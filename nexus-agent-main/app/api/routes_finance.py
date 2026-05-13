from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
import numpy as np

from app.api.schemas import FinanceAnalyzeRequest, RiskDetectRequest
from app.config import settings
from app.services.metrics import metrics_store
from app.tools.finance import analyze_finance_internal, detect_risk_internal
from app.services.caching import get_cache
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Finance"])


@router.post("/analyze/finance")
def analyze_finance(payload: FinanceAnalyzeRequest):
    cache = get_cache()
    cache_key = f"finance:{payload.ticker}:{payload.period}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if settings.secret and payload.secret != settings.secret:
        raise HTTPException(status_code=403, detail="Invalid secret.")
    result = analyze_finance_internal(
        ticker=payload.ticker, period=payload.period, with_explanation=True
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    result["analysis_type"] = payload.analysis_type or "standard"
    cache.set(cache_key, result)
    return result


@router.post("/analyze/stock")
def analyze_stock_compat(payload: FinanceAnalyzeRequest):
    return analyze_finance(payload)


@router.get("/dashboard/{ticker}")
def get_dashboard(ticker: str, period: str = "6mo"):
    cache = get_cache()
    cache_key = f"dashboard:{ticker}:{period}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        from app.ml.finance_features import summarize_finance_features
        from app.ml.risk_model import detect_market_anomaly
        from app.services.sentiment_analyzer import get_sentiment_analyzer
        from app.services.recommendation_engine import compute_recommendation
        from app.services.groq_client import explain_metrics

        features = summarize_finance_features(ticker=ticker, period=period)
        if "error" in features:
            raise HTTPException(status_code=400, detail=features["error"])

        feature_frame = features.pop("feature_frame", None)

        # Company info via yfinance
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        company_name = info.get("shortName") or info.get("longName") or ticker
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        volume = info.get("volume")

        # News & sentiment
        headlines = []
        news_items = []
        try:
            raw_news = t.news or []
            for n in raw_news[:8]:
                title = n.get("title", "")
                if title:
                    headlines.append(title)
                news_items.append({
                    "title": title,
                    "source": n.get("publisher", "Unknown"),
                    "url": n.get("link", ""),
                })
        except Exception:
            pass

        sentiment_result = {"positive_pct": 40, "negative_pct": 20, "neutral_pct": 40, "verdict": "neutral"}
        try:
            analyzer = get_sentiment_analyzer()
            sentiment_result = analyzer.analyze_headlines(headlines)
        except Exception:
            pass

        # ML anomaly
        ml_result = {"risk_score": 0.5, "anomaly": False}
        try:
            if feature_frame is not None and not feature_frame.empty:
                ml_result = detect_market_anomaly(feature_frame)
        except Exception:
            pass

        # Recommendation
        rec = {"label": "HOLD", "confidence": 50, "color": "yellow", "reasons": []}
        try:
            rec = compute_recommendation(
                rsi=features.get("rsi_14"),
                volatility=features.get("rolling_volatility"),
                trend=features.get("trend_signal"),
                sentiment=sentiment_result.get("verdict"),
                risk_score=ml_result.get("risk_score"),
                feature_frame=feature_frame,
            )
        except Exception:
            pass

        # AI explanation
        explanation = ""
        try:
            explanation = explain_metrics(
                ticker=ticker,
                metrics={
                    "rsi": features.get("rsi_14"),
                    "volatility": features.get("rolling_volatility"),
                    "trend": features.get("trend_signal"),
                    "recommendation": rec.get("label"),
                    "sentiment": sentiment_result.get("verdict"),
                }
            )
        except Exception:
            explanation = f"{ticker} analysis complete. RSI: {features.get('rsi_14')}, Trend: {features.get('trend_signal')}, Recommendation: {rec.get('label')}."

        result = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "price_summary": {
                "current_price": features.get("current_price"),
                "daily_change_pct": features.get("daily_change_pct"),
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "volume": volume,
            },
            "indicators": {
                "rsi_14": features.get("rsi_14"),
                "rolling_volatility": features.get("rolling_volatility"),
                "ma_20": features.get("ma_20"),
                "ma_50": features.get("ma_50"),
                "trend_signal": features.get("trend_signal"),
            },
            "ml_prediction": ml_result,
            "recommendation": rec,
            "sentiment": sentiment_result,
            "explanation": explanation,
            "news": news_items[:5],
            "trend_summary": {"direction": features.get("trend_signal", "Unknown")},
        }
        cache.set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart/{ticker}")
def get_chart_data(ticker: str, period: str = "6mo"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return {"dates": [], "close": [], "volume": [], "ma_20": [], "ma_50": []}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series([])
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        dates = [str(d.date()) for d in df.index]
        def safe_list(s):
            return [None if pd.isna(v) else round(float(v), 2) for v in s]
        return {
            "ticker": ticker.upper(),
            "dates": dates,
            "close": safe_list(close),
            "volume": safe_list(volume) if len(volume) else [],
            "ma_20": safe_list(ma_20),
            "ma_50": safe_list(ma_50),
        }
    except Exception as e:
        logger.error(f"Chart error for {ticker}: {e}")
        return {"dates": [], "close": [], "volume": [], "ma_20": [], "ma_50": []}


@router.post("/detect/risk")
def detect_risk(payload: RiskDetectRequest):
    cache = get_cache()
    cache_key = f"risk:{payload.ticker}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if settings.secret and payload.secret != settings.secret:
        raise HTTPException(status_code=403, detail="Invalid secret.")
    result = detect_risk_internal(ticker=payload.ticker)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    cache.set(cache_key, result)
    return result
