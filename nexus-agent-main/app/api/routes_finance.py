from fastapi import APIRouter, HTTPException

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
        logger.debug(f"Cache hit: {cache_key}")
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

        metrics_store.inc("finance_analyses_completed")
        features = summarize_finance_features(ticker=ticker, period=period)
        if "error" in features:
            raise HTTPException(status_code=400, detail=features["error"])
        # Store feature_frame BEFORE popping
        feature_frame = features["feature_frame"]
        ml_prediction = detect_market_anomaly(feature_frame)
        features.pop("feature_frame", None)
        sentiment = get_sentiment_analyzer()
        import yfinance as yf
        headlines = [n.get("title", "") for n in yf.Ticker(ticker).news[:8]]
        sentiment_result = sentiment.analyze_news_sentiment(headlines)
        recommendation = compute_recommendation(ticker, feature_frame, ml_prediction, sentiment_result)
        explanation = explain_metrics(
            "finance_analysis",
            {
                "ticker": ticker, "period": period,
                "price_summary": features["price_summary"],
                "indicators": features["indicators"],
                "trend_summary": features["trend_summary"],
                "signal": features["signal"],
                "ml_prediction": ml_prediction,
            },
        )
        response = {
            "ticker": ticker,
            "company_name": features["price_summary"].get("company_name", ticker),
            "price_summary": features["price_summary"],
            "indicators": features["indicators"],
            "ml_prediction": ml_prediction,
            "sentiment": sentiment_result,
            "recommendation": recommendation,
            "explanation": explanation,
        }
        cache.set(cache_key, response)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart/{ticker}")
def get_chart_data(ticker: str, period: str = "6mo"):
    try:
        import yfinance as yf
        import pandas as pd
        cache = get_cache()
        cache_key = f"chart:{ticker}:{period}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if df.empty:
            return {"dates": [], "close": [], "volume": []}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        close = df["Close"].tolist()
        ma20 = df["Close"].rolling(20).mean().tolist()
        ma50 = df["Close"].rolling(50).mean().tolist()
        volume = df["Volume"].tolist()
        dates = [str(d)[:10] for d in df.index]
        result = {
            "dates": dates,
            "close": close,
            "volume": volume,
            "ma_20": ma20,
            "ma_50": ma50,
        }
        cache.set(cache_key, result)
        return result
    except Exception as e:
        return {"error": str(e)}


@router.post("/detect/risk")
def detect_risk(payload: RiskDetectRequest):
    if settings.secret and payload.secret != settings.secret:
        raise HTTPException(status_code=403, detail="Invalid secret.")
    metrics_store.inc("risk_analyses_completed")
    return detect_risk_internal(
        observation=payload.observation, with_explanation=True
    )