import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


def compute_recommendation(
    ticker: str,
    features: pd.DataFrame,
    ml_output: Dict,
    sentiment: Optional[Dict] = None,
) -> Dict:
    """
    Simple, explainable recommendation logic.
    NOT investment advice -- for demo/educational purposes only.
    """
    if features is None or features.empty:
        return _null_recommendation(ticker)

    try:
        last_close = features["Close"].iloc[-1]
        ma20 = features["ma_20"].iloc[-1] if "ma_20" in features else None
        ma50 = features["ma_50"].iloc[-1] if "ma_50" in features else None
        rsi = features["rsi_14"].iloc[-1] if "rsi_14" in features else None
        vulatility = features["rolling_volatility_20d"].iloc[-1] if (
            "rolling_volatility_20d" in features) else None
        anomaly_score = ml_output.get("anomaly_score", 0)

        score = 50  # baseline neutral
        reasons = []

        # Price vs moving averages
        if ma20 is not None and ma50 is not None:
            if last_close > ma20 > ma50:
                score += 20
                reasons.append("Price trading above key moving averages")
            elif last_close < ma50 < ma20:
                score -= 15
                reasons.append("Price below key moving averages")
            else:
                score += 5
                reasons.append("Price consolidating near moving averages")

        # RSI analysis
        if rsi is not None:
            if rsi < 30:
                score += 15
                reasons.append("RSI indicates oversold conditions")
            elif rsi > 70:
                score -= 10
                reasons.append("RSI indicates overbought conditions")
            elif 40 <= rsi <= 60:
                score += 10
                reasons.append("RSI in healthy neutral zone")

        # Volatility
        if vulatility is not None:
            if vulatility < 0.02:
                score += 10
                reasons.append("Low volatility indicates stability")
            elif vulatility > 0.05:
                score -= 5
                reasons.append("Elevated volatility increases risk")

        # Anomaly score
        if anomaly_score is not None and anomaly_score < 0.5:
            score += 15
            reasons.append("Low market anomaly detected")
        elif anomaly_score is not None and anomaly_score > 0.7:
            score -= 15
            reasons.append("High anomaly score suggests unusual activity")

        # Sentiment bonus
        if sentiment:
            sentiment_score = (
                sentiment.get("positive_pct", 50) -
                sentiment.get("negative_pct", 50))
            if sentiment_score > 10:
                score += 10
                reasons.append("Favorable news sentiment")
            elif sentiment_score < -10:
                score -= 10
                reasons.append("Negative news sentiment")

        # Clamp score
        score = max(0, min(100, score))

        # Map to recommendation
        if score >= 70:
            label = "BUY"
            color = "green"
        elif score >= 55:
            label = "HOLD"
            color = "yellow"
        elif score >= 40:
            label = "WATCH"
            color = "orange"
        else:
            label = "AVOID"
            color = "red"

        confidence = min(score if label == "BUY" else (100 - score) if label in (
            "AVOID", "WATCH") else 70, 95)

        return {
            "label": label,
            "confidence": round(confidence, 1),
            "score": round(score, 1),
            "color": color,
            "reasons": reasons if reasons else ["Insufficient data for detailed analysis"],
            "disclaimer": "For educational/demo purposes only. Not financial advice.",
            "ticker": ticker,
        }

    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return _null_recommendation(ticker)


def _null_recommendation(ticker: str) -> Dict:
    return {
        "label": "HOLD",
        "confidence": 0.0,
        "score": 50.0,
        "color": "yellow",
        "reasons": ["Unable to compute recommendation at this time"],
        "disclaimer": "For educational/demo purposes only. Not financial advice.",
        "ticker": ticker,
    }