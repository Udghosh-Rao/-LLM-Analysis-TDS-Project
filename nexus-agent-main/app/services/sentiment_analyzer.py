import pandas as pd
from typing import Dict, List, Optional

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

from app.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    _instance = None
    _classifier = None

    def __init__(self):
        if not HAS_TRANSFORMERS:
            logger.warning("transformers not available, sentiment analysis disabled")
            return
        if SentimentAnalyzer._classifier is None:
            try:
                SentimentAnalyzer._classifier = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                    device="cpu"
                )
                logger.info("FinBERT sentiment classifier loaded")
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")

    @classmethod
    def get_instance(cls) -> "SentimentAnalyzer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_news_sentiment(self, headlines: List[str]) -> Dict:
        if not isinstance(headlines, list) or not headlines:
            return self._null_sentiment()

        classifier = SentimentAnalyzer._classifier
        if classifier is None:
            return self._null_sentiment()

        try:
            results = []
            for h in headlines[:10]:
                try:
                    result = classifier(h[:512])[0]
                    results.append({
                        "headline": h,
                        "sentiment": result["label"],
                        "score": round(float(result["score"]), 3),
                    })
                except Exception:
                    continue

            if not results:
                return self._null_sentiment()

            positive = sum(1 for r in results if r["sentiment"] == "POSITIVE")
            negative = sum(1 for r in results if r["sentiment"] == "NEGATIVE")
            neutral = len(results) - positive - negative

            total = len(results)
            scores = []
            for r in results:
                if r["sentiment"] == "POSITIVE":
                    scores.append(r["score"])
                elif r["sentiment"] == "NEGATIVE":
                    scores.append(-r["score"])
                else:
                    scores.append(0)

            avg_score = sum(scores) / total if total > 0 else 0

            if avg_score > 0.15:
                verdict = "bullish"
            elif avg_score < -0.15:
                verdict = "bearish"
            else:
                verdict = "neutral"

            return {
                "positive_pct": round(positive / total * 100, 1),
                "negative_pct": round(negative / total * 100, 1),
                "neutral_pct": round(neutral / total * 100, 1),
                "verdict": verdict,
                "headline_sentiments": results,
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._null_sentiment()

    def _null_sentiment(self) -> Dict:
        return {
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 100.0,
            "verdict": "neutral",
            "headline_sentiments": [],
        }


# Convenience function
_sentiment_analyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer