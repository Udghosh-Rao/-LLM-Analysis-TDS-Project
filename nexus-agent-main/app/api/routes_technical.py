from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["Technical"])


@router.get("/tech-info")
def get_technical_info():
    return {
        "project_name": "Nexus AI Agent",
        "version": settings.app_version,
        "architecture": {
            "pattern": "LangGraph + FastAPI",
            "layers": [
                "Frontend (HTML/Tailwind CSS)",
                "FastAPI Router Layer",
                "Service Layer (sentiment, caching, recommendation)",
                "ML Layer (features, risk detection, forecasting)",
                "Data Layer (yfinance, Groq LLM)",
            ],
            "data_flow": "User input -> API -> Service -> ML -> LLM -> Response",
        },
        "ml_pipeline": {
            "models": [
                "IsolationForest (anomaly detection)",
                "FinBERT (sentiment analysis)",
                "LangChain LLM (natural language)",
            ],
            "features": [
                "RSI(14) - momentum indicator",
                "Moving Average 20 & 50 - trend",
                "Rolling Volatility 20d - risk",
                "Return 1d - daily performance",
                "Drawdown - risk/reward",
                "Support/Resistance - price levels",
            ],
            "process": "fetch OHLCV -> engineer features -> ML inference -> LLM explanation",
        },
        "apis_used": [
            "yfinance (market data)",
            f"Groq API - {settings.groq_model} (LLM)",
            "HuggingFace Transformers - ProsusAI/finbert (sentiment)",
            "LangGraph (agent orchestration)",
            "FastAPI (REST API framework)",
        ],
        "key_concepts": [
            "Feature Engineering: RSI, MA, volatility, drawdown",
            "Anomaly Detection: IsolationForest for risk scoring",
            "Sentiment Analysis: FinBERT on news headlines",
            "Recommendation Logic: Score-based BUY/HOLD/WATCH",
            "LLM Grounding: Computed metrics passed to LLM",
            "Caching: LRU cache with TTL for performance",
            "LangGraph: Autonomous agent task routing",
        ],
        "design_patterns": [
            "Singleton (sentiment analyzer, cache)",
            "Dependency Injection (router imports services)",
            "Repository pattern (ML feature engineering)",
            "Middleware (metrics tracking, CORS)",
        ],
        "deployment": {
            "platform": "Hugging Face Spaces (Docker)",
            "config": "Environment variables (.env)",
            "port": "7860 (ASGI)",
        },
    }