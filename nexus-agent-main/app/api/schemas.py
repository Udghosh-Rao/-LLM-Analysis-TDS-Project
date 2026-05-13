from __future__ import annotations
from typing import Any, Optional, List
from pydantic import BaseModel, Field





class SolveRequest(BaseModel):
    url: str
    secret: str = Field(default="")


class FinanceAnalyzeRequest(BaseModel):
    ticker: str = Field(default="AAPL")
    period: str = Field(default="6mo")
    analysis_type: Optional[str] = Field(default="standard")
    secret: str = Field(default="")


class RiskDetectRequest(BaseModel):
    observation: dict[str, Any]
    secret: str = Field(default="")


class AgentRunRequest(BaseModel):
    prompt: str
    secret: str = Field(default="")


class ChatRequest(BaseModel):
    message: str
    ticker: Optional[str] = Field(default=None)
    context: Optional[dict[str, Any]] = Field(default=None)
    secret: str = Field(default="")


class NewsRequest(BaseModel):
    ticker: str
    limit: int = Field(default=5)
    secret: str = Field(default="")


# ============ RESPONSE SCHEMAS ============

class PriceSummary(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    current_price: Optional[float] = None
    open_price: Optional[float] = None
    prev_close: Optional[float] = None
    daily_change: Optional[float] = None
    daily_change_pct: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None


class ChartData(BaseModel):
    dates: List[str] = Field(default_factory=list)
    close: List[float] = Field(default_factory=list)
    volume: List[float] = Field(default_factory=list)
    ma_20: Optional[List[float]] = Field(default=None)
    ma_50: Optional[List[float]] = Field(default=None)
    forecast_dates: Optional[List[str]] = Field(default=None)
    forecast_values: Optional[List[float]] = Field(default=None)


class Indicators(BaseModel):
    rsi_14: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    rolling_volatility: Optional[float] = None
    return_1d: Optional[float] = None
    trend_signal: Optional[str] = None


class MLPrediction(BaseModel):
    anomaly_score: Optional[float] = None
    risk_score: Optional[float] = None
    label: Optional[str] = None
    confidence: Optional[float] = None


class SentimentItem(BaseModel):
    headline: str
    sentiment: str  # POSITIVE | NEGATIVE | NEUTRAL
    score: Optional[float] = None
    source: Optional[str] = None
    published: Optional[str] = None


class SentimentSummary(BaseModel):
    positive_pct: float = 0.0
    negative_pct: float = 0.0
    neutral_pct: float = 0.0
    verdict: str = "neutral"  # bullish | bearish | neutral
    headline_sentiments: List[SentimentItem] = Field(default_factory=list)


class NewsItem(BaseModel):
    title: str
    url: Optional[str] = None
    source: Optional[str] = None
    published: Optional[str] = None
    sentiment: Optional[str] = None


class Recommendation(BaseModel):
    label: str = "HOLD"  # BUY | HOLD | WATCH
    confidence: float = 0.0
    score: float = 0.0
    color: str = "yellow"
    reasons: List[str] = Field(default_factory=list)
    disclaimer: str = "For educational/demo purposes only. Not financial advice."


class Forecast(BaseModel):
    forecast_dates: List[str] = Field(default_factory=list)
    forecast_values: List[float] = Field(default_factory=list)
    direction: str = "flat"
    explanation: str = ""


class SentimentResponse(BaseModel):
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    verdict: str
    headline_sentiments: List[SentimentItem]


class NewsResponse(BaseModel):
    news: List[NewsItem]


class ChatResponse(BaseModel):
    response: str
    ticker_mentioned: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TechnicalInfo(BaseModel):
    architecture: dict[str, Any]
    ml_pipeline: dict[str, Any]
    apis_used: List[str]
    data_flow: str
    key_concepts: List[str]
    models: List[str]


class DashboardResponse(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    price_summary: PriceSummary
    indicators: Indicators
    chart_data: ChartData
    ml_prediction: MLPrediction
    sentiment: SentimentSummary
    news: List[NewsItem]
    recommendation: Recommendation
    forecast: Forecast