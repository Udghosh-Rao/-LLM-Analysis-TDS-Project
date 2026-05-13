# Nexus AI Agent

**AI-Powered Stock Analysis Assistant**

A polished, production-ready AI system that analyzes stocks, generates natural language insights, runs sentiment analysis, and provides actionable recommendations. Built with FastAPI, LangGraph, and cutting-edge ML models.

---

## Quick Start: What This Does

**User types:** `"Analyze Tesla stock"`
**System delivers:**
- Live price data and interactive charts
- AI-generated explanation in plain English
- News sentiment analysis (positive/negative/neutral)
- BUY/HOLD/WATCH recommendation with confidence score
- Technical indicators (RSI, MA, volatility)
- ML-powered risk/anomaly scoring

---

## For HR & Non-Technical Reviewers

> **This is a real product demo, not a college assignment.**

Nexus AI Agent combines:

1. **Real-time financial data** from Yahoo Finance
2. **Machine Learning** (IsolationForest for risk detection)
3. **Sentiment Analysis** (FinBERT for news processing)
4. **Large Language Models** (Groq's llama-3.3-70b for explanations)
5. **AI Agent Orchestration** (LangGraph for task routing)
6. **RESTful APIs** (FastAPI for clean backend architecture)

The result is a system that can explain complex financial data in simple language — something real fintech startups need.

---

## For Data Science Interviewers

### Tech Stack
| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Agent | LangGraph + LangChain |
| LLM | Groq (llama-3.3-70b-versatile) |
| ML | scikit-learn (IsolationForest), HuggingFace (FinBERT) |
| Data | yfinance (market data) |
| Cache | Custom LRU cache with TTL |
| Deploy | Hugging Face Spaces (Docker) |

### Data Science Concepts Demonstrated
- **Feature Engineering**: RSI(14), MA-20/50, rolling volatility, drawdown
- **Anomaly Detection**: IsolationForest for market risk scoring
- **Sentiment Analysis**: FinBERT on financial news headlines
- **Recommendation Engine**: Score-based BUY/HOLD/WATCH logic
- **LLM Grounding**: Computed metrics passed to LLM (no hallucination)
- **Caching**: LRU cache with TTL for API performance
- **Agent Orchestration**: LangGraph state machine with task routing

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/dashboard/{ticker}` | GET | Full dashboard data: price, indicators, sentiment, recommendation |
| `/chart/{ticker}` | GET | Time series chart data with moving averages |
| `/chat` | POST | AI chat assistant for stock questions |
| `/tech-info` | GET | Technical architecture overview (for interviewers) |
| `/analyze/finance` | POST | Legacy quant analysis endpoint |
| `/detect/risk` | POST | Transaction/observation risk detection |
| `/agent/run` | POST | LangGraph autonomous agent |
| `/status` | GET | System status and counters |
| `/metrics` | GET | Observability metrics |
| `/docs` | GET | Swagger API documentation |

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Set GROQ_API_KEY and SECRET in .env
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

---

## File Structure

```
nexus-agent-main/
├── app/
│   ├── api/
│   │   ├── app.py              # FastAPI entry + routers
│   │   ├── routes_finance.py   # Dashboard, chart endpoints
│   │   ├── routes_chat.py      # AI chat assistant
│   │   ├── routes_technical.py # Interview-facing info
│   │   ├── schemas.py          # Pydantic models
│   │   └── ...
│   ├── services/
│   │   ├── sentiment_analyzer.py    # FinBERT sentiment
│   │   ├── recommendation_engine.py # BUY/HOLD logic
│   │   ├── caching.py         # LRU cache
│   │   └── groq_client.py     # LLM client
│   ├── ml/
│   │   ├── finance_features.py # Feature engineering
│   │   └── risk_model.py       # IsolationForest
│   ├── agents/
│   │   └── graph.py            # LangGraph agent
│   └── static/
│       └── index.html          # Frontend UI
└── main.py
```

---

## Key Design Decisions

1. **Compute First, Explain Later**: ML/statistical analysis runs before LLM generates text. No hallucinations.
2. **Grounded LLM Responses**: All LLM explanations receive computed metrics as context.
3. **Caching for Performance**: LRU cache reduces redundant API calls.
4. **Modular Architecture**: Clean separation of concerns (routers, services, ML layers).
5. **Demo-Friendly UI**: Modern dark theme with cards, charts, and natural language explanations.

---

## Disclaimer

**This is for educational and demonstration purposes only. It is NOT financial advice. Do not use for real investment decisions.**

---

## Credits

- **YFinance**: Market data
- **Groq**: Ultra-fast LLM inference
- **HuggingFace**: FinBERT sentiment model
- **LangGraph**: Agent orchestration framework