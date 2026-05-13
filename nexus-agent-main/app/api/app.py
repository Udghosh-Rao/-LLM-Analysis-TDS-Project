import time
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes_agent import router as agent_router
from app.api.routes_finance import router as finance_router
from app.api.routes_monitoring import router as monitoring_router
from app.api.routes_chat import router as chat_router
from app.api.routes_technical import router as technical_router
from app.config import settings
from app.services.metrics import metrics_store

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if not os.path.exists(STATIC_DIR):
    STATIC_DIR = None

app = FastAPI(
    title=settings.app_name,
    description=(
        f"{settings.app_subtitle}. An AI-powered stock analysis assistant "
        "with LangGraph orchestration, financial indicators, ML anomaly detection, "
        "sentiment analysis, and natural language explanations."
    ),
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    path = request.url.path
    with metrics_store.track_latency("api_routes", path):
        response = await call_next(request)
    if response.status_code >= 400:
        metrics_store.inc("api_failed_runs")
    metrics_store.inc("api_total_runs")
    return response

app.include_router(agent_router)
app.include_router(finance_router)
app.include_router(monitoring_router)
app.include_router(chat_router)
app.include_router(technical_router)

if STATIC_DIR and os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    @app.get("/")
    def serve_index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "running", "docs": "/docs"}