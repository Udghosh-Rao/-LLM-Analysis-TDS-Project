import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.agent.graph import run_agent
from app.api.schemas import AgentRunRequest, SolveRequest
from app.config import settings
from shared_store import BASE64_STORE, run_counter, url_time

router = APIRouter(tags=["Agent"])

_executor = ThreadPoolExecutor(max_workers=4)


@router.post("/solve")
async def solve(payload: SolveRequest, background_tasks: BackgroundTasks):
    if settings.secret and payload.secret != settings.secret:
        raise HTTPException(status_code=403, detail="Invalid secret.")

    url_time.clear()
    BASE64_STORE.clear()
    run_counter["total"] += 1
    os.environ["url"] = payload.url
    os.environ["offset"] = "0"
    url_time[payload.url] = time.time()
    background_tasks.add_task(run_agent, payload.url)
    return {"status": "ok", "message": "Agent started.", "run_id": run_counter["total"]}


@router.post("/agent/run")
async def run_agent_endpoint(payload: AgentRunRequest):
    if settings.secret and payload.secret != settings.secret:
        raise HTTPException(status_code=403, detail="Invalid secret.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, run_agent, payload.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return {"status": "success", "response": "No output"}

    last = messages[-1]
    content = getattr(last, "content", str(last))
    return {"status": "success", "response": content}
