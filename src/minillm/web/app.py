"""Web admin console: FastAPI app + self-contained static frontend.

``minillm web`` serves:
- ``/``                  → 管理控制台（原生 JS，无构建）
- ``/api/status``        → 各引擎实时状态（up/down、模型、指标、tokens/s）
- ``/api/engines``       → 引擎配置列表
- ``/api/bench``         → Benchmark 报告列表（runs/bench/*/bench_report.json）
- ``/api/train``         → 微调运行摘要（runs/*/metrics.json）
- ``/api/prometheus``    → PromQL 代理（可选，配置 prometheus_url 后可用）
- ``/api/health``        → 探活
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from minillm.web.collector import collect_bench_runs, collect_engine_status, collect_train_runs
from minillm.web.config import WebConfig

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class ChatBody(BaseModel):
    """对话请求体（转发到引擎的 /v1/chat/completions）。"""

    engine: str = Field(..., description="admin.yaml engines 列表中的引擎 name")
    messages: list[dict[str, str]] = Field(..., min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = True


def build_web_app(cfg: WebConfig) -> FastAPI:
    app = FastAPI(title=f"miniLLM Admin Console ({cfg.serve_name})", version="0.1.0")

    # ------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "serve_name": cfg.serve_name}

    @app.get("/api/engines")
    def engines():
        return [e.model_dump() for e in cfg.engines]

    @app.get("/api/status")
    async def status():
        engines_data = await collect_engine_status(cfg)
        return {
            "serve_name": cfg.serve_name,
            "refresh_interval_s": cfg.refresh_interval_s,
            "engines": engines_data,
            "prometheus_url": cfg.prometheus_url,
        }

    @app.get("/api/bench")
    def bench(limit: int = Query(default=20, ge=1, le=100)):
        return {"runs": collect_bench_runs(cfg.bench_dir, limit)}

    @app.get("/api/train")
    def train(limit: int = Query(default=20, ge=1, le=100)):
        return {"runs": collect_train_runs(cfg.train_dir, limit)}

    @app.get("/api/prometheus")
    async def prometheus(query: str = Query(..., description="PromQL query")):
        if not cfg.prometheus_url:
            raise HTTPException(status_code=400, detail="prometheus_url 未配置")
        url = f"{cfg.prometheus_url.rstrip('/')}/api/v1/query"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params={"query": query})
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------ chat

    @app.post("/api/chat")
    async def chat(req: ChatBody):
        """对话代理：把请求转发到所选引擎的 OpenAI 兼容端点（支持 SSE 流式）。"""
        engine = next((e for e in cfg.engines if e.name == req.engine), None)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"引擎 {req.engine!r} 未配置（见 engines 列表）")
        base = engine.base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                models_resp = await client.get(f"{base}/v1/models")
                models_resp.raise_for_status()
                model_id = models_resp.json()["data"][0]["id"]
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"引擎 {req.engine} 不可达: {e}") from e

        payload = {
            "model": model_id,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": req.stream,
        }
        url = f"{base}/v1/chat/completions"

        if not req.stream:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPError as e:
                raise HTTPException(status_code=502, detail=f"引擎 {req.engine} 请求失败: {e}") from e

        async def _forward():
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"
            except httpx.HTTPError as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"

        return StreamingResponse(_forward(), media_type="text/event-stream")

    # static assets (app.js / style.css)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
