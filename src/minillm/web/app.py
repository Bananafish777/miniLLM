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

import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from minillm.web.collector import collect_bench_runs, collect_engine_status, collect_train_runs
from minillm.web.config import WebConfig

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


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

    # static assets (app.js / style.css)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
