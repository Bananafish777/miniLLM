"""Web admin console configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WebEngineConfig(BaseModel):
    """One engine endpoint shown in the admin console."""

    name: str = Field(..., description="display name, e.g. vllm-main")
    type: Literal["hf", "vllm", "sglang"] = "vllm"
    base_url: str = Field(..., description="engine base URL, e.g. http://127.0.0.1:8000")
    model: str | None = Field(default=None, description="model id (optional, from /v1/models otherwise)")


class WebConfig(BaseModel):
    """Top-level config for the web admin console."""

    serve_name: str = "minillm-admin"
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    engines: list[WebEngineConfig] = Field(default_factory=list)
    bench_dir: str = Field(default="runs/bench", description="benchmark reports directory (glob <dir>/*/bench_report.json)")
    train_dir: str = Field(default="runs", description="train run root (glob <dir>/*/metrics.json)")
    refresh_interval_s: int = Field(default=5, ge=1, description="UI auto-refresh interval")
    engine_timeout_s: float = Field(default=3.0, gt=0, description="per-engine metrics fetch timeout")
    prometheus_url: str | None = Field(
        default=None, description="optional Prometheus URL to proxy PromQL queries (e.g. GPU panels)"
    )
