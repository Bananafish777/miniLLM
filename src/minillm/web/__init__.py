"""Web admin console subsystem."""

from minillm.web.app import build_web_app
from minillm.web.collector import collect_bench_runs, collect_engine_status, collect_train_runs
from minillm.web.config import WebConfig, WebEngineConfig

__all__ = [
    "WebConfig",
    "WebEngineConfig",
    "build_web_app",
    "collect_bench_runs",
    "collect_engine_status",
    "collect_train_runs",
]
