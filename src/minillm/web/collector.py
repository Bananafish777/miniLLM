"""Data collection for the admin console.

- engine status: pull /metrics (+ /v1/models) from each configured engine
- benchmark runs: read runs/bench/*/bench_report.json
- training runs: read runs/*/metrics.json
All sources degrade gracefully (engine down / dir missing → null, never crash).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from minillm.serve.engine.openai import parse_prometheus_text
from minillm.web.config import WebConfig

log = logging.getLogger(__name__)

# 上次抓取快照（用于计算 tokens/s 速率）：{engine_name: (ts, tokens_total)}
_snapshots: dict[str, tuple[float, float]] = {}


async def collect_engine_status(cfg: WebConfig) -> list[dict[str, Any]]:
    """Fetch live status of every configured engine (concurrently)."""
    results: list[dict[str, Any]] = []
    for engine in cfg.engines:
        results.append(await _one_engine(engine, cfg.engine_timeout_s))
    return results


async def _one_engine(engine, timeout: float) -> dict[str, Any]:
    base = engine.base_url.rstrip("/")
    entry: dict[str, Any] = {"name": engine.name, "type": engine.type, "up": False, "error": None,
                             "model": engine.model, "metrics": {}, "tokens_per_sec": None}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base}/metrics")
            resp.raise_for_status()
            metrics = parse_prometheus_text(resp.text)
            try:
                models_resp = await client.get(f"{base}/v1/models")
                if models_resp.status_code == 200:
                    entry["model"] = entry["model"] or models_resp.json()["data"][0]["id"]
            except Exception:  # noqa: BLE001
                pass
        entry["up"] = True
        entry["metrics"] = metrics
        # tokens/s 速率（相邻两次抓取差值）；兼容三种引擎的 token 计数指标名
        tokens_total = next(
            (metrics[k] for k in ("minillm_tokens_generated_total", "vllm:generation_tokens_total",
                                  "sglang:generation_tokens_total") if k in metrics),
            0.0,
        )
        now = time.monotonic()
        prev = _snapshots.get(engine.name)
        if prev is not None and tokens_total >= prev[1]:
            dt = now - prev[0]
            if dt > 0:
                entry["tokens_per_sec"] = round((tokens_total - prev[1]) / dt, 2)
        _snapshots[engine.name] = (now, tokens_total)
    except Exception as e:  # noqa: BLE001
        entry["error"] = str(e)[:120]
        log.debug("engine %s unreachable: %s", engine.name, e)
    return entry


def collect_bench_runs(bench_dir: str, limit: int = 20) -> list[dict[str, Any]]:
    """List benchmark reports (newest first)."""
    root = Path(bench_dir)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for report in sorted(root.glob("*/bench_report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            runs.append({
                "dir": str(report.parent),
                "experiment": data.get("experiment"),
                "timestamp": data.get("timestamp"),
                "n_cases": len(data.get("metrics", [])),
                "metrics": data.get("metrics", []),
                "findings": data.get("findings", []),
            })
            if len(runs) >= limit:
                break
        except (json.JSONDecodeError, OSError) as e:
            log.warning("skip broken bench report %s: %s", report, e)
    return runs


def collect_train_runs(train_dir: str, limit: int = 20) -> list[dict[str, Any]]:
    """List finetuning run summaries (newest first)."""
    root = Path(train_dir)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for summary in sorted(root.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        # 排除 bench 目录自身的 metrics.json
        if "bench" in summary.parts:
            continue
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            tm = data.get("train_metrics", {})
            runs.append({
                "dir": str(summary.parent),
                "experiment": data.get("experiment"),
                "finetune_mode": data.get("finetune_mode"),
                "model": data.get("model"),
                "device": data.get("device"),
                "eval_loss": tm.get("eval_loss"),
                "train_tokens_per_second": tm.get("train_tokens_per_second"),
                "export_dir": (data.get("export") or {}).get("export_dir"),
            })
            if len(runs) >= limit:
                break
        except (json.JSONDecodeError, OSError) as e:
            log.warning("skip broken train summary %s: %s", summary, e)
    return runs
