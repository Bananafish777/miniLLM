"""Benchmark → Pushgateway 指标序列构造（跨引擎对比入 Grafana 大盘）。"""

from __future__ import annotations

import time
from typing import Any

from minillm.bench.metrics import CaseMetrics

# 指标名 → (帮助文本, 值提取函数)
_SERIES_SPEC: list[tuple[str, str, Any]] = [
    ("minillm_bench_throughput_tps", "Benchmark throughput (tokens/s)", lambda m: m.throughput_tps),
    ("minillm_bench_ttft_p99_seconds", "TTFT p99 (s)", lambda m: m.ttft.get("p99", float("nan"))),
    ("minillm_bench_ttft_p50_seconds", "TTFT p50 (s)", lambda m: m.ttft.get("p50", float("nan"))),
    ("minillm_bench_itl_p50_seconds", "Inter-token latency p50 (s)", lambda m: m.itl.get("p50", float("nan"))),
    ("minillm_bench_e2e_p99_seconds", "End-to-end latency p99 (s)", lambda m: m.latency.get("p99", float("nan"))),
    ("minillm_bench_success_rate", "Request success rate", lambda m: m.success_rate),
]


def build_bench_series(metrics: list[CaseMetrics], experiment: str) -> list[dict[str, Any]]:
    """Expand aggregated case metrics into pushable series (one per metric × case)."""
    series: list[dict[str, Any]] = []
    for m in metrics:
        labels = {
            "experiment": experiment,
            "engine": m.case.engine,
            "model": m.case.model_label,
            "concurrency": str(m.case.concurrency),
            "input_tokens": str(m.case.input_tokens),
            "output_tokens": str(m.case.output_tokens),
        }
        for name, help_text, extract in _SERIES_SPEC:
            series.append({"name": name, "help": help_text, "value": extract(m), "labels": labels})
    return series


def push_bench_metrics(
    metrics: list[CaseMetrics],
    *,
    experiment: str,
    gateway: str | None = None,
) -> bool:
    """Push a benchmark run's results to the push gateway (job=minillm-bench)."""
    from minillm.monitor.push import push_series

    return push_series(
        build_bench_series(metrics, experiment),
        job="minillm-bench",
        gateway=gateway,
        grouping_key={"experiment": experiment, "run": time.strftime("%Y%m%d-%H%M%S")},
    )
