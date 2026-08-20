"""Real benchmark run on the in-process HF engine (marked `bench`, excluded from `make test`).

Verifies the full pipeline end-to-end: config → case expansion → concurrent
load → aggregation → bottleneck analysis → JSON+Markdown reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minillm.bench.bench import run_bench
from minillm.bench.config import BenchRunConfig
from minillm.common.utils import load_model_config

pytestmark = pytest.mark.bench


def test_bench_pipeline_end_to_end(tmp_path):
    cfg = load_model_config(
        BenchRunConfig,
        "configs/bench/smoke_local.yaml",
        overrides=[f"output_dir={tmp_path}/out"],
    )
    run_bench(cfg)  # type: ignore[arg-type]

    out = Path(tmp_path) / "out"
    assert (out / "bench_report.json").exists()
    assert (out / "bench_report.md").exists()

    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    assert report["experiment"] == "bench-smoke-local"
    assert len(report["metrics"]) == 2  # c=1, c=2

    rows = {r["concurrency"]: r for r in report["metrics"]}
    assert rows[1]["success_rate"] == 1.0
    assert rows[1]["throughput_tps"] > 0
    assert rows[2]["throughput_tps"] > 0
    assert rows[1]["ttft_p50"] > 0
    assert rows[2]["ttft_p50"] > 0

    # 服务端计数交叉验证（进程内计数器增量 ≈ 客户端计数）
    assert "throughput_server_tps" in rows[1]

    # 报告 markdown 包含总表与口径
    md = (out / "bench_report.md").read_text(encoding="utf-8")
    assert "瓶颈分析" in md
    assert "吞吐" in md
