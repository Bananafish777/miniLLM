"""Benchmark orchestrator: matrix → cases → load → aggregate → analyze → report."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from minillm.bench.cases import expand_cases
from minillm.bench.client import make_client
from minillm.bench.config import BenchRunConfig
from minillm.bench.metrics import bottleneck_analysis
from minillm.bench.report import print_console, write_report
from minillm.bench.runner import CaseResult, run_case

log = logging.getLogger(__name__)


def run_bench(cfg: BenchRunConfig) -> dict:
    """Execute the full benchmark matrix; returns the report summary."""
    cases = expand_cases(cfg)
    log.info("=== minillm bench | %s | %d cases ===", cfg.experiment, len(cases))

    # one client per engine (model loaded once per engine)
    clients = {name: make_client(target) for name, target in cfg.engines.items()}

    results: list[CaseResult] = []
    for case in cases:
        client = clients[case.engine]
        log.info("running case %s ...", case.label())
        results.append(asyncio.run(run_case(cfg, case, client)))

    findings = bottleneck_analysis(results)
    for f in findings:
        log.info("[%s] %s", f.severity.upper(), f.message)

    output_dir = cfg.output_dir or f"runs/bench/{cfg.experiment}-{time.strftime('%Y%m%d-%H%M%S')}"
    paths = write_report(cfg, results, findings, output_dir)

    print_console(results)
    log.info("report written: %s", paths["json"])
    return {
        "experiment": cfg.experiment,
        "output_dir": str(Path(output_dir).resolve()),
        "cases": len(cases),
        "report": paths,
        "findings": [{"severity": f.severity, "message": f.message} for f in findings],
    }
