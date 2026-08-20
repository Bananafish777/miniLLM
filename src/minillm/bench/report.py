"""Benchmark report generation: JSON + Markdown + console table."""

from __future__ import annotations

import time
from pathlib import Path

from minillm.bench.config import BenchRunConfig
from minillm.bench.metrics import CaseMetrics, Finding, aggregate
from minillm.bench.runner import CaseResult
from minillm.common.utils import save_json


def _fmt(v: float) -> str:
    return f"{v:.2f}" if v == v else "-"


def build_markdown(cfg: BenchRunConfig, metrics: list[CaseMetrics], findings: list[Finding]) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark 报告 — {cfg.experiment}")
    lines.append(f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 引擎: {', '.join(cfg.matrix.engines)} | 并发: {cfg.matrix.concurrency}")
    lines.append(f"- 采样: temperature={cfg.sampling.temperature} | stream={cfg.run.stream}")
    lines.append("")

    lines.append("## 指标总表（吞吐 tokens/s · TTFT/ITL/端到端 秒）")
    lines.append("| engine | model | conc | in | out | 吞吐 | 吞吐(服务端) | TTFT p50 | TTFT p99 | ITL p50 | E2E p99 | 成功率 | 显存峰值MB |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in sorted(metrics, key=lambda x: (x.case.engine, x.case.concurrency)):
        r = m.row()
        lines.append(
            f"| {r['engine']} | {r['model']} | {r['concurrency']} | {r['input_tokens']} | {r['output_tokens']} "
            f"| {_fmt(m.throughput_tps)} | {_fmt(m.throughput_server_tps) if m.throughput_server_tps else '-'} "
            f"| {_fmt(m.ttft.get('p50', float('nan')))} | {_fmt(m.ttft.get('p99', float('nan')))} "
            f"| {_fmt(m.itl.get('p50', float('nan')))} | {_fmt(m.latency.get('p99', float('nan')))} "
            f"| {m.success_rate:.0%} | {r['gpu_mem_peak_mb'] if r['gpu_mem_peak_mb'] else '-'} |"
        )
    lines.append("")

    lines.append("## 瓶颈分析")
    if findings:
        for f in findings:
            lines.append(f"- **[{f.severity}]** {f.message}")
    else:
        lines.append("- 未触发内置启发式规则（可扩展）。")
    lines.append("")
    lines.append("## 口径说明")
    lines.append("- 吞吐 = 总输出 tokens / 用例总耗时；`吞吐(服务端)` 来自引擎 /metrics 计数增量（交叉验证）。")
    lines.append("- TTFT 通过流式首 chunk 计时；ITL 为相邻 chunk 间隔。")
    lines.append("- 显存为压测期间采样峰值（无 GPU 环境为 None）。")
    return "\n".join(lines)


def write_report(
    cfg: BenchRunConfig,
    results: list[CaseResult],
    findings: list[Finding],
    output_dir: str | Path,
) -> dict:
    """Persist JSON + Markdown reports; returns artifact paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = [aggregate(r) for r in results]

    payload = {
        "experiment": cfg.experiment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.model_dump(),
        "metrics": [m.row() for m in metrics],
        "findings": [{"severity": f.severity, "message": f.message} for f in findings],
    }
    json_path = out / "bench_report.json"
    save_json(payload, json_path)

    md_path = out / "bench_report.md"
    md_path.write_text(build_markdown(cfg, metrics, findings), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def print_console(results: list[CaseResult]) -> None:
    """Compact console summary table."""
    metrics = [aggregate(r) for r in results]
    header = f"{'engine':<8}{'conc':>5}{'in':>6}{'out':>6}{'tps':>9}{'ttft_p50':>10}{'ttft_p99':>10}{'e2e_p99':>10}{'ok%':>7}"
    print(header)
    print("-" * len(header))
    for m in sorted(metrics, key=lambda x: (x.case.engine, x.case.concurrency)):
        print(
            f"{m.case.engine:<8}{m.case.concurrency:>5}{m.case.input_tokens:>6}{m.case.output_tokens:>6}"
            f"{m.throughput_tps:>9.1f}"
            f"{_fmt(m.ttft.get('p50', float('nan'))):>10}"
            f"{_fmt(m.ttft.get('p99', float('nan'))):>10}"
            f"{_fmt(m.latency.get('p99', float('nan'))):>10}"
            f"{m.success_rate * 100:>6.0f}%"
        )
