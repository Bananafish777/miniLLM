"""Metric aggregation & bottleneck analysis (pure functions, unit-testable).

指标口径（与架构文档 5.3 一致）：
- 吞吐     : 总输出 tokens / 用例总耗时（另附服务端计数 delta 交叉验证）
- TTFT     : 首 token 延迟（流式首 chunk 到达时刻 - 请求发出时刻）
- ITL      : 相邻 chunk 间隔（解码阶段逐 token 延迟）
- 端到端   : 请求发出 → 最后 chunk
- GPU 显存 : 压测期间采样峰值（DCGM/pynvml 通道，无 GPU 时为 None）
"""

from __future__ import annotations

from dataclasses import dataclass

from minillm.bench.cases import BenchCase
from minillm.bench.runner import CaseResult


def percentiles(values: list[float], ps: tuple[float, ...] = (50, 90, 99)) -> dict[str, float]:
    """Nearest-rank percentiles (1-based rank = ceil(p/100 * n))."""
    if not values:
        return {f"p{int(p)}": float("nan") for p in ps}
    ordered = sorted(values)
    n = len(ordered)
    out: dict[str, float] = {}
    for p in ps:
        rank = max(1, min(n, int(p / 100 * n + 0.999999)))
        out[f"p{int(p)}"] = ordered[rank - 1]
    return out


@dataclass
class CaseMetrics:
    """Aggregated metrics for one case."""

    case: BenchCase
    throughput_tps: float
    throughput_server_tps: float | None
    ttft: dict[str, float]
    itl: dict[str, float]
    latency: dict[str, float]
    success_rate: float
    gpu_mem_peak_bytes: float | None
    wall_s: float
    requests: int

    def row(self) -> dict:
        return {
            "engine": self.case.engine,
            "model": self.case.model_label,
            "concurrency": self.case.concurrency,
            "input_tokens": self.case.input_tokens,
            "output_tokens": self.case.output_tokens,
            "throughput_tps": round(self.throughput_tps, 2),
            "throughput_server_tps": round(self.throughput_server_tps, 2) if self.throughput_server_tps else None,
            "ttft_p50": round(self.ttft.get("p50", float("nan")), 4),
            "ttft_p99": round(self.ttft.get("p99", float("nan")), 4),
            "itl_p50": round(self.itl.get("p50", float("nan")), 4),
            "e2e_p99": round(self.latency.get("p99", float("nan")), 4),
            "success_rate": round(self.success_rate, 3),
            "gpu_mem_peak_mb": round(self.gpu_mem_peak_bytes / 1e6, 1) if self.gpu_mem_peak_bytes else None,
        }


def aggregate(result: CaseResult) -> CaseMetrics:
    """Aggregate per-request samples into case-level metrics."""
    ok = [s for s in result.samples if s.error is None]
    ttfts = [s.ttft_s for s in ok]
    latencies = [s.latency_s for s in ok]
    itls = [x for s in ok for x in s.itls()]

    server_delta = result.server_tokens_after - result.server_tokens_before
    throughput_server = (
        server_delta / result.wall_s if server_delta > 0 and result.wall_s > 0 else None
    )
    gpu_peak = max(result.gpu_mem_samples) if result.gpu_mem_samples else None

    return CaseMetrics(
        case=result.case,
        throughput_tps=result.total_output_tokens / result.wall_s if result.wall_s > 0 else 0.0,
        throughput_server_tps=throughput_server,
        ttft=percentiles(ttfts),
        itl=percentiles(itls),
        latency=percentiles(latencies),
        success_rate=result.success_rate,
        gpu_mem_peak_bytes=gpu_peak,
        wall_s=result.wall_s,
        requests=len(result.samples),
    )


@dataclass
class Finding:
    severity: str  # info | warn | critical
    message: str


def bottleneck_analysis(results: list[CaseResult]) -> list[Finding]:
    """内置启发式瓶颈定位（可扩展；规则见 docs/architecture.md 5.3）。"""
    findings: list[Finding] = []
    metrics = {r.case.key(): aggregate(r) for r in results}

    if not metrics:
        return findings

    # 1) 错误率
    for r in results:
        if r.success_rate < 1.0:
            findings.append(Finding(
                "critical",
                f"[{r.case.label()}] 失败率 {(1 - r.success_rate) * 100:.1f}% "
                f"({sum(1 for s in r.samples if s.error) }/{len(r.samples)} 请求) — 检查服务健康与超时",
            ))

    # 2) 吞吐饱和：同引擎下吞吐随并发不再增长
    by_engine: dict[str, list[CaseMetrics]] = {}
    for m in metrics.values():
        by_engine.setdefault(m.case.engine, []).append(m)
    for engine, ms in by_engine.items():
        ms = sorted(ms, key=lambda m: m.case.concurrency)
        for prev, cur in zip(ms, ms[1:], strict=False):
            if prev.case.input_tokens != cur.case.input_tokens or prev.case.output_tokens != cur.case.output_tokens:
                continue
            if cur.throughput_tps <= prev.throughput_tps * 1.05:
                findings.append(Finding(
                    "warn",
                    f"[{engine}] 吞吐在 c={prev.case.concurrency}→{cur.case.concurrency} 饱和 "
                    f"({prev.throughput_tps:.0f}→{cur.throughput_tps:.0f} tokens/s) — decode 或显存受限",
                ))

    # 3) TTFT 随并发劣化（排队/prefill 瓶颈）
    for engine, ms in by_engine.items():
        ms = sorted(ms, key=lambda m: m.case.concurrency)
        base = next((m for m in ms if m.case.concurrency == 1), None)
        if base is None:
            continue
        worst = max(ms, key=lambda m: m.ttft.get("p99", 0))
        if worst.case.concurrency > 1 and worst.ttft.get("p99", 0) > base.ttft.get("p99", 0) * 5:
            findings.append(Finding(
                "warn",
                f"[{engine}] TTFT-p99 随并发上升 {base.ttft['p99']:.2f}s(c=1) → "
                f"{worst.ttft['p99']:.2f}s(c={worst.case.concurrency}) — 排队或 prefill 抢占",
            ))

    # 4) 引擎横向对比（同并发同长度）：优化收益
    concs = {m.case.concurrency for m in metrics.values()}
    for conc in concs:
        same = [m for m in metrics.values() if m.case.concurrency == conc]
        if len(same) < 2:
            continue
        best = max(same, key=lambda m: m.throughput_tps)
        worst = min(same, key=lambda m: m.throughput_tps)
        if best.throughput_tps > 0 and worst.throughput_tps > 0:
            ratio = best.throughput_tps / worst.throughput_tps
            if ratio > 1.5:
                findings.append(Finding(
                    "info",
                    f"[c={conc}] {best.case.engine} 吞吐是 {worst.case.engine} 的 {ratio:.1f}× "
                    f"({best.throughput_tps:.0f} vs {worst.throughput_tps:.0f} tokens/s) — 优化引擎收益显著",
                ))
            elif ratio < 1.1:
                findings.append(Finding(
                    "info",
                    f"[c={conc}] 引擎间吞吐差异 <10% — 疑似共享瓶颈（单卡显存/CPU offload/数据管线）",
                ))

    # 5) 显存水位
    for m in metrics.values():
        if m.gpu_mem_peak_bytes:
            findings.append(Finding(
                "info",
                f"[{m.case.engine}] 压测峰值显存 {m.gpu_mem_peak_bytes / 1e6:.0f} MB "
                f"(c={m.case.concurrency}) — 可对比模型驻留与 KV Cache 预算",
            ))
    return findings
