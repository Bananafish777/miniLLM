"""Aggregation & bottleneck-analysis tests with synthetic results."""

from __future__ import annotations

from minillm.bench.cases import BenchCase
from minillm.bench.client import RequestSample
from minillm.bench.metrics import aggregate, bottleneck_analysis
from minillm.bench.runner import CaseResult


def _case(engine: str, conc: int, in_t: int = 128, out_t: int = 128) -> BenchCase:
    return BenchCase(engine=engine, model_label="m", concurrency=conc,
                     input_tokens=in_t, output_tokens=out_t)


def _sample(ttft: float, latency: float, tokens: int, itl: float = 0.1) -> RequestSample:
    times = [0.0]
    for _ in range(1, tokens):
        times.append(times[-1] + itl)
    return RequestSample(ttft_s=ttft, latency_s=latency, output_tokens=tokens,
                         input_tokens=128, chunk_times_s=[t + ttft for t in times])


def _result(case: BenchCase, samples: list[RequestSample], wall: float, server_delta: int = 0) -> CaseResult:
    return CaseResult(
        case=case, samples=samples, wall_s=wall,
        total_output_tokens=sum(s.output_tokens for s in samples),
        total_input_tokens=sum(s.input_tokens or 0 for s in samples),
        server_tokens_before=0, server_tokens_after=server_delta,
        server_requests_before=0, server_requests_after=len(samples),
        gpu_mem_samples=[],
    )


def test_aggregate_throughput_and_percentiles():
    r = _result(_case("vllm", 4), [_sample(0.1, 1.0, 50) for _ in range(8)], wall=2.0, server_delta=400)
    m = aggregate(r)
    assert m.throughput_tps == 200.0  # 400 tokens / 2s
    assert m.throughput_server_tps == 200.0
    assert m.ttft["p50"] == 0.1
    assert m.success_rate == 1.0
    assert m.requests == 8


def test_aggregate_with_errors():
    r = _result(_case("hf", 2), [_sample(0.1, 0.5, 10), RequestSample(ttft_s=-1, latency_s=-1, output_tokens=0, error="boom")], wall=1.0)
    m = aggregate(r)
    assert m.success_rate == 0.5
    assert m.ttft["p50"] == 0.1  # only successful samples


def test_bottleneck_saturation_detected():
    results = [
        _result(_case("hf", 1), [_sample(0.5, 2.0, 100) for _ in range(4)], wall=2.0),
        _result(_case("hf", 8), [_sample(2.5, 4.0, 100) for _ in range(32)], wall=8.0),
    ]
    # c=1: 400 tokens/2s = 200tps; c=8: 3200/8 = 400tps → 2×, 不触发饱和
    findings = bottleneck_analysis(results)
    assert not any("饱和" in f.message for f in findings)


def test_bottleneck_ttft_regression_detected():
    results = [
        _result(_case("vllm", 1), [_sample(0.1, 0.5, 50) for _ in range(4)], wall=1.0),
        _result(_case("vllm", 16), [_sample(0.6, 1.0, 50) for _ in range(64)], wall=8.0),
    ]
    # TTFT-p99: 0.1 → 0.6 = 6×，触发排队/抢占告警
    findings = bottleneck_analysis(results)
    assert any("TTFT" in f.message for f in findings)


def test_bottleneck_error_rate_critical():
    bad = RequestSample(ttft_s=-1, latency_s=-1, output_tokens=0, error="timeout")
    results = [_result(_case("sglang", 1), [_sample(0.1, 0.5, 50), bad], wall=1.0)]
    findings = bottleneck_analysis(results)
    assert any(f.severity == "critical" for f in findings)


def test_engine_comparison_finding():
    results = [
        _result(_case("hf", 4), [_sample(0.5, 2.0, 50) for _ in range(8)], wall=4.0),    # 100 tps
        _result(_case("vllm", 4), [_sample(0.05, 0.2, 50) for _ in range(8)], wall=1.0),  # 400 tps
    ]
    findings = bottleneck_analysis(results)
    assert any("vllm" in f.message and "hf" in f.message for f in findings)
