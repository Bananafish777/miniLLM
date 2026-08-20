"""Pushgateway integration tests (local capture server, no external deps)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from minillm.bench.cases import BenchCase
from minillm.bench.client import RequestSample
from minillm.bench.metrics import aggregate
from minillm.bench.runner import CaseResult
from minillm.monitor import build_bench_series, push_series


class _CaptureHandler(BaseHTTPRequestHandler):
    bodies: list[str] = []

    def _capture(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        type(self).bodies.append(body)
        self.send_response(200)
        self.end_headers()

    def do_PUT(self) -> None:  # noqa: N802
        self._capture()

    def do_POST(self) -> None:  # noqa: N802
        self._capture()

    def log_message(self, *args) -> None:  # noqa: ARG002
        pass


@pytest.fixture
def capture_server():
    _CaptureHandler.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def test_push_series_sends_prometheus_format(capture_server):
    ok = push_series(
        [
            {"name": "minillm_bench_throughput_tps", "help": "tps", "value": 123.4,
             "labels": {"engine": "vllm", "concurrency": "8"}},
            {"name": "minillm_bench_success_rate", "help": "ok", "value": 1.0,
             "labels": {"engine": "vllm", "concurrency": "8"}},
        ],
        job="minillm-bench",
        gateway=capture_server,
        grouping_key={"experiment": "e1", "run": "20260101-000000"},
    )
    assert ok is True
    assert len(_CaptureHandler.bodies) == 1
    body = _CaptureHandler.bodies[0]
    # prometheus_client 按字母序输出标签
    assert 'minillm_bench_throughput_tps{concurrency="8",engine="vllm"} 123.4' in body
    assert 'minillm_bench_success_rate{concurrency="8",engine="vllm"} 1.0' in body


def test_push_without_gateway_is_noop():
    assert push_series([{"name": "x", "value": 1, "labels": {}}], job="j", gateway=None) is False


def _sample(ttft: float, latency: float, tokens: int) -> RequestSample:
    return RequestSample(ttft_s=ttft, latency_s=latency, output_tokens=tokens,
                         input_tokens=128, chunk_times_s=[ttft + i * 0.05 for i in range(tokens)])


def _metrics(engine: str):
    case = BenchCase(engine=engine, model_label="m", concurrency=8, input_tokens=128, output_tokens=64)
    result = CaseResult(
        case=case, samples=[_sample(0.1, 1.0, 64) for _ in range(8)], wall_s=2.0,
        total_output_tokens=512, total_input_tokens=1024,
        server_tokens_before=0, server_tokens_after=512,
        server_requests_before=0, server_requests_after=8,
    )
    return aggregate(result)


def test_build_bench_series_expands_all_metrics():
    series = build_bench_series([_metrics("vllm"), _metrics("hf")], "bench-e2e")
    names = {s["name"] for s in series}
    assert {
        "minillm_bench_throughput_tps",
        "minillm_bench_ttft_p99_seconds",
        "minillm_bench_itl_p50_seconds",
        "minillm_bench_e2e_p99_seconds",
        "minillm_bench_success_rate",
    } <= names
    # 每用例 × 6 指标
    assert len(series) == 2 * 6
    # 标签携带引擎与实验
    assert all(s["labels"]["experiment"] == "bench-e2e" for s in series)
    assert {s["labels"]["engine"] for s in series} == {"vllm", "hf"}
    # 吞吐值来自聚合结果
    tps = [s for s in series if s["name"] == "minillm_bench_throughput_tps"]
    assert all(s["value"] == 256.0 for s in tps)  # 512 tokens / 2s
