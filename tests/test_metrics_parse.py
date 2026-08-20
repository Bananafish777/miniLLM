"""Prometheus text-format metrics parser tests."""

from __future__ import annotations

from minillm.serve.engine.openai import parse_prometheus_text

SAMPLE = """# HELP vllm:generation_tokens_total Number of generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model="m",} 1234.0
vllm:generation_tokens_total{model="m",engine="gpu"} 100.0
vllm:num_requests_running 5
vllm:cache_hit_rate{model="m"} 0.42
minillm_ttft_seconds_bucket{le="0.1"} 7.0
# EOF
"""


def test_parse_sums_labeled_series():
    out = parse_prometheus_text(SAMPLE)
    assert out["vllm:generation_tokens_total"] == 1334.0  # 1234 + 100
    assert out["vllm:num_requests_running"] == 5.0
    assert out["vllm:cache_hit_rate"] == 0.42


def test_parse_histogram_buckets():
    out = parse_prometheus_text(SAMPLE)
    assert out["minillm_ttft_seconds_bucket"] == 7.0


def test_parse_empty_and_comments():
    assert parse_prometheus_text("") == {}
    assert parse_prometheus_text("# only a comment\n\n") == {}


def test_parse_scientific_notation():
    out = parse_prometheus_text("minillm_thing 1.5e-3\n")
    assert out["minillm_thing"] == 0.0015
