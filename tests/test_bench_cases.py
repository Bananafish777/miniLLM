"""Benchmark case expansion / prompt synthesis / percentile tests."""

from __future__ import annotations

from minillm.bench.cases import BenchCase, expand_cases
from minillm.bench.client import make_prompt
from minillm.bench.config import BenchRunConfig
from minillm.bench.metrics import percentiles
from minillm.common.utils import load_model_config

CFG = "configs/bench/smoke_local.yaml"


def test_expand_cases_product():
    cfg = load_model_config(BenchRunConfig, CFG)
    cases = expand_cases(cfg)  # type: ignore[arg-type]
    assert len(cases) == 1 * 1 * 2 * 1 * 1  # engines×models×conc×in×out
    assert cases[0].engine == "hf"
    assert cases[0].concurrency == 1
    assert cases[0].key().startswith("hf|tiny-llama|c1|in32|out16")


def test_expand_cases_larger_matrix():
    cfg = load_model_config(
        BenchRunConfig, CFG,
        overrides=[
            'matrix.engines=["hf","vllm","sglang"]',
            "matrix.concurrency=[1,8,32]",
            "matrix.input_tokens=[128,2048]",
            "matrix.output_tokens=[128]",
        ],
    )
    cases = expand_cases(cfg)  # type: ignore[arg-type]
    assert len(cases) == 3 * 1 * 3 * 2 * 1


def test_make_prompt_approximates_length():
    p = make_prompt(128)
    assert len(p) == 512  # 4 chars/token
    assert "fox" in p


def test_percentiles():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    out = percentiles(values)
    assert out["p50"] == 5.0
    assert out["p90"] == 9.0
    assert out["p99"] == 10.0


def test_percentiles_empty():
    out = percentiles([])
    assert out["p50"] != out["p50"]  # nan


def test_bench_case_label():
    c = BenchCase(engine="vllm", model_label="m1", concurrency=16, input_tokens=128, output_tokens=256)
    assert "c=16" in c.label()
