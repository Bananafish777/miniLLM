"""Benchmark configuration: engine registry + matrix of load cases."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EngineTarget(BaseModel):
    """How to reach one engine for benchmarking."""

    type: Literal["hf", "vllm", "sglang"]
    model: str = Field(..., description="model id/path reported to the engine")
    base_url: str | None = Field(default=None, description="required for vllm/sglang (OpenAI endpoint)")
    dtype: str = "auto"  # hf only
    attn_impl: str = "sdpa"  # hf only
    max_model_len: int = 2048  # hf only


class MatrixConfig(BaseModel):
    """Cartesian product expanded into one case per combination."""

    engines: list[str] = Field(..., min_length=1, description="keys of the `engines` registry to sweep")
    models: list[str] = Field(default_factory=lambda: ["default"], min_length=1, description="model labels (report only)")
    concurrency: list[int] = Field(..., min_length=1, description="concurrent requests per case")
    input_tokens: list[int] = Field(..., min_length=1)
    output_tokens: list[int] = Field(..., min_length=1)


class SamplingConfig(BaseModel):
    temperature: float = 0.0  # greedy: deterministic, comparable across engines
    top_p: float = 1.0


class RunConfig(BaseModel):
    warmup_requests: int = Field(default=2, ge=0)
    requests_per_concurrency: int = Field(default=8, ge=1, description="requests per worker at each concurrency level")
    stream: bool = Field(default=True, description="measure TTFT precisely via streaming")
    timeout_s: float = Field(default=300.0, gt=0)
    max_parallel_cases: int = Field(default=1, ge=1, description="cases run sequentially by default (fairness)")


class BenchRunConfig(BaseModel):
    """Top-level benchmark config."""

    experiment: str = "minillm-bench"
    output_dir: str | None = Field(default=None, description="default: runs/bench/<experiment>-<timestamp>")
    engines: dict[str, EngineTarget]
    matrix: MatrixConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    run: RunConfig = Field(default_factory=RunConfig)
