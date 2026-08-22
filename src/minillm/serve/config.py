"""Serving configuration: engine-agnostic surface for inference services."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ServeModelConfig(BaseModel):
    """Model used by the serving engine."""

    name_or_path: str = Field(..., description="HF hub id or local path of the (exported) model")
    dtype: Literal["auto", "fp32", "fp16", "bf16", "int8", "int4"] = "auto"
    attn_impl: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    trust_remote_code: bool = False
    tokenizer_name: str | None = None
    max_model_len: int = Field(default=2048, ge=16, description="context window (KV cache bound)")


class ServeConfig(BaseModel):
    """Top-level config for one inference service instance."""

    engine: Literal["hf", "vllm", "sglang"] = Field(
        default="hf", description="hf: in-process Transformers; vllm/sglang: external OpenAI-compatible server"
    )
    serve_name: str = Field(default="minillm", description="server identity reported by /v1/models")
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    model: ServeModelConfig

    # --- engine-specific knobs (ignored by engine=hf) ---
    gpu_memory_utilization: float = Field(default=0.85, ge=0.1, le=0.95, description="vLLM: KV cache budget")
    tensor_parallel_size: int = Field(default=1, ge=1, description="vLLM/SGLang: GPU count for TP")
    enable_prefix_caching: bool = Field(default=True, description="vLLM: reuse KV of common prefixes")
    max_num_seqs: int = Field(default=256, ge=1, description="vLLM: continuous-batching concurrency cap")
    max_num_batched_tokens: int = Field(default=8192, ge=1, description="vLLM: prefill batch token cap")

    # --- SGLang on Apple Silicon (MLX runtime) ---
    sglang_use_mlx: bool = Field(
        default=False,
        description="SGLang MLX runtime (Apple Silicon). Requires sglang built with srt_mps extra; "
                    "launches with SGLANG_USE_MLX=1 and --disable-cuda-graph",
    )
    sglang_metrics: bool = Field(default=True, description="SGLang: expose Prometheus /metrics")
    engine_python: str | None = Field(
        default=None,
        description="Python interpreter for the engine (e.g. .venv-sglang/bin/python when sglang "
                    "lives in a separate venv); defaults to the current interpreter",
    )

    # --- client defaults (when requests omit sampling params) ---
    default_temperature: float = Field(default=0.7, gt=0.0)
    default_max_tokens: int = Field(default=512, ge=1)
    request_timeout: float = Field(default=300.0, gt=0, description="client-side timeout (s)")
