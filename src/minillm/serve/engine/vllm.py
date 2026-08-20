"""vLLM adapter: OpenAI-compatible client pointed at a vLLM server.

vLLM runs as an external process/container (CUDA required); this adapter
speaks the OpenAI protocol it exposes, so the whole platform (serving layer,
benchmark, monitoring) is engine-agnostic. vLLM-specific optimizations are
configured via ``ServeConfig`` (gpu_memory_utilization / prefix caching /
continuous-batching caps) and applied by the launcher, not this client.
"""

from __future__ import annotations

from minillm.serve.engine.openai import OpenAICompatibleClient


class VLLMAdapter(OpenAICompatibleClient):
    """EngineClient for a vLLM OpenAI-compatible server."""

    engine = "vllm"

    def __init__(self, base_url: str, model: str, *, timeout: float = 300.0, api_key: str | None = None):
        super().__init__(base_url, model, timeout=timeout, api_key=api_key)
