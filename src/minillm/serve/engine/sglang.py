"""SGLang adapter: OpenAI-compatible client pointed at an SGLang server.

SGLang's ``launch_server`` exposes the OpenAI protocol on the given port;
this adapter is identical in shape to the vLLM one (they share the same
protocol), which is exactly the point of the engine abstraction: swapping
engines is a config change, not a code change.
"""

from __future__ import annotations

from minillm.serve.engine.openai import OpenAICompatibleClient


class SGLangAdapter(OpenAICompatibleClient):
    """EngineClient for an SGLang OpenAI-compatible server."""

    engine = "sglang"

    def __init__(self, base_url: str, model: str, *, timeout: float = 300.0, api_key: str | None = None):
        super().__init__(base_url, model, timeout=timeout, api_key=api_key)
