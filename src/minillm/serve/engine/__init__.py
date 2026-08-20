"""Engine adapters: unified EngineClient interface for every backend."""

from minillm.serve.engine.base import Completion, EngineClient, EngineStats, RequestParams
from minillm.serve.engine.hf import HFAdapter
from minillm.serve.engine.openai import OpenAICompatibleClient, parse_prometheus_text
from minillm.serve.engine.sglang import SGLangAdapter
from minillm.serve.engine.vllm import VLLMAdapter

__all__ = [
    "Completion",
    "EngineClient",
    "EngineStats",
    "HFAdapter",
    "OpenAICompatibleClient",
    "RequestParams",
    "SGLangAdapter",
    "VLLMAdapter",
    "parse_prometheus_text",
]
