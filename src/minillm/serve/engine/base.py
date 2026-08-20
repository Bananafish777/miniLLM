"""Engine abstraction: one interface for every inference backend.

Design (ADR-1): serving and benchmarking share this interface, so "bench =
production" holds by construction. Implementations:

- :class:`OpenAICompatibleClient` — speaks the OpenAI HTTP protocol
  (used by the vLLM and SGLang adapters)
- :class:`HFAdapter` — in-process Transformers generation (no server needed)
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

RequestParams = dict[str, Any]  # sampling params forwarded to the engine


@dataclass
class Completion:
    """A completed (non-streaming) generation result."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float = 0.0
    raw: dict[str, Any] | None = field(default=None, repr=False)


@dataclass
class EngineStats:
    """Engine-level runtime statistics."""

    engine: str
    model: str
    device: str
    requests_total: int = 0
    tokens_generated: int = 0
    gpu_mem_used_bytes: int | None = None
    gpu_mem_total_bytes: int | None = None
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class EngineClient(Protocol):
    """Unified interface implemented by every engine adapter."""

    engine: str
    model: str

    def list_models(self) -> list[str]: ...

    def completions(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> Completion | Iterable[str]: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> Completion | Iterable[str]: ...

    def metrics(self) -> dict[str, float]: ...

    def system_stats(self) -> EngineStats: ...


def _now() -> float:
    return time.perf_counter()
