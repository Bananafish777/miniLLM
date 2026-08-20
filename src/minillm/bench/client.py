"""Async benchmark clients.

- :class:`AsyncOpenAIClient` — httpx async streaming client for vLLM/SGLang
  (and any OpenAI-compatible endpoint); measures TTFT / ITL / latency from
  chunk arrival times.
- :class:`AsyncHFClient` — in-process Transformers engine (the no-GPU
  baseline), serialized by an async lock (honest single-stream baseline).

Token accounting is double-checked against the engine's own counters
(``/metrics`` for remote servers, adapter counters in-process) — the
"client-side + server-side cross-validation" channel from the architecture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from minillm.bench.config import EngineTarget
from minillm.serve.engine.hf import HFAdapter
from minillm.serve.engine.openai import parse_prometheus_text

log = logging.getLogger(__name__)

# ~4 chars per token for English text — used only for prompt synthesis
# (exact input lengths come from engine usage/counters where available).
CHARS_PER_TOKEN = 4


def make_prompt(input_tokens: int) -> str:
    """Deterministic synthetic prompt of approximately `input_tokens` tokens."""
    base = "The quick brown fox jumps over the lazy dog. "
    n = max(1, input_tokens * CHARS_PER_TOKEN // len(base) + 1)
    return (base * n)[: input_tokens * CHARS_PER_TOKEN]


@dataclass
class RequestSample:
    """Per-request timing and token measurements."""

    ttft_s: float
    latency_s: float
    output_tokens: int
    input_tokens: int | None = None
    chunk_times_s: list[float] = field(default_factory=list)
    error: str | None = None

    def itls(self) -> list[float]:
        """Inter-token latencies derived from chunk arrival times."""
        return [b - a for a, b in zip(self.chunk_times_s, self.chunk_times_s[1:], strict=False)]


class AsyncOpenAIClient:
    """Async OpenAI-compatible streaming client (vLLM / SGLang)."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def stream_completions(
        self, prompt: str, *, max_tokens: int, temperature: float, top_p: float
    ) -> RequestSample:
        url = urljoin(self.base_url, "/v1/completions")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        t0 = time.perf_counter()
        first: float | None = None
        times: list[float] = []
        tokens = 0
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        choice = data.get("choices", [{}])[0]
                        piece = choice.get("text") or choice.get("delta", {}).get("content", "")
                        if not piece:
                            continue
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        times.append(now)
                        tokens += 1
        except Exception as e:  # noqa: BLE001
            return RequestSample(ttft_s=-1, latency_s=-1, output_tokens=0, error=str(e))
        if first is None:
            return RequestSample(ttft_s=-1, latency_s=-1, output_tokens=0, error="empty stream")
        return RequestSample(
            ttft_s=first - t0,
            latency_s=times[-1] - t0,
            output_tokens=tokens,
            input_tokens=max(1, len(prompt) // CHARS_PER_TOKEN),
            chunk_times_s=times,
        )

    async def metrics(self) -> dict[str, float]:
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 30.0)) as client:
                resp = await client.get(urljoin(self.base_url, "/metrics"))
                resp.raise_for_status()
                return parse_prometheus_text(resp.text)
        except Exception as e:  # noqa: BLE001
            log.warning("engine metrics unavailable: %s", e)
            return {}


class AsyncHFClient:
    """In-process Transformers engine, requests serialized by an async lock."""

    def __init__(self, target: EngineTarget):
        self.adapter = HFAdapter(
            target.model,
            dtype=target.dtype,
            attn_impl=target.attn_impl,
            max_model_len=target.max_model_len,
        )
        self._lock = asyncio.Lock()

    async def stream_completions(
        self, prompt: str, *, max_tokens: int, temperature: float, top_p: float
    ) -> RequestSample:
        async with self._lock:
            t0 = time.perf_counter()

            def _collect():
                gen = self.adapter.completions(
                    prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True
                )
                first: float | None = None
                times: list[float] = []
                tokens = 0
                for _ in gen:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    times.append(now)
                    tokens += 1
                return first, times, tokens

            try:
                first, times, tokens = await asyncio.to_thread(_collect)
            except Exception as e:  # noqa: BLE001
                return RequestSample(ttft_s=-1, latency_s=-1, output_tokens=0, error=str(e))
        if first is None:
            return RequestSample(ttft_s=-1, latency_s=-1, output_tokens=0, error="empty stream")
        return RequestSample(
            ttft_s=first - t0,
            latency_s=times[-1] - t0,
            output_tokens=tokens,
            input_tokens=None,
            chunk_times_s=times,
        )

    async def metrics(self) -> dict[str, float]:
        return self.adapter.metrics()


def make_client(target: EngineTarget) -> AsyncOpenAIClient | AsyncHFClient:
    """Factory: engine target → async benchmark client."""
    if target.type == "hf":
        return AsyncHFClient(target)
    if target.type in ("vllm", "sglang"):
        if not target.base_url:
            raise ValueError(f"engine {target.type!r} requires base_url")
        return AsyncOpenAIClient(target.base_url, target.model)
    raise ValueError(f"unknown engine type: {target.type!r}")  # pragma: no cover
