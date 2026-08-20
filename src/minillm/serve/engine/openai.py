"""OpenAI-compatible HTTP client + Prometheus text-format metrics parser.

One implementation serves both the vLLM and SGLang adapters (they expose the
same OpenAI protocol), and is integration-tested against our own HF server, so
protocol correctness is verified without a GPU.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urljoin

import httpx

from minillm.serve.engine.base import Completion, EngineStats, _now

log = logging.getLogger(__name__)

SSE_DONE = "[DONE]"


# ------------------------------------------------------------------ metrics

def parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format into {metric_name: sum of samples}.

    Label dimensions are collapsed by summation — good enough for the
    cross-validation numbers benchmark needs (throughput, cache hit rate, ...).
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{.*\}\s+(-?[\d.eE+-]+)$", line)
        if not m:
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(-?[\d.eE+-]+)$", line)
        if not m:
            continue
        name, value = m.group(1), float(m.group(2))
        result[name] = result.get(name, 0.0) + value
    return result


# ------------------------------------------------------------------ client

class OpenAICompatibleClient:
    """Adapter base speaking the OpenAI HTTP protocol (v1/completions & chat)."""

    engine: str = "openai"
    path_prefix: str = "/v1"

    def __init__(self, base_url: str, model: str, *, timeout: float = 300.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # ------------------------------------------------------------ protocol

    def list_models(self) -> list[str]:
        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            resp = client.get(urljoin(self.base_url, f"{self.path_prefix}/models"))
            resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def completions(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> Completion | Iterable[str]:
        payload = self._payload(
            {"prompt": prompt}, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            stream=stream, **extra,
        )
        return self._send("POST", "/completions", payload, stream=stream, text_field="text")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> Completion | Iterable[str]:
        payload = self._payload(
            {"messages": messages}, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            stream=stream, **extra,
        )
        return self._send("POST", "/chat/completions", payload, stream=stream, text_field="delta")

    def metrics(self) -> dict[str, float]:
        resp = httpx.get(urljoin(self.base_url, "/metrics"), timeout=min(self._timeout, 30.0))
        resp.raise_for_status()
        return parse_prometheus_text(resp.text)

    def system_stats(self) -> EngineStats:
        stats = self.metrics()
        return EngineStats(
            engine=self.engine,
            model=self.model,
            device="remote",
            requests_total=int(stats.get("minillm_requests_total", 0)),
            tokens_generated=int(stats.get("minillm_tokens_generated_total", 0)),
            extra=stats,
        )

    # ------------------------------------------------------------ internals

    def _payload(self, body: dict, *, max_tokens, temperature, top_p, stream, **extra) -> dict:
        payload = {"model": self.model, "stream": stream, **body, **extra}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return payload

    def _send(
        self, method: str, path: str, payload: dict, *, stream: bool, text_field: str
    ) -> Completion | Iterable[str]:
        url = urljoin(self.base_url, f"{self.path_prefix}{path}")
        t0 = _now()

        if not stream:
            with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return Completion(
                text=choice.get("text") or choice.get("message", {}).get("content", ""),
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=_now() - t0,
                raw=data,
            )

        # streaming: the generator outlives this call, so the client must live
        # as long as the stream is consumed (closed in the generator's finally).
        client = httpx.Client(timeout=self._timeout, headers=self._headers)

        def _stream():
            ttft: float | None = None
            try:
                with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == SSE_DONE:
                            break
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        if ttft is None:
                            ttft = _now() - t0
                        choice = data["choices"][0]
                        piece = choice.get("text") or choice.get("delta", {}).get("content", "")
                        if piece:
                            yield piece
            finally:
                client.close()

        return _stream()
