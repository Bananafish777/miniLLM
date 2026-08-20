"""In-process Transformers engine adapter (the no-GPU / baseline path).

Used directly by the HF OpenAI-compatible server and by benchmark baselines.
Supports real token streaming via ``TextIteratorStreamer`` on a background
thread, so latency characteristics match a production streaming setup.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

import torch

from minillm.common.utils import detect_device, resolve_dtype, torch_dtype
from minillm.serve.engine.base import Completion, EngineStats, _now

log = logging.getLogger(__name__)


class HFAdapter:
    """EngineClient implementation backed by ``transformers`` in-process."""

    engine = "hf"

    def __init__(self, name_or_path: str, *, dtype: str = "auto", attn_impl: str = "sdpa",
                 tokenizer_name: str | None = None, trust_remote_code: bool = False,
                 max_model_len: int = 2048):
        self.model_id = name_or_path
        self.device = detect_device()
        self.max_model_len = max_model_len
        self._requests = 0
        self._tokens = 0
        self._lock = threading.Lock()

        dtype_name = resolve_dtype(dtype, self.device)
        torch_dtype_name = torch_dtype(dtype_name)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype_name, "trust_remote_code": trust_remote_code}
        if attn_impl == "eager":
            kwargs["attn_implementation"] = "eager"
        elif attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "flash_attention_2"
        else:
            kwargs["attn_implementation"] = "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs).to(self.device)
        tok_name = tokenizer_name or name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        # 显式设定 pad_token_id，避免 transformers>=5 对 generate 传参告警
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        log.info("HFAdapter ready: model=%s device=%s dtype=%s", name_or_path, self.device, dtype_name)

    # ------------------------------------------------------------ protocol

    def list_models(self) -> list[str]:
        return [self.model_id]

    def completions(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float | None = None,
        top_p: float | None = None, stream: bool = False, **extra: Any,
    ) -> Completion | Iterable[str]:
        max_tokens = max_tokens or 64
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = self._gen_kwargs(max_tokens, temperature, top_p)
        if not stream:
            t0 = _now()
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            new_ids = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            with self._lock:
                self._requests += 1
                self._tokens += new_ids.numel()
            return Completion(
                text=text, finish_reason="stop",
                prompt_tokens=inputs["input_ids"].shape[1], completion_tokens=new_ids.numel(),
                latency_s=_now() - t0,
            )

        return self._stream_generate(inputs, gen_kwargs)

    def chat(
        self, messages: list[dict[str, str]], *, max_tokens: int | None = None,
        temperature: float | None = None, top_p: float | None = None,
        stream: bool = False, **extra: Any,
    ) -> Completion | Iterable[str]:
        if self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self._plain_chat_prompt(messages)
        return self.completions(prompt, max_tokens=max_tokens, temperature=temperature,
                                top_p=top_p, stream=stream, **extra)

    def metrics(self) -> dict[str, float]:
        with self._lock:
            return {
                "minillm_requests_total": float(self._requests),
                "minillm_tokens_generated_total": float(self._tokens),
                "minillm_active_engines": 1.0,
            }

    def system_stats(self) -> EngineStats:
        stats = EngineStats(engine=self.engine, model=self.model_id, device=self.device)
        with self._lock:
            stats.requests_total = self._requests
            stats.tokens_generated = self._tokens
        if self.device == "cuda":
            stats.gpu_mem_used_bytes = torch.cuda.memory_allocated()
            stats.gpu_mem_total_bytes = torch.cuda.get_device_properties(0).total_memory
        try:
            import psutil

            stats.extra["rss_bytes"] = float(psutil.Process().memory_info().rss)
        except ImportError:  # pragma: no cover
            pass
        return stats

    # ------------------------------------------------------------ internals

    @staticmethod
    def _gen_kwargs(max_tokens: int, temperature: float | None, top_p: float | None) -> dict:
        kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature is not None and temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _stream_generate(self, inputs: dict, gen_kwargs: dict) -> Iterable[str]:
        """Yield tokens as they are decoded, via a streamer thread."""
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        gen_kwargs = {**gen_kwargs, "streamer": streamer}
        thread = threading.Thread(
            target=self._generate_in_thread, args=(inputs, gen_kwargs), daemon=True
        )
        thread.start()
        return _token_iter(streamer)

    def _generate_in_thread(self, inputs: dict, gen_kwargs: dict) -> None:
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        with self._lock:
            self._requests += 1
            self._tokens += out[0][inputs["input_ids"].shape[1]:].numel()

    @staticmethod
    def _plain_chat_prompt(messages: list[dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = "Human" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        return "\n\n".join(parts) + "\n\nAssistant: "


def _token_iter(streamer):
    for piece in streamer:
        if piece:
            yield piece
