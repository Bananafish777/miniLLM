"""Integration tests: OpenAI-compatible protocol against a live server.

Boots the HF-backed FastAPI server on a random port (uvicorn in a thread),
then drives it through the *vLLM and SGLang adapters* — proving the engine
abstraction's core claim: any OpenAI-compatible backend is interchangeable,
and protocol correctness is verified without a GPU.
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from minillm.serve.config import ServeConfig
from minillm.serve.engine import SGLangAdapter, VLLMAdapter
from minillm.serve.server import build_app

MODEL = "data/models/tiny-random-LlamaForCausalLM"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    cfg = ServeConfig(
        engine="hf", serve_name="minillm-test", host="127.0.0.1", port=port,
        model={"name_or_path": MODEL, "dtype": "fp32", "attn_impl": "eager", "max_model_len": 256},
        default_max_tokens=16,
    )
    server = uvicorn.Server(
        uvicorn.Config(build_app(cfg), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                yield url
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:  # pragma: no cover
        raise RuntimeError("server did not start")
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def vllm(server_url):
    return VLLMAdapter(server_url, MODEL)


@pytest.fixture(scope="module")
def sglang(server_url):
    return SGLangAdapter(server_url, MODEL)


def test_list_models(vllm, sglang):
    assert vllm.list_models() == [MODEL]
    assert sglang.list_models() == [MODEL]


def test_completions_non_stream(vllm):
    result = vllm.completions("Hello", max_tokens=8, temperature=0.0)
    assert isinstance(result.text, str) and result.text
    assert result.finish_reason == "stop"
    assert result.completion_tokens == 8
    assert result.prompt_tokens > 0
    assert result.latency_s >= 0


def test_chat_non_stream(sglang):
    result = sglang.chat([{"role": "user", "content": "你好"}], max_tokens=8, temperature=0.0)
    assert isinstance(result.text, str) and result.text


def test_streaming_protocol(vllm):
    pieces = list(vllm.completions("Stream me", max_tokens=8, temperature=0.0, stream=True))
    assert pieces
    assert "".join(pieces).strip()


def test_streaming_chat_protocol(sglang):
    pieces = list(sglang.chat([{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.0, stream=True))
    assert pieces and "".join(pieces).strip()


def test_unknown_model_rejected(vllm):
    bad = VLLMAdapter(vllm.base_url, "some/other-model")
    with pytest.raises(httpx.HTTPStatusError):
        bad.completions("Hello", max_tokens=4)


def test_metrics_endpoint(vllm):
    m = vllm.metrics()
    assert m["minillm_requests_total"] >= 4
    assert m["minillm_tokens_generated_total"] >= 24
    # histogram 桶存在
    assert any(k.startswith("minillm_ttft_seconds_bucket") for k in m)


def test_system_stats(vllm):
    stats = vllm.system_stats()
    assert stats.engine == "vllm"
    assert stats.requests_total >= 4
