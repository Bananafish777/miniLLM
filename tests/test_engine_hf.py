"""HF adapter (in-process Transformers engine) tests."""

from __future__ import annotations

import pytest

from minillm.serve.engine import HFAdapter

MODEL = "data/models/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def adapter():
    return HFAdapter(MODEL, dtype="fp32", attn_impl="eager", max_model_len=256)


def test_list_models(adapter):
    assert adapter.list_models() == [MODEL]


def test_completions_non_stream(adapter):
    result = adapter.completions("Hello", max_tokens=8, temperature=0.0)
    assert isinstance(result.text, str) and result.text
    assert result.completion_tokens == 8
    assert result.prompt_tokens > 0
    assert result.latency_s >= 0


def test_chat_plain_fallback(adapter):
    result = adapter.chat(
        [{"role": "user", "content": "你好"}], max_tokens=8, temperature=0.0
    )
    assert isinstance(result.text, str) and result.text


def test_streaming_yields_chunks(adapter):
    stream = adapter.completions("Hello world", max_tokens=8, temperature=0.0, stream=True)
    pieces = list(stream)
    assert pieces
    joined = "".join(pieces).strip()
    assert joined
    # 8 个 token 的流式输出 chunk 数不超过 8（streamer 可能合并相邻 token）
    assert len(pieces) <= 8
    # 非流式路径 token 计数正确（注：MPS 内核非确定性，不比较具体文本）
    non_stream = adapter.completions("Hello world", max_tokens=8, temperature=0.0)
    assert non_stream.completion_tokens == 8


def test_metrics_and_stats(adapter):
    m = adapter.metrics()
    assert m["minillm_requests_total"] >= 1
    assert m["minillm_tokens_generated_total"] > 0
    stats = adapter.system_stats()
    assert stats.engine == "hf"
    assert stats.model == MODEL
    assert stats.requests_total >= 1
