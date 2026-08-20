"""Dataset loaders for the finetuning pipeline.

Supported formats:

- ``synthetic``: locally generated instruction samples (no network, smoke tests / demos)
- ``alpaca``:    JSON/JSONL with ``instruction`` (+optional ``input``) / ``output``
- ``sharegpt``:  JSON/JSONL with ``conversations: [{from: human|gpt, value}]``
- ``plain``:     raw text documents (``text`` field), next-token prediction

All loaders return a ``datasets.Dataset`` with the canonical fields
``prompt`` and ``response`` (``plain`` returns ``text`` only), which the
tokenizer stage consumes uniformly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset

from minillm.common.logging import get_logger

log = get_logger(__name__)

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# ------------------------------------------------------------------ synthetic

_SYNTHETIC_INSTRUCTIONS = [
    "解释什么是大语言模型。",
    "用一句话总结注意力机制。",
    "写一个 Python 函数计算斐波那契数列。",
    "什么是 KV Cache？",
    "介绍一下 LoRA 微调的原理。",
    "vLLM 如何实现 Continuous Batching？",
    "什么是 Paged Attention？",
    "列举三个深度学习框架。",
    "如何评估一个聊天机器人的质量？",
    "简述梯度下降算法的步骤。",
]

_SYNTHETIC_RESPONSES = [
    "大语言模型是基于海量文本预训练的神经网络模型，能够通过自回归方式生成自然语言。",
    "注意力机制让模型根据上下文动态加权输入信息，是 Transformer 的核心组件。",
    "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
    "KV Cache 缓存历史 token 的 Key 和 Value，避免解码阶段重复计算注意力。",
    "LoRA 通过低秩矩阵分解冻结原权重，只训练少量新增参数，大幅降低显存占用。",
    "Continuous Batching 在请求粒度动态调度，前一个请求解码完成后立即插入新请求。",
    "Paged Attention 将 KV Cache 分页存储，按需分配显存，减少碎片和浪费。",
    "PyTorch、TensorFlow、JAX 是三个主流深度学习框架。",
    "可以从准确性、流畅性、事实一致性、多轮连贯性等维度评估聊天机器人。",
    "梯度下降通过计算损失对参数的梯度，沿负梯度方向迭代更新参数以最小化损失。",
]


def load_synthetic(n: int, seed: int = 42) -> Dataset:
    """Generate ``n`` deterministic instruction-following samples."""
    rng = random.Random(seed)
    samples: list[dict[str, str]] = []
    for i in range(n):
        instruction = _SYNTHETIC_INSTRUCTIONS[i % len(_SYNTHETIC_INSTRUCTIONS)]
        response = _SYNTHETIC_RESPONSES[i % len(_SYNTHETIC_RESPONSES)]
        samples.append(
            {
                "prompt": ALPACA_TEMPLATE.format(instruction=instruction, input=""),
                "response": response,
            }
        )
    log.info("generated %d synthetic samples (seed=%d)", n, seed)
    return Dataset.from_list(samples)


# ------------------------------------------------------------------ alpaca

def _iter_json_lines(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        else:  # {"train": [...]} or {"data": [...]}
            for key in ("train", "data", "samples"):
                if key in data:
                    yield from data[key]
                    break
            else:
                raise ValueError(f"unsupported alpaca JSON shape: {list(data)}")


def load_alpaca(path: str | Path) -> Dataset:
    """Load Alpaca-style JSON/JSONL into prompt/response pairs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    samples: list[dict[str, str]] = []
    for raw in _iter_json_lines(path):
        instruction = str(raw.get("instruction", "")).strip()
        output = str(raw.get("output", "")).strip()
        if not instruction or not output:
            continue
        inp = str(raw.get("input", "")).strip()
        prompt = ALPACA_TEMPLATE.format(instruction=instruction, input=inp)
        samples.append({"prompt": prompt, "response": output})
    if not samples:
        raise ValueError(f"no valid alpaca samples in {path}")
    log.info("loaded %d alpaca samples from %s", len(samples), path)
    return Dataset.from_list(samples)


# ------------------------------------------------------------------ sharegpt

_ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def load_sharegpt(path: str | Path) -> Dataset:
    """Load ShareGPT-style multi-turn conversations.

    Each conversation is split into training pairs: every assistant turn
    becomes a (prompt=history up to that turn, response=turn) sample.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    samples: list[dict[str, str]] = []
    for raw in _iter_json_lines(path):
        convs = raw.get("conversations") or raw.get("messages")
        if not convs:
            continue
        messages: list[dict[str, str]] = []
        for turn in convs:
            role = _ROLE_MAP.get(str(turn.get("from", turn.get("role", ""))).lower())
            value = str(turn.get("value", "")).strip()
            if role is None or not value:
                continue
            messages.append({"role": role, "content": value})
        # every assistant turn is a supervised target
        for i in range(1, len(messages)):
            if messages[i]["role"] == "assistant":
                history = messages[: i + 1]
                prompt_msgs = history[:-1]
                response = history[-1]["content"]
                samples.append({"messages": prompt_msgs, "response": response})
    if not samples:
        raise ValueError(f"no valid sharegpt conversations in {path}")
    log.info("loaded %d sharegpt turn-samples from %s", len(samples), path)
    return Dataset.from_list(samples)


# ------------------------------------------------------------------ plain

def load_plain(path: str | Path) -> Dataset:
    """Load raw text documents for next-token prediction."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in (".json", ".jsonl"):
        samples = [{"text": str(r["text"]).strip()} for r in _iter_json_lines(path) if r.get("text")]
    else:  # .txt
        samples = [{"text": t.strip()} for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
    if not samples:
        raise ValueError(f"no text samples in {path}")
    log.info("loaded %d plain-text samples from %s", len(samples), path)
    return Dataset.from_list(samples)


# ------------------------------------------------------------------ dispatch

def load_dataset_by_format(fmt: str, path: str | None, synthetic_n: int = 64, seed: int = 42) -> Dataset:
    """Unified loader entry point used by the train pipeline."""
    if fmt == "synthetic":
        return load_synthetic(synthetic_n, seed)
    if path is None:
        raise ValueError(f"data.path is required for format={fmt!r}")
    if fmt == "alpaca":
        return load_alpaca(path)
    if fmt == "sharegpt":
        return load_sharegpt(path)
    if fmt == "plain":
        return load_plain(path)
    raise ValueError(f"unknown data format: {fmt!r} (expected alpaca|sharegpt|plain|synthetic)")
