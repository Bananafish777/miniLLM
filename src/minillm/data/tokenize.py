"""Tokenization & collation for the finetuning pipeline.

Masking strategy
----------------
Instruction-style samples (alpaca / synthetic / sharegpt) are converted to
``(prompt_ids, response_ids)`` pairs; labels mask the prompt with ``-100`` so
the model only learns to produce the response (completion-only loss).

Plain-text samples use standard next-token prediction (labels == input_ids).

Chat templates
--------------
ShareGPT samples use ``tokenizer.apply_chat_template`` when the tokenizer
provides one (e.g. Qwen2.5); otherwise a plain ``\n\n``-joined text fallback
is used so the pipeline works with any tokenizer.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

EOS_SUFFIX = "\n"  # appended to responses without EOS token (text-based formats)

# ------------------------------------------------------------------ helpers

def _tokenize_pair(tokenizer, prompt: str, response: str, max_seq_len: int) -> dict[str, list[int]]:
    """Tokenize (prompt, response) with prompt-masked labels and truncation.

    Budget plan (eos accounted for): response keeps at least a 16-token floor
    and at most ``max_seq_len - 1 - 16`` tokens; the prompt tail fills the rest.
    """
    eos_id = tokenizer.eos_token_id
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids

    resp_cap = max_seq_len - 1 - 16 if eos_id is not None else max_seq_len - 16
    if len(resp_ids) > resp_cap:
        resp_ids = resp_ids[:resp_cap]
    budget = max_seq_len - len(resp_ids) - (1 if eos_id is not None else 0)
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]  # keep the tail (closest to response)

    if eos_id is not None:
        resp_ids = resp_ids + [eos_id]

    input_ids = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids
    return {"input_ids": input_ids, "labels": labels}


def _format_messages_text(messages: list[dict[str, str]]) -> str:
    """Chat-template fallback: join turns as plain text."""
    parts = []
    for m in messages:
        role = "Human" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content']}")
    return "\n\n".join(parts) + "\n\nAssistant: "


# ------------------------------------------------------------------ main entry

def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_seq_len: int = 2048,
    num_proc: int = 1,
) -> Dataset:
    """Convert a loader-output dataset into tokenized ``input_ids``/``labels``."""

    def _tokenize(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        for i in range(len(examples.get("prompt", [""]))):
            prompt, response = examples["prompt"][i], examples["response"][i]
            out = _tokenize_pair(tokenizer, prompt, response, max_seq_len)
            input_ids.append(out["input_ids"])
            labels.append(out["labels"])
        return {"input_ids": input_ids, "labels": labels}

    def _tokenize_plain(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        for text in examples["text"]:
            ids = tokenizer(text, add_special_tokens=False).input_ids[:max_seq_len]
            if tokenizer.eos_token_id is not None:
                ids = ids + [tokenizer.eos_token_id]
            input_ids.append(ids)
            labels.append(list(ids))  # next-token prediction
        return {"input_ids": input_ids, "labels": labels}

    if "text" in dataset.column_names:
        return dataset.map(_tokenize_plain, batched=True, remove_columns=dataset.column_names, num_proc=num_proc)

    if "messages" in dataset.column_names:

        def _tokenize_chat(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            input_ids, labels = [], []
            for i in range(len(examples["messages"])):
                prompt_msgs: list[dict[str, str]] = examples["messages"][i]
                response: str = examples["response"][i]
                if tokenizer.chat_template is not None:
                    prompt = tokenizer.apply_chat_template(
                        prompt_msgs, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt = _format_messages_text(prompt_msgs)
                out = _tokenize_pair(tokenizer, prompt, response, max_seq_len)
                input_ids.append(out["input_ids"])
                labels.append(out["labels"])
            return {"input_ids": input_ids, "labels": labels}

        return dataset.map(_tokenize_chat, batched=True, remove_columns=dataset.column_names, num_proc=num_proc)

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names, num_proc=num_proc)


# ------------------------------------------------------------------ collator

class CompletionCollator:
    """Pad tokenized samples to the longest sequence in the batch.

    - ``input_ids``/``attention_mask`` padded with ``pad_token_id`` / 0
    - ``labels`` padded with -100 (ignored by the loss)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                log.info("pad_token unset; reusing eos_token as pad_token")
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                log.info("pad_token unset; added '<pad>' special token")
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
