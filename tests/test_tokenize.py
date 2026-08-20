"""Tokenization & masking tests (offline, scratch tokenizer)."""

from __future__ import annotations

from minillm.data import CompletionCollator, load_synthetic, tokenize_dataset
from minillm.train.model import build_scratch_tokenizer


def _make_tokenizer():
    ds = load_synthetic(8)
    texts = [f"{p}{r}" for p, r in zip(ds["prompt"], ds["response"])]
    return build_scratch_tokenizer(texts, vocab_size=256)


def test_tokenize_masks_prompt():
    tok = _make_tokenizer()
    ds = load_synthetic(4)
    tok_ds = tokenize_dataset(ds, tok, max_seq_len=64)
    row = tok_ds[0]
    n_prompt = len(tok(ds[0]["prompt"], add_special_tokens=False).input_ids)
    # prompt 部分 labels 全为 -100
    assert row["labels"][:n_prompt] == [-100] * n_prompt
    # response 部分 labels 等于 input_ids
    assert row["labels"][n_prompt:] == row["input_ids"][n_prompt:]
    assert len(row["input_ids"]) == len(row["labels"])


def test_truncation_keeps_response():
    tok = _make_tokenizer()
    ds = load_synthetic(2)
    tok_ds = tokenize_dataset(ds, tok, max_seq_len=32)
    for row in tok_ds:
        assert len(row["input_ids"]) <= 32
        # response 尾部至少保留 16 个 token
        assert len(row["labels"]) - sum(1 for x in row["labels"] if x == -100) >= 16


def test_collator_pads_batch():
    tok = _make_tokenizer()
    ds = load_synthetic(4)
    tok_ds = tokenize_dataset(ds, tok, max_seq_len=64)
    collator = CompletionCollator(tok)
    batch = collator([tok_ds[i] for i in range(4)])
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape[1] == batch["labels"].shape[1]
    assert batch["labels"].shape[1] == batch["attention_mask"].shape[1]
    # pad 位置 label 为 -100
    import torch

    pad_mask = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][pad_mask] == -100)
