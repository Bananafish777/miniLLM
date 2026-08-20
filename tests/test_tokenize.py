"""Tokenization & masking tests (offline, scratch tokenizer)."""

from __future__ import annotations

from minillm.data import CompletionCollator, load_synthetic, tokenize_dataset
from minillm.train.model import build_scratch_tokenizer


def _make_tokenizer():
    ds = load_synthetic(8)
    texts = [f"{p}{r}" for p, r in zip(ds["prompt"], ds["response"], strict=True)]
    return build_scratch_tokenizer(texts, vocab_size=256)


def test_tokenize_masks_prompt():
    tok = _make_tokenizer()
    ds = load_synthetic(4)
    tok_ds = tokenize_dataset(ds, tok, max_seq_len=128)  # generous: no truncation
    row = tok_ds[0]
    # response 部分 labels 等于 input_ids，且从第一个非 -100 开始
    resp_start = next(i for i, x in enumerate(row["labels"]) if x != -100)
    assert row["labels"][:resp_start] == [-100] * resp_start
    assert row["labels"][resp_start:] == row["input_ids"][resp_start:]
    assert len(row["input_ids"]) == len(row["labels"])
    # prompt 部分与原始 tokenize 一致
    n_prompt = len(tok(ds[0]["prompt"], add_special_tokens=False).input_ids)
    assert resp_start == n_prompt


def test_truncation_keeps_response():
    tok = _make_tokenizer()
    ds = load_synthetic(2)
    tok_ds = tokenize_dataset(ds, tok, max_seq_len=32)
    for i, row in enumerate(tok_ds):
        assert len(row["input_ids"]) <= 32
        resp_len = sum(1 for x in row["labels"] if x != -100)
        orig_resp_len = len(tok(ds[i]["response"], add_special_tokens=False).input_ids)
        # response 保留 min(原始长度, 15) + eos；prompt 截尾填充剩余
        assert resp_len == min(orig_resp_len, 15) + 1
        # prompt 截尾：input_ids 尾部 = 完整 response token 序列
        assert row["input_ids"][-resp_len:] == tok(ds[i]["response"], add_special_tokens=False).input_ids[: min(orig_resp_len, 15)] + [tok.eos_token_id]


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
