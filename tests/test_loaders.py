"""Data loader tests (offline, tmp files)."""

from __future__ import annotations

import json

import pytest

from minillm.data import (
    load_alpaca,
    load_dataset_by_format,
    load_plain,
    load_sharegpt,
    load_synthetic,
)


def test_synthetic_loader():
    ds = load_synthetic(16, seed=42)
    assert len(ds) == 16
    assert "prompt" in ds.column_names and "response" in ds.column_names
    assert "### Instruction:" in ds[0]["prompt"]
    assert ds[0]["response"].strip()


def test_synthetic_deterministic():
    a = load_synthetic(8, seed=1)
    b = load_synthetic(8, seed=1)
    assert a[0] == b[0]


def test_alpaca_loader_json(tmp_path):
    samples = [
        {"instruction": "求和", "input": "1+1", "output": "2"},
        {"instruction": "无输入", "output": "好的"},
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(samples), encoding="utf-8")
    ds = load_alpaca(path)
    assert len(ds) == 2
    assert "1+1" in ds[0]["prompt"] and ds[0]["response"] == "2"


def test_alpaca_loader_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"instruction": "A", "output": "B"}) + "\n" + json.dumps({"instruction": "C", "output": "D"}),
        encoding="utf-8",
    )
    ds = load_alpaca(path)
    assert len(ds) == 2


def test_sharegpt_loader_multi_turn(tmp_path):
    conv = {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "gpt", "value": "你好！"},
            {"from": "human", "value": "你是谁"},
            {"from": "gpt", "value": "我是 AI 助手"},
        ]
    }
    path = tmp_path / "conv.json"
    path.write_text(json.dumps([conv]), encoding="utf-8")
    ds = load_sharegpt(path)
    assert len(ds) == 2  # 两个 assistant turn -> 两个样本
    assert ds[0]["response"] == "你好！"
    assert ds[0]["messages"][-1]["content"] == "你好"
    assert ds[1]["messages"][-1]["content"] == "你是谁"


def test_plain_loader(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("第一行文本\n第二行文本\n", encoding="utf-8")
    ds = load_plain(path)
    assert len(ds) == 2
    assert ds[0]["text"] == "第一行文本"


def test_dispatch():
    ds = load_dataset_by_format("synthetic", None, synthetic_n=8)
    assert len(ds) == 8
    with pytest.raises(ValueError):
        load_dataset_by_format("alpaca", None)
    with pytest.raises(ValueError):
        load_dataset_by_format("bogus", None)
