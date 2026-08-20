"""Config loading / override / validation tests (offline)."""

from __future__ import annotations

import pytest

from minillm.common.utils import apply_overrides, load_model_config, load_yaml
from minillm.train.config import TrainRunConfig

CFG = "configs/train/smoke_scratch.yaml"


def test_load_yaml_roundtrip():
    data = load_yaml(CFG)
    assert data["finetune_mode"] == "lora"
    assert data["model"]["source"] == "scratch"


def test_config_validates_full():
    cfg = load_model_config(TrainRunConfig, CFG)
    assert cfg.train.max_steps == 3
    assert cfg.data.max_seq_len == 128
    assert cfg.lora is not None and cfg.lora.r == 8


def test_apply_overrides_nested():
    data = load_yaml(CFG)
    apply_overrides(data, ["train.learning_rate=1e-4", "data.synthetic_n=64", "lora.r=4"])
    assert data["train"]["learning_rate"] == 1e-4
    assert data["data"]["synthetic_n"] == 64
    assert data["lora"]["r"] == 4


def test_override_type_coercion():
    data = load_yaml(CFG)
    apply_overrides(data, ["export.merge_adapter=false", "model.dtype=fp32"])
    assert data["export"]["merge_adapter"] is False
    assert data["model"]["dtype"] == "fp32"


def test_override_missing_key_creates():
    data = load_yaml(CFG)
    apply_overrides(data, ["train.new_field=1"])
    assert data["train"]["new_field"] == 1


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_model_config(TrainRunConfig, "configs/train/does-not-exist.yaml")


def test_lora_required_for_lora_mode():
    import copy

    data = load_yaml(CFG)
    data.pop("lora")
    with pytest.raises(ValueError, match="requires a `lora` section"):
        TrainRunConfig.model_validate(copy.deepcopy(data)).model_validate_after_defaults()


def test_qlora_requires_4bit():
    import copy

    data = load_yaml("configs/train/qlora_qwen25_3b.yaml")
    data["model"]["load_in_4bit"] = False
    with pytest.raises(ValueError, match="load_in_4bit"):
        TrainRunConfig.model_validate(copy.deepcopy(data)).model_validate_after_defaults()


def test_bad_format_rejected():
    import copy

    data = load_yaml(CFG)
    data["data"]["format"] = "csv"
    with pytest.raises(ValueError):
        TrainRunConfig.model_validate(copy.deepcopy(data))
