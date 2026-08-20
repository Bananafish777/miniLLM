"""Hub-path smoke test: load a real (tiny) pretrained model + tokenizer.

Requires HuggingFace hub access; use the mirror when the official hub is
blocked::

    HF_ENDPOINT=https://hf-mirror.com pytest -m hub -q

Marked `hub` so it is excluded from the default offline unit run.
"""

from __future__ import annotations

import os

import pytest

from minillm.common.utils import load_model_config
from minillm.train.config import TrainRunConfig
from minillm.train.trainer import run_train

pytestmark = pytest.mark.hub


@pytest.fixture(scope="module", autouse=True)
def _ensure_endpoint():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield


def test_hub_tiny_llama_lora(tmp_path):
    cfg = load_model_config(
        TrainRunConfig,
        "configs/train/smoke_hub.yaml",
        overrides=[f"train.output_dir={tmp_path}/run"],
    )
    summary = run_train(cfg)  # type: ignore[arg-type]
    assert summary["finetune_mode"] == "lora"
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "run" / "export" / "model.safetensors").exists()
