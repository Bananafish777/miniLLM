"""Hub-path smoke test: load a real (tiny) pretrained model + tokenizer.

The checkpoint is fetched into ``data/models/`` by ``scripts/fetch_model.sh``
(see ``make smoke-hub``) because some networks block the Python hub download
path while curl works. The test itself verifies the real-model path:
``AutoModelForCausalLM.from_pretrained`` + ``AutoTokenizer`` + LoRA training
on an actual Llama checkpoint.

Marked `hub` so it is excluded from the default offline unit run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from minillm.common.utils import load_model_config
from minillm.train.config import TrainRunConfig
from minillm.train.trainer import run_train

pytestmark = pytest.mark.hub

MODEL_DIR = Path("data/models/tiny-random-LlamaForCausalLM")


@pytest.fixture(scope="module", autouse=True)
def _ensure_endpoint():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield


def test_hub_tiny_llama_lora(tmp_path):
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip(
            "model not fetched — run `make smoke-hub` (fetches via scripts/fetch_model.sh)"
        )
    cfg = load_model_config(
        TrainRunConfig,
        "configs/train/smoke_hub.yaml",
        overrides=[f"train.output_dir={tmp_path}/run"],
    )
    summary = run_train(cfg)  # type: ignore[arg-type]
    assert summary["finetune_mode"] == "lora"
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "run" / "export" / "model.safetensors").exists()
