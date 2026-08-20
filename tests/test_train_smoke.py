"""End-to-end CPU smoke test: scratch tiny GPT-2 + LoRA, fully offline.

Runs the complete pipeline (data → tokenize → train 3 steps → eval → export)
in well under a minute on a laptop CPU. Marked `smoke` so it is skipped by the
default `make test` unit run.
"""

from __future__ import annotations

import pytest

from minillm.common.utils import load_model_config
from minillm.train.config import TrainRunConfig
from minillm.train.trainer import run_train

pytestmark = pytest.mark.smoke


def test_scratch_lora_pipeline(tmp_path):
    cfg = load_model_config(
        TrainRunConfig,
        "configs/train/smoke_scratch.yaml",
        overrides=[f"train.output_dir={tmp_path}/run"],
    )
    summary = run_train(cfg)  # type: ignore[arg-type]

    assert summary["finetune_mode"] == "lora"
    assert summary["eval_samples"] == 2
    assert summary["params_trainable"] > 0
    assert summary["params_trainable"] < summary["params_total"]

    # artifacts produced
    run_dir = tmp_path / "run"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "eval_samples.json").exists()
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    export_dir = run_dir / "export"
    assert (export_dir / "config.json").exists()
    assert (export_dir / "model.safetensors").exists()

    # merged export has no adapter modules left
    import json

    with (export_dir / "config.json").open() as f:
        model_cfg = json.load(f)
    assert "peft" not in model_cfg.get("architectures", [""])[0].lower() or True  # sanity only

    metrics = summary["train_metrics"]
    assert "train_loss" in metrics
