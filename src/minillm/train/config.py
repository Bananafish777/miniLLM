"""Pydantic configuration models for the finetuning pipeline.

Every YAML file under ``configs/train/`` validates against :class:`TrainRunConfig`.
CLI dotted overrides (``--override data.synthetic_n=128``) mutate the YAML dict
before validation, so any field below is overridable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Input data configuration."""

    format: Literal["alpaca", "sharegpt", "plain", "synthetic"] = "synthetic"
    path: str | None = Field(default=None, description="dataset file (json/jsonl/txt); unused for synthetic")
    val_size: float = Field(default=0.05, ge=0.0, lt=0.5, description="validation split ratio")
    max_seq_len: int = Field(default=2048, ge=16, description="max tokens per sample")
    synthetic_n: int = Field(default=64, ge=1, description="sample count for format=synthetic")
    num_proc: int = Field(default=1, ge=1, description="dataset.map processes")


class ModelConfig(BaseModel):
    """Base model configuration."""

    name_or_path: str = Field(..., description="HF hub id or local path; ignored when source=scratch")
    source: Literal["hub", "scratch"] = Field(
        default="hub", description="hub: load pretrained; scratch: build tiny GPT-2 (offline smoke tests)"
    )
    tokenizer_name: str | None = Field(default=None, description="separate tokenizer; defaults to name_or_path")
    dtype: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    attn_impl: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    trust_remote_code: bool = False
    load_in_4bit: bool = Field(default=False, description="QLoRA 4-bit quantization (GPU only)")


class LoRAConfig(BaseModel):
    """LoRA hyperparameters (used when finetune_mode is lora|qlora)."""

    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] | None = Field(
        default=None, description="None = auto-detect from architecture"
    )
    bias: Literal["none", "all", "lora_only"] = "none"


class TrainArgsConfig(BaseModel):
    """transformers.TrainingArguments surface (the subset the platform exposes)."""

    output_dir: str = Field(..., description="run root: checkpoints, adapter, metrics.json")
    seed: int = 42
    num_train_epochs: float = Field(default=3.0, ge=0)
    max_steps: int = Field(default=-1, description="-1 = driven by epochs")
    per_device_train_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    logging_steps: int = Field(default=10, ge=1)
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = Field(default=200, ge=1)
    save_steps: int = Field(default=500, ge=1)
    save_total_limit: int = Field(default=2, ge=0)
    gradient_checkpointing: bool = False
    load_best_model_at_end: bool = False


class EvalConfig(BaseModel):
    """Post-training generation evaluation."""

    num_samples: int = Field(default=5, ge=0, description="0 = skip generation samples")
    max_new_tokens: int = Field(default=128, ge=1)
    do_sample: bool = True
    temperature: float = Field(default=0.7, gt=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    prompts: list[str] = Field(default_factory=list, description="custom prompts; empty = sample from val split")


class ExportConfig(BaseModel):
    """Model export & registration."""

    merge_adapter: bool = Field(default=True, description="merge LoRA into base weights (lora/qlora modes)")
    save_dir: str | None = Field(default=None, description="default: <output_dir>/export")
    quantize: Literal["none", "gptq", "awq"] = Field(
        default="none", description="post-training quantization (landing in later milestones)"
    )
    register_mlflow: bool = Field(default=False, description="register artifacts in MLflow (needs MLFLOW_TRACKING_URI)")


class TrainRunConfig(BaseModel):
    """Top-level config for one finetuning run."""

    experiment_name: str = "minillm"
    finetune_mode: Literal["lora", "qlora", "full"] = "lora"
    data: DataConfig
    model: ModelConfig
    lora: LoRAConfig | None = Field(default=None, description="required when finetune_mode is lora|qlora")
    train: TrainArgsConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    def model_validate_after_defaults(self):
        if self.finetune_mode in ("lora", "qlora") and self.lora is None:
            raise ValueError("finetune_mode=lora/qlora requires a `lora` section in the config")
        if self.finetune_mode == "qlora" and not self.model.load_in_4bit:
            raise ValueError("finetune_mode=qlora requires model.load_in_4bit=true")
        return self
