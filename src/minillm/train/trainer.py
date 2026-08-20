"""Finetuning pipeline orchestrator: data → tokenize → train → evaluate → export → track.

Runs on CUDA / MPS / CPU; ``model.source=scratch`` provides a fully offline
tiny-GPT-2 path used by smoke tests and demos.
"""

from __future__ import annotations

import os
from pathlib import Path

from minillm.common.logging import get_logger
from minillm.common.utils import (
    count_parameters,
    detect_device,
    format_params,
    resolve_dtype,
    save_json,
    set_seed,
    torch_dtype,
)
from minillm.data import CompletionCollator, load_dataset_by_format, tokenize_dataset
from minillm.train.config import TrainRunConfig
from minillm.train.evaluate import evaluate_generations
from minillm.train.export import export_model
from minillm.train.model import apply_lora, build_model, build_scratch_tokenizer

log = get_logger(__name__)


def _split_dataset(dataset, val_size: float, seed: int):
    """Deterministic train/validation split."""
    if val_size <= 0 or len(dataset) < 2:
        return dataset, None
    n_val = max(1, int(len(dataset) * val_size))
    split = dataset.train_test_split(test_size=n_val, seed=seed)
    return split["train"], split["test"]


def _load_tokenizer(cfg: TrainRunConfig, device: str, texts: list[str] | None):
    """Load a hub tokenizer, or train a scratch one (fully offline)."""
    if cfg.model.source == "scratch":
        if not texts:
            raise ValueError("scratch tokenizer requires sample texts")
        return build_scratch_tokenizer(texts)
    from transformers import AutoTokenizer

    name = cfg.model.tokenizer_name or cfg.model.name_or_path
    tok = AutoTokenizer.from_pretrained(
        name, trust_remote_code=cfg.model.trust_remote_code
    )
    if tok.pad_token_id is None:  # e.g. Llama family: reuse eos
        tok.pad_token = tok.eos_token
        log.info("pad_token unset; reusing eos_token as pad_token")
    log.info("tokenizer loaded: %s (vocab=%d)", name, tok.vocab_size)
    return tok


def _build_training_args(cfg: TrainRunConfig, dtype_name: str, device: str):
    import inspect

    from transformers import TrainingArguments

    t = cfg.train
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    # transformers >=5 移除了 warmup_ratio；warmup_steps 传小数即为比例
    if "warmup_ratio" in ta_params:
        warmup_kwargs = {"warmup_ratio": t.warmup_ratio}
    else:
        warmup_kwargs = {"warmup_steps": t.warmup_ratio}
    return TrainingArguments(
        output_dir=t.output_dir,
        seed=t.seed,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        lr_scheduler_type=t.lr_scheduler_type,
        **warmup_kwargs,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        gradient_checkpointing=t.gradient_checkpointing,
        load_best_model_at_end=t.load_best_model_at_end,
        bf16=(dtype_name == "bf16" and device == "cuda"),
        fp16=(dtype_name == "fp16" and device == "cuda"),
        report_to=[],
        remove_unused_columns=False,
        save_only_model=True,
    )


def _log_mlflow(cfg: TrainRunConfig, params: dict, metrics: dict, artifacts: list[str]) -> bool:
    """Best-effort MLflow tracking; no-op unless mlflow is installed + configured."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        log.info("MLFLOW_TRACKING_URI not set — skipping experiment tracking")
        return False
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed — skipping experiment tracking")
        return False
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=Path(cfg.train.output_dir).name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifacts(artifact, artifact_path=Path(artifact).name)
    log.info("run tracked in MLflow (%s)", uri)
    return True


def run_train(cfg: TrainRunConfig) -> dict:
    """Execute the full finetuning pipeline; returns a run summary dict."""
    cfg.model_validate_after_defaults()
    set_seed(cfg.train.seed)
    device = detect_device()
    log.info("=== minillm train | experiment=%s | mode=%s | device=%s ===",
             cfg.experiment_name, cfg.finetune_mode, device)
    log.info("model: %s (dtype=%s, attn=%s)", cfg.model.name_or_path,
             resolve_dtype(cfg.model.dtype, device), cfg.model.attn_impl)

    # 1) data ---------------------------------------------------------------
    raw = load_dataset_by_format(cfg.data.format, cfg.data.path,
                                 cfg.data.synthetic_n, cfg.train.seed)
    train_raw, val_raw = _split_dataset(raw, cfg.data.val_size, cfg.train.seed)
    log.info("samples: train=%d val=%s", len(train_raw), len(val_raw) if val_raw else "n/a")

    # 2) tokenizer ----------------------------------------------------------
    texts = [f"{p}\n{r}" for p, r in zip(raw["prompt"], raw["response"], strict=True)] if "prompt" in raw.column_names else raw["text"]
    tokenizer = _load_tokenizer(cfg, device, texts)

    # 3) tokenize -----------------------------------------------------------
    train_ds = tokenize_dataset(train_raw, tokenizer, max_seq_len=cfg.data.max_seq_len, num_proc=cfg.data.num_proc)
    val_ds = tokenize_dataset(val_raw, tokenizer, max_seq_len=cfg.data.max_seq_len, num_proc=cfg.data.num_proc) if val_raw else None

    # 4) model --------------------------------------------------------------
    dtype_name = resolve_dtype(cfg.model.dtype, device)
    model = build_model(cfg.model, torch_dtype(dtype_name), device,
                        scratch_vocab=tokenizer.vocab_size if cfg.model.source == "scratch" else None)

    # 5) LoRA / QLoRA -------------------------------------------------------
    if cfg.finetune_mode in ("lora", "qlora"):
        if cfg.lora is None:
            raise ValueError("finetune_mode=lora/qlora requires a `lora` section")
        model = apply_lora(model, cfg.lora)
    total, trainable = count_parameters(model)
    log.info("parameters: total=%s trainable=%s (%.2f%%)",
             format_params(total), format_params(trainable), 100 * trainable / total)

    # 6) train --------------------------------------------------------------
    from transformers import Trainer

    args = _build_training_args(cfg, dtype_name, device)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CompletionCollator(tokenizer),
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    if val_ds is not None:
        metrics.update(trainer.evaluate())

    # 7) save adapter -------------------------------------------------------
    adapter_dir = Path(cfg.train.output_dir) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log.info("adapter saved to %s", adapter_dir)

    # 8) generation evaluation ----------------------------------------------
    fallback_prompts = []
    if val_raw is not None and "prompt" in val_raw.column_names:
        fallback_prompts = list(val_raw["prompt"])[: cfg.eval.num_samples]
    samples = evaluate_generations(model, tokenizer, cfg.eval, device, fallback_prompts)
    if samples:
        save_json(samples, Path(cfg.train.output_dir) / "eval_samples.json")
        for s in samples[:3]:
            log.info("eval sample | prompt: %s", s["prompt"][:60].replace("\n", " "))
            log.info("           | generation: %s", s["generation"][:120].replace("\n", " "))

    # 9) export --------------------------------------------------------------
    export = export_model(model, tokenizer, cfg.export, cfg.train.output_dir, cfg.finetune_mode)

    # 10) tracking + summary --------------------------------------------------
    summary = {
        "experiment": cfg.experiment_name,
        "finetune_mode": cfg.finetune_mode,
        "model": cfg.model.name_or_path,
        "device": device,
        "params_total": total,
        "params_trainable": trainable,
        "train_metrics": metrics,
        "eval_samples": len(samples),
        "export": export,
    }
    save_json(summary, Path(cfg.train.output_dir) / "metrics.json")

    mlflow_ok = _log_mlflow(
        cfg,
        params={"finetune_mode": cfg.finetune_mode, "model": cfg.model.name_or_path, "lr": cfg.train.learning_rate,
                "epochs": cfg.train.num_train_epochs, "batch_size": cfg.train.per_device_train_batch_size,
                "lora_r": cfg.lora.r if cfg.lora else None, "max_seq_len": cfg.data.max_seq_len},
        metrics={"final_eval_loss": metrics.get("eval_loss", float("nan")),
                 "train_samples_per_second": metrics.get("train_samples_per_second", 0.0),
                 "train_tokens_per_second": metrics.get("train_tokens_per_second", 0.0)},
        artifacts=[str(adapter_dir), export["export_dir"]] if mlflow_importable() else [],
    )
    summary["mlflow_logged"] = mlflow_ok

    # 训练指标入库（Pushgateway → Grafana 训练视图）
    from minillm.monitor.push import push_series

    push_ok = push_series(
        [
            {
                "name": "minillm_train_final_eval_loss",
                "help": "Final evaluation loss",
                "value": metrics.get("eval_loss", float("nan")),
                "labels": {"experiment": cfg.experiment_name, "model": cfg.model.name_or_path,
                           "mode": cfg.finetune_mode},
            },
            {
                "name": "minillm_train_tokens_per_second",
                "help": "Training throughput (tokens/s)",
                "value": metrics.get("train_tokens_per_second", 0.0),
                "labels": {"experiment": cfg.experiment_name, "model": cfg.model.name_or_path,
                           "mode": cfg.finetune_mode},
            },
        ],
        job="minillm-train",
        gateway=cfg.push_gateway,
        grouping_key={"experiment": cfg.experiment_name, "run": Path(cfg.train.output_dir).name},
    )
    summary["pushgateway_logged"] = push_ok
    log.info("=== train finished: %s | eval_loss=%s | export=%s ===",
             cfg.train.output_dir, metrics.get("eval_loss", "n/a"), export["export_dir"])
    return summary


def mlflow_importable() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False
