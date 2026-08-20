"""Model export & registration after finetuning."""

from __future__ import annotations

import logging
from pathlib import Path

from minillm.train.config import ExportConfig

log = logging.getLogger(__name__)


def export_model(model, tokenizer, cfg: ExportConfig, run_dir: str, finetune_mode: str) -> dict:
    """Export the finetuned model to a deployable directory.

    - lora/qlora: merge adapter into base weights (``merge_and_unload``)
    - full:       save the trained weights directly
    - quantization hooks (gptq/awq) land with the serving milestone (M2/M3)
    """
    if cfg.quantize != "none":
        raise NotImplementedError(
            f"quantize={cfg.quantize!r} is not implemented yet; "
            "post-training quantization lands in a later milestone"
        )

    export_dir = Path(cfg.save_dir) if cfg.save_dir else Path(run_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    if finetune_mode in ("lora", "qlora") and cfg.merge_adapter:
        log.info("merging LoRA adapter into base weights ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(export_dir)
        merged_kind = "lora-merged"
    else:
        model.save_pretrained(export_dir)
        merged_kind = "full" if finetune_mode == "full" else "adapter-only"

    tokenizer.save_pretrained(export_dir)
    log.info("exported model to %s (%s)", export_dir, merged_kind)
    return {"export_dir": str(export_dir), "kind": merged_kind, "quantize": cfg.quantize}
