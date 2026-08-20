"""Post-training generation evaluation: sample prompts → generated responses."""

from __future__ import annotations

import logging

import torch

from minillm.train.config import EvalConfig

log = logging.getLogger(__name__)


def evaluate_generations(
    model,
    tokenizer,
    cfg: EvalConfig,
    device: str,
    fallback_prompts: list[str] | None = None,
) -> list[dict[str, str]]:
    """Generate responses for configured prompts (or a fallback sample) and return them.

    Pure inference: no gradients, model left in eval mode on return.
    """
    if cfg.num_samples <= 0:
        return []
    prompts = list(cfg.prompts)
    if not prompts:
        prompts = [p for p in (fallback_prompts or []) if p][: cfg.num_samples]
    if not prompts:
        log.warning("no prompts available for generation evaluation")
        return []

    model.eval()
    results: list[dict[str, str]] = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append({"prompt": prompt, "generation": text})
    log.info("generated %d evaluation samples", len(results))
    return results
