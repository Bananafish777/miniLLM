"""Model construction: pretrained (hub) or scratch tiny GPT-2, plus LoRA setup."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from minillm.train.config import LoRAConfig, ModelConfig

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ scratch

SCRATCH_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_scratch_tokenizer(texts: list[str], vocab_size: int = 512):
    """Train a tiny BPE tokenizer on the dataset (fully offline smoke tests)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SCRATCH_SPECIAL_TOKENS, min_frequency=1
    )
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def build_scratch_model(vocab_size: int):
    """Build a tiny GPT-2 (2 layers, 64 hidden) with random weights."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=64,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    log.info("built scratch GPT-2: %d params", sum(p.numel() for p in model.parameters()))
    return model


# ------------------------------------------------------------------ hub

def build_hub_model(cfg: ModelConfig, dtype: torch.dtype):
    """Load a pretrained causal LM from the HF hub or a local path."""
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"torch_dtype": dtype, "trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_impl == "eager":
        kwargs["attn_implementation"] = "eager"
    elif cfg.attn_impl == "flash_attention_2":
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["attn_implementation"] = "sdpa"

    if cfg.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:  # pragma: no cover
            raise RuntimeError("load_in_4bit requires transformers>=4.30 with bitsandbytes") from None
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
        log.info("quantizing to 4-bit NF4 (QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **kwargs)
    return model


def build_model(cfg: ModelConfig, dtype: torch.dtype, device: str, scratch_vocab: int | None = None):
    """Unified model factory: hub pretrained or scratch tiny GPT-2."""
    if cfg.source == "scratch":
        if scratch_vocab is None:
            raise ValueError("scratch model requires tokenizer vocab size")
        model = build_scratch_model(scratch_vocab)
    else:
        model = build_hub_model(cfg, dtype)
        if device in ("cpu", "mps") and not cfg.load_in_4bit:
            model = model.to(device)
    if device == "cuda" and not cfg.load_in_4bit:
        model = model.to(device)
    return model


# ------------------------------------------------------------------ LoRA

_KNOWN_LINEAR_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "c_attn", "c_proj", "c_fc",
}


def auto_target_modules(model: nn.Module) -> list[str]:
    """Auto-detect LoRA target modules from the architecture.

    Prefers known attention/MLP projection names (Llama/Qwen/GPT-2 families);
    falls back to every Linear except the LM head.
    """
    linear_names = [
        name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)
    ]
    matched = [n for n in linear_names if n.split(".")[-1] in _KNOWN_LINEAR_SUFFIXES]
    if matched:
        return matched
    return [n for n in linear_names if not n.endswith("lm_head")]


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """Wrap the model with PEFT LoRA."""
    from peft import LoraConfig, get_peft_model

    target_modules = cfg.target_modules or auto_target_modules(model)
    log.info("LoRA target modules: %s", target_modules)
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=target_modules,
        bias=cfg.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model
