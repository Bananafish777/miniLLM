"""Shared helpers: YAML config loading, dotted overrides, device/dtype resolution."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from minillm.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------- config io

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data)}: {p}")
    return data


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `key.path=value` overrides onto a nested dict (in place, returns it).

    Examples::

        --override train.learning_rate=1e-4
        --override model.name_or_path=Qwen/Qwen2.5-1.5B-Instruct
        --override data.synthetic_n=128
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = _parse_scalar(raw)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override path {key!r} conflicts with non-mapping value")
        node[parts[-1]] = value
        log.info("override: %s = %r", key, value)
    return data


def _parse_scalar(raw: str) -> Any:
    """Parse override values: JSON types first, then YAML scalar semantics."""
    raw = raw.strip()
    lowered = raw.lower()
    if lowered in ("true", "false", "null", "none"):
        return {"true": True, "false": False, "null": None, "none": None}[lowered]
    try:
        return json.loads(raw)  # numbers, lists, dicts, quoted strings
    except json.JSONDecodeError:
        return raw  # plain string


def load_model_config(config_cls: type[BaseModel], path: str | Path, overrides: list[str] | None = None) -> BaseModel:
    """Load, override and validate a pydantic config from YAML."""
    data = load_yaml(path)
    if overrides:
        apply_overrides(data, overrides)
    try:
        return config_cls.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"invalid config {path}:\n{e}") from e


# ---------------------------------------------------------------- misc

def set_seed(seed: int) -> None:
    """Deterministic reproducibility across torch/numpy/random."""
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    """Pick the best available compute device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> str:
    """Map a config dtype ('auto'|'fp32'|'fp16'|'bf16') to a concrete dtype name.

    CPU/MPS default to fp32 (broadest compatibility); CUDA defaults to bf16
    (modern GPUs) unless explicitly requested otherwise.
    """
    import torch

    r = requested.lower()
    if r == "auto":
        r = "bf16" if device == "cuda" else "fp32"
    if r in ("fp16", "half"):
        return "fp16"
    if r in ("bf16", "bfloat16"):
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested but not supported by this GPU")
        return "bf16"
    return "fp32"


def torch_dtype(dtype_name: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]


def count_parameters(model) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n < 1e9 else f"{n / 1e9:.2f}B"


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
