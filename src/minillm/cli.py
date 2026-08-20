"""minillm command-line interface.

Usage::

    minillm train --config configs/train/lora_qwen25_1p5b.yaml [--override key=value ...]
    minillm serve --config configs/serve/hf_qwen25_1p5b.yaml  [--override key=value ...]
    minillm bench --config configs/bench/matrix.yaml          # M3
"""

from __future__ import annotations

import argparse
import sys

from minillm.bench.bench import run_bench
from minillm.bench.config import BenchRunConfig
from minillm.common.logging import setup_logging
from minillm.common.utils import load_model_config
from minillm.serve.config import ServeConfig
from minillm.serve.launch import serve
from minillm.train.config import TrainRunConfig
from minillm.train.trainer import run_train


def _add_config_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--config", required=required, help="path to YAML config")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="override a nested config field, repeatable (e.g. train.learning_rate=1e-4)")


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_model_config(TrainRunConfig, args.config, args.override)
    run_train(cfg)  # type: ignore[arg-type]
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    cfg = load_model_config(ServeConfig, args.config, args.override)
    return serve(cfg)  # type: ignore[arg-type]


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_model_config(BenchRunConfig, args.config, args.override)
    run_bench(cfg)  # type: ignore[arg-type]
    return 0


def _cmd_not_ready(name: str, milestone: str) -> int:
    print(f"[minillm] `{name}` is scheduled for {milestone} — not implemented yet.", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minillm", description="End-to-end LLM infrastructure platform"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="run the finetuning pipeline (LoRA/QLoRA/full)")
    _add_config_arg(p_train)

    p_serve = sub.add_parser("serve", help="start an OpenAI-compatible inference service (M2)")
    _add_config_arg(p_serve)

    p_bench = sub.add_parser("bench", help="run the benchmark matrix (Transformers/vLLM/SGLang)")
    _add_config_arg(p_bench)

    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "train":
        return _cmd_train(args)
    if args.command == "serve":
        return _cmd_serve(args)
    return _cmd_bench(args)

    # unreachable
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
