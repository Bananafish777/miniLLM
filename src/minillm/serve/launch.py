"""Engine launcher: dispatch ``minillm serve`` to the right backend.

- ``engine=hf``     → run the in-process FastAPI OpenAI-compatible server
- ``engine=vllm``   → launch ``vllm serve`` (local CLI) or print the docker command
- ``engine=sglang`` → launch ``python -m sglang.launch_server`` (local CLI) or print docker command
"""

from __future__ import annotations

import logging
import shutil
import sys

from minillm.serve.config import ServeConfig
from minillm.serve.engine import HFAdapter, SGLangAdapter, VLLMAdapter

log = logging.getLogger(__name__)


def build_client(cfg: ServeConfig):
    """Return an EngineClient for the configured engine.

    hf → in-process adapter; vllm/sglang → HTTP clients (external server).
    """
    base = f"http://{cfg.host}:{cfg.port}"
    if cfg.engine == "hf":
        return HFAdapter(
            cfg.model.name_or_path,
            dtype=cfg.model.dtype,
            attn_impl=cfg.model.attn_impl,
            tokenizer_name=cfg.model.tokenizer_name,
            trust_remote_code=cfg.model.trust_remote_code,
            max_model_len=cfg.model.max_model_len,
        )
    if cfg.engine == "vllm":
        return VLLMAdapter(base, cfg.model.name_or_path, timeout=cfg.request_timeout)
    if cfg.engine == "sglang":
        return SGLangAdapter(base, cfg.model.name_or_path, timeout=cfg.request_timeout)
    raise ValueError(f"unknown engine: {cfg.engine}")


def vllm_command(cfg: ServeConfig) -> list[str]:
    """The ``vllm serve`` command equivalent of this config (also used in docs/docker)."""
    return [
        "vllm", "serve", cfg.model.name_or_path,
        "--host", cfg.host, "--port", str(cfg.port),
        "--max-model-len", str(cfg.model.max_model_len),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--tensor-parallel-size", str(cfg.tensor_parallel_size),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
        *(["--enable-prefix-caching"] if cfg.enable_prefix_caching else []),
    ]


def sglang_command(cfg: ServeConfig) -> list[str]:
    """The ``sglang.launch_server`` command equivalent of this config.

    ``sglang_use_mlx`` 时面向 Apple Silicon（MLX runtime）：加 ``--disable-cuda-graph``，
    调用方需通过环境变量 ``SGLANG_USE_MLX=1`` 启用（见 serve()）。
    ``engine_python`` 允许指定独立 venv 的解释器（如 .venv-sglang/bin/python）。
    """
    cmd = [
        cfg.engine_python or sys.executable, "-m", "sglang.launch_server",
        "--model-path", cfg.model.name_or_path,
        "--host", cfg.host, "--port", str(cfg.port),
        "--tp", str(cfg.tensor_parallel_size),
    ]
    if cfg.sglang_metrics:
        cmd.append("--enable-metrics")
    if cfg.sglang_use_mlx:
        cmd.append("--disable-cuda-graph")
    else:
        cmd += ["--mem-fraction-static", str(cfg.gpu_memory_utilization)]
    return cmd


def docker_hint(cfg: ServeConfig) -> str:
    """One-line docker run command for GPU environments (used when CLI absent)."""
    if cfg.engine == "vllm":
        image = "vllm/vllm-openai:latest"
        cmd = " ".join(vllm_command(cfg)[1:])  # strip leading "vllm"
        return (
            f"docker run --gpus all --shm-size 8g -p {cfg.port}:{cfg.port} "
            f"{image} {cmd}"
        )
    image = "lmsysorg/sglang:latest"
    args = sglang_command(cfg)[2:]  # strip "python -m sglang.launch_server"
    return (
        f"docker run --gpus all --shm-size 8g -p {cfg.port}:{cfg.port} "
        f"{image} python3 -m sglang.launch_server {' '.join(args)}"
    )


def serve(cfg: ServeConfig) -> int:
    """Run the configured serving engine (blocking)."""
    if cfg.engine == "hf":
        import uvicorn

        from minillm.serve.server import build_app

        log.info("starting HF OpenAI-compatible server on %s:%s (model=%s)",
                 cfg.host, cfg.port, cfg.model.name_or_path)
        uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
        return 0

    if cfg.engine == "vllm":
        cmd = vllm_command(cfg)
        if shutil.which("vllm"):
            log.info("launching vLLM: %s", " ".join(cmd))
            import subprocess

            return subprocess.call(cmd)
        log.warning("vLLM CLI not found (CUDA environment required). Run instead:\n  %s", docker_hint(cfg))
        return 2

    if cfg.engine == "sglang":
        cmd = sglang_command(cfg)
        if cfg.sglang_use_mlx:
            log.info("SGLang MLX runtime (Apple Silicon) — SGLANG_USE_MLX=1")
            import os

            env = {**os.environ, "SGLANG_USE_MLX": "1"}
            # engine_python 显式指定了引擎解释器（独立 venv），直接信任启动；
            # 否则要求当前解释器可 import sglang
            if cfg.engine_python or _sglang_python_available():
                log.info("launching SGLang(MLX): %s", " ".join(cmd))
                import subprocess

                return subprocess.call(cmd, env=env)
            log.warning("SGLang(MLX) 未安装。Mac 上安装步骤见 docs/serving.md（独立 venv + srt_mps extra）。")
            return 2
        if shutil.which("sglang.launch_server"):
            log.info("launching SGLang: %s", " ".join(cmd))
            import subprocess

            return subprocess.call(cmd)
        log.warning("SGLang CLI not found (CUDA environment required). Run instead:\n  %s", docker_hint(cfg))
        return 2

    raise ValueError(f"unknown engine: {cfg.engine}")


def _sglang_python_available() -> bool:
    """True when the current interpreter can import sglang (used for MLX mode)."""
    import importlib.util

    return importlib.util.find_spec("sglang") is not None
