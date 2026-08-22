"""SGLang / vLLM launcher command generation tests (GPU vs MLX modes)."""

from __future__ import annotations

from minillm.common.utils import load_model_config
from minillm.serve.config import ServeConfig
from minillm.serve.launch import sglang_command, vllm_command


def _cfg(**overrides) -> ServeConfig:
    base = {
        "engine": "sglang",
        "model": {"name_or_path": "data/models/Qwen3-0.6B-4bit"},
    }
    base.update(overrides)
    return ServeConfig(**base)


def test_sglang_command_gpu_mode():
    cmd = sglang_command(_cfg())
    assert cmd[0].endswith("python")  # 默认当前解释器
    assert "--model-path" in cmd
    assert "--mem-fraction-static" in cmd
    assert "--disable-cuda-graph" not in cmd
    assert "--enable-metrics" in cmd


def test_sglang_command_mlx_mode():
    cfg = _cfg(sglang_use_mlx=True, engine_python=".venv-sglang/bin/python")
    cmd = sglang_command(cfg)
    assert cmd[0] == ".venv-sglang/bin/python"
    assert "--disable-cuda-graph" in cmd
    assert "--mem-fraction-static" not in cmd  # MLX 模式不使用
    assert "--enable-metrics" in cmd


def test_sglang_metrics_opt_out():
    cmd = sglang_command(_cfg(sglang_metrics=False))
    assert "--enable-metrics" not in cmd


def test_sglang_mac_config_loads():
    cfg = load_model_config(ServeConfig, "configs/serve/sglang_mac.yaml")
    assert cfg.sglang_use_mlx is True
    assert cfg.engine_python == ".venv-sglang/bin/python"
    assert cfg.model.name_or_path == "data/models/Qwen3-0.6B"
    cmd = sglang_command(cfg)  # type: ignore[arg-type]
    assert "--disable-cuda-graph" in cmd


# ---------------------------------------------------------------- vLLM (MLX)


def test_vllm_command_gpu_mode():
    cmd = vllm_command(_cfg(engine="vllm"))
    assert cmd[0] == "vllm"
    assert "--gpu-memory-utilization" in cmd
    assert "--tensor-parallel-size" in cmd


def test_vllm_command_mlx_mode():
    cfg = _cfg(engine="vllm", vllm_use_mlx=True, engine_python=".venv-vllm-metal/bin/python")
    cmd = vllm_command(cfg)
    assert cmd[0] == ".venv-vllm-metal/bin/vllm"  # 从 engine_python 推断
    assert "--gpu-memory-utilization" not in cmd   # Metal 无显存管理概念
    assert "--tensor-parallel-size" not in cmd
    assert "--max-num-seqs" in cmd
    assert "--enable-prefix-caching" in cmd


def test_vllm_mac_config_loads():
    cfg = load_model_config(ServeConfig, "configs/serve/vllm_mac.yaml")
    assert cfg.vllm_use_mlx is True
    assert cfg.engine_python == ".venv-vllm-metal/bin/python"
    cmd = vllm_command(cfg)  # type: ignore[arg-type]
    assert cmd[0] == ".venv-vllm-metal/bin/vllm"
    assert "--gpu-memory-utilization" not in cmd
