# 推理服务使用说明（M2）

`minillm serve` 启动 **OpenAI API 兼容**推理服务，多引擎统一通过 `EngineClient` 抽象接入：

| 引擎 | 后端 | 运行环境 | 用途 |
| --- | --- | --- | --- |
| `hf` | Transformers（进程内） | Mac / CPU / GPU | 本地开发、无优化基线、协议演示 |
| `vllm` | vLLM（外部服务/容器） | NVIDIA GPU | 主推理引擎：Paged Attention / Continuous Batching / KV Cache |
| `sglang` | SGLang（外部服务/容器） | NVIDIA GPU | Benchmark 对照组 |
| `sglang`（MLX） | SGLang MLX runtime | **Apple Silicon** | Mac 上的高性能推理（见下节） |

## Mac (Apple Silicon) 上跑 SGLang — MLX runtime

> SGLang 官方已支持 Apple Metal（通过 MLX runtime，`SGLANG_USE_MLX=1`）；vLLM 官方暂不支持 Mac（社区 PR 进行中）。

### 安装（独立 venv，避免污染项目环境）

```bash
# ① 克隆源码（Apple Metal 支持在 master，PyPI 版暂未发布）
git clone --depth 1 https://github.com/sgl-project/sglang.git .tools/sglang

# ② 独立 venv + 换 pyproject（srt_mps extra 在 pyproject_other.toml）
cd .tools/sglang
uv venv -p 3.12 ../../.venv-sglang
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml

# ③ 安装（SGLANG_BUILD_RUST_EXTS=none 跳过 Rust 扩展，无需 cargo；sgl-kernel 原生 kernel 需完整 Xcode，可选）
export SGLANG_BUILD_RUST_EXTS=none
uv pip install --python ../../.venv-sglang -e "python[srt_mps]"   # mlx + mlx-lm + torch 2.13

# ④ 下载模型（mlx-community 预量化 4bit，~335MB）
cd ../.. && scripts/fetch_model.sh mlx-community/Qwen3-0.6B-4bit
```

### 启动

```bash
SGLANG_USE_MLX=1 minillm serve --config configs/serve/sglang_mac.yaml
# → http://127.0.0.1:30000 （OpenAI API + /metrics 同端口）
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"data/models/Qwen3-0.6B-4bit","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

### 说明

- 配置 `sglang_mac.yaml` 关键字段：`sglang_use_mlx: true`（自动加 `--disable-cuda-graph`）、
  `engine_python: .venv-sglang/bin/python`（sglang 在独立 venv）、`sglang_metrics: true`（`/metrics` 同端口暴露，Benchmark 交叉验证通道可用）
- 模型：`mlx-community/<model>-4bit` 预量化直接加载；fp16 模型可加 `--quantization mlx_q4` 现场量化
- 已验证：本机 M4 Pro 跑 Qwen3-0.6B-4bit，中文生成正常、`SGLangAdapter` 全接口（models/completions/stream/metrics）通过

## 本地启动（engine=hf）

```bash
# 微型模型（冒烟）：data/models/tiny-random-LlamaForCausalLM
minillm serve --config configs/serve/hf_tiny.yaml

# 真实模型：加载 M1 微调产物（runs/<run>/export）
minillm serve --config configs/serve/hf_qwen25_1p5b.yaml
```

验证：

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<模型路径>","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<模型路径>","prompt":"从前有座山","stream":true}'
```

> 任意 OpenAI SDK 均可直接对接（`base_url="http://127.0.0.1:8000/v1"`）。

## GPU 环境（engine=vllm / sglang）

```bash
# vLLM（Paged Attention / Continuous Batching / KV Cache 配置见 configs/serve/vllm_qwen25_1p5b.yaml）
minillm serve --config configs/serve/vllm_qwen25_1p5b.yaml

# 等价 docker 命令（本机无 vllm CLI 时自动打印提示）
docker run --gpus all --shm-size 8g -p 8000:8000 vllm/vllm-openai:latest \
  serve runs/qwen25-1.5b-lora/export --gpu-memory-utilization 0.85 \
  --max-model-len 8192 --enable-prefix-caching ...

# SGLang 对照组（默认端口 30000）
minillm serve --config configs/serve/sglang_qwen25_1p5b.yaml
```

## 题目核心技术点 ↔ 配置映射（vLLM）

| 题目要求 | vLLM 机制 | 配置字段 |
| --- | --- | --- |
| Paged Attention | KV Cache 分页管理 | `gpu_memory_utilization: 0.85` |
| Continuous Batching | 请求级动态调度 | `max_num_seqs: 256`、`max_num_batched_tokens: 8192` |
| KV Cache | 显存预算与上下文窗口 | `max_model_len: 8192` |
| 前缀复用优化 | Prefix Caching | `enable_prefix_caching: true` |

## 监控指标（Prometheus 格式）

- `GET /metrics`：`minillm_requests_total`、`minillm_tokens_generated_total`、
  `minillm_request_duration_seconds`（直方图）、`minillm_ttft_seconds`（直方图）
- vLLM 服务自带 `/metrics`（`vllm:*` 系列：吞吐/排队数/cache 命中率），M3 Benchmark 交叉验证、M6 Grafana 大盘均消费这两路指标

## EngineClient（引擎抽象）

```python
from minillm.serve.engine import VLLMAdapter, SGLangAdapter, HFAdapter

# 三个引擎同一接口（M3 Benchmark 复用）
client = VLLMAdapter("http://gpu-host:8000", "runs/qwen25-1.5b-lora/export")
client.chat([{"role": "user", "content": "hi"}], max_tokens=64)
client.completions("Hello", stream=True)   # 逐 token 迭代
client.metrics()                           # Prometheus 指标字典
client.system_stats()                      # 请求/显存/进程统计
```
