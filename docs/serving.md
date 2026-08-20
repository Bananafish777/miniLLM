# 推理服务使用说明（M2）

`minillm serve` 启动 **OpenAI API 兼容**推理服务，多引擎统一通过 `EngineClient` 抽象接入：

| 引擎 | 后端 | 运行环境 | 用途 |
| --- | --- | --- | --- |
| `hf` | Transformers（进程内） | Mac / CPU / GPU | 本地开发、无优化基线、协议演示 |
| `vllm` | vLLM（外部服务/容器） | NVIDIA GPU | 主推理引擎：Paged Attention / Continuous Batching / KV Cache |
| `sglang` | SGLang（外部服务/容器） | NVIDIA GPU | Benchmark 对照组 |

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
