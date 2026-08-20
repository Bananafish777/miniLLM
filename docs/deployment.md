# 部署手册（M4 — docker-compose；M5 — Kubernetes）

## 架构

```
                 ┌─────────────┐
   客户端/压测 ──►│  端口映射区   │ 8000=vLLM · 30000=SGLang · 8001=serve-hf
                 └──────┬──────┘
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   ┌─────────┐    ┌──────────┐    ┌───────────┐
   │  vllm   │    │  sglang  │    │  serve-hf │   (GPU 服务经 docker-compose.gpu.yml 声明设备)
   └────┬────┘    └────┬─────┘    └─────┬─────┘
        └──────────────┴───────────────┘
        scrape ▼ (Prometheus 15s)
   ┌───────────┐     ┌────────┐     ┌─────────┐
   │ Prometheus │◄───►│ Grafana │     │  mlflow │
   │  +DCGM 指标 │     │ 面板预置 │     │ 实验追踪 │
   └───────────┘     └────────┘     └─────────┘
```

## 快速开始

### 演示模式（无 GPU：监控 + HF 推理 + MLflow）

```bash
cd deploy/compose
cp .env.example .env
docker compose --env-file .env up -d prometheus grafana serve-hf mlflow

# 验证
curl http://127.0.0.1:8001/v1/models          # HF OpenAI 兼容服务
open http://127.0.0.1:3000                    # Grafana（admin/admin，面板已预置）
open http://127.0.0.1:9090/targets             # Prometheus 抓取目标
```

### GPU 全栈（vLLM + SGLang + DCGM + 监控）

```bash
cd deploy/compose
cp .env.example .env
# 修改 .env: VLLM_MODEL/SGLANG_MODEL 指向实际微调产物
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --env-file .env --profile gpu up -d

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/runs/qwen25-1.5b-lora/export","messages":[{"role":"user","content":"hi"}]}'
```

### 训练任务（一次性容器）

```bash
cd deploy/compose
# CPU 训练（冒烟）：TRAIN_CONFIG=configs/train/smoke_scratch.yaml
docker compose --env-file .env --profile train run --rm train

# GPU 训练（LoRA Qwen2.5-1.5B）：
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --env-file .env --profile train run --rm train
# 指标自动入库 MLflow（http://127.0.0.1:5000）
```

## 服务与端口

| 服务 | 端口 | 说明 | 镜像 |
| --- | --- | --- | --- |
| vllm | 8000 | OpenAI 兼容推理（Paged Attention/Continuous Batching/KV Cache） | `vllm/vllm-openai:v0.12.0` |
| sglang | 30000 | 对比引擎 | `lmsysorg/sglang:latest` |
| serve-hf | 8001 | 平台 Transformers 服务（无 GPU 演示） | 自建 `Dockerfile.serve` |
| train | — | 一次性训练任务 | 自建 `Dockerfile.train` |
| mlflow | 5000 | 实验追踪 + 模型注册 | `ghcr.io/mlflow/mlflow` |
| prometheus | 9090 | 指标采集（15s） | `prom/prometheus:v2.53.0` |
| grafana | 3000 | 可视化（面板预置） | `grafana/grafana:11.1.0` |
| dcgm-exporter | 9400 | GPU 硬件指标 | `nvidia/dcgm-exporter:3.3.5` |

## 监控内容

- **Grafana 面板（自动预置）**：GPU 利用率/显存/温度、推理吞吐（tokens/s）、
  vLLM 运行/排队请求、TTFT p50/p99、KV Cache 命中率与使用率、请求速率
- **告警规则**（Prometheus）：服务不可达、GPU 显存 >90%、GPU 利用率异常、
  vLLM 排队 >50、TTFT-p99 >5s、abort 率 >10%
- 数据源：vLLM `/metrics`（`vllm:*`）、serve-hf `/metrics`（`minillm_*`）、DCGM（`DCGM_FI_DEV_*`）

## 关键设计

1. **双 compose 文件**：`docker-compose.yml`（基础）+ `docker-compose.gpu.yml`（override 设备声明）
   —— CPU 演示与 GPU 生产共享同一编排源；
2. **Profile 门控**：GPU 服务（vllm/sglang/dcgm）与训练任务按需启用，不污染 CPU 栈；
3. **同一镜像双环境**：Linux 下 pip 安装的 torch 自带 CUDA 运行库，训练/服务镜像
   在 GPU（+nvidia-container-toolkit）与 CPU 主机均可运行；
4. **健康检查**：服务均带 healthcheck（/health），供编排与 HPA 使用；
5. **无 Docker 环境验证**：`make deploy-validate` 静态校验全部编排/监控工件（见 tests/test_deploy_artifacts.py）。

## 环境变量（deploy/compose/.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HF_ENDPOINT` | hf-mirror.com | HuggingFace 镜像 |
| `VLLM_MODEL` / `SGLANG_MODEL` | runs/... | 模型路径（容器内 /models） |
| `VLLM_GPU_MEM_UTIL` | 0.85 | KV Cache 显存预算 |
| `VLLM_MAX_NUM_SEQS` | 256 | Continuous Batching 并发上限 |
| `VLLM_PREFIX_CACHING` | true | 前缀缓存 |
| `VLLM_GPU_COUNT` / `SGLANG_GPU_COUNT` | 1 | 张量并行卡数 |
| `GRAFANA_ADMIN_PASSWORD` | admin | Grafana 密码 |
