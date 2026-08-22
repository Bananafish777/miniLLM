# miniLLM — 端到端大模型基础设施平台

一套从零实现的 LLM 基础设施平台，覆盖 **模型微调 → 推理服务 → 性能基准 → 容器化部署 → 监控告警** 全链路，基于 PyTorch / vLLM / SGLang / Docker / Kubernetes / Prometheus / Grafana 技术栈。用于学习实践与求职作品集展示。

## ✨ 项目亮点

1. **三引擎统一抽象（EngineClient）**：vLLM / SGLang / Transformers 通过同一接口接入推理与压测，**"压测口径 = 生产口径"**；换引擎只改配置不改代码
2. **自研 Benchmark 系统**：asyncio 并发压测 + 流式精确计时（TTFT/ITL）+ **客户端/服务端双通道 token 交叉验证** + 内置瓶颈分析启发式
3. **双轨部署**：docker-compose（base + GPU override 双文件模式）与 Kubernetes（Helm + **Kueue GPU 配额队列** + 基于 vLLM 排队指标的 HPA）
4. **可观测性闭环**：Benchmark/训练结果经 **Pushgateway 入库 Prometheus**，Grafana 两块预置面板（全栈监控 + 跨引擎对比）
5. **无 GPU 全流程可验证**：Mac（CPU/MPS）即可跑通微调→推理→压测全链路；`make demo` 一键演示，57+ 自动化测试守护

## 核心能力

| 模块 | 技术栈 | 说明 |
| --- | --- | --- |
| 🎓 微调流水线 | PyTorch · Transformers · PEFT · MLflow | LoRA / QLoRA / 全参三路径；4 种数据格式（alpaca/sharegpt/plain/synthetic）；prompt 掩码；adapter 合并导出；实验追踪 |
| ⚡ 推理服务 | **vLLM**（主）· SGLang（对比）· **SGLang-MLX**（Apple Silicon）· Transformers（本地） | OpenAI API 兼容（流式 SSE）；Paged Attention / Continuous Batching / KV Cache 全旋钮化；多引擎适配器 |
| 📊 Benchmark | 自研 asyncio 压测客户端 | 吞吐 / TTFT / ITL / 端到端 / 显存峰值；双通道交叉验证；瓶颈分析；JSON+MD 报告 |
| 🚀 部署调度 | Docker · docker-compose · Kubernetes · Helm · Kueue | 八服务 Compose 栈；GPU override；Kueue 配额队列；HPA 弹性伸缩 |
| 📈 可观测性 | Prometheus · Grafana · DCGM · Pushgateway | GPU 利用率/显存/温度；服务吞吐/排队/TTFT 分位；6 条告警规则；Benchmark/训练结果入库 |
| 🖥️ Web 控制台 | FastAPI + 原生 JS（零构建） | 引擎状态 / GPU 指标 / Benchmark 对比 / 微调运行 一站式管理面板 |

## 系统架构

```
                    ┌─────────────────────────────────────────────┐
                    │            EngineClient 统一抽象              │
                    │  vLLM ◄─► SGLang ◄─► Transformers (HF)       │
                    └──────┬──────────────┬──────────────┬─────────┘
                           │              │              │
              ┌────────────▼───┐   ┌──────▼───────┐   ┌──▼─────────────┐
              │ 微调流水线      │   │ OpenAI 兼容   │   │ Benchmark 系统  │
              │ LoRA/QLoRA/全参 │   │ 推理服务       │   │ 并发压测/报告    │
              └──────┬─────────┘   └──────┬───────┘   └──┬─────────────┘
                     │ export/register    │ /metrics     │ Pushgateway
                     ▼                    ▼              ▼
              ┌──────────────────────────────────────────────────────┐
              │       Prometheus ◄── DCGM/vLLM/minillm 指标           │
              │            │                                         │
              │       Grafana（全栈监控 + Benchmark 对比面板）          │
              └──────────────────────────────────────────────────────┘
  部署: docker-compose（开发/演示） · Kubernetes + Helm + Kueue（生产）
```

## 快速开始

```bash
make setup                  # venv (Python 3.12) + 依赖
make demo                   # 🎬 全链路演示：微调→推理(OpenAI API)→压测→报告（~2 分钟）
```

分步操作：

```bash
# 微调（M1）
make train CONFIG=configs/train/lora_qwen25_1p5b.yaml     # LoRA 微调 Qwen2.5-1.5B

# 推理服务（M2）— OpenAI 兼容
make serve CONFIG=configs/serve/hf_tiny.yaml              # 本机（Mac 可跑）
make serve CONFIG=configs/serve/vllm_qwen25_1p5b.yaml     # vLLM（GPU 环境）
SGLANG_USE_MLX=1 minillm serve --config configs/serve/sglang_mac.yaml   # SGLang-MLX（Apple Silicon，见 docs/serving.md）
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"runs/qwen25-1.5b-lora/export","messages":[{"role":"user","content":"你好"}]}'

# Benchmark（M3）— 三引擎对比
make bench CONFIG=configs/bench/matrix_qwen25.yaml        # 报告: runs/bench/*/bench_report.md

# Web 管理控制台（引擎/GPU/Benchmark/训练 面板）
make serve CONFIG=configs/serve/hf_tiny.yaml              # 先起一个引擎
minillm web --config configs/web/admin.yaml               # 打开 http://127.0.0.1:8080

# 容器化 + 监控（M4/M6）
cd deploy/compose && cp .env.example .env
docker compose --env-file .env up -d prometheus grafana serve-hf mlflow pushgateway

# Kubernetes（M5）
make helm-validate                                       # 无集群静态验证
helm upgrade --install minillm deploy/helm/minillm --namespace minillm --create-namespace
```

## 实测数据（本机 M4 Pro，MPS）

| 场景 | 结果 |
| --- | --- |
| 进程内 Transformers 基线（tiny-Llama，并发 1→2） | 72.7 → 76.1 tokens/s，瓶颈分析正确检出**并发饱和** |
| HTTP 协议路径（vllm 适配器打 HF 服务，并发 1→4） | 68.4 → **122.3 tokens/s（1.8×）**，TTFT p50 0.03s，成功率 100% |
| 测试套件 | 57 项通过（单元 + 集成 + 部署/Helm 工件校验） |

> GPU 环境的完整三引擎对比（vLLM vs SGLang vs Transformers，4 档并发 × 2 档输入/输出长度）由 `configs/bench/matrix_qwen25.yaml` 一键产出。

## 测试与质量

```bash
make test               # 57 项离线测试（含 uvicorn 真实服务协议集成测试）
make smoke              # 离线冒烟（scratch GPT-2 + LoRA 全链路）
make smoke-hub          # 真实模型路径冒烟（自动走 hf-mirror 镜像）
make bench-test         # 真实压测流水线集成测试
make lint               # ruff 全绿
make deploy-validate    # compose/监控工件静态校验
make helm-validate      # helm lint + 全量渲染
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [架构设计与技术选型](docs/architecture.md) | 需求分析 / 分层架构 / 选型总表 / ADR / 风险 / 路线图 |
| [面试技术文档](docs/interview.md) | 🎯 项目讲述稿：核心技术问答 / 难点与解决 / 实测数据 / 追问预案 |
| [微调流水线](docs/finetuning.md) | 数据格式、三模式、产物、MLflow |
| [模型微调使用指南](docs/training-guide.md) | 🎯 从选模型到部署的完整操作手册（含数据示例/超参速查/FAQ/端到端示例） |
| [推理服务](docs/serving.md) | OpenAI API、vLLM 旋钮映射、EngineClient |
| [Benchmark 方法论](docs/benchmark-methodology.md) | 指标口径、公平性、瓶颈规则 |
| [Web 控制台](docs/web.md) | 管理面板使用与 API |
| [部署手册](docs/deployment.md) | Compose / Helm / Kueue / HPA / Pushgateway |

## 路线图（全部完成 ✅）

M0 架构选型 → M1 微调流水线 → M2 推理服务 → M3 Benchmark → M4 Compose+监控 → M5 K8s+Helm+Kueue → M6 指标入库 → M7 演示闭环

## 环境要求

- **开发**：macOS（Apple Silicon 亦可，CPU/MPS 降级模式）；Python 3.12（uv 自动管理）
- **生产/完整验证**：Linux + NVIDIA GPU（≥24GB），CUDA 12.x；Docker 27+；Kubernetes 1.28+（M5）
- HuggingFace 官方不可达时自动/手动走 `https://hf-mirror.com`（`.env.example` 已配置）

## 仓库结构

```
src/minillm/        train/ 微调 · serve/ 推理+引擎抽象 · bench/ 压测 · monitor/ 指标入库 · common/
configs/            train/ · serve/ · bench/ 全部 YAML 配置
deploy/             docker/ Dockerfile · compose/ 双文件编排 · helm/minillm/ · monitoring/ Prometheus/Grafana
docs/               架构/微调/推理/Benchmark方法论/部署 五份文档
scripts/            fetch_model.sh · validate_deploy.py · demo.sh
tests/              57 项自动化测试
```
