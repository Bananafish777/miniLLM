# miniLLM — 端到端大模型基础设施平台

一套覆盖**模型微调 → 推理服务 → 性能基准 → 资源调度 → 监控可视化**全链路的大模型基础设施平台，用于学习实践与求职作品集展示。

## 核心能力

| 模块 | 技术栈 | 说明 |
| --- | --- | --- |
| 🎓 微调流水线 | PyTorch · Transformers · PEFT · MLflow | LoRA / QLoRA / 全参三条路径，适配 1B~3B 模型，产物自动导出注册 |
| ⚡ 推理服务 | **vLLM**（主）· SGLang（对比） | OpenAI API 兼容；Paged Attention · Continuous Batching · KV Cache 高并发优化（M2） |
| 📊 Benchmark 系统 | 自研 asyncio 压测客户端 | 对比 Transformers / vLLM / SGLang 的**吞吐、TTFT、ITL、显存占用**，内置瓶颈分析（M3） |
| 🚀 部署调度 | Docker · docker-compose · Kubernetes · Helm · Kueue | 开发/演示与生产双轨编排，GPU 队列调度 + 弹性伸缩（M4/M5） |
| 📈 可观测性 | Prometheus · Grafana · DCGM Exporter | GPU 利用率/显存/温度 + 服务吞吐/延迟分位数大盘（M4/M6） |

## 架构总览

```
数据 ──► 微调(LoRA/QLoRA/全参) ──► 导出/模型注册 ──► vLLM/SGLang 推理(OpenAI API)
                                                          │
                     Benchmark 并发压测 ◄──────────────────┘
                        │
                   报告 + Prometheus/Grafana 监控大盘
```

- 多引擎统一 **EngineClient 抽象**：推理与压测共用同一接口，保证"压测即生产"口径
- **开发/生产分离**：macOS 可跑 CPU 演示模式，NVIDIA GPU 环境跑完整流程
- 全流程 **YAML 配置驱动**，一次配置、处处复现

## 文档

- [架构设计与技术选型](docs/architecture.md)（M0，已定稿）
- [微调流水线使用说明](docs/finetuning.md)（M1）
- Benchmark 方法论（M3 补全）· 部署手册（M4/M5 补全）

## 开发路线图

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 架构与技术选型 | ✅ 完成 |
| M1 | 微调流水线（LoRA/QLoRA/全参 + 导出 + MLflow） | ✅ 完成 |
| M2 | vLLM OpenAI 兼容推理服务 | ⏳ 下一步 |
| M3 | Benchmark 系统（三引擎对比 + 瓶颈分析） | 待开始 |
| M4 | docker-compose 全栈编排 | 待开始 |
| M5 | Kubernetes + Helm + Kueue 调度 | 待开始 |
| M6 | 监控完善（告警 + Benchmark 结果入库） | 待开始 |
| M7 | 全链路演示闭环 + 作品集化文档 | 待开始 |

## 环境要求

- **开发**：macOS（Apple Silicon 亦可，CPU/MPS 降级模式）
- **生产/完整验证**：Linux + NVIDIA GPU（单卡 ≥24GB，如 RTX 4090），CUDA 12.x
- **依赖**：Python 3.12（uv 自动管理），Docker 27+，docker-compose 2.x，Kubernetes 1.3x（M5 起）

## 快速开始

```bash
make setup              # 创建 venv (Python 3.12) 并安装依赖
make test               # 离线单元测试（无需网络/GPU）
make smoke              # CPU 冒烟：scratch 微型 GPT-2 + LoRA 全链路
make smoke-hub          # hub 路径冒烟（自动走 hf-mirror 镜像）

# 微调（M1 已可用）
make train CONFIG=configs/train/smoke_scratch.yaml              # 离线冒烟
make train CONFIG=configs/train/lora_qwen25_1p5b.yaml           # LoRA 微调 Qwen2.5-1.5B
make train CONFIG=configs/train/qlora_qwen25_3b.yaml            # QLoRA 微调 Qwen2.5-3B（GPU）
make train CONFIG=configs/train/full_qwen25_1p5b.yaml           # 全参微调（GPU）

# 微调产物（runs/<run>/）
#   adapter/      LoRA adapter
#   export/       合并后的完整模型（M2 可直接被 vLLM 加载）
#   metrics.json  训练摘要（供监控采集）
#   eval_samples.json  生成质量样例
```

> HuggingFace 官方 hub 被墙时设置 `HF_ENDPOINT=https://hf-mirror.com`（已写入 `.env.example`）。
> 详细配置字段说明见 [docs/finetuning.md](docs/finetuning.md)。
