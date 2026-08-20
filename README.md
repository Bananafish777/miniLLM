# miniLLM — 端到端大模型基础设施平台

一套覆盖**模型微调 → 推理服务 → 性能基准 → 资源调度 → 监控可视化**全链路的大模型基础设施平台，用于学习实践与求职作品集展示。

## 核心能力

| 模块 | 技术栈 | 说明 |
| --- | --- | --- |
| 🎓 微调流水线 | PyTorch · Transformers · PEFT · TRL · MLflow | LoRA / QLoRA / 全参三条路径，适配 1B~3B 模型，产物自动注册 |
| ⚡ 推理服务 | **vLLM**（主）· SGLang（对比） | OpenAI API 兼容；Paged Attention · Continuous Batching · KV Cache 高并发优化 |
| 📊 Benchmark 系统 | 自研 asyncio 压测客户端 | 对比 Transformers / vLLM / SGLang 的**吞吐、TTFT、ITL、显存占用**，内置瓶颈分析 |
| 🚀 部署调度 | Docker · docker-compose · Kubernetes · Helm · Kueue | 开发/演示与生产双轨编排，GPU 队列调度 + 弹性伸缩 |
| 📈 可观测性 | Prometheus · Grafana · DCGM Exporter | GPU 利用率/显存/温度 + 服务吞吐/延迟分位数大盘 |

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

- [架构设计与技术选型](docs/architecture.md)（当前里程碑 M0，已定稿）
- Benchmark 方法论（M3 补全）
- 部署手册：Compose / Helm（M4/M5 补全）

## 开发路线图

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 架构与技术选型 | ✅ 完成 |
| M1 | 微调流水线（LoRA/QLoRA/全参 + 导出 + MLflow） | ⏳ 下一步 |
| M2 | vLLM OpenAI 兼容推理服务 | 待开始 |
| M3 | Benchmark 系统（三引擎对比 + 瓶颈分析） | 待开始 |
| M4 | docker-compose 全栈编排 | 待开始 |
| M5 | Kubernetes + Helm + Kueue 调度 | 待开始 |
| M6 | 监控完善（告警 + Benchmark 结果入库） | 待开始 |
| M7 | 全链路演示闭环 + 作品集化文档 | 待开始 |

## 环境要求

- **开发**：macOS（Apple Silicon 亦可，CPU/MPS/MLX 降级模式）
- **生产/完整验证**：Linux + NVIDIA GPU（单卡 ≥24GB，如 RTX 4090），CUDA 12.x
- **依赖**：Python 3.11+，Docker 27+，docker-compose 2.x，Kubernetes 1.3x（M5 起）

## 快速开始（占位，随里程碑补全）

```bash
make setup      # 安装依赖与开发环境
make train      # 运行微调流水线（M1）
make serve      # 启动 vLLM 推理服务（M2）
make bench      # 运行三引擎基准测试（M3）
make up         # 一键启动 Compose 全栈（M4）
```
