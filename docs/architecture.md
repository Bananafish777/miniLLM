# miniLLM — 端到端大模型基础设施平台 · 架构设计

| 项目 | 内容 |
| --- | --- |
| 版本 | v1.0（2025-08） |
| 状态 | 已评审定稿，作为后续所有开发里程碑（M1~M7）的基线 |
| 定位 | 学习 + 求职作品集：覆盖**微调 → 推理 → 压测 → 调度 → 监控**全链路 |

---

## 1. 项目背景与目标

### 1.1 目标

设计并实现一套端到端大模型基础设施平台，覆盖模型微调、推理服务部署、性能测试与资源调度流程：

1. **模型微调**：基于 PyTorch + HuggingFace 生态，支持 **LoRA / QLoRA** 与**全参微调**双路径，模型规模 1B~3B（如 Qwen2.5-1.5B / Qwen2.5-3B）。
2. **高性能推理**：基于 **vLLM**（Paged Attention、Continuous Batching、KV Cache）搭建 **OpenAI API 兼容**推理服务，优化高并发场景吞吐与延迟；**SGLang** 作为第二推理引擎，与 vLLM 形成对比。
3. **Benchmark 系统**：对比 **Transformers / vLLM / SGLang** 在不同模型规模与并发压力下的**吞吐（tokens/s）、首 Token 延迟（TTFT）、GPU 显存占用**等指标，定位推理性能瓶颈。
4. **容器化与调度**：Docker 容器化部署训练与推理服务，**docker-compose**（开发/演示）与 **Kubernetes + Helm**（生产）双轨编排。
5. **可观测性**：**Prometheus + Grafana** 监控 GPU 使用率、显存占用与服务性能指标。

### 1.2 环境约束（重要，决定架构形态）

| 约束 | 现状 | 架构对策 |
| --- | --- | --- |
| 个人开发机 | macOS（Apple Silicon，仅 MLX，无 NVIDIA GPU） | **开发/生产分离**：Mac 负责代码、单元测试、数据准备、CPU 演示模式；GPU 全流程在容器化目标环境运行 |
| 目标 GPU 环境 | 单卡 24GB（RTX 4090/3090 级），**可扩展多卡** | 引擎配置支持张量并行参数；资源层预留多卡声明 |
| 模型规模 | 1B~3B | 24GB 显存下可完整跑 LoRA/QLoRA 微调 + 量化推理 |
| 部署形态 | Compose + K8s 都要 | 双轨设计，Compose 保证可演示，Helm 保证生产叙事 |

---

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 说明 |
| --- | --- | --- |
| FR-1 | 数据准备与预处理 | 支持常见指令/对话数据格式（Alpaca/ShareGPT），统一 tokenize 流水线 |
| FR-2 | LoRA/QLoRA 微调 | PEFT 参数高效微调，单卡 24GB 可训 3B 模型 |
| FR-3 | 全参微调 | Trainer 原生支持，多卡可扩展（DDP/FSDP 配置预留） |
| FR-4 | 微调产物导出 | LoRA adapter 合并/保存、量化导出（GPTQ/AWQ）、上传模型注册 |
| FR-5 | OpenAI API 兼容推理 | `/v1/chat/completions`、`/v1/completions`、`/v1/models`，流式与非流式 |
| FR-6 | 多推理引擎 | vLLM（主）/ SGLang（对比）/ Transformers（基线）统一接入 |
| FR-7 | 性能基准测试 | 吞吐、TTFT、ITL、显存占用；可配置模型/引擎/并发/序列长度矩阵 |
| FR-8 | 报告生成 | 结构化 JSON + 可读 Markdown/HTML 报告，含瓶颈分析 |
| FR-9 | 容器化部署 | 训练、推理、监控全容器化，一键编排 |
| FR-10 | 资源调度 | Compose 单机编排；K8s GPU 资源声明、队列调度（Kueue）、弹性伸缩 |
| FR-11 | 指标监控 | GPU 利用率/显存/温度 + 服务级指标（吞吐、排队、延迟分位数） |
| FR-12 | 可视化 | Grafana 面板：训练、推理、Benchmark 三套视图 |

### 2.2 非功能需求

| 编号 | 需求 | 说明 |
| --- | --- | --- |
| NFR-1 | 可移植性 | 代码在 macOS（CPU/MPS 降级模式）与 Linux+CUDA 均可运行 |
| NFR-2 | 配置驱动 | 所有流程由 YAML 配置驱动，不硬编码路径/超参 |
| NFR-3 | 可复现 | 锁定依赖版本、Docker 镜像 tag；实验记录（MLflow） |
| NFR-4 | 模块化 | 训练/推理/Benchmark/监控解耦，可独立运行与替换 |
| NFR-5 | 可观测性内置 | 每个服务默认暴露 `/metrics`（Prometheus 格式） |
| NFR-6 | 文档完整 | 架构、API、部署、Benchmark 方法论均有文档 |

---

## 3. 总体架构

### 3.1 分层架构

```
┌────────────────────────────────────────────────────────────────────┐
│                          接入层  Client / API                        │
│       OpenAI SDK · curl · 自研压测客户端 · Grafana Web                │
├────────────────────────────────────────────────────────────────────┤
│                       编排与调度层  Orchestration                    │
│    docker-compose（开发/演示） · Kubernetes + Helm（生产）             │
│    Kueue GPU 队列 · HPA 弹性伸缩 · 多实例路由                          │
├────────────────────────────────────────────────────────────────────┤
│        ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐      │
│        │  微调子系统   │  │  推理子系统   │  │  Benchmark 子系统   │      │
│        │  Finetune   │  │   Serving   │  │     Bench         │      │
│        │ 数据/训练/   │  │  vLLM 主引擎  │  │ 用例矩阵/并发压测/  │      │
│        │ 评估/导出    │  │  SGLang 备选 │  │ 指标采集/报告      │      │
│        └─────┬───────┘  └─────┬───────┘  └─────────┬─────────┘      │
├──────────────┴────────────────┴────────────────────┴────────────────┤
│                        引擎抽象层  Engine Abstraction                 │
│  统一 EngineClient 接口（generate / stream / metrics / 系统指标）      │
│  适配器：vLLM · SGLang · Transformers · MLX(仅开发)                   │
├────────────────────────────────────────────────────────────────────┤
│                      可观测性层  Observability                        │
│  Prometheus · Grafana · DCGM Exporter · 自研 Exporter · MLflow       │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心设计原则

1. **引擎抽象，多态接入**：推理与 Benchmark 只依赖统一的 `EngineClient` 接口，vLLM / SGLang / Transformers 以适配器模式接入。新增引擎不影响上层代码。
2. **开发/生产分离**：Mac（CPU/MPS）只跑开发与演示路径；GPU 专属能力（vLLM/SGLang/DCGM）通过容器与配置隔离，保证两套环境互不污染。
3. **配置驱动**：训练、推理、Benchmark 的完整参数矩阵全部 YAML 化，一次配置、处处复现。
4. **可观测性内置**：所有服务默认暴露 Prometheus 指标，监控不是后置补丁。
5. **双轨部署，一套代码**：同一镜像、同一配置源，Compose 与 Helm 只是不同的渲染出口。
6. **端到端闭环**：微调产物自动进入模型注册表 → 推理服务可一键加载 → Benchmark 结果回流监控大盘，形成"训练-部署-评测"闭环。

---

## 4. 技术选型

### 4.1 选型总表

| 领域 | 选型 | 版本基线 | 选型理由 | 备选 |
| --- | --- | --- | --- | --- |
| 语言 | Python | 3.11+ | 生态全覆盖，训练/推理/监控全栈同语言 | — |
| 深度学习框架 | PyTorch | 2.4+ | 题目指定；HuggingFace 生态根基 | JAX（不选，生态割裂） |
| 训练 | Transformers + PEFT + TRL + bitsandbytes | 4.4x / 0.1x | Trainer 统一全参与 LoRA 路径；QLoRA 量化底座 | LitGPT（不选，缺 PEFT 生态） |
| 数据 | HuggingFace datasets | 2.x | 流式加载、缓存、切分开箱即用 | 自研 DataLoader |
| 实验追踪 | MLflow（自托管） | 2.x | 指标/参数/模型注册表一体化，作品集中展示实验管理能力 | W&B（商业依赖） |
| 推理引擎（主） | vLLM | ≥0.12 | Paged Attention / Continuous Batching / KV Cache 全部内置；OpenAI API 原生兼容；`/metrics` 开箱即用 | — |
| 推理引擎（对比） | SGLang | ≥0.4 | RadixAttention、原生 OpenAI 兼容 server，与 vLLM 形成 benchmark 对照组 | TensorRT-LLM（闭源工具链重） |
| 推理基线 | Transformers（`generate`） | 4.4x | 无优化基线，量化"优化收益"的参照系 | — |
| Benchmark | 自研（asyncio 并发客户端） | — | 精确控制并发/序列长度/采样参数；三引擎统一口径 | LLMPerf / vLLM benchmark（口径不一，仅参考） |
| 容器 | Docker + docker-compose | 27.x / 2.x | 题目指定；训练/推理/监控全栈编排 | Podman |
| 编排（生产） | Kubernetes + Helm | 1.3x | 生产叙事：GPU 资源声明、滚动更新、弹性伸缩 | Docker Swarm（生态弱） |
| GPU 队列调度 | Kueue | 0.9+ | 社区标准的多租户 GPU 队列/配额调度，作品集亮点 | Volcano |
| 监控 | Prometheus + Grafana | 2.5x / 11.x | 题目指定；K8s 与容器生态事实标准 | VictoriaMetrics |
| GPU 硬件指标 | NVIDIA DCGM Exporter | 3.x | 官方 GPU 利用率/显存/温度指标源 | node-exporter（无 GPU 指标） |
| 配置管理 | pydantic-settings + YAML | 2.x | 类型安全 + 环境变量覆盖，代码内配置模型 | Hydra（学习成本高） |
| 代码质量 | ruff + pytest + pre-commit | — | 轻量统一 lint/测试门禁 | black+flake8（拆分） |
| 开发机适配 | MLX（可选 dev 通道） | 0.2x | Mac 上小模型推理/训练演示 | CPU/MPS 降级（兜底） |

### 4.2 关键版本依据

- **vLLM**：2025 年已迭代至 [v0.12.0+](https://github.com/vllm-project/vllm/releases/tag/v0.12.0)，持续演进 PagedAttention、Prefix Caching、多模态与分布式推理（[官方博客](https://vllm.ai/blog)）。
- **SGLang**：处于 [0.4.x 快速迭代](https://github.com/sgl-project/sglang)，`launch_server` 原生提供 OpenAI 兼容 API。
- **模型**：Qwen2.5 系列（1.5B/3B）HuggingFace 权重齐全，vLLM/SGLang 官方支持列表内，是 1B~3B 档位的最稳选择。

### 4.3 明确的"不选"项

| 不选 | 原因 |
| --- | --- |
| TensorRT-LLM | 工具链重、编译链路长，偏离"快速对比三引擎"的目标 |
| DeepSpeed ZeRO-3 | 1B~3B 单卡场景无收益；全参多卡用 PyTorch DDP/FSDP 即可 |
| FastAPI 自研推理服务 | 无法复刻 Paged Attention/Continuous Batching，违背题目核心诉求 |
| W&B | 商业 SaaS，自托管 MLflow 更符合作品集完整性 |

---

## 5. 子系统设计

### 5.1 微调子系统（Finetune）

**流水线阶段**（每个阶段独立可跑、可断点续跑）：

```
数据准备 ──► 预处理/Tokenize ──► 训练 ──► 评估 ──► 导出 ──► 模型注册
raw JSON    HF Dataset 缓存     LoRA/全参    eval loss/    合并adapter/   MLflow
(Alpaca/    + 训练/验证切分      QLoRA       生成样例       量化(GPTQ/AWQ)   Model Registry
ShareGPT)                                                            │
                                                              v 推理子系统加载
```

**关键设计**：

| 设计点 | 方案 |
| --- | --- |
| 统一入口 | `minillm train --config configs/train/lora_qwen25_1p5b.yaml`；`finetune_mode: lora \| qlora \| full` 三选一，同一 Trainer 包装 |
| LoRA 配置 | `target_modules`（q/k/v/o/gate/up/down）、`r`、`alpha`、`dropout` 全部可配 |
| QLoRA | 4bit NF4 + Double Quant + bnb 优化器；显存目标：3B 模型 < 16GB |
| 全参路径 | 直接走 `Trainer`，预留 DDP/FSDP 启动参数（多卡扩展） |
| 精度 | bf16（GPU 支持时）/ fp16 fallback |
| 序列长度 | 默认 2048，可配 |
| 评估 | 训练中 eval loss + 训练后生成样例对比（模板化 prompt） |
| 导出 | LoRA 合并保存 → 可选 AutoGPTQ/AWQ 量化 → 记录到 MLflow 注册表 |
| 可观测 | 训练指标（loss/lr/显存）→ MLflow + Prometheus push 通道（Pushgateway） |

### 5.2 推理子系统（Serving）

**主服务：vLLM OpenAI API 兼容服务**

```
┌──────────────────────────────────────────────────────┐
│               vLLM OpenAI 兼容 Server                  │
│  /v1/models · /v1/completions · /v1/chat/completions  │
│  （流式 SSE / 非流式）                                  │
├──────────────────────────────────────────────────────┤
│  Paged Attention   KV Cache 分页管理，显存碎片≈0        │
│  Continuous Batching  请求级动态调度，无固定 batch 墙   │
│  KV Cache 管理  gpu_memory_utilization / max-model_len │
│  Prefix Caching（可选） 公共前缀 KV 复用                │
│  /metrics  Prometheus：吞吐/排队/延迟/cache 命中        │
└──────────────────────────────────────────────────────┘
```

**题目核心技术点在 vLLM 中的配置映射**：

| 题目要求 | vLLM 机制 | 平台配置项（示例） |
| --- | --- | --- |
| Paged Attention | Block 级 KV 缓存管理 | `--enforce-eager` 与 CUDA Graph 平衡（默认开 graph 加速） |
| Continuous Batching | 请求级动态调度 | 默认开启；`--max-num-seqs` 控制并发上限 |
| KV Cache | 显存利用率分配 | `--gpu-memory-utilization 0.85`、`--max-model-len 8192`、`--kv-cache-dtype auto` |
| 高并发优化 | 前缀缓存 + 调度器参数 | `--enable-prefix-caching`、`--max-num-batched-tokens` |

**多引擎接入（Engine Abstraction）**：

```
EngineClient（统一接口）
 ├── generate(prompt, params) -> 迭代器[str]        # 流式生成
 ├── chat(messages, params) -> 响应                 # OpenAI 对话格式
 ├── list_models() -> [...]
 ├── metrics() -> 服务端指标（拉取 /metrics）
 └── system_stats() -> 进程内资源占用（pynvml / psutil）

适配器：
 ├── VLLMAdapter     → http://<host>:8000/v1（OpenAI 兼容协议）
 ├── SGLangAdapter   → http://<host>:30000/v1（OpenAI 兼容协议）
 ├── HFAdapter       → 进程内 Transformers pipeline（无优化基线）
 └── MLXAdapter      → Mac 开发演示（可选）
```

同一接口同时服务**推理层**（聊天 UI/API 透传）与 **Benchmark 层**（压测），保证"压测即生产"口径一致。

**服务拓扑（Compose 演示形态）**：

```
                   ┌──────────────┐
  Client/压测 ────► │  nginx 网关   │（可选，路由/限流）
                   └──────┬───────┘
              ┌───────────┴───────────┐
              ▼                       ▼
     ┌──────────────┐        ┌──────────────┐
     │ vLLM 服务实例 │        │ SGLang 对比实例│
     │  :8000       │        │  :30000       │
     └──────────────┘        └──────────────┘
```

### 5.3 Benchmark 子系统（Bench）

**指标体系**（三引擎统一口径）：

| 指标 | 定义 | 采集方式 |
| --- | --- | --- |
| 吞吐（Throughput） | 输出 tokens/s（总输出 tokens / 总耗时） | 压测客户端统计 |
| 并发吞吐（Concurrent TPS） | 固定并发下的稳态吞吐 | 压测客户端统计 |
| TTFT（首 Token 延迟） | 请求发出 → 首个输出 token 到达 | 客户端逐请求计时 |
| ITL（Token 间延迟） | 相邻输出 token 间隔，p50/p90/p99 | 客户端逐 token 计时 |
| TPOT（每输出 token 时间） | 解码阶段单 token 耗时 | 客户端统计 |
| 端到端延迟 | 请求 → 完整响应 | 客户端统计 |
| GPU 利用率 | 压测期间平均/峰值 | DCGM / pynvml 轮询 |
| GPU 显存占用 | 模型驻留 + KV Cache 峰值 | DCGM / pynvml 轮询 |
| 服务端指标 | 排队请求数、cache 命中率 | vLLM `/metrics` 交叉验证 |

**用例矩阵**（配置驱动）：

```
模型 × 引擎 × 并发 × 输入长度 × 输出长度 × 采样参数
[Qwen2.5-1.5B, Qwen2.5-3B] × [transformers, vllm, sglang]
× [1, 8, 16, 32, 64] × [128, 512, 2048] × [128, 512]
```

**执行流程**：

```
bench.yaml ──► 用例矩阵展开 ──► 预热(warmup) ──► 并发压测(asyncio 客户端)
                                                        │
              报告(bench_report.json/md) ◄── 瓶颈分析 ◄──┴── 指标采集
                                                        │
                                              (DCGM/服务端 metrics/客户端计时)
```

**瓶颈分析规则**（内置启发式，可扩展）：
- TTFT 随并发陡增 → prefill 瓶颈 / KV Cache 不足 / 排队过长；
- 吞吐不再随并发增长 → 达到 decode 吞吐上限 / 显存受限；
- 显存峰值接近上限 → 降低 `gpu-memory-utilization` 或序列长度；
- vLLM 与 SGLang 差异显著 → 检查调度策略差异（RadixAttention vs Prefix Caching）。

**对比口径**：三引擎使用**相同模型权重、相同采样参数（如 greedy 或固定 temperature）**，仅引擎不同，保证归因正确。

### 5.4 调度与部署子系统（Deploy）

**5.4.1 docker-compose（开发/演示，单机一键起）**

```
miniLLM compose 栈：
  ├─ train-worker     （一次性任务：微调流水线）
  ├─ vllm-server      （推理，GPU 直通，shm 调大）
  ├─ sglang-server    （对比引擎实例，可选）
  ├─ mlflow           （实验追踪 + 模型注册）
  ├─ prometheus       （指标采集）
  ├─ grafana          （可视化，预置面板）
  └─ dcgm-exporter    （GPU 指标，仅 GPU 主机）
```

**5.4.2 Kubernetes + Helm（生产）**

```
deploy/helm/minillm/
  ├── Chart.yaml
  ├── values.yaml            # 环境差异收敛点
  ├── templates/
  │   ├── inference/         # vLLM Deployment + Service + HPA
  │   ├── bench/             # benchmark Job（一次性压测任务）
  │   ├── monitoring/        # Prometheus/Grafana/DCGM（或引用 kube-prometheus-stack）
  │   ├── training/          # 微调 Job（Kueue 队列）
  │   └── queue.yaml         # Kueue ClusterQueue/LocalQueue
```

**GPU 调度策略**：

| 策略 | 实现 | 说明 |
| --- | --- | --- |
| GPU 资源声明 | `resources.limits: nvidia.com/gpu: 1` + nodeSelector | 硬性绑定 GPU 节点 |
| 队列调度 | Kueue ClusterQueue（配额/抢占/优先级） | 训练 Job 与推理共享 GPU 池的排队秩序 |
| 弹性伸缩 | HPA 基于 vLLM 指标（排队请求数/吞吐） | 高并发自动扩容副本 |
| 张量并行扩展 | vLLM `--tensor-parallel-size N`，多卡声明 | 为多卡环境预留（Compose 与 Helm 均暴露该参数） |
| 亲和性 | nodeAffinity + tolerations | GPU 节点打标（`gpu=1`）隔离 |

### 5.5 可观测性子系统（Observability）

**指标来源拓扑**：

```
┌─────────────┐   ┌────────────────┐   ┌─────────────────┐   ┌───────────────┐
│ DCGM        │   │ vLLM /metrics   │   │ 自研 Exporter    │   │ Pushgateway   │
│ GPU 利用率/  │   │ 吞吐/排队/      │   │ (训练进程指标、   │   │ (训练脚本推送  │
│ 显存/温度    │   │ cache 命中      │   │ benchmark 结果)  │   │  loss/lr)     │
└──────┬──────┘   └───────┬────────┘   └────────┬────────┘   └───────┬───────┘
       └──────────────────┴────────────────────┴────────────────────┘
                                    │ scrape
                              ┌─────▼─────┐
                              │ Prometheus │──► Grafana（三套面板）
                              └───────────┘    训练视图 / 推理视图 / Benchmark 视图
```

**指标清单（核心）**：

| 来源 | 指标示例 |
| --- | --- |
| DCGM | `DCGM_FI_DEV_GPU_UTIL`、`DCGM_FI_DEV_FB_USED`、`DCGM_FI_DEV_MEM_CLOCK`、`DCGM_FI_DEV_GPU_TEMP` |
| vLLM | `vllm:generation_tokens_total`、`vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:cache_hit_rate`、`vllm:time_to_first_token_seconds`、`vllm:e2e_request_latency_seconds` |
| 自研训练 | `minillm_train_loss`、`minillm_train_lr`、`minillm_train_tokens_per_sec`、`minillm_train_gpu_mem` |
| Benchmark | `minillm_bench_throughput`、`minillm_bench_ttft_p99`（按 模型×引擎×并发 打标签） |

**Grafana 面板设计**：训练视图（loss 曲线/学习率/显存水位）、推理视图（吞吐/排队数/TTFT 分位/GPU 利用率）、Benchmark 视图（三引擎对比柱状图/热力图）。

---

## 6. 端到端数据流

```
 原始数据 ──► 预处理 ──► 微调(LoRA/全参) ──► 评估/导出 ──► MLflow 模型注册
                                                              │
                        ┌─────────────────────────────────────┘
                        ▼
                 vLLM/SGLang 推理服务（OpenAI API）
                        ▲
        ┌───────────────┴───────────────┐
        │                               │
   Benchmark 并发压测            聊天客户端 / 业务调用
   （三引擎对比 + 瓶颈分析）
        │
        ▼
   报告 + Prometheus/Grafana 监控大盘
```

---

## 7. 仓库结构规划

```
miniLLM/
├── README.md                    # 项目总览、快速开始、文档索引
├── docs/                        # 全部文档（架构/API/部署/方法论）
│   ├── architecture.md          # 本文档
│   ├── benchmark-methodology.md # Benchmark 口径与指标定义（M3 补全）
│   └── deployment.md            # Compose/Helm 部署手册（M4/M5 补全）
├── configs/                     # 全部 YAML 配置（训练/推理/benchmark/告警）
│   ├── train/
│   ├── serve/
│   └── bench/
├── src/minillm/
│   ├── __init__.py
│   ├── train/                   # 微调流水线（prepare/train/evaluate/export）
│   ├── serve/                   # 推理启动器、EngineClient 抽象与适配器
│   │   ├── engine/              #   vllm.py / sglang.py / hf.py / mlx.py
│   │   └── server.py            #   OpenAI 兼容入口（vLLM 直出）
│   ├── bench/                   # Benchmark（matrix/cases/runner/collector/report）
│   ├── monitor/                 # 自研指标 exporter、告警规则
│   └── common/                  # 配置模型、日志、工具函数
├── deploy/
│   ├── docker/                  # Dockerfile（train/vllm/sglang/bench 多阶段）
│   ├── compose/                 # docker-compose 栈 + env 模板
│   ├── helm/minillm/            # Helm chart（inference/bench/training/queue）
│   └── monitoring/              # prometheus.yml、grafana-provisioning、dcgm
├── data/                        # 数据集与缓存（gitignore）
├── scripts/                     # 开发辅助（setup/launch/clean）
├── tests/                       # 单元与集成测试
├── pyproject.toml               # 依赖与工具链
├── .env.example
└── Makefile                     # 常用命令入口（make train / make serve / make bench）
```

---

## 8. 关键设计决策（ADR 摘要）

| # | 决策 | 理由 | 影响 |
| --- | --- | --- | --- |
| ADR-1 | 推理与 Benchmark 共用 `EngineClient` 抽象 | 压测口径=生产口径；三引擎无差别接入 | 上层零引擎感知 |
| ADR-2 | OpenAI 兼容 API 作为对外唯一协议 | 生态标准，客户端（OpenAI SDK/curl）即插即用 | 屏蔽引擎差异 |
| ADR-3 | Compose + Helm 双轨，配置单一来源 | 演示与生产共享同一配置模型 | 部署文档两套渲染出口 |
| ADR-4 | Mac 开发 / GPU 生产分离 | 个人无 NVIDIA GPU；vLLM/SGLang 仅 CUDA | 提供 CPU 降级演示模式 |
| ADR-5 | Transformers 作为 benchmark 基线而非服务 | 无优化基线必须存在，才能量化 vLLM/SGLang 收益 | HFAdapter 进程内运行 |
| ADR-6 | 显存指标以 DCGM + pynvml 双通道 | DCGM 面向大盘，pynvml 面向压测精确采样 | 交叉验证 |
| ADR-7 | 配置用 pydantic-settings（非 Hydra） | 类型安全、环境变量覆盖、学习成本低 | 复杂组合实验靠 YAML 矩阵 |
| ADR-8 | 微调默认 LoRA，全参为扩展路径 | 24GB 单卡 + 1B~3B 模型的最优匹配 | Trainer 统一入口双模式 |

---

## 9. 开发路线图

| 里程碑 | 内容 | 验收标准 |
| --- | --- | --- |
| **M0** | 架构与技术选型（本文档） | 架构评审通过 ✅ |
| **M1** | 微调流水线 | ✅ 完成：LoRA/QLoRA/全参三路径、4 种数据格式、prompt 掩码、评估、adapter 合并导出、MLflow 可选追踪；21 个测试通过（含 scratch 与真实 tiny-Llama hub 路径），CLI `minillm train` 可用 |
| **M2** | 推理服务 | ✅ 完成：EngineClient 抽象（vLLM/SGLang/Transformers 适配器）、OpenAI 兼容服务器（流式 SSE/非流式、/v1/models、/metrics Prometheus 指标）、vLLM/SGLang 启动命令与 docker 提示、38 个测试通过（含真实 uvicorn 服务 + 双适配器协议集成验证） |
| **M3** | Benchmark 系统 | ✅ 完成：矩阵展开 → asyncio 并发压测（流式 TTFT/ITL 精确计时）→ 双通道 token 交叉验证 → 聚合 + 内置瓶颈分析 → JSON/Markdown 报告；54 个测试通过；本机实测：进程内 HF 73→76 tps（检出并发饱和）、HTTP 协议路径 c=4 达 122 tps（1.8×） |
| **M4** | Compose 全栈 | ✅ 完成：双 compose 文件（base + GPU override）、Profile 门控、训练/推理/MLflow/Prometheus/Grafana/DCGM 八服务、Grafana 面板与 6 条告警规则预置、`make deploy-validate` 静态校验工件（无 Docker 环境可测） |
| **M5** | K8s + Helm | ✅ 完成：Helm chart（vLLM Deployment/Service/HPA external 指标、SGLang 可选、训练/压测 Job、Kueue ClusterQueue+LocalQueue、ServiceMonitor 可选、PVC 模板）、`helm lint` + 全量渲染验证（12 资源）、配置漂移测试 |
| **M6** | 监控完善 | ✅ 完成：Benchmark/训练结果经 Pushgateway 入库（6+2 指标、实验分组键防覆盖）、Grafana Benchmark 对比面板（实验下拉过滤）、compose 内置 pushgateway 服务与采集、推送格式测试（本地捕获服务器） |
| **M7** | 演示闭环 + 文档 | 全链路演示脚本 + README 作品集化（架构图/结果截图/方法论） |

> M1~M3 可在无 GPU 时先以 **CPU 演示模式**开发联调（Transformers 路径完整可测），GPU 专属验证留待目标环境。

---

## 10. 风险与对策

| 风险 | 等级 | 对策 |
| --- | --- | --- |
| 个人无 NVIDIA GPU，vLLM/SGLang 无法本地验证 | 高 | CPU 降级模式开发联调；预留 GPU CI/远端服务器接入点；文档明确最低硬件要求 |
| CUDA/PyTorch/vLLM 版本兼容矩阵复杂 | 中 | 锁定镜像 tag（如 `vllm/vllm-openai:v0.12.x-cu124`），Docker 构建脚本统一管理 |
| 24GB 显存下 3B 模型 QLoRA + KV Cache 同卡竞争 | 中 | 显存预算表（模型权重/优化器/adapter/KV Cache 分区），`gpu-memory-utilization` 可调 |
| Benchmark 口径不一致导致对比失真 | 中 | 统一权重/采样参数/预热策略；客户端计时 + 服务端指标双通道交叉验证 |
| K8s 环境成本高（无集群） | 中 | Helm chart 先静态验证（`helm template`/kind 单节点），Compose 保证演示闭环 |
| 作品集差异化不足 | 低 | 强调：三引擎对比方法论、Kueue 调度、Benchmark 结果入库大盘这三点是同类项目少见完整闭环 |

---

## 11. 参考链接

- [vLLM 官方博客](https://vllm.ai/blog) · [vLLM Releases](https://github.com/vllm-project/vllm/releases)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [vLLM OpenAI 兼容 Server 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Kueue（K8s GPU 队列调度）](https://kueue.sigs.k8s.io/)
- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [Qwen2.5 系列模型](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
