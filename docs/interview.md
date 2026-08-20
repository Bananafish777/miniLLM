# miniLLM 面试技术文档

> 面向算法/后端/ML Infra 岗位的面试讲述稿：项目定位 → 架构 → 核心技术问答 → 难点与解决 → 实测数据 → 追问预案。
> 建议讲述时长：10~15 分钟（亮点 3 分钟 → 架构 3 分钟 → 核心技术 5 分钟 → 难点 3 分钟 → 数据 1 分钟）。

---

## 1. 项目一句话定位

**"一个端到端的大模型基础设施平台：PyTorch 微调 → vLLM/SGLang OpenAI 兼容推理 → 三引擎性能基准 → Docker/K8s 部署 → Prometheus/Grafana 可观测，全链路自研、全链路可验证。"**

加分句（根据岗位选一句）：
- 算法向：*"用自研 Benchmark 量化了 Paged Attention / Continuous Batching 等优化机制的实际收益"*
- 工程向：*"用 EngineClient 统一抽象让'压测口径 = 生产口径'，换引擎只改配置不改代码"*
- 系统向：*"从零搭建了 GPU 配额队列（Kueue）+ 基于排队指标的 HPA 弹性伸缩的调度体系"*

---

## 2. 架构总览

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
                     │                    │ /metrics     │ Pushgateway
                     ▼                    ▼              ▼
              ┌──────────────────────────────────────────────────────┐
              │       Prometheus ◄── DCGM/vLLM/minillm 指标           │
              │            │                                         │
              │       Grafana（全栈监控 + Benchmark 对比面板）          │
              └──────────────────────────────────────────────────────┘
  部署: docker-compose（开发/演示） · Kubernetes + Helm + Kueue（生产）
  Web: minillm web 管理控制台（引擎状态/GPU/Benchmark/训练 一站式面板）
```

**五个子系统一句话职责**：
| 子系统 | 职责 | 关键设计 |
| --- | --- | --- |
| 微调 | LoRA/QLoRA/全参三路径，产物自动导出注册 | Trainer 统一入口；prompt 掩码；adapter 合并 → 可直接喂 vLLM |
| 推理 | OpenAI API 兼容服务，vLLM 为主引擎 | EngineClient 抽象；Paged Attention/Continuous Batching/KV Cache 全旋钮化 |
| Benchmark | 三引擎 × 并发 × 长度矩阵压测 | asyncio 并发；流式精确计时 TTFT/ITL；双通道交叉验证 |
| 调度部署 | Compose（演示）+ Helm/Kueue（生产） | 双 compose 文件；GPU 配额队列；HPA 外部指标 |
| 可观测 | Prometheus/Grafana + 结果入库 | DCGM 硬件指标；Pushgateway 承接批任务指标 |

---

## 3. 核心技术问答（面试高频）

### Q1. 讲一下 Paged Attention 的原理？为什么能省显存？
- **问题本质**：传统 KV Cache 按请求预分配连续显存，最长序列假设导致**内部碎片**（预留不用）+ **外部碎片**（无法复用）。
- **Paged Attention**：把 KV Cache 切成固定大小 block（如 16 token/block），用**块表（block table）**像操作系统的虚拟内存分页一样映射逻辑块 → 物理块。
- **收益**：① 按需分配，碎片率从 60~80% 降到 <4%；② 相邻请求的物理块可共享（prefix caching）；③ 支持抢占时把 block 换出到 CPU。
- 回答加分：*"本质是把显存管理从'连续分配'变成'分页分配'，这是 vLLM 能在 24GB 卡上跑更大 batch 的根本原因。"*

### Q2. Continuous Batching 是什么？和传统 Static Batching 的区别？
- 传统：按 batch 为单位调度，一个 batch 内所有请求必须全部完成后才能释放；**慢请求拖累快请求**，batch 之间有空档。
- Continuous Batching（vLLM 的调度器）：**迭代级调度（iteration-level scheduling）**——每个 decode 步（iteration）都重新决定哪些请求进入执行；某个请求生成完（EOS/达 max_tokens）立刻释放其 KV Cache 插槽，新请求**立刻**插入 prefill。
- 收益：GPU 利用率显著提升；混合 prefill/decode 阶段（chunked prefill 进一步切分长 prompt）。
- 回答加分：*"它把调度粒度从'请求'降到'迭代'，代价是调度器复杂度上升（需要维护 running/waiting 队列、抢占策略）。"*

### Q3. KV Cache 是什么？显存怎么估算？
- 推理时每个 token 的 K/V 向量需缓存供后续 token 注意力使用，避免重复计算（计算量 O(n²) → 缓存后 decode 每步 O(n)）。
- 估算公式：`显存 ≈ 2 × num_layers × num_heads × head_dim × seq_len × batch × bytes_per_elem`
  - 例：Qwen2.5-1.5B（28 层 × 28 头 × 128 head_dim），fp16，8K 序列 × 32 并发 ≈ **~2.5GB/请求 × ...**（按公式现场估算，展示你懂推导）。
- 相关旋钮：`gpu_memory_utilization`（KV 预算）、`max_model_len`（窗口上限）、prefix caching（复用）。

### Q4. vLLM vs SGLang vs Transformers 的差异？
| 维度 | Transformers | vLLM | SGLang |
| --- | --- | --- | --- |
| 调度 | 无（单请求） | 迭代级 Continuous Batching | 同左 + RadixAttention |
| KV 管理 | 连续分配 | Paged Attention 分页 | RadixAttention（前缀树复用，细到 token 级） |
| 前缀复用 | 无 | Prefix Caching（块级） | RadixAttention 树结构（更细粒度） |
| 吞吐/延迟 | 基线（最低） | 高 | 高（长共享前缀场景更优） |
| 成熟度 | 最稳 | 生态最全 | 迭代快，部分特性先行 |
- 回答加分：*"Benchmark 里我把 Transformers 作为'无优化基线'，vLLM/SGLang 的收益都是相对它量化的——这也是 benchmark 设计里'必须有参照系'的思路。"*

### Q5. LoRA / QLoRA / 全参怎么选？
- **LoRA**：冻结原权重，注入低秩矩阵（r=16 等），可训练参数 <1%；显存省、速度快、adapter 可插拔；`W' = W + BA`，推理时可合并回原权重（本项目 merge_and_unload 导出）。
- **QLoRA**：4-bit NF4 量化底座 + Double Quant + paged optimizer，3B 模型可在 <16GB 显存微调；量化损失 <1%。
- **全参**：效果上限最高，但 1.5B 模型也需要大显存/多卡；本项目三条路径由 `finetune_mode` 配置切换，同一 Trainer 包装。
- 选型逻辑：*"资源约束 → 效果需求 → 框架支持"*。

### Q6. Benchmark 如何保证三引擎对比公平？（重点，必问）
口径五原则：
1. **同权重同采样**：同一模型目录、temperature=0（greedy），消除采样随机性；
2. **预热**：每用例先跑 warmup（CUDA graph、显存分配、prefix cache 就绪后再计时）；
3. **流式计时**：TTFT 只能从流式接口精确测得，三引擎统一 stream；
4. **双通道 token 验证**：客户端 chunk 计数 vs 服务端 `/metrics` 计数增量——两者偏差本身就是流式开销的度量；
5. **串行用例**：引擎间不并行，避免资源争抢污染对比。
- 回答加分：*"我发现 TextIteratorStreamer 会把相邻 token 合并成 chunk，导致客户端计数偏小——所以报告里同时给出服务端计数算的吞吐，这是交叉验证的价值。"*

### Q7. 瓶颈分析是怎么做的？
内置 5 条启发式规则（可扩展）：
- 吞吐饱和：并发翻倍增益 <5% → decode/显存受限；
- TTFT 劣化：c>1 的 TTFT-p99 > c=1 的 5× → 排队/prefill 抢占；
- 引擎对比：>1.5× → 优化收益显著；<10% → 疑似共享瓶颈；
- 失败率 >0 → critical；显存水位 → 与 KV Cache 预算对照。
- 实测例子：*"本机 MPS 上进程内 HF 并发 1→2 吞吐 73→76 t/s（检出饱和，符合单流基线预期）；HTTP 路径并发 1→4 达 122 t/s（1.8×），TTFT 仅 0.03s。"*

### Q8. 为什么批任务指标用 Pushgateway 而不是被 Prometheus 直接抓？
- Prometheus 是 pull 模型：需要**常驻且可寻址**的 /metrics 端点。
- 训练/压测是**一次性批任务**（容器跑完就退出），Prometheus 抓不到 → Pushgateway 作为指标"邮局"：任务结束时 push 结果，Prometheus 照常 pull Pushgateway。
- 注意事项（体现深度）：Pushgateway 是"最后一值"语义，所以本项目用 grouping_key（experiment+timestamp）区分每次运行，避免旧实验覆盖新实验；同时要配**指标过期清理**策略。

### Q9. Kueue 队列调度解决了什么问题？
- 多租户/多任务共享 GPU 池时，需要**配额管理 + 排队秩序 + 优先级抢占**——原生 K8s 只有"资源够就调度，不够就 Pending"。
- Kueue 的 ClusterQueue（集群级配额：cpu/memory/nvidia.com/gpu nominalQuota）+ LocalQueue（命名空间入口），Workload 被 admitted 后才真正创建 Pod。
- 本项目：训练/压测 Job 通过 `kueue.x-k8s.io/queue-name` 注解接入；推理服务是常驻负载不走队列，弹性由 HPA 负责——**常驻 vs 批任务用不同的调度策略**，这是设计取舍。

### Q10. HPA 为什么用 external 指标？怎么落地？
- K8s 内置 HPA 支持 CPU/内存（resource 指标）和自定义指标，但 vLLM 的排队请求数在 **Prometheus** 里（`vllm:num_requests_waiting`）。
- 方案：prometheus-adapter 把 Prometheus 查询暴露为 `external.metrics.k8s.io`，HPA 用 `type: External + AverageValue` 引用；排队 >20 → 扩容 1→4 副本。
- 比 CPU 指标更贴业务：排队数是**服务端真实拥塞信号**，而容器 CPU 在 vLLM 的 CUDA 图执行下往往不是瓶颈。

### Q11. 可观测性指标体系怎么设计的？
- 硬件层：DCGM（利用率/显存/温度/时钟）；
- 服务层（RED）：Rate（请求速率）、Errors（abort 率告警）、Duration（TTFT/E2E 直方图分位）；
- 业务层：Benchmark 结果（吞吐/TTFT/ITL 按 engine/concurrency 打标签入库）、训练指标（loss/tokens-per-sec）；
- 告警规则 6 条：服务 down、显存 >90%、GPU 利用率异常、排队 >50、TTFT-p99 >5s、abort >10%。

### Q12. EngineClient 抽象的价值？
- 推理层与压测层共用同一接口 → **压测即生产**，benchmark 不会"测了个寂寞"；
- 三引擎（含本地 Transformers）协议级可替换 → 无 GPU 时也能完整验证协议路径；
- 新引擎接入 = 实现 5 个方法，上层零改动。

---

## 4. 项目难点与解决（真实踩坑，展示工程能力）

| # | 难点 | 现象 | 解决 |
| --- | --- | --- | --- |
| 1 | transformers 5.x 破坏性变更 | `TrainingArguments` 移除 `warmup_ratio`；`Conv1D` 迁移到 `pytorch_utils` | 运行时 inspect 签名做版本兼容（warmup_steps 传小数）；导入路径回退链 |
| 2 | httpx 流式 client 生命周期 | 流式生成器在 `with httpx.Client` 退出后才被消费 → "client has been closed" | client 生命周期跟随生成器，finally 中关闭 |
| 3 | MPS 内核非确定性 | 同参数两次 greedy 生成结果不同，测试断言失败 | 测试改为断言 token 计数与结构性质，不比较具体文本（注释说明环境特性） |
| 4 | 沙箱/受限网络下 HF 不可达 + 镜像对 Python 客户端 TLS 指纹拦截 | huggingface_hub 全部 308 到被墙源站，curl 正常 | 用 curl 拉取模型本地化（fetch_model.sh），hub 路径测试改走本地真实权重 |
| 5 | 无 GPU 如何验证 vLLM/SGLang | 引擎无法本地运行 | 适配器走 OpenAI HTTP 协议 → 用自建 HF 服务做协议级集成测试；引擎专属能力（CUDA）标注为 GPU 环境验证项 |
| 6 | 指标口径漂移 | 客户端 chunk 计数 < 服务端真实 token 数（streamer 合并 chunk） | 双通道交叉验证，报告中并列展示 |

**讲述建议**：挑 2~3 个讲细节（推荐 2/3/6），其余一句话带过。重点展示"现象 → 定位 → 修复 → 防止回归（测试）"的完整链路。

---

## 5. 实测数据（诚实标注环境）

| 场景 | 环境 | 结果 |
| --- | --- | --- |
| 进程内 Transformers 基线并发 1→2 | M4 Pro (MPS) | 72.7 → 76.1 tokens/s，瓶颈分析正确检出并发饱和 |
| HTTP 协议路径并发 1→4（vllm 适配器打 HF 服务） | M4 Pro (MPS) | 68.4 → 122.3 tokens/s（1.8×），TTFT p50 0.03s，成功率 100% |
| 测试套件 | — | 59+ 项自动化测试全绿（单元/集成/部署/Helm 工件） |

**尚未验证（如实说明）**：真实 NVIDIA GPU 上的三引擎对比矩阵（vLLM vs SGLang vs Transformers）已配置就绪（`configs/bench/matrix_qwen25.yaml`），但本开发环境无 GPU——此部分留待 GPU 环境执行。诚实陈述 + 已准备好的执行方案，比假装跑过更可信。

---

## 6. 面试追问预案

| 追问 | 回答要点 |
| --- | --- |
| "为什么不用现成的 LLMPerf/vLLM benchmark？" | 现成工具口径不统一、无法精确控制矩阵与双通道交叉验证；自研 300 行内可控，且是作品集差异化点 |
| "吞吐上不去可能有哪些原因？怎么排查？" | 数据管线（tokenize 慢）→ 单步延迟（ITL 高=decode 慢）→ 并发不足（利用率低）→ 排队（TTFT 高）→ 显存（KV 受限）；用 RED 指标逐层定位 |
| "多卡怎么扩展？" | 推理：tensor_parallel_size；训练：DDP/FSDP 参数预留；Kueue 配额按卡数声明 |
| "模型微调效果如何评估？" | eval loss + 生成样例（本项目 eval_samples.json）+ 领域指标（若业务可定义）|
| "如果让你继续做，下一步做什么？" | ① GPU 环境跑完整三引擎矩阵并沉淀报告；② 增加量化推理（GPTQ/AWQ）与 LoRA 动态加载；③ 多租户鉴权与限流；④ 训练/推理共享 GPU 池的 Kueue 抢占策略调优 |
| "这个项目你个人贡献是什么？" | 从架构设计到全部实现（M0~M7 八阶段），含文档与测试；指出可 Review 的代码文件（见项目地图） |

---

## 7. 项目地图（面试前过一遍）

| 面试讲述点 | 代码位置 |
| --- | --- |
| 引擎抽象 | `src/minillm/serve/engine/`（base/openai/hf/vllm/sglang） |
| OpenAI 兼容服务 | `src/minillm/serve/server.py` |
| 微调流水线 | `src/minillm/train/trainer.py` + `data/` |
| Benchmark 口径 | `src/minillm/bench/`（client/runner/metrics/report） |
| 指标入库 | `src/minillm/monitor/` |
| Web 控制台 | `src/minillm/web/` |
| Compose 编排 | `deploy/compose/docker-compose*.yml` |
| K8s 调度 | `deploy/helm/minillm/templates/`（queue.yaml / inference-vllm.yaml） |
| 监控面板 | `deploy/monitoring/grafana/dashboards/` |
| 全部配置 | `configs/`（train/serve/bench/web） |

---

## 8. 一句话总结（收尾用）

**"这个项目我独立完成了从架构设计到落地验证的完整闭环：用统一的引擎抽象打通了训练、推理与评测，用自研基准量化了优化收益，用双轨部署和可观测性体系覆盖了生产所需——它证明了我能把一个复杂系统从 0 到 1 做出来，并且每一步都有测试和证据。"**
