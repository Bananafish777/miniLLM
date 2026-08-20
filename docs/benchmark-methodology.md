# Benchmark 方法论（M3）

## 目标

在**相同模型权重、相同采样参数**下对比 Transformers / vLLM / SGLang 的吞吐与延迟特征，
量化优化引擎（Paged Attention / Continuous Batching / KV Cache）的实际收益，定位瓶颈。

## 指标体系

| 指标 | 定义 | 采集通道 |
| --- | --- | --- |
| 吞吐（tokens/s） | 用例总输出 tokens / 总耗时 | 客户端计数 + **服务端计数增量交叉验证** |
| TTFT | 请求发出 → 首个输出 token | 流式首 chunk 计时（逐请求） |
| ITL | 相邻输出 token 间隔 | 流式 chunk 时间戳序列 |
| 端到端延迟 | 请求 → 完整响应 | 客户端计时 |
| 成功率 | 成功请求 / 总请求 | 客户端 |
| GPU 显存峰值 | 压测期间采样最大值 | pynvml / 引擎指标（无 GPU 为 None） |

## 口径与公平性

1. **同权重同参数**：三引擎加载同一模型目录、`temperature=0.0`（greedy），消除采样随机性；
2. **预热**：每个用例先跑 `warmup_requests` 次（CUDA graph、prefix cache、显存分配就绪）；
3. **并发模型**：`asyncio.Semaphore(concurrency)` 恒定并发，每并发点跑
   `concurrency × requests_per_concurrency` 个请求，全程计时；
4. **流式测量**：TTFT 只能从流式接口精确测得，所有引擎统一走 `stream=true`；
5. **双通道 token 校验**：客户端 chunk 计数受流式实现影响（如 TextIteratorStreamer 会合并 chunk），
   报告同时给出引擎 `/metrics` 的 token 计数增量计算的吞吐，两者偏差即流式开销/合并的度量；
6. **用例顺序执行**：引擎之间不并行，避免资源争抢污染对比。

## 矩阵设计

```
引擎 × 模型 × 并发 × 输入长度 × 输出长度
hf/vllm/sglang × qwen25-1.5b × [1,8,32,64] × [128,2048] × [128,512]
```

- 输入长度变化 → prefill 阶段压力（影响 TTFT）
- 输出长度变化 → decode 阶段压力（影响吞吐）
- 并发阶梯 → 排队与批处理效率

## 瓶颈分析启发式（内置，可扩展）

| 规则 | 触发条件 | 含义 |
| --- | --- | --- |
| 吞吐饱和 | 并发翻倍但吞吐增益 <5% | decode 或显存受限 |
| TTFT 劣化 | c>1 的 TTFT-p99 > c=1 的 5× | 排队过长 / prefill 抢占 |
| 引擎对比 | 最佳/最差吞吐 >1.5× | 优化引擎收益显著 |
| 引擎持平 | 差异 <10% | 共享瓶颈（单卡显存/数据管线） |
| 失败率 | 任何失败请求 | 服务健康问题（critical） |
| 显存水位 | 采样峰值 | 与 KV Cache 预算对照 |

## 报告产物（runs/bench/<experiment>-<ts>/）

- `bench_report.json` — 结构化数据（矩阵配置 + 指标行 + findings），供入库/二次分析
- `bench_report.md` — 可读报告（指标总表 + 瓶颈分析 + 口径说明）

## 三引擎差异说明

| 引擎 | 接入方式 | 测量注意点 |
| --- | --- | --- |
| Transformers | 进程内（`AsyncHFClient`，异步锁串行） | 单流基线；chunk 计数偏小（streamer 合并） |
| vLLM | HTTP OpenAI 协议（`AsyncOpenAIClient`） | `/metrics` 提供 `vllm:*` 交叉验证指标 |
| SGLang | HTTP OpenAI 协议 | 同上 |
