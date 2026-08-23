# 项目知识与实测经验库

> 面向面试准备：系统整理 miniLLM 项目涉及的技术知识 + 本人在开发中真实踩过的坑与调优经验。
> 配套：`docs/interview.md`（问答版）、`docs/interview-self-intro.md`（自我介绍）、`docs/benchmark-results.md`（实测数据）。

---

## Part 1：技术知识体系（按项目模块）

### 1. 模型微调（LoRA / QLoRA / 全参）

| 主题 | 核心知识 |
| --- | --- |
| LoRA | 冻结原权重，注入低秩矩阵 `W' = W + BA`（r=16 等）；可训练参数 <1%；推理时 merge_and_unload 合并回原权重 |
| QLoRA | 4-bit NF4 量化底座 + Double Quant + paged optimizer；3B 模型 <16GB 显存；质量损失 <1% |
| 全参 | 效果上限最高；1.5B 也要 batch=1 + 梯度累积；lr 比 LoRA 小 10 倍（1e-5 vs 2e-4） |
| 损失掩码 | completion-only loss：prompt 部分 labels 置 -100，只对 response 计算损失 |
| 数据格式 | alpaca（指令）/ sharegpt（多轮，chat template）/ plain（续写）/ synthetic（冒烟） |
| 显存估算 | 权重 + 优化器状态 + 梯度 + 激活 + KV Cache（LoRA 只省优化器/梯度中冻结部分） |

### 2. 高性能推理（核心，JD 最相关）

| 机制 | 原理一句话 | 面试要能展开的细节 |
| --- | --- | --- |
| **Paged Attention** | KV Cache 分页（block 表映射），像虚拟内存一样按需分配 | 碎片率从 60~80% 降到 <4%；支持共享/换出；`gpu_memory_utilization` 控制预算 |
| **Continuous Batching** | 迭代级调度：每个 decode 步重排请求，完成即释放、新请求立即插入 | 对比 static batching 的短板（慢请求拖累）；chunked prefill 切长 prompt |
| **KV Cache** | 缓存历史 token 的 K/V，避免重复计算 | 显存公式 `2 × layers × kv_heads × head_dim × seq × batch × bytes` |
| **Prefix Caching** | 公共前缀 KV 复用 | vLLM 块级 vs SGLang RadixAttention 树级（token 级，更细） |
| EngineClient 抽象 | 统一 generate/chat/stream/metrics 接口，多引擎适配器 | "压测即生产"，换引擎只改配置 |
| OpenAI 兼容 | /v1/models、/v1/chat/completions、SSE 流式、/metrics | SSE 格式 `data: {...}` 结束 `[DONE]` |

### 3. Benchmark（性能评测方法论）

| 主题 | 知识 |
| --- | --- |
| 指标 | 吞吐（tokens/s）、TTFT（首 token）、ITL（token 间隔）、TPOT、E2E、成功率、显存峰值 |
| 公平性五原则 | 同权重同采样（temp=0）、预热、流式计时、双通道 token 验证、串行隔离 |
| 双通道验证 | 客户端 chunk 计数 vs 服务端 /metrics 计数增量，偏差=流式开销度量 |
| 瓶颈启发式 | 吞吐饱和、TTFT 劣化（排队/prefill）、引擎对比、失败率、显存水位 |
| 压测客户端 | asyncio + Semaphore 恒定并发；流式首 chunk 计时 |

### 4. 部署与调度

| 主题 | 知识 |
| --- | --- |
| Docker Compose 双轨 | base（监控+HF 服务）+ GPU override（设备声明），profile 门控 |
| K8s/Helm | Deployment/Service/HPA；GPU 资源声明 `nvidia.com/gpu` |
| Kueue | ClusterQueue（配额 nominalQuota）+ LocalQueue（注解接入）；常驻负载走 HPA，批任务走队列 |
| HPA 外部指标 | prometheus-adapter 暴露 `vllm:num_requests_waiting`，排队>20 扩容 |

### 5. 可观测性

| 主题 | 知识 |
| --- | --- |
| 指标体系 | DCGM（硬件）/ RED（服务：速率/错误/延迟）/ USE；vLLM `vllm:*`、自研 `minillm_*` |
| Pushgateway | 批任务指标入口（Prometheus pull 模型够不到一次性任务）；grouping_key 区分运行 |
| 告警 | 服务 down、显存>90%、排队>50、TTFT-p99>5s、abort 率>10% |

### 6. Apple Silicon 推理（差异化亮点）

| 主题 | 知识 |
| --- | --- |
| MLX | Apple 官方统一内存框架；SGLang `SGLANG_USE_MLX=1`、vllm-metal 插件都以它为后端 |
| 三引擎在 Mac | vLLM（vllm-metal 插件，macOS wheel v0.26+）/ SGLang（srt_mps extra 源码装）/ HF（MPS 原生） |
| 统一内存 | 内存带宽是绝对瓶颈；"带宽受限"是 Mac 上压测的预期结论 |

---

## Part 2：实测经验与踩坑（面试差异化核心）

> 每个都是真实发生过的，讲述用"现象 → 定位 → 修复 → 防回归"结构。

### 1. 三引擎同机压测数据（最硬的证据）

| 引擎 | c=1 | c=4 | c=8 | 结论 |
| --- | --- | --- | --- | --- |
| vLLM-MLX | ~150 | ~250 | **~480 t/s** | 近线性扩展、100% 稳定 |
| SGLang-MLX | ~155 | ~440（调参后） | ❌ Metal OOM | 单点最强、高并发缺陷 |
| HF | ~40 | ~27 | ~14 t/s | 反向下行、E2E 延迟 64s |

**面试价值**：量化了 Continuous Batching 收益（480 vs 14 = 34 倍）；证明"懂原理 + 会测量 + 会归因"。

### 2. SGLang 高并发 Metal OOM 排查

- **现象**：并发 4 起成功率骤降为 0，进程崩溃
- **定位**：查服务日志 → `RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`，发生在 `async_chained_decode_mlx`
- **原因**：SGLang MLX 后端默认 `mem_fraction_static=0.88` 把显存预算设得太满，fp16 + 并发 4 的 decode 阶段运行时内存耗尽
- **修复**：降到 0.6 后并发 4 恢复（493 t/s）；并发 8 仍 OOM → 判断为 MLX 后端高并发显存管理缺陷（vllm-metal 同条件稳定）
- **防回归**：`scripts/bench_mac.py` 内置该参数并注释原因

### 3. BPE tokenizer 压缩坑（max_len 探测脚本）

- **现象**：构造 N-token prompt，服务端只收到约一半（1000 → 501）
- **原因**：BPE 词表含 `" token"`（带前导空格）类合并 token；重复文本 re-tokenize 被压缩；且 `decode` 默认 `clean_up_tokenization_spaces=True` 破坏往返
- **修复**：超量构造 + token 空间截断 + 本地精确计数、以服务端回显为准
- **结果**：vLLM max=8192 时实测边界 **8183**（8183 prompt + 8 template + 1 输出 = 8192 数学自洽），8184 精确报 400

### 4. transformers 5.x 破坏性变更

- `TrainingArguments` 移除 `warmup_ratio` → runtime inspect 签名做版本兼容（`warmup_steps` 传小数）
- `Conv1D` 从 `modeling_utils` 迁移到 `pytorch_utils` → 导入路径回退链

### 5. httpx 流式 client 生命周期 bug

- **现象**：流式生成器在 `with httpx.Client` 退出后才被消费 → "client has been closed"
- **修复**：client 生命周期跟随生成器，finally 中关闭

### 6. MPS 内核非确定性

- **现象**：同参数两次 greedy 生成结果不同，测试断言失败
- **修复**：断言改为 token 计数/结构性质，不比较具体文本（环境特性，非代码 bug）

### 7. 网络受限环境的多重绕过

| 问题 | 方案 |
| --- | --- |
| HuggingFace 官方被墙 | `HF_ENDPOINT=https://hf-mirror.com` |
| hf-mirror 对 Python 客户端 TLS 指纹拦截（curl 正常、httpx 308 到源站） | `fetch_model.sh` 用 curl 拉模型本地化 |
| GitHub release 下载超时 | 用户协助下载 + gh-proxy.com 加速 |

### 8. uv 缓存 ≠ 共享安装

- **现象**：三个 venv 都装 torch 2.13，`du` 显示各占一份
- **实测**：inode 各不相同 → 独立副本；uv 缓存只保证不重复**下载**，安装仍是独立解压
- **结论**：多 venv 隔离是正确设计，磁盘多花 2G 买稳定性

### 9. git-filter-repo 清理大文件的两个连锁问题

- **问题 A**：327MB 模型误入库（.git 膨胀 1.1G，GitHub 单文件限 100MB）→ filter-repo 重写历史 → .git 瘦到 3.3M
- **问题 B**（连锁）：filter-repo 顺带删了磁盘上的模型目录 → 配置仍指向它 → transformers 把本地路径当 HF repo id 报 "Repo id must be..." → 改用仍在盘的 fp16 权重

### 10. 其他运维经验

- **ulimit fd 警告**：`ulimit -n 65536` 临时 / shell rc 持久；低并发可无视
- **端口冲突**：hf 服务 8000 被旧进程占用 → `lsof -ti:PORT | xargs kill`
- **vLLM/SGLang 装独立 venv**：依赖强约束（setuptools/numba/fastapi 锁死）会污染主环境；`engine_python` 字段解耦

---

## Part 3：面试高频考点速查（详见 docs/interview.md）

| 考点 | 一句话答案锚点 |
| --- | --- |
| Paged Attention 为什么省显存 | 分页管理消除碎片（60~80% → <4%） |
| Continuous Batching 本质 | 调度粒度从请求降到迭代 |
| KV Cache 显存公式 | `2×layers×kv_heads×head_dim×seq×batch×bytes` |
| vLLM vs SGLang | RadixAttention 前缀树 vs 块级 prefix caching |
| 三引擎为什么公平 | 同权重同采样 + 串行隔离 + 双通道验证 |
| 批任务指标为何走 Pushgateway | pull 模型够不到一次性任务 |
| Kueue 解决什么 | GPU 配额 + 排队 + 优先级，非"资源够就调度" |
