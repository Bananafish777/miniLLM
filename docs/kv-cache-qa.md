# KV Cache 及优化手段考点（可背诵版）

> 推理加速方向**必考**。KV Cache 是"自回归推理为什么慢/为什么吃显存"的根源，也是 Paged Attention、GQA、前缀缓存等一系列优化的起点。
> 配套：`docs/project-knowledge.md`（实测）、`docs/benchmark-results.md`（三引擎数据）。

---

## Q1. KV Cache 是什么？为什么需要它？

**一句话**：自回归生成时，每生成一个新 token 都要对**所有历史 token** 算注意力；KV Cache 把历史 token 的 Key/Value 向量缓存下来，避免每步重复计算。

**展开**：
- 自回归是"一步出一个 token，下一个 token 依赖前面所有 token"
- 没有缓存：生成第 n 个 token 时要重新算前 n-1 个 token 的 K/V → 计算量 O(n²)，且是重复劳动
- 有缓存：历史 token 的 K/V 只算一次存起来，新 token 只需算自己的 K/V，再做 attention → **每步计算从 O(n²) 降到 O(n)**
- 本质：**用显存换计算**——省掉重复计算，代价是显存随序列长度线性增长
- 记忆点：`自回归 + attention 重复计算 → 缓存 K/V → 显存换计算`

**追问应对**：若问"为什么只缓存 K/V 不缓存 Q"→ Q 只和当前 token 有关（下一个 token 用不到历史的 Q），而每个历史 token 的 K/V 都会被后续所有 token 用到。

---

## Q2. KV Cache 显存怎么估算？（必考公式）

**一句话**：`显存 = 2 × 层数 × KV头数 × head_dim × 序列长度 × batch × 字节数`（×2 是 K 和 V）。

**展开**：
- 每层的 KV Cache = `2（K+V）× num_kv_heads × head_dim × seq_len × batch × dtype_bytes`
- 例（Qwen3-0.6B 举例）：28 层 × 4 kv_heads × 128 head_dim × 8192 seq × 1 batch × 2 字节(fp16) ≈ **2.3GB** —— 单条 8K 请求的 KV 就要这么多
- 关键洞察：**KV Cache 随 seq_len × batch 线性增长**，高并发（batch 大）+ 长上下文（seq 长）时，KV 常超过模型权重本身，成为显存大头
- 记忆点：`2 × L × kv_heads × head_dim × seq × batch × bytes`；**并发和长度是 KV 的两个放大器**

**追问应对**：若让估算"模型权重 vs KV 谁大"→ 权重固定，KV 随并发/长度涨，高并发长上下文下 KV 反超权重，这也是 `gpu_memory_utilization` 要留预算的原因。

---

## Q3. KV Cache 带来什么问题？

**一句话**：显存占用大、且有碎片和浪费，直接限制**并发数**和**最大上下文长度**。

**展开**：
- **显存压力**：并发 × 长序列，KV 轻松几十 GB
- **碎片浪费**（传统实现）：按请求**最大可能长度**预分配连续显存 → 内部碎片（预留不用）+ 外部碎片（无法复用），碎片率可达 60~80%
- **不能复用**：不同请求的公共前缀（如 system prompt、few-shot 例子）重复存储
- 后果：显存不够 → 要么降并发（吞吐↓），要么截断长度（质量↓）
- 记忆点：`三宗罪：占得多、有碎片、不复用`

---

## Q4. 优化手段总览（先给框架，面试先答这个）

**一句话**：从"存得省、存得少、存得巧"三个角度：**分页（去碎片）→ 复用（去冗余）→ 压缩（降精度/减头数/减长度）→ 换出（offload）**。

| 角度 | 手段 | 收益 |
| --- | --- | --- |
| 存得巧（存储管理） | Paged Attention | 碎片 60~80% → <4% |
| 存得巧（复用） | Prefix Caching / RadixAttention | 公共前缀只存一份 |
| 存得少（架构） | GQA / MQA | KV 头数降到 1/4~1/8 |
| 存得少（精度） | KV 量化（int8/fp8） | 显存减半 |
| 存得少（长度） | 滑动窗口 / 稀疏注意力 | 只存最近/部分 |
| 存得巧（换出） | CPU offload | 冷 KV 放 CPU |

**记忆点**：`分页 → 复用 → 压缩 → 换出`，按"最通用到最极端"排序。

---

## Q5. Paged Attention（存储管理，vLLM 核心）

**一句话**：把 KV Cache 切成固定大小的 block，用**块表（block table）**做虚拟→物理映射，像操作系统分页一样按需分配。

**展开**：
- 传统：按请求最长序列预分配**连续**显存 → 碎片
- Paged：KV 按 block（如 16 token/块）存，逻辑块 → 物理块经块表映射
- 收益：① 按需分配，碎片率降到 <4%；② 相邻请求可共享物理块（prefix sharing）；③ 支持抢占时把 block 换出
- 类比：**虚拟内存分页**——逻辑连续、物理不连续
- 记忆点：`块表映射、按需分配、碎片<4%、可共享可换出`

**追问应对**：若问和 FlashAttention 关系 → 正交：Paged 管"KV 存哪、怎么分块管理"，FlashAttention 管"单个 attention 计算怎么高效（分块 kernel）"，vLLM 两者同时用。

---

## Q6. Prefix Caching / RadixAttention（复用）

**一句话**：不同请求的**公共前缀**（system prompt、few-shot 例子、多轮历史）的 KV 只算一次、共享复用。

**展开**：
- 场景：同一 system prompt + 多个用户问题、多轮对话的历史部分、few-shot 前缀——这些前缀的 KV 每次都会重复计算且相同
- **vLLM 前缀缓存**：块级（block 粒度）复用，`--enable-prefix-caching`
- **SGLang RadixAttention**：**token 级**前缀树（radix tree），更细粒度，长共享前缀场景收益更大
- 收益：**TTFT 大幅下降**（prefill 阶段省掉了公共前缀的计算）+ 显存复用
- 记忆点：`公共前缀只算一次；vLLM 块级 vs SGLang 树级（更细）`

**追问应对**：若问"什么场景收益最大"→ 长 system prompt、多轮对话、RAG 里同一上下文问多个问题。

---

## Q7. GQA / MQA（架构层，KV 头数减少）

**一句话**：让多个 Query 头**共享一组 KV 头**，直接把 KV Cache 缩小数倍。

**展开**：
- **MHA**（多头）：每个 Q 头都有自己的 K/V 头 → KV 头数 = Q 头数（最大）
- **MQA**（多查询）：所有 Q 头共享 1 组 K/V → KV 缩小到 1/头数（省最多，但质量略降）
- **GQA**（分组查询）：折中，几组 Q 头共享 1 组 K/V（如 8:1 或 4:1）→ 显存和质量平衡，**主流模型（Llama-2/3、Qwen2/3）都用 GQA**
- 收益：KV Cache 直接缩小 `num_heads / num_kv_heads` 倍
- 记忆点：`MHA 每个头独立 KV → MQA 全共享 → GQA 分组折中，主流是 GQA`

**追问应对**：若问为什么 GQA 几乎不掉精度 → KV 头只是压缩了注意力键值维度，且实验证明分组共享对多数任务影响极小。

---

## Q8. KV Cache 量化（精度换显存）

**一句话**：把 KV 从 fp16/bf16 量化到 int8/fp8，显存直接减半，代价是精度损失。

**展开**：
- 思路：KV 是"缓存值"而非权重，对精度相对不敏感 → 适合激进量化
- 主流：**fp8 / int8**（vLLM 支持 `--kv-cache-dtype fp8`）；更激进 int4（损失明显，需谨慎）
- 收益：显存减半 → 并发翻倍或长度翻倍
- 权衡：量化误差累积，超长序列/敏感任务可能掉点，需验证
- 记忆点：`KV 对精度不敏感 → int8/fp8 减半显存`

**追问应对**：若问"为什么 KV 比权重好量化"→ 权重影响所有前向计算，KV 只影响注意力打分，且注意力本身有 softmax 归一化，对量化误差更鲁棒。

---

## Q9. 滑动窗口 / 稀疏注意力（少存）

**一句话**：不需要所有历史 token 的 KV，只存最近的（滑动窗口）或重要的（稀疏），直接减少缓存量。

**展开**：
- **滑动窗口 attention**：每个 token 只看最近 W 个 → KV 只存 W 长度（如 Mistral 的 4096 窗口）
- **稀疏/局部注意力**：长文本里只对局部或选定位置做注意力
- 权衡：牺牲"无限长上下文"换显存和速度，适合局部性强的任务
- 记忆点：`只存窗口内/重要的，适合局部任务`

---

## Q10. CPU offload 与长度控制（工程兜底）

**一句话**：冷 KV 换到 CPU（offload），或用 `max_model_len`/`gpu_memory_utilization` 硬性控制显存预算。

**展开**：
- **CPU offload**：不活跃的 KV block 换到内存，用到再换回（慢但能跑更长/更多并发）
- **max_model_len**：截断最大序列长度，直接限制 KV 上限（本项目 8192）
- **gpu_memory_utilization**：控制 GPU 给"权重 + KV Cache"的总预算比例（vLLM 默认 0.9，本项目 0.85）
- 记忆点：`offload 换出、max_model_len 限长度、gpu_memory_utilization 定预算`

---

## Q11. 实战：vLLM 里 KV Cache 相关参数怎么调？

**一句话**：`gpu_memory_utilization`（总预算）+ `max_model_len`（长度上限）+ `enable_prefix_caching`（复用）+ `--kv-cache-dtype`（量化）。

**展开**（对应本项目 `configs/serve/vllm_mac.yaml` / GPU 版）：
| 参数 | 作用 | 调优方向 |
| --- | --- | --- |
| `gpu_memory_utilization` | 显存给权重+KV 的预算比例 | 高并发/长上下文调低给 KV 留空间，反之调高 |
| `max_model_len` | KV 长度上限 | 按业务最长需求设，别拍太大浪费 |
| `enable_prefix_caching` | 前缀 KV 复用 | 多轮对话/长 system prompt 必开 |
| `--kv-cache-dtype fp8` | KV 量化 | 显存紧张时开，先验证精度 |
| `max_num_seqs` | 并发批处理上限 | 受 KV 预算约束，显存不够就降 |

**记忆点**：`预算、长度、复用、量化` 四旋钮。

---

## Q12. 结合项目实测：三引擎压测里的 KV 相关现象

- **SGLang 高并发 Metal OOM**：fp16 + 并发 4 时 decode 阶段显存耗尽 → 本质是**KV Cache + 运行时中间量超出预算**；调低 `mem_fraction_static`（0.88→0.6）恢复 → 这正是"KV 显存预算是并发瓶颈"的实证
- **vLLM 高并发稳定**：并发 8 达 480 t/s → Paged Attention 的分页管理在高并发下更健壮
- **max_len 边界实测**：配置 8192，实测 prompt 完整接收上限 8183（8183 + 8 template + 1 输出 = 8192 精确自洽）→ 理解"长度上限如何吃掉 KV 预算"

**面试价值**：能把"KV 显存预算"从公式讲到**自己踩过的 OOM 实测**，是"懂原理 + 有实操"的证明。

---

## 追问应对汇总

| 追问 | 要点 |
| --- | --- |
| 为什么不缓存 Q | Q 只服务当前 token，历史 Q 不会被后续用到 |
| KV 和权重谁占显存多 | 权重固定，KV 随并发×长度涨，高并发长上下文下 KV 反超 |
| FlashAttention 和 Paged Attention 区别 | 前者优化单次计算（分块 kernel），后者优化存储（分页管理），正交可叠加 |
| 什么时候用前缀缓存 | 长 system prompt、多轮对话、同一上下文多问题（RAG） |
| GQA 为什么主流 | 压缩 KV 数倍且几乎不掉精度 |
| 怎么定位"显存不够" | 看 KV 预算：降并发 / 限长度 / 开量化 / 调 gpu_memory_utilization |

---

## 30 秒速记卡

```
KV Cache = 缓存历史 token 的 K/V，显存换计算（O(n²)→O(n)）
显存公式 = 2 × L × kv_heads × head_dim × seq × batch × bytes
问题 = 占得多、有碎片、不复用
优化 = 分页(去碎片) → 复用(前缀缓存) → 压缩(GQA/量化) → 换出(offload)
Paged Attention = 块表映射，碎片 60~80% → <4%
GQA = 多 Q 共享 KV 头，显存缩小 num_heads/kv_heads 倍
vLLM 四旋钮 = gpu_memory_utilization / max_model_len / prefix_caching / kv-cache-dtype
```
