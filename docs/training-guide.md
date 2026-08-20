# miniLLM 模型微调使用指南（生成式模型）

> 面向"用本项目微调一个自己的大模型"的完整操作手册。适用：**decoder-only 生成式模型**（Qwen2.5 / Llama-3 / Mistral 等一切 HuggingFace causal LM），规模 1B~13B。
>
> 注意：本项目微调流水线为自回归生成范式，**不适用于 BERT/GLiNER 等 encoder 模型**（NER/分类等任务请另走相应工具链）。

---

## 0. 模型选型（先想清楚训什么）

| 模型 | fp16 权重显存 | 单卡 24GB 推荐模式 | 适用 |
| --- | --- | --- | --- |
| Qwen2.5-0.5B | ~1.1 GB | LoRA（极轻） | 冒烟/入门 |
| Qwen2.5-1.5B | ~3.3 GB | **LoRA（默认推荐）** | 指令微调、风格迁移、轻任务 |
| Qwen2.5-3B / Llama-3.2-3B | ~6.5 GB | LoRA / QLoRA | 中文任务首选 Qwen 系 |
| 7B~8B（Llama-3.1-8B 等） | ~16 GB | QLoRA（需 bitsandbytes） | 复杂任务 |
| 13B+ | >26 GB | 需多卡/量化，超出本指南默认场景 | — |

**模式选择决策**：
- **LoRA**（默认）：只训 0.1%~1% 参数，显存省、速度快、adapter 可插拔；1B~3B 指令微调首选
- **QLoRA**：4-bit 量化底座，显存最省（3B < 16GB），质量损失 <1%；需要 NVIDIA GPU
- **全参**：效果上限最高，但 1.5B 也需要大显存（batch=1 + 梯度累积）；追求极致效果且资源充足时用

---

## 1. 环境准备

```bash
# ① 本机（Mac 开发机）：安装依赖 + 自检
make setup
make test          # 67 项测试，确认环境健康
make smoke         # 离线冒烟：scratch GPT-2 + LoRA 全链路（无需 GPU/网络）

# ② GPU 训练机（Linux + NVIDIA，推荐单卡 ≥24GB）：
#    方式一：直接装环境（同 make setup）
#    方式二：容器化（GPU 训练，见 docs/deployment.md）
cd deploy/compose
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --env-file .env --profile train run --rm train

# ③ HuggingFace 官方被墙时使用镜像（已写入 .env.example）
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 2. 准备训练数据（决定效果的最关键一步）

把数据放进 `data/raw/`，四种格式任选（`data.format` 指定）：

**① alpaca（指令微调，最常用）** — `data/raw/my_data.jsonl`：
```json
{"instruction": "把下面的句子翻译成英文", "input": "今天天气很好", "output": "The weather is nice today."}
{"instruction": "写一个 Python 函数求平方", "input": "", "output": "def square(x):\n    return x * x"}
```

**② sharegpt（多轮对话）**：
```json
{"conversations": [
  {"from": "human", "value": "你好"},
  {"from": "gpt", "value": "你好！有什么可以帮你？"},
  {"from": "human", "value": "介绍一下你自己"},
  {"from": "gpt", "value": "我是经过微调的 AI 助手。"}
]}
```

**③ plain（续写/领域语料）**：
```json
{"text": "这里是领域文档的正文内容……"}
```

**④ synthetic（无数据冒烟用，不需要文件）**：`data.format: synthetic`

**数据质量要点**：
- 数量：LoRA 指令微调建议 **1k~10k 条**起步（质量 > 数量，宁可 2k 条干净数据不要 20k 条脏数据）
- 覆盖：把你的任务拆成 3~5 类变体，每类均匀覆盖
- 清洗：去重（近似重复）、过滤超长（>2048 token）、格式一致（标点/换行）、去掉含错别字的样本
- 切分：`data.val_size: 0.05` 自动留验证集，不要手动混入

---

## 3. 配置并启动训练

三份现成配置（`configs/train/`）改数据路径即可：

```bash
# LoRA 微调 Qwen2.5-1.5B（默认推荐，单卡 24GB 首选）
make train CONFIG=configs/train/lora_qwen25_1p5b.yaml \
    --override data.path=data/raw/my_data.jsonl \
    --override data.format=alpaca

# QLoRA 微调 Qwen2.5-3B（显存最省）
make train CONFIG=configs/train/qlora_qwen25_3b.yaml \
    --override data.path=data/raw/my_data.jsonl \
    --override data.format=sharegpt

# 全参微调（追求效果上限，需要大显存/多卡）
make train CONFIG=configs/train/full_qwen25_1p5b.yaml \
    --override data.path=data/raw/my_data.jsonl
```

**常用超参速查**（`--override key=value` 覆盖）：

| 参数 | LoRA 推荐 | 全参推荐 | 说明 |
| --- | --- | --- | --- |
| `train.learning_rate` | 2e-4 | 1e-5 | 全参必须小 10 倍 |
| `train.num_train_epochs` | 3 | 2~3 | 数据多可减少 |
| `train.per_device_train_batch_size` | 4（24GB 卡） | 1 + grad_accum 8 | 显存不够就降 batch 加累积 |
| `data.max_seq_len` | 2048 | 2048 | 长文本任务可升 4096（显存↑） |
| `lora.r` | 16 | — | 任务复杂可升 32 |
| `train.gradient_checkpointing` | true | true | 显存换速度 |

**训练中看什么**：
- 日志：`train_loss` 应随步数下降，`eval_loss` 同步下降（过拟合时 eval 先回升）
- 显存：`nvidia-smi -l 1`，峰值别超卡显存 90%
- 产物：`runs/<output_dir>/checkpoint-*` 自动轮换（save_total_limit=2）

---

## 4. 评估与验收

```bash
# ① 自动评估：训练结束自动产出
runs/<run>/metrics.json         # eval loss / tokens-per-sec / 导出信息
runs/<run>/eval_samples.json    # 微调后生成样例（prompt → generation）

# ② 自定义评测问题（训练前在配置里写好，训练完自动生成回答）
# configs/train/xxx.yaml
eval:
  prompts:
    - "把这句话翻译成英文：今天天气很好"
    - "写一个二分查找的 Python 实现"
```

**验收标准**：eval loss 明显低于初始值；生成样例质量符合任务要求；拿 20~50 条真实任务输入做人工盲测，对比微调前后差异。

---

## 5. 导出与部署（微调 → 推理闭环）

```bash
# 训练自动导出（LoRA 已合并到 base 权重）：
runs/<run>/export/    # 完整模型：config.json + model.safetensors + tokenizer

# ① 部署 vLLM（OpenAI 兼容，GPU 环境）
make serve CONFIG=configs/serve/vllm_qwen25_1p5b.yaml \
    --override model.name_or_path=runs/<run>/export

# ② 验证
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"runs/<run>/export","messages":[{"role":"user","content":"测试你的任务"}],"max_tokens":256}'

# ③ 压测验证性能（吞吐/TTFT）
make bench CONFIG=configs/bench/smoke_http.yaml --override engines.vllm.model=runs/<run>/export
```

**可选增强**：`export.register_mlflow: true` + 设置 `MLFLOW_TRACKING_URI` 记录实验；`push_gateway` 配置后训练指标自动入库 Grafana。

---

## 6. Mac 开发机的边界（诚实说明）

| 能做 | 不能做 |
| --- | --- |
| 数据准备、配置编写、代码调试 | 1.5B+ 模型的有效微调（CPU/MPS 慢到无意义） |
| `make smoke`（scratch 微型模型全链路） | vLLM / SGLang（仅 CUDA） |
| tiny 模型（<100M）LoRA 演示 | QLoRA（bitsandbytes 仅 Linux+CUDA） |
| 单元测试 / 部署工件校验 | 真实 GPU 指标（DCGM 等） |

**推荐工作流**：Mac 上准备数据/配置/脚本 → 推送到 GPU 机器（或容器）执行训练 → 结果回传 Mac 分析。全程配置驱动，两边行为一致。

---

## 7. 常见问题 FAQ

| 问题 | 处理 |
| --- | --- |
| CUDA out of memory | 降 batch → 开 gradient_checkpointing → 降 max_seq_len → 换 QLoRA |
| 训练 loss 不降 | lr 过大（全参试试 1e-5）；数据格式错误（检查 tokenize 后 labels 掩码）；epoch 太少 |
| 输出乱码/无限重复 | 数据问题优先（重复样本/格式混用）；推理时 temperature 调低或 greedy |
| HF 下载失败 | `export HF_ENDPOINT=https://hf-mirror.com`；或 `scripts/fetch_model.sh` 拉到本地后用本地路径 |
| 中文效果差 | 换 Qwen 系（中文预训练最强）；确保数据是目标语言；不要混语言样本 |
| pad_token 报错 | 已自动处理（复用 eos 或新增 <pad>）；自定义模型需手动配置 |
| 训练完模型"没学会" | 检查 eval_samples.json 是否真的用了微调权重（export 是合并后的）；数据量是否过少 |
| 想中途换数据/超参 | 一切可 `--override`，配置驱动，改完重跑即可（产物按 output_dir 隔离） |

---

## 8. 完整端到端示例（照抄可跑）

```bash
# ① 环境
make setup && export HF_ENDPOINT=https://hf-mirror.com

# ② 数据（示例：3 条中文指令）
mkdir -p data/raw
cat > data/raw/my_data.jsonl <<'EOF'
{"instruction": "把下面的话翻译成英文", "input": "今天天气很好", "output": "The weather is nice today."}
{"instruction": "写一个函数计算阶乘", "input": "", "output": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"}
{"instruction": "总结这段话", "input": "大模型通过海量文本预训练获得语言能力，再通过指令微调对齐人类偏好。", "output": "大模型先预训练再指令微调以对齐人类偏好。"}
EOF

# ③ 训练（GPU 机器；Mac 上仅 tiny 模型可跑）
make train CONFIG=configs/train/lora_qwen25_1p5b.yaml \
    --override data.path=data/raw/my_data.jsonl \
    --override data.synthetic_n=1 \
    --override train.output_dir=runs/my-first-finetune \
    --override train.num_train_epochs=2

# ④ 检查产物
ls runs/my-first-finetune/{adapter,export}
cat runs/my-first-finetune/eval_samples.json

# ⑤ 部署（GPU）
make serve CONFIG=configs/serve/vllm_qwen25_1p5b.yaml \
    --override model.name_or_path=runs/my-first-finetune/export
```
