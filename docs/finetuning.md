# 微调流水线使用说明（M1）

`minillm train` 覆盖完整微调流程：**数据准备 → Tokenize → 训练（LoRA/QLoRA/全参）→ 评估 → 导出 → 实验追踪**。

## 快速上手

```bash
# 1) 安装环境
make setup

# 2) 离线冒烟测试（scratch 微型 GPT-2 + 合成数据，无需 GPU/网络，<1 分钟）
make smoke

# 3) 真实微调（需要数据文件；HF 官方被墙时自动走 hf-mirror.com）
make train CONFIG=configs/train/lora_qwen25_1p5b.yaml
```

## 三种微调模式

| 模式 | 配置 | 适用场景 |
| --- | --- | --- |
| LoRA | `finetune_mode: lora` | 默认；24GB 单卡可微调 3B 模型 |
| QLoRA | `finetune_mode: qlora` + `model.load_in_4bit: true` | 显存更省（NF4 量化底座，需 bitsandbytes + GPU） |
| 全参 | `finetune_mode: full` | 追求上限效果；显存要求高，建议多卡 |

## 数据格式

| format | 字段 | 说明 |
| --- | --- | --- |
| `alpaca` | `instruction` / `input`(可选) / `output` | JSON 或 JSONL，Alpaca 风格指令数据 |
| `sharegpt` | `conversations: [{from, value}]` | 多轮对话；每个 assistant 轮次生成一个监督样本，自动用 chat template |
| `plain` | `text` | 纯文本续写（next-token prediction） |
| `synthetic` | — | 本地合成指令数据（冒烟/演示，无需数据文件） |

```bash
make train CONFIG=configs/train/lora_qwen25_1p5b.yaml \
    --override data.format=alpaca data.path=data/raw/my_data.jsonl
```

## 产物（runs/<output_dir>/）

| 产物 | 说明 |
| --- | --- |
| `adapter/` | LoRA adapter 权重 + tokenizer（PEFT 可独立加载） |
| `export/` | **合并后的完整模型**（`merge_and_unload`），M2 中直接喂给 vLLM |
| `metrics.json` | 训练摘要：loss / samples-per-sec / tokens-per-sec / 导出信息 |
| `eval_samples.json` | 微调后生成质量样例（prompt → generation） |
| `checkpoint-*` | transformers 检查点（按 `save_total_limit` 轮换） |

## 实验追踪（可选）

```bash
# 启动自托管 MLflow（或复用已有服务）
uv run mlflow server --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root data/mlruns

export MLFLOW_TRACKING_URI=sqlite:///data/mlflow.db
make train CONFIG=configs/train/lora_qwen25_1p5b.yaml   # 自动记录参数/指标/产物
```

未设置 `MLFLOW_TRACKING_URI` 时自动跳过，不影响训练主流程。

## 配置字段速查

完整字段见 `src/minillm/train/config.py`（pydantic 模型即文档）。常用覆盖：

```bash
--override model.name_or_path=Qwen/Qwen2.5-3B-Instruct   # 换模型
--override data.max_seq_len=4096                          # 序列长度
--override train.learning_rate=1e-4                       # 学习率
--override train.num_train_epochs=5                       # 轮数
--override lora.r=32                                      # LoRA 秩
--override export.merge_adapter=false                     # 只保留 adapter 不合并
```

## 本地开发（无 GPU）

- 训练自动降级：CUDA → MPS → CPU（`dtype: auto` 时 CPU/MPS 用 fp32）
- 全链路可离线验证：`configs/train/smoke_scratch.yaml`
- hub 路径验证（走镜像）：`make smoke-hub`
