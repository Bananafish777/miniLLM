# Web 管理控制台使用说明

`minillm web` 提供一站式管理面板：**引擎状态 · GPU 硬件指标 · Benchmark 结果 · 微调运行 · 多引擎对话**。自包含实现（FastAPI + 原生 JS），无构建工具链。

## 启动

```bash
# 先启动要监控/对话的推理引擎（任意数量，可为 vLLM/SGLang/HF 混合）
minillm serve --config configs/serve/vllm_mac.yaml         # vLLM-MLX（:8010）
minillm serve --config configs/serve/hf_tiny.yaml          # HF（:8000）

# 启动控制台（默认 http://127.0.0.1:8080）
minillm web --config configs/web/admin.yaml
```

## 面板内容

| 区块 | 数据来源 | 内容 |
| --- | --- | --- |
| 推理引擎状态 | 各引擎 `/metrics` + `/v1/models` | 在线状态、模型、请求总数、生成 tokens、**tokens/s（相邻抓取差值）**、排队/运行请求、Cache 命中率（兼容 `minillm_*` / `vllm:*` / `sglang:*` 三类指标名） |
| GPU 硬件指标 | Prometheus（`prometheus_url` 配置后启用） | GPU 利用率 / 显存 / 温度（DCGM 指标经 `/api/prometheus` PromQL 代理） |
| Benchmark 结果 | `runs/bench/*/bench_report.json` | 实验列表（点击展开：引擎吞吐柱状对比 + 瓶颈分析 findings） |
| 微调运行 | `runs/*/metrics.json` | 实验、模式、模型、设备、eval loss、训练 tokens/s |
| **对话界面** | `/api/chat`（转发到所选引擎） | 多引擎聊天：引擎下拉切换、SSE 流式逐 token 渲染、多轮对话、停止/清空 |

## 对话界面

顶部 **对话** tab：从 `admin.yaml` 的 `engines` 列表选择引擎（如 vllm-mlx / hf-local / sglang-mlx），
输入消息后经 `/api/chat` 转发到该引擎的 `/v1/chat/completions`（OpenAI 兼容格式透传，支持流式 SSE）。

```bash
minillm serve --config configs/serve/vllm_mac.yaml   # 目标引擎在线
minillm web --config configs/web/admin.yaml          # 打开 http://127.0.0.1:8080 → 对话 tab
```

## API

| 端点 | 说明 |
| --- | --- |
| `/` | 控制台页面 |
| `/api/status` | 引擎实时状态（含 tokens/s 速率计算） |
| `/api/engines` | 引擎配置列表 |
| `/api/chat` | 对话代理（POST：engine/messages/max_tokens/temperature/stream，转发到引擎并透传 SSE） |
| `/api/bench?limit=N` | Benchmark 报告列表（新→旧） |
| `/api/train?limit=N` | 微调运行摘要 |
| `/api/prometheus?query=<PromQL>` | PromQL 代理（需配置 prometheus_url） |
| `/api/health` | 探活 |

## 配置（configs/web/admin.yaml）

```yaml
engines:
  - name: vllm-mlx            # 展示名（对话界面下拉也按此列表）
    type: vllm               # hf | vllm | sglang（仅影响展示）
    base_url: http://127.0.0.1:8010
  - name: hf-local
    type: hf
    base_url: http://127.0.0.1:8000
bench_dir: runs/bench        # 报告目录
train_dir: runs              # 训练运行目录
refresh_interval_s: 5        # 前端轮询间隔
prometheus_url: http://127.0.0.1:9090   # 可选：GPU 面板
```

> 引擎离线时卡片显示红色状态与错误信息，不影响其他区块；目录缺失时对应区块显示"暂无数据"。
> 对话界面选择离线引擎时返回 503 错误提示。
