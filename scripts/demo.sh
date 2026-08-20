#!/usr/bin/env bash
# miniLLM 全链路演示：微调 → 推理服务 → OpenAI API 调用 → 压测 → 报告
# Mac (CPU/MPS) 可完整运行，约 1~2 分钟；无需 GPU、无需网络（模型已本地化）
# 用法: make demo   或   bash scripts/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export HF_HOME="${HF_HOME:-data/cache/huggingface}"

step() { echo; echo "══════════════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════════════"; }

step "[1/5] 质量基线：单元测试 + lint"
"$PY" -m pytest tests -m "not hub and not bench" -q
"$PY" -m ruff check src tests scripts

step "[2/5] 微调流水线：tiny-Llama LoRA（合成指令数据，5 步）"
"$PY" -m minillm.cli train --config configs/train/smoke_hub.yaml \
    --override train.output_dir=runs/demo-finetune \
    --override data.synthetic_n=32 \
    --override train.max_steps=5
echo "  产物: runs/demo-finetune/{adapter,export,metrics.json,eval_samples.json}"

step "[3/5] 启动 OpenAI 兼容推理服务（加载微调产物）"
"$PY" -m minillm.cli serve --config configs/serve/hf_tiny.yaml \
    --override model.name_or_path=runs/demo-finetune/export &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
    curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 1
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "服务启动失败"; exit 1; }
echo "  服务就绪: http://127.0.0.1:8000 (v1/models, v1/completions, v1/chat/completions, metrics)"

step "[4/5] OpenAI API 调用（/v1/models + /v1/chat/completions + 流式）"
curl -s http://127.0.0.1:8000/v1/models | "$PY" -c "import json,sys; print('  模型:', [m['id'] for m in json.load(sys.stdin)['data']])"
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"runs/demo-finetune/export\",\"messages\":[{\"role\":\"user\",\"content\":\"什么是 LoRA？\"}],\"max_tokens\":16,\"temperature\":0.0}" \
    | "$PY" -c "import json,sys; d=json.load(sys.stdin); print('  回复:', d['choices'][0]['message']['content'][:60]); print('  usage:', d['usage'])"
echo "  流式示例（前 3 个 chunk）:"
curl -sN --max-time 30 http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"runs/demo-finetune/export","prompt":"Once upon a time","max_tokens":4,"stream":true}' \
    > runs/demo-stream.txt
head -3 runs/demo-stream.txt | cut -c1-120

step "[5/5] 压测：经 vllm 适配器并发打服务（吞吐/TTFT/ITL）"
"$PY" -m minillm.cli bench --config configs/bench/smoke_http.yaml \
    --override engines.vllm.model=runs/demo-finetune/export
echo "  报告: runs/bench/bench-smoke-http-*/bench_report.{json,md}"

step "✅ 全链路演示完成：微调 → 推理(OpenAI API) → 压测 → 报告"
