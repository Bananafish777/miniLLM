#!/usr/bin/env python3
"""同机三引擎压测编排：vLLM-MLX vs SGLang-MLX vs HF（统一 Qwen3-0.6B fp16 权重）。

纪律（保证公平性）：
- 三引擎**串行**执行：启动引擎 → 压测 → 关闭 → 下一个（统一内存带宽是共享资源）
- 同一模型权重（data/models/Qwen3-0.6B，fp16 原始）
- 同一压测参数（temperature=0、流式计时、预热、双通道 token 验证）

用法:
    python scripts/bench_mac.py [--matrix smoke|full] [--engines hf,sglang,vllm]

产出:
    runs/bench/bench-mac-three-engine-<ts>/  各引擎报告 + summary.md 对比总表
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
MODEL = "data/models/Qwen3-0.6B"
PY = sys.executable  # 项目 venv 解释器

MATRICES = {
    "smoke": {
        "concurrency": ["[1,2]"],
        "input_tokens": ["[32]"],
        "output_tokens": ["[32]"],
        "requests_per_concurrency": "2",
    },
    "full": {
        "concurrency": ["[1,4,8]"],
        "input_tokens": ["[32,256]"],
        "output_tokens": ["[32,128]"],
        "requests_per_concurrency": "4",
    },
}

# (引擎名, HTTP 压测用 type, 端口, 启动命令[list], 额外环境变量)
ENGINES = {
    "hf": {
        "type": "vllm",  # HF 服务同样用 OpenAI 协议压测（vllm adapter 打任意 OpenAI 端点）
        "port": 8000,
        "cmd": [
            PY, "-m", "minillm.cli", "serve",
            "--config", "configs/serve/hf_tiny.yaml",
            "--override", f"model.name_or_path={MODEL}",
            "--override", "model.attn_impl=sdpa",
        ],
        "env": {},
        "log": "bench-hf.log",
    },
    "sglang": {
        "type": "sglang",
        "port": 30000,
        "cmd": [
            str(ROOT / ".venv-sglang/bin/python"), "-m", "sglang.launch_server",
            "--model-path", MODEL, "--disable-cuda-graph",
            "--host", "127.0.0.1", "--port", "30000", "--enable-metrics",
            # MLX 后端默认 mem_fraction_static=0.88 会在并发 decode 时 Metal OOM；
            # 降到 0.6 给运行时留余量（实测 vllm-metal 同并发无此问题）
            "--mem-fraction-static", "0.6",
        ],
        "env": {"SGLANG_USE_MLX": "1"},
        "log": "bench-sglang.log",
    },
    "vllm": {
        "type": "vllm",
        "port": 8010,
        "cmd": [
            str(ROOT / ".venv-vllm-metal/bin/vllm"), "serve", MODEL,
            "--host", "127.0.0.1", "--port", "8010",
            "--max-model-len", "8192", "--max-num-seqs", "256",
        ],
        "env": {},
        "log": "bench-vllm.log",
    },
}

_COMMON_ENV = {
    "HF_HOME": str(ROOT / "data/cache/huggingface"),
    "SGLANG_CACHE_DIR": str(ROOT / "data/cache/sglang"),
    "TORCHINDUCTOR_CACHE_DIR": str(ROOT / "data/cache/sglang/inductor"),
    "VLLM_CACHE_DIR": str(ROOT / "data/cache/vllm"),
}


def _wait_ready(url: str, timeout: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = httpx.get(f"{url}/v1/models", timeout=3)
            if resp.status_code == 200 and MODEL in resp.text:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    return False


def run_bench(engine: str, out_dir: Path, matrix: dict) -> None:
    cfg = ENGINES[engine]
    overrides = [
        f"engines.vllm.type={cfg['type']}",
        f"engines.vllm.base_url=http://127.0.0.1:{cfg['port']}",
        f"engines.vllm.model={MODEL}",
        f"matrix.concurrency={matrix['concurrency'][0]}",
        f"matrix.input_tokens={matrix['input_tokens'][0]}",
        f"matrix.output_tokens={matrix['output_tokens'][0]}",
        f"run.requests_per_concurrency={matrix['requests_per_concurrency']}",
        f"output_dir={out_dir / f'engine-{engine}'}",
    ]
    cmd = [PY, "-m", "minillm.cli", "bench", "--config", "configs/bench/smoke_http.yaml"]
    for o in overrides:
        cmd += ["--override", o]
    print(f"  [bench] {engine}: " + " ".join(o.split('=')[0] for o in overrides))
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, **_COMMON_ENV})


def run_one(engine: str, out_dir: Path, matrix: dict) -> bool:
    cfg = ENGINES[engine]
    print(f"=== [{engine}] 启动引擎 (port {cfg['port']}) ===")
    log_path = out_dir / cfg["log"]
    env = {**os.environ, **_COMMON_ENV, **cfg["env"]}
    proc = subprocess.Popen(
        cfg["cmd"], cwd=ROOT, env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )
    try:
        ok = _wait_ready(f"http://127.0.0.1:{cfg['port']}", timeout=240)
        if not ok:
            print(f"  ✗ [{engine}] 引擎未就绪（看 {log_path}）")
            return False
        print(f"  ✓ [{engine}] 就绪")
        run_bench(engine, out_dir, matrix)
        return True
    finally:
        print(f"  [{engine}] 关闭引擎")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def summarize(out_dir: Path) -> None:
    """合并各引擎 bench_report.json 为对比总表。"""
    rows: list[dict] = []
    for engine in ENGINES:
        report = out_dir / f"engine-{engine}" / "bench_report.json"
        if not report.exists():
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        for m in data.get("metrics", []):
            rows.append({
                "engine": engine, "concurrency": m["concurrency"],
                "input_tokens": m["input_tokens"], "output_tokens": m["output_tokens"],
                "throughput_tps": m["throughput_tps"],
                "ttft_p50": m["ttft_p50"], "ttft_p99": m["ttft_p99"],
                "e2e_p99": m["e2e_p99"], "success_rate": m["success_rate"],
            })
    rows.sort(key=lambda r: (r["concurrency"], r["engine"]))

    lines = ["# 同机三引擎压测对比（Qwen3-0.6B fp16, M4 Pro）", ""]
    lines.append("| engine | conc | in | out | 吞吐 t/s | TTFT p50 | TTFT p99 | E2E p99 | 成功率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['engine']} | {r['concurrency']} | {r['input_tokens']} | {r['output_tokens']} "
            f"| {r['throughput_tps']:.1f} | {r['ttft_p50']:.3f} | {r['ttft_p99']:.3f} "
            f"| {r['e2e_p99']:.2f} | {r['success_rate']:.0%} |"
        )
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


def main() -> int:
    ap = argparse.ArgumentParser(description="同机三引擎压测编排")
    ap.add_argument("--matrix", choices=list(MATRICES), default="smoke")
    ap.add_argument("--engines", default="hf,sglang,vllm", help="逗号分隔引擎列表")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip() in ENGINES]
    out_dir = ROOT / "runs/bench" / f"bench-mac-three-engine-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = MATRICES[args.matrix]
    print(f"矩阵: {matrix} | 引擎: {engines} | 输出: {out_dir}")

    results = {}
    for engine in engines:
        results[engine] = run_one(engine, out_dir, matrix)

    summarize(out_dir)
    failed = [e for e, ok in results.items() if not ok]
    if failed:
        print(f"✗ 失败引擎: {failed}（日志见 {out_dir}/*.log）")
        return 1
    print(f"✓ 全部完成: {out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
