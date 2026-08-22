#!/usr/bin/env python3
"""max_len 边界探测：自动找出引擎能完整处理的单请求最大 prompt 长度（截断点）。

方法
----
- 用模型自带的 tokenizer 精确构造 N-token prompt（token 计数可控）
- 二分搜索：每个 N 发一次请求（max_tokens=1），观察引擎行为
    ok        → 200 且 usage.prompt_tokens == N（完整接收）
    truncated → 200 但 prompt_tokens < N（服务静默截断，输入被丢弃）
    error     → 400/异常（超出服务 max_model_len，如 vLLM 的 "maximum context length"）
- 边界 = 最后一个 ok 的长度；另报告首个 error 的长度与引擎错误信息

用法
----
    python scripts/test_max_len.py --base-url http://127.0.0.1:8010 \
        --model data/models/Qwen3-0.6B --tokenizer data/models/Qwen3-0.6B
    # --max-len 覆盖服务上报值；--verify 在边界附近做 3 次确认
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

import httpx

# ---------------------------------------------------------------- 二分核心（纯逻辑，可单测）


def find_boundary(predicate: Callable[[int], bool], lo: int, hi: int) -> int | None:
    """二分找满足 predicate 的最大值（[lo, hi] 内）。predicate 需单调：小→True，大→False。

    返回最后一个 True 的值；若 lo 都不满足返回 None。
    """
    if not predicate(lo):
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------- 探测器


class MaxLenProbe:
    def __init__(self, base_url: str, model: str, tokenizer_dir: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        # 探测用重复 token（避免真实语义长度差异）。
        # 注意：BPE 词表含 " token"（带前导空格）类合并 token，decode→re-tokenize 会压缩，
        # 因此"构造恰好 N"不可靠 —— 改为：本地精确计数 + 以服务端回显为准（见 probe）。
        seed = self.tok("token ", add_special_tokens=False).input_ids
        if not seed:
            seed = [self.tok.bos_token_id or 1]
        self.seed = seed

    def server_max_len(self) -> int:
        resp = httpx.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        for m in resp.json()["data"]:
            if m.get("max_model_len"):
                return int(m["max_model_len"])
        return 0

    def build_text(self, approx_n: int) -> str:
        """构造本地 tokenize 计数 ≈ approx_n 的 prompt 文本。

        BPE 合并导致"ids→文本→ids"约损失一半，因此先超量构造（本地计数达标）
        再从 token 空间截断到 approx_n。
        """
        k = 1
        while True:
            text = self.tok.decode(
                self.seed * k, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if len(self.tok(text, add_special_tokens=False).input_ids) >= approx_n:
                break
            k *= 2
        ids = self.tok(text, add_special_tokens=False).input_ids[:approx_n]
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def probe(self, approx_n: int) -> tuple[str, dict]:
        """返回 (状态, 详情)。状态判定以**本地精确计数**为基准：
        - ok        : 200 且服务端回显 prompt_tokens == 本地计数（完整接收）
        - truncated : 200 但回显 < 本地计数（服务端静默截断）
        - error     : 非 200 / 异常
        """
        prompt = self.build_text(approx_n)
        sent = len(self.tok(prompt, add_special_tokens=False).input_ids)  # 本地精确计数
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 1, "temperature": 0.0},
                timeout=self.timeout,
            )
            body = resp.json()
        except Exception as e:  # noqa: BLE001
            return "error", {"approx_n": approx_n, "error": str(e)[:120]}
        if resp.status_code != 200:
            return "error", {"approx_n": approx_n, "sent": sent,
                             "status": resp.status_code,
                             "detail": str(body.get("error") or body)[:160]}
        got = usage_prompt_tokens(body)
        if got < sent:
            return "truncated", {"approx_n": approx_n, "sent": sent,
                                 "prompt_tokens": got, "usage": body.get("usage")}
        return "ok", {"approx_n": approx_n, "sent": sent,
                      "prompt_tokens": got, "usage": body.get("usage")}

    def run(self, max_len: int, steps: int = 24, verify: bool = False) -> dict:
        def is_ok(n: int) -> bool:
            return self.probe(n)[0] == "ok"

        # 上界略超服务 max：验证超过后的截断/报错行为
        lo, hi = 1, max(1, max_len + 512)
        boundary_approx = find_boundary(is_ok, lo, hi)

        # 边界处以服务端回显为准，重测一次拿精确值
        boundary = None
        if boundary_approx:
            st, detail = self.probe(boundary_approx)
            boundary = detail.get("sent") or detail.get("prompt_tokens")

        # 在边界附近补测，确认单调性与截断行为
        samples: dict[int, tuple[str, dict]] = {}
        for n in sorted({1, hi, hi // 2, boundary_approx or 0, (boundary_approx or 0) + 1,
                         (boundary_approx or 0) + 64, hi - 1, hi}):
            if 1 <= n <= hi and n not in samples:
                samples[n] = self.probe(n)

        # verify: 边界处多次确认
        confirm = []
        if verify and boundary:
            for approx in (boundary_approx, min(boundary_approx + 1, hi)):
                st, detail = self.probe(approx)
                confirm.append({"len": detail.get("sent", approx), "status": st, **detail})

        return {
            "engine": self.base_url,
            "model": self.model,
            "server_max_model_len": max_len,
            "max_full_prompt_tokens": boundary,
            "samples": {str(k): {"status": v[0], **v[1]} for k, v in sorted(samples.items())},
            "confirm": confirm,
        }


def usage_prompt_tokens(body: dict) -> int:
    """从 OpenAI 响应中提取 prompt_tokens（兼容 usage 缺失等异常）。"""
    usage = body.get("usage") or {}
    got = usage.get("prompt_tokens", 0)
    return int(got or 0)


def report(result: dict) -> str:
    lines = [
        "=" * 60,
        f"引擎      : {result['engine']}",
        f"模型      : {result['model']}",
        f"服务上报  : max_model_len = {result['server_max_model_len']}",
        f"实测边界  : 最长完整接收 prompt = {result['max_full_prompt_tokens']} tokens",
        "=" * 60,
        f"{'prompt_len':>10}  {'状态':<10} 详情",
        "-" * 60,
    ]
    for n, s in result["samples"].items():
        detail = s.get("detail") or f"sent={s.get('sent','-')} recv={s.get('prompt_tokens','-')}"
        lines.append(f"{n:>10}  {s['status']:<10} {detail}")
    if result["confirm"]:
        lines.append("-" * 60)
        for c in result["confirm"]:
            lines.append(f"确认 {c['len']:>8} → {c['status']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="max_len 截断点探测")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default="data/models/Qwen3-0.6B")
    ap.add_argument("--tokenizer", default="data/models/Qwen3-0.6B")
    ap.add_argument("--max-len", type=int, default=0, help="覆盖服务上报的 max_model_len")
    ap.add_argument("--verify", action="store_true", help="边界处多次确认")
    args = ap.parse_args()

    probe = MaxLenProbe(args.base_url, args.model, args.tokenizer)
    max_len = args.max_len or probe.server_max_len()
    if not max_len:
        print("无法从 /v1/models 获取 max_model_len，请用 --max-len 指定", file=sys.stderr)
        return 2

    print(f"探测中（服务 max_model_len={max_len}，二分 ~{max_len.bit_length()} 步）...")
    result = probe.run(max_len, verify=args.verify)
    print(report(result))
    with open("runs/max_len_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n报告已存: runs/max_len_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
