#!/usr/bin/env python3
"""静态验证 deploy/ 下的编排与监控工件（无需 Docker 引擎）。

检查项：
- compose 文件可解析、服务互相引用（depends_on）存在、GPU override 服务名一致
- Prometheus 配置/rules、Grafana provisioning/面板 JSON 可解析且路径存在
- Dockerfile 与构建上下文存在
- .env.example 存在

用法: python scripts/validate_deploy.py   （退出码 0 = 全部通过）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "deploy/compose/docker-compose.yml"
COMPOSE_GPU = ROOT / "deploy/compose/docker-compose.gpu.yml"
MONITORING = ROOT / "deploy/monitoring"

FAILURES: list[str] = []


def check(ok: bool, msg: str) -> None:
    if ok:
        print(f"  ok   {msg}")
    else:
        FAILURES.append(msg)
        print(f"  FAIL {msg}")


def main() -> int:
    print("== compose ==")
    base = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    check(isinstance(base, dict) and "services" in base, "docker-compose.yml 可解析且含 services")

    gpu = yaml.safe_load(COMPOSE_GPU.read_text(encoding="utf-8"))
    check(isinstance(gpu, dict) and "services" in gpu, "docker-compose.gpu.yml 可解析且含 services")

    services = base["services"]
    for svc, cfg in services.items():
        for dep in (cfg.get("depends_on") or []):
            check(dep in services, f"depends_on 引用存在: {svc} -> {dep}")
        ports = cfg.get("ports") or []
        for p in ports:
            check(str(p).count(":") >= 1, f"{svc} 端口映射格式: {p}")
        if cfg.get("build"):
            # compose 的相对路径基于 compose 文件所在目录解析
            ctx = (COMPOSE.parent / str(cfg["build"]["context"])).resolve()
            df = ctx / str(cfg["build"]["dockerfile"])
            check(ctx.exists(), f"{svc} build context 存在: {ctx}")
            check(df.exists(), f"{svc} Dockerfile 存在: {df}")

    for svc in gpu["services"]:
        check(svc in services, f"GPU override 服务存在于 base: {svc}")

    print("== monitoring ==")
    prom = yaml.safe_load((MONITORING / "prometheus/prometheus.yml").read_text(encoding="utf-8"))
    jobs = [j["job_name"] for j in prom.get("scrape_configs", [])]
    check(len(jobs) == len(set(jobs)), "scrape job 无重名")
    for target_svc in ["vllm", "sglang", "serve-hf", "dcgm-exporter", "pushgateway"]:
        check(
            any(target_svc in str(j.get("static_configs")) for j in prom.get("scrape_configs", [])),
            f"scrape 目标包含 {target_svc}",
        )
    check("pushgateway" in base["services"], "compose 含 pushgateway 服务")
    check("PUSHGATEWAY_URL" in yaml.dump(base["services"]["train"].get("environment", {})),
          "train 容器注入 PUSHGATEWAY_URL")
    rules = yaml.safe_load((MONITORING / "prometheus/rules/minillm-alerts.yml").read_text(encoding="utf-8"))
    n_rules = sum(len(g["rules"]) for g in rules.get("groups", []))
    check(n_rules >= 4, f"告警规则 >= 4 条（实际 {n_rules}）")

    ds = yaml.safe_load((MONITORING / "grafana/provisioning/datasources/datasources.yml").read_text(encoding="utf-8"))
    check(ds["datasources"][0]["url"] == "http://prometheus:9090", "Grafana datasource 指向 compose 内 Prometheus")

    dash_provider = yaml.safe_load((MONITORING / "grafana/provisioning/dashboards/dashboards.yml").read_text(encoding="utf-8"))
    check(dash_provider["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards", "面板挂载路径一致")

    dash = json.loads((MONITORING / "grafana/dashboards/minillm-overview.json").read_text(encoding="utf-8"))
    check(dash["title"] == "miniLLM 全栈监控", "面板 JSON 可解析")
    exprs = [t["expr"] for p in dash["panels"] for t in p["targets"]]
    check(all("DCGM_FI_DEV" in e or "vllm:" in e or "minillm_" in e for e in exprs), "面板表达式均为真实指标")

    bench_dash = json.loads((MONITORING / "grafana/dashboards/minillm-bench.json").read_text(encoding="utf-8"))
    bench_exprs = [t["expr"] for p in bench_dash["panels"] for t in p["targets"]]
    check(bench_dash["title"] == "miniLLM Benchmark 对比", "Benchmark 面板 JSON 可解析")
    check(all("minillm_bench_" in e or "minillm_train_" in e for e in bench_exprs), "Benchmark 面板引用已入库指标")

    check((ROOT / "deploy/compose/.env.example").exists(), ".env.example 存在")
    check((ROOT / "deploy/docker/Dockerfile.train").exists(), "Dockerfile.train 存在")
    check((ROOT / "deploy/docker/Dockerfile.serve").exists(), "Dockerfile.serve 存在")

    print()
    if FAILURES:
        print(f"验证失败: {len(FAILURES)} 项")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("全部部署工件验证通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
