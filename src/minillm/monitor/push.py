"""Monitoring helpers: push metrics to Prometheus Pushgateway.

Benchmark 结果与训练指标经 Pushgateway 入库，Grafana 即可展示跨引擎/跨实验对比。
（批任务无法被 Prometheus 常规 scrape，Pushgateway 是标准解法。）
"""

from __future__ import annotations

import logging
import os
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

log = logging.getLogger(__name__)


def gateway_url_from_env() -> str | None:
    """Resolve push gateway URL from env ($MINILLM_PUSHGATEWAY or $PUSHGATEWAY_URL)."""
    return os.environ.get("MINILLM_PUSHGATEWAY") or os.environ.get("PUSHGATEWAY_URL")


def build_registry(series: list[dict[str, Any]]) -> CollectorRegistry:
    """Build a Prometheus registry from [{name, value, labels{...}}] series."""
    registry = CollectorRegistry()
    for s in series:
        name = s["name"]
        labels = dict(s.get("labels", {}))
        gauge = Gauge(
            name,
            s.get("help", name),
            list(labels),
            registry=registry,
        )
        gauge.labels(**labels).set(float(s["value"]))
    return registry


def push_series(
    series: list[dict[str, Any]],
    *,
    job: str,
    gateway: str | None = None,
    grouping_key: dict[str, str] | None = None,
) -> bool:
    """Push a metric series batch to the gateway. Returns True on success.

    ``grouping_key`` 区分同一 job 下的多个运行（如按 experiment+timestamp），
    避免旧实验覆盖新实验。
    """
    gateway = gateway or gateway_url_from_env()
    if not gateway:
        log.info("push gateway 未配置（MINILLM_PUSHGATEWAY）— 跳过指标入库")
        return False
    if not series:
        log.warning("无指标可推送")
        return False
    registry = build_registry(series)
    try:
        push_to_gateway(
            gateway,
            job=job,
            registry=registry,
            grouping_key=grouping_key,
        )
        log.info("pushed %d series to pushgateway %s (job=%s)", len(series), gateway, job)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("pushgateway 推送失败（不影响主流程）: %s", e)
        return False
