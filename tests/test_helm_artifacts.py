"""Helm chart validation (requires helm binary; skips when absent)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CHART = ROOT / "deploy/helm/minillm"
HELM = ROOT / ".tools/helm"


def _helm() -> str | None:
    if HELM.exists():
        return str(HELM)
    return shutil.which("helm")


@pytest.fixture(scope="module")
def helm_bin() -> str | None:
    return _helm()


def test_helm_lint(helm_bin):
    if not helm_bin:
        pytest.skip("helm binary not found (run `make helm-tool` or install helm)")
    proc = subprocess.run(
        [helm_bin, "lint", str(CHART)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"helm lint failed:\n{proc.stdout}\n{proc.stderr}"
    assert "0 chart(s) failed" in proc.stdout


@pytest.mark.parametrize(
    "extra_args,expect_kinds",
    [
        ([], ["Deployment", "Service", "HorizontalPodAutoscaler", "ClusterQueue", "LocalQueue"]),
        (
            ["--set", "training.enabled=true", "--set", "bench.enabled=true", "--set", "sglang.enabled=true"],
            ["Job", "PersistentVolumeClaim", "Service"],
        ),
    ],
)
def test_helm_template_renders(helm_bin, extra_args, expect_kinds):
    if not helm_bin:
        pytest.skip("helm binary not found (run `make helm-tool` or install helm)")
    proc = subprocess.run(
        [helm_bin, "template", "demo", str(CHART), *extra_args],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"helm template failed:\n{proc.stderr}"
    docs = [d for d in yaml.safe_load_all(proc.stdout) if d]
    kinds = [d["kind"] for d in docs]
    for kind in expect_kinds:
        assert kind in kinds, f"rendered manifests missing kind {kind} (got {kinds})"
    # 关键资源名称
    names = {d["kind"]: d["metadata"]["name"] for d in docs}
    assert names["ClusterQueue"] == "minillm-gpu"
    assert "LocalQueue" in kinds


def test_helm_values_referenced(helm_bin):
    """核心 values 均被模板消费（防配置漂移）。"""
    if not helm_bin:
        pytest.skip("helm binary not found")
    template_dir = CHART / "templates"
    templates = "".join(p.read_text(encoding="utf-8") for p in template_dir.glob("*.yaml"))
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))

    # 每个 vllm 性能旋钮都出现在模板中
    for key in ["gpuMemoryUtilization", "maxModelLen", "maxNumSeqs", "maxNumBatchedTokens", "enablePrefixCaching"]:
        assert f".Values.vllm.{key}" in templates, f"values.vllm.{key} 未被模板引用"
    assert ".Values.kueue.localQueues" in templates
    assert ".Values.vllm.hpa.metricName" in templates
    assert values["vllm"]["enablePrefixCaching"] is True
