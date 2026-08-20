"""Deployment artifact validation (no Docker engine required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_validate_deploy_artifacts():
    script = ROOT / "scripts/validate_deploy.py"
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT, timeout=60
    )
    assert proc.returncode == 0, f"validation failed:\n{proc.stdout}\n{proc.stderr}"
    assert "全部部署工件验证通过" in proc.stdout
