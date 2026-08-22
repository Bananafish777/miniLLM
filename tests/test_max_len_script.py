"""max_len 探测脚本：二分逻辑单元测试（无需真实引擎）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("minillm_test_max_len", ROOT / "scripts/test_max_len.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
find_boundary = _mod.find_boundary


def test_find_boundary_basic():
    assert find_boundary(lambda n: n <= 5000, 1, 8192) == 5000


def test_find_boundary_at_hi():
    assert find_boundary(lambda n: True, 1, 8192) == 8192


def test_find_boundary_none_when_lo_false():
    assert find_boundary(lambda n: False, 1, 8192) is None


def test_find_boundary_small_range():
    assert find_boundary(lambda n: n < 100, 1, 200) == 99


def test_find_boundary_call_count_logarithmic():
    calls = []

    def pred(n: int) -> bool:
        calls.append(n)
        return n <= 40960

    find_boundary(pred, 1, 40960)
    assert len(calls) <= 20  # log2(40960) ≈ 16
