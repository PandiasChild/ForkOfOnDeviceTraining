"""Test examples/_shared/parity.py."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest

from examples._shared.parity import ParityCheck, ParityResult, run_parity_checks


def test_absolute_check_within_tolerance_passes():
    check = ParityCheck("test_acc", abs_tol=0.025)
    result = check.evaluate(pt_value=0.91, c_value=0.89)
    assert result.passed is True
    assert abs(result.diff - 0.02) < 1e-9


def test_absolute_check_outside_tolerance_fails():
    check = ParityCheck("test_acc", abs_tol=0.025)
    result = check.evaluate(pt_value=0.91, c_value=0.85)
    assert result.passed is False


def test_relative_check_within_tolerance_passes():
    check = ParityCheck("test_mse", rel_tol=0.10)
    result = check.evaluate(pt_value=0.020, c_value=0.022)  # 10% drift exactly
    assert result.passed is True


def test_relative_check_outside_tolerance_fails():
    check = ParityCheck("test_mse", rel_tol=0.10)
    result = check.evaluate(pt_value=0.020, c_value=0.024)  # 20% drift
    assert result.passed is False


def test_run_parity_checks_returns_overall_status_and_per_check_results():
    checks = [
        ParityCheck("a", abs_tol=0.05),
        ParityCheck("b", rel_tol=0.10),
    ]
    pt = {"a": 1.0, "b": 1.0}
    c =  {"a": 1.02, "b": 1.20}  # b is 20% off → fails
    overall_pass, results = run_parity_checks(checks, pt, c)
    assert overall_pass is False
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False


def test_parity_check_requires_exactly_one_tolerance_type():
    with pytest.raises(ValueError):
        ParityCheck("x")  # neither
    with pytest.raises(ValueError):
        ParityCheck("x", abs_tol=0.1, rel_tol=0.1)  # both
