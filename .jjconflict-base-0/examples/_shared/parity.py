"""Final-state parity tolerance helpers.

A ParityCheck is a per-metric tolerance. Each metric specifies *either*
abs_tol *or* rel_tol (never both, never neither). The compare.py for
each example builds a list of ParityChecks, evaluates them against
final values from the two RunLogs, and reports per-metric pass/fail
plus an overall pass flag.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParityResult:
    metric: str
    pt_value: float
    c_value: float
    diff: float          # absolute difference, |pt - c|
    rel_diff: float      # relative, |pt - c| / |pt|; 0 if pt is 0
    tolerance: float
    tolerance_type: str  # "abs" or "rel"
    passed: bool


@dataclass(frozen=True)
class ParityCheck:
    metric: str
    abs_tol: float | None = None
    rel_tol: float | None = None

    def __post_init__(self) -> None:
        if (self.abs_tol is None) == (self.rel_tol is None):
            raise ValueError(
                f"ParityCheck({self.metric!r}): specify exactly one of abs_tol, rel_tol"
            )

    def evaluate(self, pt_value: float, c_value: float) -> ParityResult:
        diff = abs(pt_value - c_value)
        rel_diff = diff / abs(pt_value) if pt_value != 0 else 0.0
        if self.abs_tol is not None:
            return ParityResult(
                metric=self.metric, pt_value=pt_value, c_value=c_value,
                diff=diff, rel_diff=rel_diff,
                tolerance=self.abs_tol, tolerance_type="abs",
                passed=diff <= self.abs_tol,
            )
        assert self.rel_tol is not None
        return ParityResult(
            metric=self.metric, pt_value=pt_value, c_value=c_value,
            diff=diff, rel_diff=rel_diff,
            tolerance=self.rel_tol, tolerance_type="rel",
            passed=rel_diff <= self.rel_tol,
        )


def run_parity_checks(
    checks: list[ParityCheck],
    pt_finals: dict[str, float],
    c_finals: dict[str, float],
) -> tuple[bool, list[ParityResult]]:
    results = [c.evaluate(pt_finals[c.metric], c_finals[c.metric]) for c in checks]
    return all(r.passed for r in results), results
