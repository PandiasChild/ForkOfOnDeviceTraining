"""Localize the first tensor where C and PyTorch training diverge.

Pairs examples/<ex>/dump_c/stepNNN/<probe>.<phase>.npy against the dump_pt
counterpart (identical filenames on both sides), computes max-abs / max-rel error
per pair, prints a table ordered by tier then network depth, and flags the FIRST
probe whose error jumps orders of magnitude above the running per-tier floor
(relative-jump test, not a flat epsilon). The noise floor resets at each tier
boundary because tiers have independent magnitudes.

Self-test: `uv run examples/_shared/trace_compare.py --self-test`.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

# Network depth order (must equal probe_manifest.h / FWD_PROBES) — 17-layer model:
PROBES = ["pool0","conv1","ln1","relu1","pool1","conv2","ln2","relu2","pool2",
          "conv3","ln3","relu3","pool3","adaptpool","flatten","fc","softmax"]
DEPTH = {name: i for i, name in enumerate(PROBES)}
DEPTH["loss"] = len(PROBES)  # the loss-grad probe sits after the last layer
# Table tier order, by phase prefix:
TIERS = [("fwd", 0), ("lossgrad", 1), ("agrad", 2), ("grad_raw", 3),
         ("grad_scaled", 4), ("w_before", 5), ("w_after", 6)]
JUMP_FACTOR = 1e3  # error >1000x the running per-tier floor = first drift


def tier_of(phase: str) -> int:
    for prefix, rank in TIERS:
        if phase.startswith(prefix):
            return rank
    return len(TIERS)


def sample_of(phase: str) -> int:
    if ".s" in phase:
        try:
            return int(phase.rsplit(".s", 1)[1])
        except ValueError:
            return -1
    return -1


def sort_key(p: Path):
    probe, _, phase = p.name[:-4].partition(".")
    return (tier_of(phase), DEPTH.get(probe, 99), sample_of(phase), phase)


def errs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.shape != b.shape:
        return float("inf"), float("inf")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(b.astype(np.float64)), 1e-12)
    return float(diff.max()), float((diff / denom).max())


def compare_dir(c_dir: Path, pt_dir: Path) -> int:
    files = sorted(c_dir.glob("*.npy"), key=sort_key)
    if not files:
        print(f"no dumps in {c_dir}", file=sys.stderr)
        return 2
    floor, cur_tier, first_drift = 1e-6, None, None
    print(f"{'probe':12}{'phase':24}{'max_abs':>12}{'max_rel':>12}  status")
    for f in files:
        probe, _, phase = f.name[:-4].partition(".")
        tier = tier_of(phase)
        if tier != cur_tier:
            floor, cur_tier = 1e-6, tier  # reset the noise floor per tier
        pt = pt_dir / f.name
        if not pt.exists():
            print(f"{probe:12}{phase:24}{'':>12}{'':>12}  (no PyTorch counterpart)")
            continue
        ma, mr = errs(np.load(f), np.load(pt))
        drift = mr > floor * JUMP_FACTOR and first_drift is None
        status = "<= FIRST DRIFT" if drift else "ok"
        if drift:
            first_drift = (probe, phase, ma, mr)
        print(f"{probe:12}{phase:24}{ma:12.2e}{mr:12.2e}  {status}")
        if not drift and mr < 1.0:
            floor = max(floor, mr)  # raise the running per-tier floor
    if first_drift:
        print(f"\nFIRST DRIFT: {first_drift[0]}.{first_drift[1]} "
              f"(max_abs={first_drift[2]:.2e}, max_rel={first_drift[3]:.2e})")
    else:
        print("\nno drift above threshold - all tiers agree")
    return 0


def self_test() -> int:
    import tempfile
    rs = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as d:
        c, pt = Path(d) / "c", Path(d) / "pt"
        c.mkdir(); pt.mkdir()
        base = rs.randn(1, 16, 8).astype(np.float32)  # per-sample activation, [1,C,L]
        for nm in ("conv1.fwd.s000.npy", "conv1.fwd.s001.npy"):
            np.save(c / nm, base); np.save(pt / nm, base.copy())
        wbase = rs.randn(16, 1, 3).astype(np.float32)
        bad = wbase.copy(); bad[0, 0, 0] += 5.0
        np.save(c / "conv1.grad_raw.weight.npy", bad)
        np.save(pt / "conv1.grad_raw.weight.npy", wbase)
        rc = compare_dir(c, pt)
        assert rc == 0
    print("self-test OK")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="kws_raw")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    root = Path(__file__).resolve().parents[1] / args.example
    step = f"step{args.step:03d}"
    sys.exit(compare_dir(root / "dump_c" / step, root / "dump_pt" / step))


if __name__ == "__main__":
    main()
