"""#279 dead-zone A/B analysis: aggregate a det-vs-SR sweep and render the verdict.

Reads the per-run RunLog JSON emitted by run_matrix.py (``<config>_seed<N>.json``)
and compares, per SYM width, the two write-back rounding arms against the FLOAT32
reference:

    float          -- FLOAT32 params (the twin / upper bound)
    sym<N>det      -- SYM@N params, deterministic HALF_AWAY write-back (dead-zone)
    sym<N>         -- SYM@N params, seeded SR_HALF_AWAY write-back (the escape)

The decision this feeds (#279): does seeded SR close the convergence gap that
HALF_AWAY plateaus on, in aggregate over >=10 seeds? Reports mean +/- std of the
final test accuracy / loss and the final train loss, plus the gap each SYM arm
leaves to the FLOAT32 twin and the fraction of that gap SR recovers.

    uv run examples/har_classifier/analyze_deadzone.py --logs examples/har_classifier/logs
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

_SEED_RE = re.compile(r"^(?P<config>.+)_seed(?P<seed>\d+)\.json$")


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return math.nan, math.nan
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def load_runs(logs_dir: Path) -> dict[str, list[dict]]:
    """config -> list of parsed run dicts (one per seed)."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(logs_dir.glob("*_seed*.json")):
        m = _SEED_RE.match(path.name)
        if not m:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  WARN: {path.name} is not valid JSON (run may have crashed) -- skipped")
            continue
        data["_seed"] = int(m.group("seed"))
        runs[m.group("config")].append(data)
    return runs


def summarize(run_list: list[dict]) -> dict:
    """Aggregate final metrics + final train loss across seeds for one config."""
    accs, test_losses, train_losses = [], [], []
    for r in run_list:
        final = r.get("final", {})
        if final.get("test_acc") is not None:
            accs.append(float(final["test_acc"]))
        if final.get("test_loss") is not None:
            test_losses.append(float(final["test_loss"]))
        epochs = r.get("epochs", [])
        if epochs and epochs[-1].get("train_loss") is not None:
            train_losses.append(float(epochs[-1]["train_loss"]))
    return {
        "n": len(run_list),
        "acc": _mean_std(accs),
        "test_loss": _mean_std(test_losses),
        "train_loss": _mean_std(train_losses),
        "accs": accs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--logs", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--ref", default="float", help="reference config (default: float)")
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    runs = load_runs(logs_dir)
    if not runs:
        raise SystemExit(f"no *_seed*.json runs found in {logs_dir}")

    summ = {cfg: summarize(rl) for cfg, rl in runs.items()}

    # ---- per-config table -------------------------------------------------
    print(f"\n#279 dead-zone A/B  ({logs_dir})\n")
    hdr = f"{'config':10s} {'n':>3s}  {'test_acc':>16s}  {'test_loss':>16s}  {'train_loss':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for cfg in sorted(summ):
        s = summ[cfg]
        am, asd = s["acc"]
        tlm, tlsd = s["test_loss"]
        trm, trsd = s["train_loss"]
        print(
            f"{cfg:10s} {s['n']:>3d}  {am:6.4f} +/- {asd:6.4f}  "
            f"{tlm:6.4f} +/- {tlsd:6.4f}  {trm:6.4f} +/- {trsd:6.4f}"
        )

    # ---- verdict: pair each sym<N>det with sym<N> vs the reference ---------
    ref = summ.get(args.ref)
    if ref is None or math.isnan(ref["acc"][0]):
        print(f"\n(no reference '{args.ref}' runs -- skipping gap verdict)")
        return
    ref_acc = ref["acc"][0]

    widths = sorted(
        {m.group(1) for cfg in summ if (m := re.match(r"sym(\d+)$", cfg))},
        key=int,
    )
    print(f"\nVerdict vs {args.ref} (test_acc={ref_acc:.4f}):\n")
    for w in widths:
        sr = summ.get(f"sym{w}")
        det = summ.get(f"sym{w}det")
        if sr is None or det is None:
            continue
        sr_acc, det_acc = sr["acc"][0], det["acc"][0]
        gap_det = ref_acc - det_acc
        gap_sr = ref_acc - sr_acc
        recovered = (gap_det - gap_sr) / gap_det if abs(gap_det) > 1e-9 else math.nan
        print(f"  SYM@{w}:")
        print(f"    HALF_AWAY (det): acc={det_acc:.4f}  gap-to-float={gap_det:+.4f}")
        print(f"    SR_HALF_AWAY   : acc={sr_acc:.4f}  gap-to-float={gap_sr:+.4f}")
        print(f"    -> SR recovers {recovered*100:5.1f}% of the HALF_AWAY dead-zone gap")


if __name__ == "__main__":
    main()
