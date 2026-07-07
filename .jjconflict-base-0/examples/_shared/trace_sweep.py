"""Multi-batch aggregator for the C-vs-PyTorch divergence diagnosis.

Runs N non-overlapping controlled steps, collects compare_pairs output for each
batch, and aggregates the per-probe error statistics so that robust divergence
(consistently large across batches) is distinguished from accumulation noise
(varies batch to batch).

CLI:
    uv run examples/_shared/trace_sweep.py [options]

Options:
    --example     example name (default: kws_raw)
    --batches     number of non-overlapping batches (default: 10)
    --batch       samples per batch B (default: 32)
    --act-samples activation-dump samples per batch (default: 1)
    --classes     number of output classes (default: 6)
    --start0      sample-start for batch 0 (default: 0)
    --stride      step between batch start indices (default: B)
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import trace_compare  # noqa: E402


def run_c(example: str, start: int, batch: int, act: int, classes: int) -> str:
    binary = ROOT / "build" / "examples" / "examples" / example / f"trace_c_{example}"
    if not binary.exists():
        raise FileNotFoundError(
            f"C harness not found: {binary}\n"
            "Build it first: cmake --preset examples && "
            f"cmake --build --preset examples --target trace_c_{example}"
        )
    env = os.environ.copy()
    env["KWS_CLASSES"] = str(classes)
    result = subprocess.run(
        [str(binary), "--sample-start", str(start), "--batch", str(batch),
         "--act-samples", str(act)],
        cwd=ROOT, capture_output=True, text=True, env=env, check=True,
    )
    return result.stdout.strip()


def run_pt(example: str, start: int, batch: int, act: int, classes: int) -> str:
    result = subprocess.run(
        ["uv", "run", f"examples/{example}/trace_pytorch.py",
         "--sample-start", str(start), "--batch", str(batch),
         "--act-samples", str(act), "--classes", str(classes)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def extract_loss(text: str, key: str = "mean_loss=") -> float | None:
    """Parse a 'mean_loss=<float>' token from a whitespace-separated output line."""
    for token in text.split():
        if token.startswith(key):
            try:
                return float(token[len(key):])
            except ValueError:
                pass
    return None


def row_sort_key(item: tuple) -> tuple:
    """Sort aggregate rows by tier, then network depth, then sample index, then phase."""
    (probe, phase), entry = item
    return (entry["tier"] if entry["tier"] is not None else 99,
            trace_compare.DEPTH.get(probe, 99),
            trace_compare.sample_of(phase),
            phase)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", default="kws_raw")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--act-samples", type=int, default=1)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--start0", type=int, default=0)
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args()

    B = args.batch
    stride = args.stride if args.stride is not None else B

    c_dump = ROOT / "examples" / args.example / "dump_c" / "step000"
    pt_dump = ROOT / "examples" / args.example / "dump_pt" / "step000"

    # --- Per-batch loop ---
    batch_results: list[tuple[float | None, float | None, list[dict]]] = []
    for i in range(args.batches):
        start = args.start0 + i * stride
        print(f"\n--- batch {i:2d}  sample_start={start} ---", flush=True)
        shutil.rmtree(c_dump, ignore_errors=True)
        shutil.rmtree(pt_dump, ignore_errors=True)
        try:
            c_out = run_c(args.example, start, B, args.act_samples, args.classes)
        except subprocess.CalledProcessError as exc:
            print(f"  C harness FAILED (exit {exc.returncode}):\n{exc.stderr}", file=sys.stderr)
            raise
        try:
            pt_out = run_pt(args.example, start, B, args.act_samples, args.classes)
        except subprocess.CalledProcessError as exc:
            print(f"  PyTorch script FAILED (exit {exc.returncode}):\n{exc.stderr}",
                  file=sys.stderr)
            raise
        print(f"  C:  {c_out}")
        print(f"  PT: {pt_out}", flush=True)
        c_loss = extract_loss(c_out)
        pt_loss = extract_loss(pt_out)
        pairs = trace_compare.compare_pairs(c_dump, pt_dump)
        batch_results.append((c_loss, pt_loss, pairs))

    # --- Aggregate per (probe, phase) across all batches ---
    accum: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"max_abs_list": [], "max_rel_list": [], "tier": None}
    )
    for _, _, pairs in batch_results:
        for d in pairs:
            key = (d["probe"], d["phase"])
            entry = accum[key]
            entry["max_abs_list"].append(d["max_abs"])
            entry["max_rel_list"].append(d["max_rel"])
            if entry["tier"] is None:
                entry["tier"] = d["tier"]

    # --- Header: loss sanity check ---
    print("\n" + "=" * 76)
    print("LOSS SANITY (C vs PyTorch mean_loss per batch):")
    for i, (cl, pl, _) in enumerate(batch_results):
        c_str = f"{cl:.6f}" if cl is not None else "N/A"
        p_str = f"{pl:.6f}" if pl is not None else "N/A"
        delta = ""
        if cl is not None and pl is not None:
            delta = f"  |diff|={abs(cl - pl):.2e}"
        print(f"  batch {i:2d}: C={c_str}  PT={p_str}{delta}")

    # --- Full aggregate table ---
    rows = sorted(accum.items(), key=row_sort_key)
    print("\nAGGREGATE TABLE (sorted by tier then network depth):")
    hdr = f"{'probe':12}{'phase':30}{'mean(maxabs)':>12}{'max_abs':>12}{'mean_rel':>12}{'n':>4}"
    print(hdr)
    print("-" * len(hdr))
    for (probe, phase), entry in rows:
        abs_list = entry["max_abs_list"]
        rel_list = entry["max_rel_list"]
        n = len(abs_list)
        mean_abs = float(np.mean(abs_list))
        max_abs = float(np.max(abs_list))
        mean_rel = float(np.mean(rel_list))
        print(f"{probe:12}{phase:30}{mean_abs:12.2e}{max_abs:12.2e}{mean_rel:12.2e}{n:4d}")

    # --- Focused summary: param-grad tiers only, sorted by mean_abs desc ---
    print("\nFOCUSED SUMMARY — param-grad mean_abs across batches (descending):")
    grad_rows = [
        ((probe, phase), entry)
        for (probe, phase), entry in accum.items()
        if phase.startswith("grad_raw") or phase.startswith("grad_scaled")
    ]
    grad_rows.sort(key=lambda kv: -float(np.mean(kv[1]["max_abs_list"])))
    hdr2 = f"{'probe':12}{'phase':30}{'mean(maxabs)':>12}{'max_abs':>12}{'n':>4}"
    print(hdr2)
    print("-" * len(hdr2))
    for (probe, phase), entry in grad_rows:
        abs_list = entry["max_abs_list"]
        n = len(abs_list)
        mean_abs = float(np.mean(abs_list))
        max_abs = float(np.max(abs_list))
        print(f"{probe:12}{phase:30}{mean_abs:12.2e}{max_abs:12.2e}{n:4d}")


if __name__ == "__main__":
    main()
