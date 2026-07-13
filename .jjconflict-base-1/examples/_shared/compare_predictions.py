"""Compare C-side predictions/reconstructions against PyTorch reference outputs.

Used by the bit-parity CI step. Exits 0 on match, 1 on mismatch.

Usage:
    uv run examples/_shared/compare_predictions.py \\
        --pytorch <path-to-pytorch.npy> \\
        --c <path-to-c.npy> \\
        --dtype {int32,float32} \\
        [--rtol 1e-4] [--atol 1e-5]
"""

import argparse
import sys
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", required=True, help="PyTorch reference .npy")
    parser.add_argument("--c", required=True, help="C-side .npy")
    parser.add_argument("--dtype", required=True, choices=["int32", "float32"])
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    py = np.load(args.pytorch)
    c = np.load(args.c)

    if py.shape != c.shape:
        print(f"FAIL: shape mismatch — pytorch={py.shape}, c={c.shape}", file=sys.stderr)
        return 1

    if args.dtype == "int32":
        if not np.array_equal(py, c):
            mismatches = np.flatnonzero(py != c)
            print(f"FAIL: int32 mismatch at {mismatches.size}/{py.size} positions",
                  file=sys.stderr)
            for idx in mismatches[:5]:
                print(f"  idx={idx}: pytorch={py.flat[idx]}, c={c.flat[idx]}", file=sys.stderr)
            return 1
        print(f"PASS: int32 arrays bit-identical ({py.size} elements)")
        return 0

    # float32
    if not np.allclose(py, c, rtol=args.rtol, atol=args.atol):
        diffs = np.abs(py - c)
        max_diff = diffs.max()
        rel_diffs = diffs / (np.abs(py) + args.atol)
        max_rel = rel_diffs.max()
        print(f"FAIL: float32 mismatch — max_abs={max_diff:.6e}, "
              f"max_rel={max_rel:.6e}, rtol={args.rtol}, atol={args.atol}", file=sys.stderr)
        worst = np.argmax(diffs)
        print(f"  worst idx={worst}: pytorch={py.flat[worst]:.6e}, c={c.flat[worst]:.6e}",
              file=sys.stderr)
        return 1
    print(f"PASS: float32 arrays close (rtol={args.rtol}, atol={args.atol}, "
          f"{py.size} elements)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
