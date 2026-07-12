"""ACC/BWT from train_c_continual JSON logs (GEM definitions, 0-indexed).

Single log:  uv run compare_continual.py logs/run.json
Sweep dir:   uv run compare_continual.py logs/sweep/   (aggregates mean+std)
"""

import json
import sys
from pathlib import Path

import numpy as np


def acc_bwt(log: dict) -> tuple[float, float]:
    rows = log["accuracy_matrix"]
    t_final = len(rows) - 1
    r_final = rows[t_final]
    acc = float(np.mean(r_final))
    if t_final == 0:
        return acc, 0.0
    bwt = float(np.mean([r_final[j] - rows[j][j] for j in range(t_final)]))
    return acc, bwt


def main() -> int:
    target = Path(sys.argv[1])
    logs = sorted(target.glob("*.json")) if target.is_dir() else [target]
    results = []
    for path in logs:
        log = json.loads(path.read_text())
        acc, bwt = acc_bwt(log)
        replay = log.get("replay", {}).get("enabled", 0)
        results.append((path.name, replay, acc, bwt))
        print(f"{path.name}: replay={replay} ACC={acc:.4f} BWT={bwt:+.4f}")
    if len(results) > 1:
        for flag in (0, 1):
            sel = [(a, b) for _, r, a, b in results if r == flag]
            if sel:
                accs, bwts = zip(*sel)
                print(
                    f"replay={flag}: n={len(sel)} "
                    f"ACC={np.mean(accs):.4f}±{np.std(accs):.4f} "
                    f"BWT={np.mean(bwts):+.4f}±{np.std(bwts):.4f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
