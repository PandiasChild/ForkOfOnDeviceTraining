"""Re-split cached raw UCI HAR data into T=5 sequential domains by subject.

No downloads: requires prepare_data.py to have populated data/raw first.
Protocol (pinned for the #326 post-merge study): subjects sorted ascending,
contiguous chunks of sizes 5,4,4,4,4; per-domain 80/20 train/eval split with
numpy default_rng(42).

Output dtypes mirror prepare_data.py's on-disk convention: x float32
[N,9,128], y int32 class indices 0..5 (labels are 1-based in the raw UCI
files; both scripts subtract 1 the same way).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "UCI HAR Dataset" / "train"
OUT = HERE / "data" / "domains"
CHUNKS = [5, 4, 4, 4, 4]
SEED = 42

# Order must mirror prepare_data.py's CHANNELS exactly -- the demo's model
# consumes the same [N,9,128] channel layout as train_x.npy.
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def main() -> int:
    if not RAW.exists():
        print(f"raw data missing at {RAW} — run prepare_data.py first", file=sys.stderr)
        return 1
    subjects = np.loadtxt(RAW / "subject_train.txt", dtype=np.int64)
    labels = np.loadtxt(RAW / "y_train.txt", dtype=np.int32) - 1  # 1..6 -> 0..5
    channels = [
        np.loadtxt(RAW / "Inertial Signals" / f"{s}_train.txt", dtype=np.float32)
        for s in SIGNALS
    ]
    x = np.stack(channels, axis=1)  # [N, 9, 128]
    assert x.shape[1:] == (9, 128), x.shape
    assert len(subjects) == len(labels) == len(x)

    unique_subjects = np.sort(np.unique(subjects))
    assert len(unique_subjects) == sum(CHUNKS), (
        f"expected {sum(CHUNKS)} train subjects, got {len(unique_subjects)}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta = {"seed": SEED, "domains": []}
    start = 0
    for t, size in enumerate(CHUNKS):
        domain_subjects = unique_subjects[start : start + size]
        start += size
        mask = np.isin(subjects, domain_subjects)
        dx, dy = x[mask], labels[mask]
        order = rng.permutation(len(dx))
        cut = int(0.8 * len(dx))
        tr, ev = order[:cut], order[cut:]
        np.save(OUT / f"domain{t}_train_x.npy", dx[tr])
        np.save(OUT / f"domain{t}_train_y.npy", dy[tr])
        np.save(OUT / f"domain{t}_eval_x.npy", dx[ev])
        np.save(OUT / f"domain{t}_eval_y.npy", dy[ev])
        meta["domains"].append(
            {
                "subjects": domain_subjects.tolist(),
                "train": int(len(tr)),
                "eval": int(len(ev)),
            }
        )
        print(f"domain {t}: subjects {domain_subjects.tolist()} train={len(tr)} eval={len(ev)}")
    (OUT / "domains_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
