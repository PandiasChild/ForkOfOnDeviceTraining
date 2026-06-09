from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import wave

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.seeds import SHUFFLE_SEED  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "raw"

def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        audio = wf.readframes(wf.getnframes())
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
    return sr, audio

# TODO: does work for yes no but not for speech commands. speech commands have no seperate folder
def _download_if_missing(archive_path: Path, extracted_dir: Path, zip_url: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"downloading {zip_url}")
        urllib.request.urlretrieve(zip_url, archive_path)

    if not extracted_dir.exists():
        print("extracting archive...")
        import tarfile
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(RAW_DIR, filter="data")


def _split(X: np.ndarray, y: np.ndarray):
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(X))

    n_val = len(X) // 10
    n_test = len(X) // 10

    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )

