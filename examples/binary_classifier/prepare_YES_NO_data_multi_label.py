"""
Prepare YESNO dataset


Output:
  train_x.npy [N_train, 1, 56000]
  train_y.npy [N_train, 8]  //multi label problem!!!
  val_x.npy
  val_y.npy
  test_x.npy
  test_y.npy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from prepare_audio_data import DATA_DIR, RAW_DIR, _download_if_missing, _split, read_wav

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.seeds import SHUFFLE_SEED  # noqa: E402
TARGET_LEN = 56000

ZIP_URL = "https://www.openslr.org/resources/1/waves_yesno.tar.gz"
ARCHIVE_PATH = RAW_DIR / "yesno.tar.gz"
EXTRACTED_DIR = RAW_DIR / "waves_yesno"
NPY_DIRECTORY = DATA_DIR / "YES_NO_multi_label"

def _load_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    wav_files = sorted(EXTRACTED_DIR.rglob("*.wav"))

    for f in wav_files:
        sr, audio = read_wav(f)
        assert sr == 8000, f"Unexpected sample rate: {sr}"

        audio = audio.astype(np.float32)

        if len(audio) < TARGET_LEN:
            audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
        else:
            audio = audio[:TARGET_LEN]

        X.append(audio[None, :])

        label = list(map(int, f.stem.split("_")))
        assert len(label) == 8
        assert all(v in (0, 1) for v in label)

        y.append(label)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

def main() -> None:
    _download_if_missing(ARCHIVE_PATH, EXTRACTED_DIR, ZIP_URL)
    NPY_DIRECTORY.mkdir(parents=True, exist_ok=True)

    X, y = _load_dataset()

    train_x, train_y, val_x, val_y, test_x, test_y = _split(X, y)

    np.save(NPY_DIRECTORY / "train_x.npy", train_x)
    np.save(NPY_DIRECTORY / "train_y.npy", train_y)

    np.save(NPY_DIRECTORY / "val_x.npy", val_x)
    np.save(NPY_DIRECTORY / "val_y.npy", val_y)

    np.save(NPY_DIRECTORY / "test_x.npy", test_x)
    np.save(NPY_DIRECTORY / "test_y.npy", test_y)

    print(f"train: {train_x.shape}, val: {val_x.shape}, test: {test_x.shape}")


if __name__ == "__main__":
    main()