"""Generate a tiny, deterministic rank-3 weight fixture for the npyLoadFlat test.

Produces proxy_w.npy of shape (3, 2, 4), dtype float32, with flat value i at
flat index i (i.e. np.arange(24)). This mirrors a Conv1d weight layout
[out, in, k]; testNpyLoadFlat_LoadsFullMultiDimTensor in UnitTestDataLoader.c
asserts every element loads correctly. The elements at flat index >= 8 (beyond
the first dim0 row of size 2*4) are exactly what the issue #177 loader bug
corrupted, so they make the regression sharp.

Run with:  uv run --with numpy test/unit/data_loader/ProxyWeightGenerator.py
"""

import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS = np.arange(24, dtype=np.float32).reshape(3, 2, 4)


if __name__ == "__main__":
    np.save(os.path.join(BASE_DIR, "proxy_w.npy"), WEIGHTS)
