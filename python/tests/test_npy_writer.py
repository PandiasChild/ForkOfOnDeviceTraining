"""Test examples/_shared/npy_writer.{h,c} by compiling a tiny C harness
that calls npyWriteFloat32 / npyWriteInt32 with known buffers, then
np.load()-ing the output and asserting it matches.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compile_writer_test_harness(harness_src: Path) -> str:
    """Compile harness_src against examples/_shared/npy_writer.c."""
    writer_c = REPO_ROOT / "examples" / "_shared" / "npy_writer.c"
    writer_h = REPO_ROOT / "examples" / "_shared"
    binary = tempfile.NamedTemporaryFile(suffix="", delete=False).name
    subprocess.run(
        [
            "cc", "-std=c11", "-O0", f"-I{writer_h}",
            str(harness_src), str(writer_c), "-o", binary,
        ],
        check=True,
    )
    return binary


def test_npy_writer_round_trips_float32(tmp_path):
    harness = tmp_path / "harness.c"
    out = tmp_path / "out.npy"
    harness.write_text(f"""
        #include "npy_writer.h"
        #include <stddef.h>
        int main(void) {{
            float data[] = {{1.5f, -2.25f, 0.0f, 3.125f, 4.0f, -5.5f}};
            size_t shape[] = {{2, 3}};
            return npyWriteFloat32("{out}", data, shape, 2);
        }}
    """)
    binary = _compile_writer_test_harness(harness)
    subprocess.run([binary], check=True)
    arr = np.load(out)
    assert arr.dtype == np.float32
    assert arr.shape == (2, 3)
    assert np.array_equal(arr, np.array([[1.5, -2.25, 0.0], [3.125, 4.0, -5.5]], dtype=np.float32))


def test_npy_writer_round_trips_int32(tmp_path):
    harness = tmp_path / "harness.c"
    out = tmp_path / "out.npy"
    harness.write_text(f"""
        #include "npy_writer.h"
        #include <stddef.h>
        #include <stdint.h>
        int main(void) {{
            int32_t data[] = {{0, 1, 2, 3, 4, 5, 6, 7}};
            size_t shape[] = {{8}};
            return npyWriteInt32("{out}", data, shape, 1);
        }}
    """)
    binary = _compile_writer_test_harness(harness)
    subprocess.run([binary], check=True)
    arr = np.load(out)
    assert arr.dtype == np.int32
    assert arr.shape == (8,)
    assert np.array_equal(arr, np.arange(8, dtype=np.int32))
