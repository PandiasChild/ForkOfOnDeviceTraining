"""Test the Python XorShift32 mirror in examples/_shared/xorshift32.py."""
import subprocess
import sys
import tempfile
from pathlib import Path

# Make examples/ importable from python/tests/.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.xorshift32 import xorshift32_next, shuffle_indices


def test_xorshift32_next_known_vectors():
    """Hand-computed first 4 outputs from seed=1 with shifts (13,17,5)."""
    state = 1
    state = xorshift32_next(state); assert state == 270369
    state = xorshift32_next(state); assert state == 67634689
    state = xorshift32_next(state); assert state == 2647435461
    state = xorshift32_next(state); assert state == 307599695


def test_xorshift32_seed_zero_maps_to_uint32_max():
    """Seed 0 must alias to UINT32_MAX inside shuffle_indices, matching rngSetSeed."""
    expected = shuffle_indices(8, 0xFFFFFFFF)
    actual   = shuffle_indices(8, 0)
    assert actual == expected


def test_shuffle_indices_is_permutation_of_range():
    """Output must be a permutation of [0, n)."""
    out = shuffle_indices(100, 42)
    assert sorted(out) == list(range(100))


def test_shuffle_indices_n_lt_2_returns_identity():
    """rngShuffleIndices in C returns immediately if n < 2."""
    assert shuffle_indices(0, 42) == []
    assert shuffle_indices(1, 42) == [0]


def test_shuffle_indices_deterministic():
    """Same (n, seed) yields same output."""
    a = shuffle_indices(1000, 42)
    b = shuffle_indices(1000, 42)
    assert a == b


def _compile_and_run_c_harness(n: int, seed: int) -> list[int]:
    """Compile the C harness against src/rng/RNG.c, run it, parse stdout."""
    harness_src = REPO_ROOT / "examples" / "_shared" / "verify_xorshift32_harness.c"
    rng_src = REPO_ROOT / "src" / "rng" / "RNG.c"
    rng_inc = REPO_ROOT / "src" / "rng" / "include"
    with tempfile.NamedTemporaryFile(suffix="", delete=False) as out:
        binary = out.name
    subprocess.run(
        [
            "cc", "-std=c11", "-O0",
            f"-I{rng_inc}",
            str(harness_src), str(rng_src),
            "-o", binary,
        ],
        check=True,
    )
    result = subprocess.run(
        [binary, str(n), str(seed)], check=True, capture_output=True, text=True,
    )
    return [int(tok) for tok in result.stdout.split()]


def test_python_mirror_matches_c_for_n10_seed42():
    expected = _compile_and_run_c_harness(10, 42)
    actual = shuffle_indices(10, 42)
    assert actual == expected


def test_python_mirror_matches_c_for_n1000_seed1():
    expected = _compile_and_run_c_harness(1000, 1)
    actual = shuffle_indices(1000, 1)
    assert actual == expected


def test_python_mirror_matches_c_for_seed_zero():
    expected = _compile_and_run_c_harness(50, 0)
    actual = shuffle_indices(50, 0)
    assert actual == expected
