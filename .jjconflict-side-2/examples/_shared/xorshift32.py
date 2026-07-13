"""Python mirror of the framework's XorShift32 RNG (src/rng/RNG.c).

This module exists so PyTorch DataLoaders shuffle samples in the same
order as the C framework's DataLoader, which uses rngShuffleIndices
seeded by shuffleSeed in dataLoaderInit (called once at init time, no
per-epoch reshuffle).

Single-shuffle-at-init means every epoch sees samples in the same
order. This matches DataLoader.c:41-43.
"""
from __future__ import annotations

UINT32_MAX = 0xFFFFFFFF


def xorshift32_next(state: int) -> int:
    """One step of Marsaglia 32-bit XorShift with shifts (13, 17, 5).

    All XOR-shifts are masked to 32 bits because Python ints are
    arbitrary-precision; without the mask, drift appears within ~30
    iterations.
    """
    x = state
    x ^= (x << 13) & UINT32_MAX
    x ^= x >> 17
    x ^= (x << 5) & UINT32_MAX
    return x


def _bounded(state: int, bound: int) -> tuple[int, int]:
    """Rejection-sampled uniform in [0, bound), mirroring rngBounded.

    Returns (new_state, sample).
    """
    limit = UINT32_MAX - (UINT32_MAX % bound)
    while True:
        state = xorshift32_next(state)
        if state < limit:
            return state, state % bound


def shuffle_indices(n: int, seed: int) -> list[int]:
    """Fisher-Yates shuffle mirroring rngShuffleIndices.

    Returns a permutation of [0, n). For n < 2 returns the identity
    permutation (matches the C early-return).
    """
    if n < 2:
        return list(range(n))
    state = seed if seed != 0 else UINT32_MAX
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        state, j = _bounded(state, i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return indices
