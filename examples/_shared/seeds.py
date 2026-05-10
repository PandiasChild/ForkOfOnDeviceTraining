"""Single source of truth for example seeds.

Both PyTorch and C training programs read SEED for any deterministic init,
and SHUFFLE_SEED is passed to the data loader's shuffle on both sides.
"""

SEED: int = 42
SHUFFLE_SEED: int = 42
