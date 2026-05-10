# Determinism Contract

Both PyTorch and C training programs in `examples/` must seed every
source of randomness from `examples/_shared/seeds.py` constants:

- `SEED = 42` — used for any deterministic init (`torch.manual_seed`, etc.).
- `SHUFFLE_SEED = 42` — passed to the data loader's shuffle.

## Init weights

Each side uses its own native init. PyTorch uses `torch.nn.init`
defaults (Kaiming uniform for `Conv1d`/`Linear` weights, uniform
`[-1/√fan_in, 1/√fan_in]` for biases). C uses the framework's native
parameter init via `tensorInitWithDistribution(XAVIER_UNIFORM, ...)`.
Init weights are NOT exchanged between implementations — that's the
design choice that keeps the parity story robust to known C-side init
drift.

## Shuffle ordering

Both sides shuffle the training set's index list once at init time
using a Marsaglia 32-bit XorShift (shifts 13/17/5) seeded from
`SHUFFLE_SEED`. The Python side mirrors `src/rng/RNG.c` via
`examples/_shared/xorshift32.py`. The C side uses the framework's
`rngSetSeed` + `rngShuffleIndices` directly. Per-epoch reshuffling is
NOT performed on either side (matches `DataLoader.c:41-43`).

## Parity assertion regime

Final-state only. After `N` epochs both sides should reach the same
test accuracy / loss within per-example tolerance tables in `compare.py`.
The framework's `parity.ParityCheck` either uses absolute or relative
tolerance, never both.

## Verifying the XorShift32 mirror

The Python test `python/tests/test_xorshift32.py::test_python_mirror_matches_c_for_*`
compiles a tiny C harness against `src/rng/RNG.c` at test time and
asserts byte-identical permutations. If you change either the Python
mirror or `src/rng/RNG.c`, that test will catch divergence.
