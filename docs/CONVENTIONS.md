# Project Conventions

## Data Shape Convention

Datasets deliver samples in their natural geometric shape (e.g. `[C, H, W]`
for images, `[C, L]` for time series). Any `reshape`, `flatten`, or `view`
operation is the **first layer of the model**, not a preprocessing step in
the dataset. This:

- keeps dataset code independent of downstream model topology
- allows one dataset to feed models with different input ranks
- matches the PyTorch / Keras / elastic-ai.creator IR convention, so a future
  ir2c can compile each shape transform to a corresponding C layer

For flatten-to-2D, use `flattenLayerInit()` from `FlattenApi.h`.

## Allocation Locality

Only `src/userApi/` may call `malloc`, `calloc`, `realloc`, or `free` directly. All other code (sub-layers under `src/`, tests under `test/`) must route allocations through `reserveMemory` and `freeReservedMemory` in `src/userApi/StorageApi.{c,h}`.

Why:
- MCU stack overflows are silent killers; routing through StorageApi keeps stack usage predictable and small.
- Reviewers know exactly where to look for memory issues: `src/userApi/`.
- A future handle-based allocator can subsume the entire allocation surface in one API change instead of touching every call site.

Enforcement:
- A CI job (`alloc-locality` in `.github/workflows/ci.yml`) runs `git grep` against `src/` and `test/` (excluding `src/userApi/`) and fails the build on any match. Comments are excluded from the match.
- Exceptions: none today. If a use-case arises that genuinely needs a direct alloc primitive outside `src/userApi/`, escalate via a PR comment so the rule itself can be revisited.
