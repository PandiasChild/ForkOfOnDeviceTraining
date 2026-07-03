# Project Conventions

Contributor conventions for OnDeviceTraining. Detailed per-subsystem conventions
live under `docs/conventions/`; this file is the index and the cross-cutting
vision. (Claude sessions receive each subsystem's conventions
path-scoped automatically via `.claude/rules/`.)

## Vision: memory over float accuracy

ODT is a memory-light on-device-training research framework. SYM_INT32 paths
may be deliberately inaccurate with no float-matching — that is by design, not
a defect. FLOAT32-twin comparisons are a **ballpark sanity check**, not a tight
acceptance gate; SYM acceptance is "trains and converges to a useful model".
This does not license UB — overflow/garbage is still a bug (hence the #189 guard).

## Comment guidelines

A good comment carries one of two things the code cannot say for itself:

1. **Non-obvious rationale** — why this approach, not what it does (the code
   already says what). Deviations from an obvious/expected design, tricky
   invariants, or research-scope decisions belong here.
2. **When-to-use guidance at a decision point** — where two or more
   similar-looking alternatives exist (an `*Init` vs. `*InitOwning` factory
   pair, a mode enum, a documented exception to a general rule) and picking
   wrong is easy, say which one to reach for and why, right at the
   declaration.

What-narration ("this loop iterates over the array") drifts out of sync with
the code it describes and carries no information a reader can't get by
reading the next line. When you find it, delete it — don't rewrite it into
something that will drift again.

## Subsystem conventions

- [`conventions/tensor.md`](conventions/tensor.md) — `SYM_INT32` is a compute
  format, not storage (#261); the `SYM ↔ *` conversion bridge (#227).
- [`conventions/arithmetic-sym.md`](conventions/arithmetic-sym.md) — #189
  seed-rescale guard; Conv1d/Conv1dTransposed SYM_INT32 (#45); the int12-operand /
  int32-accumulator contract (no int64); the quantized grad-accumulation open
  problem (#218).
- [`conventions/loss.md`](conventions/loss.md) — loss forward/backward/reduction
  microbatch contracts; where the macro-batch divisor lives.
- [`conventions/allocation.md`](conventions/allocation.md) — allocation locality
  (alloc primitives only in `src/userApi/`; everything else via StorageApi).
- [`conventions/testing.md`](conventions/testing.md) — sanitizer gating; heap-tier
  test memory discipline; build-time gold-value generators.
- [`conventions/data-shape.md`](conventions/data-shape.md) — datasets deliver the
  natural geometric shape; reshape/flatten is the first model layer.
