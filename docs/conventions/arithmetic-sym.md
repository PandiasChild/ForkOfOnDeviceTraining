# Arithmetic & SYM_INT32 kernels

Conventions for the integer-math path: `src/arithmetic/**` and the SYM kernels of
`src/layer/{Conv1d,Conv1dTransposed,Linear,LayerNorm}*`. Path-scoped for Claude
via `.claude/rules/arithmetic-sym.md`.

## SYM_INT32 seed-rescale + the #189 guard

A SYM_INT32 parameter that must enter an integer accumulator at a *different*
scale — the forward bias seed (Matmul today; Conv when #45 lands) and the
LayerNorm affine beta seed — is converted via `rescaleIntoAccumulatorScale`
(`src/arithmetic/Rounding.c`): `seed = round(param_q * param_scale /
accumulator_scale)`. The `float -> int32` cast is data-dependent and is UB on
overflow (#189); the helper guards it NaN-robustly (`!(x <= T)`, reserving one
worst-case int16 product `32768*32767` of headroom) under `-DODT_SEED_GUARD`
(default ON; a future MCU/release build disables it, with UBSan #204 covering
occurrences). All seed-rescale sites route through this one helper.

This refold is deliberate, not a wart: it holds the real-valued bias **constant**
under ODT's dynamic per-input activation scaling. A fixed integer added raw
(`seed = b_int`, ignoring the bias scale) would apply the bias at
`s_acc / s_bias` of its value (≈0.01-0.05% on real layers — effectively deleting
it) and make it co-scale with input magnitude; the refold recomputes the seed
each forward (`∝ 1/s_acc`) so the bias stays a constant offset. The bias stays
SYM_INT32 (never a float master — the optimizer is single-dtype); a wide
raw-integer bias (qMaxBits=32, scale=1) would need a structurally different
scheme and is out of scope.

## Conv1d / Conv1dTransposed SYM_INT32 (#45)

Two integer sliding-window cores live in `src/arithmetic/`, siblings of the
FLOAT kernels with identical loop nest + `SlidingWindow1d` geometry:

- `conv1dKernelSymInt32` — gather forward; Conv1d forward, and Conv1dTransposed's
  `dx` adjoint in PR3.
- `convTranspose1dKernelSymInt32` — scatter forward; Conv1d's `dx` adjoint, and
  Conv1dTransposed's forward in PR3.

Both emit **raw accumulator-range int32 mantissas** at output scale `s_in·s_w`
(NOT range-restored). As of PR1b the `executeOp` epilogue restores operand width at the
**producer** on the migrated paths (the dx wire; Linear/LayerNorm backward; the Quantization
layer itself — `OUT_WRITE` routes SYM→SYM through the conversionMatrix diagonal). SYM
**forward** chains are not yet funnel-routed: an explicitly-chained Quantization layer after
each SYM producer remains REQUIRED (enforced by `validateModelQuantization`) until the forward
migration (PR1b.2), after which it becomes an optional storage node. Per-output-
channel bias is refolded into the product scale via `rescaleIntoAccumulatorScale`
(the #189 guarded helper); never raw-added.

Conv1d backward dispatches on **three independent qConfigs** (`weightGradQ`,
`biasGradQ`, `propLossQ`), like `linearBackward`:

- **weightGrad (SYM)** = strategy A: integer gather into a fresh `reserveMemory`
  intermediate at scale `s_loss·s_in`, then `addSymInt32TensorsInplace` into the
  SYM grad accumulator (fresh absmax scale).
- **biasGrad (SYM)** = an int32 `(batch × outputLength)` accumulator per output
  channel, then `rescaleIntoAccumulatorScale(sum, s_loss, s_bg, mode)` at the
  bias-grad's fixed scale (the #218 scheme).
- **dx / propLoss (SYM)** = `convTranspose1dKernelSymInt32(lossGrad, weights)`,
  scale `s_loss·s_w`, guarded by the #187 fail-fast if `propLoss` is not SYM.

### Operand bit-width: int12, not int16 (int32-accumulator soundness)

SYM kernels accumulate **products** of operands in an **int32** accumulator (no
int64 — hard rule). For symmetric `b`-bit operands each product is ≤ 2^(2b−2),
so an int32 accumulator (~2^31) holds only ~2^(33−2b) worst-case product terms
before signed overflow (UB):

| operand width | max product | int32 term at which overflow first occurs |
|---|---|---|
| int16 (qMaxBits=16) | 2^30 | 2 |
| int12 (qMaxBits=12) | 2^22 | 512 |
| int8  (qMaxBits=8)  | 2^14 | 131072 |

The number of worst-case terms that still **fit** is one less: int16 survives 1,
int12 survives **511**, int8 survives 131071 — i.e. int12 is sound for reductions
of length **N ≤ 511** (`512·2^22 = 2^31 > INT32_MAX`).

int16×int16→int32 is **unsound for product-accumulation** (forward, dx,
weightGrad) — it overflows after ~2 full-scale terms; it is sound only for
*value* sums (biasGrad). Conv SYM therefore uses **int12 operands**
(`quantizationInitSymInt32WithBits(rm, 12)`): products ≤ 2047² ≈ 4.2e6, ~512-term
int32 headroom — ample for the batch=1 MCU regime ODT targets, matching the
low-bit×low-bit→int32 arithmetic of the Deutel FQT paper (arXiv:2407.10734) /
TFLite. The **grad accumulators stay int16** (wider accumulator, free since SYM
stores int32 regardless of qMaxBits). The **kernels are bit-width-agnostic** —
only the quantization configs change; the int32 accumulator (no int64) is kept.

**Realized framework-wide int12 contract (PR-A, #227):**

- The SYM_INT32 **operand** default is int12 via the compile-time knob
  `ODT_SYM_OPERAND_QMAXBITS` (=12), set in `initSymInt32QConfig`
  (`src/tensor/include/Quantization.h`). Override per-build with
  `-DODT_SYM_OPERAND_QMAXBITS=N` (e.g. =8 for layers wider than 511).
- `matmulIntCore` (Linear forward / propLoss / weightGrad) and the LayerNorm
  **affine product** now run on int12 operands, enforced by op-entry guards
  (`matmulValidateSymOperand` at both Matmul SYM entries;
  `layerNormValidateSymTensor` lowered to the knob). LayerNorm's per-group
  mantissa-sum is a value-sum and stays sound at any qMaxBits ≤ 16.
- **Grad accumulators stay int16** via `ODT_SYM_GRAD_QMAXBITS` (=16), pinned
  in `gradInitSymInt32` (`getQLike` preserves the source width). biasGrad is a
  value-sum; weightGrad is a sum of products (int32 accumulate → requantize).
  Whether grads should be stored SYM_INT32 at all is under redesign — #261.
- int12 is sound only for reductions **N ≤ 511**; the runtime N-vs-budget check
  is a deferred follow-up. The #189 policy (release runs free, CI UBSan #204)
  backstops residual overflow.
- Note: the conv weightGrad product mixes an int12 input with an int16 grad
  operand under the #218 grad-accumulator scheme — its budget is governed by
  #218/#45, not closed by this operand flip.
- The unit-test gold suite validates the **default** int12/int16 contract
  (`ODT_SYM_OPERAND_QMAXBITS=12`, `ODT_SYM_GRAD_QMAXBITS=16`); building with a
  knob override (e.g. `-DODT_SYM_OPERAND_QMAXBITS=8`) diverges from those gold
  fixtures, which is expected and intentional.

The training loop (`CalculateGradsSequential.c`) allocates each dx-wire buffer
from the **producing layer's declared backward config** (`backwardWireQ`:
`propLossQ` for Linear/Conv/pools, `backwardQ` for Relu/Softmax/Dropout/LayerNorm/
Quantization; Flatten and the loss seed pass through the upstream dtype) — #221.
Uniform chains behave exactly as before. The Conv→Quant→…→MSE chain
wiring + FLOAT32-twin convergence check is PR3.

### Conv1dTransposed SYM_INT32 (PR3)

Conv1dTransposed is Conv1d's adjoint with roles swapped, so it reuses BOTH PR2
cores — no new kernels:

- **forward** = `convTranspose1dKernelSymInt32` (the scatter core; its internal
  per-channel bias-seed refold gives ConvT bias for free). Pass `outputPadding`.
- **dx / propLoss** = `conv1dKernelSymInt32` (the gather core, the VALID adjoint),
  guarded by the #187 fail-fast if `propLoss` is not SYM_INT32.
- **weightGrad** = strategy A: a scatter-style integer gather (ConvT weight layout
  `[Cin, Cout/groups, K]`, index `(ic·outChPerGroup + ocOffset)·K + k`) into a fresh
  `reserveMemory` int32 intermediate at scale `s_in·s_loss`, then
  `addSymInt32TensorsInplace` into the SYM grad accumulator.
- **biasGrad** = the same fixed-scale refold as Conv1d (`rescaleIntoAccumulatorScale`
  over the `batch × outputLength` int32 sum).

Backward dispatches on three independent qConfigs (`weightGradQ`/`biasGradQ`/
`propLossQ`), like `conv1dBackward`/`linearBackward`. Operands are int12, grad
accumulators int16, accumulators int32 — no int64. Conv1dTransposed is VALID-only
(Phase 1), so the adjoint never hits a SAME/EXPLICIT padLeft.

### Validator (PR3)

`producerForwardQ` (`ModelValidationApi.c`) now returns the conv layer's `forwardQ`
for CONV1D and CONV1D_TRANSPOSED, bringing SYM-producing conv layers under the
int16 inter-layer contract: a SYM conv producer must be followed by a Quantization
layer (or sit in the last position).

### SYM training chains

The training loop allocates each dx-wire buffer from the **producing layer's
declared backward config** (`backwardWireQ`: `propLossQ` for Linear/Conv/pools,
`backwardQ` for Relu/Softmax/Dropout/LayerNorm/Quantization; Flatten and the
loss seed pass through the upstream dtype) — #221. Uniform chains behave exactly
as before: a uniformly-SYM chain (every `forwardQ` SYM_INT32) makes every grad
tensor SYM_INT32 and every layer's `propLossQ` match — the #187 guard passes.
SYM-trainable conv layers are built via the low-level
`initConv1dTransposedConfigWithWeightsAndBias` with SYM `parameter_t`s (the
high-level factory keeps grads FLOAT32, matching the Linear KAIMING factory).
`Conv1dTransposed → Quant → MSE` trains under
`calculateGradsSequential` + `sgdStepM(SYM_INT32)`.

## Quantized gradient accumulation — known precision Open Problem

As of the quantized-gradient prerequisite (`gradInit`, 2026-06-05) a trainable
layer's parameter gradient can be stored in `param->grad->quantization`'s dtype
(grad-storage axis); `backwardMath` is the declared backward *arithmetic*.
Accumulation is the `OUT_ACC_DYNAMIC_RESCALE` epilogue mode of `executeOp` (no
longer inside kernels). For SYM_INT32 grads, the per-microbatch accumulation
delegates to the funnel's rescaling epilogue, which dequantizes both the running
grad and the new microbatch grad to float, adds, and re-quantizes the running sum
to a new absmax-derived scale **on every microbatch**.

This is functionally correct end-to-end today, but **not** numerically ideal:

- Quantization noise compounds with the number of microbatches M.
- The running-sum absmax is pinned by the heaviest microbatch, coarsening the
  LSB for the accumulated small-gradient mass.

Preliminary characterization (internal simulation, M=100, N=64, σ=1e-3 with a
10% ×50 heavy tail — *problem characterization only, not a basis for a chosen
solution*):

| Strategy | Final rel. error vs float64 | Float-free? |
|---|---|---|
| A — dynamic-rescale (current) | ~1.5e-4, **grows with M** (2.0e-5 @ step1 → 1.7e-4 @ step100) | No |
| B — fixed-scale integer accum | ~9.9e-5 | Yes |
| C — float accum, quantize-at-read | ~2.2e-5 | No |

We deliberately ship strategy A now and do **not** adopt B/C or any homegrown
numerical scheme. The resolution path is a literature review (stochastic-rounding
accumulators, error-feedback / residual accumulation, higher-precision master
grads, block/group scaling, …) → implement or improve a **published** technique.
Tracked as a separate research task (#218). This note is intentionally public
(not buried in a private spec) so contributors hitting accuracy issues in
quantized training know this is a known, expected limitation rather than a bug.

### Two accumulation schemes in-tree (both intentional)

- **Strategy A (dynamic-rescale)** — Linear SYM weight grads and LayerNorm
  gamma/beta grads: per-microbatch `addSymInt32TensorsInplace` (dequantize
  both operands with their own scales, float-add, requantize the running sum
  to a fresh absmax scale). Not float-free.
- **Fixed-scale integer accumulation** — Linear SYM bias grads
  (`linearCalcBiasGradsSymInt32`): increments are rescaled into the running
  grad's EXISTING scale and added in integer arithmetic; the scale is never
  re-derived during accumulation. The coarser resolution (LSB pinned by the
  running scale, which inits to 1.0) is inherent to the scheme.

The two schemes are now the named funnel epilogue modes chosen per call in layer
code: `OUT_ACC_DYNAMIC_RESCALE` (Strategy A) and `OUT_ACC_FIXED_SCALE` (Linear
SYM bias). On FLOAT32 targets, both collapse to the exact float add (no
quantization).

  **Attribution note:** this fixed-scale integer bias-GRADIENT accumulation is
  ODT's own construction and is NOT prescribed by Deutel et al.
  (arXiv:2407.10734). The paper's quantization is *dynamic*: scales are
  re-derived from observed data — weights every SGD update (Eqs. 6-7) — and the
  method is framed throughout as "dynamic adaptation of the zero-point and
  scale parameters" (Sec. IV-E). The paper has a forward bias (int32 bias on
  the int32 MAC accumulator, Fig. 2) but describes no bias-*gradient*
  accumulation, and it nowhere states that any scale is held static *during
  training* (the only static/PTQ mention is post-training, at deployment) — so
  absent evidence to the contrary, assume its scales are dynamic. ODT's
  fixed-scale bias-grad scheme, which never re-derives the scale during
  accumulation, therefore DEVIATES from the paper's dynamic scaling; the ODT
  scheme that corresponds to Deutel is Strategy A (dynamic-rescale, above).
  What ODT also follows from Deutel: per-layer error requant (~Eq. 4) and the
  float-space SGD step (~Eqs. 5-7). Scheme choice + the init-scale resolution
  limit: #218.

This is a research framework: deliberate scheme differences like this one
MUST be documented here, so experimental design stays separable from
accidental inconsistency. LayerNorm uses strategy A for BOTH gamma and beta
per the 2026-06-05 LayerNorm spec.

