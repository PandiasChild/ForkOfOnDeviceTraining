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

## Sanitizer-driven memory bug detection

The C unit-test suite is run twice in CI: once normally (`c-build-and-test`),
and once under AddressSanitizer + UndefinedBehaviorSanitizer
(`c-asan-build-and-test`). The sanitizer job is a hard gate — any heap-OOB,
use-after-free, double-free, or UB diagnoses fails the PR. LeakSanitizer is
deliberately **off** (`detect_leaks=0`) in CI; see the opt-in recipe below.

### Local reproduction

The `unit_test_asan` preset is the source of truth. Same flags, same runtime
options as CI:

```bash
cmake --preset unit_test_asan
cmake --build --preset unit_test_asan
ctest --preset unit_test_asan
```

Or, in the devenv shell, the composite script:

```bash
run_asan_tests
```

Sanitizer flags (`-fsanitize=address,undefined -fno-sanitize=function
-fno-omit-frame-pointer -fno-sanitize-recover=all -g -O1`) propagate to every
target in the link graph via the configure preset — there is no opt-in per
target.

Runtime options the test preset sets:

- `ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`
- `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`

`halt_on_error=1` plus `-fno-sanitize-recover=all` means the **first** finding
aborts the test binary — earlier tests must run cleanly to surface later ones.
When triaging multiple unrelated failures, isolate by running individual test
binaries from `build/unit_test_asan/test/unit/...` directly.

### macOS toolchain requirement (LLVM ≥ 22)

macOS 26.4 changed the dyld shared-cache layout in a way that hangs
AddressSanitizer startup — `__asan_init` livelocks before `main()` (zero output,
~100% CPU) — for any compiler-rt **≤ 21.1.8**, which is the nixpkgs Darwin
default that `pkgs.clang` would otherwise provide. The upstream fix (LLVM
PR #182943, backported to `release/22.x`) ships in **LLVM ≥ 22**, so the devenv
`run_asan_tests` and `ci` scripts pin the ASan compiler to clang 22 (the
`nixpkgs-llvm22` input → `asanClang` in `devenv.nix`). The normal `gcc` build
and CI (Linux / apt-clang) are unaffected.

Running ASan outside devenv on macOS? Use clang ≥ 22, or Apple Command Line
Tools ≥ 26.5 (Apple backported the same fix into their clang 21). Apple CLT
≤ 26.3 will hang.

### Opt-in LeakSanitizer recipe

LSan is staged separately because it requires a cleanup convention every test
honours; see #82 for the umbrella. To run a single test or directory under LSan
during incremental cleanup work, override `detect_leaks` at the call site:

```bash
ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:halt_on_error=1" \
  build/unit_test_asan/test/unit/<module>/UnitTest<Name>
```

For broader recon (e.g. surveying which tests currently leak), prefer the
valgrind-based recipe in `docs/superpowers/tools/lsan-recon/` — it produces
reproducible, fully-attributed per-test reports.

## Allocation Locality

Only `src/userApi/` may call `malloc`, `calloc`, `realloc`, or `free` directly. All other code (sub-layers under `src/`, tests under `test/`) must route allocations through `reserveMemory` and `freeReservedMemory` in `src/userApi/StorageApi.{c,h}`.

Why:
- MCU stack overflows are silent killers; routing through StorageApi keeps stack usage predictable and small.
- Reviewers know exactly where to look for memory issues: `src/userApi/`.
- A future handle-based allocator can subsume the entire allocation surface in one API change instead of touching every call site.

Enforcement:
- A CI job (`alloc-locality` in `.github/workflows/ci.yml`) runs `git grep` against `src/` and `test/` (excluding `src/userApi/`) and fails the build on any match. Comments are excluded from the match.
- Exceptions: none today. If a use-case arises that genuinely needs a direct alloc primitive outside `src/userApi/`, escalate via a PR comment so the rule itself can be revisited.

## Test memory discipline

Unit tests in `test/unit/**` follow a tiered idiom for memory cleanup. The
tier boundary is mechanical: tests that contain no `*Init*` calls (i.e.,
purely stack-allocated `tensor_t`/`shape_t`/`quantization_t` designated
initializers) stay in the **stack-only tier** and need no cleanup. Any test
that calls `*Init*` (= heap allocation through `reserveMemory`) is in the
**heap tier** and follows three rules.

### Rule 1 — Build via the post-#106 primitives

Heap tensors are built by:

```c
size_t *dims  = reserveMemory(N * sizeof(size_t));
/* ... populate dims[i] ... */
size_t *order = reserveMemory(N * sizeof(size_t));
setOrderOfDimsForNewTensor(N, order);
shape_t *s    = reserveMemory(sizeof(shape_t));
setShape(s, dims, N, order);
tensor_t *t   = initTensor(s, quantizationInitFloat(), NULL);
tensorFillFromFloatBuffer(t, src, count);   /* or initDistribution(t, &d); */
```

The deprecated `tensorInitFloat` / `tensorInitSymInt32` / `tensorInit*`
family must not be used in new tests. Their attributes emit
`-Wdeprecated-declarations` to surface accidental adoption.

A file-local factory like `makeFloatTensorForDistTest` in
`test/unit/tensor/UnitTestTensorApi.c` is fine when 3+ tests in the same
file repeat the construction. A *cross-file* helper is deferred until 3+
test files repeat the same construction.

### Rule 2 — Free in reverse-init order

`freeTensor` cascades to data + shape (with its dims and order blocks) +
quantization + sparsity + the tensor struct itself. Do not call
`freeShape` or `freeQuantization` on a shape/quantization that was already
consumed by `initTensor` — that is a double-free. The cascade table:

| Allocation                                | Cleanup call         | Cascades to                         |
|-------------------------------------------|----------------------|-------------------------------------|
| `initTensor(s, q, sp)`                    | `freeTensor(t)`      | data, shape (+dims, +order), q, sp  |
| `parameterInit(p, g)`                     | `freeParameter(par)` | param tensor + grad (if non-NULL)   |
| `linearLayerInitLegacy(...)`              | `freeLinearLayerLegacy(l)` | layer config wrapper only     |
| `reluLayerInitLegacy(...)`                | `freeReluLayerLegacy(l)` | layer config wrapper only       |
| `softmaxLayerInit(...)`                   | `freeSoftmaxLayer(l)`| layer config wrapper only           |
| `sgdMCreateOptim(...)`                    | `freeOptimSgdM(o)`   | all registered parameters + states  |
| `inference(...)` (returns `tensor_t *`)   | `freeTensor(out)`    | as above                            |
| `inferenceWithLoss(...)`                  | `freeInferenceStats` | stats struct + output tensor        |
| `calculateGradsSequential(...)`           | `freeTrainingStats`  | stats struct                        |

Layer free-functions release only the config wrapper, not the parameters
they reference. When an optimizer is in play, `freeOptimSgdM` takes
ownership of the parameter cleanup — do not also call `freeParameter` on
the same pointers.

### Rule 3 — Assert-last (capture, free, assert)

ODT's Unity build defines `UNITY_INCLUDE_SETJMP`, so a failing
`TEST_ASSERT_*` longjmps out of the test function and any code after it
does not run. To keep LSan output meaningful — failing tests should still
report zero leaks attributable to the test fixture — every heap-tier test
follows this three-block shape:

```c
void testFoo(void) {
    /* 1. Build heap fixtures (Rule 1). */
    quantization_t *q = quantizationInitFloat();
    /* ... etc ... */

    /* 2. Exercise the system, capture every assertion value into a
     *    stack local. Do not assert here. */
    float capturedLoss = inferenceWithLoss(model, ...)->loss;
    /* (capture more if needed) */

    /* 3. Free in reverse-init order (Rule 2). */
    freeTensor(t);
    /* ... etc ... */

    /* 4. Assert on the captured locals. */
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, EXPECTED_LOSS, capturedLoss);
}
```

Reference exemplars in the tree: `test/unit/userAPI/UnitTestInferenceApi.c`,
`test/unit/userAPI/UnitTestMultiLayerTraining.c`,
`test/unit/tensor/UnitTestTensorApi.c::testInitDistribution_*`.

### Verification

A test file is considered idiom-compliant when, run under valgrind in the
`odt-lsan-recon:2026-04-22` Docker image with
`--leak-check=full --show-leak-kinds=all`, all four LEAK SUMMARY
categories report 0 bytes in 0 blocks (or valgrind emits "All heap blocks
were freed -- no leaks are possible"). The reproducible recipe and
container Dockerfile live in `docs/superpowers/tools/lsan-recon/`.

## Build-time gold-value generators (CMake + uv + PyTorch)

Some unit tests compare C-side numerics against PyTorch reference values. The
references are not committed: a Python script in the test directory emits a C
header (`expected_*.h`) at build time, which the test then `#include`s.

The wiring lives in `test/unit/<module>/CMakeLists.txt`:

```cmake
add_custom_command(
        OUTPUT ${GEN_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_<thing>.py
                --out ${GEN_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_<thing>.py
        VERBATIM
)
add_custom_target(generate_expected_<thing> DEPENDS ${GEN_HEADER})
add_dependencies(UnitTest<Name> generate_expected_<thing>)
target_include_directories(UnitTest<Name> PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
```

Reference exemplars:
`test/unit/arithmetic/generate_expected_conv1d_kernel.py`,
`test/unit/arithmetic/generate_expected_conv_transpose_1d_kernel.py`.

### Generator-script conventions

- Use `repr(v) + "f"` to format C float literals, **not** `f"{v:.9g}"`.
  `repr` always preserves a decimal point or exponent, so `10.0f` stays valid.
  `:.9g` produces `10` and the trailing `f` then makes it an invalid integer
  suffix that gcc rejects.
- Self-check fixtures with `assert torch.allclose(...)` before emitting them,
  so generator-side numerical drift fails the build instead of silently
  shifting expected values.
- `torch` and `torchvision` are declared as direct dependencies in
  `pyproject.toml`. The decoupling is intentional: generator scripts
  import `torch` directly, so the dependency belongs at the project
  level rather than inherited from `elasticai-creator`.

### CI implication: every job that runs `cmake --build` MUST install uv

The custom command above is invoked by ninja during the build phase, not by
configure. Any CI job that produces or runs targets depending on a generated
header must therefore have `uv` on `PATH` at build time. In
`.github/workflows/ci.yml` this is `c-build-and-test` and
`c-asan-build-and-test`; both install uv via `astral-sh/setup-uv@v6` and
`uv sync` before `cmake --preset ...`.

Locally this is silent: `devenv.nix` puts `uv` on `PATH` for the whole shell,
so `cmake --build` finds it without any explicit setup. CI is stricter and
catches drift here before merge.

When introducing a new generator under a new test target, audit every CI job
that builds the affected preset and add the uv setup steps if missing.

## Loss API: microbatch contracts

Each loss function in `src/loss_functions/` exposes:

- `forward(modelOutput, label, reduction) → float`
- `backward(modelOutput, label, result) → void`
- `computeMeanScale(totalSamples, modelOutput) → float`

### Reduction split

`lossConfig_t.backwardReduction` is the user's training-strategy choice — it
drives whether `scaleOptimizerGradients` runs between `trainingBatchDefault`
and `optimFns.step`. It is a config field.

`forwardReduction` is a per-call parameter on every aggregator
(`trainingBatchDefault`, `evaluationBatch`, `evaluationEpoch`, `inferenceWithLoss`,
`calculateGradsFn_t`). It controls how the per-microbatch loss value is
reported. `trainingRun` is the only function that hardcodes it
(to `REDUCTION_MEAN`) so train and eval losses are comparable; lower-level
callers pick freely.

### Microbatch shape

`modelOutput->shape->dimensions[0]` is the microbatch dimension `B`. For
`B=1` today, output shape is `[F]` (the leading 1 is implicit). For `B>=1`
in the future, output shape is `[B, F]` and `numFeaturesPerSample = numElements / B`.

**Uniform-B assumption** (DataLoader contract): all microbatches in one
macro batch have equal `B`. The MEAN aggregator divides by total samples
(`Σ batch->size`) rather than by `(numberOfBatches × B)`, so non-uniform B
would skew the mean. ODT's DataLoader currently always produces uniform
batches via `dropLast=true`; non-uniform B is out of contract.

### Backward macro-scaling

Backward writes raw per-element gradients (`2(o-l)` for MSE, `(p-y)` for CE).
The macro-batch divisor lives at the optimizer:

- `lossFunctions[lossConfig.funcType].computeMeanScale(N, modelOutput)`
  returns the PyTorch-parity divisor (`1/(N*F)` for MSE, `1/N` for CE).
- `scaleOptimizerGradients(optimizer, factor)` multiplies every parameter's
  `grad` field by the factor in place.
- `trainingEpochDefault` calls these between accumulation and `step`,
  but only when `backwardReduction == REDUCTION_MEAN`.

For SUM (or future per-sample weighted variants — see #150), the backward
gradient flows through unscaled.

### Shape assertion (deferred)

Runtime assertion of the `dimensions[0] >= 1` contract is deferred to the
microbatch-B>1 umbrella (#152) — specifically #153. Today (B=1 only) the
assertion would be effectively a no-op; the protective value materialises
when B>1 becomes a real feature target.

## Quantized gradient accumulation — known precision Open Problem

As of the quantized-gradient prerequisite (`gradInit`, 2026-06-05) a trainable
layer's parameter gradient can be stored in the dtype its `backwardMath`
declares. For SYM_INT32 grads, the per-microbatch accumulation reuses the
existing `addSymInt32TensorsInplace` ("strategy A", dynamic-rescale): it
dequantizes both the running grad and the new microbatch grad to float, adds,
and re-quantizes the running sum to a new absmax-derived scale **on every
microbatch**.

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
(NOT range-restored). An explicitly-chained Quantization layer (#192) restores
the operand width downstream — the same contract as Linear/LayerNorm. Per-output-
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
  in `gradInitSymInt32` (`getQLike` preserves the source width). They are
  value-sums; wider is free.
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

The training loop (`CalculateGradsSequential.c`) allocates grad/activation
tensors from the **forward** qConfig, not the backward qConfigs — so a full-SYM
chain needs each layer's `propLossQ` to agree with the forward-derived grad dtype
(else the #187 guard fires), exactly as for Linear. The Conv→Quant→…→MSE chain
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

The training loop allocates every grad/activation tensor from the FORWARD output
qConfig (`initGradTensor`), so a uniformly-SYM chain (every `forwardQ` SYM_INT32)
makes every grad tensor SYM_INT32 and every layer's `propLossQ` match — the #187
guard passes. SYM-trainable conv layers are built via the low-level
`initConv1dTransposedConfigWithWeightsAndBias` with SYM `parameter_t`s (the
high-level factory keeps grads FLOAT32, matching the Linear KAIMING factory).
`Conv1dTransposed → Quant → MSE` trains under
`calculateGradsSequential` + `sgdStepM(SYM_INT32)`.

## SYM ↔ * conversion bridge (#227)

`SYM` is the sub-byte bit-packed **storage** dtype; `SYM_INT32` is the int32-slot
**compute** dtype. The MCU lifecycle is store-packed (`SYM`) → unpack to int32
(`SYM_INT32`) → compute → repack. `conversionMatrix`
(`src/tensor/TensorConversion.c`) fills these cells: PR-B implements the **unpack
row** (`SYM → {SYM_INT32, FLOAT32, INT32, ASYM}`); the pack column (`* → SYM`) is
PR-C.

**Sign-extend on unpack.** `byteConversion` is a pure bit-copy that ZERO-FILLS on
widen, so a packed signed mantissa (e.g. `−3` at qBits=6 = `0b111101`) would read
back as `61`. Every `SYM →` cell routes through the shared
`unpackSignExtend(src, srcBits, dst, n)` helper, which widens then sign-extends the
two's-complement payload from `srcBits` (`(v ^ signBit) − signBit`). ASYM codes are
non-negative, so the ASYM **pack** path does not sign-extend.

**`int_repr` vs `dequantize` (deliberate, documented asymmetry).** A conversion
whose destination is `INT32` emits the integer **codes** and drops the scale
(`int_repr`); a conversion whose destination is `FLOAT32` emits the **values** with
the scale applied (`dequantize`). This mirrors PyTorch `int_repr()` vs
`dequantize()` and is consistent across both source dtypes: `SYM → INT32` and
`SYM_INT32 → INT32` are both `int_repr`; `SYM → FLOAT32` and `SYM_INT32 → FLOAT32`
are both `dequantize`. No value-rounding `→INT32` variant exists (YAGNI;
near-useless for `scale ≪ 1`).

**Rescale on the symmetric↔asymmetric transition.** `SYM → ASYM` always rescales
(dequantize → derive a fresh asym `scale`+`zeroPoint` from min/max → requantize →
pack): a symmetric code grid cannot hold an off-center `+zeroPoint` band at the
carried scale, independent of width.

## Vision: memory over float accuracy

ODT is a memory-light on-device-training research framework. SYM_INT32 paths
may be deliberately inaccurate with no float-matching — that is by design, not
a defect. FLOAT32-twin comparisons are a **ballpark sanity check**, not a tight
acceptance gate; SYM acceptance is "trains and converges to a useful model".
This does not license UB — overflow/garbage is still a bug (hence the #189 guard).
