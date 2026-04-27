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
| `linearLayerInit(...)`                    | `freeLinearLayer(l)` | layer config wrapper only           |
| `reluLayerInit(...)`                      | `freeReluLayer(l)`   | layer config wrapper only           |
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
