#ifndef HAR_MEM_INSTRUMENT_H
#define HAR_MEM_INSTRUMENT_H

/* Shared memory instrumentation for the two HAR trainer binaries (train_c.c
 * FLOAT32 + train_c_sym.c packed-SYM). Both share the identical 12-layer model
 * and training loop, so the measurement + analytic sizing + JSON-emit logic
 * lives here ONCE and both binaries call it (no copy-paste).
 *
 * The heap counters (memProfile*Bytes) are no-ops returning 0 unless the whole
 * program is built with -DODT_MEM_PROFILE (PUBLIC on StorageApi). The stack
 * high-water and RSS probes work regardless. Callers gate the whole report on
 * #ifdef ODT_MEM_PROFILE so the CI bit-parity build carries zero instrumentation.
 */

#include <stddef.h>
#include <stdio.h>

#include "CalculateGradsSequential.h"
#include "MemProfile.h" /* measurePeakStackBytes, memProfileRssPeakKb */
#include "Optimizer.h"
#include "StorageApi.h" /* memProfileReset/Mark/CurrentBytes/PeakBytes */
#include "TrainingLoopApi.h"

/* One honest per-run memory breakdown. All *_b fields are bytes; rss is KiB.
 * A Python consumer (examples/_shared/log_schema.py memory contract) reads the
 * emitted keys verbatim, so field <-> key names must stay in lockstep with
 * memInstrumentEmitJson. */
typedef struct memReport {
    int sym_bits; /* SYM_BITS for the sym binary; -1 for the float binary */

    /* Instrumented phase marks (memProfileMark deltas over the heap counter). */
    size_t dataset_b;      /* live bytes after initDataSets */
    size_t params_grads_b; /* delta across buildModel (+ requantize for sym) */
    size_t optstate_b;     /* delta across sgdMCreateOptim */

    /* Analytic categories — what an MCU deployment would actually hold. */
    size_t params_b;            /* sum of weight+bias tensor bytes (dtype-aware) */
    size_t grads_b;             /* sum of grad tensor bytes */
    size_t optstate_analytic_b; /* sum of optimizer momentum-buffer bytes */
    size_t activations_b;       /* peak concurrent activation bytes, one batch */
    size_t io_b;                /* batched input + one-hot label bytes */
    size_t mcu_total_b;         /* params+grads+optstate+activations+io */

    /* Instrumented process-level anchors. */
    size_t heap_peak_b;  /* memProfilePeakBytes() */
    size_t stack_peak_b; /* measurePeakStackBytes() on one training step */
    size_t rss_peak_kb;  /* memProfileRssPeakKb() */

    /* heap_peak_b - mcu_total_b. RECORDED, never massaged (integrity rule). */
    long reconciliation_gap_b;
} memReport_t;

/* Analytic byte sums over the optimizer's authoritative trainable-parameter
 * array (optim->parameter / optim->states). Preferred over a traceModel* walk:
 * the optimizer already holds exactly the trainable weight/bias parameters and
 * their grads + momentum buffers, and calcBytesPerTensor is dtype-aware (packed
 * SYM weights count as ceil(qBits*N/8), FLOAT32 grads as 4*N), so this measures
 * the ACTUAL storage whatever it is. */
size_t memInstrumentParamBytes(optimizer_t *optim);
size_t memInstrumentGradBytes(optimizer_t *optim);
size_t memInstrumentOptStateBytes(optimizer_t *optim);

/* HAR-classifier-specific analytic activation / IO sizing for one MICRO-batch.
 * Encodes the fixed 12-layer topology (see mem_instrument.c); identical for the
 * float and the SYM binary because activation WIRES are FLOAT32 in both.
 *
 * Pass the MICRO-batch (concurrent samples per forward/backward), NOT the loader
 * macro-batch: trainingBatchDefault streams the macro-batch one sample at a time
 * (loss.md: dimensions[0]=B, today B=1) and accumulates grads at the optimizer,
 * so only one sample's activations are ever live. Passing the macro-batch would
 * over-count activations + IO by that factor. */
size_t memInstrumentHarActivationBytes(size_t microBatch);
size_t memInstrumentHarIoBytes(size_t microBatch);

/* Everything the stack thunk needs to run one representative training step
 * (zeroGrad -> calculateGradsSequential -> step -> zeroGrad) on one sample. */
typedef struct memStepCtx {
    layer_t **model;
    size_t modelSize;
    lossConfig_t lossConfig;
    tensor_t *input;
    tensor_t *label;
    optimizer_t *optim;
} memStepCtx_t;

/* Runs one training step on a painted pthread stack and returns the high-water
 * bytes touched. NOTE: performs a REAL gradient + optimizer step, mutating the
 * model and momentum state — call it AFTER any output the run must preserve. */
size_t memInstrumentStackPeakBytes(memStepCtx_t *ctx, size_t stackBytes);

/* Fill mcu_total_b and reconciliation_gap_b from the populated fields. */
void memInstrumentFinalize(memReport_t *r);

/* Emit the bare "memory" object body: {"sym_bits": ..., ...}. The caller places
 * it (writes the "memory": key and any surrounding commas). */
void memInstrumentEmitJson(FILE *f, const memReport_t *r);

/* Print the reconciliation line to stdout. Integrity: prints the gap as-is. */
void memInstrumentPrintReconciliation(const memReport_t *r);

#endif /* HAR_MEM_INSTRUMENT_H */
