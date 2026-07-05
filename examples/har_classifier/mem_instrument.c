#define SOURCE_FILE "har_mem_instrument"

#include "mem_instrument.h"

#include "MemProfile.h"
#include "Optimizer.h"
#include "Tensor.h"

/* ---- Analytic sums over the optimizer's trainable-parameter array --------- */

size_t memInstrumentParamBytes(optimizer_t *optim) {
    size_t total = 0;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        total += calcBytesPerTensor(optim->parameter[i]->param);
    }
    return total;
}

size_t memInstrumentGradBytes(optimizer_t *optim) {
    size_t total = 0;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        total += calcBytesPerTensor(optim->parameter[i]->grad);
    }
    return total;
}

size_t memInstrumentOptStateBytes(optimizer_t *optim) {
    size_t total = 0;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        states_t *s = optim->states[i];
        for (size_t j = 0; j < s->statesPerParameter; j++) {
            total += calcBytesPerTensor(s->stateBuffers[j]);
        }
    }
    return total;
}

/* ---- HAR-specific analytic activation / IO model -------------------------- */

/* Per-sample output element count of every layer of the fixed HAR topology
 * (train_c.c / train_c_sym.c):
 *   input [C=9, L=128]
 *     conv1 9->16 K7 SAME   -> [16,128]   relu -> [16,128]   maxpool/2 -> [16,64]
 *     conv2 16->32 K5 SAME  -> [32,64]    relu -> [32,64]    maxpool/2 -> [32,32]
 *     conv3 32->64 K3 SAME  -> [64,32]    relu -> [64,32]    avgpool/32 -> [64,1]
 *     flatten -> [64]       linear 64->6 -> [6]   softmax -> [6]
 * All wires are FLOAT32 in BOTH binaries (the SYM binary packs only WEIGHTS),
 * so activations_b / io_b are identical for float and SYM. */
static const size_t HAR_LAYER_OUT_ELEMS_PER_SAMPLE[] = {
    16 * 128, 16 * 128, 16 * 64, /* conv1, relu1, maxpool1 */
    32 * 64,  32 * 64,  32 * 32, /* conv2, relu2, maxpool2 */
    64 * 32,  64 * 32,  64 * 1,  /* conv3, relu3, avgpool  */
    64,       6,        6,       /* flatten, linear, softmax */
};

size_t memInstrumentHarActivationBytes(size_t microBatch) {
    /* Sum of EVERY layer's output-tensor bytes for ONE micro-batch.
     * calculateGradsSequential allocates all forward activations up front
     * (initLayerOutputs) and frees them only AFTER the full backward pass
     * (deInitLayerOutputs), so every forward activation is concurrently live
     * during backprop — this sum is a faithful peak, not a loose upper bound.
     * (It still ignores the transient per-op conversion scratch executeOp
     * allocates, so it slightly UNDER-counts the instantaneous peak.)
     *
     * microBatch is the CONCURRENT sample count, NOT the macro-batch:
     * trainingBatchDefault loops the macro-batch one sample at a time (loss.md:
     * dimensions[0]=B, today B=1) and accumulates grads at the optimizer, so
     * macro-batching does NOT multiply activation memory. Passing the macro-batch
     * here would over-count activations by that factor. */
    size_t elems = 0;
    size_t n = sizeof(HAR_LAYER_OUT_ELEMS_PER_SAMPLE) / sizeof(HAR_LAYER_OUT_ELEMS_PER_SAMPLE[0]);
    for (size_t i = 0; i < n; i++) {
        elems += HAR_LAYER_OUT_ELEMS_PER_SAMPLE[i];
    }
    return elems * microBatch * sizeof(float);
}

size_t memInstrumentHarIoBytes(size_t microBatch) {
    /* One micro-batch of input [microBatch, 9, 128] float + one-hot labels
     * [microBatch, 6] float. The macro-batch is streamed one sample at a time
     * (see the activation note), so this is the micro-batch, not the loader batch. */
    return (size_t)(9 * 128 + 6) * microBatch * sizeof(float);
}

/* ---- Stack high-water of one training step -------------------------------- */

static void memOneStepThunk(void *p) {
    memStepCtx_t *c = (memStepCtx_t *)p;
    optimizerFunctions_t fns = optimizerFunctions[c->optim->type];

    fns.zero(c->optim);
    trainingStats_t *stats = calculateGradsSequential(c->model, c->modelSize, c->lossConfig,
                                                      REDUCTION_MEAN, c->input, c->label);
    freeTrainingStats(stats);
    /* No scaleOptimizerGradients: the macro-batch mean scale is a scalar grad
     * multiply that does not deepen the call stack; the step itself does. */
    fns.step(c->optim);
    fns.zero(c->optim);
}

size_t memInstrumentStackPeakBytes(memStepCtx_t *ctx, size_t stackBytes) {
    return measurePeakStackBytes(memOneStepThunk, ctx, stackBytes);
}

/* ---- Reconciliation + emit ------------------------------------------------ */

void memInstrumentFinalize(memReport_t *r) {
    r->mcu_total_b = r->params_b + r->grads_b + r->optstate_analytic_b + r->activations_b + r->io_b;
    /* Signed on purpose: a positive gap = unaccounted heap (dataset, dataloaders,
     * per-op scratch, bookkeeping); a negative gap would mean the analytic model
     * over-counts. RECORD it — never tune the categories to shrink it. */
    r->reconciliation_gap_b = (long)r->heap_peak_b - (long)r->mcu_total_b;
}

void memInstrumentEmitJson(FILE *f, const memReport_t *r) {
    fprintf(f,
            "{\"sym_bits\": %d, "
            "\"dataset_b\": %zu, \"params_grads_b\": %zu, \"optstate_b\": %zu, "
            "\"params_b\": %zu, \"grads_b\": %zu, \"optstate_analytic_b\": %zu, "
            "\"activations_b\": %zu, \"io_b\": %zu, \"mcu_total_b\": %zu, "
            "\"heap_peak_b\": %zu, \"stack_peak_b\": %zu, \"rss_peak_kb\": %zu, "
            "\"reconciliation_gap_b\": %ld}",
            r->sym_bits, r->dataset_b, r->params_grads_b, r->optstate_b, r->params_b, r->grads_b,
            r->optstate_analytic_b, r->activations_b, r->io_b, r->mcu_total_b, r->heap_peak_b,
            r->stack_peak_b, r->rss_peak_kb, r->reconciliation_gap_b);
}

void memInstrumentPrintReconciliation(const memReport_t *r) {
    fprintf(stdout, "RECONCILIATION heap_peak=%zu mcu_total=%zu gap=%ld\n", r->heap_peak_b,
            r->mcu_total_b, r->reconciliation_gap_b);
    fflush(stdout);
}
