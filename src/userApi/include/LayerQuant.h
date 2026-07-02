#ifndef LAYER_QUANT_H
#define LAYER_QUANT_H

#include "ArithmeticType.h"
#include "Quantization.h"

/*! Per-layer quantization profile: 4 by-value declared-arithmetic slots +
 *  6 storage-config pointers (spec 2026-07-02 arithmetic-type-split, D5).
 *
 *  Each layer factory reads only the fields it needs (documented in each
 *  layer's header).  Arithmetic fields are by-value: there is no "unset"
 *  sentinel, so factories validate the storage pointers they consume
 *  instead (NULL is a fatal error via PRINT_ERROR + exit).  Fields the
 *  layer doesn't read are ignored, so a single fully-populated
 *  layerQuant_t can be shared across all layers in a model.  */
typedef struct layerQuant {
    arithmetic_t forwardMath;    /* declared forward compute representation */
    arithmetic_t weightGradMath; /* declared weight-grad arithmetic (GEMM family) */
    arithmetic_t biasGradMath;   /* declared bias-grad arithmetic (GEMM family) */
    arithmetic_t propLossMath;   /* declared dx-wire arithmetic (kernel selection) */

    quantization_t *outputQ;   /* REQUIRED for every layer except Flatten: forward-wire storage */
    quantization_t *propLossQ; /* REQUIRED for every layer except Flatten: dx-wire storage */
    quantization_t *weightStorage;     /* REQUIRED for Conv1d / Conv1dTransposed / Linear /
                                           LayerNorm (gamma) */
    quantization_t *biasStorage;       /* REQUIRED for Conv1d / Conv1dTransposed / Linear
                                           (bias) / LayerNorm (beta) */
    quantization_t *weightGradStorage; /* grad storage knob; NULL = per-layer default */
    quantization_t *biasGradStorage;   /* grad storage knob; NULL = per-layer default */
} layerQuant_t;

/*! Populate `lq` from a single `q`: all four arithmetic slots derive via
 *  `arithmeticFromQuantization(q)`; `outputQ`/`propLossQ`/`weightStorage`/
 *  `biasStorage` all alias `q`; both grad-storage slots are NULL (factory
 *  default). Convenience for the common all-same-quantization case. Caller
 *  retains ownership of `q`. */
void layerQuantInitUniform(layerQuant_t *lq, quantization_t *q);

/*! Deep-copy a `quantization_t` and its `qConfig`. Returns NULL if `src` is NULL.
 *
 *  Caller owns the returned allocation. Free via:
 *      freeReservedMemory(result->qConfig);
 *      freeReservedMemory(result);
 *
 *  The `qConfig` size is dispatched by `src->type`; BOOL/INT32/FLOAT32 have
 *  no qConfig (result->qConfig == NULL). Unknown types fire PRINT_ERROR +
 *  exit(1).
 *
 *  Used by every `*LayerInitOwning` factory to materialize per-layer copies
 *  of the four math quantizations referenced by `layerQuant_t`. */
quantization_t *deepCopyQuantization(quantization_t *src);

#endif /* LAYER_QUANT_H */
