#ifndef LAYER_QUANT_H
#define LAYER_QUANT_H

#include "Quantization.h"

/*! Per-layer quantization profile.
 *
 *  Each layer factory reads only the fields it needs (documented in each
 *  layer's header).  For every field it reads, NULL is a fatal error
 *  (PRINT_ERROR + exit).  Fields the layer doesn't read are ignored, so a
 *  single fully-populated layerQuant_t can be shared across all layers in
 *  a model.  */
typedef struct layerQuant {
    quantization_t *forwardMath;   /* REQUIRED for every layer except Flatten */
    quantization_t *backwardMath;  /* REQUIRED for every layer except Flatten */
    quantization_t *weightStorage; /* REQUIRED for Conv1d / Conv1dTransposed / Linear */
    quantization_t
        *biasStorage; /* REQUIRED for Conv1d / Conv1dTransposed / Linear when bias is enabled */
} layerQuant_t;

/*! Populate all four slots of `lq` with the same `q`.  Convenience for the
 *  common all-same-quantization case.  Caller retains ownership of `q`. */
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
