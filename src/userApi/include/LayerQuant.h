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

#endif /* LAYER_QUANT_H */
