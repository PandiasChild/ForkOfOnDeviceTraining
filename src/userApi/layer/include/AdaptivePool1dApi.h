#ifndef ADAPTIVE_POOL1D_API_H
#define ADAPTIVE_POOL1D_API_H

#include <stddef.h>

#include "Layer.h"
#include "LayerQuant.h"

/*! AdaptiveAvgPool1d factory configuration. Produces a fixed output length
 *  regardless of input length. No kernel/stride/padding/dilation — the window
 *  geometry is derived from (inputLength, outputSize) at run time.
 *
 *  Usage:
 *      adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){ .outputSize = 1 }, lq);
 */
typedef struct adaptiveAvgPool1dInit {
    size_t outputSize; /* REQUIRED, >= 1 */
} adaptiveAvgPool1dInit_t;

/*! Borrowing variant — stores lq->outputQ in outputQ and lq->propLossQ
 *  in propLossQ verbatim (ownsQuantizations = false). */
layer_t *adaptiveAvgPool1dLayerInit(adaptiveAvgPool1dInit_t *init, layerQuant_t *lq);

/*! Owning variant — deep-copies outputQ / propLossQ (ownsQuantizations = true). */
layer_t *adaptiveAvgPool1dLayerInitOwning(adaptiveAvgPool1dInit_t *init, layerQuant_t *lq);

/*! Tears down everything the factory allocated; frees the two math quantizations
 *  only when ownsQuantizations is set. */
void freeAdaptiveAvgPool1dLayer(layer_t *layer);

#endif /* ADAPTIVE_POOL1D_API_H */
