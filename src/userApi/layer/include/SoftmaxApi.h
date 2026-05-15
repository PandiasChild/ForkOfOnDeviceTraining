#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "Layer.h"
#include "LayerQuant.h"
#include "Tensor.h"

/* Legacy (pre-2026-05-15 factory API) — retained during PR 1/2 coexistence window. */
layer_t *softmaxLayerInitLegacy(quantization_t *forwardQ, quantization_t *backwardQ);
void freeSoftmaxLayerLegacy(layer_t *softmaxLayer);

/*! Borrowing variant — stores lq->forwardMath in forwardQ and
 *  lq->backwardMath in backwardQ verbatim. */
layer_t *softmaxLayerInit(layerQuant_t *lq);

/*! Owning variant — deep-copies forwardMath + backwardMath via
 *  deepCopyQuantization. */
layer_t *softmaxLayerInitOwning(layerQuant_t *lq);

/*! Tears down the layer. Reads config->ownsQuantizations to decide
 *  whether to also free the two quantization_t and their qConfigs. */
void freeSoftmaxLayer(layer_t *softmaxLayer);

#endif // SOFTMAXAPI_H
