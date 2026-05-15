#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "Layer.h"
#include "Tensor.h"

/* Legacy (pre-2026-05-15 factory API) — retained during PR 1/2 coexistence window. */
layer_t *softmaxLayerInitLegacy(quantization_t *forwardQ, quantization_t *backwardQ);
void freeSoftmaxLayerLegacy(layer_t *softmaxLayer);

#endif // SOFTMAXAPI_H
