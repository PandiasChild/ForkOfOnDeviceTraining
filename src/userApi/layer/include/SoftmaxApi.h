#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "Layer.h"
#include "Tensor.h"

layer_t *softmaxLayerInit(quantization_t *forwardQ, quantization_t *backwardQ);

void freeSoftmaxLayer(layer_t *softmaxLayer);

#endif // SOFTMAXAPI_H
