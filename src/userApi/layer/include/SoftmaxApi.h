#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "Layer.h"
#include "LayerQuant.h"
#include "Tensor.h"

/*! Borrowing variant — stores lq->outputQ in outputQ and
 *  lq->propLossQ in propLossQ verbatim. */
layer_t *softmaxLayerInit(layerQuant_t *lq);

/*! Owning variant — deep-copies outputQ + propLossQ via
 *  deepCopyQuantization. */
layer_t *softmaxLayerInitOwning(layerQuant_t *lq);

/*! Tears down the layer. Reads config->ownsQuantizations to decide
 *  whether to also free the two quantization_t and their qConfigs. */
void freeSoftmaxLayer(layer_t *softmaxLayer);

#endif // SOFTMAXAPI_H
