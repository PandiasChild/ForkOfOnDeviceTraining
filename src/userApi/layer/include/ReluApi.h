#ifndef ODT_RELU_H
#define ODT_RELU_H

#include "Layer.h"
#include "LayerQuant.h"

/*! Borrowing variant — stores lq->outputQ and lq->propLossQ as the
 *  layer's outputQ / propLossQ pointers. Caller retains ownership of lq. */
layer_t *reluLayerInit(layerQuant_t *lq);

/*! Owning variant — deep-copies lq->outputQ and lq->propLossQ.
 *  Caller can drop lq + quantizations immediately. */
layer_t *reluLayerInitOwning(layerQuant_t *lq);

/*! Tears down the layer. Reads config->ownsQuantizations to decide whether
 *  to also free the two quantization_t* and their qConfigs. */
void freeReluLayer(layer_t *reluLayer);

#endif // ODT_RELU_H
