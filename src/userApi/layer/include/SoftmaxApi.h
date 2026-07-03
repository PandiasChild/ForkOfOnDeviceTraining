#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "Layer.h"
#include "LayerQuant.h"
#include "Tensor.h"

/*! Borrowing variant — stores lq->outputQ in outputQ and
 *  lq->propLossQ in propLossQ verbatim.
 *  Use when: outputQ/propLossQ are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *softmaxLayerInit(layerQuant_t *lq);

/*! Owning variant — deep-copies outputQ + propLossQ via
 *  deepCopyQuantization.
 *  Use when: outputQ/propLossQ are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeSoftmaxLayer tears them down too). */
layer_t *softmaxLayerInitOwning(layerQuant_t *lq);

/*! Tears down the layer. Reads config->ownsQuantizations to decide
 *  whether to also free the two quantization_t and their qConfigs. */
void freeSoftmaxLayer(layer_t *softmaxLayer);

#endif // SOFTMAXAPI_H
