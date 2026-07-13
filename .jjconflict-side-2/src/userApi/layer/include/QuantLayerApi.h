#ifndef ODT_QUANT_LAYER_API_H
#define ODT_QUANT_LAYER_API_H

#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationLayer.h"

/*! Borrowing variant — factory stores lq->outputQ / lq->propLossQ
 *  directly (pure conversion node — no consumed arithmetic). Caller retains
 *  ownership of `lq` and the quantizations; `lq` itself can be a compound
 *  literal that dies after the call. outputQ = dtype/qConfig target of the
 *  forward output; propLossQ = dtype/qConfig target of the backward propLoss.
 *  weightStorage/biasStorage are ignored (the Quantization layer has no
 *  parameters).
 *  Use when: outputQ/propLossQ are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *quantLayerInit(layerQuant_t *lq);

/*! Owning variant — factory deep-copies outputQ and propLossQ.
 *  Caller can free `lq` and both quantization_t* immediately after the
 *  factory returns. freeQuantLayer will tear down the copies.
 *  Use when: outputQ/propLossQ are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeQuantLayer tears them down too). */
layer_t *quantLayerInitOwning(layerQuant_t *lq);

/*! Tears down everything the factory allocated (internal config, layer).
 *  Reads config->ownsQuantizations to decide whether to also tear down the
 *  two quantization_t* and their qConfigs. */
void freeQuantLayer(layer_t *quantLayer);

#endif // ODT_QUANT_LAYER_API_H
