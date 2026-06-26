#define SOURCE_FILE "SOFTMAX_API"

#include <stdbool.h>
#include <stdlib.h>

#include "Common.h"
#include "LayerQuant.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include <stdlib.h>
#include "Common.h"

layer_t *softmaxLayerInitLegacy(quantization_t *forwardQ, quantization_t *backwardQ) {
    layer_t *softmaxLayer = reserveMemory(sizeof(layer_t));

    softmaxLayer->type = SOFTMAX;

    layerConfig_t *layerConfig = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *softmaxConfig = reserveMemory(sizeof(softmaxConfig_t));
    if( layerConfig == NULL || softmaxConfig == NULL){
        PRINT_ERROR("Memory Allocation Failed");
        exit(1);
    }
    layerConfig->softmax = softmaxConfig;

    softmaxConfig->forwardQ = forwardQ;
    softmaxConfig->backwardQ = backwardQ;
    softmaxConfig->ownsQuantizations = false;
    softmaxLayer->config = layerConfig;

    return softmaxLayer;
}

void freeSoftmaxLayerLegacy(layer_t *softmaxLayer) {
    freeReservedMemory(softmaxLayer->config->softmax);
    freeReservedMemory(softmaxLayer->config);
    freeReservedMemory(softmaxLayer);
}

/* ============================================================================
 * New factory API — layerQuant_t profile (PR 2).
 * ========================================================================== */

static void validateLayerQuantForSoftmax(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("softmaxLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->forwardMath == NULL) {
        PRINT_ERROR("softmaxLayerInit: layerQuant.forwardMath must be set");
        exit(1);
    }
    if (lq->backwardMath == NULL) {
        PRINT_ERROR("softmaxLayerInit: layerQuant.backwardMath must be set");
        exit(1);
    }
}

layer_t *softmaxLayerInit(layerQuant_t *lq) {
    validateLayerQuantForSoftmax(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = SOFTMAX;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *cfg = reserveMemory(sizeof(softmaxConfig_t));
    layerCfg->softmax = cfg;
    layer->config = layerCfg;

    cfg->forwardQ = lq->forwardMath;
    cfg->backwardQ = lq->backwardMath;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *softmaxLayerInitOwning(layerQuant_t *lq) {
    validateLayerQuantForSoftmax(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = SOFTMAX;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *cfg = reserveMemory(sizeof(softmaxConfig_t));
    layerCfg->softmax = cfg;
    layer->config = layerCfg;

    cfg->forwardQ = deepCopyQuantization(lq->forwardMath);
    cfg->backwardQ = deepCopyQuantization(lq->backwardMath);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeSoftmaxLayer(layer_t *softmaxLayer) {
    if (softmaxLayer == NULL) {
        return;
    }
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;

    if (cfg->ownsQuantizations) {
        if (cfg->forwardQ != NULL) {
            freeReservedMemory(cfg->forwardQ->qConfig);
            freeReservedMemory(cfg->forwardQ);
        }
        if (cfg->backwardQ != NULL && cfg->backwardQ != cfg->forwardQ) {
            freeReservedMemory(cfg->backwardQ->qConfig);
            freeReservedMemory(cfg->backwardQ);
        }
    }
    freeReservedMemory(cfg);
    freeReservedMemory(softmaxLayer->config);
    freeReservedMemory(softmaxLayer);
}
