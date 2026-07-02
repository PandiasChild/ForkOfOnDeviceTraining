#define SOURCE_FILE "RELU_API"

#include <stdbool.h>
#include <stdlib.h> /* exit */

#include "ArithmeticType.h"
#include "Common.h" /* PRINT_ERROR */
#include "LayerQuant.h"
#include "Relu.h"
#include "ReluApi.h"
#include "StorageApi.h"

layer_t *reluLayerInitLegacy(quantization_t *forwardQ, quantization_t *backwardQ) {
    layer_t *reluLayer = reserveMemory(sizeof(layer_t));

    reluLayer->type = RELU;

    layerConfig_t *reluConfig = reserveMemory(sizeof(layerConfig_t));

    reluConfig_t *reluCfg = reserveMemory(sizeof(reluConfig_t));
    reluConfig->relu = reluCfg;
    reluCfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    reluCfg->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    reluCfg->outputQ = forwardQ;
    reluCfg->propLossQ = backwardQ;
    reluCfg->ownsQuantizations = false;

    reluLayer->config = reluConfig;

    return reluLayer;
}

void freeReluLayerLegacy(layer_t *reluLayer) {
    freeReservedMemory(reluLayer->config->relu);
    freeReservedMemory(reluLayer->config);
    freeReservedMemory(reluLayer);
}

/* ============================================================================
 * New factory API — layerQuant_t profile (PR 1).
 * ========================================================================== */

static void validateLayerQuantForRelu(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("reluLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->forwardMath == NULL) {
        PRINT_ERROR("reluLayerInit: layerQuant.forwardMath must be set");
        exit(1);
    }
    if (lq->backwardMath == NULL) {
        PRINT_ERROR("reluLayerInit: layerQuant.backwardMath must be set");
        exit(1);
    }
}

layer_t *reluLayerInit(layerQuant_t *lq) {
    validateLayerQuantForRelu(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = RELU;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    reluConfig_t *cfg = reserveMemory(sizeof(reluConfig_t));
    layerCfg->relu = cfg;
    layer->config = layerCfg;

    cfg->forwardMath = arithmeticFromQuantization(lq->forwardMath);
    cfg->propLossMath = arithmeticFromQuantization(lq->backwardMath);
    cfg->outputQ = lq->forwardMath;
    cfg->propLossQ = lq->backwardMath;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *reluLayerInitOwning(layerQuant_t *lq) {
    validateLayerQuantForRelu(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = RELU;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    reluConfig_t *cfg = reserveMemory(sizeof(reluConfig_t));
    layerCfg->relu = cfg;
    layer->config = layerCfg;

    cfg->forwardMath = arithmeticFromQuantization(lq->forwardMath);
    cfg->propLossMath = arithmeticFromQuantization(lq->backwardMath);
    cfg->outputQ = deepCopyQuantization(lq->forwardMath);
    cfg->propLossQ = deepCopyQuantization(lq->backwardMath);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeReluLayer(layer_t *reluLayer) {
    if (reluLayer == NULL) {
        return;
    }
    reluConfig_t *cfg = reluLayer->config->relu;

    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeReservedMemory(cfg->outputQ->qConfig);
            freeReservedMemory(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeReservedMemory(cfg->propLossQ->qConfig);
            freeReservedMemory(cfg->propLossQ);
        }
    }
    freeReservedMemory(cfg);
    freeReservedMemory(reluLayer->config);
    freeReservedMemory(reluLayer);
}
