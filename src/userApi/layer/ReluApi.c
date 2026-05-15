#define SOURCE_FILE "RELU_API"

#include <stdbool.h>
#include <stdlib.h> /* exit */
#include <string.h> /* memcpy */

#include "Common.h" /* PRINT_ERROR */
#include "LayerQuant.h"
#include "Relu.h"
#include "ReluApi.h"
#include "StorageApi.h"

layer_t *reluLayerInit(quantization_t *forwardQ, quantization_t *backwardQ) {
    layer_t *reluLayer = reserveMemory(sizeof(layer_t));

    reluLayer->type = RELU;

    layerConfig_t *reluConfig = reserveMemory(sizeof(layerConfig_t));

    reluConfig_t *reluCfg = reserveMemory(sizeof(reluConfig_t));
    reluConfig->relu = reluCfg;
    reluCfg->forwardQ = forwardQ;
    reluCfg->backwardQ = backwardQ;
    reluCfg->ownsQuantizations = false;

    reluLayer->config = reluConfig;

    return reluLayer;
}

void freeReluLayer(layer_t *reluLayer) {
    freeReservedMemory(reluLayer->config->relu);
    freeReservedMemory(reluLayer->config);
    freeReservedMemory(reluLayer);
}

/* ============================================================================
 * New factory API — layerQuant_t profile (PR 1).
 * ========================================================================== */

static quantization_t *reluDeepCopyQuantization(quantization_t *src) {
    if (src == NULL) {
        return NULL;
    }

    quantization_t *dst = reserveMemory(sizeof(quantization_t));
    dst->type = src->type;

    size_t cfgSize = 0;
    switch (src->type) {
    case FLOAT32:
        cfgSize = 0;
        break;
    case INT32:
        cfgSize = 0;
        break;
    case BOOL:
        cfgSize = 0;
        break;
    case SYM_INT32:
        cfgSize = sizeof(symInt32QConfig_t);
        break;
    case SYM:
        cfgSize = sizeof(symQConfig_t);
        break;
    case ASYM:
        cfgSize = sizeof(asymQConfig_t);
        break;
    default:
        PRINT_ERROR("reluLayerInitOwning: unknown quantization type %d", (int)src->type);
        exit(1);
    }
    if (cfgSize == 0) {
        dst->qConfig = NULL;
    } else {
        dst->qConfig = reserveMemory(cfgSize);
        memcpy(dst->qConfig, src->qConfig, cfgSize);
    }
    return dst;
}

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

    cfg->forwardQ = lq->forwardMath;
    cfg->backwardQ = lq->backwardMath;
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

    cfg->forwardQ = reluDeepCopyQuantization(lq->forwardMath);
    cfg->backwardQ = reluDeepCopyQuantization(lq->backwardMath);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeReluLayer(layer_t *reluLayer) {
    if (reluLayer == NULL) {
        return;
    }
    reluConfig_t *cfg = reluLayer->config->relu;

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
    freeReservedMemory(reluLayer->config);
    freeReservedMemory(reluLayer);
}
