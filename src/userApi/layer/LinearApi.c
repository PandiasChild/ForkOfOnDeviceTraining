#define SOURCE_FILE "LINEAR_API"

#include <stdbool.h>
#include <stdlib.h>

#include "Common.h"
#include "Distributions.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "Common.h"

layer_t *linearLayerInitLegacy(parameter_t *weights, parameter_t *bias, quantization_t *forwardQ,
                               quantization_t *weightGradsQ, quantization_t *biasGradsQ,
                               quantization_t *propLossQ) {
    layer_t *linearLayer = reserveMemory(sizeof(layer_t));

    linearLayer->type = LINEAR;

    layerConfig_t *layerConfig = reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *linearConfig = reserveMemory(sizeof(linearConfig_t));
    if( layerConfig == NULL || linearConfig == NULL){
        PRINT_ERROR("Memory Allocation Failed");
        exit(1);
    }
    layerConfig->linear = linearConfig;

    linearConfig->weights = weights;
    linearConfig->bias = bias;
    linearConfig->forwardQ = forwardQ;
    linearConfig->weightGradQ = weightGradsQ;
    linearConfig->biasGradQ = biasGradsQ;
    linearConfig->propLossQ = propLossQ;
    linearConfig->ownsQuantizations = false;

    linearLayer->config = layerConfig;

    return linearLayer;
}

layer_t *linearLayerInitNonTrainableLegacy(tensor_t *weights, tensor_t *bias,
                                           quantization_t *forwardQ) {
    layer_t *linearLayer = reserveMemory(sizeof(layer_t));

    linearLayer->type = LINEAR;

    layerConfig_t *layerConfig = reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *linearConfig = reserveMemory(sizeof(linearConfig_t));
    if( layerConfig == NULL || linearConfig == NULL){
        PRINT_ERROR("Memory Allocation Failed");
        exit(1);
    }
    layerConfig->linear = linearConfig;

    linearConfig->weights = parameterInit(weights, NULL);
    linearConfig->bias = parameterInit(bias, NULL);
    linearConfig->forwardQ = forwardQ;
    linearConfig->ownsQuantizations = false;

    linearLayer->config = layerConfig;

    return linearLayer;
}

void freeLinearLayerLegacy(layer_t *linearLayer) {
    freeReservedMemory(linearLayer->config->linear);
    freeReservedMemory(linearLayer->config);
    freeReservedMemory(linearLayer);
}

/* ============================================================================
 * New factory API — layerQuant_t profile + linearInit_t struct (PR 1).
 * ========================================================================== */

static bool resolveLinearBias(bias_t b) {
    switch (b) {
    case BIAS_DEFAULT:
        return true; /* PyTorch parity for Linear */
    case BIAS_TRUE:
        return true;
    case BIAS_FALSE:
        return false;
    default:
        PRINT_ERROR("linearLayerInit: invalid bias value (got %d)", (int)b);
        exit(1);
    }
}

/*! Build a heap-owned shape_t with the given dims; the tensor that this shape
 *  is passed to takes ownership and freeTensor cascades into freeShape. */
static shape_t *buildOwnedShape(const size_t *srcDims, size_t numberOfDims) {
    size_t *dims = reserveMemory(numberOfDims * sizeof(size_t));
    for (size_t i = 0; i < numberOfDims; i++) {
        dims[i] = srcDims[i];
    }
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numberOfDims, order);
    return shape;
}

static parameter_t *allocateLinearWeights(size_t inFeatures, size_t outFeatures,
                                          quantization_t *storageQ, quantization_t *gradQ) {
    /* Weight tensor: shape [outFeatures, inFeatures]. The tensor takes ownership
     * of `shape` and `quantization`, so we clone the borrowed storageQ via
     * getQLike to avoid tying the tensor's lifetime to the caller's quant. */
    shape_t *shape = buildOwnedShape((size_t[]){outFeatures, inFeatures}, 2);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);

    /* PyTorch-aligned default: Kaiming uniform with fan_in mode.
     * Note: PyTorch's actual default uses a=sqrt(5); bit-identical parity
     * requires Issue C (distribution parametrization). The current
     * tensorInitWithDistribution gain (sqrtf(2.0f)) is preserved here. */
    if (storageQ->type != FLOAT32) {
        PRINT_ERROR("linearLayerInit: KAIMING_UNIFORM init currently requires FLOAT32 "
                    "weight storage (Issue C will lift this limit)");
        exit(1);
    }
    distribution_t dist = {
        .type = KAIMING_UNIFORM,
        .params.kaiming = {.gain = 1.4142135623730951f /* sqrtf(2.0f) */, .fanMode = inFeatures},
    };
    initDistribution(paramTensor, &dist);

    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

static parameter_t *allocateLinearBias(size_t outFeatures, quantization_t *storageQ,
                                       quantization_t *gradQ) {
    /* Bias tensor: shape [outFeatures]. Initialized to ZEROS, which initTensor
     * already provides (reserveMemory == calloc), so no fill is needed. */
    shape_t *shape = buildOwnedShape((size_t[]){outFeatures}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    /* No initDistribution(ZEROS) call needed: data is already zero from calloc. */

    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

static void validateLinearInit(linearInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("linearLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->inFeatures == 0) {
        PRINT_ERROR("linearLayerInit: inFeatures must be > 0 (got 0)");
        exit(1);
    }
    if (init->outFeatures == 0) {
        PRINT_ERROR("linearLayerInit: outFeatures must be > 0 (got 0)");
        exit(1);
    }
}

static void validateLayerQuantForLinear(layerQuant_t *lq, bool hasBias) {
    if (lq == NULL) {
        PRINT_ERROR("linearLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->forwardMath == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.forwardMath must be set");
        exit(1);
    }
    if (lq->backwardMath == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.backwardMath must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.weightStorage must be set");
        exit(1);
    }
    if (hasBias && lq->biasStorage == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.biasStorage must be set when bias is enabled");
        exit(1);
    }
}

layer_t *linearLayerInit(linearInit_t *init, layerQuant_t *lq) {
    validateLinearInit(init);
    bool hasBias = resolveLinearBias(init->bias);
    validateLayerQuantForLinear(lq, hasBias);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LINEAR;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    layerCfg->linear = cfg;
    layer->config = layerCfg;

    cfg->weights = allocateLinearWeights(init->inFeatures, init->outFeatures, lq->weightStorage,
                                         lq->backwardMath);
    cfg->bias =
        hasBias ? allocateLinearBias(init->outFeatures, lq->biasStorage, lq->backwardMath) : NULL;

    /* Borrowing: store the four quant pointers verbatim, no copy.
     * Per design spec section 4: collapse to a single math Q for forward and
     * a single math Q for backward (the 4-slot split was empirically never
     * used). */
    cfg->forwardQ = lq->forwardMath;
    cfg->weightGradQ = lq->backwardMath;
    cfg->biasGradQ = lq->backwardMath;
    cfg->propLossQ = lq->backwardMath;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *linearLayerInitOwning(linearInit_t *init, layerQuant_t *lq) {
    validateLinearInit(init);
    bool hasBias = resolveLinearBias(init->bias);
    validateLayerQuantForLinear(lq, hasBias);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LINEAR;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    layerCfg->linear = cfg;
    layer->config = layerCfg;

    /* Allocate parameters using lq->weightStorage / lq->biasStorage as the
     * SOURCE of the tensor's owned quantization.  The allocator helpers (from
     * T12) internally clone via getQLike, so the parameter tensors hold their
     * own quantization_t copies — the caller can immediately drop the lq's
     * weightStorage/biasStorage pointers without breaking the parameters. */
    cfg->weights = allocateLinearWeights(init->inFeatures, init->outFeatures, lq->weightStorage,
                                         lq->backwardMath);
    cfg->bias =
        hasBias ? allocateLinearBias(init->outFeatures, lq->biasStorage, lq->backwardMath) : NULL;

    /* Owning: deep-copy each of the four math quantizations.  Always allocate
     * four separate copies, even if multiple lq slots pointed to the same
     * physical instance — this keeps free* simple (no dedup logic needed
     * for the math slots).  */
    cfg->forwardQ = deepCopyQuantization(lq->forwardMath);
    cfg->weightGradQ = deepCopyQuantization(lq->backwardMath);
    cfg->biasGradQ = deepCopyQuantization(lq->backwardMath);
    cfg->propLossQ = deepCopyQuantization(lq->backwardMath);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeLinearLayer(layer_t *linearLayer) {
    if (linearLayer == NULL) {
        return;
    }
    linearConfig_t *cfg = linearLayer->config->linear;

    /* Tear down weight parameter (param tensor + grad tensor + their data + shape).
     * freeParameter handles the NULL-grad case post-#106. */
    if (cfg->weights != NULL) {
        freeParameter(cfg->weights);
    }
    if (cfg->bias != NULL) {
        freeParameter(cfg->bias);
    }

    /* Owning-variant only — tear down the four quantization_t and their qConfigs.
     * Defensive dedup: the Owning factory (Task 14) will always allocate four
     * separate copies, but if a caller of the Borrowing variant happened to use
     * the same pointer in multiple lq slots, we'd double-free without these
     * checks. Borrowing has ownsQuantizations=false, so this branch is skipped
     * for it; the dedup is purely a defensive guard for future maintenance. */
    if (cfg->ownsQuantizations) {
        if (cfg->forwardQ != NULL) {
            freeReservedMemory(cfg->forwardQ->qConfig);
            freeReservedMemory(cfg->forwardQ);
        }
        if (cfg->weightGradQ != NULL && cfg->weightGradQ != cfg->forwardQ) {
            freeReservedMemory(cfg->weightGradQ->qConfig);
            freeReservedMemory(cfg->weightGradQ);
        }
        if (cfg->biasGradQ != NULL && cfg->biasGradQ != cfg->forwardQ &&
            cfg->biasGradQ != cfg->weightGradQ) {
            freeReservedMemory(cfg->biasGradQ->qConfig);
            freeReservedMemory(cfg->biasGradQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->forwardQ &&
            cfg->propLossQ != cfg->weightGradQ && cfg->propLossQ != cfg->biasGradQ) {
            freeReservedMemory(cfg->propLossQ->qConfig);
            freeReservedMemory(cfg->propLossQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(linearLayer->config);
    freeReservedMemory(linearLayer);
}
