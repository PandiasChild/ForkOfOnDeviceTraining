#define SOURCE_FILE "CONV1D_API"

#include <stdbool.h>
#include <stdlib.h>

#include "Common.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Distributions.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/* ============================================================================
 * Legacy factory (renamed in Task 4).
 * ========================================================================== */

layer_t *conv1dLayerInitLegacy(parameter_t *weights, parameter_t *bias, kernel_t *kernel,
                               quantization_t *forwardQ, quantization_t *weightGradQ,
                               quantization_t *biasGradQ, quantization_t *propLossQ) {
    layer_t *conv1dLayer = reserveMemory(sizeof(layer_t));
    layerConfig_t *layerConfig = reserveMemory(sizeof(layerConfig_t));
    conv1dConfig_t *conv1dConfig = reserveMemory(sizeof(conv1dConfig_t));

    initConv1dConfigWithWeightsAndBias(conv1dConfig, kernel, weights, bias, 1u, forwardQ,
                                       weightGradQ, biasGradQ, propLossQ);
    conv1dConfig->ownsQuantizations = false;

    conv1dLayer->type = CONV1D;
    layerConfig->conv1d = conv1dConfig;
    conv1dLayer->config = layerConfig;

    return conv1dLayer;
}

void freeConv1dLayerLegacy(layer_t *conv1dLayer) {
    conv1dConfig_t *conv1dConfig = conv1dLayer->config->conv1d;

    freeParameter(conv1dConfig->weights);
    if (conv1dConfig->bias) {
        freeParameter(conv1dConfig->bias);
    }

    freeQuantization(conv1dConfig->forwardQ);
    freeQuantization(conv1dConfig->weightGradQ);
    freeQuantization(conv1dConfig->biasGradQ);
    freeQuantization(conv1dConfig->propLossQ);
    freeReservedMemory(conv1dConfig);
    freeReservedMemory(conv1dLayer);
}

/* ============================================================================
 * New factory API — conv1dInit_t struct + layerQuant_t profile (PR 2).
 * ========================================================================== */

static bool resolveConv1dBias(bias_t b) {
    switch (b) {
    case BIAS_DEFAULT:
        return true; /* PyTorch parity for Conv1d */
    case BIAS_TRUE:
        return true;
    case BIAS_FALSE:
        return false;
    default:
        PRINT_ERROR("conv1dLayerInit: invalid bias value (got %d)", (int)b);
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

static parameter_t *allocateConv1dWeights(size_t outChannels, size_t inChannels, size_t groups,
                                          size_t kernelSize, quantization_t *storageQ,
                                          quantization_t *gradQ) {
    /* Conv1d weight shape: [outChannels, inChannels/groups, kernelSize].
     * Per Conv1d.h:11. */
    if (inChannels % groups != 0) {
        PRINT_ERROR("conv1dLayerInit: inChannels (%zu) must be divisible by groups (%zu)",
                    inChannels, groups);
        exit(1);
    }
    if (outChannels % groups != 0) {
        PRINT_ERROR("conv1dLayerInit: outChannels (%zu) must be divisible by groups (%zu)",
                    outChannels, groups);
        exit(1);
    }
    size_t inPerGroup = inChannels / groups;

    shape_t *shape = buildOwnedShape((size_t[]){outChannels, inPerGroup, kernelSize}, 3);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);

    /* PyTorch-aligned default: Kaiming uniform with fan_in mode.
     * Note: PyTorch's actual default uses a=sqrt(5); bit-identical parity
     * requires Issue C (distribution parametrization). */
    if (storageQ->type != FLOAT32) {
        PRINT_ERROR("conv1dLayerInit: KAIMING_UNIFORM init currently requires FLOAT32 "
                    "weight storage (Issue C will lift this limit)");
        exit(1);
    }
    distribution_t dist = {
        .type = KAIMING_UNIFORM,
        .params.kaiming = {.gain = 1.4142135623730951f /* sqrtf(2.0f) */,
                           .fanMode = inPerGroup * kernelSize},
    };
    initDistribution(paramTensor, &dist);

    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

static parameter_t *allocateConv1dBias(size_t outChannels, quantization_t *storageQ,
                                       quantization_t *gradQ) {
    /* Bias tensor: shape [outChannels]. Zero-initialized via calloc (reserveMemory). */
    shape_t *shape = buildOwnedShape((size_t[]){outChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    /* No initDistribution(ZEROS) — calloc already gave us zeros. */

    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

static void validateConv1dInit(conv1dInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("conv1dLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->inChannels == 0) {
        PRINT_ERROR("conv1dLayerInit: inChannels must be > 0");
        exit(1);
    }
    if (init->outChannels == 0) {
        PRINT_ERROR("conv1dLayerInit: outChannels must be > 0");
        exit(1);
    }
    if (init->kernelSize == 0) {
        PRINT_ERROR("conv1dLayerInit: kernelSize must be > 0");
        exit(1);
    }
}

static void validateLayerQuantForConv1d(layerQuant_t *lq, bool hasBias) {
    if (lq == NULL) {
        PRINT_ERROR("conv1dLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->forwardMath == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.forwardMath must be set");
        exit(1);
    }
    if (lq->backwardMath == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.backwardMath must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.weightStorage must be set");
        exit(1);
    }
    if (hasBias && lq->biasStorage == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.biasStorage must be set when bias is enabled");
        exit(1);
    }
}

/*! Build a heap-owned kernel_t from the conv1dInit_t fields, applying
 *  zero-init defaults (stride=1, dilation=1, padding=VALID). */
static kernel_t *buildConv1dKernel(conv1dInit_t *init) {
    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    size_t stride = init->stride == 0 ? 1 : init->stride;
    size_t dilation = init->dilation == 0 ? 1 : init->dilation;
    if (init->padding == EXPLICIT) {
        /* Symmetric integer padding (PyTorch padding=N). Required to match a
         * reference conv trained with explicit padding under stride>1, where
         * SAME's minimal/asymmetric padding would diverge (issue #177 enc1). */
        initKernelExplicit(kernel, init->kernelSize, init->paddingAmount, dilation, stride);
    } else {
        initKernel(kernel, init->kernelSize, init->padding, dilation, stride);
    }
    return kernel;
}

layer_t *conv1dLayerInit(conv1dInit_t *init, layerQuant_t *lq) {
    validateConv1dInit(init);
    bool hasBias = resolveConv1dBias(init->bias);
    validateLayerQuantForConv1d(lq, hasBias);

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = CONV1D;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    layerCfg->conv1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = buildConv1dKernel(init);
    quantization_t *gradQ = quantizationInitFloat(); /* Conv1d backward is FLOAT32-only */
    cfg->weights = allocateConv1dWeights(init->outChannels, init->inChannels, groups,
                                         init->kernelSize, lq->weightStorage, gradQ);
    cfg->bias = hasBias ? allocateConv1dBias(init->outChannels, lq->biasStorage, gradQ) : NULL;
    freeQuantization(gradQ);
    cfg->groups = groups;
    cfg->forwardQ = lq->forwardMath;
    cfg->weightGradQ = lq->backwardMath;
    cfg->biasGradQ = lq->backwardMath;
    cfg->propLossQ = lq->backwardMath;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *conv1dLayerInitOwning(conv1dInit_t *init, layerQuant_t *lq) {
    validateConv1dInit(init);
    bool hasBias = resolveConv1dBias(init->bias);
    validateLayerQuantForConv1d(lq, hasBias);

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = CONV1D;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    layerCfg->conv1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = buildConv1dKernel(init);
    /* allocateConv1dWeights / allocateConv1dBias internally clone via getQLike,
     * so the parameter tensors own their quantization_t — caller can drop
     * lq->weightStorage / lq->biasStorage immediately. */
    quantization_t *gradQ = quantizationInitFloat(); /* Conv1d backward is FLOAT32-only */
    cfg->weights = allocateConv1dWeights(init->outChannels, init->inChannels, groups,
                                         init->kernelSize, lq->weightStorage, gradQ);
    cfg->bias = hasBias ? allocateConv1dBias(init->outChannels, lq->biasStorage, gradQ) : NULL;
    freeQuantization(gradQ);
    cfg->groups = groups;

    /* Owning: deep-copy each of the four math quantizations. Always four
     * separate copies (no aliasing), keeping freeConv1dLayer simple. */
    cfg->forwardQ = deepCopyQuantization(lq->forwardMath);
    cfg->weightGradQ = deepCopyQuantization(lq->backwardMath);
    cfg->biasGradQ = deepCopyQuantization(lq->backwardMath);
    cfg->propLossQ = deepCopyQuantization(lq->backwardMath);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeConv1dLayer(layer_t *conv1dLayer) {
    if (conv1dLayer == NULL) {
        return;
    }
    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;

    /* Always factory-owned: parameters + kernel. */
    if (cfg->weights != NULL) {
        freeParameter(cfg->weights);
    }
    if (cfg->bias != NULL) {
        freeParameter(cfg->bias);
    }
    freeReservedMemory(cfg->kernel);

    /* Conditionally factory-owned: quantizations (Owning variant only).
     * Defensive dedup: the Owning factory in Task 9 allocates four
     * separate copies (no aliasing), so the dedup is a no-op there but
     * protects against future aliasing. */
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
    freeReservedMemory(conv1dLayer->config);
    freeReservedMemory(conv1dLayer);
}
