#define SOURCE_FILE "LAYERNORM_API"

#include <stdlib.h>

#include "LayerNormApi.h"

#include "Common.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

#define LAYERNORM_DEFAULT_EPS 1e-5f

/* Heap-owned shape_t with default orderOfDimensions; ownership transfers to the
 * tensor (freeTensor cascades into freeShape). */
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

/* Constant fill via tensorFillFromFloatBuffer: plain memcpy for FLOAT32; for
 * SYM_INT32 it routes through convertFloatTensorToSymInt32Tensor, which IS
 * the spec's parameter quantization (all-ones -> mantissa 32767, scale
 * 1/32767; all-zeros -> mantissa 0, scale 1.0 via the absMax==0 guard).
 * initDistribution cannot be used here: it is FLOAT32-only by guard, and
 * extending it is Issue-C scope. */
static void fillParamTensorWithConstant(tensor_t *paramTensor, float value) {
    size_t count = calcNumberOfElementsByTensor(paramTensor);
    float *buf = reserveMemory(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        buf[i] = value;
    }
    tensorFillFromFloatBuffer(paramTensor, buf, count);
    freeReservedMemory(buf);
}

/* gamma: shape normalizedShape, init all-ones (FLOAT32: 1.0f each; SYM_INT32:
 * mantissa 32767, scale 1/32767); grad dtype from gradQ (= the profile's
 * backwardMath). */
static parameter_t *allocateLayerNormGamma(const size_t *normalizedShape, size_t numNormDims,
                                           quantization_t *storageQ, quantization_t *gradQ) {
    shape_t *shape = buildOwnedShape(normalizedShape, numNormDims);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    fillParamTensorWithConstant(paramTensor, 1.0f);
    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

/* beta: shape normalizedShape, init all-zeros (FLOAT32: 0.0f each; SYM_INT32:
 * mantissa 0, scale 1.0 — the explicit fill exercises the absMax==0 constant
 * guard instead of relying on calloc zeros + the default scale). */
static parameter_t *allocateLayerNormBeta(const size_t *normalizedShape, size_t numNormDims,
                                          quantization_t *storageQ, quantization_t *gradQ) {
    shape_t *shape = buildOwnedShape(normalizedShape, numNormDims);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    fillParamTensorWithConstant(paramTensor, 0.0f);
    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

static void validateLayerNormInit(layerNormInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("layerNormLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->normalizedShape == NULL) {
        PRINT_ERROR("layerNormLayerInit: normalizedShape is NULL");
        exit(1);
    }
    if (init->numNormDims == 0) {
        PRINT_ERROR("layerNormLayerInit: numNormDims must be > 0 (got 0)");
        exit(1);
    }
    for (size_t d = 0; d < init->numNormDims; d++) {
        if (init->normalizedShape[d] == 0) {
            PRINT_ERROR("layerNormLayerInit: normalizedShape[%zu] must be > 0", d);
            exit(1);
        }
    }
    if (init->eps < 0.0f) {
        PRINT_ERROR("layerNormLayerInit: eps must be >= 0 (got %f)", (double)init->eps);
        exit(1);
    }
}

static void validateLayerQuantForLayerNorm(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("layerNormLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->forwardMath == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.forwardMath must be set");
        exit(1);
    }
    if (lq->backwardMath == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.backwardMath must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.weightStorage must be set (gamma storage)");
        exit(1);
    }
    if (lq->biasStorage == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.biasStorage must be set (beta storage)");
        exit(1);
    }
    if (lq->weightStorage->type != FLOAT32 && lq->weightStorage->type != SYM_INT32) {
        PRINT_ERROR("layerNormLayerInit: gamma storage must be FLOAT32 or SYM_INT32");
        exit(1);
    }
    if (lq->biasStorage->type != FLOAT32 && lq->biasStorage->type != SYM_INT32) {
        PRINT_ERROR("layerNormLayerInit: beta storage must be FLOAT32 or SYM_INT32");
        exit(1);
    }
    /* The kernels read gamma/beta in the forward dtype: a FLOAT32 kernel over
     * SYM mantissas (or vice versa) is silent garbage. Fail at construction. */
    if (lq->weightStorage->type != lq->forwardMath->type ||
        lq->biasStorage->type != lq->forwardMath->type) {
        PRINT_ERROR("layerNormLayerInit: gamma/beta storage type must match forwardMath");
        exit(1);
    }
    if (lq->backwardMath->type != FLOAT32 && lq->backwardMath->type != SYM_INT32) {
        PRINT_ERROR("layerNormLayerInit: backwardMath must be FLOAT32 or SYM_INT32");
        exit(1);
    }
    /* The SYM_INT32 backward recomputes group stats from forwardInput's int32
     * mantissas and reads gamma mantissas; with a FLOAT32 forward those
     * buffers hold float bits — silent garbage. The REVERSE (SYM forwardMath +
     * FLOAT32 backwardMath) stays constructible: it is the inference-only
     * profile, and the runtime backward guard rejects training it. */
    if (lq->backwardMath->type == SYM_INT32 && lq->forwardMath->type != SYM_INT32) {
        PRINT_ERROR("layerNormLayerInit: SYM_INT32 backwardMath requires SYM_INT32 forwardMath");
        exit(1);
    }
}

/* Shared scaffolding for both factories: validate, allocate the layer/config
 * wrappers, allocate gamma(=1)/beta(=0) + grads, and copy normalizedShape into
 * factory-owned memory. Leaves forwardQ/backwardQ/ownsQuantizations for the
 * caller (the specific factory variant) to set. Returns the layer; `*outCfg`
 * receives the config so the variant can finish wiring the math quant slots. */
static layer_t *layerNormLayerInitCommon(layerNormInit_t *init, layerQuant_t *lq,
                                         layerNormConfig_t **outCfg) {
    validateLayerNormInit(init);
    validateLayerQuantForLayerNorm(lq);

    float eps = (init->eps == 0.0f) ? LAYERNORM_DEFAULT_EPS : init->eps;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LAYERNORM;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerNormConfig_t *cfg = reserveMemory(sizeof(layerNormConfig_t));
    layerCfg->layerNorm = cfg;
    layer->config = layerCfg;

    parameter_t *gamma = allocateLayerNormGamma(init->normalizedShape, init->numNormDims,
                                                lq->weightStorage, lq->backwardMath);
    parameter_t *beta = allocateLayerNormBeta(init->normalizedShape, init->numNormDims,
                                              lq->biasStorage, lq->backwardMath);

    /* Factory-owned copy of normalizedShape (caller may free its own array). */
    size_t *normShapeCopy = reserveMemory(init->numNormDims * sizeof(size_t));
    for (size_t d = 0; d < init->numNormDims; d++) {
        normShapeCopy[d] = init->normalizedShape[d];
    }

    /* forwardQ/backwardQ filled by the variant below; pass NULL here. */
    initLayerNormConfig(cfg, gamma, beta, normShapeCopy, init->numNormDims, eps, NULL, NULL);

    *outCfg = cfg;
    return layer;
}

layer_t *layerNormLayerInit(layerNormInit_t *init, layerQuant_t *lq) {
    layerNormConfig_t *cfg;
    layer_t *layer = layerNormLayerInitCommon(init, lq, &cfg);

    /* Borrowing: store the two math quant pointers verbatim. The caller owns
     * forwardMath/backwardMath and frees them; freeLayerNormLayer leaves them
     * untouched (ownsQuantizations=false). */
    cfg->forwardQ = lq->forwardMath;
    cfg->backwardQ = lq->backwardMath;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *layerNormLayerInitOwning(layerNormInit_t *init, layerQuant_t *lq) {
    layerNormConfig_t *cfg;
    layer_t *layer = layerNormLayerInitCommon(init, lq, &cfg);

    /* Owning: deep-copy each math quantization so the caller can drop its
     * forwardMath/backwardMath pointers immediately. freeLayerNormLayer tears
     * the copies down (ownsQuantizations=true). Mirrors linearLayerInitOwning. */
    cfg->forwardQ = deepCopyQuantization(lq->forwardMath);
    cfg->backwardQ = deepCopyQuantization(lq->backwardMath);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeLayerNormLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    layerNormConfig_t *cfg = layer->config->layerNorm;

    /* ALWAYS tear down the factory-allocated parameters (gamma/beta + their grad
     * tensors + data + shapes). The param tensors own their storage quant via
     * getQLike, so storage quant is freed here regardless of ownsQuantizations. */
    if (cfg->gamma != NULL) {
        freeParameter(cfg->gamma);
    }
    if (cfg->beta != NULL) {
        freeParameter(cfg->beta);
    }
    freeReservedMemory(cfg->normalizedShape);

    /* Owning-variant only — tear down the two math quantization_t (qConfig +
     * struct). Dedup guard exactly like freeLinearLayer: free backwardQ only if
     * it is a distinct allocation from forwardQ. The Owning factory always
     * deep-copies into two separate instances, so the guard is a defensive
     * measure; the Borrowing variant has ownsQuantizations=false and skips this
     * branch (caller frees them). */
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
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}
