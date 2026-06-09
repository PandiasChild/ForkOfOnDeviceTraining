#define SOURCE_FILE "LAYERNORM"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "LayerNorm.h"

#include "Arithmetic.h"
#include "Common.h"
#include "Layer.h"
#include "Quantization.h"
#include "Tensor.h"

void initLayerNormConfig(layerNormConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                         size_t *normalizedShape, size_t numNormDims, float eps,
                         quantization_t *forwardQ, quantization_t *backwardQ) {
    cfg->gamma = gamma;
    cfg->beta = beta;
    cfg->normalizedShape = normalizedShape;
    cfg->numNormDims = numNormDims;
    cfg->eps = eps;
    cfg->forwardQ = forwardQ;
    cfg->backwardQ = backwardQ;
}

/* Compute G (groups) and N (per-group element count) from a logical shape:
 *   N = product of the last numNormDims logical dims
 *   G = total / N
 * Logical dim sizes honor orderOfDimensions via getDimensionsByIndex. */
static void layerNormGroupSizes(tensor_t *t, size_t numNormDims, size_t *outG, size_t *outN) {
    size_t rank = t->shape->numberOfDimensions;
    size_t total = calcNumberOfElementsByTensor(t);
    size_t n = 1;
    for (size_t d = rank - numNormDims; d < rank; d++) {
        n *= getDimensionsByIndex(t, d);
    }
    *outN = n;
    *outG = (n == 0) ? 0 : total / n;
}

/* Fail fast if the input's trailing D logical dims differ from the configured
 * normalizedShape (mirrors Matmul's shape checks). gamma/beta hold exactly
 * prod(normalizedShape) elements, so a mismatched input would otherwise read
 * gamma/beta out of bounds in forward and WRITE the grad tensors out of
 * bounds in backward — silently. */
static void layerNormValidateInputShape(layerNormConfig_t *cfg, tensor_t *input) {
    size_t rank = input->shape->numberOfDimensions;
    if (cfg->numNormDims > rank) {
        PRINT_ERROR("LayerNorm: numNormDims (%zu) exceeds input rank (%zu)", cfg->numNormDims,
                    rank);
        exit(1);
    }
    for (size_t d = 0; d < cfg->numNormDims; d++) {
        size_t inputDim = getDimensionsByIndex(input, rank - cfg->numNormDims + d);
        if (inputDim != cfg->normalizedShape[d]) {
            PRINT_ERROR("LayerNorm: input trailing dim %zu is %zu but normalizedShape[%zu] is %zu",
                        rank - cfg->numNormDims + d, inputDim, d, cfg->normalizedShape[d]);
            exit(1);
        }
    }
}

/* Physical flat offset of logical element (group g, inner j) in tensor t.
 * The logical multi-index is built by decomposing g over the leading
 * (rank - D) dims and j over the last D dims, then mapped through
 * calcElementIndexByIndices (which applies orderOfDimensions). */
static size_t layerNormPhysOffset(tensor_t *t, size_t numNormDims, size_t g, size_t j) {
    size_t rank = t->shape->numberOfDimensions;
    size_t idx[rank];

    /* Decompose j over the last D logical dims (row-major within the group). */
    size_t rem = j;
    for (size_t d = rank; d-- > rank - numNormDims;) {
        size_t dimSize = getDimensionsByIndex(t, d);
        idx[d] = rem % dimSize;
        rem /= dimSize;
    }
    /* Decompose g over the leading (rank - D) logical dims. */
    rem = g;
    for (size_t d = rank - numNormDims; d-- > 0;) {
        size_t dimSize = getDimensionsByIndex(t, d);
        idx[d] = rem % dimSize;
        rem /= dimSize;
    }
    return calcElementIndexByIndices(rank, t->shape->dimensions, idx, t->shape->orderOfDimensions);
}

/* Two-pass per-group stats: pass 1 mean, pass 2 biased variance (÷N, NOT N-1),
 * eps inside the sqrt. Shared by forward and backward so the backward
 * recompute can never desync from the forward definition. */
static void layerNormGroupStats(tensor_t *t, size_t numNormDims, size_t g, size_t N, float eps,
                                float *outMean, float *outInvSigma) {
    float mean = 0.0f;
    for (size_t j = 0; j < N; j++) {
        mean += ((float *)t->data)[layerNormPhysOffset(t, numNormDims, g, j)];
    }
    mean /= (float)N;

    float var = 0.0f;
    for (size_t j = 0; j < N; j++) {
        float d = ((float *)t->data)[layerNormPhysOffset(t, numNormDims, g, j)] - mean;
        var += d * d;
    }
    var /= (float)N; /* BIASED — divide by N, not N-1 */

    *outMean = mean;
    *outInvSigma = 1.0f / sqrtf(var + eps); /* eps INSIDE sqrt */
}

static void layerNormForwardFloat(layerNormConfig_t *cfg, tensor_t *input, tensor_t *output) {
    float *in = (float *)input->data;
    float *out = (float *)output->data;
    float *gamma = (float *)cfg->gamma->param->data;
    float *beta = (float *)cfg->beta->param->data;

    size_t G, N;
    layerNormGroupSizes(input, cfg->numNormDims, &G, &N);

    for (size_t g = 0; g < G; g++) {
        float mean;
        float invSigma;
        layerNormGroupStats(input, cfg->numNormDims, g, N, cfg->eps, &mean, &invSigma);

        for (size_t j = 0; j < N; j++) {
            size_t off = layerNormPhysOffset(input, cfg->numNormDims, g, j);
            float nval = (in[off] - mean) * invSigma;
            size_t outOff = layerNormPhysOffset(output, cfg->numNormDims, g, j);
            out[outOff] = gamma[j] * nval + beta[j];
        }
    }
}

void layerNormForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    layerNormConfig_t *cfg = layer->config->layerNorm;
    layerNormValidateInputShape(cfg, input);
    switch (cfg->forwardQ->type) {
    case FLOAT32:
        layerNormForwardFloat(cfg, input, output);
        break;
    default:
        PRINT_ERROR("LayerNorm forward: quantization type not implemented (FLOAT32 only in PR-1)");
        exit(1);
    }
}

static void layerNormBackwardFloat(layerNormConfig_t *cfg, tensor_t *forwardInput, tensor_t *loss,
                                   tensor_t *propLoss) {
    float *x = (float *)forwardInput->data;
    float *dy = (float *)loss->data;
    float *dx = (float *)propLoss->data;
    float *gamma = (float *)cfg->gamma->param->data;
    float *dgamma = (float *)cfg->gamma->grad->data; /* accumulated += */
    float *dbeta = (float *)cfg->beta->grad->data;   /* accumulated += */

    size_t G, N;
    layerNormGroupSizes(forwardInput, cfg->numNormDims, &G, &N);

    for (size_t g = 0; g < G; g++) {
        /* Recompute stats from forwardInput (no cache). */
        float mean;
        float invSigma;
        layerNormGroupStats(forwardInput, cfg->numNormDims, g, N, cfg->eps, &mean, &invSigma);

        /* Pass over the group: build n, accumulate dgamma/dbeta, and the two
         * reductions meanDn, meanDnN. */
        float meanDn = 0.0f;
        float meanDnN = 0.0f;
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = (x[xoff] - mean) * invSigma;
            float dyv = dy[dyoff];
            dbeta[j] += dyv;         /* SUM over groups */
            dgamma[j] += dyv * nval; /* SUM over groups */
            float dn = dyv * gamma[j];
            meanDn += dn;
            meanDnN += dn * nval;
        }
        meanDn /= (float)N;
        meanDnN /= (float)N;

        /* dx scattered back to the same physical offset its x came from. */
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = (x[xoff] - mean) * invSigma;
            float dn = dy[dyoff] * gamma[j];
            float dxv = invSigma * (dn - meanDn - nval * meanDnN);
            size_t dxoff = layerNormPhysOffset(propLoss, cfg->numNormDims, g, j);
            dx[dxoff] = dxv;
        }
    }
}

void layerNormBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    layerNormConfig_t *cfg = layer->config->layerNorm;
    layerNormValidateInputShape(cfg, forwardInput);
    switch (cfg->backwardQ->type) {
    case FLOAT32:
        layerNormBackwardFloat(cfg, forwardInput, loss, propLoss);
        break;
    default:
        PRINT_ERROR("LayerNorm backward: quantization type not implemented (FLOAT32 only in PR-1)");
        exit(1);
    }
}

void layerNormCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    (void)layer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
