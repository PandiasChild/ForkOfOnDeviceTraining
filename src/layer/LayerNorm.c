#define SOURCE_FILE "LAYERNORM"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "LayerNorm.h"

#include "Arithmetic.h"
#include "Common.h"
#include "Layer.h"
#include "Quantization.h"
#include "Rounding.h"
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
    cfg->ownsQuantizations = false;
}

/* Compute G (groups) and N (per-group element count) from a logical shape:
 *   N = product of the last numNormDims logical dims
 *   G = total / N
 * Logical dim sizes honor orderOfDimensions via getDimensionsByIndex. */
static void layerNormGroupSizes(tensor_t *t, size_t numNormDims, size_t *outG, size_t *outN) {
    size_t rank = t->shape->numberOfDimensions;
    if (numNormDims > rank) {
        PRINT_ERROR("LayerNorm: numNormDims (%zu) exceeds input rank (%zu)", numNormDims, rank);
        exit(1);
    }
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

/* The SYM_INT32 path reinterprets tensor data as int32 mantissas; a FLOAT32
 * buffer read that way is silent garbage, so fail fast. qMaxBits > 16 would
 * break two int32 bounds: the per-group mantissa-sum accumulator and the
 * affine product q*gamma_q <= qMax^2 (spec: int64 is NOT used). NOTE: the
 * guard can only check the config — int16-range mantissas are the SYM_INT32
 * input CONTRACT; upstream ops must deliver requantized mantissas — an open
 * inter-layer requantization question for the quantized-training epic (#137). */
static void layerNormValidateSymTensor(tensor_t *t, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("LayerNorm SYM_INT32: %s must be SYM_INT32", what);
        exit(1);
    }
    symInt32QConfig_t *qc = t->quantization->qConfig;
    if (qc->qMaxBits > 16) {
        PRINT_ERROR("LayerNorm SYM_INT32: %s qMaxBits (%u) exceeds 16", what,
                    (unsigned)qc->qMaxBits);
        exit(1);
    }
}

/* Per-group stats from SYM_INT32 mantissas: int32 accumulator for the mantissa
 * sum (mantissas are int16-range per the qMaxBits<=16 guard, so an int32 sum
 * holds ~65536 terms), then a float pass for the biased variance (/N, NOT N-1)
 * with eps INSIDE the sqrt. SYM twin of layerNormGroupStats (which reads float
 * data directly and remains the single source of truth for the FLOAT32
 * forward+backward); PR-3's SYM backward must recompute through THIS helper. */
static void layerNormGroupStatsSymInt32(tensor_t *t, size_t numNormDims, size_t g, size_t N,
                                        float eps, float inScale, float *outMean,
                                        float *outInvSigma) {
    int32_t *in = (int32_t *)t->data;
    int32_t sumQ = 0;
    for (size_t j = 0; j < N; j++) {
        sumQ += in[layerNormPhysOffset(t, numNormDims, g, j)];
    }
    float mean = inScale * ((float)sumQ / (float)N);

    float var = 0.0f;
    for (size_t j = 0; j < N; j++) {
        float d = (float)in[layerNormPhysOffset(t, numNormDims, g, j)] * inScale - mean;
        var += d * d;
    }
    var /= (float)N; /* BIASED — divide by N, not N-1 */

    *outMean = mean;
    *outInvSigma = 1.0f / sqrtf(var + eps); /* eps INSIDE sqrt */
}

/* Affine y = gamma*n + beta as a SEPARATE quantized elementwise stage, applied
 * in-place over the freshly written normalized mantissas (spec: it destroys
 * the abs-max=qMax / var~1 invariants and has its own requantization + output
 * scale). Lives in LayerNorm.c, NOT arithmetic/: it needs broadcast over
 * groups (gamma_j shared by all G groups) and the layout-agnostic logical
 * index map — the flat equal-count arithmetic elementwise ops can express
 * neither. Scale bookkeeping:
 *   s_y    = s_norm * s_gamma                  (product idiom, mulSymInt32Tensors)
 *   seed_j = round(beta_q,j * s_beta / s_y)    (bias-rescale idiom,
 *            matmulSymInt32TensorsWithBias — a raw beta_q add would silently
 *            drop beta under dynamic per-tensor scales)
 *   y_q    = q * gamma_q,j + seed_j            (the product q*gamma_q <= qMax^2
 *            fits int32, but the rescaled seed is DATA-DEPENDENT and unbounded:
 *            |seed| ~ |beta| * qMax^2 / (absmax_n * absmax_gamma). The guard
 *            below fails fast outside the safe envelope instead of casting an
 *            out-of-range float to int32 (UB). Spec errata: "int32 is safe
 *            throughout" holds only while the rescaled seed leaves qMax^2 of
 *            int32 headroom for the gamma-product, i.e. |beta| <~
 *            absmax_n * absmax_gamma.)
 * gamma/beta are contiguous default-order rank-D tensors -> flat index j. */
static void layerNormAffineSymInt32(layerNormConfig_t *cfg, tensor_t *output, float sNorm) {
    int32_t *out = (int32_t *)output->data;
    int32_t *gammaQ = (int32_t *)cfg->gamma->param->data;
    int32_t *betaQ = (int32_t *)cfg->beta->param->data;
    symInt32QConfig_t *outQC = output->quantization->qConfig;
    symInt32QConfig_t *gammaQC = cfg->gamma->param->quantization->qConfig;
    symInt32QConfig_t *betaQC = cfg->beta->param->quantization->qConfig;

    float sY = sNorm * gammaQC->scale;
    float betaRescale = betaQC->scale / sY;

    size_t G, N;
    layerNormGroupSizes(output, cfg->numNormDims, &G, &N);

    /* Seed-overflow guard: y_q = q*gamma_q + seed must fit int32. The product
     * is bounded by qMax^2; the seed is not — fail fast before UB. */
    const float qMax = powf(2, (float)outQC->qMaxBits - 1) - 1;
    int32_t maxAbsBetaQ = 0;
    for (size_t j = 0; j < N; j++) {
        int32_t a = (betaQ[j] < 0) ? -betaQ[j] : betaQ[j];
        if (a > maxAbsBetaQ) {
            maxAbsBetaQ = a;
        }
    }
    /* !(x <= T) instead of (x > T): also catches NaN (0 * inf when sY underflows). */
    if (!(fabsf((float)maxAbsBetaQ * betaRescale) <= 2147483647.0f - qMax * qMax)) {
        PRINT_ERROR("LayerNorm SYM_INT32 affine: rescaled beta leaves no int32 headroom for the "
                    "gamma-product (|beta| exceeds ~absmax(n)*absmax(gamma) — see #189)");
        exit(1);
    }

    for (size_t g = 0; g < G; g++) {
        for (size_t j = 0; j < N; j++) {
            size_t off = layerNormPhysOffset(output, cfg->numNormDims, g, j);
            int32_t seed = roundByMode((float)betaQ[j] * betaRescale, outQC->roundingMode);
            out[off] = out[off] * gammaQ[j] + seed;
        }
    }
    outQC->scale = sY;
}

/* SYM_INT32 forward (spec 2026-06-05, verified scale-folding scheme):
 * pass 1: per-group float stats + GLOBAL absmax of the normalized values.
 *         Multi-group REQUIRES the per-group 1/sigma_g to hit the DATA — one
 *         per-tensor scale cannot encode G different sigmas; only the global
 *         stretch lives in the scale.
 * pass 2: recompute stats per group (recompute-over-store: no scratch at all,
 *         not even G floats — stateless and MCU-friendly), normalize, stretch
 *         by K = qMax/absmax, round-clamp.
 * Output scale s_norm = 1/K. Per-group reconstructed variance of the output is
 * var/(var+eps) — exactly PyTorch's eps behavior. The affine stage follows
 * separately (Task 3). */
static void layerNormForwardSymInt32(layerNormConfig_t *cfg, tensor_t *input, tensor_t *output) {
    layerNormValidateSymTensor(input, "input");
    layerNormValidateSymTensor(output, "output");
    layerNormValidateSymTensor(cfg->gamma->param, "gamma");
    layerNormValidateSymTensor(cfg->beta->param, "beta");

    symInt32QConfig_t *inQC = input->quantization->qConfig;
    symInt32QConfig_t *outQC = output->quantization->qConfig;
    int32_t *in = (int32_t *)input->data;
    int32_t *out = (int32_t *)output->data;
    float inScale = inQC->scale;
    const float qMax = powf(2, (float)outQC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qMaxBits - 1);

    size_t G, N;
    layerNormGroupSizes(input, cfg->numNormDims, &G, &N);

    if (G == 0 || N == 0) {
        outQC->scale = 1.0f; /* nothing to normalize; neutral scale (cf. #160) */
        return;
    }

    /* Integer range pre-check: if every mantissa is identical (global int32 min
     * == max), every group's centered value is exactly zero in integer space.
     * A float absMax==0.0f check alone is fragile here: gcc's default
     * -ffp-contract=fast may fuse (float)in[off]*inScale - mean into an fma,
     * so the product is not rounded before the subtraction and the cancellation
     * x*s - mean is NOT guaranteed to be exactly 0 even when all mantissas are
     * equal. The integer check is portable and catches this common case. */
    int32_t intMin = in[layerNormPhysOffset(input, cfg->numNormDims, 0, 0)];
    int32_t intMax = intMin;
    bool allConstant = true;
    for (size_t g = 0; g < G; g++) {
        for (size_t j = 0; j < N; j++) {
            int32_t v = in[layerNormPhysOffset(input, cfg->numNormDims, g, j)];
            if (v < intMin) {
                intMin = v;
                allConstant = false;
            }
            if (v > intMax) {
                intMax = v;
                allConstant = false;
            }
        }
    }

    /* Pass 1: global absmax of n = (x_q*s_x - mu)/sigma over all groups.
     * Skipped when allConstant — absMax stays 0.0f, caught by the unified guard
     * below. */
    float absMax = 0.0f;
    if (!allConstant) {
        for (size_t g = 0; g < G; g++) {
            float mean;
            float invSigma;
            layerNormGroupStatsSymInt32(input, cfg->numNormDims, g, N, cfg->eps, inScale, &mean,
                                        &invSigma);
            for (size_t j = 0; j < N; j++) {
                size_t off = layerNormPhysOffset(input, cfg->numNormDims, g, j);
                float a = fabsf(((float)in[off] * inScale - mean) * invSigma);
                if (a > absMax) {
                    absMax = a;
                }
            }
        }
    }

    float sNorm;
    if (allConstant || absMax == 0.0f) {
        /* Constant input (integer fast-path) OR exact float cancellation
         * (groups internally constant, products exactly representable):
         * emit all-zero mantissas with scale 1.0 — mirrors the absMax==0
         * idiom in convertFloatTensorToSymInt32Tensor. */
        sNorm = 1.0f;
        for (size_t g = 0; g < G; g++) {
            for (size_t j = 0; j < N; j++) {
                out[layerNormPhysOffset(output, cfg->numNormDims, g, j)] = 0;
            }
        }
    } else {
        float K = qMax / absMax;
        sNorm = 1.0f / K;

        /* Pass 2: recompute stats, normalize, quantize. */
        for (size_t g = 0; g < G; g++) {
            float mean;
            float invSigma;
            layerNormGroupStatsSymInt32(input, cfg->numNormDims, g, N, cfg->eps, inScale, &mean,
                                        &invSigma);
            for (size_t j = 0; j < N; j++) {
                size_t inOff = layerNormPhysOffset(input, cfg->numNormDims, g, j);
                float n = ((float)in[inOff] * inScale - mean) * invSigma;
                size_t outOff = layerNormPhysOffset(output, cfg->numNormDims, g, j);
                out[outOff] = roundByMode(clamp(n * K, qMin, qMax), outQC->roundingMode);
            }
        }
    }

    layerNormAffineSymInt32(cfg, output, sNorm);
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
    case SYM_INT32:
        layerNormForwardSymInt32(cfg, input, output);
        break;
    default:
        PRINT_ERROR(
            "LayerNorm forward: quantization type not implemented (FLOAT32/SYM_INT32 only)");
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
        /* PR-2 allows SYM_INT32 forward configs; their backward lands in PR-3
         * (#187 propLoss dtype asymmetry). Reading a SYM_INT32 forwardInput /
         * loss / gamma as float* here would be silent garbage — fail fast. */
        if (forwardInput->quantization->type != FLOAT32 || loss->quantization->type != FLOAT32 ||
            cfg->gamma->param->quantization->type != FLOAT32) {
            PRINT_ERROR("LayerNorm backward: FLOAT32 backward requires FLOAT32 tensors "
                        "(SYM_INT32 backward lands in PR-3, see #187)");
            exit(1);
        }
        layerNormBackwardFloat(cfg, forwardInput, loss, propLoss);
        break;
    default:
        PRINT_ERROR(
            "LayerNorm backward: quantization type not implemented (FLOAT32 only until PR-3, see "
            "#187)");
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
