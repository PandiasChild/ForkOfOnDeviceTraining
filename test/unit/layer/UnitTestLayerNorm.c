#define SOURCE_FILE "UNIT_TEST_LAYERNORM"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "Arithmetic.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Build a FLOAT32 tensor of the given rank with the given dims and (optional)
 * row-major data. Caller frees via freeTensor. */
static tensor_t *buildFloatTensorND(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, (float *)vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

/* Build a γ or β parameter_t of shape normalizedShape, filled from `vals`
 * (NULL → leave at calloc-zero). Grad is FLOAT32 zero. Caller frees via
 * freeParameter. */
static parameter_t *buildFloatParam(size_t numDims, const size_t *dimsIn, const float *vals) {
    tensor_t *p = buildFloatTensorND(numDims, dimsIn, vals);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

void testConfigStructIsPopulated(void) {
    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 1.f, 1.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);

    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();

    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);

    size_t numNormDims = cfg.numNormDims;
    float eps = cfg.eps;
    bool gammaOk = (cfg.gamma == gamma);
    bool betaOk = (cfg.beta == beta);
    bool fqOk = (cfg.forwardQ == fq);
    bool bqOk = (cfg.backwardQ == bq);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);

    TEST_ASSERT_EQUAL_UINT(1, numNormDims);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1e-5f, eps);
    TEST_ASSERT_TRUE(gammaOk);
    TEST_ASSERT_TRUE(betaOk);
    TEST_ASSERT_TRUE(fqOk);
    TEST_ASSERT_TRUE(bqOk);
}

void testCalcOutputShapeIsIdentity(void) {
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, NULL);

    size_t *odims = reserveMemory(2 * sizeof(size_t));
    size_t *oorder = reserveMemory(2 * sizeof(size_t));
    shape_t *outShape = reserveMemory(sizeof(shape_t));
    outShape->dimensions = odims;
    outShape->orderOfDimensions = oorder;
    outShape->numberOfDimensions = 0;

    layer_t layer = {.type = LAYERNORM, .config = NULL};
    layerNormCalcOutputShape(&layer, in->shape, outShape);

    size_t nd = outShape->numberOfDimensions;
    size_t d0 = outShape->dimensions[0];
    size_t d1 = outShape->dimensions[1];

    freeReservedMemory(outShape);
    freeReservedMemory(oorder);
    freeReservedMemory(odims);
    freeTensor(in);

    TEST_ASSERT_EQUAL_UINT(2, nd);
    TEST_ASSERT_EQUAL_UINT(2, d0);
    TEST_ASSERT_EQUAL_UINT(3, d1);
}

/* Wrap a layerNormConfig in a stack layer_t. */
static layer_t makeLayerNormLayer(layerNormConfig_t *cfg, layerConfig_t *lcfg) {
    lcfg->layerNorm = cfg;
    layer_t layer = {.type = LAYERNORM, .config = lcfg};
    return layer;
}

/* Derivation for testForwardFloatSingleGroup:
 *   input  = [1, 2, 3, 4]
 *   mean   = 2.5
 *   biased var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
 *   invSigma = 1/sqrt(1.25 + 1e-5) ≈ 0.8944236
 *   n      = [-1.3416354, -0.4472118, +0.4472118, +1.3416354]
 *   gamma  = [2, 3, 4, 5],  beta = [1, -1, 0.5, 0]
 *   y      = gamma*n + beta = [-1.6832708, -2.3416354, +2.2888472, +6.7081771]
 */
void testForwardFloatSingleGroup(void) {
    size_t dims[] = {4};
    tensor_t *in = buildFloatTensorND(1, dims, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *out = buildFloatTensorND(1, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){2.f, 3.f, 4.f, 5.f});
    parameter_t *beta = buildFloatParam(1, ns, (float[]){1.f, -1.f, 0.5f, 0.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float captured[4];
    for (size_t i = 0; i < 4; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {-1.6832708f, -2.3416354f, 2.2888472f, 6.7081771f};
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], captured[i]);
    }
}

/* Derivation for testForwardFloatMultiGroup:
 *   rank-2 [2,3], normalizedShape=[3], D=1 → G=2 rows.
 *   Row0=[0,1,2]: mu=1, var=2/3, sigma≈0.81652,
 *     n=[-1.22472, 0, +1.22472]; gamma=[1,2,3], beta=0
 *     → [-1.22472, 0, +3.67417]
 *   Row1=[10,10,10]: mu=10, var=0 → n=[0,0,0] → [0,0,0]
 *   Also exercises shared gamma across groups.
 */
void testForwardFloatMultiGroup(void) {
    size_t dims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, dims, (float[]){0.f, 1.f, 2.f, 10.f, 10.f, 10.f});
    tensor_t *out = buildFloatTensorND(2, dims, NULL);

    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 2.f, 3.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {-1.22472f, 0.f, 3.67417f, 0.f, 0.f, 0.f};
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected[i], captured[i]);
    }
}

void testForwardFloatTransposedInput(void) {
    /* Physical [3,2] buffer; after transpose logical shape is [2,3] with
     * logical row0=[0,1,2], row1=[10,10,10] (matches the multi-group test). */
    size_t pdims[] = {3, 2};
    tensor_t *in = buildFloatTensorND(2, pdims, (float[]){0.f, 10.f, 1.f, 10.f, 2.f, 10.f});
    transposeTensor(in, 0, 1); /* logical [2,3], physical unchanged */

    /* Output: same logical [2,3], same transposed layout, so dx/y scatter back
     * to matching physical offsets. */
    size_t pdimsOut[] = {3, 2};
    tensor_t *out = buildFloatTensorND(2, pdimsOut, NULL);
    transposeTensor(out, 0, 1);

    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 2.f, 3.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    /* Read back logically via the same offset map the kernel used:
     * logical row0 -> [-1.22472, 0, 3.67417], row1 -> [0,0,0]. */
    float captured[6];
    float *od = (float *)out->data;
    /* physical offsets: logical (r,c) -> phys c*2 + r */
    captured[0] = od[0 * 2 + 0]; /* (0,0) */
    captured[1] = od[1 * 2 + 0]; /* (0,1) */
    captured[2] = od[2 * 2 + 0]; /* (0,2) */
    captured[3] = od[0 * 2 + 1]; /* (1,0) */
    captured[4] = od[1 * 2 + 1]; /* (1,1) */
    captured[5] = od[2 * 2 + 1]; /* (1,2) */

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {-1.22472f, 0.f, 3.67417f, 0.f, 0.f, 0.f};
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected[i], captured[i]);
    }
}

void testVtableForwardFloatSingleGroup(void) {
    size_t dims[] = {4};
    tensor_t *in = buildFloatTensorND(1, dims, (float[]){1.f, -1.f, 1.f, -1.f});
    tensor_t *out = buildFloatTensorND(1, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){2.f, 2.f, 2.f, 2.f});
    parameter_t *beta = buildFloatParam(1, ns, (float[]){1.f, 1.f, 1.f, 1.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerFunctions[LAYERNORM].forward(&layer, in, out);

    float captured[4];
    for (size_t i = 0; i < 4; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {2.99999f, -0.99999f, 2.99999f, -0.99999f};
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], captured[i]);
    }
}

/* x=[1,-1,1,-1]: mu=0, var=1, invSigma=1/sqrt(1+1e-5)=0.9999950, n=±0.9999950.
 * dy=[1,2,3,4]: dbeta=dy; dgamma=dy*n; dn=dy (gamma=1), meanDn=2.5,
 * meanDnN=-0.4999975, dx=invSigma*(dn-meanDn-n*meanDnN). */
void testBackwardFloatSingleGroup(void) {
    size_t dims[] = {4};
    tensor_t *fwdIn = buildFloatTensorND(1, dims, (float[]){1.f, -1.f, 1.f, -1.f});
    tensor_t *loss = buildFloatTensorND(1, dims, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *propLoss = buildFloatTensorND(1, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 1.f, 1.f, 1.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL); /* grads start at zero */
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerFunctions[LAYERNORM].backward(&layer, fwdIn, loss, propLoss);

    float dx[4], dgamma[4], dbeta[4];
    for (size_t i = 0; i < 4; i++) {
        dx[i] = ((float *)propLoss->data)[i];
        dgamma[i] = ((float *)gamma->grad->data)[i];
        dbeta[i] = ((float *)beta->grad->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    float expDx[] = {-1.0f, -0.99999f, 0.99999f, 1.0f};
    float expDgamma[] = {0.999995f, -1.99999f, 2.999985f, -3.99998f};
    float expDbeta[] = {1.f, 2.f, 3.f, 4.f};
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDx[i], dx[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDgamma[i], dgamma[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDbeta[i], dbeta[i]);
    }
}

/* Two groups, uniform dy: dbeta sums over groups ([2,2]); dgamma sums dy*n over
 * groups (N=2 saturation: n=[-0.999995,+0.999995] in both rows regardless of
 * the row values -> [-1.99999,+1.99999]).
 * dx = 0 exactly: uniform dy with gamma=1 makes dn constant per group, so
 * dn-meanDn = 0 and meanDnN = dn*mean(n) = 0 (holds for any N; LayerNorm
 * backward annihilates uniform upstream gradients). */
void testBackwardFloatMultiGroupAccumulatesGradients(void) {
    size_t dims[] = {2, 2};
    tensor_t *fwdIn = buildFloatTensorND(2, dims, (float[]){-1.f, 1.f, 2.f, 4.f});
    tensor_t *loss = buildFloatTensorND(2, dims, (float[]){1.f, 1.f, 1.f, 1.f});
    tensor_t *propLoss = buildFloatTensorND(2, dims, NULL);

    size_t ns[] = {2};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 1.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 2;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss, propLoss);

    float dgamma[2], dbeta[2], dx[4];
    for (size_t i = 0; i < 2; i++) {
        dgamma[i] = ((float *)gamma->grad->data)[i];
        dbeta[i] = ((float *)beta->grad->data)[i];
    }
    for (size_t i = 0; i < 4; i++) {
        dx[i] = ((float *)propLoss->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    float expDgamma[] = {-1.99999f, 1.99999f};
    float expDbeta[] = {2.f, 2.f};
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDgamma[i], dgamma[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDbeta[i], dbeta[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.f, dx[i]);
    }
}

/* Divergent layouts: forwardInput is TRANSPOSED (physical [3,2] buffer
 * [0,1,1,3,2,8], logical [2,3] with (r,c) at buf[c*2+r]), while loss/propLoss
 * are natural [2,3]. Catches routing any offset through the wrong tensor map.
 * Logical fixture: x=[[0,1,2],[1,3,8]], dy=[[1,2,3],[-1,0.5,2]], gamma=[1,2,3],
 * beta=0, normalizedShape=[3], eps=1e-5.
 * Expected values derived one-off via PyTorch float64 autograd:
 *   uv run python -c "... F.layer_norm(x,[3],weight=g,bias=b,eps=1e-5);
 *                     y.backward(dy)"
 *   dx     = [0.4081717, -0.8164905, 0.4083187,
 *             -1.175824e-6, -3.919414e-7, 1.567765e-6]
 *   dgamma = [-0.2056869, -0.1698415, 6.391670]
 *   dbeta  = [0, 2.5, 5]  (= sum_g dy, exact) */
void testBackwardFloatDivergentLayouts(void) {
    size_t pdims[] = {3, 2};
    tensor_t *fwdIn = buildFloatTensorND(2, pdims, (float[]){0.f, 1.f, 1.f, 3.f, 2.f, 8.f});
    transposeTensor(fwdIn, 0, 1); /* logical [2,3], physical unchanged */

    size_t dims[] = {2, 3};
    tensor_t *loss = buildFloatTensorND(2, dims, (float[]){1.f, 2.f, 3.f, -1.f, 0.5f, 2.f});
    tensor_t *propLoss = buildFloatTensorND(2, dims, NULL);

    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 2.f, 3.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss, propLoss);

    float dx[6], dgamma[3], dbeta[3];
    for (size_t i = 0; i < 6; i++) {
        dx[i] = ((float *)propLoss->data)[i]; /* natural flat indices */
    }
    for (size_t i = 0; i < 3; i++) {
        dgamma[i] = ((float *)gamma->grad->data)[i];
        dbeta[i] = ((float *)beta->grad->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    float expDx[] = {0.4081717f,    -0.8164905f,   0.4083187f,
                     -1.175824e-6f, -3.919414e-7f, 1.567765e-6f};
    float expDgamma[] = {-0.2056869f, -0.1698415f, 6.391670f};
    float expDbeta[] = {0.f, 2.5f, 5.f};
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDx[i], dx[i]);
    }
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDgamma[i], dgamma[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDbeta[i], dbeta[i]);
    }
}

void testFactoryBuildsGammaOnesBetaZerosAndForwards(void) {
    size_t normShape[] = {4};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 0.0f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = fwdMath,
                       .backwardMath = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = layerNormLayerInit(&init, &lq);

    bool typeOk = (layer->type == LAYERNORM);
    layerNormConfig_t *cfg = layer->config->layerNorm;
    /* eps==0 → factory substitutes default 1e-5 */
    float capturedEps = cfg->eps;
    /* gamma all ones, beta all zeros */
    float g0 = ((float *)cfg->gamma->param->data)[0];
    float g3 = ((float *)cfg->gamma->param->data)[3];
    float b0 = ((float *)cfg->beta->param->data)[0];
    bool gammaGradFloat = (cfg->gamma->grad->quantization->type == FLOAT32);
    bool betaGradFloat = (cfg->beta->grad->quantization->type == FLOAT32);
    bool fwdMapped = (cfg->forwardQ == fwdMath);
    bool bwdMapped = (cfg->backwardQ == bwdMath);

    /* Forward: x=[1,-1,1,-1], gamma=1, beta=0 -> n ~ [+1,-1,+1,-1] (eps=1e-5). */
    size_t dims[] = {4};
    tensor_t *in = buildFloatTensorND(1, dims, (float[]){1.f, -1.f, 1.f, -1.f});
    tensor_t *out = buildFloatTensorND(1, dims, NULL);
    layerFunctions[LAYERNORM].forward(layer, in, out);
    float y0 = ((float *)out->data)[0];
    float y1 = ((float *)out->data)[1];

    freeTensor(out);
    freeTensor(in);
    freeLayerNormLayer(layer);
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    TEST_ASSERT_TRUE(typeOk);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1e-5f, capturedEps);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.f, g0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.f, g3);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.f, b0);
    TEST_ASSERT_TRUE(gammaGradFloat);
    TEST_ASSERT_TRUE(betaGradFloat);
    TEST_ASSERT_TRUE(fwdMapped);
    TEST_ASSERT_TRUE(bwdMapped);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.999995f, y0);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.999995f, y1);
}

void testFactoryOwningDeepCopiesQuantizations(void) {
    size_t normShape[] = {3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = fwdMath,
                       .backwardMath = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = layerNormLayerInitOwning(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    /* Owning: forwardQ/backwardQ are fresh allocations, NOT the caller's. */
    bool fwdIsCopy = (cfg->forwardQ != fwdMath);
    bool bwdIsCopy = (cfg->backwardQ != bwdMath);
    bool fwdTypeOk = (cfg->forwardQ->type == fwdMath->type);
    bool owns = cfg->ownsQuantizations;

    /* Caller drops its math quant configs IMMEDIATELY — the layer holds copies.
     * Storage quant is cloned into the param tensors via getQLike, so freeing the
     * caller's wStore/bStore here is also safe. */
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    /* Now tear down the layer — frees gamma/beta + the OWNED forwardQ/backwardQ
     * copies. No double-free (the caller's originals are already gone and were
     * never aliased). */
    freeLayerNormLayer(layer);

    TEST_ASSERT_TRUE(fwdIsCopy);
    TEST_ASSERT_TRUE(bwdIsCopy);
    TEST_ASSERT_TRUE(fwdTypeOk);
    TEST_ASSERT_TRUE(owns);
}

void testFactoryBorrowingDoesNotFreeCallerQuantizations(void) {
    size_t normShape[] = {3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = fwdMath,
                       .backwardMath = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = layerNormLayerInit(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    /* Borrowing: verbatim pointers, no ownership. */
    bool fwdVerbatim = (cfg->forwardQ == fwdMath);
    bool bwdVerbatim = (cfg->backwardQ == bwdMath);
    bool owns = cfg->ownsQuantizations;

    /* Free the layer FIRST — it must NOT touch the borrowed math quantizations. */
    freeLayerNormLayer(layer);

    /* Caller frees its own quant configs AFTER. If freeLayerNormLayer had freed
     * forwardQ/backwardQ, these would be double-frees (ASan/valgrind catch them). */
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    TEST_ASSERT_TRUE(fwdVerbatim);
    TEST_ASSERT_TRUE(bwdVerbatim);
    TEST_ASSERT_FALSE(owns);
}

void testFactoryRank2NormShapeAndExplicitEps(void) {
    size_t normShape[] = {2, 3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 2, .eps = 0.1f};

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = q, .backwardMath = q, .weightStorage = q, .biasStorage = q};

    layer_t *layer = layerNormLayerInit(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    size_t numNormDims = cfg->numNormDims;
    float capturedEps = cfg->eps;
    size_t ns0 = cfg->normalizedShape[0];
    size_t ns1 = cfg->normalizedShape[1];
    size_t gammaCount = calcNumberOfElementsByTensor(cfg->gamma->param);

    freeLayerNormLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_UINT(2, numNormDims);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.1f, capturedEps);
    TEST_ASSERT_EQUAL_UINT(2, ns0);
    TEST_ASSERT_EQUAL_UINT(3, ns1);
    TEST_ASSERT_EQUAL_UINT(6, gammaCount);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConfigStructIsPopulated);
    RUN_TEST(testCalcOutputShapeIsIdentity);
    RUN_TEST(testForwardFloatSingleGroup);
    RUN_TEST(testForwardFloatMultiGroup);
    RUN_TEST(testForwardFloatTransposedInput);
    RUN_TEST(testVtableForwardFloatSingleGroup);
    RUN_TEST(testBackwardFloatSingleGroup);
    RUN_TEST(testBackwardFloatMultiGroupAccumulatesGradients);
    RUN_TEST(testBackwardFloatDivergentLayouts);
    RUN_TEST(testFactoryBuildsGammaOnesBetaZerosAndForwards);
    RUN_TEST(testFactoryOwningDeepCopiesQuantizations);
    RUN_TEST(testFactoryBorrowingDoesNotFreeCallerQuantizations);
    RUN_TEST(testFactoryRank2NormShapeAndExplicitEps);
    return UNITY_END();
}
