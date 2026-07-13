#define SOURCE_FILE "UNIT_TEST_LAYER_CONFIG_ACCESS"

#include <stddef.h>

#include "AdaptiveAvgPool1d.h"
#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "DropoutApi.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "MaxPool1d.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "QuantizationApi.h"
#include "Relu.h"
#include "ReluApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Default arithmetic for layers with no consumed arithmetic (Flatten,
 * Quantization — D4) and the shared arithmetic every uniform-FLOAT32-config
 * layer below derives (all uniform-profile fixtures in this file — every
 * fixture but LINEAR, which uses a divergent SYM_INT32 profile to make the
 * accessors discriminating, see testLinearAccessorsMatchConfig — use a
 * single float quantization_t, so forwardMath is always this same value for
 * real arithmetic-bearing layers too). */
static void assertUniformArithmetic(arithmetic_t a) {
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, a.roundingMode);
}

static tensor_t *buildBoolMask(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

/* Discriminating fixture (mirrors #221's testDxWireHonorsProducerPropLossQ in
 * UnitTestCalculateGradsSequential.c): forwardMath (-> outputQ) and
 * backwardMath (-> propLossQ) are two DISTINCT SYM_INT32 objects, so this
 * test only passes if layerOutputQ/backwardWireQ each read their own arm —
 * every other fixture below shares one quantization_t across both arms and
 * would pass even if the LINEAR case of one accessor returned the other's
 * config. */
void testLinearAccessorsMatchConfig(void) {
    /* weightStorage/biasStorage stay FLOAT32: initWeightTensor/initBiasTensor
     * (PyTorch-parity random init, called unconditionally by linearLayerInit)
     * currently require FLOAT32 storage (LAYER_COMMON: requireFloat32) — a
     * pre-existing constraint orthogonal to what this fixture exercises
     * (outputQ/propLossQ pointer routing), so weight/bias storage is kept
     * real but out of the way of the forwardMath/backwardMath divergence. */
    quantization_t *qStorage = quantizationInitFloat();
    quantization_t *qA = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *qB = quantizationInitSymInt32(SR_HALF_AWAY);
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(qA),
        .weightGradMath = arithmeticFromQuantization(qB),
        .biasGradMath = arithmeticFromQuantization(qB),
        .propLossMath = arithmeticFromQuantization(qB),
        .outputQ = qA,
        .propLossQ = qB,
        .weightStorage = qStorage,
        .biasStorage = qStorage,
    };
    layer_t *layer = linearLayerInit(&(linearInit_t){.inFeatures = 1, .outFeatures = 1}, &lq);

    linearConfig_t *cfg = layer->config->linear;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    TEST_ASSERT_EQUAL_PTR(qA, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(qB, backwardWireQ(layer));
    arithmetic_t fm = layerForwardMath(layer);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, fm.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, fm.roundingMode);

    freeLinearLayer(layer);
    freeQuantization(qStorage);
    freeQuantization(qA);
    freeQuantization(qB);
}

void testReluAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = reluLayerInit(&lq);

    reluConfig_t *cfg = layer->config->relu;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeReluLayer(layer);
    freeQuantization(q);
}

void testConv1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1}, &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1, .stride = 1},
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 1, .inputChannels = 1, .inputLength = 1}, &lq);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testAvgPool1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 1}, &lq);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testAdaptiveAvgPool1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, &lq);

    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeAdaptiveAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testSoftmaxAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = softmaxLayerInit(&lq);

    softmaxConfig_t *cfg = layer->config->softmax;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeSoftmaxLayer(layer);
    freeQuantization(q);
}

void testDropoutAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    tensor_t *mask = buildBoolMask(4);
    layer_t *layer = dropoutLayerInit(0.5f, mask, q, q);

    dropoutConfig_t *cfg = layer->config->dropout;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeDropoutLayer(layer);
    freeTensor(mask);
    freeQuantization(q);
}

void testLayerNormAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){1}, .numNormDims = 1}, &lq);

    layerNormConfig_t *cfg = layer->config->layerNorm;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeLayerNormLayer(layer);
    freeQuantization(q);
}

void testQuantizationAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = quantLayerInit(&lq);

    quantizationConfig_t *cfg = layer->config->quantization;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    /* Quantization is a pure conversion node (D4) — no consumed arithmetic. */
    assertUniformArithmetic(layerForwardMath(layer));

    freeQuantLayer(layer);
    freeQuantization(q);
}

void testFlattenAccessorsAreNullAndDefaultArithmetic(void) {
    layer_t *layer = flattenLayerInit();

    TEST_ASSERT_NULL(layerOutputQ(layer));
    TEST_ASSERT_NULL(backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeFlattenLayer(layer);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLinearAccessorsMatchConfig);
    RUN_TEST(testReluAccessorsMatchConfig);
    RUN_TEST(testConv1dAccessorsMatchConfig);
    RUN_TEST(testConv1dTransposedAccessorsMatchConfig);
    RUN_TEST(testMaxPool1dAccessorsMatchConfig);
    RUN_TEST(testAvgPool1dAccessorsMatchConfig);
    RUN_TEST(testAdaptiveAvgPool1dAccessorsMatchConfig);
    RUN_TEST(testSoftmaxAccessorsMatchConfig);
    RUN_TEST(testDropoutAccessorsMatchConfig);
    RUN_TEST(testLayerNormAccessorsMatchConfig);
    RUN_TEST(testQuantizationAccessorsMatchConfig);
    RUN_TEST(testFlattenAccessorsAreNullAndDefaultArithmetic);
    return UNITY_END();
}
