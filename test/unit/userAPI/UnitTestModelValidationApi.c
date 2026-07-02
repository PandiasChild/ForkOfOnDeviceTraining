#define SOURCE_FILE "UNIT_TEST_MODEL_VALIDATION_API"

#include <stdbool.h>

#include "ArithmeticType.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "ModelValidationApi.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* The validator reads ONLY layer->type + layerForwardMath(layer) (which reads
 * config->...->forwardMath), so a parameter-free stub suffices — same shape
 * as buildConv1dStub/buildLayerNormStub below (the deleted linearLayerInitLegacy
 * used to store forwardMath without dereferencing weights/bias). */
static layer_t *buildLinearStub(quantization_t *fq) {
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    cfg->weights = NULL;
    cfg->bias = NULL;
    cfg->forwardMath = arithmeticFromQuantization(fq);
    cfg->weightGradMath = arithmeticFromQuantization(fq);
    cfg->biasGradMath = arithmeticFromQuantization(fq);
    cfg->propLossMath = arithmeticFromQuantization(fq);
    cfg->outputQ = fq;
    cfg->propLossQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->linear = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LINEAR, lc);
    return layer;
}

static void freeLinearStub(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static layer_t *buildLayerNormStub(quantization_t *fq) {
    layerNormConfig_t *cfg = reserveMemory(sizeof(layerNormConfig_t));
    cfg->gamma = NULL;
    cfg->beta = NULL;
    cfg->normalizedShape = NULL;
    cfg->numNormDims = 0;
    cfg->eps = 1e-5f;
    cfg->forwardMath = arithmeticFromQuantization(fq);
    cfg->propLossMath = arithmeticFromQuantization(fq);
    cfg->outputQ = fq;
    cfg->propLossQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->layerNorm = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LAYERNORM, lc);
    return layer;
}

static layer_t *buildQuantStub(quantization_t *fq) {
    quantizationConfig_t *cfg = reserveMemory(sizeof(quantizationConfig_t));
    cfg->outputQ = fq;
    cfg->propLossQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->quantization = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, QUANTIZATION, lc);
    return layer;
}

static void freeLayerNormStub(layer_t *layer) {
    freeReservedMemory(layer->config->layerNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static layer_t *buildConv1dStub(quantization_t *fq) {
    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    cfg->kernel = NULL;
    cfg->weights = NULL;
    cfg->bias = NULL;
    cfg->groups = 1;
    cfg->forwardMath = arithmeticFromQuantization(fq);
    cfg->weightGradMath = arithmeticFromQuantization(fq);
    cfg->biasGradMath = arithmeticFromQuantization(fq);
    cfg->propLossMath = arithmeticFromQuantization(fq);
    cfg->outputQ = fq;
    cfg->propLossQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->conv1d = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, CONV1D, lc);
    return layer;
}

static void freeConv1dStub(layer_t *layer) {
    freeReservedMemory(layer->config->conv1d);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static layer_t *buildConv1dTransposedStub(quantization_t *fq) {
    conv1dTransposedConfig_t *cfg = reserveMemory(sizeof(conv1dTransposedConfig_t));
    cfg->kernel = NULL;
    cfg->weights = NULL;
    cfg->bias = NULL;
    cfg->groups = 1;
    cfg->outputPadding = 0;
    cfg->forwardMath = arithmeticFromQuantization(fq);
    cfg->weightGradMath = arithmeticFromQuantization(fq);
    cfg->biasGradMath = arithmeticFromQuantization(fq);
    cfg->propLossMath = arithmeticFromQuantization(fq);
    cfg->outputQ = fq;
    cfg->propLossQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->conv1dTransposed = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, CONV1D_TRANSPOSED, lc);
    return layer;
}

static void freeConv1dTransposedStub(layer_t *layer) {
    freeReservedMemory(layer->config->conv1dTransposed);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void freeQuantStub(layer_t *layer) {
    freeReservedMemory(layer->config->quantization);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

void testValidatorAcceptsEmptyModel(void) {
    /* modelSize == 0: loop never runs, valid stays true. */
    layer_t *model[1] = {NULL}; /* pointer needed; element never read */
    bool valid = validateModelQuantization(model, 0);
    TEST_ASSERT_TRUE_MESSAGE(valid, "empty model (size 0) should be accepted");
}

void testValidatorRejectsNullModel(void) {
    /* model == NULL: guarded at entry, returns false. */
    bool valid = validateModelQuantization(NULL, 0);
    TEST_ASSERT_FALSE_MESSAGE(valid, "NULL model pointer should be rejected");
}

void testValidatorRejectsNullElementMidArray(void) {
    /* NULL element mid-array: guarded in loop body (model[i]==NULL branch),
     * sets valid=false and continues without crashing. */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildLinearStub(symQ);
    layer_t *model[2] = {linear, NULL};

    bool valid = validateModelQuantization(model, 2);

    freeLinearStub(linear);
    freeQuantization(symQ);

    TEST_ASSERT_FALSE_MESSAGE(valid, "NULL element in model array should be rejected");
}

void testValidatorRejectsSymProducerFollowedByNonQuantLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildLinearStub(symQ);
    layer_t *norm = buildLayerNormStub(symQ);
    layer_t *model[2] = {linear, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeLinearStub(linear);
    freeQuantization(symQ);

    TEST_ASSERT_FALSE_MESSAGE(valid, "Linear-SYM feeding LayerNorm-SYM without a Quant layer "
                                     "violates the int16 inter-layer contract");
}

void testValidatorAcceptsChainWithQuantLayersBetweenProducers(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear0 = buildLinearStub(symQ);
    layer_t *quant1 = buildQuantStub(symQ);
    layer_t *norm = buildLayerNormStub(symQ);
    layer_t *quant2 = buildQuantStub(symQ);
    layer_t *linear1 = buildLinearStub(symQ); /* producer in last position: allowed */
    layer_t *model[5] = {linear0, quant1, norm, quant2, linear1};

    bool valid = validateModelQuantization(model, 5);

    freeLinearStub(linear1);
    freeQuantStub(quant2);
    freeLayerNormStub(norm);
    freeQuantStub(quant1);
    freeLinearStub(linear0);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(valid);
}

void testValidatorAcceptsSymProducerInLastPosition(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildLinearStub(symQ);
    layer_t *model[1] = {linear};

    bool valid = validateModelQuantization(model, 1);

    freeLinearStub(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE_MESSAGE(valid, "producer at the loss boundary is allowed");
}

void testValidatorAcceptsFloatOnlyModel(void) {
    quantization_t *fQ = quantizationInitFloat();
    layer_t *linear = buildLinearStub(fQ);
    layer_t *norm = buildLayerNormStub(fQ);
    layer_t *model[2] = {linear, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeLinearStub(linear);
    freeQuantization(fQ);

    TEST_ASSERT_TRUE(valid);
}

void testValidatorRejectsConv1dSymProducerFollowedByNonQuantLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *conv = buildConv1dStub(symQ);
    layer_t *norm = buildLayerNormStub(symQ);
    layer_t *model[2] = {conv, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeConv1dStub(conv);
    freeQuantization(symQ);

    TEST_ASSERT_FALSE_MESSAGE(valid, "Conv1d-SYM feeding a non-Quant layer violates the "
                                     "int16 inter-layer contract");
}

void testValidatorAcceptsConv1dTransposedSymProducerInLastPosition(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *convT = buildConv1dTransposedStub(symQ);
    layer_t *model[1] = {convT};

    bool valid = validateModelQuantization(model, 1);

    freeConv1dTransposedStub(convT);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE_MESSAGE(valid, "a SYM producer in the last position is allowed");
}

void testValidatorRejectsConv1dTransposedSymProducerFollowedByNonQuantLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *convT = buildConv1dTransposedStub(symQ);
    layer_t *norm = buildLayerNormStub(symQ);
    layer_t *model[2] = {convT, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeConv1dTransposedStub(convT);
    freeQuantization(symQ);

    TEST_ASSERT_FALSE_MESSAGE(valid, "Conv1dTransposed-SYM feeding a non-Quant layer violates "
                                     "the int16 inter-layer contract");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testValidatorAcceptsEmptyModel);
    RUN_TEST(testValidatorRejectsNullModel);
    RUN_TEST(testValidatorRejectsNullElementMidArray);
    RUN_TEST(testValidatorRejectsSymProducerFollowedByNonQuantLayer);
    RUN_TEST(testValidatorAcceptsChainWithQuantLayersBetweenProducers);
    RUN_TEST(testValidatorAcceptsSymProducerInLastPosition);
    RUN_TEST(testValidatorAcceptsFloatOnlyModel);
    RUN_TEST(testValidatorRejectsConv1dSymProducerFollowedByNonQuantLayer);
    RUN_TEST(testValidatorAcceptsConv1dTransposedSymProducerInLastPosition);
    RUN_TEST(testValidatorRejectsConv1dTransposedSymProducerFollowedByNonQuantLayer);
    return UNITY_END();
}
