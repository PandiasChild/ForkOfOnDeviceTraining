#define SOURCE_FILE "UNIT_TEST_CONV1D_API"

#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testConv1dLayerInitBorrowingBuildsLayerWithCorrectShape(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 3,
            .outChannels = 4,
            .kernelSize = 5,
            .padding = VALID,
            .stride = 1,
            .dilation = 1,
            .groups = 1,
            .bias = BIAS_TRUE,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(CONV1D, layer->type);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    /* Borrowing variant stores pointers verbatim */
    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->weightGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->biasGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    /* Weights allocated with shape [outChannels, inChannels/groups, kernelSize] */
    TEST_ASSERT_NOT_NULL(cfg->weights);
    tensor_t *weightTensor = cfg->weights->param;
    TEST_ASSERT_NOT_NULL(weightTensor);
    TEST_ASSERT_EQUAL_UINT(3, weightTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(4, weightTensor->shape->dimensions[0]); /* outChannels */
    TEST_ASSERT_EQUAL_UINT(3, weightTensor->shape->dimensions[1]); /* inChannels / groups */
    TEST_ASSERT_EQUAL_UINT(5, weightTensor->shape->dimensions[2]); /* kernelSize */

    /* Bias allocated with shape [outChannels] */
    TEST_ASSERT_NOT_NULL(cfg->bias);
    tensor_t *biasTensor = cfg->bias->param;
    TEST_ASSERT_NOT_NULL(biasTensor);
    TEST_ASSERT_EQUAL_UINT(1, biasTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(4, biasTensor->shape->dimensions[0]);

    /* Kernel populated from init struct */
    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->stride);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->dilation);

    /* groups defaulted to 1 explicitly via init */
    TEST_ASSERT_EQUAL_UINT(1, cfg->groups);

    freeConv1dLayer(layer);
}

void testConv1dLayerInitBorrowingBiasDefaultResolvesToTrue(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 1,
            .outChannels = 2,
            .kernelSize = 3,
            /* .bias omitted → BIAS_DEFAULT (0) → resolves to true */
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_NOT_NULL(cfg->bias);

    freeConv1dLayer(layer);
}

void testConv1dLayerInitBorrowingBiasFalseLeavesBiasNull(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 1,
            .outChannels = 2,
            .kernelSize = 3,
            .bias = BIAS_FALSE,
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_NULL(cfg->bias);

    freeConv1dLayer(layer);
}

void testConv1dLayerInitBorrowingPaddingDefaultIsValid(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 1,
            .outChannels = 1,
            .kernelSize = 3,
            /* .padding omitted → VALID (enum value 0) */
            /* .stride, .dilation, .groups omitted → 1 (resolved from 0) */
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->stride);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->dilation);
    TEST_ASSERT_EQUAL_UINT(1, cfg->groups);

    freeConv1dLayer(layer);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConv1dLayerInitBorrowingBuildsLayerWithCorrectShape);
    RUN_TEST(testConv1dLayerInitBorrowingBiasDefaultResolvesToTrue);
    RUN_TEST(testConv1dLayerInitBorrowingBiasFalseLeavesBiasNull);
    RUN_TEST(testConv1dLayerInitBorrowingPaddingDefaultIsValid);
    return UNITY_END();
}
