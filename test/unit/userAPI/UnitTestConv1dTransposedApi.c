#define SOURCE_FILE "UNIT_TEST_CONV1D_TRANSPOSED_API"

#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testConv1dTransposedLayerInitBorrowingBuildsLayerWithCorrectShape(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 16,
            .outChannels = 8,
            .kernelSize = 5,
            .stride = 5,
            .padding = VALID,
            .bias = BIAS_TRUE,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(CONV1D_TRANSPOSED, layer->type);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    /* Borrowing variant stores pointers verbatim */
    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->weightGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->biasGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    /* Weight shape: [inChannels, outChannels/groups, kernelSize] per Conv1dTransposed.h:12.
     * Note SWAP from Conv1d. */
    TEST_ASSERT_NOT_NULL(cfg->weights);
    tensor_t *weightTensor = cfg->weights->param;
    TEST_ASSERT_NOT_NULL(weightTensor);
    TEST_ASSERT_EQUAL_UINT(3, weightTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(16, weightTensor->shape->dimensions[0]); /* inChannels */
    TEST_ASSERT_EQUAL_UINT(8, weightTensor->shape->dimensions[1]);  /* outChannels / groups */
    TEST_ASSERT_EQUAL_UINT(5, weightTensor->shape->dimensions[2]);  /* kernelSize */

    /* Bias shape: [outChannels] */
    TEST_ASSERT_NOT_NULL(cfg->bias);
    tensor_t *biasTensor = cfg->bias->param;
    TEST_ASSERT_EQUAL_UINT(1, biasTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(8, biasTensor->shape->dimensions[0]);

    /* Kernel populated from init struct */
    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->stride);

    /* groups + outputPadding defaulted to 1 / 0 */
    TEST_ASSERT_EQUAL_UINT(1, cfg->groups);
    TEST_ASSERT_EQUAL_UINT(0, cfg->outputPadding);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitBorrowingBiasFalseLeavesBiasNull(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 4,
            .outChannels = 2,
            .kernelSize = 3,
            .bias = BIAS_FALSE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_NULL(cfg->bias);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitBorrowingOutputPaddingPropagatesToConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 4,
            .outChannels = 2,
            .kernelSize = 3,
            .stride = 2,
            .outputPadding = 1,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_EQUAL_UINT(1, cfg->outputPadding);
    TEST_ASSERT_EQUAL_UINT(2, cfg->kernel->stride);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitOwningDeepCopiesQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInitOwning(
        &(conv1dTransposedInit_t){
            .inChannels = 8,
            .outChannels = 4,
            .kernelSize = 3,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;

    TEST_ASSERT_NOT_EQUAL(q, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->weightGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->biasGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(q->type, cfg->forwardQ->type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitOwningFreesAllAllocationsWithoutLeak(void) {
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitFloat();
        layerQuant_t lq;
        layerQuantInitUniform(&lq, q);

        layer_t *layer = conv1dTransposedLayerInitOwning(
            &(conv1dTransposedInit_t){
                .inChannels = 4,
                .outChannels = 2,
                .kernelSize = 3,
                .bias = BIAS_TRUE,
            },
            &lq);

        freeConv1dTransposedLayer(layer);
        freeQuantization(q);
    }
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConv1dTransposedLayerInitBorrowingBuildsLayerWithCorrectShape);
    RUN_TEST(testConv1dTransposedLayerInitBorrowingBiasFalseLeavesBiasNull);
    RUN_TEST(testConv1dTransposedLayerInitBorrowingOutputPaddingPropagatesToConfig);
    RUN_TEST(testConv1dTransposedLayerInitOwningDeepCopiesQuantizations);
    RUN_TEST(testConv1dTransposedLayerInitOwningFreesAllAllocationsWithoutLeak);
    return UNITY_END();
}
