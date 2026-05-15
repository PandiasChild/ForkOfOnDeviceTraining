#define SOURCE_FILE "UNIT_TEST_POOL1D_API"

#include "AvgPool1d.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "MaxPool1d.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* ============================================================================
 * MaxPool1d
 * ========================================================================== */

void testMaxPool1dLayerInitBorrowingBuildsLayerWithKernelAndArgmax(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* For K=2 S=2 VALID on inputLength=64, outputLength = (64 - 2)/2 + 1 = 32 */
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2,
            .stride = 2,
            .inputChannels = 16,
            .inputLength = 64,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(MAXPOOL1D, layer->type);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    /* Kernel correctness */
    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(2, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(2, cfg->kernel->stride);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->dilation);

    /* Argmax tensor shape: [1, inputChannels, outputLength] = [1, 16, 32] */
    TEST_ASSERT_NOT_NULL(cfg->argmaxIndices);
    TEST_ASSERT_EQUAL_UINT(3, cfg->argmaxIndices->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(1, cfg->argmaxIndices->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(16, cfg->argmaxIndices->shape->dimensions[1]);
    TEST_ASSERT_EQUAL_UINT(32, cfg->argmaxIndices->shape->dimensions[2]);
    TEST_ASSERT_EQUAL_INT(INT32, cfg->argmaxIndices->quantization->type);

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dLayerInitBorrowingStrideDefaultsToKernelSize(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* stride omitted → defaults to kernelSize per PyTorch convention */
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 4,
            .inputChannels = 1,
            .inputLength = 16,
        },
        &lq);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_EQUAL_UINT(4, cfg->kernel->size);
    TEST_ASSERT_EQUAL_UINT(4, cfg->kernel->stride);
    /* outputLength = (16 - 4)/4 + 1 = 4 */
    TEST_ASSERT_EQUAL_UINT(4, cfg->argmaxIndices->shape->dimensions[2]);

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dLayerInitOwningDeepCopiesTwoQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = maxPool1dLayerInitOwning(
        &(maxPool1dInit_t){
            .kernelSize = 2,
            .stride = 2,
            .inputChannels = 4,
            .inputLength = 8,
        },
        &lq);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_NOT_EQUAL(q, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(q->type, cfg->forwardQ->type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dLayerInitOwningRepeatedBuildFreeNoLeak(void) {
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitFloat();
        layerQuant_t lq;
        layerQuantInitUniform(&lq, q);

        layer_t *layer = maxPool1dLayerInitOwning(
            &(maxPool1dInit_t){
                .kernelSize = 2,
                .stride = 2,
                .inputChannels = 4,
                .inputLength = 8,
            },
            &lq);

        freeMaxPool1dLayer(layer);
        freeQuantization(q);
    }
    TEST_PASS();
}

/* ============================================================================
 * AvgPool1d
 * ========================================================================== */

void testAvgPool1dLayerInitBorrowingBuildsLayerWithKernel(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = avgPool1dLayerInit(
        &(avgPool1dInit_t){
            .kernelSize = 5,
            .stride = 5,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(AVGPOOL1D, layer->type);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->stride);

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testAvgPool1dLayerInitBorrowingStrideDefaultsToKernelSize(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = avgPool1dLayerInit(
        &(avgPool1dInit_t){
            .kernelSize = 3,
            /* stride omitted → kernelSize=3 */
        },
        &lq);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_EQUAL_UINT(3, cfg->kernel->stride);

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testAvgPool1dLayerInitOwningDeepCopiesTwoQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = avgPool1dLayerInitOwning(
        &(avgPool1dInit_t){
            .kernelSize = 2,
            .stride = 2,
        },
        &lq);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_NOT_EQUAL(q, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMaxPool1dLayerInitBorrowingBuildsLayerWithKernelAndArgmax);
    RUN_TEST(testMaxPool1dLayerInitBorrowingStrideDefaultsToKernelSize);
    RUN_TEST(testMaxPool1dLayerInitOwningDeepCopiesTwoQuantizations);
    RUN_TEST(testMaxPool1dLayerInitOwningRepeatedBuildFreeNoLeak);
    RUN_TEST(testAvgPool1dLayerInitBorrowingBuildsLayerWithKernel);
    RUN_TEST(testAvgPool1dLayerInitBorrowingStrideDefaultsToKernelSize);
    RUN_TEST(testAvgPool1dLayerInitOwningDeepCopiesTwoQuantizations);
    return UNITY_END();
}
