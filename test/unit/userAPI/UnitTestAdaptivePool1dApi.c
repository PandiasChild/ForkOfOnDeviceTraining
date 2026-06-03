#define SOURCE_FILE "UNIT_TEST_ADAPTIVE_POOL1D_API"

#include "AdaptiveAvgPool1d.h"
#include "AdaptivePool1dApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testBorrowingBuildsLayer(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 8}, &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(ADAPTIVE_AVGPOOL1D, layer->type);

    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);
    TEST_ASSERT_EQUAL_UINT(8, cfg->outputSize);
    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    freeAdaptiveAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testOwningDeepCopiesTwoQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer =
        adaptiveAvgPool1dLayerInitOwning(&(adaptiveAvgPool1dInit_t){.outputSize = 4}, &lq);

    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    TEST_ASSERT_NOT_EQUAL(q, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    /* The two deep copies must be distinct allocations, else freeAdaptive...
       would double-free and the borrowing/owning contract would be broken. */
    TEST_ASSERT_NOT_EQUAL(cfg->forwardQ, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(q->type, cfg->forwardQ->type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeAdaptiveAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testOwningRepeatedBuildFreeNoLeak(void) {
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitFloat();
        layerQuant_t lq;
        layerQuantInitUniform(&lq, q);
        layer_t *layer =
            adaptiveAvgPool1dLayerInitOwning(&(adaptiveAvgPool1dInit_t){.outputSize = 2}, &lq);
        freeAdaptiveAvgPool1dLayer(layer);
        freeQuantization(q);
    }
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testBorrowingBuildsLayer);
    RUN_TEST(testOwningDeepCopiesTwoQuantizations);
    RUN_TEST(testOwningRepeatedBuildFreeNoLeak);
    return UNITY_END();
}
