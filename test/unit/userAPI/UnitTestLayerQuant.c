#define SOURCE_FILE "UNIT_TEST_LAYER_QUANT"

#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testLayerQuantInitUniformSetsAllFourSlotsToTheSamePointer(void) {
    quantization_t *q = quantizationInitFloat();

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_PTR(q, lq.forwardMath);
    TEST_ASSERT_EQUAL_PTR(q, lq.backwardMath);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);
}

void testLayerQuantInitUniformDoesNotMutateTheQuantization(void) {
    quantization_t *q = quantizationInitFloat();
    qtype_t typeBefore = q->type;
    void *configBefore = q->qConfig;

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(typeBefore, q->type);
    TEST_ASSERT_EQUAL_PTR(configBefore, q->qConfig);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerQuantInitUniformSetsAllFourSlotsToTheSamePointer);
    RUN_TEST(testLayerQuantInitUniformDoesNotMutateTheQuantization);
    return UNITY_END();
}
