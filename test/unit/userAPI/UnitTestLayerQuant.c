#define SOURCE_FILE "UNIT_TEST_LAYER_QUANT"

#include "ArithmeticType.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testLayerQuantInitUniformFloat32DerivesFloatArithmeticAndSharesStorage(void) {
    quantization_t *q = quantizationInitFloat();

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.forwardMath.type);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, lq.forwardMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.weightGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.biasGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.propLossMath.type);

    TEST_ASSERT_EQUAL_PTR(q, lq.outputQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.propLossQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);

    TEST_ASSERT_NULL(lq.weightGradStorage);
    TEST_ASSERT_NULL(lq.biasGradStorage);
}

void testLayerQuantInitUniformSymInt32DerivesSymArithmeticWithProfileRoundingMode(void) {
    quantization_t *q = quantizationInitSymInt32(SR_HALF_AWAY);

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.forwardMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.forwardMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.weightGradMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.weightGradMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.biasGradMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.biasGradMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.propLossMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.propLossMath.roundingMode);

    TEST_ASSERT_EQUAL_PTR(q, lq.outputQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.propLossQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);

    TEST_ASSERT_NULL(lq.weightGradStorage);
    TEST_ASSERT_NULL(lq.biasGradStorage);
}

void testLayerQuantInitUniformAsymBridgesThroughFloatArithmeticButKeepsAsymStorage(void) {
    /* Storage-only dtype (spec D5): arithmetic bridges through ARITH_FLOAT32,
     * but the storage slots keep the real ASYM quantization untouched. */
    quantization_t *q = quantizationInitAsym(8, HALF_AWAY);

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.forwardMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.weightGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.biasGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.propLossMath.type);

    TEST_ASSERT_EQUAL_PTR(q, lq.outputQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.propLossQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);
    TEST_ASSERT_EQUAL_INT(ASYM, lq.outputQ->type);
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

void testDeepCopyQuantizationReturnsNullForNullInput(void) {
    TEST_ASSERT_NULL(deepCopyQuantization(NULL));
}

void testDeepCopyQuantizationFloat32ReturnsFreshAllocationWithNullQConfig(void) {
    quantization_t *src = quantizationInitFloat();
    quantization_t *dst = deepCopyQuantization(src);

    TEST_ASSERT_NOT_NULL(dst);
    TEST_ASSERT_NOT_EQUAL(src, dst); /* fresh allocation */
    TEST_ASSERT_EQUAL_INT(FLOAT32, dst->type);
    TEST_ASSERT_NULL(dst->qConfig);

    freeReservedMemory(dst->qConfig);
    freeReservedMemory(dst);
}

void testDeepCopyQuantizationSymInt32DuplicatesQConfigBytes(void) {
    quantization_t *src = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *dst = deepCopyQuantization(src);

    TEST_ASSERT_NOT_NULL(dst);
    TEST_ASSERT_NOT_EQUAL(src, dst);
    TEST_ASSERT_EQUAL_INT(SYM_INT32, dst->type);
    TEST_ASSERT_NOT_NULL(dst->qConfig);
    TEST_ASSERT_NOT_EQUAL(src->qConfig, dst->qConfig);

    symInt32QConfig_t *srcCfg = (symInt32QConfig_t *)src->qConfig;
    symInt32QConfig_t *dstCfg = (symInt32QConfig_t *)dst->qConfig;
    TEST_ASSERT_EQUAL_MEMORY(srcCfg, dstCfg, sizeof(symInt32QConfig_t));

    freeReservedMemory(dst->qConfig);
    freeReservedMemory(dst);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerQuantInitUniformFloat32DerivesFloatArithmeticAndSharesStorage);
    RUN_TEST(testLayerQuantInitUniformSymInt32DerivesSymArithmeticWithProfileRoundingMode);
    RUN_TEST(testLayerQuantInitUniformAsymBridgesThroughFloatArithmeticButKeepsAsymStorage);
    RUN_TEST(testLayerQuantInitUniformDoesNotMutateTheQuantization);
    RUN_TEST(testDeepCopyQuantizationReturnsNullForNullInput);
    RUN_TEST(testDeepCopyQuantizationFloat32ReturnsFreshAllocationWithNullQConfig);
    RUN_TEST(testDeepCopyQuantizationSymInt32DuplicatesQConfigBytes);
    return UNITY_END();
}
