#define SOURCE_FILE "UNIT_TEST_QUANTIZATION_API_BITS_CONTROL"

#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testQuantizationInitSymInt32WithBitsSetsBitsAndRoundingMode(void) {
    quantization_t *q = quantizationInitSymInt32WithBits(SR_HALF_AWAY, 12);

    TEST_ASSERT_EQUAL_INT(SYM_INT32, q->type);
    TEST_ASSERT_NOT_NULL(q->qConfig);

    symInt32QConfig_t *cfg = (symInt32QConfig_t *)q->qConfig;
    TEST_ASSERT_EQUAL_UINT8(12, cfg->qMaxBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, cfg->roundingMode);
}

void testQuantizationInitSymInt32WithBitsRejectsMaxBits32(void) {
    /* #202: qMaxBits=32 makes the float32 clamp bound powf(2, 31) - 1 round up to
     * 2^31 > INT32_MAX, an out-of-range (int32_t) cast (UB) in the converter.
     * Reject it at the init chokepoint before any such config can be built. */
    ASSERT_EXITS_WITH_FAILURE(quantizationInitSymInt32WithBits(HALF_AWAY, 32));
}

void testQuantizationInitSymInt32WithBitsAcceptsMaxBits31(void) {
    /* 31 is the largest cast-safe width (powf(2, 30) - 1 is exactly representable);
     * the #227 raw-int/scale=1 regime (widths 17-31) must stay valid. */
    quantization_t *q = quantizationInitSymInt32WithBits(HALF_AWAY, 31);

    symInt32QConfig_t *cfg = (symInt32QConfig_t *)q->qConfig;
    TEST_ASSERT_EQUAL_UINT8(31, cfg->qMaxBits);
}

void testQuantizationInitSymBuildsQConfigWithSpecifiedBitsAndRounding(void) {
    quantization_t *q = quantizationInitSym(4, HALF_AWAY);

    TEST_ASSERT_EQUAL_INT(SYM, q->type);
    TEST_ASSERT_NOT_NULL(q->qConfig);

    symQConfig_t *cfg = (symQConfig_t *)q->qConfig;
    TEST_ASSERT_EQUAL_UINT8(4, cfg->qBits);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, cfg->roundingMode);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testQuantizationInitSymInt32WithBitsSetsBitsAndRoundingMode);
    RUN_TEST(testQuantizationInitSymInt32WithBitsRejectsMaxBits32);
    RUN_TEST(testQuantizationInitSymInt32WithBitsAcceptsMaxBits31);
    RUN_TEST(testQuantizationInitSymBuildsQConfigWithSpecifiedBitsAndRounding);
    return UNITY_END();
}
