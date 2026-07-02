#define SOURCE_FILE "UNIT-TEST-ARITHMETIC-TYPE"

#include "ArithmeticType.h"
#include "Quantization.h"
#include "Rounding.h"
#include "unity.h"

static void testFloat32QuantizationDerivesFloatArithmeticWithHalfAway(void) {
    quantization_t q;
    initFloat32Quantization(&q);
    arithmetic_t a = arithmeticFromQuantization(&q);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, a.roundingMode);
}

static void testInt32QuantizationDerivesFloatArithmeticWithHalfAway(void) {
    /* INT32 is a spec-named storage-only dtype (raw 32-bit integer, no
     * qConfig) — bridges through float like the other storage-only types. */
    quantization_t q;
    initInt32Quantization(&q);
    arithmetic_t a = arithmeticFromQuantization(&q);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, a.roundingMode);
}

static void testSymInt32QuantizationDerivesSymArithmeticWithItsRoundingMode(void) {
    symInt32QConfig_t qc;
    initSymInt32QConfig(SR_HALF_AWAY, &qc);
    quantization_t q;
    initSymInt32Quantization(&qc, &q);
    arithmetic_t a = arithmeticFromQuantization(&q);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, a.type);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, a.roundingMode);
}

static void testStorageOnlyDtypesDeriveFloatArithmetic(void) {
    /* ASYM/SYM/BOOL/INT32 are storage formats; compute bridges through float
     * (spec D5, project ASYM design: conversion between native ops). */
    asymQConfig_t aqc;
    initAsymQConfig(8, SR_HALF_AWAY, &aqc);
    quantization_t asymQ;
    initAsymQuantization(&aqc, &asymQ);
    arithmetic_t a = arithmeticFromQuantization(&asymQ);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, a.roundingMode); /* roundingMode carried over */

    symQConfig_t sqc;
    initSymQConfig(8, SR_HALF_AWAY, &sqc);
    quantization_t symQ;
    initSymQuantization(&sqc, &symQ);
    arithmetic_t s = arithmeticFromQuantization(&symQ);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, s.type);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, s.roundingMode); /* roundingMode carried over */

    quantization_t boolQ;
    initBoolQuantization(&boolQ);
    arithmetic_t b = arithmeticFromQuantization(&boolQ);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, b.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, b.roundingMode); /* no qConfig -> default HALF_AWAY */
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testFloat32QuantizationDerivesFloatArithmeticWithHalfAway);
    RUN_TEST(testInt32QuantizationDerivesFloatArithmeticWithHalfAway);
    RUN_TEST(testSymInt32QuantizationDerivesSymArithmeticWithItsRoundingMode);
    RUN_TEST(testStorageOnlyDtypesDeriveFloatArithmetic);

    return UNITY_END();
}
