#define SOURCE_FILE "UNIT_TEST_BERNOULLI"

#include <stdbool.h>
#include <stdlib.h>

#include "Bernoulli.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *buildBoolTensor(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

static size_t countSetBits(tensor_t *mask, size_t n) {
    size_t set = 0;
    for (size_t i = 0; i < n; i++) {
        if (tensorBoolGet(mask, i)) {
            set++;
        }
    }
    return set;
}

void testReferenceRateApproxProbTrue(void) {
    size_t n = 10000;
    tensor_t *mask = buildBoolTensor(n);
    rngSetSeed(12345u);

    bernoulliFillMask(mask, 0.3f);
    size_t set = countSetBits(mask, n);

    freeTensor(mask);

    // ~30% of bits set; tolerate sampling noise on 10k draws.
    TEST_ASSERT_FLOAT_WITHIN(0.03f, 0.3f, (float)set / (float)n);
}

void testReferenceEdgeAllZero(void) {
    size_t n = 256;
    tensor_t *mask = buildBoolTensor(n);
    rngSetSeed(1u);

    bernoulliFillMask(mask, 0.0f);
    size_t set = countSetBits(mask, n);

    freeTensor(mask);
    TEST_ASSERT_EQUAL_UINT(0, set);
}

void testReferenceEdgeAllOne(void) {
    size_t n = 256;
    tensor_t *mask = buildBoolTensor(n);
    rngSetSeed(1u);

    bernoulliFillMask(mask, 1.0f);
    size_t set = countSetBits(mask, n);

    freeTensor(mask);
    TEST_ASSERT_EQUAL_UINT(256, set);
}

void testDeterministicUnderSameSeed(void) {
    size_t n = 512;
    tensor_t *a = buildBoolTensor(n);
    tensor_t *b = buildBoolTensor(n);

    rngSetSeed(999u);
    bernoulliFillMask(a, 0.5f);
    rngSetSeed(999u);
    bernoulliFillMask(b, 0.5f);

    bool identical = true;
    for (size_t i = 0; i < n; i++) {
        if (tensorBoolGet(a, i) != tensorBoolGet(b, i)) {
            identical = false;
            break;
        }
    }

    freeTensor(b);
    freeTensor(a);
    TEST_ASSERT_TRUE(identical);
}

static void stubFillAllSet(tensor_t *mask, float probTrue) {
    (void)probTrue;
    size_t n = calcNumberOfElementsByTensor(mask);
    for (size_t i = 0; i < n; i++) {
        tensorBoolSet(mask, i, true);
    }
}

void testSwapMechanismInstallsAndRestores(void) {
    size_t n = 64;
    tensor_t *mask = buildBoolTensor(n);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubFillAllSet);

    bool active = (bernoulliGetFillMaskFn() == stubFillAllSet);
    bernoulliFillMask(mask, 0.0f); // probTrue 0 but stub ignores it → all set
    size_t setAfterStub = countSetBits(mask, n);

    bernoulliSetFillMaskFn(saved);
    bool restored = (bernoulliGetFillMaskFn() == saved);

    freeTensor(mask);
    TEST_ASSERT_TRUE(active);
    TEST_ASSERT_EQUAL_UINT(64, setAfterStub);
    TEST_ASSERT_TRUE(restored);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReferenceRateApproxProbTrue);
    RUN_TEST(testReferenceEdgeAllZero);
    RUN_TEST(testReferenceEdgeAllOne);
    RUN_TEST(testDeterministicUnderSameSeed);
    RUN_TEST(testSwapMechanismInstallsAndRestores);
    return UNITY_END();
}
