#define SOURCE_FILE "UNIT_TEST_TENSOR_API"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

/* Compile-time contract: initTensor takes shape_t *, quantization_t *,
 * sparsity_t * and returns tensor_t *. No data buffer parameter. */
_Static_assert(_Generic((&initTensor),
                   tensor_t *(*)(shape_t *, quantization_t *, sparsity_t *): 1,
                   default: 0),
               "initTensor must take (shape_t *, quantization_t *, sparsity_t *)");

/* Compile-time contract: initDistribution takes tensor_t * and const distribution_t *. */
_Static_assert(_Generic((&initDistribution),
                   void (*)(tensor_t *, const distribution_t *): 1,
                   default: 0),
               "initDistribution must take (tensor_t *, const distribution_t *)");

/* Compile-time contract: tensorFillFromFloatBuffer takes (tensor_t *, const float *, size_t). */
_Static_assert(_Generic((&tensorFillFromFloatBuffer),
                   void (*)(tensor_t *, const float *, size_t): 1,
                   default: 0),
               "tensorFillFromFloatBuffer must take (tensor_t *, const float *, size_t)");

void setUp() {}
void tearDown() {}

void testTensorInitWithDistribution_Zeros_InitializesProductOfDimsValues() {
    // dims = {2, 5} → product = 10, sum = 7
    // Bug: += gives 7, *= gives 10
    // Fill data with sentinel 42.0f, then ZEROS should overwrite exactly 10 values
    float data[10];
    for (size_t i = 0; i < 10; i++) {
        data[i] = 42.0f;
    }
    size_t dims[] = {2, 5};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(ZEROS, data, dims, 2, &q, NULL, 2, 5);

    // All 10 values should be zero
    float *values = (float *)t->data;
    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.0f, values[i]);
    }
}

void testTensorInitWithDistribution_Ones_InitializesAllValues() {
    // dims = {3, 4} → product = 12, sum = 7
    // Fill data with 0.0f, then ONES should set exactly 12 values to 1.0f
    float data[12];
    memset(data, 0, sizeof(data));
    size_t dims[] = {3, 4};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(ONES, data, dims, 2, &q, NULL, 3, 4);

    float *values = (float *)t->data;
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f, values[i]);
    }
}

void testTensorInitWithDistribution_Normal_InitializesAllValues() {
    // dims = {4, 5} → product = 20, sum = 9
    // If only 9 values are initialized, remaining 11 stay at sentinel
    float data[20];
    float sentinel = -999.0f;
    for (size_t i = 0; i < 20; i++) {
        data[i] = sentinel;
    }
    size_t dims[] = {4, 5};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(NORMAL, data, dims, 2, &q, NULL, 4, 5);

    // With NORMAL distribution, values should NOT be the sentinel
    float *values = (float *)t->data;
    size_t sentinelCount = 0;
    for (size_t i = 0; i < 20; i++) {
        if (values[i] == sentinel) {
            sentinelCount++;
        }
    }
    // All 20 values should have been overwritten — none should remain as sentinel
    TEST_ASSERT_EQUAL_UINT(0, sentinelCount);
}

void testTensorInitWithDistribution_ShapeIsCorrect() {
    // Verify the resulting tensor has the correct shape dimensions
    float data[6] = {0};
    size_t dims[] = {2, 3};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(ZEROS, data, dims, 2, &q, NULL, 2, 3);

    TEST_ASSERT_EQUAL_UINT(2, t->shape->numberOfDimensions);
    size_t numElements = calcNumberOfElementsByTensor(t);
    TEST_ASSERT_EQUAL_UINT(6, numElements);
}

static tensor_t *makeFloatTensorForDistTest(size_t d0, size_t d1) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitFloat(), NULL);
}

void testInitDistribution_Zeros_AllValuesAreZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(3, 4);
    /* Pre-write a sentinel so we can prove ZEROS overwrites. */
    float *vals = (float *)t->data;
    for (size_t i = 0; i < 12; ++i) {
        vals[i] = 42.0f;
    }

    distribution_t d = {.type = ZEROS};
    initDistribution(t, &d);

    for (size_t i = 0; i < 12; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.0f, vals[i]);
    }
    freeTensor(t);
}

void testInitDistribution_Ones_AllValuesAreOne(void) {
    tensor_t *t = makeFloatTensorForDistTest(3, 4);
    distribution_t d = {.type = ONES};
    initDistribution(t, &d);
    float *vals = (float *)t->data;
    for (size_t i = 0; i < 12; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f, vals[i]);
    }
    freeTensor(t);
}

void testInitDistribution_Uniform_AllValuesInRange(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = UNIFORM, .params.uniform = {-0.5f, 0.5f}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        TEST_ASSERT_TRUE(vals[i] >= -0.5f && vals[i] <= 0.5f);
        if (vals[i] != 0.0f) {
            any_nonzero = true;
        }
    }
    TEST_ASSERT_TRUE(any_nonzero);
    freeTensor(t);
}

void testInitDistribution_Normal_NotAllSentinel(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    float *vals = (float *)t->data;
    for (size_t i = 0; i < 20; ++i) {
        vals[i] = -999.0f;
    }
    distribution_t d = {.type = NORMAL, .params.normal = {0.0f, 0.01f}};
    initDistribution(t, &d);
    size_t sentinelCount = 0;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] == -999.0f) {
            sentinelCount++;
        }
    }
    TEST_ASSERT_EQUAL_UINT(0, sentinelCount);
    freeTensor(t);
}

void testInitDistribution_XavierUniform_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = XAVIER_UNIFORM,
                        .params.xavier = {.gain = 1.0f, .fanIn = 4, .fanOut = 5}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(any_nonzero);
    freeTensor(t);
}

void testInitDistribution_XavierNormal_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = XAVIER_NORMAL,
                        .params.xavier = {.gain = 1.0f, .fanIn = 4, .fanOut = 5}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(any_nonzero);
    freeTensor(t);
}

void testInitDistribution_KaimingUniform_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = KAIMING_UNIFORM,
                        .params.kaiming = {.gain = sqrtf(2.0f), .fanMode = 4}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(any_nonzero);
    freeTensor(t);
}

void testInitDistribution_KaimingNormal_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = KAIMING_NORMAL,
                        .params.kaiming = {.gain = sqrtf(2.0f), .fanMode = 4}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(any_nonzero);
    freeTensor(t);
}

static void fillTensorFromStackArrayThatGoesOutOfScope(tensor_t *t) {
    /* Local stack array — exits scope when this function returns. */
    float src[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensorFillFromFloatBuffer(t, src, 6);
    /* `src` goes out of scope on return; tensor must keep its own copy. */
}

void testTensorFillFromFloatBuffer_CopiesValues_SourceCanGoOutOfScope(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);

    fillTensorFromStackArrayThatGoesOutOfScope(t);

    /* Force stack reuse — analogous to issue #93's regression pattern. */
    for (int i = 0; i < 100; ++i) {
        volatile float junk[6] = {(float)i, (float)~i, 0, 0, 0, 0};
        (void)junk;
    }

    float *vals = (float *)t->data;
    float expected[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < 6; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, expected[i], vals[i]);
    }

    freeTensor(t);
}

void testInitTensor_Int32_AllocatesFourBytesPerElement(void) {
    /* Closes the calcNumberOfBytesForData gap surfaced by code-review on Task A:
     * the INT32 arm was missing, which would have made gradInitInt32 (Task D)
     * exit(1) on its first call. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    quantization_t *q = quantizationInitInt32();
    tensor_t *t = initTensor(shape, q, NULL);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_NOT_NULL(t->data);
    /* 6 elements × 4 bytes = 24 bytes; all zero. */
    for (size_t i = 0; i < 24; ++i) {
        TEST_ASSERT_EQUAL_UINT8(0, t->data[i]);
    }
    freeTensor(t);
}

void testInitTensor_AllocatesOwnZeroDataBuffer_FreeTensorIsSafe(void) {
    /* Build shape via reserveMemory so caller doesn't bypass the locality rule. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    quantization_t *q = quantizationInitFloat();

    tensor_t *t = initTensor(shape, q, NULL);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_NOT_NULL(t->data);
    TEST_ASSERT_EQUAL_PTR(shape, t->shape);
    TEST_ASSERT_EQUAL_PTR(q, t->quantization);
    TEST_ASSERT_NULL(t->sparsity);

    /* All bytes of the data buffer must be zero (calloc semantics). */
    size_t bytes = calcBytesPerTensor(t);
    for (size_t i = 0; i < bytes; ++i) {
        TEST_ASSERT_EQUAL_UINT8(0, t->data[i]);
    }

    /* freeTensor must release everything cleanly without external buffers. */
    freeTensor(t);
    /* If we reach here without an abort/segfault, freeTensor is unconditional-safe
     * for an initTensor-allocated tensor. */
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTensorInitWithDistribution_Zeros_InitializesProductOfDimsValues);
    RUN_TEST(testTensorInitWithDistribution_Ones_InitializesAllValues);
    RUN_TEST(testTensorInitWithDistribution_Normal_InitializesAllValues);
    RUN_TEST(testTensorInitWithDistribution_ShapeIsCorrect);
    RUN_TEST(testInitTensor_AllocatesOwnZeroDataBuffer_FreeTensorIsSafe);
    RUN_TEST(testInitTensor_Int32_AllocatesFourBytesPerElement);
    RUN_TEST(testTensorFillFromFloatBuffer_CopiesValues_SourceCanGoOutOfScope);
    RUN_TEST(testInitDistribution_Zeros_AllValuesAreZero);
    RUN_TEST(testInitDistribution_Ones_AllValuesAreOne);
    RUN_TEST(testInitDistribution_Uniform_AllValuesInRange);
    RUN_TEST(testInitDistribution_Normal_NotAllSentinel);
    RUN_TEST(testInitDistribution_XavierUniform_NotAllZero);
    RUN_TEST(testInitDistribution_XavierNormal_NotAllZero);
    RUN_TEST(testInitDistribution_KaimingUniform_NotAllZero);
    RUN_TEST(testInitDistribution_KaimingNormal_NotAllZero);
    return UNITY_END();
}
