#define SOURCE_FILE "UNIT_TEST_TENSOR_API"

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
    return UNITY_END();
}
