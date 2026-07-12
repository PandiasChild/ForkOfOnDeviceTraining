#define SOURCE_FILE "UNIT_TEST_AXPBY"

#include <stddef.h>
#include <stdlib.h>

#include "Axpby.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *buildFloat32TensorND(size_t rank, const size_t *srcDims, const float *fill) {
    size_t *dims = reserveMemory(rank * sizeof(size_t));
    size_t *order = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        dims[i] = srcDims[i];
        order[i] = i;
    }
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = rank;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (fill != NULL) {
        float *d = (float *)t->data;
        size_t n = calcNumberOfElementsByTensor(t);
        for (size_t i = 0; i < n; i++) {
            d[i] = fill[i];
        }
    }
    return t;
}

void testAxpbyBasic(void) {
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){4}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    tensor_t *y = buildFloat32TensorND(1, (size_t[]){4}, (float[]){10.0f, 20.0f, 30.0f, 40.0f});
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){4}, NULL);

    axpbyFloat32Tensors(2.0f, x, 0.5f, y, out);

    float expected[] = {7.0f, 14.0f, 21.0f, 28.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)out->data, 4);
    freeTensor(out);
    freeTensor(y);
    freeTensor(x);
}

void testAxpbyAliasedOutput(void) {
    /* out == x must work: same-index read-then-write. */
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){3}, (float[]){1.0f, 2.0f, 3.0f});
    tensor_t *y = buildFloat32TensorND(1, (size_t[]){3}, (float[]){4.0f, 5.0f, 6.0f});

    axpbyFloat32Tensors(1.0f, x, 1.0f, y, x);

    float expected[] = {5.0f, 7.0f, 9.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)x->data, 3);
    freeTensor(y);
    freeTensor(x);
}

void testAxpbyPermutationAwareRead(void) {
    /* x is a transposed VIEW [3,2] over a [2,3] row-major buffer:
     * storage {1,2,3,4,5,6} viewed transposed reads logically
     * {{1,4},{2,5},{3,6}}. y zeros, a=1, b=0 -> out must be the
     * TRANSPOSED element order, flat. */
    tensor_t *xBase =
        buildFloat32TensorND(2, (size_t[]){2, 3}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    /* Build the transposed view: logical dims [3,2] over the same data.
     * Copy the transpose idiom from src/layer/Linear.c (transposeTensor);
     * if transposeTensor mutates in place, apply it to xBase directly. */
    transposeTensor(xBase, 0, 1);

    tensor_t *y = buildFloat32TensorND(2, (size_t[]){3, 2}, NULL);
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){3, 2}, NULL);

    axpbyFloat32Tensors(1.0f, xBase, 0.0f, y, out);

    float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)out->data, 6);
    freeTensor(out);
    freeTensor(y);
    transposeTensor(xBase, 0, 1); /* untranspose before freeing (Linear.c idiom) */
    freeTensor(xBase);
}

void testAxpbyRejectsDimMismatch(void) {
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){3}, NULL);
    tensor_t *y = buildFloat32TensorND(1, (size_t[]){4}, NULL);
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){3}, NULL);
    ASSERT_EXITS_WITH_FAILURE(axpbyFloat32Tensors(1.0f, x, 1.0f, y, out));
    freeTensor(out);
    freeTensor(y);
    freeTensor(x);
}

void testAxpbyRejectsNonInvolutionPermutation(void) {
    /* Two composed transposes turn orderOfDimensions into a 3-cycle
     * ([0,1,2] -> [1,0,2] -> [1,2,0]): the shared index algebra
     * (calcElementIndexByIndices) only addresses involutions (identity or
     * one axis-pair swap) correctly, so the guard must fail fast instead of
     * reading out of bounds. y/out use x's LOGICAL dims [4,2,3] so this
     * test cannot pass vacuously via the dim-mismatch guard. */
    tensor_t *x = buildFloat32TensorND(3, (size_t[]){2, 3, 4}, NULL);
    transposeTensor(x, 0, 1);
    transposeTensor(x, 1, 2);

    tensor_t *y = buildFloat32TensorND(3, (size_t[]){4, 2, 3}, NULL);
    tensor_t *out = buildFloat32TensorND(3, (size_t[]){4, 2, 3}, NULL);

    ASSERT_EXITS_WITH_FAILURE(axpbyFloat32Tensors(1.0f, x, 1.0f, y, out));

    freeTensor(out);
    freeTensor(y);
    transposeTensor(x, 1, 2); /* unwind the cycle before freeing (Linear.c idiom) */
    transposeTensor(x, 0, 1);
    freeTensor(x);
}

void testAxpbyRejectsNonFloat(void) {
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){3}, NULL);
    size_t *dims = reserveMemory(sizeof(size_t));
    size_t *order = reserveMemory(sizeof(size_t));
    dims[0] = 3;
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initInt32Quantization(q);
    tensor_t *y = initTensor(shape, q, NULL);
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){3}, NULL);
    ASSERT_EXITS_WITH_FAILURE(axpbyFloat32Tensors(1.0f, x, 1.0f, y, out));
    freeTensor(out);
    freeTensor(y);
    freeTensor(x);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAxpbyBasic);
    RUN_TEST(testAxpbyAliasedOutput);
    RUN_TEST(testAxpbyPermutationAwareRead);
    RUN_TEST(testAxpbyRejectsDimMismatch);
    RUN_TEST(testAxpbyRejectsNonInvolutionPermutation);
    RUN_TEST(testAxpbyRejectsNonFloat);
    return UNITY_END();
}
