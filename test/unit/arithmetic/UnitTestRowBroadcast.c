#define SOURCE_FILE "UNIT_TEST_ROW_BROADCAST"

#include <stddef.h>
#include <stdlib.h>

#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RowBroadcast.h"
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

void testScaleRows(void) {
    tensor_t *mat =
        buildFloat32TensorND(2, (size_t[]){2, 3}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    tensor_t *scales = buildFloat32TensorND(1, (size_t[]){2}, (float[]){2.0f, -1.0f});
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);

    scaleRowsFloat32(mat, scales, out);

    float expected[] = {2.0f, 4.0f, 6.0f, -4.0f, -5.0f, -6.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)out->data, 6);
    freeTensor(out);
    freeTensor(scales);
    freeTensor(mat);
}

void testScaleRowsInPlace(void) {
    tensor_t *mat = buildFloat32TensorND(2, (size_t[]){2, 2}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    tensor_t *scales = buildFloat32TensorND(1, (size_t[]){2}, (float[]){10.0f, 0.5f});
    scaleRowsFloat32(mat, scales, mat);
    float expected[] = {10.0f, 20.0f, 1.5f, 2.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)mat->data, 4);
    freeTensor(scales);
    freeTensor(mat);
}

void testSubRowBroadcast(void) {
    tensor_t *mat =
        buildFloat32TensorND(2, (size_t[]){2, 3}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    tensor_t *row = buildFloat32TensorND(1, (size_t[]){3}, (float[]){1.0f, 1.0f, 2.0f});
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);

    subRowBroadcastFloat32(mat, row, out);

    float expected[] = {0.0f, 1.0f, 1.0f, 3.0f, 4.0f, 4.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)out->data, 6);
    freeTensor(out);
    freeTensor(row);
    freeTensor(mat);
}

void testScaleRowsRejectsScaleLenMismatch(void) {
    tensor_t *mat = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    tensor_t *scales = buildFloat32TensorND(1, (size_t[]){3}, NULL);
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    ASSERT_EXITS_WITH_FAILURE(scaleRowsFloat32(mat, scales, out));
    freeTensor(out);
    freeTensor(scales);
    freeTensor(mat);
}

void testSubRowBroadcastRejectsRowLenMismatch(void) {
    tensor_t *mat = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    tensor_t *row = buildFloat32TensorND(1, (size_t[]){2}, NULL);
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    ASSERT_EXITS_WITH_FAILURE(subRowBroadcastFloat32(mat, row, out));
    freeTensor(out);
    freeTensor(row);
    freeTensor(mat);
}

void testScaleRowsRejectsPermutedMat(void) {
    /* PR #366 hardening (same gate family as PointwiseFused, #339): the
     * kernels read/write flat row-major and ignore orderOfDimensions -- a
     * permuted operand would be silently misindexed (dims stay physical, so
     * no OOB -- just wrong values). Fail fast instead. */
    tensor_t *mat = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    tensor_t *scales = buildFloat32TensorND(1, (size_t[]){2}, NULL);
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    mat->shape->orderOfDimensions[0] = 1;
    mat->shape->orderOfDimensions[1] = 0;
    ASSERT_EXITS_WITH_FAILURE(scaleRowsFloat32(mat, scales, out));
    freeTensor(out);
    freeTensor(scales);
    freeTensor(mat);
}

void testSubRowBroadcastRejectsPermutedOut(void) {
    tensor_t *mat = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    tensor_t *row = buildFloat32TensorND(1, (size_t[]){3}, NULL);
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){2, 3}, NULL);
    out->shape->orderOfDimensions[0] = 1;
    out->shape->orderOfDimensions[1] = 0;
    ASSERT_EXITS_WITH_FAILURE(subRowBroadcastFloat32(mat, row, out));
    freeTensor(out);
    freeTensor(row);
    freeTensor(mat);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testScaleRows);
    RUN_TEST(testScaleRowsInPlace);
    RUN_TEST(testSubRowBroadcast);
    RUN_TEST(testScaleRowsRejectsScaleLenMismatch);
    RUN_TEST(testSubRowBroadcastRejectsRowLenMismatch);
    RUN_TEST(testScaleRowsRejectsPermutedMat);
    RUN_TEST(testSubRowBroadcastRejectsPermutedOut);
    return UNITY_END();
}
