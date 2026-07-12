#define SOURCE_FILE "ROW_BROADCAST"

#include <stdlib.h>

#include "Common.h"
#include "RowBroadcast.h"
#include "Tensor.h"

static void validateRowBroadcast(tensor_t *mat, tensor_t *vec, size_t vecDimIndex, tensor_t *out,
                                 const char *op) {
    if (mat->quantization->type != FLOAT32 || vec->quantization->type != FLOAT32 ||
        out->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: FLOAT32 tensors only", op);
        exit(1);
    }
    if (mat->shape->numberOfDimensions != 2) {
        PRINT_ERROR("%s: mat must be rank 2", op);
        exit(1);
    }
    if (vec->shape->numberOfDimensions != 1 ||
        vec->shape->dimensions[0] != mat->shape->dimensions[vecDimIndex]) {
        PRINT_ERROR("%s: vector length must match mat dim %zu", op, vecDimIndex);
        exit(1);
    }
    if (out->shape->numberOfDimensions != 2 ||
        out->shape->dimensions[0] != mat->shape->dimensions[0] ||
        out->shape->dimensions[1] != mat->shape->dimensions[1]) {
        PRINT_ERROR("%s: out dims must match mat", op);
        exit(1);
    }
}

void scaleRowsFloat32(tensor_t *mat, tensor_t *rowScales, tensor_t *out) {
    validateRowBroadcast(mat, rowScales, 0, out, "scaleRowsFloat32");
    size_t r = mat->shape->dimensions[0];
    size_t c = mat->shape->dimensions[1];
    float *m = (float *)mat->data;
    float *s = (float *)rowScales->data;
    float *o = (float *)out->data;
    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            o[i * c + j] = s[i] * m[i * c + j];
        }
    }
}

void subRowBroadcastFloat32(tensor_t *mat, tensor_t *row, tensor_t *out) {
    validateRowBroadcast(mat, row, 1, out, "subRowBroadcastFloat32");
    size_t r = mat->shape->dimensions[0];
    size_t c = mat->shape->dimensions[1];
    float *m = (float *)mat->data;
    float *v = (float *)row->data;
    float *o = (float *)out->data;
    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            o[i * c + j] = m[i * c + j] - v[j];
        }
    }
}
