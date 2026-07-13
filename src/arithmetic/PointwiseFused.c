#define SOURCE_FILE "POINTWISE_FUSED"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "PointwiseFused.h"
#include "Tensor.h"

/* Rounding order is the torch-parity contract: forbid contraction of the
 * separate mul/add/div roundings. lerp's fmaf() is the one deliberate
 * fusion and is explicit. */
#pragma STDC FP_CONTRACT OFF

size_t lerpInstructionCounter = 0;
size_t addcmulInstructionCounter = 0;
size_t addcdivDenomInstructionCounter = 0;

/* Shared operand gate (see header contract). One pass, fail-fast. */
static void requireIdentityOrderSameDimsFloat32(tensor_t **ops, size_t n, const char *op) {
    size_t numberOfDims = ops[0]->shape->numberOfDimensions;
    for (size_t k = 0; k < n; k++) {
        if (ops[k]->quantization->type != FLOAT32) {
            PRINT_ERROR("%s: operand %zu is not FLOAT32", op, k);
            exit(1);
        }
        if (ops[k]->shape->numberOfDimensions != numberOfDims) {
            PRINT_ERROR("%s: operand %zu rank mismatch", op, k);
            exit(1);
        }
        for (size_t d = 0; d < numberOfDims; d++) {
            if (ops[k]->shape->orderOfDimensions[d] != d) {
                PRINT_ERROR("%s: operand %zu is permuted (identity orderOfDimensions "
                            "required; see #339)",
                            op, k);
                exit(1);
            }
            if (ops[k]->shape->dimensions[d] != ops[0]->shape->dimensions[d]) {
                PRINT_ERROR("%s: operand %zu dimension mismatch", op, k);
                exit(1);
            }
        }
    }
}

void lerpFloat32TensorsInplace(tensor_t *a, tensor_t *b, float weight) {
    requireIdentityOrderSameDimsFloat32((tensor_t *[]){a, b}, 2, "lerpFloat32TensorsInplace");
    size_t numberOfElements = calcNumberOfElementsByTensor(a);
    float *aData = (float *)a->data;
    const float *bData = (const float *)b->data;
    for (size_t i = 0; i < numberOfElements; i++) {
        float diff = bData[i] - aData[i];
        aData[i] = fmaf(weight, diff, aData[i]);
#ifdef TRACK_INSTRUCTIONS
        ++lerpInstructionCounter;
#endif
    }
}

void addcmulFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *c, float s) {
    requireIdentityOrderSameDimsFloat32((tensor_t *[]){a, b, c}, 3, "addcmulFloat32TensorsInplace");
    size_t numberOfElements = calcNumberOfElementsByTensor(a);
    float *aData = (float *)a->data;
    const float *bData = (const float *)b->data;
    const float *cData = (const float *)c->data;
    for (size_t i = 0; i < numberOfElements; i++) {
        float t = s * bData[i];
        float t2 = t * cData[i];
        aData[i] = aData[i] + t2;
#ifdef TRACK_INSTRUCTIONS
        ++addcmulInstructionCounter;
#endif
    }
}

void addcdivDenomFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *v, float dScale,
                                       float eps, float s) {
    requireIdentityOrderSameDimsFloat32((tensor_t *[]){a, b, v}, 3,
                                        "addcdivDenomFloat32TensorsInplace");
    size_t numberOfElements = calcNumberOfElementsByTensor(a);
    float *aData = (float *)a->data;
    const float *bData = (const float *)b->data;
    const float *vData = (const float *)v->data;
    for (size_t i = 0; i < numberOfElements; i++) {
        float d = sqrtf(vData[i]);
        d = d / dScale;
        d = d + eps;
        float numer = s * bData[i];
        float quotient = numer / d;
        aData[i] = aData[i] + quotient;
#ifdef TRACK_INSTRUCTIONS
        ++addcdivDenomInstructionCounter;
#endif
    }
}

size_t getLerpInstructionCounter(void) {
    return lerpInstructionCounter;
}

size_t getAddcmulInstructionCounter(void) {
    return addcmulInstructionCounter;
}

size_t getAddcdivDenomInstructionCounter(void) {
    return addcdivDenomInstructionCounter;
}
