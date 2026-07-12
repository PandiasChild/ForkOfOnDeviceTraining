#define SOURCE_FILE "AXPBY"

#include <stdlib.h>

#include "Arithmetic.h"
#include "Axpby.h"
#include "Common.h"
#include "DTypes.h"
#include "Tensor.h"

static void requireFloat32(tensor_t *tensor, const char *name) {
    if (tensor->quantization->type != FLOAT32) {
        PRINT_ERROR("axpbyFloat32Tensors: %s must be FLOAT32", name);
        exit(1);
    }
}

/* The shared index algebra (calcElementIndexByIndices, Arithmetic.c) inverts
 * orderOfDimensions by re-applying it, which is only correct for involutions
 * (identity or disjoint axis-pair swaps, order == order^-1). Composed
 * transposes (any cycle of length >= 3, e.g. order [1,2,0]) get mis-addressed
 * -- up to out-of-bounds reads -- so reject them fail-fast until the shared
 * helper is fixed (epic #326 follow-up). */
static void requireInvolutionOrder(tensor_t *tensor, const char *name) {
    size_t numberOfDims = tensor->shape->numberOfDimensions;
    const size_t *order = tensor->shape->orderOfDimensions;
    for (size_t i = 0; i < numberOfDims; i++) {
        if (order[order[i]] != i) {
            PRINT_ERROR("axpbyFloat32Tensors: %s has non-involution orderOfDimensions -- "
                        "unsupported (disjoint axis-pair swaps only; shared index algebra "
                        "limitation)",
                        name);
            exit(1);
        }
    }
}

void axpbyFloat32Tensors(float a, tensor_t *x, float b, tensor_t *y, tensor_t *out) {
    requireFloat32(x, "x");
    requireFloat32(y, "y");
    requireFloat32(out, "out");

    requireInvolutionOrder(x, "x");
    requireInvolutionOrder(y, "y");

    if (!doDimensionsMatch(x, y)) {
        PRINT_ERROR("axpbyFloat32Tensors: x/y dimensions don't match");
        exit(1);
    }
    if (!doDimensionsMatch(x, out)) {
        PRINT_ERROR("axpbyFloat32Tensors: x/out dimensions don't match");
        exit(1);
    }

    size_t numberOfElements = calcNumberOfElementsByTensor(out);
    size_t bytesPerElement = sizeof(float);
    size_t numberOfDims = x->shape->numberOfDimensions;
    size_t *xDims = x->shape->dimensions;
    size_t *yDims = y->shape->dimensions;

    /* Decode the logical (output-order) raw index against each tensor's
     * ORDERED dims -- not its physical storage dims -- then re-encode with
     * the physical dims + orderOfDimensions to get the storage offset. Using
     * the physical dims for the decode step (as Arithmetic.c's shared
     * int32/floatPointWiseArithmetic does) silently degenerates to identity
     * whenever the transposed axes happen to share a size (or one of them is
     * 1), and reads out of bounds otherwise -- see task-3 report. */
    size_t xOrderedDims[numberOfDims];
    size_t yOrderedDims[numberOfDims];
    orderDims(x, xOrderedDims);
    orderDims(y, yOrderedDims);

    for (size_t i = 0; i < numberOfElements; i++) {
        size_t xIndices[numberOfDims];
        calcIndicesByRawIndex(numberOfDims, xOrderedDims, i, xIndices);
        size_t xElementIndex =
            calcElementIndexByIndices(numberOfDims, xDims, xIndices, x->shape->orderOfDimensions);

        size_t yIndices[numberOfDims];
        calcIndicesByRawIndex(numberOfDims, yOrderedDims, i, yIndices);
        size_t yElementIndex =
            calcElementIndexByIndices(numberOfDims, yDims, yIndices, y->shape->orderOfDimensions);

        size_t xByteIndex = xElementIndex * bytesPerElement;
        size_t yByteIndex = yElementIndex * bytesPerElement;

        float xValue = readBytesAsFloat(&x->data[xByteIndex]);
        float yValue = readBytesAsFloat(&y->data[yByteIndex]);

        float result = a * xValue + b * yValue;

        size_t outputByteIndex = i * bytesPerElement;
        writeFloatToByteArray(result, &out->data[outputByteIndex]);
    }
}
