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

/* The write path is flat contiguous row-major (see Axpby.h), so out must be
 * an identity-order tensor or every result lands in the wrong slot. Reads are
 * permutation-aware (post-#344 index algebra), so x/y may be arbitrary views. */
static void requireIdentityOrder(tensor_t *tensor, const char *name) {
    size_t numberOfDims = tensor->shape->numberOfDimensions;
    const size_t *order = tensor->shape->orderOfDimensions;
    for (size_t i = 0; i < numberOfDims; i++) {
        if (order[i] != i) {
            PRINT_ERROR("axpbyFloat32Tensors: %s is a permuted view -- the flat row-major "
                        "write requires identity orderOfDimensions",
                        name);
            exit(1);
        }
    }
}

void axpbyFloat32Tensors(float a, tensor_t *x, float b, tensor_t *y, tensor_t *out) {
    requireFloat32(x, "x");
    requireFloat32(y, "y");
    requireFloat32(out, "out");

    requireIdentityOrder(out, "out");

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
     * the physical dims + orderOfDimensions to get the storage offset.
     * Decoding against physical dims would silently degenerate to identity
     * whenever the transposed axes happen to share a size (or one of them is
     * 1), and read out of bounds otherwise. */
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
