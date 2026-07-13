#ifndef ODT_SLIDING_WINDOW_1D_H
#define ODT_SLIDING_WINDOW_1D_H

#include <stdlib.h>

#include "Kernel.h"

typedef struct windowGeometry1d {
    size_t inputLength;
    size_t outputLength;
    size_t kernelSize;
    size_t stride;
    size_t dilation;
    size_t padLeft;
    size_t padRight;
} windowGeometry1d_t;

windowGeometry1d_t windowGeometry1dCalc(size_t inputLength, kernel_t const *kernel);

/*! VALID-mode transposed-conv forward output length:
 *    Lout = (inputLength - 1)*stride + dilation*(kernelSize - 1) + outputPadding + 1
 *  Uses kernel->size as the kernel tap count (enforced == weight kernelSize at
 *  layer init, see initConv1dConfigWithWeightsAndBias /
 *  initConv1dTransposedConfigWithWeightsAndBias). SAME/EXPLICIT transpose geometry
 *  is NOT this — it is recovered via windowGeometry1dCalc(outputLength, kernel). */
size_t convTranspose1dOutputLength(size_t inputLength, kernel_t const *kernel,
                                   size_t outputPadding);

typedef struct windowSlice1d {
    size_t firstValidInputIdx;
    size_t firstValidKernelOffset;
    size_t validCount;
} windowSlice1d_t;

windowSlice1d_t windowSlice1dAt(windowGeometry1d_t const *geometry, size_t outputPos);

#endif // ODT_SLIDING_WINDOW_1D_H
