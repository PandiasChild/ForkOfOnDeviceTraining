#ifndef ODT_KERNEL_H
#define ODT_KERNEL_H

#include <stdlib.h>

typedef enum { VALID, SAME, EXPLICIT } paddingType_t;

typedef struct kernel {
    size_t size;
    paddingType_t paddingType; /*! Default is 0 (VALID) */
    size_t stride;             /*! Default is 1 */
    size_t dilation;           /*! Default is 1 */
    size_t padding; /*! Symmetric pad per side; used only when paddingType == EXPLICIT. */
} kernel_t;

void initKernel(kernel_t *kernel, size_t size, paddingType_t paddingType, size_t dilation,
                size_t stride);

/*! Initialize a kernel with EXPLICIT symmetric integer padding (PyTorch-style:
 *  `padding` zeros are prepended and appended before the windowed op). Sets
 *  paddingType = EXPLICIT and stores the pad amount. Unlike SAME — which picks
 *  the MINIMAL padding that preserves ceil(L/stride) and may split it
 *  asymmetrically — EXPLICIT pads exactly `padding` on each side, so a stride>1
 *  conv can reproduce a PyTorch conv trained with `padding=N` bit-for-bit. */
void initKernelExplicit(kernel_t *kernel, size_t size, size_t padding, size_t dilation,
                        size_t stride);

#endif // ODT_KERNEL_H
