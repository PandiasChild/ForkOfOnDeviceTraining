#define SOURCE_FILE "ODT_KERNEL"

#include "Kernel.h"

#include "Common.h"

void initKernel(kernel_t *kernel, size_t size, paddingType_t paddingType, size_t dilation,
                size_t stride) {
    kernel->dilation = dilation;
    kernel->paddingType = paddingType;
    kernel->size = size;
    kernel->stride = stride;
    kernel->padding = 0;
}

void initKernelExplicit(kernel_t *kernel, size_t size, size_t padding, size_t dilation,
                        size_t stride) {
    kernel->dilation = dilation;
    kernel->paddingType = EXPLICIT;
    kernel->size = size;
    kernel->stride = stride;
    kernel->padding = padding;
}
