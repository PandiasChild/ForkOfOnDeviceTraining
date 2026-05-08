#ifndef ODT_KERNEL_H
#define ODT_KERNEL_H

#include <stdlib.h>

typedef enum { VALID, SAME } paddingType_t;

typedef struct kernel {
    size_t size;
    paddingType_t paddingType; /*! Default is 0 */
    size_t stride;             /*! Default is 1 */
    size_t dilation;           /*! Default is 1 */
} kernel_t;

void initKernel(kernel_t *kernel, size_t size, paddingType_t paddingType, size_t dilation,
                size_t stride);

#endif // ODT_KERNEL_H
