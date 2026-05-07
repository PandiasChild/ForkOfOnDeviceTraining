#ifndef ODT_CONV1D_KERNEL_H
#define ODT_CONV1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "Tensor.h"

/*! 1D convolution forward (FLOAT32 only). Correlation, not flipped.
 *
 *  @param input  [batch, in_channels, input_length], FLOAT32
 *  @param weight [out_channels, in_channels/groups, kernel_size], FLOAT32
 *  @param bias   [out_channels] or NULL, FLOAT32
 *  @param kernel kernel_t with size/stride/dilation/paddingType
 *  @param groups must divide in_channels and out_channels
 *  @param output [batch, out_channels, output_length], FLOAT32, pre-allocated
 */
void conv1dKernelFloat32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                         kernel_t const *kernel, size_t groups, tensor_t *output);

#endif // ODT_CONV1D_KERNEL_H
