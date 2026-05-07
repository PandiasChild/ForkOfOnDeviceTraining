#ifndef ODT_CONV_TRANSPOSE_1D_KERNEL_H
#define ODT_CONV_TRANSPOSE_1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "Tensor.h"

/*! Transposed 1D convolution forward (FLOAT32). Scatter form: each input
 *  value is multiplied into a window region of the output.
 *
 *  Phase 1 supports kernel.paddingType == VALID only. SAME for transposed
 *  conv has subtle PyTorch-incompatible semantics and is rejected at
 *  runtime (PRINT_ERROR + exit).
 *
 *  Output length (VALID, no input-side padding) is:
 *    Lout = (Lin - 1) * stride + dilation * (K - 1) + outputPadding + 1
 *
 *  PyTorch convention: outputPadding must satisfy
 *    outputPadding == 0  ||  outputPadding < max(stride, dilation)
 *
 *  @param input          [batch, in_channels, input_length], FLOAT32
 *  @param weight         [in_channels, out_channels/groups, kernel_size], FLOAT32
 *  @param bias           [out_channels] or NULL, FLOAT32
 *  @param kernel         kernel_t (paddingType must be VALID)
 *  @param groups         must divide in_channels and out_channels
 *  @param outputPadding  trailing zeros at output end; default 0
 *  @param output         [batch, out_channels, output_length], FLOAT32, pre-allocated
 */
void convTranspose1dKernelFloat32(tensor_t const *input, tensor_t const *weight,
                                  tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                  size_t outputPadding, tensor_t *output);

#endif // ODT_CONV_TRANSPOSE_1D_KERNEL_H
