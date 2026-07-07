#ifndef ODT_CONV_TRANSPOSE_1D_KERNEL_H
#define ODT_CONV_TRANSPOSE_1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "Tensor.h"

/*! Transposed 1D convolution forward (FLOAT32). Scatter form: each input
 *  value is multiplied into a window region of the output.
 *
 *  Supports both VALID and SAME paddingType. SAME is used by Conv1d-backward
 *  via the adjoint identity (dL/dx of Conv1d(SAME) = convTranspose1d(lossGrad,
 *  W, NULL, kernel-with-SAME, groups, 0, propLoss) where propLoss takes the
 *  shape of the original Conv1d's input). In SAME mode, padLeft/padRight are
 *  recovered from windowGeometry1dCalc(outputLength, kernel) — that is, the
 *  forward Conv1d's geometry on the adjoint output shape.
 *
 *  VALID output length:
 *    Lout = (Lin - 1) * stride + dilation * (K - 1) + outputPadding + 1
 *  SAME output length: caller-determined; the kernel asserts that
 *    windowGeometry1dCalc(Lout, kernel).outputLength == Lin.
 *
 *  PyTorch convention: outputPadding must satisfy
 *    outputPadding == 0  ||  outputPadding < max(stride, dilation)
 *  outputPadding must be 0 in SAME mode.
 *
 *  @param input          [batch, in_channels, input_length], FLOAT32
 *  @param weight         [in_channels, out_channels/groups, kernel_size], FLOAT32
 *  @param bias           [out_channels] or NULL, FLOAT32
 *  @param kernel         kernel_t (paddingType VALID or SAME)
 *  @param groups         must divide in_channels and out_channels
 *  @param outputPadding  trailing zeros at output end (VALID only); 0 for SAME
 *  @param output         [batch, out_channels, output_length], FLOAT32, pre-allocated
 */
void convTranspose1dKernelFloat32(tensor_t const *input, tensor_t const *weight,
                                  tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                  size_t outputPadding, tensor_t *output);

/*! Transposed 1D convolution forward (SYM_INT32). Integer scatter sibling of
 *  convTranspose1dKernelFloat32: int32 accumulator, mulInt32s, per-output-channel
 *  bias seed (refold) added in a separate pass. Output mantissas are raw
 *  accumulator range at scale s_in*s_w. Same VALID/SAME/EXPLICIT geometry; used
 *  as Conv1d's dx adjoint with bias==NULL (SAME/EXPLICIT branch), and as
 *  Conv1dTransposed's forward in PR3.
 *
 *  @param input  [batch, in_channels, input_length], SYM_INT32
 *  @param weight [in_channels, out_channels/groups, kernel_size], SYM_INT32
 *  @param bias   [out_channels] or NULL, SYM_INT32
 */
void convTranspose1dKernelSymInt32(tensor_t const *input, tensor_t const *weight,
                                   tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                   size_t outputPadding, tensor_t *output);

#endif // ODT_CONV_TRANSPOSE_1D_KERNEL_H
