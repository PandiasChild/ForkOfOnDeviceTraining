#ifndef ODT_CONV1D_TRANSPOSED_H
#define ODT_CONV1D_TRANSPOSED_H

#include <stdlib.h>

#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct conv1dTransposedConfig {
    kernel_t *kernel;
    parameter_t *weights; // [Cin, Cout/groups, K]
    parameter_t *bias;    // [Cout] or NULL
    size_t groups;        // must divide Cin and Cout
    size_t outputPadding; // PyTorch parameter; default 0; < max(stride, dilation)
    quantization_t *forwardQ;
    quantization_t *weightGradQ;
    quantization_t *biasGradQ;
    quantization_t *propLossQ;
} conv1dTransposedConfig_t;

void initConv1dTransposedConfigWithWeightsAndBias(
    conv1dTransposedConfig_t *cfg, kernel_t *kernel, parameter_t *weights, parameter_t *bias,
    size_t groups, size_t outputPadding, quantization_t *forwardQ, quantization_t *weightGradQ,
    quantization_t *biasGradQ, quantization_t *propLossQ);

void conv1dTransposedForward(layer_t *layer, tensor_t *input, tensor_t *output);
void conv1dTransposedForwardFloat(layer_t *layer, tensor_t *input, tensor_t *output);

void conv1dTransposedBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                              tensor_t *propLoss);
void conv1dTransposedBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                                   tensor_t *propLoss);

void conv1dTransposedCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_CONV1D_TRANSPOSED_H
