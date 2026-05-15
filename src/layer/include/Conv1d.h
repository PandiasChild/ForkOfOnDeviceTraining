#ifndef ODT_CONV1D_H
#define ODT_CONV1D_H

#include <stdbool.h>
#include <stdlib.h>

#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct conv1dConfig {
    kernel_t *kernel;
    parameter_t *weights; // [Cout, Cin/groups, K]
    parameter_t *bias;    // [Cout] or NULL
    size_t groups;        // must divide Cin and Cout
    quantization_t *forwardQ;
    quantization_t *weightGradQ;
    quantization_t *biasGradQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} conv1dConfig_t;

void initConv1dConfigWithWeightsAndBias(conv1dConfig_t *conv1dConfig, kernel_t *kernel,
                                        parameter_t *weights, parameter_t *bias, size_t groups,
                                        quantization_t *forwardQ, quantization_t *weightGradQ,
                                        quantization_t *biasGradQ, quantization_t *propLossQ);

void conv1dForward(layer_t *layer, tensor_t *input, tensor_t *output);
void conv1dForwardFloat(layer_t *layer, tensor_t *input, tensor_t *output);

void conv1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad, tensor_t *propLoss);
void conv1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                         tensor_t *propLoss);

void conv1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_CONV1D_H
