#ifndef ODT_MAX_POOL_1D_H
#define ODT_MAX_POOL_1D_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct maxPool1dConfig {
    kernel_t *kernel;
    tensor_t *argmaxIndices; // INT32, shape == output shape; pre-allocated by caller
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} maxPool1dConfig_t;

void initMaxPool1dConfig(maxPool1dConfig_t *cfg, kernel_t *kernel, tensor_t *argmaxIndices,
                         quantization_t *forwardQ, quantization_t *propLossQ);

void maxPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output);
void maxPool1dForwardFloat(layer_t *layer, tensor_t *input, tensor_t *output);

void maxPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss);
void maxPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss);

void maxPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_MAX_POOL_1D_H
