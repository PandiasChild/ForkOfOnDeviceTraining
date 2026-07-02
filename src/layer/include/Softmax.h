#ifndef ENV5_RUNTIME_SOFTMAX_H
#define ENV5_RUNTIME_SOFTMAX_H

#include <stdbool.h>

#include "ArithmeticType.h"
#include "Layer.h"

typedef struct softmaxConfig {
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} softmaxConfig_t;

void softmaxInitConfig(softmaxConfig_t *softmaxConfig, quantization_t *forwardQ,
                       quantization_t *backwardQ);

void softmaxInitLayer(layerConfig_t *softmaxConfig, layer_t *softmaxLayer);

void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output);

void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss);

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_SOFTMAX_H
