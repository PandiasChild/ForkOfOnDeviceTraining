#ifndef ODT_FLATTEN_H
#define ODT_FLATTEN_H

#include "Layer.h"
#include "Tensor.h"

void flattenForward(layer_t *flattenLayer, tensor_t *input, tensor_t *output);
void flattenBackward(layer_t *flattenLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss);
void flattenCalcOutputShape(layer_t *flattenLayer, shape_t *inputShape, shape_t *outputShape);

#endif
