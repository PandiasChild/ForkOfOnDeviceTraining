#define SOURCE_FILE "FLATTEN"

#include <stdlib.h>

#include "Common.h"
#include "Flatten.h"

void flattenForward(layer_t *flattenLayer, tensor_t *input, tensor_t *output) {
  (void)flattenLayer;
  (void)input;
  (void)output;
  PRINT_ERROR("flattenForward not implemented yet");
  exit(1);
}

void flattenBackward(layer_t *flattenLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss) {
  (void)flattenLayer;
  (void)forwardInput;
  (void)loss;
  (void)propLoss;
  PRINT_ERROR("flattenBackward not implemented yet");
  exit(1);
}

void flattenCalcOutputShape(layer_t *flattenLayer, shape_t *inputShape, shape_t *outputShape) {
  (void)flattenLayer;
  (void)inputShape;
  (void)outputShape;
  PRINT_ERROR("flattenCalcOutputShape not implemented yet");
  exit(1);
}
