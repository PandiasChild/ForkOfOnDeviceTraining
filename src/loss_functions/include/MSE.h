#ifndef MSE_H
#define MSE_H

#include "LossFunction.h"
#include "Tensor.h"

float mseLossForward(tensor_t *output, tensor_t *label);

void mseLossBackwardFloat(tensor_t *modelOutput, tensor_t *label, tensor_t *result,
                          size_t batchSize, reduction_t reduction);

void mseLossBackwardAsym(tensor_t *modelOutput, tensor_t *label, tensor_t *result, size_t batchSize,
                         reduction_t reduction);

void mseLossBackward(tensor_t *modelOutput, tensor_t *label, tensor_t *result, size_t batchSize,
                     reduction_t reduction);

#endif // MSE_H
