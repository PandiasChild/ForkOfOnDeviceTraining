#ifndef MSE_H
#define MSE_H

#include "LossFunction.h"
#include "Tensor.h"

float mseLossForward(tensor_t *output, tensor_t *label, reduction_t reduction);

void mseLossBackwardFloat(tensor_t *modelOutput, tensor_t *label, tensor_t *result);

void mseLossBackward(tensor_t *modelOutput, tensor_t *label, tensor_t *result);

/* Per-loss MEAN-reduction scale factor (PyTorch parity).
 *
 * Returns 1 / (totalSamples * numFeaturesPerSample) for MSE, where
 * numFeaturesPerSample is derived from the model output's shape:
 * numElements(modelOutput) / dimensions[0]. The microbatch dimension
 * (B = dimensions[0]) is treated as 1 if shape is degenerate (B=0).
 *
 * Caller must check backwardReduction == REDUCTION_MEAN before invoking. */
float computeMeanScaleMSE(size_t totalSamples, tensor_t *modelOutput);

#endif // MSE_H
