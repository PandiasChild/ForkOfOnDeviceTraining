#ifndef CALCULATE_GRADS_SEQUENTIAL_H
#define CALCULATE_GRADS_SEQUENTIAL_H

#include "TrainingLoopApi.h"

trainingStats_t *calculateGradsSequential(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label);

#endif // CALCULATE_GRADS_SEQUENTIAL_H
