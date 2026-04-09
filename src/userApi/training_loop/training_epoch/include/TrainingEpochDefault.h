#ifndef TRAINING_EPOCH_DEFAULT_H
#define TRAINING_EPOCH_DEFAULT_H

#include "TrainingLoopApi.h"

float trainingEpochDefault(layer_t **model, size_t modelSize, lossType_t lossType,
                           dataLoader_t *dataLoader, optimizer_t *optimizer,
                           calculateGradsFn_t calculateGradsFn);

#endif // TRAINING_EPOCH_DEFAULT_H
