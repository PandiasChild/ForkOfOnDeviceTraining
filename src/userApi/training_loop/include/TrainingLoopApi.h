#ifndef TRAINING_LOOP_API_H
#define TRAINING_LOOP_API_H

#include "Tensor.h"
#include "Optimizer.h"
#include "LossFunction.h"
#include "DataLoader.h"
#include "InferenceApi.h"

typedef struct trainingStats {
    tensor_t *output;
    float loss;
} trainingStats_t;

typedef struct trainingRunResult {
    float finalTrainLoss;
    float finalEvalLoss;
} trainingRunResult_t;

typedef trainingStats_t *(*calculateGradsFn_t)(layer_t **model, size_t modelSize,
                                               lossType_t lossType, tensor_t *input,
                                               tensor_t *label);

typedef inferenceStats_t *(*inferenceWithLossFn_t)(layer_t **model, size_t numberOfLayers,
                                                   tensor_t *input, tensor_t *label,
                                                   lossType_t lossType);

typedef tensor_t *(*inferenceFn_t)(layer_t **model, size_t numberOfLayers, tensor_t *input);

typedef void (*epochCallbackFn_t)(size_t epoch, float trainLoss, float evalLoss);

void freeTrainingStats(trainingStats_t *trainingStats);

float evaluationBatch(layer_t **model, size_t modelSize, lossType_t lossType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn);

size_t evaluationBatchAccuracy(layer_t **model, size_t modelSize, batch_t *batch,
                               size_t numClasses, inferenceFn_t inferenceFn);

float evaluationEpoch(layer_t **model, size_t modelSize, lossType_t lossType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn);

float evaluationEpochAccuracy(layer_t **model, size_t modelSize, dataLoader_t *dataLoader,
                              size_t numClasses, inferenceFn_t inferenceFn);

trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossType_t lossType,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn,
                                epochCallbackFn_t callback);

#endif // TRAINING_LOOP_API_H
