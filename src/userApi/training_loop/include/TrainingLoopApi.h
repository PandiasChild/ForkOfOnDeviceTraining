#ifndef TRAINING_LOOP_API_H
#define TRAINING_LOOP_API_H

#include "DataLoader.h"
#include "InferenceApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "Tensor.h"

typedef struct trainingStats {
    tensor_t *output;
    float loss;
} trainingStats_t;

/*! Aggregate evaluation metrics for a full epoch.
 *
 * Loss is averaged across batches. Accuracy is over all evaluated samples.
 * Precision, recall and F1 are macro-averaged (unweighted mean across classes).
 */
typedef struct epochStats {
    float loss;
    float accuracy;
    float precision;
    float recall;
    float f1;
} epochStats_t;

/*! Full classification report returned by evaluationEpochWithReport().
 *
 * The confusion matrix lives in a caller-provided buffer of size
 * numClasses * numClasses, indexed as cm[predicted * numClasses + actual].
 * Ownership of confusionMatrix stays with the caller.
 */
typedef struct classificationReport {
    epochStats_t stats;
    size_t *confusionMatrix;
    size_t numClasses;
} classificationReport_t;

/*! Final result of trainingRun() after the last epoch completed. */
typedef struct trainingRunResult {
    float finalTrainLoss;
    epochStats_t finalEvalStats;
} trainingRunResult_t;

typedef trainingStats_t *(*calculateGradsFn_t)(layer_t **model, size_t modelSize,
                                               lossConfig_t lossConfig, size_t batchSize,
                                               tensor_t *input, tensor_t *label);

typedef inferenceStats_t *(*inferenceWithLossFn_t)(layer_t **model, size_t numberOfLayers,
                                                   tensor_t *input, tensor_t *label,
                                                   lossFuncType_t funcType);

/*! Callback invoked once per training epoch, after evaluation completes. */
typedef void (*epochCallbackFn_t)(size_t epoch, float trainLoss, epochStats_t evalStats);

void freeTrainingStats(trainingStats_t *trainingStats);

float evaluationBatch(layer_t **model, size_t modelSize, lossFuncType_t funcType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn);

float evaluationEpoch(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn);

epochStats_t evaluationEpochWithMetrics(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                                        dataLoader_t *dataLoader,
                                        inferenceWithLossFn_t inferenceFn);

classificationReport_t evaluationEpochWithReport(layer_t **model, size_t modelSize,
                                                 lossFuncType_t funcType, dataLoader_t *dataLoader,
                                                 inferenceWithLossFn_t inferenceFn,
                                                 size_t *cmBuffer, size_t numClasses);

trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn, epochCallbackFn_t callback);

#endif // TRAINING_LOOP_API_H
