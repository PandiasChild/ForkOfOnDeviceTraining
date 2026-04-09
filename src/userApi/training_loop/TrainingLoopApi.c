#define SOURCE_FILE "TRAINING_LOOP_API"

#include <stddef.h>

#include "TrainingLoopApi.h"
#include "TrainingEpochDefault.h"
#include "InferenceApi.h"
#include "TensorApi.h"
#include "StorageApi.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "Optimizer.h"


void freeTrainingStats(trainingStats_t *trainingStats) {
    freeTensor(trainingStats->output);
    freeReservedMemory(trainingStats);
}

float evaluationBatch(layer_t **model, size_t modelSize, lossType_t lossType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        inferenceStats_t *stats =
            inferenceFn(model, modelSize, batch->samples[i]->item, batch->samples[i]->label,
                        lossType);
        totalLoss += stats->loss;
        freeInferenceStats(stats);
    }

    return totalLoss / (float)batch->size;
}

static size_t argmax(const float *data, size_t n) {
    size_t maxIdx = 0;
    float maxVal = data[0];

    for (size_t i = 1; i < n; i++) {
        if (data[i] > maxVal) {
            maxVal = data[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

size_t evaluationBatchAccuracy(layer_t **model, size_t modelSize, batch_t *batch,
                               size_t numClasses, inferenceFn_t inferenceFn) {
    size_t correct = 0;

    for (size_t i = 0; i < batch->size; i++) {
        tensor_t *output = inferenceFn(model, modelSize, batch->samples[i]->item);

        float *outputData = (float *)output->data;
        float *labelData = (float *)batch->samples[i]->label->data;

        size_t predicted = argmax(outputData, numClasses);
        size_t target = argmax(labelData, numClasses);

        if (predicted == target) {
            correct++;
        }

        freeTensor(output);
    }

    return correct;
}

float evaluationEpoch(layer_t **model, size_t modelSize, lossType_t lossType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn) {
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    float totalLoss = 0.0f;

    for (size_t i = 0; i < numberOfBatches; i++) {
        batch_t *batch = dataLoader->getBatch(dataLoader, i);
        totalLoss += evaluationBatch(model, modelSize, lossType, batch, inferenceFn);
        freeBatch(batch);
    }

    return totalLoss / (float)numberOfBatches;
}

float evaluationEpochAccuracy(layer_t **model, size_t modelSize, dataLoader_t *dataLoader,
                              size_t numClasses, inferenceFn_t inferenceFn) {
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    size_t totalCorrect = 0;

    for (size_t i = 0; i < numberOfBatches; i++) {
        batch_t *batch = dataLoader->getBatch(dataLoader, i);
        totalCorrect += evaluationBatchAccuracy(model, modelSize, batch, numClasses, inferenceFn);
        freeBatch(batch);
    }

    size_t evaluatedSamples = numberOfBatches * dataLoader->batchSize;
    return (float)totalCorrect / (float)evaluatedSamples;
}

trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossType_t lossType,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn,
                                epochCallbackFn_t callback) {
    trainingRunResult_t result = {0.0f, 0.0f};

    for (size_t epoch = 0; epoch < numberOfEpochs; epoch++) {
        float trainLoss = trainingEpochDefault(model, modelSize, lossType, trainDataLoader,
                                               optimizer, calculateGradsFn);
        float evalLoss = evaluationEpoch(model, modelSize, lossType, evalDataLoader, inferenceFn);

        if (callback != NULL) {
            callback(epoch, trainLoss, evalLoss);
        }

        result.finalTrainLoss = trainLoss;
        result.finalEvalLoss = evalLoss;
    }

    return result;
}
