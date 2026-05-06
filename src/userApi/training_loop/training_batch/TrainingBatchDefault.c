#define SOURCE_FILE "TRAINING_BATCH_DEFAULT"

#include "TrainingBatchDefault.h"
#include "Common.h"
#include "DataLoaderApi.h"

float trainingBatchDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           batch_t *batch, calculateGradsFn_t calculateGradsFn,
                           reduction_t forwardReduction) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        trainingStats_t *stats =
            calculateGradsFn(model, modelSize, lossConfig, forwardReduction,
                             batch->samples[i]->item, batch->samples[i]->label);
        totalLoss += stats->loss;
        freeTrainingStats(stats);
        freeSample(batch->samples[i]);
    }

    if (forwardReduction == REDUCTION_MEAN) {
        return totalLoss / (float)batch->size;
    }
    return totalLoss;
}
