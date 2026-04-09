#define SOURCE_FILE "TRAINING_BATCH_DEFAULT"

#include "TrainingBatchDefault.h"
#include "Common.h"

float trainingBatchDefault(layer_t **model, size_t modelSize, lossType_t lossType,
                           batch_t *batch, calculateGradsFn_t calculateGradsFn) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        trainingStats_t *stats =
            calculateGradsFn(model, modelSize, lossType,
                             batch->samples[i]->item, batch->samples[i]->label);
        totalLoss += stats->loss;
        freeTrainingStats(stats);
    }

    return totalLoss / (float)batch->size;
}
