#ifndef TRAINING_H
#define TRAINING_H

#include "Tensor.h"
#include "Optimizer.h"
#include "LossFunction.h"
#include "DataLoader.h"


typedef struct trainingStats
{
    tensor_t* output;
    float loss;
} trainingStats_t;

/*!
 * Calculates gradients sequentially by given model.
 *
 * IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 *
 * \param model: Pointer to array of layers
 * \param modelSize: Number of layers
 * \param lossFunctionType: Enum of loss function to be used
 * \param input: Model input
 * \param label: Label for loss calculation
 * \returns Training stats containing loss and model output
 */
trainingStats_t* calculateGrads(layer_t** model, size_t modelSize, lossType_t lossFunctionType,
                                tensor_t* input, tensor_t* label);

/*!
 * Calculates gradients sequentially by given model with batch processing.
 *
 * IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 *
 * \param model: Pointer to array of layers
 * \param modelSize: Number of layers
 * \param lossType: Enum of loss function to be used
 * \param batch: Current batch containing items and labels
 * \returns Array of training stats with losses and model outputs
 */
trainingStats_t** calculateGradsBatched(layer_t** model, size_t modelSize,
                                        lossType_t lossType, batch_t* batch);

/*!
 * Calculates gradients sequentially by given model.
 *
 * Afterward a given optimizer will be used.
 *
 * IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 *
 * \param model: Pointer to array of layers
 * \param modelSize: Number of layers
 * \param lossFunctionType: Enum of loss function to be used
 * \param input: Model input
 * \param label: Label for loss calculation
 * \param optimizer: Pointer to optimizer to be used
 * \returns Training stats containing loss and model output
 */
trainingStats_t* trainingEpoch(layer_t** model, size_t modelSize,
                               lossType_t lossFunctionType, tensor_t* input,
                               tensor_t* label, optimizer_t* optimizer);

/*!
 * Frees training stat output and pointer.
 *
 * \param trainingStats: Pointer to training stats
 */
void freeTrainingStats(trainingStats_t* trainingStats);

/*!
 * Frees all training stat output and pointers.
 *
 * \param trainingStatsArr: Pointer to array of training stats
 */
void freeTrainingStatsBatched(trainingStats_t** trainingStatsArr, size_t batchSize);

#endif //TRAINING_H
