#ifndef TRAININGAPIINTERNAL_H
#define TRAININGAPIINTERNAL_H

#include <stddef.h>

#include "Tensor.h"
#include "Layer.h"

/*! Initializes buffer array to match each layer output.
 * \param layerOutputs: Pointer to buffer array
 * \param model: Pointer to array of layers
 * \param sizeNetwork: Number of layers
 */
static void initLayerOutputs(tensor_t **layerOutputs, layer_t **model, size_t sizeNetwork);

/*! Frees all tensors in tensor array.
 *
 * \param layerOutputs: Pointer to tensor array
 * \param modelSize: Number of tensors
 */
static void deInitLayerOutputs(tensor_t **layerOutputs, size_t modelSize);

/*! Initializes a gradient tensor to match a given layer output tensor.
 *
 * \param grad: Pointer to gradient tensor
 * \param layerOutput: Pointer to layer output
 */
static void initGradTensor(tensor_t *grad, tensor_t *layerOutput);

/*! Frees data, shape and quantization of given tensor.
 *
 * \param tensor: Pointer to tensor
 */
static void deInitGradTensor(tensor_t *tensor);

/*! Initializes training stats to match given output
 *
 * \param output: Pointer to output
 * \returns Pointer to initialized training stats
 */
static trainingStats_t *initTrainingStats(tensor_t *output);

#endif //TRAININGAPIINTERNAL_H
