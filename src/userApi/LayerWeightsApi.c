#define SOURCE_FILE "LAYER_WEIGHTS_API"

#include "LayerWeightsApi.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Linear.h"
#include "Tensor.h"
#include <stdlib.h>
#include <string.h>

void layerLoadWeights(layer_t *layer, float *weightData, float *biasData) {
    if (layer == NULL) {
        PRINT_ERROR("layerLoadWeights: layer is NULL");
        exit(1);
    }
    if (weightData == NULL) {
        PRINT_ERROR("layerLoadWeights: weightData is NULL");
        exit(1);
    }

    switch (layer->type) {
    case LINEAR: {
        linearConfig_t *cfg = layer->config->linear;
        if (cfg->weights == NULL) {
            PRINT_ERROR("layerLoadWeights LINEAR: layer has no weight parameter");
            exit(1);
        }
        tensor_t *weightTensor = cfg->weights->param;
        size_t numWeightElements = calcNumberOfElementsByTensor(weightTensor);
        memcpy(weightTensor->data, weightData, numWeightElements * sizeof(float));

        if (cfg->bias != NULL) {
            if (biasData == NULL) {
                PRINT_ERROR("layerLoadWeights LINEAR: layer has bias but biasData is NULL");
                exit(1);
            }
            tensor_t *biasTensor = cfg->bias->param;
            size_t numBiasElements = calcNumberOfElementsByTensor(biasTensor);
            memcpy(biasTensor->data, biasData, numBiasElements * sizeof(float));
        } else if (biasData != NULL) {
            PRINT_ERROR("layerLoadWeights LINEAR: layer has no bias but biasData is non-NULL");
            exit(1);
        }
        break;
    }
    case CONV1D: {
        conv1dConfig_t *cfg = layer->config->conv1d;
        if (cfg->weights == NULL) {
            PRINT_ERROR("layerLoadWeights CONV1D: layer has no weight parameter");
            exit(1);
        }
        tensor_t *weightTensor = cfg->weights->param;
        size_t numWeightElements = calcNumberOfElementsByTensor(weightTensor);
        memcpy(weightTensor->data, weightData, numWeightElements * sizeof(float));

        if (cfg->bias != NULL) {
            if (biasData == NULL) {
                PRINT_ERROR("layerLoadWeights CONV1D: layer has bias but biasData is NULL");
                exit(1);
            }
            tensor_t *biasTensor = cfg->bias->param;
            size_t numBiasElements = calcNumberOfElementsByTensor(biasTensor);
            memcpy(biasTensor->data, biasData, numBiasElements * sizeof(float));
        } else if (biasData != NULL) {
            PRINT_ERROR("layerLoadWeights CONV1D: layer has no bias but biasData is non-NULL");
            exit(1);
        }
        break;
    }
    case RELU:
    case SOFTMAX:
    case FLATTEN:
    case MAXPOOL1D:
    case AVGPOOL1D:
    case ADAPTIVE_AVGPOOL1D:
    case DROPOUT:
        PRINT_ERROR("layerLoadWeights: layer type %d has no parameters to load", (int)layer->type);
        exit(1);
    case CONV1D_TRANSPOSED: {
        conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
        if (cfg->weights == NULL) {
            PRINT_ERROR("layerLoadWeights CONV1D_TRANSPOSED: layer has no weight parameter");
            exit(1);
        }
        tensor_t *weightTensor = cfg->weights->param;
        size_t numWeightElements = calcNumberOfElementsByTensor(weightTensor);
        memcpy(weightTensor->data, weightData, numWeightElements * sizeof(float));

        if (cfg->bias != NULL) {
            if (biasData == NULL) {
                PRINT_ERROR("layerLoadWeights CONV1D_TRANSPOSED: layer has bias but biasData "
                            "is NULL");
                exit(1);
            }
            tensor_t *biasTensor = cfg->bias->param;
            size_t numBiasElements = calcNumberOfElementsByTensor(biasTensor);
            memcpy(biasTensor->data, biasData, numBiasElements * sizeof(float));
        } else if (biasData != NULL) {
            PRINT_ERROR("layerLoadWeights CONV1D_TRANSPOSED: layer has no bias but biasData "
                        "is non-NULL");
            exit(1);
        }
        break;
    }
    default:
        PRINT_ERROR("layerLoadWeights: dispatch not implemented for layer type %d",
                    (int)layer->type);
        exit(1);
    }
}
