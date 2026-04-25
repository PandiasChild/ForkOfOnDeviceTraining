#define SOURCE_FILE "SOFTMAX_API"

#include "StorageApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"

layer_t *softmaxLayerInit(quantization_t *forwardQ, quantization_t *backwardQ) {
    layer_t *softmaxLayer = reserveMemory(sizeof(layer_t));

    softmaxLayer->type = SOFTMAX;

    layerConfig_t *layerConfig = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *softmaxConfig = reserveMemory(sizeof(softmaxConfig_t));
    layerConfig->softmax = softmaxConfig;

    softmaxConfig->forwardQ = forwardQ;
    softmaxConfig->backwardQ = backwardQ;
    softmaxLayer->config = layerConfig;

    return  softmaxLayer;
}