#define SOURCE_FILE "SOFTMAX_API"

#include "SoftmaxApi.h"
#include "Softmax.h"
#include "StorageApi.h"

layer_t *softmaxLayerInitLegacy(quantization_t *forwardQ, quantization_t *backwardQ) {
    layer_t *softmaxLayer = reserveMemory(sizeof(layer_t));

    softmaxLayer->type = SOFTMAX;

    layerConfig_t *layerConfig = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *softmaxConfig = reserveMemory(sizeof(softmaxConfig_t));
    layerConfig->softmax = softmaxConfig;

    softmaxConfig->forwardQ = forwardQ;
    softmaxConfig->backwardQ = backwardQ;
    softmaxConfig->ownsQuantizations = false;
    softmaxLayer->config = layerConfig;

    return softmaxLayer;
}

void freeSoftmaxLayerLegacy(layer_t *softmaxLayer) {
    freeReservedMemory(softmaxLayer->config->softmax);
    freeReservedMemory(softmaxLayer->config);
    freeReservedMemory(softmaxLayer);
}
