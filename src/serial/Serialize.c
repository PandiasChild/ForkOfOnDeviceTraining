#define SOURCE_FILE "SERIALIZE"

#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
#include "Linear.h"
#include "Relu.h"
#include "Rounding.h"
#include "Serialize.h"
#include "SerializeInternal.h"
#include "Softmax.h"
#include "Tensor.h"

void serializeTensor(tensor_t *tensor, FILE *f) {
    size_t numberOfValues = calcNumberOfElementsByTensor(tensor);
    size_t bytesPerValue = calcBytesPerElement(tensor->quantization);

    serializeShape(tensor->shape, f);
    serializeQuantization(tensor->quantization, f);
    serializeData(tensor->data, numberOfValues, bytesPerValue, f);
    serializeSparsity();
}

void serializeParameter(parameter_t *parameter, FILE *f) {
    serializeTensor(parameter->param, f);
    serializeTensor(parameter->grad, f);
}

void serializeModel(layer_t **model, size_t sizeModel, FILE *f) {
    for (size_t i = 0; i < sizeModel; i++) {
        serializeLayer(model[i], f);
    }
}

// Helper Functions

static void serialize(void *values, size_t numberOfElements, size_t sizeOfElement, FILE *f) {
    fwrite(values, numberOfElements, sizeOfElement, f);
}

static void serializeShape(shape_t *shape, FILE *f) {
    serialize(&shape->numberOfDimensions, 1, sizeof(size_t), f);
    serialize(shape->dimensions, shape->numberOfDimensions, sizeof(size_t), f);
    serialize(shape->orderOfDimensions, shape->numberOfDimensions, sizeof(size_t), f);
}

static void serializeQuantization(quantization_t *q, FILE *f) {
    serialize(&q->type, 1, sizeof(qtype_t), f);
    serializeQConfig(q, f);
}

static void serializeData(uint8_t *data, size_t numberOfValues, size_t bytesPerValue, FILE *f) {
    serialize(data, numberOfValues, bytesPerValue, f);
}

static void serializeQConfig(quantization_t *q, FILE *f) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
        break;
    case SYM_INT32:
        symInt32QConfig_t *symIntQC = q->qConfig;
        serialize(&symIntQC->scale, 1, sizeof(float), f);
        serialize(&symIntQC->roundingMode, 1, sizeof(roundingMode_t), f);
        serialize(&symIntQC->qMaxBits, 1, sizeof(uint8_t), f);
        break;
    case SYM:
        symQConfig_t *symQC = q->qConfig;
        serialize(&symQC->scale, 1, sizeof(float), f);
        serialize(&symQC->qBits, 1, sizeof(uint8_t), f);
        break;
    case ASYM:
        asymQConfig_t *asymQC = q->qConfig;
        serialize(&asymQC->scale, 1, sizeof(float), f);
        serialize(&asymQC->qBits, 1, sizeof(uint8_t), f);
        serialize(&asymQC->roundingMode, 1, sizeof(roundingMode_t), f);
        serialize(&asymQC->zeroPoint, 1, sizeof(int16_t), f);
        break;
    default:
        PRINT_ERROR("Unknown qType!");
        exit(1);
    }
}

// TODO
static void serializeSparsity() {}

static void serializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR:
        linearConfig_t *linearConfig = layer->config->linear;
        serializeParameter(linearConfig->weights, f);
        serializeParameter(linearConfig->bias, f);
        serializeQuantization(linearConfig->forwardQ, f);
        serializeQuantization(linearConfig->weightGradQ, f);
        serializeQuantization(linearConfig->biasGradQ, f);
        serializeQuantization(linearConfig->propLossQ, f);
        break;
    case RELU:
        reluConfig_t *reluConfig = layer->config->relu;
        serializeQuantization(reluConfig->forwardQ, f);
        serializeQuantization(reluConfig->backwardQ, f);
        break;
    case SOFTMAX:
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        serializeQuantization(softmaxConfig->forwardQ, f);
        serializeQuantization(softmaxConfig->backwardQ, f);
        break;
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    default:
        PRINT_ERROR("Unsupported qtype!\n");
        exit(1);
    }
}
