#define SOURCE_FILE "DESERIALIZE"

#include <stdlib.h>

#include "Common.h"
#include "Deserialize.h"
#include "DeserializeInternal.h"
#include "Linear.h"
#include "Relu.h"
#include "Rounding.h"
#include "Softmax.h"
#include "Tensor.h"

void deserializeTensor(tensor_t *tensor, FILE *f) {
    deserializeShape(tensor->shape, f);
    deserializeQuantization(tensor->quantization, f);

    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    size_t bytesPerValue = calcBytesPerElement(tensor->quantization);

    deserializeData(tensor->data, numberOfValues, bytesPerValue, f);
    deserializeSparsity();
}

void deserializeParameter(parameter_t *parameter, FILE *f) {
    deserializeTensor(parameter->param, f);
    deserializeTensor(parameter->grad, f);
}

void deserializeModel(layer_t **model, size_t sizeModel, FILE *f) {
    for (size_t i = 0; i < sizeModel; i++) {
        deserializeLayer(model[i], f);
    }
}

// Helper Functions

static void deserialize(void *values, size_t numberOfElements, size_t sizeOfElement, FILE *f) {
    fread(values, numberOfElements, sizeOfElement, f);
}

static void deserializeShape(shape_t *shape, FILE *f) {
    deserialize(&shape->numberOfDimensions, 1, sizeof(size_t), f);
    deserialize(shape->dimensions, shape->numberOfDimensions, sizeof(size_t), f);
    deserialize(shape->orderOfDimensions, shape->numberOfDimensions, sizeof(size_t), f);
}

static void deserializeQuantization(quantization_t *q, FILE *f) {
    deserialize(&q->type, 1, sizeof(qtype_t), f);
    deserializeQConfig(q, f);
}

static void deserializeData(uint8_t *data, size_t numberOfValues, size_t bytesPerValue, FILE *f) {
    deserialize(data, numberOfValues, bytesPerValue, f);
}

static void deserializeQConfig(quantization_t *q, FILE *f) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
        break;
    case SYM_INT32:
        symInt32QConfig_t *symIntQC = q->qConfig;
        deserialize(&symIntQC->scale, 1, sizeof(float), f);
        deserialize(&symIntQC->roundingMode, 1, sizeof(roundingMode_t), f);
        deserialize(&symIntQC->qMaxBits, 1, sizeof(uint8_t), f);
        break;
    case SYM:
        symQConfig_t *symQC = q->qConfig;
        deserialize(&symQC->scale, 1, sizeof(float), f);
        deserialize(&symQC->qBits, 1, sizeof(uint8_t), f);
        break;
    case ASYM:
        asymQConfig_t *asymQC = q->qConfig;
        deserialize(&asymQC->scale, 1, sizeof(float), f);
        deserialize(&asymQC->qBits, 1, sizeof(uint8_t), f);
        deserialize(&asymQC->roundingMode, 1, sizeof(roundingMode_t), f);
        deserialize(&asymQC->zeroPoint, 1, sizeof(int16_t), f);
        break;
    default:
        PRINT_ERROR("Unknown qType!");
        exit(1);
    }
}

// TODO
static void deserializeSparsity() {}

static void deserializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR:
        linearConfig_t *linearConfig = layer->config->linear;
        deserializeParameter(linearConfig->weights, f);
        deserializeParameter(linearConfig->bias, f);
        deserializeQuantization(linearConfig->outputQ, f);
        deserializeQuantization(linearConfig->propLossQ, f);
        break;
    case RELU:
        reluConfig_t *reluConfig = layer->config->relu;
        deserializeQuantization(reluConfig->outputQ, f);
        deserializeQuantization(reluConfig->propLossQ, f);
        break;
    case SOFTMAX:
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        deserializeQuantization(softmaxConfig->outputQ, f);
        deserializeQuantization(softmaxConfig->propLossQ, f);
        break;
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    default:
        PRINT_ERROR("Unsupported qtype!\n");
        exit(1);
    }
}
