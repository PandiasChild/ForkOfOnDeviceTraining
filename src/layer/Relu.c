#define SOURCE_FILE "RELU"

#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Comparison.h"
#include "DTypes.h"
#include "Layer.h"
#include "Relu.h"
#include "Tensor.h"

void reluInitConfig(reluConfig_t *reluConfig, quantization_t *forwardQ, quantization_t *backwardQ) {
    reluConfig->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    reluConfig->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    reluConfig->outputQ = forwardQ;
    reluConfig->propLossQ = backwardQ;
}

void reluForwardFloat(tensor_t *input, tensor_t *output) {
    gteFloatValue(input, 0, 0, output);
}

void reluForwardSymInt32(tensor_t *input, tensor_t *output) {
    symInt32QConfig_t *inputSymInt32QC = input->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = output->quantization->qConfig;
    gteSymInt32Zero(input, 0, output);
    outputSymInt32QC->scale = inputSymInt32QC->scale;
}

void reluForward(layer_t *reluLayer, tensor_t *input, tensor_t *output) {
    reluConfig_t *reluConfig = reluLayer->config->relu;

    switch (reluConfig->forwardMath.type) {
    case ARITH_FLOAT32:
        reluForwardFloat(input, output);
        break;
    case ARITH_SYM_INT32:
        reluForwardSymInt32(input, output);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void reluBackwardFloat(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    float *inputArray = (float *)forwardInput->data;
    float *gradOutArray = (float *)loss->data;
    float *gradInArray = (float *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInArray[i] = 0;
        } else {
            gradInArray[i] = gradOutArray[i];
        }
    }
}

void reluBackwardSymInt32(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    int32_t *inputArray = (int32_t *)forwardInput->data;
    int32_t *gradOutputArray = (int32_t *)loss->data;
    int32_t *gradInputArray = (int32_t *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInputArray[i] = 0;
        } else {
            gradInputArray[i] = gradOutputArray[i];
        }
    }

    symInt32QConfig_t *lossQC = loss->quantization->qConfig;
    symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
    propLossQC->scale = lossQC->scale;
}

void reluBackward(layer_t *reluLayer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    reluConfig_t *reluConfig = reluLayer->config->relu;

    switch (reluConfig->propLossMath.type) {
    case ARITH_FLOAT32:
        /* Relu backward bypasses the executeOp funnel (scale-transparent) and
         * raw-casts the wire data pointers. The FLOAT32 arm reads/writes
         * forwardInput/loss/propLoss as float*; fed a SYM_INT32 wire it silently
         * reads int mantissa codes as floats — garbage grads propagated with no
         * diagnostic. Guard the actual wire dtypes and fail fast, mirroring the
         * LayerNorm/GroupNorm backward guards (#315, #261). */
        if (forwardInput->quantization->type != FLOAT32 || loss->quantization->type != FLOAT32 ||
            propLoss->quantization->type != FLOAT32) {
            PRINT_ERROR("ReLU backward: FLOAT32 arm requires FLOAT32 wires — got forwardInput %d, "
                        "loss %d, propLoss %d",
                        (int)forwardInput->quantization->type, (int)loss->quantization->type,
                        (int)propLoss->quantization->type);
            exit(1);
        }
        reluBackwardFloat(forwardInput, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        /* The SYM_INT32 arm raw-casts to int32* and derefs loss/propLoss->qConfig
         * as symInt32QConfig_t*; a FLOAT32 wire carries qConfig == NULL, so the
         * mismatch is a NULL deref rather than mere garbage — same fail-fast. */
        if (forwardInput->quantization->type != SYM_INT32 ||
            loss->quantization->type != SYM_INT32 || propLoss->quantization->type != SYM_INT32) {
            PRINT_ERROR("ReLU backward: SYM_INT32 arm requires SYM_INT32 wires — got forwardInput "
                        "%d, loss %d, propLoss %d",
                        (int)forwardInput->quantization->type, (int)loss->quantization->type,
                        (int)propLoss->quantization->type);
            exit(1);
        }
        reluBackwardSymInt32(forwardInput, loss, propLoss);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void reluCalcOutputShape(layer_t *reluLayer, shape_t *inputShape, shape_t *outputShape) {
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
