#define SOURCE_FILE "FLATTEN"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Flatten.h"

void flattenForward(layer_t *flattenLayer, tensor_t *input, tensor_t *output) {

    (void)flattenLayer;

    size_t numberOfElements = calcNumberOfElementsByTensor(input);

    size_t numberOfBytes = calcNumberOfBytesForData(input->quantization, numberOfElements);
    memcpy(output->data, input->data, numberOfBytes);

    if (input->quantization->type == SYM_INT32) {
        symInt32QConfig_t *inputQC = input->quantization->qConfig;
        symInt32QConfig_t *outputQC = output->quantization->qConfig;
        outputQC->scale = inputQC->scale;
    }
}

void flattenBackward(layer_t *flattenLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss) {
    (void)flattenLayer;
    (void)forwardInput;

    size_t numberOfLossElements = calcNumberOfElementsByTensor(loss);
    size_t numberOfPropLossElements = calcNumberOfElementsByTensor(propLoss);
    if (numberOfLossElements != numberOfPropLossElements) {
        PRINT_DEBUG("FLATTEN ERROR: shape mismatch\n");
        return;
    }
    size_t numberOfBytes = calcNumberOfBytesForData(loss->quantization, numberOfLossElements);
    memcpy(propLoss->data, loss->data, numberOfBytes);

    if (loss->quantization->type == SYM_INT32) {
        symInt32QConfig_t *lossQC = loss->quantization->qConfig;
        symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
        propLossQC->scale = lossQC->scale;
    }
}

void flattenCalcOutputShape(layer_t *flattenLayer, shape_t *inputShape, shape_t *outputShape) {
    (void)flattenLayer;

    size_t batch = inputShape->dimensions[0];
    size_t features = 1;
    for (size_t i = 1; i < inputShape->numberOfDimensions; i++) {
        features *= inputShape->dimensions[i];
    }

    // Precondition: caller allocates outputShape->dimensions and
    // ->orderOfDimensions with >= 2 slots, regardless of input rank.
    outputShape->dimensions[0] = batch;
    outputShape->dimensions[1] = features;
    outputShape->numberOfDimensions = 2;
    setOrderOfDimsForNewTensor(outputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
