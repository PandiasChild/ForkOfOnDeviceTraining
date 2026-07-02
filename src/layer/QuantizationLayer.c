#define SOURCE_FILE "QUANTIZATION_LAYER"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "ExecuteOp.h"
#include "QuantizationLayer.h"

/* The Quantization layer is a pure conversion node (#266): no arithmetic, no
 * kernel — executeConvert dispatches straight through the conversionMatrix
 * (SYM_INT32->SYM_INT32 routes through the diagonal / requant). A same-dtype
 * non-SYM pair stays a config error: a Quantization layer that neither
 * changes dtype nor requantizes is a misconfigured no-op. */
static void dispatchQuantization(tensor_t *input, tensor_t *output) {
    qtype_t inputDType = input->quantization->type;
    qtype_t outputDType = output->quantization->type;

    if (inputDType == outputDType && inputDType != SYM_INT32) {
        PRINT_ERROR("Quantization layer: same input/output dtype %d is only supported "
                    "for SYM_INT32 (requant); other same-dtype pairs are a config error",
                    (int)inputDType);
        exit(1);
    }
    executeConvert(input, output);
}

void quantizationForward(layer_t *layer, tensor_t *inputTensor, tensor_t *outputTensor) {
    (void)layer; /* dtype dispatch reads the tensors; config->quantization->forwardQ
                  * already determined outputTensor's dtype at allocation time
                  * (initLayerOutputs / initBufferOutput). */
    dispatchQuantization(inputTensor, outputTensor);
}

void quantizationBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss,
                          tensor_t *propLoss) {
    (void)layer;
    (void)forwardInput; /* straight-through: dy does not depend on x. Both forward
                         * and backward requant map their own absmax exactly onto
                         * qMax, so neither pass saturates and no STE clipping mask
                         * is needed (spec D3). */
    dispatchQuantization(loss, propLoss);
}

void quantizationCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    (void)layer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
