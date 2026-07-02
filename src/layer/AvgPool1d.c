#define SOURCE_FILE "ODT_AVG_POOL_1D"

#include "AvgPool1d.h"

#include "ArithmeticType.h"
#include "Common.h"
#include "Layer.h"
#include "SlidingWindow1d.h"
#include "Tensor.h"

void initAvgPool1dConfig(avgPool1dConfig_t *cfg, kernel_t *kernel, quantization_t *forwardQ,
                         quantization_t *propLossQ) {
    if (kernel->size == 0) {
        PRINT_ERROR("AvgPool1d: kernel size must be >= 1");
        exit(1);
    }
    cfg->kernel = kernel;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
}

void avgPool1dForwardFloat(layer_t *layer, tensor_t *input, tensor_t *output) {
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    if (output->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("AvgPool1d forward: output length (%zu) does not match "
                    "geometry-derived (%zu)",
                    output->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *xArr = (float const *)input->data;
    float *yArr = (float *)output->data;
    float divisor = (float)cfg->kernel->size; // count_include_pad=true: divisor = K always

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                float sum = 0.0f;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    sum += xArr[(b * channels + c) * inputLength + inputIdx];
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                yArr[outIdx] = sum / divisor;
            }
        }
    }
}

void avgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        avgPool1dForwardFloat(layer, input, output);
        break;
    default:
        PRINT_ERROR("AvgPool1d forward: quantization type not implemented");
        exit(1);
    }
}

void avgPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss) {
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    (void)forwardInput; // not needed: window geometry is determined by kernel + input shape

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("AvgPool1d backward: lossGrad outputLength (%zu) does not match "
                    "geometry-derived (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    float *gxArr = (float *)propLoss->data;
    float divisor = (float)cfg->kernel->size;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                float contribution = gyArr[outIdx] / divisor;

                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    gxArr[(b * channels + c) * inputLength + inputIdx] += contribution;
                }
            }
        }
    }
}

void avgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss) {
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        avgPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    default:
        PRINT_ERROR("AvgPool1d backward: quantization type not implemented");
        exit(1);
    }
}

void avgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("AvgPool1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    size_t inputLength = inputShape->dimensions[2];
    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    outputShape->numberOfDimensions = 3;
    outputShape->dimensions[0] = inputShape->dimensions[0]; // B
    outputShape->dimensions[1] = inputShape->dimensions[1]; // C
    outputShape->dimensions[2] = geom.outputLength;
    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
