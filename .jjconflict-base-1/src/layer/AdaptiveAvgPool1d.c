#define SOURCE_FILE "ODT_ADAPTIVE_AVG_POOL_1D"

#include "AdaptiveAvgPool1d.h"

#include "AdaptiveWindow1d.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Tensor.h"

void initAdaptiveAvgPool1dConfig(adaptiveAvgPool1dConfig_t *cfg, size_t outputSize,
                                 quantization_t *forwardQ, quantization_t *propLossQ) {
    if (outputSize == 0) {
        PRINT_ERROR("AdaptiveAvgPool1d: outputSize must be >= 1");
        exit(1);
    }
    cfg->outputSize = outputSize;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
}

/* executeOp forward kernel adapter — ctx = adaptiveAvgPool1dConfig_t* (outputSize
 * geometry, mirrors AvgPool1d's ctx convention, AvgPool1d.c); 1 input, no
 * auxOut. Only a FLOAT32 arm exists (no SYM kernel body), but routing it
 * through the funnel still gains real capability: the prologue now
 * dequantizes a mismatched-dtype (e.g. SYM_INT32) input into FLOAT32 scratch
 * first, where the pre-migration direct cast would have silently
 * reinterpreted the producer's raw int32 bits as floats. */
static void adaptiveAvgPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut,
                                           tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outputLength = cfg->outputSize;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("AdaptiveAvgPool1d forward: output length (%zu) does not match "
                    "configured outputSize (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *xArr = (float const *)input->data;
    float *yArr = (float *)rawOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);

                float sum = 0.0f;
                for (size_t i = 0; i < w.count; i++) {
                    sum += xArr[(b * channels + c) * inputLength + w.start + i];
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                yArr[outIdx] = sum / (float)w.count;
            }
        }
    }
}

void adaptiveAvgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        executeOp(
            &(opSpec_t){
                .kernel = adaptiveAvgPool1dForwardKernel,
                .ctx = cfg,
                .inputs = (tensor_t *[]){input},
                .nInputs = 1,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
            },
            output);
        break;
    default:
        PRINT_ERROR("AdaptiveAvgPool1d forward: quantization type not implemented");
        exit(1);
    }
}

void adaptiveAvgPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                                    tensor_t *propLoss) {
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    (void)forwardInput; // not needed: window geometry is determined by shapes

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    if (outputLength != cfg->outputSize) {
        PRINT_ERROR("AdaptiveAvgPool1d backward: lossGrad outputLength (%zu) does not match "
                    "configured outputSize (%zu)",
                    outputLength, cfg->outputSize);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    float *gxArr = (float *)propLoss->data;
    // propLoss arrives calloc-zeroed (StorageApi reserveMemory); overlapping
    // windows accumulate correctly via +=.

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                float contribution = gyArr[outIdx] / (float)w.count;

                for (size_t i = 0; i < w.count; i++) {
                    gxArr[(b * channels + c) * inputLength + w.start + i] += contribution;
                }
            }
        }
    }
}

void adaptiveAvgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                               tensor_t *propLoss) {
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        adaptiveAvgPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    default:
        PRINT_ERROR("AdaptiveAvgPool1d backward: quantization type not implemented");
        exit(1);
    }
}

void adaptiveAvgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("AdaptiveAvgPool1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    outputShape->numberOfDimensions = 3;
    outputShape->dimensions[0] = inputShape->dimensions[0]; // B
    outputShape->dimensions[1] = inputShape->dimensions[1]; // C
    outputShape->dimensions[2] = cfg->outputSize;
    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
