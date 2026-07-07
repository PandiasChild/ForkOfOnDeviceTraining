#define SOURCE_FILE "ODT_MAX_POOL_1D"

#include <math.h>

#include "MaxPool1d.h"

#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "SlidingWindow1d.h"
#include "Tensor.h"

void initMaxPool1dConfig(maxPool1dConfig_t *cfg, kernel_t *kernel, tensor_t *argmaxIndices,
                         quantization_t *forwardQ, quantization_t *propLossQ) {
    if (argmaxIndices == NULL) {
        PRINT_ERROR("MaxPool1d: argmaxIndices must not be NULL — caller must pre-allocate");
        exit(1);
    }
    cfg->kernel = kernel;
    cfg->argmaxIndices = argmaxIndices;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
}

/* executeOp forward kernel adapter — ctx = maxPool1dConfig_t* for kernel_t
 * geometry (mirrors AvgPool1d/Conv1d's ctx convention). auxOut = the layer's
 * pre-allocated argmaxIndices tensor (opSpec_t.auxOut, spec D1): the funnel
 * never converts it (kernel-written verbatim, in ITS OWN storage format,
 * INT32) — this is exactly the dual-output shape auxOut was added for (D1's
 * "MaxPool argmaxIndices lives here"). Only a FLOAT32 arm exists (no SYM
 * kernel body); routing the data output through the funnel still gains real
 * capability, same as AvgPool1d/AdaptiveAvgPool1d — see those files' comments. */
static void maxPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx) {
    (void)n;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d forward: output length (%zu) does not match "
                    "geometry-derived (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }
    if (auxOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d forward: argmaxIndices length (%zu) does not match "
                    "geometry-derived (%zu)",
                    auxOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *xArr = (float const *)input->data;
    float *yArr = (float *)rawOut->data;
    int32_t *argmaxArr = (int32_t *)auxOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                float bestVal = -INFINITY;
                int32_t bestInputIdx = -1;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    float v = xArr[(b * channels + c) * inputLength + inputIdx];
                    if (v > bestVal) {
                        bestVal = v;
                        bestInputIdx = (int32_t)inputIdx;
                    }
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                if (slice.validCount > 0) {
                    yArr[outIdx] = bestVal;
                    argmaxArr[outIdx] = bestInputIdx;
                } else {
                    // spec §6.3: empty window is theoretically possible but in
                    // practice unreachable; log + sentinel-encode rather than exit
                    yArr[outIdx] = 0.0f;
                    argmaxArr[outIdx] = -1;
                    PRINT_ERROR("MaxPool1d: empty window at outPos=%zu — likely user misconfig",
                                outPos);
                }
            }
        }
    }
}

void maxPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        executeOp(
            &(opSpec_t){
                .kernel = maxPool1dForwardKernel,
                .ctx = cfg,
                .inputs = (tensor_t *[]){input},
                .nInputs = 1,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
                .auxOut = cfg->argmaxIndices,
            },
            output);
        break;
    default:
        PRINT_ERROR("MaxPool1d forward: quantization type not implemented");
        exit(1);
    }
}

void maxPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss) {
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    (void)forwardInput; // not needed: argmax already encodes which input position to update.

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    // Defensive: argmax shape must match lossGrad shape.
    if (cfg->argmaxIndices->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d backward: argmaxIndices length (%zu) does not match "
                    "lossGrad outputLength (%zu)",
                    cfg->argmaxIndices->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    int32_t const *argmaxArr = (int32_t const *)cfg->argmaxIndices->data;
    float *gxArr = (float *)propLoss->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                int32_t inputIdx = argmaxArr[outIdx];
                if (inputIdx < 0) {
                    continue; // sentinel: empty window, no gradient flows
                }
                gxArr[(b * channels + c) * inputLength + (size_t)inputIdx] += gyArr[outIdx];
            }
        }
    }
}

void maxPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss) {
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        maxPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    default:
        PRINT_ERROR("MaxPool1d backward: quantization type not implemented");
        exit(1);
    }
}

void maxPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("MaxPool1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    size_t inputLength = inputShape->dimensions[2];
    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    outputShape->numberOfDimensions = 3;
    outputShape->dimensions[0] = inputShape->dimensions[0]; // B
    outputShape->dimensions[1] = inputShape->dimensions[1]; // C
    outputShape->dimensions[2] = geom.outputLength;
    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
