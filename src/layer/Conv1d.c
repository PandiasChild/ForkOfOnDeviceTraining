#define SOURCE_FILE "ODT_CONV1D"

#include "Conv1d.h"

#include "Common.h"
#include "Conv1dKernel.h"
#include "ConvTranspose1dKernel.h"
#include "Layer.h"
#include "SlidingWindow1d.h"
#include "Tensor.h"

void initConv1dConfigWithWeightsAndBias(conv1dConfig_t *conv1dConfig, kernel_t *kernel,
                                        parameter_t *weights, parameter_t *bias, size_t groups,
                                        quantization_t *forwardQ, quantization_t *weightGradQ,
                                        quantization_t *biasGradQ, quantization_t *propLossQ) {
    if (groups == 0) {
        PRINT_ERROR("Conv1d: groups must be >= 1");
        exit(1);
    }
    conv1dConfig->kernel = kernel;
    conv1dConfig->weights = weights;
    conv1dConfig->bias = bias;
    conv1dConfig->groups = groups;
    conv1dConfig->forwardQ = forwardQ;
    conv1dConfig->weightGradQ = weightGradQ;
    conv1dConfig->biasGradQ = biasGradQ;
    conv1dConfig->propLossQ = propLossQ;
}

void conv1dForwardFloat(layer_t *conv1dLayer, tensor_t *input, tensor_t *output) {
    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;
    tensor_t *weightTensor = cfg->weights->param;
    tensor_t *biasTensor = cfg->bias ? cfg->bias->param : NULL;

    conv1dKernelFloat32(input, weightTensor, biasTensor, cfg->kernel, cfg->groups, output);
}

void conv1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    conv1dConfig_t *cfg = layer->config->conv1d;
    switch (cfg->forwardQ->type) {
    case FLOAT32:
        conv1dForwardFloat(layer, input, output);
        break;
    default:
        PRINT_ERROR("Conv1d forward: quantization type not implemented");
        exit(1);
    }
}

static void conv1dCalcWeightGradsFloat32(conv1dConfig_t *cfg, tensor_t *forwardInput,
                                         tensor_t *lossGrad) {
    size_t batch = forwardInput->shape->dimensions[0];
    size_t inChannels = forwardInput->shape->dimensions[1];
    size_t inputLength = forwardInput->shape->dimensions[2];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t kernelSize = cfg->weights->param->shape->dimensions[2];

    size_t groups = cfg->groups;
    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("Conv1d backward: lossGrad outputLength (%zu) does not match "
                    "geometry derived from forwardInput (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    float const *xArr = (float const *)forwardInput->data;
    float const *gyArr = (float const *)lossGrad->data;
    float *gwArr = (float *)cfg->weights->grad->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                size_t oc = outLo + ocOffset;
                for (size_t outPos = 0; outPos < outputLength; outPos++) {
                    windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                    float gy = gyArr[(b * outChannels + oc) * outputLength + outPos];

                    for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                        size_t ic = inLo + icOffset;
                        for (size_t i = 0; i < slice.validCount; i++) {
                            size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                            size_t kernelIdx = slice.firstValidKernelOffset + i;

                            float xv = xArr[(b * inChannels + ic) * inputLength + inputIdx];
                            gwArr[(oc * inChPerGroup + icOffset) * kernelSize + kernelIdx] +=
                                xv * gy;
                        }
                    }
                }
            }
        }
    }
}

static void conv1dCalcBiasGradsFloat32(conv1dConfig_t *cfg, tensor_t *lossGrad) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t outChannels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];

    float const *gyArr = (float const *)lossGrad->data;
    float *gbArr = (float *)cfg->bias->grad->data;

    for (size_t oc = 0; oc < outChannels; oc++) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch; b++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                sum += gyArr[(b * outChannels + oc) * outputLength + outPos];
            }
        }
        gbArr[oc] += sum;
    }
}

void conv1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                         tensor_t *propLoss) {
    conv1dConfig_t *cfg = layer->config->conv1d;

    conv1dCalcWeightGradsFloat32(cfg, forwardInput, lossGrad);

    if (cfg->bias) {
        conv1dCalcBiasGradsFloat32(cfg, lossGrad);
    }

    convTranspose1dKernelFloat32(lossGrad, cfg->weights->param, NULL, cfg->kernel, cfg->groups, 0u,
                                 propLoss);
}

void conv1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                    tensor_t *propLoss) {
    conv1dConfig_t *cfg = layer->config->conv1d;
    switch (cfg->weightGradQ->type) {
    case FLOAT32:
        conv1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    default:
        PRINT_ERROR("Conv1d backward: quantization type not implemented");
        exit(1);
    }
}

void conv1dCalcOutputShape(layer_t *conv1dLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;
    size_t batchSize = inputShape->dimensions[0];
    size_t inputLength = inputShape->dimensions[2];
    size_t outChannels = cfg->weights->param->shape->dimensions[0];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outChannels;
    outputShape->dimensions[2] = geom.outputLength;
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
