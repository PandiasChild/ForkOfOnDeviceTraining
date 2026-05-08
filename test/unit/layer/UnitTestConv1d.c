#include <stdlib.h>

#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "expected_conv1d.h"
#include "unity.h"

typedef struct conv1dFixtureSetup {
    // Pointer-to-caller-stack dims: tensorInitFloat stores a raw pointer, not a copy.
    // The pointed-to arrays must outlive the tensor — i.e. must live on the test
    // function's stack, not on conv1dRunForward's stack.
    size_t const *weightDims; // length 3
    size_t const *biasDims;   // length 1 (or NULL when hasBias==0)
    size_t const *inputDims;  // length 3
    size_t const *outputDims; // length 3
    int hasBias;
    size_t kSize;
    paddingType_t padding;
    size_t dilation;
    size_t stride;
    size_t groups;
    float const *weightData;
    float const *biasData;
    float const *inputData;
} conv1dFixtureSetup_t;

typedef struct conv1dRunResult {
    parameter_t *weights;
    parameter_t *bias;
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    quantization_t *q;
} conv1dRunResult_t;

// Builds the layer (using direct initConv1dConfigWithWeightsAndBias when groups != 1
// — bypassing the UserAPI that hardcodes groups=1) and runs forward.
// Caller owns output buffer.
static conv1dRunResult_t conv1dRunForward(conv1dFixtureSetup_t s, float *outputBuf) {
    // Non-reentrant: function-local static storage for kernel/config/layer is
    // overwritten on each call. Safe for Unity tests, which execute serially
    // with no concurrent calls. If a future test invokes this helper twice
    // and tries to use both layers concurrently, the second call will silently
    // clobber the first.
    conv1dRunResult_t r = {0};

    tensor_t *weightParam = tensorInitFloat((float *)s.weightData, (size_t *)s.weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    r.weights = parameterInit(weightParam, weightGrad);

    if (s.hasBias) {
        tensor_t *biasParam = tensorInitFloat((float *)s.biasData, (size_t *)s.biasDims, 1, NULL);
        tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
        r.bias = parameterInit(biasParam, biasGrad);
    } else {
        r.bias = NULL;
    }

    // kernelStore is static so its address remains valid after this function returns;
    // conv1dLayerInit / initConv1dConfigWithWeightsAndBias both store the kernel pointer.
    static kernel_t kernelStore;
    initKernel(&kernelStore, s.kSize, s.padding, s.dilation, s.stride);
    r.q = quantizationInitFloat();

    if (s.groups == 1) {
        r.layer = conv1dLayerInit(r.weights, r.bias, &kernelStore, r.q, r.q, r.q, r.q);
    } else {
        // Phase-2 will expose groups via UserAPI; here we go around the UserAPI.
        // All statics so their addresses remain valid after this function returns.
        static conv1dConfig_t cfg;
        static layerConfig_t lc;
        static layer_t l;
        initConv1dConfigWithWeightsAndBias(&cfg, &kernelStore, r.weights, r.bias, s.groups, r.q,
                                           r.q, r.q, r.q);
        l.type = CONV1D;
        lc.conv1d = &cfg;
        l.config = &lc;
        r.layer = &l;
    }

    r.input = tensorInitFloat((float *)s.inputData, (size_t *)s.inputDims, 3, NULL);
    r.output = tensorInitFloat(outputBuf, (size_t *)s.outputDims, 3, NULL);
    conv1dForward(r.layer, r.input, r.output);

    return r;
}

void testConv1dForwardMultiChannelWithBias() {
    size_t weightDims[] = {2, 3, 3};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_multiChannelWithBias, weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    size_t biasDims[] = {2};
    tensor_t *biasParam =
        tensorInitFloat((float *)bias_conv1d_multiChannelWithBias, biasDims, 1, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInit(weights, bias, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 3, 5};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_multiChannelWithBias, inputDims, 3, NULL);

    float outputData[1 * 2 * 3] = {0};
    size_t outputDims[] = {1, 2, 3};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedForward_conv1d_multiChannelWithBias, output->data,
                                  expectedForward_conv1d_multiChannelWithBias_len);
}

void testConv1dForwardSingleChannelSingleBatch() {
    size_t weightDims[] = {1, 1, 2};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_singleChannelSingleBatch, weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInit(weights, NULL, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 1, 4};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_singleChannelSingleBatch, inputDims, 3, NULL);

    float outputData[3] = {0};
    size_t outputDims[] = {1, 1, 3};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedForward_conv1d_singleChannelSingleBatch, output->data,
                                  expectedForward_conv1d_singleChannelSingleBatch_len);
}

void testConv1dBackwardSingleChannelWithBias() {
    size_t weightDims[] = {1, 1, 2};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_singleChannelWithBias, weightDims, 3, NULL);
    float weightGradData[2] = {0};
    tensor_t *weightGrad = tensorInitFloat(weightGradData, weightDims, 3, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    size_t biasDims[] = {1};
    tensor_t *biasParam =
        tensorInitFloat((float *)bias_conv1d_singleChannelWithBias, biasDims, 1, NULL);
    float biasGradData[1] = {0};
    tensor_t *biasGrad = tensorInitFloat(biasGradData, biasDims, 1, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInit(weights, bias, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 1, 4};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_singleChannelWithBias, inputDims, 3, NULL);

    // forward (sanity — also fills output)
    float outputData[3] = {0};
    size_t outputDims[] = {1, 1, 3};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);
    conv1dForward(conv1d, input, output);

    // lossGrad = ones (matches what the generator used for autograd)
    float lossGradData[3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    // propLoss buffer caller-owned, pre-zeroed
    float propLossData[4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(conv1d, input, lossGrad, propLoss);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedPropLoss_conv1d_singleChannelWithBias, propLoss->data,
                                  expectedPropLoss_conv1d_singleChannelWithBias_len);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedWeightGrad_conv1d_singleChannelWithBias,
                                  weights->grad->data,
                                  expectedWeightGrad_conv1d_singleChannelWithBias_len);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedBiasGrad_conv1d_singleChannelWithBias, bias->grad->data,
                                  expectedBiasGrad_conv1d_singleChannelWithBias_len);
}

void testConv1dBackwardSamePaddingSymmetric() {
    size_t weightDims[] = {1, 1, 3};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_samePaddingSymmetric, weightDims, 3, NULL);
    float weightGradData[3] = {0};
    tensor_t *weightGrad = tensorInitFloat(weightGradData, weightDims, 3, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInit(weights, NULL, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 1, 5};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_samePaddingSymmetric, inputDims, 3, NULL);

    float outputData[5] = {0};
    size_t outputDims[] = {1, 1, 5};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);
    conv1dForward(conv1d, input, output);

    float lossGradData[5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(conv1d, input, lossGrad, propLoss);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedPropLoss_conv1d_samePaddingSymmetric, propLoss->data,
                                  expectedPropLoss_conv1d_samePaddingSymmetric_len);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedWeightGrad_conv1d_samePaddingSymmetric,
                                  weights->grad->data,
                                  expectedWeightGrad_conv1d_samePaddingSymmetric_len);
}

void testConv1dForwardMultiBatch() {
    size_t weightDims[3] = {2, 2, 2};
    size_t inputDims[3] = {4, 2, 4};
    size_t outputDims[3] = {4, 2, 3};
    float outputData[4 * 2 * 3] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_multiBatch,
        .biasData = NULL,
        .inputData = input_conv1d_multiBatch,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    TEST_ASSERT_EQUAL_size_t(expectedForward_conv1d_multiBatch_len, 4 * 2 * 3);
    for (size_t i = 0; i < expectedForward_conv1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dForwardGroupsDepthwise() {
    size_t weightDims[3] = {4, 1, 2};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 4, 4};
    float outputData[1 * 4 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 4,
        .weightData = weight_conv1d_groupsDepthwise,
        .biasData = NULL,
        .inputData = input_conv1d_groupsDepthwise,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_groupsDepthwise[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardGroupsDepthwise() {
    size_t weightDims[3] = {4, 1, 2};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 4, 4};
    float outputData[1 * 4 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 4,
        .weightData = weight_conv1d_groupsDepthwise,
        .biasData = NULL,
        .inputData = input_conv1d_groupsDepthwise,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[1 * 4 * 4];
    for (size_t i = 0; i < 16; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_groupsDepthwise[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_groupsDepthwise[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dForwardGroupsGrouped() {
    size_t weightDims[3] = {8, 2, 2};
    size_t biasDims[1] = {8};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 8, 4};
    float outputData[1 * 8 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_groupsGrouped,
        .biasData = bias_conv1d_groupsGrouped,
        .inputData = input_conv1d_groupsGrouped,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_groupsGrouped[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardGroupsGrouped() {
    size_t weightDims[3] = {8, 2, 2};
    size_t biasDims[1] = {8};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 8, 4};
    float outputData[1 * 8 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_groupsGrouped,
        .biasData = bias_conv1d_groupsGrouped,
        .inputData = input_conv1d_groupsGrouped,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[1 * 8 * 4];
    for (size_t i = 0; i < 32; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_groupsGrouped[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_groupsGrouped[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_conv1d_groupsGrouped[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dForwardStrideDilation() {
    size_t weightDims[3] = {1, 1, 2};
    size_t inputDims[3] = {1, 1, 9};
    size_t outputDims[3] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 2,
        .stride = 3,
        .groups = 1,
        .weightData = weight_conv1d_strideDilation,
        .biasData = NULL,
        .inputData = input_conv1d_strideDilation,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    TEST_ASSERT_EQUAL_size_t(3u, expectedForward_conv1d_strideDilation_len);
    for (size_t i = 0; i < expectedForward_conv1d_strideDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_strideDilation[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dForwardSamePaddingAsymmetric() {
    size_t weightDims[3] = {1, 1, 4};
    size_t inputDims[3] = {1, 1, 5};
    size_t outputDims[3] = {1, 1, 5};
    float outputData[1 * 1 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 4,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_samePaddingAsymmetric,
        .biasData = NULL,
        .inputData = input_conv1d_samePaddingAsymmetric,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_samePaddingAsymmetric_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_samePaddingAsymmetric[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardSamePaddingAsymmetric() {
    size_t weightDims[3] = {1, 1, 4};
    size_t inputDims[3] = {1, 1, 5};
    size_t outputDims[3] = {1, 1, 5};
    float outputData[1 * 1 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 4,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_samePaddingAsymmetric,
        .biasData = NULL,
        .inputData = input_conv1d_samePaddingAsymmetric,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_samePaddingAsymmetric_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_samePaddingAsymmetric[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_samePaddingAsymmetric_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_samePaddingAsymmetric[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dForwardSamePaddingWithGroups() {
    size_t weightDims[3] = {4, 2, 3};
    size_t biasDims[1] = {4};
    size_t inputDims[3] = {2, 4, 6};
    size_t outputDims[3] = {2, 4, 6};
    float outputData[2 * 4 * 6] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 3,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_samePaddingWithGroups,
        .biasData = bias_conv1d_samePaddingWithGroups,
        .inputData = input_conv1d_samePaddingWithGroups,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_samePaddingWithGroups[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardSamePaddingWithGroups() {
    size_t weightDims[3] = {4, 2, 3};
    size_t biasDims[1] = {4};
    size_t inputDims[3] = {2, 4, 6};
    size_t outputDims[3] = {2, 4, 6};
    float outputData[2 * 4 * 6] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 3,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_samePaddingWithGroups,
        .biasData = bias_conv1d_samePaddingWithGroups,
        .inputData = input_conv1d_samePaddingWithGroups,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[2 * 4 * 6];
    for (size_t i = 0; i < 48; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[2 * 4 * 6] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_samePaddingWithGroups[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_samePaddingWithGroups[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_conv1d_samePaddingWithGroups[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testConv1dForwardMultiChannelWithBias);
    RUN_TEST(testConv1dForwardSingleChannelSingleBatch);
    RUN_TEST(testConv1dBackwardSingleChannelWithBias);
    RUN_TEST(testConv1dBackwardSamePaddingSymmetric);
    RUN_TEST(testConv1dForwardMultiBatch);
    RUN_TEST(testConv1dForwardGroupsDepthwise);
    RUN_TEST(testConv1dBackwardGroupsDepthwise);
    RUN_TEST(testConv1dForwardGroupsGrouped);
    RUN_TEST(testConv1dBackwardGroupsGrouped);
    RUN_TEST(testConv1dForwardStrideDilation);
    RUN_TEST(testConv1dForwardSamePaddingAsymmetric);
    RUN_TEST(testConv1dBackwardSamePaddingAsymmetric);
    RUN_TEST(testConv1dForwardSamePaddingWithGroups);
    RUN_TEST(testConv1dBackwardSamePaddingWithGroups);
    return UNITY_END();
}
