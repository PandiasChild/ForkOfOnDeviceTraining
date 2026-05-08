#include <stdlib.h>

#include "Conv1dTransposed.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "expected_conv1d_transposed.h"
#include "unity.h"

// Helper: build a Conv1dTransposed layer manually (no UserAPI in Phase 1)
typedef struct convT1dRunResult {
    parameter_t *weights;
    parameter_t *bias;
    layer_t layer;
    layerConfig_t lc;
    conv1dTransposedConfig_t cfg;
    tensor_t *input;
    tensor_t *output;
    kernel_t kernel;
    quantization_t *q;
} convT1dRunResult_t;

static void convT1dBuild(convT1dRunResult_t *r, float const *weightData, size_t const *weightDims,
                         float const *biasData, size_t const *biasDims, int hasBias,
                         float const *inputData, size_t const *inputDims, size_t kSize,
                         size_t dilation, size_t stride, size_t groups, size_t outputPadding,
                         float *outputBuf, size_t const *outputDims) {
    tensor_t *weightParam = tensorInitFloat((float *)weightData, (size_t *)weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    r->weights = parameterInit(weightParam, weightGrad);

    if (hasBias) {
        tensor_t *biasParam = tensorInitFloat((float *)biasData, (size_t *)biasDims, 1, NULL);
        tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
        r->bias = parameterInit(biasParam, biasGrad);
    } else {
        r->bias = NULL;
    }

    initKernel(&r->kernel, kSize, VALID, dilation, stride);
    r->q = quantizationInitFloat();

    initConv1dTransposedConfigWithWeightsAndBias(&r->cfg, &r->kernel, r->weights, r->bias, groups,
                                                 outputPadding, r->q, r->q, r->q, r->q);
    r->layer.type = CONV1D_TRANSPOSED;
    r->lc.conv1dTransposed = &r->cfg;
    r->layer.config = &r->lc;

    r->input = tensorInitFloat((float *)inputData, (size_t *)inputDims, 3, NULL);
    r->output = tensorInitFloat(outputBuf, (size_t *)outputDims, 3, NULL);
}

void testConv1dTransposedForwardSingleChannelSingleBatch() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};
    float outputData[1 * 1 * 4] = {0};

    convT1dBuild(&r, weight_convT1d_singleChannelSingleBatch, weightDims, NULL, NULL, 0,
                 input_convT1d_singleChannelSingleBatch, inputDims, 2, 1, 1, 1, 0, outputData,
                 outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_singleChannelSingleBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_singleChannelSingleBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedForwardMultiChannelWithBias() {
    convT1dRunResult_t r;
    size_t weightDims[] = {3, 2, 2};
    size_t biasDims[] = {2};
    size_t inputDims[] = {1, 3, 4};
    size_t outputDims[] = {1, 2, 5}; // (4-1)*1 + 1*(2-1) + 0 + 1 = 5
    float outputData[1 * 2 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_multiChannelWithBias, weightDims,
                 bias_convT1d_multiChannelWithBias, biasDims, 1, input_convT1d_multiChannelWithBias,
                 inputDims, 2, 1, 1, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_multiChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_multiChannelWithBias[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedCalcOutputShape() {
    // Cin=3, Cout=2 (= 2*1 groups=1), K=2, stride=2, dilation=1, outputPadding=1.
    // outLen = (4-1)*2 + 1*(2-1) + 1 + 1 = 9
    convT1dRunResult_t r;
    size_t weightDims[] = {3, 2, 2};
    size_t inputDims[] = {1, 3, 4};
    size_t outputDims[] = {1, 1, 1}; // nonzero so gradInitFloat on output works
    float dummyOut[1] = {0};

    convT1dBuild(&r, weight_convT1d_multiChannelWithBias, weightDims, NULL, NULL, 0,
                 input_convT1d_multiChannelWithBias, inputDims, 2, 1, 2, 1, 1, dummyOut,
                 outputDims);

    // shape_t uses pointer fields — must point to valid stack arrays.
    size_t inDims[3] = {1, 3, 4};
    size_t inOrder[3] = {0, 1, 2};
    shape_t inShape = {.dimensions = inDims, .orderOfDimensions = inOrder, .numberOfDimensions = 3};

    size_t outDims[3] = {0, 0, 0};
    size_t outOrder[3] = {0, 0, 0};
    shape_t outShape = {
        .dimensions = outDims, .orderOfDimensions = outOrder, .numberOfDimensions = 0};
    conv1dTransposedCalcOutputShape(&r.layer, &inShape, &outShape);

    TEST_ASSERT_EQUAL_size_t(3u, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(1u, outShape.dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(2u, outShape.dimensions[1]);
    TEST_ASSERT_EQUAL_size_t(9u, outShape.dimensions[2]);
}

void testConv1dTransposedBackwardSingleChannelWithBias() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};
    float outputData[1 * 1 * 4] = {0};

    convT1dBuild(&r, weight_convT1d_singleChannelWithBias, weightDims,
                 bias_convT1d_singleChannelWithBias, biasDims, 1,
                 input_convT1d_singleChannelWithBias, inputDims, 2, 1, 1, 1, 0, outputData,
                 outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output); // sanity

    float lossGradData[4];
    for (size_t i = 0; i < 4; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[3] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_singleChannelWithBias[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_singleChannelWithBias[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_convT1d_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_convT1d_singleChannelWithBias[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dTransposedBackwardStride2() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 3};
    // outLen = (3-1)*2 + 1*(2-1) + 0 + 1 = 6
    size_t outputDims[] = {1, 1, 6};
    float outputData[6] = {0};

    convT1dBuild(&r, weight_convT1d_stride2, weightDims, NULL, NULL, 0, input_convT1d_stride2,
                 inputDims, 2, 1, 2, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    float lossGradData[6];
    for (size_t i = 0; i < 6; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[3] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_stride2_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_stride2[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_stride2_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_stride2[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dTransposedForwardMultiBatch() {
    convT1dRunResult_t r;
    size_t weightDims[] = {2, 2, 2};
    size_t inputDims[] = {3, 2, 4};
    // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5
    size_t outputDims[] = {3, 2, 5};
    float outputData[3 * 2 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_multiBatch, weightDims, NULL, NULL, 0, input_convT1d_multiBatch,
                 inputDims, 2, 1, 1, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedForwardGroupsDepthwise() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 1, 2};
    size_t inputDims[] = {1, 4, 4};
    // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5; Cout = 1*4 = 4
    size_t outputDims[] = {1, 4, 5};
    float outputData[1 * 4 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsDepthwise, weightDims, NULL, NULL, 0,
                 input_convT1d_groupsDepthwise, inputDims, 2, 1, 1, 4, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_groupsDepthwise[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedBackwardGroupsDepthwise() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 1, 2};
    size_t inputDims[] = {1, 4, 4};
    size_t outputDims[] = {1, 4, 5};
    float outputData[1 * 4 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsDepthwise, weightDims, NULL, NULL, 0,
                 input_convT1d_groupsDepthwise, inputDims, 2, 1, 1, 4, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    float lossGradData[1 * 4 * 5];
    for (size_t i = 0; i < 20; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_groupsDepthwise[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_groupsDepthwise[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dTransposedForwardGroupsGrouped() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 4, 2};
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 4};
    // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5; Cout = 4*2 = 8
    size_t outputDims[] = {1, 8, 5};
    float outputData[1 * 8 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsGrouped, weightDims, bias_convT1d_groupsGrouped, biasDims,
                 1, input_convT1d_groupsGrouped, inputDims, 2, 1, 1, 2, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_groupsGrouped[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedBackwardGroupsGrouped() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 4, 2};
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 4};
    size_t outputDims[] = {1, 8, 5};
    float outputData[1 * 8 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsGrouped, weightDims, bias_convT1d_groupsGrouped, biasDims,
                 1, input_convT1d_groupsGrouped, inputDims, 2, 1, 1, 2, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    float lossGradData[1 * 8 * 5];
    for (size_t i = 0; i < 40; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_groupsGrouped[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_groupsGrouped[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_convT1d_groupsGrouped[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dTransposedForwardStride2WithOutputPadding() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    // outLen = (3-1)*2 + 1*(2-1) + 1 + 1 = 7
    size_t outputDims[] = {1, 1, 7};
    float outputData[7] = {0};

    convT1dBuild(&r, weight_convT1d_stride2WithOutputPadding, weightDims,
                 bias_convT1d_stride2WithOutputPadding, biasDims, 1,
                 input_convT1d_stride2WithOutputPadding, inputDims, 2, 1, 2, 1, 1, outputData,
                 outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_stride2WithOutputPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_stride2WithOutputPadding[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedForwardDilation2() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 4};
    // outLen = (4-1)*1 + 2*(2-1) + 0 + 1 = 6
    size_t outputDims[] = {1, 1, 6};
    float outputData[6] = {0};

    convT1dBuild(&r, weight_convT1d_dilation2, weightDims, NULL, NULL, 0, input_convT1d_dilation2,
                 inputDims, 2, 2, 1, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_dilation2_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_dilation2[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedCalcOutputShapeWithGroups() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 4, 2};
    size_t inputDims[] = {1, 4, 4};
    size_t outputDims[] = {0, 0, 0};
    float dummyOut[1] = {0};

    convT1dBuild(&r, weight_convT1d_groupsGrouped, weightDims, NULL, NULL, 0,
                 input_convT1d_groupsGrouped, inputDims, 2, 1, 1, 2, 0, dummyOut, outputDims);

    // Caller-allocated dimensions/orderOfDimensions (shape_t.dimensions is size_t* not array)
    size_t inDimsForShape[3] = {1, 4, 4};
    size_t inOrder[3] = {0, 1, 2};
    shape_t inShape = {
        .numberOfDimensions = 3, .dimensions = inDimsForShape, .orderOfDimensions = inOrder};

    size_t outDimsForShape[3] = {0, 0, 0};
    size_t outOrder[3] = {0, 0, 0};
    shape_t outShape = {
        .numberOfDimensions = 0, .dimensions = outDimsForShape, .orderOfDimensions = outOrder};

    conv1dTransposedCalcOutputShape(&r.layer, &inShape, &outShape);

    TEST_ASSERT_EQUAL_size_t(3u, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(1u, outShape.dimensions[0]); // batch
    TEST_ASSERT_EQUAL_size_t(8u,
                             outShape.dimensions[1]); // Cout = outChPerGroup * groups = 4 * 2 = 8
    TEST_ASSERT_EQUAL_size_t(5u, outShape.dimensions[2]); // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5
}

void testConv1dTransposedRegistryDispatch() {
    // Verify layerFunctions[CONV1D_TRANSPOSED] entries point at the right fns.
    TEST_ASSERT_NOT_NULL(layerFunctions[CONV1D_TRANSPOSED].forward);
    TEST_ASSERT_NOT_NULL(layerFunctions[CONV1D_TRANSPOSED].backward);
    TEST_ASSERT_NOT_NULL(layerFunctions[CONV1D_TRANSPOSED].calcOutputShape);
    // Identity check: dispatch matches direct call.
    TEST_ASSERT_TRUE(layerFunctions[CONV1D_TRANSPOSED].forward == conv1dTransposedForward);
    TEST_ASSERT_TRUE(layerFunctions[CONV1D_TRANSPOSED].backward == conv1dTransposedBackward);
    TEST_ASSERT_TRUE(layerFunctions[CONV1D_TRANSPOSED].calcOutputShape ==
                     conv1dTransposedCalcOutputShape);
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testConv1dTransposedForwardSingleChannelSingleBatch);
    RUN_TEST(testConv1dTransposedForwardMultiChannelWithBias);
    RUN_TEST(testConv1dTransposedCalcOutputShape);
    RUN_TEST(testConv1dTransposedBackwardSingleChannelWithBias);
    RUN_TEST(testConv1dTransposedBackwardStride2);
    RUN_TEST(testConv1dTransposedForwardMultiBatch);
    RUN_TEST(testConv1dTransposedForwardGroupsDepthwise);
    RUN_TEST(testConv1dTransposedBackwardGroupsDepthwise);
    RUN_TEST(testConv1dTransposedForwardGroupsGrouped);
    RUN_TEST(testConv1dTransposedBackwardGroupsGrouped);
    RUN_TEST(testConv1dTransposedForwardStride2WithOutputPadding);
    RUN_TEST(testConv1dTransposedForwardDilation2);
    RUN_TEST(testConv1dTransposedCalcOutputShapeWithGroups);
    RUN_TEST(testConv1dTransposedRegistryDispatch);
    return UNITY_END();
}
