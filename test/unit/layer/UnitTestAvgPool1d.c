#include <stdint.h>
#include <stdlib.h>

#include "AvgPool1d.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "expected_avg_pool_1d.h"
#include "unity.h"

typedef struct avgPool1dRunResult {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    quantization_t *q;
} avgPool1dRunResult_t;

static avgPool1dRunResult_t avgPool1dBuild(float const *inputData, size_t const *inputDims,
                                           size_t kSize, paddingType_t padding, size_t dilation,
                                           size_t stride, float *outputBuf,
                                           size_t const *outputDims) {
    static kernel_t kernelStore;
    static avgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, kSize, padding, dilation, stride);

    quantization_t *q = quantizationInitFloat();
    initAvgPool1dConfig(&cfgStore, &kernelStore, q, q);

    layerStore.type = AVGPOOL1D;
    lcStore.avgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    avgPool1dRunResult_t r = {0};
    r.layer = &layerStore;
    r.input = tensorInitFloat((float *)inputData, (size_t *)inputDims, 3, NULL);
    r.output = tensorInitFloat(outputBuf, (size_t *)outputDims, 3, NULL);
    r.q = q;
    return r;
}

void testAvgPool1dForwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};

    avgPool1dRunResult_t r =
        avgPool1dBuild(input_avgPool1d_basic, inputDims, 2, VALID, 1, 1, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_avgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_basic[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testAvgPool1dBackwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};

    avgPool1dRunResult_t r =
        avgPool1dBuild(input_avgPool1d_basic, inputDims, 2, VALID, 1, 1, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);

    float lossGradData[1 * 1 * 3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, (size_t *)outputDims, 3, NULL);

    float propLossData[1 * 1 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, (size_t *)inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_avgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_basic[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dMultiChannel(void) {
    size_t inputDims[] = {1, 3, 5};
    size_t outputDims[] = {1, 3, 4};
    float outputData[1 * 3 * 4] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_multiChannel, inputDims, 2, VALID, 1, 1,
                                            outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_avgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_multiChannel[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[1 * 3 * 4];
    for (size_t i = 0; i < 12; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, (size_t *)outputDims, 3, NULL);

    float propLossData[1 * 3 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, (size_t *)inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_multiChannel[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dMultiBatch(void) {
    size_t inputDims[] = {4, 2, 4};
    size_t outputDims[] = {4, 2, 3};
    float outputData[4 * 2 * 3] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_multiBatch, inputDims, 2, VALID, 1, 1,
                                            outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_avgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[4 * 2 * 3];
    for (size_t i = 0; i < 24; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, (size_t *)outputDims, 3, NULL);

    float propLossData[4 * 2 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, (size_t *)inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_multiBatch[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dWithStrideAndDilation(void) {
    size_t inputDims[] = {1, 1, 9};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_withStrideAndDilation, inputDims, 2,
                                            VALID, 2, 3, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_avgPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_withStrideAndDilation[i],
                                 ((float *)r.output->data)[i]);
    }

    tensor_t *lossGrad = tensorInitFloat((float *)lossGrad_avgPool1d_withStrideAndDilation,
                                         (size_t *)outputDims, 3, NULL);

    float propLossData[1 * 1 * 9] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, (size_t *)inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_withStrideAndDilation[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dWithSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5}; // SAME -> outLen = inLen
    float outputData[1 * 1 * 5] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_withSamePadding, inputDims, 3, SAME, 1,
                                            1, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_avgPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_withSamePadding[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[1 * 1 * 5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, (size_t *)outputDims, 3, NULL);

    float propLossData[1 * 1 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, (size_t *)inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_withSamePadding[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dEdgeCases(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 1}; // K=L=4, stride=1, VALID -> outLen = 4-4+1 = 1
    float outputData[1] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_edgeCases, inputDims, 4, VALID, 1, 1,
                                            outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);
    // input [1,2,3,4] mean = 2.5
    for (size_t i = 0; i < expectedForward_avgPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_edgeCases[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[1] = {1.0f};
    tensor_t *lossGrad = tensorInitFloat(lossGradData, (size_t *)outputDims, 3, NULL);

    float propLossData[1 * 1 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, (size_t *)inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    // 1/K = 0.25 contribution to each of 4 input positions.
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_edgeCases[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAvgPool1dForwardBasic);
    RUN_TEST(testAvgPool1dBackwardBasic);
    RUN_TEST(testAvgPool1dMultiChannel);
    RUN_TEST(testAvgPool1dMultiBatch);
    RUN_TEST(testAvgPool1dWithStrideAndDilation);
    RUN_TEST(testAvgPool1dWithSamePadding);
    RUN_TEST(testAvgPool1dEdgeCases);
    return UNITY_END();
}
