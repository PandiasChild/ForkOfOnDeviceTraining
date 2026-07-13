#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AvgPool1d.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "expected_avg_pool_1d.h"
#include "unity.h"

typedef struct avgPool1dRunResult {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    quantization_t *q;
} avgPool1dRunResult_t;

static tensor_t *makeFloatTensor(size_t const *dims, size_t numDims, float const *data) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (data != NULL) {
        tensorFillFromFloatBuffer(t, data, calcNumberOfElementsByTensor(t));
    }
    return t;
}

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
    r.input = makeFloatTensor(inputDims, 3, inputData);
    r.output = makeFloatTensor(outputDims, 3, NULL);
    (void)outputBuf;
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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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

    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_avgPool1d_withStrideAndDilation);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    // 1/K = 0.25 contribution to each of 4 input positions.
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_edgeCases[i],
                                 ((float *)propLoss->data)[i]);
    }
}

// Smoke test for the funnel's new SYM-input capability (spec Testing list):
// the layer's own forwardMath stays FLOAT32 (no SYM kernel body exists for
// this op), but a SYM_INT32-typed *producer* tensor feeding it must now be
// dequantized by the executeOp prologue rather than silently reinterpreted
// as raw float bits (the pre-migration direct-cast hazard). Tolerance is
// widened relative to the FLOAT32 fixture tests above: it must cover the
// SYM_INT32@12 quantization step (qMax=2047) on top of the exact average,
// not just float rounding noise. absMax(1,4,2,3)=4.0 -> scale=4/2047;
// per-element error <= 0.5*scale ~ 9.8e-4, unchanged by the K=2 average
// (linear in the elements) -> 5e-3 leaves a >5x margin, not vacuous.
void testAvgPool1dForwardWithSymInt32Input(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    static kernel_t kernelStore;
    static avgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, 2, VALID, 1, 1);
    quantization_t *floatQ = quantizationInitFloat();
    initAvgPool1dConfig(&cfgStore, &kernelStore, floatQ, floatQ);
    layerStore.type = AVGPOOL1D;
    lcStore.avgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    size_t *ownedDims = reserveMemory(3 * sizeof(size_t));
    memcpy(ownedDims, inputDims, 3 * sizeof(size_t));
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, 3, order);
    tensor_t *symInput = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    tensorFillFromFloatBuffer(symInput, input_avgPool1d_basic,
                              calcNumberOfElementsByTensor(symInput));

    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    avgPool1dForward(&layerStore, symInput, output);

    for (size_t i = 0; i < expectedForward_avgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-3f, expectedForward_avgPool1d_basic[i],
                                 ((float *)output->data)[i]);
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
    RUN_TEST(testAvgPool1dForwardWithSymInt32Input);
    return UNITY_END();
}
