#include <stdint.h>
#include <stdlib.h>

#include "AdaptiveAvgPool1d.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "expected_adaptive_avg_pool_1d.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

typedef struct adaptivePoolRun {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    quantization_t *q;
} adaptivePoolRun_t;

static adaptivePoolRun_t build(float const *inputData, size_t const *inputDims, size_t outputSize,
                               float *outputBuf, size_t const *outputDims) {
    static adaptiveAvgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    quantization_t *q = quantizationInitFloat();
    initAdaptiveAvgPool1dConfig(&cfgStore, outputSize, q, q);

    lcStore.adaptiveAvgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    adaptivePoolRun_t r = {0};
    r.layer = &layerStore;
    r.input = tensorInitFloat((float *)inputData, (size_t *)inputDims, 3, NULL);
    r.output = tensorInitFloat(outputBuf, (size_t *)outputDims, 3, NULL);
    r.q = q;
    return r;
}

void testForwardBasic(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 2};
    float outData[1 * 1 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_basic, inDims, 2, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_basic[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardMultiChannelOverlap(void) {
    size_t inDims[] = {1, 3, 5};
    size_t outDims[] = {1, 3, 2};
    float outData[1 * 3 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiChannel, inDims, 2, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_multiChannel[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardMultiBatch(void) {
    size_t inDims[] = {4, 2, 6};
    size_t outDims[] = {4, 2, 4};
    float outData[4 * 2 * 4] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiBatch, inDims, 4, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardGlobal(void) {
    size_t inDims[] = {1, 2, 7};
    size_t outDims[] = {1, 2, 1};
    float outData[1 * 2 * 1] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_global, inDims, 1, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_global_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_global[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardIdentity(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 4};
    float outData[1 * 1 * 4] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_identity, inDims, 4, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_identity_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_identity[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardUpsample(void) {
    size_t inDims[] = {1, 1, 3};
    size_t outDims[] = {1, 1, 5};
    float outData[1 * 1 * 5] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_upsample, inDims, 5, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_upsample_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_upsample[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testBackwardBasic(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 2};
    float outData[1 * 1 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_basic, inDims, 2, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    float gyData[1 * 1 * 2] = {1.0f, 1.0f};
    tensor_t *lossGrad = tensorInitFloat(gyData, (size_t *)outDims, 3, NULL);
    float gxData[1 * 1 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(gxData, (size_t *)inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_basic[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardMultiChannelOverlap(void) {
    size_t inDims[] = {1, 3, 5};
    size_t outDims[] = {1, 3, 2};
    float outData[1 * 3 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiChannel, inDims, 2, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad = tensorInitFloat((float *)lossGrad_adaptiveAvgPool1d_multiChannel,
                                         (size_t *)outDims, 3, NULL);
    float gxData[1 * 3 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(gxData, (size_t *)inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_multiChannel[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardMultiBatch(void) {
    size_t inDims[] = {4, 2, 6};
    size_t outDims[] = {4, 2, 4};
    float outData[4 * 2 * 4] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiBatch, inDims, 4, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad =
        tensorInitFloat((float *)lossGrad_adaptiveAvgPool1d_multiBatch, (size_t *)outDims, 3, NULL);
    float gxData[4 * 2 * 6] = {0};
    tensor_t *propLoss = tensorInitFloat(gxData, (size_t *)inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_multiBatch[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardGlobal(void) {
    size_t inDims[] = {1, 2, 7};
    size_t outDims[] = {1, 2, 1};
    float outData[1 * 2 * 1] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_global, inDims, 1, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    float gyData[1 * 2 * 1] = {1.0f, 1.0f};
    tensor_t *lossGrad = tensorInitFloat(gyData, (size_t *)outDims, 3, NULL);
    float gxData[1 * 2 * 7] = {0};
    tensor_t *propLoss = tensorInitFloat(gxData, (size_t *)inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_global_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_global[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardUpsample(void) {
    size_t inDims[] = {1, 1, 3};
    size_t outDims[] = {1, 1, 5};
    float outData[1 * 1 * 5] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_upsample, inDims, 5, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad =
        tensorInitFloat((float *)lossGrad_adaptiveAvgPool1d_upsample, (size_t *)outDims, 3, NULL);
    float gxData[1 * 1 * 3] = {0};
    tensor_t *propLoss = tensorInitFloat(gxData, (size_t *)inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_upsample_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_upsample[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testCalcOutputShapeFixedRegardlessOfInput(void) {
    static adaptiveAvgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;
    quantization_t *q = quantizationInitFloat();
    initAdaptiveAvgPool1dConfig(&cfgStore, 3, q, q);
    lcStore.adaptiveAvgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    size_t inDims[3] = {1, 5, 20};
    size_t inOrder[3];
    setOrderOfDimsForNewTensor(3, inOrder);
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);

    size_t outDims[3] = {0, 0, 0};
    size_t outOrder[3] = {0, 0, 0};
    shape_t outShape;
    outShape.dimensions = outDims;
    outShape.numberOfDimensions = 3;
    outShape.orderOfDimensions = outOrder;

    adaptiveAvgPool1dCalcOutputShape(&layerStore, &inShape, &outShape);

    TEST_ASSERT_EQUAL_UINT(3, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(1, outShape.dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(5, outShape.dimensions[1]);
    TEST_ASSERT_EQUAL_UINT(3, outShape.dimensions[2]);
    freeQuantization(q);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testForwardBasic);
    RUN_TEST(testForwardMultiChannelOverlap);
    RUN_TEST(testForwardMultiBatch);
    RUN_TEST(testForwardGlobal);
    RUN_TEST(testForwardIdentity);
    RUN_TEST(testForwardUpsample);
    RUN_TEST(testCalcOutputShapeFixedRegardlessOfInput);
    RUN_TEST(testBackwardBasic);
    RUN_TEST(testBackwardMultiChannelOverlap);
    RUN_TEST(testBackwardMultiBatch);
    RUN_TEST(testBackwardGlobal);
    RUN_TEST(testBackwardUpsample);
    return UNITY_END();
}
