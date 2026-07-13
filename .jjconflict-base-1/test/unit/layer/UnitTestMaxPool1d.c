#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Layer.h"
#include "MaxPool1d.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "expected_max_pool_1d.h"
#include "unity.h"

// Helper: build a MaxPool1d layer manually (no UserAPI in Phase 1).
// Uses function-local statics for kernel/cfg/layer storage so addresses
// survive the return-by-value (per PR-2 plan-bug helper-pattern dangling pointers).
typedef struct maxPool1dRunResult {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    tensor_t *argmax;
    quantization_t *q;
} maxPool1dRunResult_t;

static size_t *ownedDims(size_t const *dims, size_t numDims) {
    size_t *owned = reserveMemory(numDims * sizeof(size_t));
    memcpy(owned, dims, numDims * sizeof(size_t));
    return owned;
}

static shape_t *makeShape(size_t const *dims, size_t numDims) {
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims(dims, numDims), numDims, order);
    return shape;
}

static tensor_t *makeFloatTensor(size_t const *dims, size_t numDims, float const *data) {
    tensor_t *t = initTensor(makeShape(dims, numDims), quantizationInitFloat(), NULL);
    if (data != NULL) {
        tensorFillFromFloatBuffer(t, data, calcNumberOfElementsByTensor(t));
    }
    return t;
}

static tensor_t *makeInt32Tensor(size_t const *dims, size_t numDims) {
    return initTensor(makeShape(dims, numDims), quantizationInitInt32(), NULL);
}

static maxPool1dRunResult_t maxPool1dBuild(float const *inputData, size_t const *inputDims,
                                           size_t kSize, paddingType_t padding, size_t dilation,
                                           size_t stride, float *outputBuf, int32_t *argmaxBuf,
                                           size_t const *outputDims) {
    static kernel_t kernelStore;
    static maxPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, kSize, padding, dilation, stride);

    quantization_t *q = quantizationInitFloat();

    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    initMaxPool1dConfig(&cfgStore, &kernelStore, argmax, q, q);

    layerStore.type = MAXPOOL1D;
    lcStore.maxPool1d = &cfgStore;
    layerStore.config = &lcStore;

    maxPool1dRunResult_t r = {0};
    r.layer = &layerStore;
    r.input = makeFloatTensor(inputDims, 3, inputData);
    r.output = makeFloatTensor(outputDims, 3, NULL);
    (void)outputBuf;
    (void)argmaxBuf;
    r.argmax = argmax;
    r.q = q;
    return r;
}

void testMaxPool1dForwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_maxPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_basic[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testMaxPool1dCalcOutputShapeValidAndSame(void) {
    quantization_t *q = quantizationInitFloat();

    // VALID: K=3, stride=1, dilation=1 -> outLen = inLen - K + 1 = 5
    {
        kernel_t kernel;
        initKernel(&kernel, 3, VALID, 1, 1);
        maxPool1dConfig_t cfg = {0};
        // argmax tensor not used by calcOutputShape — pass a dummy via minimal init.
        size_t dummyDims[] = {1, 1, 1};
        tensor_t *dummyArgmax = makeInt32Tensor(dummyDims, 3);
        initMaxPool1dConfig(&cfg, &kernel, dummyArgmax, q, q);

        layer_t layer;
        layerConfig_t lc;
        layer.type = MAXPOOL1D;
        lc.maxPool1d = &cfg;
        layer.config = &lc;

        // shape_t uses pointer fields — must point to valid stack arrays.
        size_t inDimsBacking[] = {2, 4, 7};
        size_t inOrderBacking[] = {0, 1, 2};
        size_t outDimsBacking[] = {0, 0, 0};
        size_t outOrderBacking[] = {0, 0, 0};
        shape_t inShape = {.dimensions = inDimsBacking,
                           .orderOfDimensions = inOrderBacking,
                           .numberOfDimensions = 3};
        shape_t outShape = {.dimensions = outDimsBacking,
                            .orderOfDimensions = outOrderBacking,
                            .numberOfDimensions = 0};

        maxPool1dCalcOutputShape(&layer, &inShape, &outShape);

        TEST_ASSERT_EQUAL_size_t(3, outShape.numberOfDimensions);
        TEST_ASSERT_EQUAL_size_t(2, outShape.dimensions[0]);
        TEST_ASSERT_EQUAL_size_t(4, outShape.dimensions[1]);
        TEST_ASSERT_EQUAL_size_t(5, outShape.dimensions[2]);
    }

    // SAME: K=3, stride=1, dilation=1 -> outLen = inLen
    {
        kernel_t kernel;
        initKernel(&kernel, 3, SAME, 1, 1);
        maxPool1dConfig_t cfg = {0};
        size_t dummyDims[] = {1, 1, 1};
        tensor_t *dummyArgmax = makeInt32Tensor(dummyDims, 3);
        initMaxPool1dConfig(&cfg, &kernel, dummyArgmax, q, q);

        layer_t layer;
        layerConfig_t lc;
        layer.type = MAXPOOL1D;
        lc.maxPool1d = &cfg;
        layer.config = &lc;

        size_t inDimsBacking[] = {1, 1, 7};
        size_t inOrderBacking[] = {0, 1, 2};
        size_t outDimsBacking[] = {0, 0, 0};
        size_t outOrderBacking[] = {0, 0, 0};
        shape_t inShape = {.dimensions = inDimsBacking,
                           .orderOfDimensions = inOrderBacking,
                           .numberOfDimensions = 3};
        shape_t outShape = {.dimensions = outDimsBacking,
                            .orderOfDimensions = outOrderBacking,
                            .numberOfDimensions = 0};

        maxPool1dCalcOutputShape(&layer, &inShape, &outShape);

        TEST_ASSERT_EQUAL_size_t(7, outShape.dimensions[2]);
    }
}

void testMaxPool1dBackwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    // Forward populates argmax — required precondition for backward.
    maxPool1dForward(r.layer, r.input, r.output);

    // lossGrad = ones (matches what the generator used for autograd on `basic`).
    float lossGradData[1 * 1 * 3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    // (Mutation: sentinel-skip removal is vacuous in basic fixture; no empty-window
    //  fixture is in this PR's test set per spec §6.3 / Q3.)
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_basic[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dArgmaxIndicesContent(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);

    // Compare argmax tensor content against generator-emitted gold values.
    int32_t const *actual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_basic_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_basic[i], actual[i]);
    }
}

void testMaxPool1dMultiChannel(void) {
    size_t inputDims[] = {1, 3, 5};  // B=1, C=3, L=5
    size_t outputDims[] = {1, 3, 4}; // outLen = (5-2)/1 + 1 = 4
    float outputData[1 * 3 * 4] = {0};
    int32_t argmaxData[1 * 3 * 4] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_multiChannel, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_maxPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_multiChannel[i],
                                 ((float *)r.output->data)[i]);
    }

    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_multiChannel_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_multiChannel[i], argmaxActual[i]);
    }

    float lossGradData[1 * 3 * 4];
    for (size_t i = 0; i < 12; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_multiChannel[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dMultiBatch(void) {
    size_t inputDims[] = {4, 2, 4};  // B=4, C=2, L=4
    size_t outputDims[] = {4, 2, 3}; // outLen = (4-2)/1 + 1 = 3
    float outputData[4 * 2 * 3] = {0};
    int32_t argmaxData[4 * 2 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_multiBatch, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_maxPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[4 * 2 * 3];
    for (size_t i = 0; i < 24; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_multiBatch[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dWithStrideAndDilation(void) {
    size_t inputDims[] = {1, 1, 9}; // B=1, C=1, L=9
    // K=2, stride=3, dilation=2 -> effective_K = (2-1)*2+1 = 3, outLen = (9-3)/3+1 = 3
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_withStrideAndDilation, inputDims, 2,
                                            VALID, 2, 3, outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_withStrideAndDilation[i],
                                 ((float *)r.output->data)[i]);
    }

    // auxOut integration (spec Testing list): argmaxIndices now flows through
    // opSpec_t.auxOut (kernel-written verbatim, never funnel-converted) —
    // assert it is byte-identical to this pre-migration, unregenerated
    // fixture (a non-trivial dilation/stride pattern, unlike the other
    // argmax-checking tests' simpler geometries).
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_withStrideAndDilation[i], argmaxActual[i]);
    }

    // Use the gold-emitted random lossGrad (NOT ones), so positional mutations
    // on the backward path are non-vacuous (codebase_uniform_lossgrad_mutation_vacuity).
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_maxPool1d_withStrideAndDilation);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_withStrideAndDilation[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dWithSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5}; // SAME -> outLen = inLen
    float outputData[1 * 1 * 5] = {0};
    int32_t argmaxData[1 * 1 * 5] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_withSamePadding, inputDims, 3, SAME, 1,
                                            1, outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_maxPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_withSamePadding[i],
                                 ((float *)r.output->data)[i]);
    }

    // Verify argmax content for both edge windows (outPos=0 and outPos=4).
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_withSamePadding[i], argmaxActual[i]);
    }

    float lossGradData[1 * 1 * 5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_withSamePadding[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dEdgeCases(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 4}; // K=1 stride=1 -> outLen = inLen
    float outputData[1 * 1 * 4] = {0};
    int32_t argmaxData[1 * 1 * 4] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_edgeCases, inputDims, 1, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_maxPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_edgeCases[i],
                                 ((float *)r.output->data)[i]);
    }

    // K=1 -> argmax[i] == i for every output position.
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_edgeCases_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_edgeCases[i], argmaxActual[i]);
    }

    float lossGradData[1 * 1 * 4];
    for (size_t i = 0; i < 4; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_edgeCases[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMaxPool1dForwardBasic);
    RUN_TEST(testMaxPool1dCalcOutputShapeValidAndSame);
    RUN_TEST(testMaxPool1dBackwardBasic);
    RUN_TEST(testMaxPool1dArgmaxIndicesContent);
    RUN_TEST(testMaxPool1dMultiChannel);
    RUN_TEST(testMaxPool1dMultiBatch);
    RUN_TEST(testMaxPool1dWithStrideAndDilation);
    RUN_TEST(testMaxPool1dWithSamePadding);
    RUN_TEST(testMaxPool1dEdgeCases);
    return UNITY_END();
}
