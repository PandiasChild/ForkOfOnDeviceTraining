#include <string.h>

#include "Conv1dKernel.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "expected_conv1d_kernel.h"
#include "unity.h"

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

void testConv1dKernelSingleChannelSingleBatch() {
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t xDims[] = {1, 1, 4};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    size_t yDims[] = {1, 1, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1); // size, padding, dilation, stride

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_singleChannelSingleBatch, y->data,
                                  expectedConv1dForward_singleChannelSingleBatch_len);
}

void testConv1dKernelMultiChannelWithBias() {
    // x: [1, 3, 5] = arange(15)
    float xData[15];
    for (size_t i = 0; i < 15; i++) {
        xData[i] = (float)i;
    }
    size_t xDims[] = {1, 3, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    // w: [2, 3, 3] = arange(18) * 0.1
    float wData[18];
    for (size_t i = 0; i < 18; i++) {
        wData[i] = (float)i * 0.1f;
    }
    size_t wDims[] = {2, 3, 3};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    float bData[] = {0.5f, -0.5f};
    size_t bDims[] = {2};
    tensor_t *b = makeFloatTensor(bDims, 1, bData);

    size_t yDims[] = {1, 2, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    conv1dKernelFloat32(x, w, b, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_multiChannelWithBias, y->data,
                                  expectedConv1dForward_multiChannelWithBias_len);
}

void testConv1dKernelMultiBatch() {
    // x and w come from the generator; their seeds are fixed.
    // x: [4, 2, 4]; w: [2, 2, 2]; output: [4, 2, 3]
    size_t yDims[] = {4, 2, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    size_t xDims[] = {4, 2, 4};
    tensor_t *x = makeFloatTensor(xDims, 3, inputConv1dForward_multiBatch);

    size_t wDims[] = {2, 2, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, weightConv1dForward_multiBatch);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_multiBatch, y->data,
                                  expectedConv1dForward_multiBatch_len);
}

void testConv1dKernelGroupsDepthwise() {
    size_t yDims[] = {1, 4, 4}; // [B=1, Cout=4, Lout=5-2+1=4]
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    size_t xDims[] = {1, 4, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, inputConv1dForward_groupsDepthwise);

    size_t wDims[] = {4, 1, 2}; // [Cout=4, Cin/groups=1, K=2]
    tensor_t *w = makeFloatTensor(wDims, 3, weightConv1dForward_groupsDepthwise);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 4, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_groupsDepthwise, y->data,
                                  expectedConv1dForward_groupsDepthwise_len);
}

void testConv1dKernelGroupsGrouped() {
    size_t yDims[] = {1, 8, 4}; // [B=1, Cout=8, Lout=4]
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    size_t xDims[] = {1, 4, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, inputConv1dForward_groupsGrouped);

    size_t wDims[] = {8, 2, 2}; // [Cout=8, Cin/groups=2, K=2]
    tensor_t *w = makeFloatTensor(wDims, 3, weightConv1dForward_groupsGrouped);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 2, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_groupsGrouped, y->data,
                                  expectedConv1dForward_groupsGrouped_len);
}

void testConv1dKernelStrideDilation() {
    float xData[] = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 4.0f};
    size_t xDims[] = {1, 1, 9};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    size_t yDims[] = {1, 1, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 2, 3); // size, padding, dilation, stride

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_strideDilation, y->data,
                                  expectedConv1dForward_strideDilation_len);
}

void testConv1dKernelSamePadding() {
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    size_t xDims[] = {1, 1, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {1.0f, 2.0f, 3.0f};
    size_t wDims[] = {1, 1, 3};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    size_t yDims[] = {1, 1, 5};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_samePadding, y->data,
                                  expectedConv1dForward_samePadding_len);
}

void testConv1dKernelExplicitPaddingStride2() {
    // ECG enc1 geometry (issue #177): K=7, stride=2, EXPLICIT symmetric padding=3.
    // Gold from PyTorch F.conv1d(..., stride=2, padding=3) — see the generator's
    // fixture_explicit_padding. This is the layer-level parity guard for explicit
    // padding: a stride>1 conv that must reproduce PyTorch's padding=N exactly.
    float xData[10];
    for (size_t i = 0; i < 10; i++) {
        xData[i] = (float)(i + 1);
    }
    size_t xDims[] = {1, 1, 10};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f};
    size_t wDims[] = {1, 1, 7};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    float bData[] = {0.25f};
    size_t bDims[] = {1};
    tensor_t *b = makeFloatTensor(bDims, 1, bData);

    size_t yDims[] = {1, 1, 5};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, 7, 3, 1, 2); // size, padding, dilation, stride

    conv1dKernelFloat32(x, w, b, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_explicitPadding, y->data,
                                  expectedConv1dForward_explicitPadding_len);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConv1dKernelSingleChannelSingleBatch);
    RUN_TEST(testConv1dKernelMultiChannelWithBias);
    RUN_TEST(testConv1dKernelMultiBatch);
    RUN_TEST(testConv1dKernelGroupsDepthwise);
    RUN_TEST(testConv1dKernelGroupsGrouped);
    RUN_TEST(testConv1dKernelStrideDilation);
    RUN_TEST(testConv1dKernelSamePadding);
    RUN_TEST(testConv1dKernelExplicitPaddingStride2);
    return UNITY_END();
}
