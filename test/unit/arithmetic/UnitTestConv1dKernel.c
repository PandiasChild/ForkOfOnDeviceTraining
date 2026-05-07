#include "Conv1dKernel.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "expected_conv1d_kernel.h"
#include "unity.h"

void testConv1dKernelSingleChannelSingleBatch() {
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t xDims[] = {1, 1, 4};
    tensor_t *x = tensorInitFloat(xData, xDims, 3, NULL);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = tensorInitFloat(wData, wDims, 3, NULL);

    float yData[3] = {0};
    size_t yDims[] = {1, 1, 3};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

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
    tensor_t *x = tensorInitFloat(xData, xDims, 3, NULL);

    // w: [2, 3, 3] = arange(18) * 0.1
    float wData[18];
    for (size_t i = 0; i < 18; i++) {
        wData[i] = (float)i * 0.1f;
    }
    size_t wDims[] = {2, 3, 3};
    tensor_t *w = tensorInitFloat(wData, wDims, 3, NULL);

    float bData[] = {0.5f, -0.5f};
    size_t bDims[] = {2};
    tensor_t *b = tensorInitFloat(bData, bDims, 1, NULL);

    float yData[2 * 3] = {0};
    size_t yDims[] = {1, 2, 3};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    conv1dKernelFloat32(x, w, b, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_multiChannelWithBias, y->data,
                                  expectedConv1dForward_multiChannelWithBias_len);
}

void testConv1dKernelMultiBatch() {
    // x and w come from the generator; their seeds are fixed.
    // x: [4, 2, 4]; w: [2, 2, 2]; output: [4, 2, 3]
    float yData[4 * 2 * 3] = {0};
    size_t yDims[] = {4, 2, 3};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    size_t xDims[] = {4, 2, 4};
    tensor_t *x = tensorInitFloat((float *)inputConv1dForward_multiBatch, xDims, 3, NULL);

    size_t wDims[] = {2, 2, 2};
    tensor_t *w = tensorInitFloat((float *)weightConv1dForward_multiBatch, wDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_multiBatch, y->data,
                                  expectedConv1dForward_multiBatch_len);
}

void testConv1dKernelGroupsDepthwise() {
    float yData[1 * 4 * 4] = {0}; // [B=1, Cout=4, Lout=5-2+1=4]
    size_t yDims[] = {1, 4, 4};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    size_t xDims[] = {1, 4, 5};
    tensor_t *x = tensorInitFloat((float *)inputConv1dForward_groupsDepthwise, xDims, 3, NULL);

    size_t wDims[] = {4, 1, 2}; // [Cout=4, Cin/groups=1, K=2]
    tensor_t *w = tensorInitFloat((float *)weightConv1dForward_groupsDepthwise, wDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 4, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_groupsDepthwise, y->data,
                                  expectedConv1dForward_groupsDepthwise_len);
}

void testConv1dKernelGroupsGrouped() {
    float yData[1 * 8 * 4] = {0}; // [B=1, Cout=8, Lout=4]
    size_t yDims[] = {1, 8, 4};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    size_t xDims[] = {1, 4, 5};
    tensor_t *x = tensorInitFloat((float *)inputConv1dForward_groupsGrouped, xDims, 3, NULL);

    size_t wDims[] = {8, 2, 2}; // [Cout=8, Cin/groups=2, K=2]
    tensor_t *w = tensorInitFloat((float *)weightConv1dForward_groupsGrouped, wDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 2, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_groupsGrouped, y->data,
                                  expectedConv1dForward_groupsGrouped_len);
}

void testConv1dKernelStrideDilation() {
    float xData[] = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 4.0f};
    size_t xDims[] = {1, 1, 9};
    tensor_t *x = tensorInitFloat(xData, xDims, 3, NULL);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = tensorInitFloat(wData, wDims, 3, NULL);

    float yData[3] = {0};
    size_t yDims[] = {1, 1, 3};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 2, 3); // size, padding, dilation, stride

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_strideDilation, y->data,
                                  expectedConv1dForward_strideDilation_len);
}

void testConv1dKernelSamePadding() {
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    size_t xDims[] = {1, 1, 5};
    tensor_t *x = tensorInitFloat(xData, xDims, 3, NULL);

    float wData[] = {1.0f, 2.0f, 3.0f};
    size_t wDims[] = {1, 1, 3};
    tensor_t *w = tensorInitFloat(wData, wDims, 3, NULL);

    float yData[5] = {0};
    size_t yDims[] = {1, 1, 5};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_samePadding, y->data,
                                  expectedConv1dForward_samePadding_len);
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
    return UNITY_END();
}
