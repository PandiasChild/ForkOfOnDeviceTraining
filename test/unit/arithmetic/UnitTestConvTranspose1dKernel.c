#include "ConvTranspose1dKernel.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "expected_conv_transpose_1d_kernel.h"
#include "unity.h"

void testConvTranspose1dKernelBasic() {
    float xData[] = {1.0f, 2.0f, 3.0f};
    size_t xDims[] = {1, 1, 3};
    tensor_t *x = tensorInitFloat(xData, xDims, 3, NULL);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2}; // [Cin=1, Cout/groups=1, K=2]
    tensor_t *w = tensorInitFloat(wData, wDims, 3, NULL);

    float yData[4] = {0};
    size_t yDims[] = {1, 1, 4};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1); // size, paddingType, dilation, stride

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 1, 0, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_basic, y->data,
                                  expectedConvT1dForward_basic_len);
}

void testConvTranspose1dKernelWithStride() {
    float xData[] = {1.0f, 2.0f, 3.0f};
    size_t xDims[] = {1, 1, 3};
    tensor_t *x = tensorInitFloat(xData, xDims, 3, NULL);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = tensorInitFloat(wData, wDims, 3, NULL);

    // Lout = (3-1)*2 + (2-1)*1 + 0 + 1 = 6
    float yData[6] = {0};
    size_t yDims[] = {1, 1, 6};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 2); // size, padding, dilation, stride=2

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 1, 0, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_withStride, y->data,
                                  expectedConvT1dForward_withStride_len);
}

void testConvTranspose1dKernelWithOutputPadding() {
    float xData[] = {1.0f, 2.0f, 3.0f};
    size_t xDims[] = {1, 1, 3};
    tensor_t *x = tensorInitFloat(xData, xDims, 3, NULL);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = tensorInitFloat(wData, wDims, 3, NULL);

    // Lout = (3-1)*2 + (2-1)*1 + 1 + 1 = 7 with outputPadding=1
    float yData[7] = {0};
    size_t yDims[] = {1, 1, 7};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 2);

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 1, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_withOutputPadding, y->data,
                                  expectedConvT1dForward_withOutputPadding_len);
}

void testConvTranspose1dKernelWithGroups() {
    // Lout = (3-1)*1 + (2-1)*1 + 0 + 1 = 4
    float yData[1 * 4 * 4] = {0};
    size_t yDims[] = {1, 4, 4};
    tensor_t *y = tensorInitFloat(yData, yDims, 3, NULL);

    size_t xDims[] = {1, 4, 3};
    tensor_t *x = tensorInitFloat((float *)inputConvT1dForward_groups, xDims, 3, NULL);

    size_t wDims[] = {4, 2, 2}; // [Cin=4, Cout/groups=2, K=2]
    tensor_t *w = tensorInitFloat((float *)weightConvT1dForward_groups, wDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 2, 0, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_groups, y->data,
                                  expectedConvT1dForward_groups_len);
}

void testConvTranspose1dKernelIsConvBackwardAdjoint() {
    // Setup: from generator, we have:
    //   inputConvT1dForward_adjointCheck   = grad_y from a Conv1d backward,
    //   weightConvT1dForward_adjointCheck  = the forward Conv1d's W,
    //   expectedConvT1dForward_adjointCheck = autograd-derived dL/dx
    //
    // Conv1d setup that produced these: x:[1,2,5], w:[3,2,3], no bias,
    //   stride=1, padding=0, dilation=1, groups=1
    //   -> y:[1,3,3]
    //
    // For ConvTranspose1d (the adjoint): grad_y is "input" with shape [1,3,3];
    //   weight is reused as [Cin_t, Cout_t/g, K] = [3,2,3].
    //   Output shape = [1, 2, 5] (= original x shape).
    //
    // Lout = (L-1)*stride + dilation*(K-1) + outputPadding + 1
    //      = 2*1 + 1*2 + 0 + 1 = 5  ✓

    size_t xDims[] = {1, 3, 3};
    tensor_t *gy = tensorInitFloat((float *)inputConvT1dForward_adjointCheck, xDims, 3, NULL);

    size_t wDims[] = {3, 2, 3};
    tensor_t *w = tensorInitFloat((float *)weightConvT1dForward_adjointCheck, wDims, 3, NULL);

    float gxData[1 * 2 * 5] = {0};
    size_t yDims[] = {1, 2, 5};
    tensor_t *gx = tensorInitFloat(gxData, yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    convTranspose1dKernelFloat32(gy, w, NULL, &kernel, 1, 0, gx);

    // expected = autograd-derived dL/dx; should match within float tolerance
    for (size_t i = 0; i < expectedConvT1dForward_adjointCheck_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedConvT1dForward_adjointCheck[i],
                                 ((float *)gx->data)[i]);
    }
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConvTranspose1dKernelBasic);
    RUN_TEST(testConvTranspose1dKernelWithStride);
    RUN_TEST(testConvTranspose1dKernelWithOutputPadding);
    RUN_TEST(testConvTranspose1dKernelWithGroups);
    RUN_TEST(testConvTranspose1dKernelIsConvBackwardAdjoint);
    return UNITY_END();
}
