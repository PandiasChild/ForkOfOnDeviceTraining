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

void testConvTranspose1dKernelSamePaddingSymmetric() {
    // Adjoint of a Conv1d(K=3, stride=1, dilation=1, SAME) on inputLen=5.
    // Forward Conv1d geometry: padLeft=1, padRight=1, outputLen=5.
    // Adjoint takes lossGrad of shape [1,1,5] and scatters into propLoss [1,1,5].
    //
    // Hand-derived expected propLoss for an all-ones lossGrad against a known
    // small weight: this produces exactly the column sums of the unrolled
    // correlation matrix W, padded.
    //
    //   Forward: y[i] = sum_k x[i + k - 1] * w[k]   (with OOB skipped)
    //     y[0] = w[1]*x[0] + w[2]*x[1]              (padLeft=1 cuts w[0])
    //     y[1] = w[0]*x[0] + w[1]*x[1] + w[2]*x[2]
    //     y[2] = w[0]*x[1] + w[1]*x[2] + w[2]*x[3]
    //     y[3] = w[0]*x[2] + w[1]*x[3] + w[2]*x[4]
    //     y[4] = w[0]*x[3] + w[1]*x[4]              (padRight=1 cuts w[2])
    //
    //   Adjoint with lossGrad=ones:
    //     propLoss[0] = w[1] + w[0]                 = (sum of w-positions hitting x[0])
    //     propLoss[1] = w[2] + w[1] + w[0]
    //     propLoss[2] = w[2] + w[1] + w[0]
    //     propLoss[3] = w[2] + w[1] + w[0]
    //     propLoss[4] = w[2] + w[1]
    //
    // For w = [2, 4, 8]:
    //     propLoss = [4+2, 8+4+2, 8+4+2, 8+4+2, 8+4] = [6, 14, 14, 14, 12]

    float lossGradData[] = {1, 1, 1, 1, 1};
    size_t lossGradDims[] = {1, 1, 5};
    tensor_t *lossGrad = tensorInitFloat(lossGradData, lossGradDims, 3, NULL);

    float weightData[] = {2, 4, 8};
    size_t weightDims[] = {1, 1, 3};
    tensor_t *weight = tensorInitFloat(weightData, weightDims, 3, NULL);

    float propLossData[5] = {0};
    size_t propLossDims[] = {1, 1, 5};
    tensor_t *propLoss = tensorInitFloat(propLossData, propLossDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1); // size, paddingType, dilation, stride

    convTranspose1dKernelFloat32(lossGrad, weight, NULL, &kernel, 1, 0, propLoss);

    float expected[] = {6, 14, 14, 14, 12};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, propLoss->data, 5);
}

void testConvTranspose1dKernelSamePaddingAsymmetric() {
    // Adjoint of a Conv1d(K=4, stride=1, dilation=1, SAME) on inputLen=5.
    // Total pad = 4-1 = 3; padLeft=1, padRight=2 (PyTorch right-biased).
    //
    //   Forward: y[i] = sum_k x[i + k - 1] * w[k]
    //     y[0] = w[1]*x[0] + w[2]*x[1] + w[3]*x[2]              (k=0 cut)
    //     y[1] = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]
    //     y[2] = w[0]*x[1] + w[1]*x[2] + w[2]*x[3] + w[3]*x[4]
    //     y[3] = w[0]*x[2] + w[1]*x[3] + w[2]*x[4]              (k=3 cut: idx=6 >= 5)
    //     y[4] = w[0]*x[3] + w[1]*x[4]                          (k=2,3 cut)
    //
    //   Adjoint with lossGrad=ones:
    //     propLoss[0] = w[1] + w[0]                              (hits y[0], y[1])
    //     propLoss[1] = w[2] + w[1] + w[0]
    //     propLoss[2] = w[3] + w[2] + w[1] + w[0]
    //     propLoss[3] = w[3] + w[2] + w[1] + w[0]
    //     propLoss[4] = w[3] + w[2] + w[1]
    //
    // For w = [1, 2, 4, 8]:
    //     propLoss = [3, 7, 15, 15, 14]

    float lossGradData[] = {1, 1, 1, 1, 1};
    size_t lossGradDims[] = {1, 1, 5};
    tensor_t *lossGrad = tensorInitFloat(lossGradData, lossGradDims, 3, NULL);

    float weightData[] = {1, 2, 4, 8};
    size_t weightDims[] = {1, 1, 4};
    tensor_t *weight = tensorInitFloat(weightData, weightDims, 3, NULL);

    float propLossData[5] = {0};
    size_t propLossDims[] = {1, 1, 5};
    tensor_t *propLoss = tensorInitFloat(propLossData, propLossDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 4, SAME, 1, 1);

    convTranspose1dKernelFloat32(lossGrad, weight, NULL, &kernel, 1, 0, propLoss);

    float expected[] = {3, 7, 15, 15, 14};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, propLoss->data, 5);
}

void testConvTranspose1dKernelAdjointSameGrouped() {
    size_t gyDims[] = {2, 4, 6};
    tensor_t *gy = tensorInitFloat((float *)inputConvT1d_adjointSameGrouped, gyDims, 3, NULL);

    size_t wDims[] = {4, 2, 3};
    tensor_t *w = tensorInitFloat((float *)weightConvT1d_adjointSameGrouped, wDims, 3, NULL);

    size_t propLossDims[] = {2, 4, 6};
    float propLossData[2 * 4 * 6] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, propLossDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);

    convTranspose1dKernelFloat32(gy, w, NULL, &kernel, 2, 0, propLoss);

    for (size_t i = 0; i < expectedConvT1d_adjointSameGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedConvT1d_adjointSameGrouped[i],
                                 ((float *)propLoss->data)[i]);
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
    RUN_TEST(testConvTranspose1dKernelSamePaddingSymmetric);
    RUN_TEST(testConvTranspose1dKernelSamePaddingAsymmetric);
    RUN_TEST(testConvTranspose1dKernelAdjointSameGrouped);
    return UNITY_END();
}
