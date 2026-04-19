#define SOURCE_FILE "UNIT_TEST_FLATTEN_INTEGRATION"

#include <stdlib.h>

#include "CalculateGradsSequential.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "QuantizationApi.h"
#include "SoftmaxApi.h"
#include "TensorApi.h"
#include "unity.h"

// Trains [Flatten → Linear(6,2) → Softmax] for one step with MSE loss.
// Main point: initLayerOutputs FLATTEN case must derive output-Q from input tensor.
// If that case is missing, initLayerOutputs hits `default: exit(1)`.
void testCalculateGradsSequential_WithFlattenFirst_DoesNotCrash(void) {
    quantization_t *q = quantizationInitFloat();

    // Linear weights (6 -> 2)
    static float w0Data[2 * 6] = {0};
    static size_t w0Dims[] = {2, 6};
    tensor_t *w0P = tensorInitWithDistribution(XAVIER_UNIFORM, w0Data, w0Dims, 2, q, NULL, 6, 2);
    tensor_t *w0G = gradInitFloat(w0P, NULL);
    parameter_t *w0 = parameterInit(w0P, w0G);

    static float b0Data[2] = {0};
    static size_t b0Dims[] = {1, 2};
    tensor_t *b0P = tensorInitWithDistribution(ZEROS, b0Data, b0Dims, 2, q, NULL, 1, 2);
    tensor_t *b0G = gradInitFloat(b0P, NULL);
    parameter_t *b0 = parameterInit(b0P, b0G);

    layer_t *model[3];
    model[0] = flattenLayerInit();
    model[1] = linearLayerInit(w0, b0, q, q, q, q);
    model[2] = softmaxLayerInit(q, q);

    // Input [1, 2, 3] = 6 elements
    size_t inputDims[] = {1, 2, 3};
    float inputData[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 3, NULL);

    // Label [1, 2]
    size_t labelDims[] = {1, 2};
    float labelData[] = {1.0f, 0.0f};
    tensor_t *label = tensorInitFloat(labelData, labelDims, 2, NULL);

    trainingStats_t *stats = calculateGradsSequential(model, 3, MSE, input, label);

    TEST_ASSERT_NOT_NULL(stats);
    TEST_ASSERT_NOT_NULL(stats->output);
    TEST_ASSERT_EQUAL_UINT(2, stats->output->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(1, stats->output->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(2, stats->output->shape->dimensions[1]);
}

void testCalculateGradsSequential_FlattenRank1_DoesNotOOB(void) {
    // Regression: initLayerOutputs sized output-shape buffers by INPUT rank, but
    // flattenCalcOutputShape always writes 2 slots. For rank-1 input this is OOB.
    size_t inputDims[] = {5};
    float inputData[] = {1.f, 2.f, 3.f, 4.f, 5.f};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 1, NULL);

    size_t labelDims[] = {5, 1};
    float labelData[] = {0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *label = tensorInitFloat(labelData, labelDims, 2, NULL);

    layer_t *model[1];
    model[0] = flattenLayerInit();

    trainingStats_t *stats = calculateGradsSequential(model, 1, MSE, input, label);

    TEST_ASSERT_NOT_NULL(stats);
    TEST_ASSERT_EQUAL_UINT(2, stats->output->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(5, stats->output->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(1, stats->output->shape->dimensions[1]);
}

void testCalculateGradsSequential_FlattenOnly_PreservesValues(void) {
    // FLATTEN is identity on values; stats->output must equal input, reshaped.
    size_t inputDims[] = {1, 2, 3};
    float inputData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 3, NULL);

    size_t labelDims[] = {1, 6};
    float labelData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *label = tensorInitFloat(labelData, labelDims, 2, NULL);

    layer_t *model[1];
    model[0] = flattenLayerInit();

    trainingStats_t *stats = calculateGradsSequential(model, 1, MSE, input, label);

    TEST_ASSERT_NOT_NULL(stats);
    TEST_ASSERT_EQUAL_UINT(2, stats->output->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(1, stats->output->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(6, stats->output->shape->dimensions[1]);

    float *actual = (float *)stats->output->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(inputData, actual, 6);
}

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalculateGradsSequential_WithFlattenFirst_DoesNotCrash);
    RUN_TEST(testCalculateGradsSequential_FlattenRank1_DoesNotOOB);
    RUN_TEST(testCalculateGradsSequential_FlattenOnly_PreservesValues);
    return UNITY_END();
}
