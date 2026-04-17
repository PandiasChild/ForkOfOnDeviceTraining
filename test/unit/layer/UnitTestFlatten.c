#define SOURCE_FILE "UNIT_TEST_FLATTEN"

#include "DTypes.h"
#include "Flatten.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void testFlattenLayerInit_ReturnsFlattenTypedLayer(void) {
  layer_t *flattenLayer = flattenLayerInit();

  TEST_ASSERT_NOT_NULL(flattenLayer);
  TEST_ASSERT_EQUAL_INT(FLATTEN, flattenLayer->type);
  TEST_ASSERT_NULL_MESSAGE(flattenLayer->config,
                           "FLATTEN carries no per-layer config; layer->config must be NULL");

  freeFlattenLayer(flattenLayer);
}

void testFlattenCalcOutputShape_NonSquareInput(void) {
  // Regression test for the old flattenItemDims bug (dim[2]*dim[2])
  // Input [1, 3, 4, 5] must flatten to [1, 60], NOT [1, 25] or similar.
  size_t inputDims[] = {1, 3, 4, 5};
  size_t inputOrder[] = {0, 1, 2, 3};
  shape_t inputShape = {
      .dimensions = inputDims, .orderOfDimensions = inputOrder, .numberOfDimensions = 4};

  size_t outputDimsBuf[4] = {0};
  size_t outputOrderBuf[4] = {0};
  shape_t outputShape = {
      .dimensions = outputDimsBuf, .orderOfDimensions = outputOrderBuf, .numberOfDimensions = 0};

  layer_t *flatten = flattenLayerInit();
  flattenCalcOutputShape(flatten, &inputShape, &outputShape);

  TEST_ASSERT_EQUAL_UINT(2, outputShape.numberOfDimensions);
  TEST_ASSERT_EQUAL_UINT(1, outputShape.dimensions[0]);
  TEST_ASSERT_EQUAL_UINT(60, outputShape.dimensions[1]);
  TEST_ASSERT_EQUAL_UINT(0, outputShape.orderOfDimensions[0]);
  TEST_ASSERT_EQUAL_UINT(1, outputShape.orderOfDimensions[1]);

  freeFlattenLayer(flatten);
}

void testFlattenForwardFloat_PreservesBytesAndReshapes(void) {
  size_t n = 12; // 1 * 2 * 2 * 3
  float inputData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
  size_t inputDims[] = {1, 2, 2, 3};
  tensor_t *input = tensorInitFloat(inputData, inputDims, 4, NULL);

  float outputData[12] = {0};
  size_t outputDims[] = {1, 12};
  tensor_t *output = tensorInitFloat(outputData, outputDims, 2, NULL);

  layer_t *flatten = flattenLayerInit();
  flattenForward(flatten, input, output);

  float *actual = (float *)output->data;
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(inputData, actual, n);

  freeFlattenLayer(flatten);
}

void testFlattenForwardSymInt32_PropagatesScaleAndValues(void) {
  size_t n = 6;
  float inputFloatData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  size_t inputDims[] = {1, 2, 3};
  tensor_t *input = tensorInitSymInt32(inputFloatData, inputDims, 3, HTE, NULL);

  float outputFloatData[6] = {0};
  size_t outputDims[] = {1, 6};
  tensor_t *output = tensorInitSymInt32(outputFloatData, outputDims, 2, HTE, NULL);

  // Clobber the output's scale so we can verify flattenForward writes it.
  symInt32QConfig_t *outputQC = output->quantization->qConfig;
  outputQC->scale = 0.0f;

  layer_t *flatten = flattenLayerInit();
  flattenForward(flatten, input, output);

  symInt32QConfig_t *inputQC = input->quantization->qConfig;
  TEST_ASSERT_EQUAL_FLOAT(inputQC->scale, outputQC->scale);

  // Values round-trip through the same scale.
  float roundTripData[6];
  tensor_t *roundTrip = tensorInitFloat(roundTripData, outputDims, 2, NULL);
  convertTensor(output, roundTrip);
  float *actual = (float *)roundTrip->data;
  for (size_t i = 0; i < n; i++) {
    TEST_ASSERT_FLOAT_WITHIN(0.1f, inputFloatData[i], actual[i]);
  }

  freeFlattenLayer(flatten);
}

void testFlattenBackwardFloat_CopiesGradsUnchanged(void) {
  size_t n = 6;

  size_t forwardDims[] = {1, 2, 3};
  float forwardData[] = {-1.f, 0.f, 1.f, 2.f, 5.f, -6.f};
  tensor_t *forwardInput = tensorInitFloat(forwardData, forwardDims, 3, NULL);

  size_t lossDims[] = {1, 6};
  float lossData[] = {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, 0.6f};
  tensor_t *loss = tensorInitFloat(lossData, lossDims, 2, NULL);

  float propLossData[6] = {0};
  tensor_t *propLoss = tensorInitFloat(propLossData, forwardDims, 3, NULL);

  layer_t *flatten = flattenLayerInit();
  flattenBackward(flatten, forwardInput, loss, propLoss);

  float *actual = (float *)propLoss->data;
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(lossData, actual, n);

  freeFlattenLayer(flatten);
}

void setUp(void) {}
void tearDown(void) {}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(testFlattenLayerInit_ReturnsFlattenTypedLayer);
  RUN_TEST(testFlattenCalcOutputShape_NonSquareInput);
  RUN_TEST(testFlattenForwardFloat_PreservesBytesAndReshapes);
  RUN_TEST(testFlattenForwardSymInt32_PropagatesScaleAndValues);
  RUN_TEST(testFlattenBackwardFloat_CopiesGradsUnchanged);
  return UNITY_END();
}
