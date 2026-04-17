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

void setUp(void) {}
void tearDown(void) {}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(testFlattenLayerInit_ReturnsFlattenTypedLayer);
  RUN_TEST(testFlattenCalcOutputShape_NonSquareInput);
  return UNITY_END();
}
