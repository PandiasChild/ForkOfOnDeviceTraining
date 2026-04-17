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

void setUp(void) {}
void tearDown(void) {}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(testFlattenLayerInit_ReturnsFlattenTypedLayer);
  return UNITY_END();
}
