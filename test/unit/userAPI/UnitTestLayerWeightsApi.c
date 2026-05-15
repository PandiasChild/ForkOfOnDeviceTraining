#define SOURCE_FILE "UNIT_TEST_LAYER_WEIGHTS_API"

#include "LayerCommon.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testLayerLoadWeightsLinearOverwritesWeightAndBiasTensors(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
        },
        &lq);

    float weightData[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float biasData[2] = {-1.f, -2.f};

    layerLoadWeights(layer, weightData, biasData);

    linearConfig_t *cfg = layer->config->linear;
    float *loadedWeights = (float *)cfg->weights->param->data;
    float *loadedBias = (float *)cfg->bias->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(weightData, loadedWeights, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(biasData, loadedBias, 2);

    freeLinearLayer(layer);
}

void testLayerLoadWeightsLinearNoBiasAcceptsNullBiasData(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_FALSE,
        },
        &lq);

    float weightData[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    layerLoadWeights(layer, weightData, NULL);

    linearConfig_t *cfg = layer->config->linear;
    float *loadedWeights = (float *)cfg->weights->param->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(weightData, loadedWeights, 6);
    TEST_ASSERT_NULL(cfg->bias);

    freeLinearLayer(layer);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerLoadWeightsLinearOverwritesWeightAndBiasTensors);
    RUN_TEST(testLayerLoadWeightsLinearNoBiasAcceptsNullBiasData);
    return UNITY_END();
}
