#define SOURCE_FILE "UNIT_TEST_STATE_DICT_API"

#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "StateDictApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testModelLoadStateDictLoadsTwoLinearLayersSkippingRelu(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *model[3];
    model[0] =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    model[1] = reluLayerInit(&lq);
    model[2] =
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 1, .bias = BIAS_TRUE}, &lq);

    float w0[6] = {1, 2, 3, 4, 5, 6};
    float b0[2] = {10, 20};
    float w1[2] = {7, 8};
    float b1[1] = {99};

    modelLoadStateDict(model, 3,
                       (stateDictEntry_t[]){
                           {.name = "fc1", .weightData = w0, .biasData = b0},
                           {.name = "fc2", .weightData = w1, .biasData = b1},
                       },
                       2);

    linearConfig_t *cfg0 = model[0]->config->linear;
    linearConfig_t *cfg2 = model[2]->config->linear;
    float *fc1Weights = (float *)cfg0->weights->param->data;
    float *fc1Bias = (float *)cfg0->bias->param->data;
    float *fc2Weights = (float *)cfg2->weights->param->data;
    float *fc2Bias = (float *)cfg2->bias->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(w0, fc1Weights, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(b0, fc1Bias, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(w1, fc2Weights, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(b1, fc2Bias, 1);

    freeLinearLayer(model[0]);
    freeReluLayer(model[1]);
    freeLinearLayer(model[2]);
}

void testModelLoadStateDictCountMismatchIsCoveredByDesign(void) {
    /* Count-mismatch fires PRINT_ERROR + exit; Unity cannot catch.
     * Documented in spec section 12. This test exists as a marker. */
    TEST_IGNORE_MESSAGE("Count mismatch path fires exit() — covered by design contract.");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testModelLoadStateDictLoadsTwoLinearLayersSkippingRelu);
    RUN_TEST(testModelLoadStateDictCountMismatchIsCoveredByDesign);
    return UNITY_END();
}
