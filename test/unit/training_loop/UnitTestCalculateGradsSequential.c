#define SOURCE_FILE "UnitTestCalculateGradsSequential"

#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* Build a [1,2] float32 tensor from a stack buffer (data is copied into the tensor). */
static tensor_t *makeRowVec2(float a, float b) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    size_t *order = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 2;
    order[0] = 0;
    order[1] = 1;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 2;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    float vals[2] = {a, b};
    tensorFillFromFloatBuffer(t, vals, 2);
    return t;
}

void testCalculateGradsSequentialClosedForm() {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    layer_t *model[2];
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
    model[1] = softmaxLayerInit(&lq);

    /* Set known weights/bias: W = {{0.1,0.2},{0.3,0.4}}, b = {0,0}. */
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float B[2] = {0.0f, 0.0f};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);

    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f); /* one-hot class 0 */

    trainingStats_t *stats = calculateGradsSequential(
        model, 2,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        REDUCTION_MEAN, x, label);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.91300f, stats->loss);

    float *wg = (float *)getGradFromParameter(model[0]->config->linear->weights)->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, wg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, wg[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, wg[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, wg[3]);

    float *bg = (float *)getGradFromParameter(model[0]->config->linear->bias)->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, bg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, bg[1]);

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalculateGradsSequentialClosedForm);
    return UNITY_END();
}
