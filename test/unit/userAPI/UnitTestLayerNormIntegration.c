#define SOURCE_FILE "UNIT_TEST_LAYERNORM_INTEGRATION"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "CalculateGradsSequential.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *build2DFloat(size_t b, size_t f, const float *data, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = b;
    dims[1] = f;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

/* freeOptimSgdM cascades into freeParameter for every registered parameter_t
 * (Linear weights/bias, LayerNorm gamma/beta) — the SAME objects the layers
 * own. The factory frees (freeLinearLayer / freeLayerNormLayer) would call
 * freeParameter again, so after freeOptimSgdM the layers must be torn down
 * shell-only. Pattern from UnitTestSgd.c
 * (testSgdZeroGradOnSymInt32GradZeroesMantissasAndResetsScale). Both layers
 * here come from the Borrowing factories (ownsQuantizations=false), so the
 * shared math quantization is freed by the caller, not by the shells. */
static void freeLinearLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void freeLayerNormLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->layerNorm->normalizedShape);
    freeReservedMemory(layer->config->layerNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* One FLOAT32 training step through Linear(4→3) → LayerNorm([3]) → Linear(3→2):
 * forward + MSE loss + backward + SGD-M step. Verifies the loss is finite and
 * that LayerNorm's gamma AND beta actually trained (changed from their 1/0
 * factory init after the optimizer step). */
void testLinearLayerNormLinearOneTrainingStep(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *linear0 =
        linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 3, .bias = BIAS_TRUE}, &lq);
    layer_t *norm = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){3}, .numNormDims = 1}, &lq);
    layer_t *linear1 =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layer_t *model[3] = {linear0, norm, linear1};

    tensor_t *input =
        build2DFloat(2, 4, (float[]){0.5f, -1.f, 2.f, 1.5f, -0.5f, 1.f, -2.f, 0.25f}, 8);
    tensor_t *label = build2DFloat(2, 2, (float[]){1.f, -1.f, 0.5f, 2.f}, 4);

    float gammaBefore = ((float *)norm->config->layerNorm->gamma->param->data)[0];
    float betaBefore = ((float *)norm->config->layerNorm->beta->param->data)[0];

    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.9f, 0.0f, model, 3, FLOAT32);

    trainingStats_t *stats =
        calculateGradsSequential(model, 3, defaultLossConfig(MSE), REDUCTION_MEAN, input, label);
    sgdStepM(optim);

    bool statsNotNull = (stats != NULL);
    float loss = stats ? stats->loss : NAN;
    bool lossFinite = isfinite(loss);
    float gammaAfter = ((float *)norm->config->layerNorm->gamma->param->data)[0];
    float betaAfter = ((float *)norm->config->layerNorm->beta->param->data)[0];

    /* Post-step forward through the InferenceApi path: the training step above
     * routes layer outputs through CalculateGradsSequential's own switch, so
     * this is the only coverage of initBufferOutput's LAYERNORM case (mirrors
     * UnitTestDropoutIntegration). */
    tensor_t *predicted = inference(model, 3, input);
    bool predictedNotNull = (predicted != NULL);
    bool predictedFinite = predictedNotNull;
    for (size_t i = 0; predictedNotNull && i < 4; i++) {
        predictedFinite = predictedFinite && isfinite(((float *)predicted->data)[i]);
    }

    freeTensor(predicted);
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeOptimSgdM(optim); /* frees w0/b0, gamma/beta, w1/b1 parameter_t */
    freeLinearLayerShell(linear1);
    freeLayerNormLayerShell(norm);
    freeLinearLayerShell(linear0);
    freeQuantization(q);

    TEST_ASSERT_TRUE(statsNotNull);
    TEST_ASSERT_TRUE_MESSAGE(lossFinite, "loss must be finite after one training step");
    TEST_ASSERT_TRUE_MESSAGE(gammaAfter != gammaBefore,
                             "LayerNorm gamma[0] must change after the SGD-M step");
    TEST_ASSERT_TRUE_MESSAGE(betaAfter != betaBefore,
                             "LayerNorm beta[0] must change after the SGD-M step");
    TEST_ASSERT_TRUE(predictedNotNull);
    TEST_ASSERT_TRUE_MESSAGE(predictedFinite, "inference output must be finite");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLinearLayerNormLinearOneTrainingStep);
    return UNITY_END();
}
