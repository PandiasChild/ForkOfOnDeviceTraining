#define SOURCE_FILE "UNIT_TEST_MNIST_SMOKE"

#include <stddef.h>

#include "unity.h"

#include "Tensor.h"
#include "TensorApi.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "SgdApi.h"
#include "QuantizationApi.h"
#include "LossFunction.h"
#include "InferenceApi.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "StorageApi.h"
#include "TrainingLoopApi.h"
#include "CalculateGradsSequential.h"


void setUp() {}
void tearDown() {}


#define INPUT_DIM 4
#define HIDDEN_DIM 8
#define NUM_CLASSES 3
#define DATASET_SIZE 6
#define MODEL_SIZE 4


// Tiny trivially-learnable dataset: each input is a noisy one-hot prototype of its class.
static tensor_t *items[DATASET_SIZE];
static tensor_t *labels[DATASET_SIZE];

static void initDataset() {
    static float itemData[DATASET_SIZE][INPUT_DIM] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.9f, 0.1f, 0.0f, 0.0f},
        {0.1f, 0.9f, 0.0f, 0.0f},
        {0.0f, 0.1f, 0.9f, 0.0f},
    };
    static float labelData[DATASET_SIZE][NUM_CLASSES] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    };
    static size_t itemDims[] = {1, INPUT_DIM};
    static size_t labelDims[] = {1, NUM_CLASSES};

    for (size_t i = 0; i < DATASET_SIZE; i++) {
        items[i] = tensorInitFloat(itemData[i], itemDims, 2, NULL);
        labels[i] = tensorInitFloat(labelData[i], labelDims, 2, NULL);
    }
}

static sample_t *getSample(size_t id) {
    sample_t *sample = *reserveMemory(sizeof(sample_t));
    sample->item = items[id];
    sample->label = labels[id];
    return sample;
}

static size_t getDatasetSize() {
    return DATASET_SIZE;
}


// Mirrors the buildModel pattern from experiments/MnistExperiment.c: model is
// constructed inside a helper that returns to its caller before trainingRun()
// runs. This exercises the stack-lifetime contract of setShape()/tensorInit*.
static void buildModel(layer_t **model) {
    quantization_t *q = quantizationInitFloat();

    static float w0Data[HIDDEN_DIM * INPUT_DIM] = {0};
    static size_t w0Dims[] = {HIDDEN_DIM, INPUT_DIM};
    tensor_t *w0Param = tensorInitWithDistribution(XAVIER_UNIFORM, w0Data, w0Dims, 2, q, NULL,
                                                    INPUT_DIM, HIDDEN_DIM);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    static float b0Data[HIDDEN_DIM] = {0};
    static size_t b0Dims[] = {1, HIDDEN_DIM};
    tensor_t *b0Param = tensorInitWithDistribution(ZEROS, b0Data, b0Dims, 2, q, NULL, 1,
                                                    HIDDEN_DIM);
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    model[0] = linearLayerInit(w0, b0, q, q, q, q);
    model[1] = reluLayerInit(q, q);

    static float w1Data[NUM_CLASSES * HIDDEN_DIM] = {0};
    static size_t w1Dims[] = {NUM_CLASSES, HIDDEN_DIM};
    tensor_t *w1Param = tensorInitWithDistribution(XAVIER_UNIFORM, w1Data, w1Dims, 2, q, NULL,
                                                    HIDDEN_DIM, NUM_CLASSES);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    static float b1Data[NUM_CLASSES] = {0};
    static size_t b1Dims[] = {1, NUM_CLASSES};
    tensor_t *b1Param = tensorInitWithDistribution(ZEROS, b1Data, b1Dims, 2, q, NULL, 1,
                                                    NUM_CLASSES);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    model[2] = linearLayerInit(w1, b1, q, q, q, q);
    model[3] = softmaxLayerInit(q, q);
}


static size_t cbInvocations;
static float firstTrainLoss;
static float lastTrainLoss;

static void captureEpoch(size_t epoch, float trainLoss, epochStats_t evalStats) {
    (void)epoch;
    (void)evalStats;
    if (cbInvocations == 0) {
        firstTrainLoss = trainLoss;
    }
    lastTrainLoss = trainLoss;
    cbInvocations++;
}


/* End-to-end smoke test: runs the same pipeline as MnistExperiment at miniature
 * scale. Catches regressions in the Linear->ReLU->Linear->Softmax + CrossEntropy
 * path, trainingRun()'s epoch/batch orchestration, and the epochCallback
 * signature. Runs in well under a second and is suitable for CI.
 */
void testMnistSmoke_FullTrainingPipelineReducesLoss() {
    initDataset();

    dataLoader_t *trainDl = dataLoaderInit(getSample, getDatasetSize, 2,
                                            NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getSample, getDatasetSize, 1,
                                           NULL, NULL, false, 0, true);

    layer_t *model[MODEL_SIZE];
    buildModel(model);

    optimizer_t *sgd = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, MODEL_SIZE, FLOAT32);

    cbInvocations = 0;
    size_t numberOfEpochs = 20;
    trainingRunResult_t result = trainingRun(model, MODEL_SIZE, CROSS_ENTROPY,
                                              trainDl, evalDl, sgd, numberOfEpochs,
                                              calculateGradsSequential, inferenceWithLoss,
                                              captureEpoch);

    TEST_ASSERT_EQUAL_UINT(numberOfEpochs, cbInvocations);
    TEST_ASSERT_TRUE(result.finalTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(result.finalEvalStats.accuracy >= 0.0f);
    TEST_ASSERT_TRUE(result.finalEvalStats.accuracy <= 1.0f);
    TEST_ASSERT_TRUE(lastTrainLoss < firstTrainLoss);
}


int main() {
    UNITY_BEGIN();
    RUN_TEST(testMnistSmoke_FullTrainingPipelineReducesLoss);
    return UNITY_END();
}
