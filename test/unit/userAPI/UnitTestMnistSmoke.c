#define SOURCE_FILE "UNIT_TEST_MNIST_SMOKE"

#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include "unity.h"

#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "InferenceApi.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

void setUp() {}
void tearDown() {}

#define INPUT_DIM 4
#define HIDDEN_DIM 8
#define NUM_CLASSES 3
#define DATASET_SIZE 6
#define MODEL_SIZE 4

/* Tiny trivially-learnable dataset: each input is a noisy one-hot prototype of
 * its class. The arrays are file-scope because the getSample callback (fired
 * during trainingRun) must reach them; freeDataset releases them at end. */
static tensor_t *items[DATASET_SIZE];
static tensor_t *labels[DATASET_SIZE];

static const float itemDataLiteral[DATASET_SIZE][INPUT_DIM] = {
    {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f},
    {0.9f, 0.1f, 0.0f, 0.0f}, {0.1f, 0.9f, 0.0f, 0.0f}, {0.0f, 0.1f, 0.9f, 0.0f},
};
static const float labelDataLiteral[DATASET_SIZE][NUM_CLASSES] = {
    {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
    {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
};

static void initDataset() {
    for (size_t i = 0; i < DATASET_SIZE; i++) {
        /* Item tensor (1, INPUT_DIM). */
        size_t *itemDims = reserveMemory(2 * sizeof(size_t));
        itemDims[0] = 1;
        itemDims[1] = INPUT_DIM;
        size_t *itemOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, itemOrder);
        shape_t *itemShape = reserveMemory(sizeof(shape_t));
        setShape(itemShape, itemDims, 2, itemOrder);
        items[i] = initTensor(itemShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(items[i], itemDataLiteral[i], INPUT_DIM);

        /* Label tensor (1, NUM_CLASSES). */
        size_t *labelDims = reserveMemory(2 * sizeof(size_t));
        labelDims[0] = 1;
        labelDims[1] = NUM_CLASSES;
        size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, labelOrder);
        shape_t *labelShape = reserveMemory(sizeof(shape_t));
        setShape(labelShape, labelDims, 2, labelOrder);
        labels[i] = initTensor(labelShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(labels[i], labelDataLiteral[i], NUM_CLASSES);
    }
}

static void freeDataset() {
    for (size_t i = 0; i < DATASET_SIZE; i++) {
        freeTensor(items[i]);
        freeTensor(labels[i]);
    }
}

static sample_t *getSample(size_t id) {
    sample_t *sample = reserveMemory(sizeof(sample_t));
    sample->item = items[id];
    sample->label = labels[id];
    return sample;
}

static size_t getDatasetSize() {
    return DATASET_SIZE;
}

/* Builds a 4-layer Linear -> ReLU -> Linear -> Softmax model. The shared
 * layer-level quantization q is exposed via q_out so the test can free it
 * after the model layers are freed. Each parameter tensor takes its own
 * fresh quantization (initTensor takes ownership), distinct from q. */
static void buildModel(layer_t **model, quantization_t **q_out) {
    quantization_t *q = quantizationInitFloat();
    *q_out = q;

    distribution_t xavier0 = {
        .type = XAVIER_UNIFORM,
        .params.xavier = {.gain = 1.0f, .fanIn = INPUT_DIM, .fanOut = HIDDEN_DIM}};
    distribution_t zeros = {.type = ZEROS};

    /* w0 (HIDDEN_DIM x INPUT_DIM, XAVIER_UNIFORM). */
    size_t *w0Dims = reserveMemory(2 * sizeof(size_t));
    w0Dims[0] = HIDDEN_DIM;
    w0Dims[1] = INPUT_DIM;
    size_t *w0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w0Order);
    shape_t *w0Shape = reserveMemory(sizeof(shape_t));
    setShape(w0Shape, w0Dims, 2, w0Order);
    tensor_t *w0Param = initTensor(w0Shape, quantizationInitFloat(), NULL);
    initDistribution(w0Param, &xavier0);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    /* b0 (1 x HIDDEN_DIM, ZEROS). */
    size_t *b0Dims = reserveMemory(2 * sizeof(size_t));
    b0Dims[0] = 1;
    b0Dims[1] = HIDDEN_DIM;
    size_t *b0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b0Order);
    shape_t *b0Shape = reserveMemory(sizeof(shape_t));
    setShape(b0Shape, b0Dims, 2, b0Order);
    tensor_t *b0Param = initTensor(b0Shape, quantizationInitFloat(), NULL);
    initDistribution(b0Param, &zeros);
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    model[0] = linearLayerInitLegacy(w0, b0, q, q, q, q);
    model[1] = reluLayerInitLegacy(q, q);

    distribution_t xavier1 = {
        .type = XAVIER_UNIFORM,
        .params.xavier = {.gain = 1.0f, .fanIn = HIDDEN_DIM, .fanOut = NUM_CLASSES}};

    /* w1 (NUM_CLASSES x HIDDEN_DIM, XAVIER_UNIFORM). */
    size_t *w1Dims = reserveMemory(2 * sizeof(size_t));
    w1Dims[0] = NUM_CLASSES;
    w1Dims[1] = HIDDEN_DIM;
    size_t *w1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w1Order);
    shape_t *w1Shape = reserveMemory(sizeof(shape_t));
    setShape(w1Shape, w1Dims, 2, w1Order);
    tensor_t *w1Param = initTensor(w1Shape, quantizationInitFloat(), NULL);
    initDistribution(w1Param, &xavier1);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    /* b1 (1 x NUM_CLASSES, ZEROS). */
    size_t *b1Dims = reserveMemory(2 * sizeof(size_t));
    b1Dims[0] = 1;
    b1Dims[1] = NUM_CLASSES;
    size_t *b1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b1Order);
    shape_t *b1Shape = reserveMemory(sizeof(shape_t));
    setShape(b1Shape, b1Dims, 2, b1Order);
    tensor_t *b1Param = initTensor(b1Shape, quantizationInitFloat(), NULL);
    initDistribution(b1Param, &zeros);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    model[2] = linearLayerInitLegacy(w1, b1, q, q, q, q);
    model[3] = softmaxLayerInitLegacy(q, q);
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

    dataLoader_t *trainDl =
        dataLoaderInit(getSample, getDatasetSize, 2, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);

    layer_t *model[MODEL_SIZE];
    quantization_t *q = NULL;
    buildModel(model, &q);

    optimizer_t *sgd = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, MODEL_SIZE, FLOAT32);

    cbInvocations = 0;
    size_t numberOfEpochs = 20;
    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN}, trainDl,
        evalDl, sgd, numberOfEpochs, calculateGradsSequential, inferenceWithLoss, captureEpoch);

    /* CAPTURE all assertion values into stack locals BEFORE any free. */
    size_t capturedCbInvocations = cbInvocations;
    float capturedFinalTrainLoss = result.finalTrainLoss;
    float capturedAccuracy = result.finalEvalStats.accuracy;
    float capturedFirstTrainLoss = firstTrainLoss;
    float capturedLastTrainLoss = lastTrainLoss;

    /* FREE in reverse-init order.
     * NOTE: freeOptimSgdM cascades to all model parameters via freeParameter.
     * Do NOT also call freeParameter on w0/b0/w1/b1 — would be a double-free. */
    freeOptimSgdM(sgd);
    freeSoftmaxLayerLegacy(model[3]);
    freeLinearLayerLegacy(model[2]);
    freeReluLayerLegacy(model[1]);
    freeLinearLayerLegacy(model[0]);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeQuantization(q);
    freeDataset();

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs, capturedCbInvocations);
    TEST_ASSERT_TRUE(capturedFinalTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy >= 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy <= 1.0f);
    TEST_ASSERT_TRUE(capturedLastTrainLoss < capturedFirstTrainLoss);
}

/* Regression test for #94. The original audit observed a silent exit(1) on
 * macOS when an snprintf or gmtime_r call sat between init() and trainingRun()
 * in the base project; #94 hypothesised an uninitialized-read or sized-buffer
 * overrun in a static-init path whose visible symptom was gated on stack/dyld
 * layout. This test mirrors the reproducer pattern (full FLOAT32 setup,
 * gmtime_r + snprintf, then trainingRun) on the in-tree smoke pipeline. If the
 * underlying memory bug is still present, trainingRun terminates with exit(1)
 * before reaching any assertion; reaching the final TEST_ASSERT_TRUE proves
 * the run completed normally. */
void testMnistSmoke_SnprintfGmtimeRBetweenSetupAndTrainingRun_NoSilentExit() {
    initDataset();

    dataLoader_t *trainDl =
        dataLoaderInit(getSample, getDatasetSize, 2, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);

    layer_t *model[MODEL_SIZE];
    quantization_t *q = NULL;
    buildModel(model, &q);

    optimizer_t *sgd = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, MODEL_SIZE, FLOAT32);

    /* The poison block from #94's reproducer: gmtime_r + snprintf wedged
     * between setup and trainingRun. Pre-fix this triggered silent exit(1). */
    char buf[64];
    time_t t = time(NULL);
    struct tm tmStruct;
    gmtime_r(&t, &tmStruct);
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tmStruct.tm_year + 1900, tmStruct.tm_mon + 1,
             tmStruct.tm_mday);

    cbInvocations = 0;
    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN}, trainDl,
        evalDl, sgd, 1, calculateGradsSequential, inferenceWithLoss, captureEpoch);

    /* CAPTURE before free. */
    size_t capturedCbInvocations = cbInvocations;
    float capturedFinalTrainLoss = result.finalTrainLoss;
    char capturedFirstChar = buf[0];

    freeOptimSgdM(sgd);
    freeSoftmaxLayerLegacy(model[3]);
    freeLinearLayerLegacy(model[2]);
    freeReluLayerLegacy(model[1]);
    freeLinearLayerLegacy(model[0]);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeQuantization(q);
    freeDataset();

    /* Reaching these at all proves trainingRun did not silent-exit. */
    TEST_ASSERT_EQUAL_UINT(1, capturedCbInvocations);
    TEST_ASSERT_TRUE(capturedFinalTrainLoss > 0.0f);
    TEST_ASSERT_NOT_EQUAL('\0', capturedFirstChar);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testMnistSmoke_FullTrainingPipelineReducesLoss);
    RUN_TEST(testMnistSmoke_SnprintfGmtimeRBetweenSetupAndTrainingRun_NoSilentExit);
    return UNITY_END();
}
