#define SOURCE_FILE "UNIT_TEST_TRAINING_LOOP_API"

#include <stddef.h>

#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "InferenceApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TrainingBatchDefault.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"
#include "unity.h"

/* Build a fresh 2-D float tensor from a literal float buffer. Encapsulates the
 * post-#106 chain so each test stays readable. */
static tensor_t *buildFloatTensor2D(size_t d0, size_t d1, const float *src, size_t count) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, src, count);
    return t;
}

void testCalculateGradsSequential_MatchesPyTorch() {
    tensor_t *weightsParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    tensor_t *weightsGrad = gradInitFloat(weightsParam, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    tensor_t *biasParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(weights, bias, &testQ, &testQ, &testQ, &testQ);

    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    tensor_t *input0 = buildFloatTensor2D(1, 3, (float[]){-4.f, 1.f, 9.f}, 3);
    tensor_t *input1 = buildFloatTensor2D(1, 3, (float[]){5.f, -1.f, 2.f}, 3);
    tensor_t *input2 = buildFloatTensor2D(1, 3, (float[]){-7.f, -5.f, 6.f}, 3);

    tensor_t *label0 = buildFloatTensor2D(1, 2, (float[]){59.f, -23.f}, 2);
    tensor_t *label1 = buildFloatTensor2D(1, 2, (float[]){43.f, 249.f}, 2);
    tensor_t *label2 = buildFloatTensor2D(1, 2, (float[]){23.f, 457.f}, 2);

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    /* Pre-existing test hack: only step weights, leaving bias unchanged across
     * iterations. We restore sizeStates to 2 before the free below so
     * freeOptimSgdM cascades to BOTH registered parameters and their state
     * buffers (otherwise parameter[1]/states[1] would leak). */
    sgd->sizeStates = 1;

    for (size_t i = 0; i < 23; i++) {
        trainingStats_t *ts0 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, 1,
            input0, label0);
        trainingStats_t *ts1 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, 1,
            input1, label1);
        trainingStats_t *ts2 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, 1,
            input2, label2);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);

        freeTrainingStats(ts0);
        freeTrainingStats(ts1);
        freeTrainingStats(ts2);
    }

    /* CAPTURE assertion values into stack locals BEFORE any free. */
    float expectedWeights[] = {5.f, -1.f, 9.f, 22.f, -100.f, 18.f};
    linearConfig_t *linearConfig = linear->config->linear;
    float capturedActualWeights[6];
    {
        float *actualWeights = (float *)linearConfig->weights->param->data;
        for (size_t i = 0; i < 6; i++) {
            capturedActualWeights[i] = actualWeights[i];
        }
    }

    /* FREE in reverse-init order. Restore sizeStates so freeOptimSgdM cascades
     * to both weights and bias parameters + their state buffers. */
    sgd->sizeStates = 2;
    freeOptimSgdM(sgd);
    freeTensor(label2);
    freeTensor(label1);
    freeTensor(label0);
    freeTensor(input2);
    freeTensor(input1);
    freeTensor(input0);
    freeLinearLayer(linear);

    /* ASSERT on captured. */
    const float errorPercent = 0.03f;
    for (size_t i = 0; i < 6; i++) {
        float currentThreshold = capturedActualWeights[i] * errorPercent;
        TEST_ASSERT_FLOAT_WITHIN(currentThreshold, expectedWeights[i], capturedActualWeights[i]);
    }
}

void testEvaluationBatch_ReturnsAverageLoss() {
    /* Set up a simple linear model: weights=[1,1,1,1,1,1] shape=[2,3], bias=[-1,3] */
    tensor_t *wParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    /* Create 2 samples manually */
    tensor_t *input0 = buildFloatTensor2D(1, 3, (float[]){0.f, 1.f, 2.f}, 3);
    tensor_t *label0 = buildFloatTensor2D(1, 2, (float[]){10.f, 20.f}, 2);
    tensor_t *input1 = buildFloatTensor2D(1, 3, (float[]){1.f, 0.f, 0.f}, 3);
    tensor_t *label1 = buildFloatTensor2D(1, 2, (float[]){5.f, 5.f}, 2);

    /* Compute expected losses via inferenceWithLoss directly */
    inferenceStats_t *stats0 = inferenceWithLoss(model, 1, input0, label0, MSE);
    inferenceStats_t *stats1 = inferenceWithLoss(model, 1, input1, label1, MSE);
    float expectedAvgLoss = (stats0->loss + stats1->loss) / 2.0f;
    freeInferenceStats(stats0);
    freeInferenceStats(stats1);

    /* Build a batch. Samples must be heap-allocated: evaluationBatch frees
     * each via freeSample (the batch-consumer convention set by #119). */
    sample_t *s0 = reserveMemory(sizeof(sample_t));
    s0->item = input0;
    s0->label = label0;
    sample_t *s1 = reserveMemory(sizeof(sample_t));
    s1->item = input1;
    s1->label = label1;
    sample_t *samples[] = {s0, s1};
    batch_t batch = {.samples = samples, .size = 2};

    float actualAvgLoss = evaluationBatch(model, 1, MSE, &batch, inferenceWithLoss);

    /* CAPTURE. */
    float capturedExpected = expectedAvgLoss;
    float capturedActual = actualAvgLoss;

    /* FREE in reverse-init order. evaluationBatch consumed s0/s1 (freed via
     * freeSample inside the production loop). The tensors they referenced
     * (input0/label0/input1/label1) are still ours to free. */
    freeTensor(label1);
    freeTensor(input1);
    freeTensor(label0);
    freeTensor(input0);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);

    /* ASSERT on captured. */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, capturedExpected, capturedActual);
}

/* Test dataset for epoch-level tests: 4 samples, batchSize=2 → 2 batches.
 * File-scope so the dataLoader callback can reach it. Each test that calls
 * initEpochDataset() must call freeEpochDataset() at end so the per-test
 * teardown idiom holds (no shared lazy-init/never-free state).
 *
 * Mirrors the dataset pattern in UnitTestMnistSmoke.c (#120). */
static tensor_t *epochItems[4];
static tensor_t *epochLabels[4];
static dataset_t epochDataset;
static tensorArray_t epochItemsArr;
static tensorArray_t epochLabelsArr;
static bool epochDatasetInit = false;

static const float epochItemDataLiteral[4][2] = {
    {5.f, 1.f},
    {1.f, 5.f},
    {3.f, 1.f},
    {1.f, 3.f},
};
static const float epochLabelDataLiteral[4][2] = {
    {1.f, 0.f},
    {0.f, 1.f},
    {1.f, 0.f},
    {0.f, 1.f},
};

static void initEpochDataset() {
    if (epochDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        epochItems[i] = buildFloatTensor2D(1, 2, epochItemDataLiteral[i], 2);
        epochLabels[i] = buildFloatTensor2D(1, 2, epochLabelDataLiteral[i], 2);
    }

    epochItemsArr.array = epochItems;
    epochItemsArr.size = 4;
    epochLabelsArr.array = epochLabels;
    epochLabelsArr.size = 4;

    epochDataset.items = &epochItemsArr;
    epochDataset.labels = &epochLabelsArr;
    epochDatasetInit = true;
}

static void freeEpochDataset() {
    if (!epochDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        freeTensor(epochItems[i]);
        freeTensor(epochLabels[i]);
    }
    epochDatasetInit = false;
}

static sample_t *getEpochSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = epochDataset.items->array[id];
    s->label = epochDataset.labels->array[id];
    return s;
}

static size_t getEpochDatasetSize() {
    return epochDataset.items->size;
}

void testEvaluationEpoch_ReturnsAverageLossAcrossBatches() {
    initEpochDataset();

    /* Identity model */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss);

    /* CAPTURE. */
    float capturedTotalAvg = totalAvg;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    /* Identity model output = input, so loss = MSE(input, label)
     * s0: MSE([5,1],[1,0]) = ((5-1)^2+(1-0)^2)/2 = 8.5
     * s1: MSE([1,5],[0,1]) = ((1-0)^2+(5-1)^2)/2 = 8.5
     * s2: MSE([3,1],[1,0]) = ((3-1)^2+(1-0)^2)/2 = 2.5
     * s3: MSE([1,3],[0,1]) = ((1-0)^2+(3-1)^2)/2 = 2.5
     * 4 batches of 1, avg of per-batch avg = (8.5+8.5+2.5+2.5)/4 = 5.5 */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, capturedTotalAvg);
}

void testEvaluationEpoch_MinibatchMatchesMicrobatchAverage() {
    initEpochDataset();

    /* Identity model — output = input, so loss = MSE(input, label) */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    /* batchSize=2 → 2 minibatches of 2 samples each
     * Per-sample losses: 8.5, 8.5, 2.5, 2.5
     * Batch 0 avg = (8.5+8.5)/2 = 8.5; Batch 1 avg = (2.5+2.5)/2 = 2.5
     * Epoch avg = (8.5+2.5)/2 = 5.5 — identical to microbatch result */
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, 0, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss);

    /* CAPTURE. */
    float capturedTotalAvg = totalAvg;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, capturedTotalAvg);
}

void testTrainingBatchDefault_ReturnsAverageLossAndAccumulatesGrads() {
    /* Single linear layer, 2 samples → avg loss should match manually computed value */
    tensor_t *wParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    /* Compute expected: run calculateGradsSequential manually per sample */
    tensor_t *in0 = buildFloatTensor2D(1, 3, (float[]){-4.f, 1.f, 9.f}, 3);
    tensor_t *lb0 = buildFloatTensor2D(1, 2, (float[]){59.f, -23.f}, 2);
    tensor_t *in1 = buildFloatTensor2D(1, 3, (float[]){5.f, -1.f, 2.f}, 3);
    tensor_t *lb1 = buildFloatTensor2D(1, 2, (float[]){43.f, 249.f}, 2);

    /* Get expected losses from individual calculateGrads calls */
    trainingStats_t *ts0 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, 1, in0, lb0);
    trainingStats_t *ts1 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, 1, in1, lb1);
    float expectedAvg = (ts0->loss + ts1->loss) / 2.0f;
    freeTrainingStats(ts0);
    freeTrainingStats(ts1);

    /* Reset grads to zero before testing trainingBatchDefault */
    float *gArr = (float *)wGrad->data;
    for (size_t i = 0; i < 6; i++) {
        gArr[i] = 0.f;
    }
    float *bgArr = (float *)bGrad->data;
    bgArr[0] = 0.f;
    bgArr[1] = 0.f;

    /* Build batch. Samples must be heap-allocated: trainingBatchDefault frees
     * each via freeSample (the batch-consumer convention set by #119). */
    sample_t *s0 = reserveMemory(sizeof(sample_t));
    s0->item = in0;
    s0->label = lb0;
    sample_t *s1 = reserveMemory(sizeof(sample_t));
    s1->item = in1;
    s1->label = lb1;
    sample_t *samples[] = {s0, s1};
    batch_t batch = {.samples = samples, .size = 2};

    float actualAvg =
        trainingBatchDefault(model, 1, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM},
                             &batch, calculateGradsSequential);

    /* CAPTURE. */
    float capturedExpected = expectedAvg;
    float capturedActual = actualAvg;

    /* FREE. */
    freeTensor(lb1);
    freeTensor(in1);
    freeTensor(lb0);
    freeTensor(in0);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);

    /* ASSERT. */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, capturedExpected, capturedActual);
}

void testTrainingEpochDefault_DoesOptimizerStepPerBatch() {
    /* After one epoch with known data, weights should change.
     * Epoch dataset has 2 input features, 2 output classes. */
    static const float wInitData[4] = {1.f, 0.f, 0.f, 1.f};
    tensor_t *wParam = buildFloatTensor2D(2, 2, wInitData, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    /* Save initial weights */
    float initWeights[4];
    for (size_t i = 0; i < 4; i++) {
        initWeights[i] = wInitData[i];
    }

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);

    /* Use epoch dataset (batchSize=1 → 4 batches) */
    initEpochDataset();
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    float epochLoss = trainingEpochDefault(
        model, sizeModel, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, dl, sgd,
        calculateGradsSequential);

    /* CAPTURE assertion values BEFORE any free. */
    bool capturedChanged = false;
    {
        float *curWeights = (float *)wParam->data;
        for (size_t i = 0; i < 4; i++) {
            if (curWeights[i] != initWeights[i]) {
                capturedChanged = true;
                break;
            }
        }
    }
    float capturedEpochLoss = epochLoss;

    /* FREE. freeOptimSgdM cascades to w and b parameters; do NOT also free
     * those (would double-free). */
    freeOptimSgdM(sgd);
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedChanged);
    TEST_ASSERT_TRUE(capturedEpochLoss > 0.0f);
}

void testTrainingEpochDefault_MinibatchStepsOncePerMinibatch() {
    /* Same model + dataset as microbatch training test, but batchSize=2
     * means the optimizer steps twice (once per minibatch) instead of
     * four times, with gradients accumulated across two samples per step.
     * Weight trajectory differs from microbatch; only property tested
     * here is that training runs end-to-end and updates the model. */
    static const float wInitData[4] = {1.f, 0.f, 0.f, 1.f};
    tensor_t *wParam = buildFloatTensor2D(2, 2, wInitData, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    float initWeights[4];
    for (size_t i = 0; i < 4; i++) {
        initWeights[i] = wInitData[i];
    }

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);

    initEpochDataset();
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, 0, true);

    float epochLoss = trainingEpochDefault(
        model, sizeModel, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, dl, sgd,
        calculateGradsSequential);

    /* CAPTURE. */
    bool capturedChanged = false;
    {
        float *curWeights = (float *)wParam->data;
        for (size_t i = 0; i < 4; i++) {
            if (curWeights[i] != initWeights[i]) {
                capturedChanged = true;
                break;
            }
        }
    }
    float capturedEpochLoss = epochLoss;

    /* FREE. */
    freeOptimSgdM(sgd);
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedChanged);
    TEST_ASSERT_TRUE(capturedEpochLoss > 0.0f);
}

void testTrainingRun_ReturnsResult() {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, FLOAT32);

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, trainDl,
                    evalDl, sgd, 2, calculateGradsSequential, inferenceWithLoss, NULL);

    /* CAPTURE. */
    float capturedFinalTrainLoss = result.finalTrainLoss;
    float capturedEvalLoss = result.finalEvalStats.loss;
    float capturedAccuracy = result.finalEvalStats.accuracy;

    /* FREE. */
    freeOptimSgdM(sgd);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayer(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedFinalTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedEvalLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy >= 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy <= 1.0f);
}

static size_t cbCallCount;
static size_t cbLastEpoch;
static float cbLastTrainLoss;
static epochStats_t cbLastStats;

static void captureCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    cbCallCount++;
    cbLastEpoch = epoch;
    cbLastTrainLoss = trainLoss;
    cbLastStats = evalStats;
}

void testTrainingRun_CallsCallbackEachEpochWithStats() {
    cbCallCount = 0;
    cbLastEpoch = 0;
    cbLastTrainLoss = 0.0f;
    cbLastStats = (epochStats_t){0};

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, FLOAT32);

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    size_t numberOfEpochs = 3;
    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .reduction = REDUCTION_SUM}, trainDl, evalDl, sgd,
        numberOfEpochs, calculateGradsSequential, inferenceWithLoss, captureCallback);

    /* CAPTURE. */
    size_t capturedCbCallCount = cbCallCount;
    size_t capturedCbLastEpoch = cbLastEpoch;
    float capturedCbLastTrainLoss = cbLastTrainLoss;
    float capturedCbLastStatsLoss = cbLastStats.loss;
    float capturedCbLastStatsAccuracy = cbLastStats.accuracy;
    float capturedFinalTrainLoss = result.finalTrainLoss;
    float capturedFinalEvalLoss = result.finalEvalStats.loss;
    float capturedFinalEvalAccuracy = result.finalEvalStats.accuracy;

    /* FREE. */
    freeOptimSgdM(sgd);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayer(linear);
    freeEpochDataset();

    /* ASSERT. Callback was invoked once per epoch, in order. */
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs, capturedCbCallCount);
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs - 1, capturedCbLastEpoch); /* 0-indexed */

    /* Stats passed to callback are real (not zero-initialized) and match the final result. */
    TEST_ASSERT_TRUE(capturedCbLastTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedCbLastStatsLoss > 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(capturedFinalTrainLoss, capturedCbLastTrainLoss);
    TEST_ASSERT_EQUAL_FLOAT(capturedFinalEvalLoss, capturedCbLastStatsLoss);
    TEST_ASSERT_EQUAL_FLOAT(capturedFinalEvalAccuracy, capturedCbLastStatsAccuracy);
}

void testEvaluationEpochWithMetrics_AllCorrect() {
    initEpochDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    /* Identity model: all 4 samples predict correctly (same as testEvaluationEpochAccuracy) */
    epochStats_t stats = evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss);

    /* CAPTURE. */
    float capturedLoss = stats.loss;
    float capturedAccuracy = stats.accuracy;
    float capturedPrecision = stats.precision;
    float capturedRecall = stats.recall;
    float capturedF1 = stats.f1;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, capturedLoss); /* same as testEvaluationEpoch */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedAccuracy);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedPrecision);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedRecall);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedF1);
}

/* Partially-correct dataset: 4 samples, 3 correct, 1 wrong */
static tensor_t *partialItems[4];
static tensor_t *partialLabels[4];
static dataset_t partialDataset;
static tensorArray_t partialItemsArr;
static tensorArray_t partialLabelsArr;
static bool partialDatasetInit = false;

static const float partialItemDataLiteral[4][2] = {
    {5.f, 1.f}, /* pred=0 */
    {3.f, 1.f}, /* pred=0 */
    {4.f, 1.f}, /* pred=0 (wrong!) */
    {1.f, 3.f}, /* pred=1 */
};
static const float partialLabelDataLiteral[4][2] = {
    {1.f, 0.f}, /* true=0 */
    {1.f, 0.f}, /* true=0 */
    {0.f, 1.f}, /* true=1 */
    {0.f, 1.f}, /* true=1 */
};

static void initPartialDataset() {
    if (partialDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        partialItems[i] = buildFloatTensor2D(1, 2, partialItemDataLiteral[i], 2);
        partialLabels[i] = buildFloatTensor2D(1, 2, partialLabelDataLiteral[i], 2);
    }

    partialItemsArr.array = partialItems;
    partialItemsArr.size = 4;
    partialLabelsArr.array = partialLabels;
    partialLabelsArr.size = 4;

    partialDataset.items = &partialItemsArr;
    partialDataset.labels = &partialLabelsArr;
    partialDatasetInit = true;
}

static void freePartialDataset() {
    if (!partialDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        freeTensor(partialItems[i]);
        freeTensor(partialLabels[i]);
    }
    partialDatasetInit = false;
}

static sample_t *getPartialSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = partialDataset.items->array[id];
    s->label = partialDataset.labels->array[id];
    return s;
}

static size_t getPartialDatasetSize() {
    return partialDataset.items->size;
}

void testEvaluationEpochWithMetrics_PartiallyCorrect() {
    initPartialDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getPartialSample, getPartialDatasetSize, 1, NULL, NULL, false, 0, true);

    epochStats_t stats = evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss);

    /* CAPTURE. */
    float capturedAccuracy = stats.accuracy;
    float capturedPrecision = stats.precision;
    float capturedRecall = stats.recall;
    float capturedF1 = stats.f1;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);
    freePartialDataset();

    /* ASSERT.
     * tp=[2,1], predCount=[3,1], actualCount=[2,2]
     * accuracy = 3/4 = 0.75 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, capturedAccuracy);
    /* precision = mean(2/3, 1/1) = 5/6 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f / 6.0f, capturedPrecision);
    /* recall = mean(2/2, 1/2) = 0.75 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, capturedRecall);
    /* F1 = mean(2*(2/3)*1/(2/3+1), 2*1*0.5/(1+0.5)) = mean(4/5, 2/3) = 11/15 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 11.0f / 15.0f, capturedF1);
}

/* Zero-prediction dataset: model never predicts class 1.
 * 3 samples: 2 correct for class 0, 1 wrong (true class 1, predicted 0). */
static tensor_t *zeroPredItems[3];
static tensor_t *zeroPredLabels[3];
static dataset_t zeroPredDataset;
static tensorArray_t zeroPredItemsArr;
static tensorArray_t zeroPredLabelsArr;
static bool zeroPredDatasetInit = false;

static const float zeroPredItemDataLiteral[3][2] = {
    {5.f, 1.f}, /* pred=0 */
    {3.f, 1.f}, /* pred=0 */
    {4.f, 1.f}, /* pred=0 (wrong, true=1) */
};
static const float zeroPredLabelDataLiteral[3][2] = {
    {1.f, 0.f}, /* true=0 */
    {1.f, 0.f}, /* true=0 */
    {0.f, 1.f}, /* true=1 */
};

static void initZeroPredDataset() {
    if (zeroPredDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 3; i++) {
        zeroPredItems[i] = buildFloatTensor2D(1, 2, zeroPredItemDataLiteral[i], 2);
        zeroPredLabels[i] = buildFloatTensor2D(1, 2, zeroPredLabelDataLiteral[i], 2);
    }

    zeroPredItemsArr.array = zeroPredItems;
    zeroPredItemsArr.size = 3;
    zeroPredLabelsArr.array = zeroPredLabels;
    zeroPredLabelsArr.size = 3;

    zeroPredDataset.items = &zeroPredItemsArr;
    zeroPredDataset.labels = &zeroPredLabelsArr;
    zeroPredDatasetInit = true;
}

static void freeZeroPredDataset() {
    if (!zeroPredDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 3; i++) {
        freeTensor(zeroPredItems[i]);
        freeTensor(zeroPredLabels[i]);
    }
    zeroPredDatasetInit = false;
}

static sample_t *getZeroPredSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = zeroPredDataset.items->array[id];
    s->label = zeroPredDataset.labels->array[id];
    return s;
}

static size_t getZeroPredDatasetSize() {
    return zeroPredDataset.items->size;
}

void testEvaluationEpochWithMetrics_HandlesZeroPredictionClass() {
    initZeroPredDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getZeroPredSample, getZeroPredDatasetSize, 1, NULL, NULL, false, 0, true);

    epochStats_t stats = evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss);

    /* CAPTURE. */
    float capturedAccuracy = stats.accuracy;
    float capturedPrecision = stats.precision;
    float capturedRecall = stats.recall;
    float capturedF1 = stats.f1;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);
    freeZeroPredDataset();

    /* ASSERT.
     * tp=[2,0], predCount=[3,0], actualCount=[2,1]
     * Class 1 has no predictions -> precision guard skips it
     * Class 1 has zero tp -> recall=0 for class 1
     * Class 1 prec+rec=0 -> F1 guard skips it
     * accuracy = 2/3 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f / 3.0f, capturedAccuracy);
    /* precision = (2/3 + 0) / 2 = 1/3 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f / 3.0f, capturedPrecision);
    /* recall = (2/2 + 0/1) / 2 = 0.5 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.5f, capturedRecall);
    /* F1 class 0: 2*(2/3)*1 / ((2/3)+1) = 4/5; class 1: 0. Mean = 2/5 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f / 5.0f, capturedF1);

    /* No NaN or Inf from division by zero */
    TEST_ASSERT_TRUE(capturedPrecision == capturedPrecision); /* NaN check */
    TEST_ASSERT_TRUE(capturedRecall == capturedRecall);
    TEST_ASSERT_TRUE(capturedF1 == capturedF1);
}

void testEvaluationEpochWithReport_ReturnsConfusionMatrix() {
    initPartialDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getPartialSample, getPartialDatasetSize, 1, NULL, NULL, false, 0, true);

    /* Pre-fill with non-zero to verify WithReport zeroes the caller's buffer before accumulating */
    size_t cm[2 * 2] = {99, 99, 99, 99};
    classificationReport_t report =
        evaluationEpochWithReport(model, 1, MSE, dl, inferenceWithLoss, cm, 2);

    /* CAPTURE. */
    size_t capturedCM[4];
    for (size_t i = 0; i < 4; i++) {
        capturedCM[i] = report.confusionMatrix[i];
    }
    size_t capturedNumClasses = report.numClasses;
    float capturedAccuracy = report.stats.accuracy;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayer(linear);
    freeParameter(b);
    freeParameter(w);
    freePartialDataset();

    /* ASSERT.
     * Expected CM[predicted][actual]: [[2,1],[0,1]]
     * cm[0*2+0]=2, cm[0*2+1]=1, cm[1*2+0]=0, cm[1*2+1]=1 */
    TEST_ASSERT_EQUAL_UINT(2, capturedCM[0]); /* pred=0, actual=0 */
    TEST_ASSERT_EQUAL_UINT(1, capturedCM[1]); /* pred=0, actual=1 */
    TEST_ASSERT_EQUAL_UINT(0, capturedCM[2]); /* pred=1, actual=0 */
    TEST_ASSERT_EQUAL_UINT(1, capturedCM[3]); /* pred=1, actual=1 */

    TEST_ASSERT_EQUAL_UINT(2, capturedNumClasses);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, capturedAccuracy);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalculateGradsSequential_MatchesPyTorch);
    RUN_TEST(testEvaluationBatch_ReturnsAverageLoss);
    RUN_TEST(testEvaluationEpoch_ReturnsAverageLossAcrossBatches);
    RUN_TEST(testEvaluationEpoch_MinibatchMatchesMicrobatchAverage);
    RUN_TEST(testTrainingBatchDefault_ReturnsAverageLossAndAccumulatesGrads);
    RUN_TEST(testTrainingEpochDefault_DoesOptimizerStepPerBatch);
    RUN_TEST(testTrainingEpochDefault_MinibatchStepsOncePerMinibatch);
    RUN_TEST(testTrainingRun_ReturnsResult);
    RUN_TEST(testEvaluationEpochWithMetrics_AllCorrect);
    RUN_TEST(testEvaluationEpochWithMetrics_PartiallyCorrect);
    RUN_TEST(testEvaluationEpochWithMetrics_HandlesZeroPredictionClass);
    RUN_TEST(testEvaluationEpochWithReport_ReturnsConfusionMatrix);
    RUN_TEST(testTrainingRun_CallsCallbackEachEpochWithStats);
    return UNITY_END();
}
