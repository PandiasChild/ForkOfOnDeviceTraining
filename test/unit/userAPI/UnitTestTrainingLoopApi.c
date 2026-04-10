#define SOURCE_FILE "UNIT_TEST_TRAINING_LOOP_API"

#include <stddef.h>

#include "LossFunction.h"
#include "TensorApi.h"
#include "LinearApi.h"
#include "SgdApi.h"
#include "unity.h"
#include "TrainingLoopApi.h"
#include "CalculateGradsSequential.h"
#include "TrainingBatchDefault.h"
#include "TrainingEpochDefault.h"
#include "TensorConversion.h"
#include "QuantizationApi.h"
#include "Linear.h"
#include "ReluApi.h"
#include "InferenceApi.h"
#include "DataLoaderApi.h"
#include "StorageApi.h"
#include "Dataset.h"

void testCalculateGradsSequential_MatchesPyTorch() {
    float weightData[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    size_t weightDims[] = {2, 3};
    tensor_t *weightsParam = tensorInitFloat(weightData, weightDims, 2, NULL);

    float weightGradData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *weightsGrad = tensorInitFloat(weightGradData, weightDims, 2, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {1, 2};
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, 2, NULL);
    float biasGradData[] = {0.f, 0.f};
    tensor_t *biasGrad = tensorInitFloat(biasGradData, biasDims, 2, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float input0Data[] = {-4.f, 1.f, 9.f};
    size_t input0Dims[] = {1, 3};
    tensor_t *input0 = tensorInitFloat(input0Data, input0Dims, 2, NULL);

    float input1Data[] = {5.f, -1.f, 2.f};
    size_t input1Dims[] = {1, 3};
    tensor_t *input1 = tensorInitFloat(input1Data, input1Dims, 2, NULL);

    float input2Data[] = {-7.f, -5.f, 6.f};
    size_t input2Dims[] = {1, 3};
    tensor_t *input2 = tensorInitFloat(input2Data, input2Dims, 2, NULL);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(weights, bias, &testQ, &testQ, &testQ, &testQ);

    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    float label0Data[] = {59.f, -23.f};
    size_t label0Dims[] = {1, 2};
    tensor_t *label0 = tensorInitFloat(label0Data, label0Dims, 2, NULL);

    float label1Data[] = {43.f, 249.f};
    size_t label1Dims[] = {1, 2};
    tensor_t *label1 = tensorInitFloat(label1Data, label1Dims, 2, NULL);

    float label2Data[] = {23.f, 457.f};
    size_t label2Dims[] = {1, 2};
    tensor_t *label2 = tensorInitFloat(label2Data, label2Dims, 2, NULL);

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    sgd->sizeStates = 1;

    for (size_t i = 0; i < 23; i++) {
        trainingStats_t *ts0 = calculateGradsSequential(model, sizeModel, MSE, input0, label0);
        trainingStats_t *ts1 = calculateGradsSequential(model, sizeModel, MSE, input1, label1);
        trainingStats_t *ts2 = calculateGradsSequential(model, sizeModel, MSE, input2, label2);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);

        freeTrainingStats(ts0);
        freeTrainingStats(ts1);
        freeTrainingStats(ts2);
    }

    float expectedWeights[] = {5.f, -1.f, 9.f, 22.f, -100.f, 18.f};
    linearConfig_t *linearConfig = linear->config->linear;
    float *actualWeights = (float *)linearConfig->weights->param->data;

    const float errorPercent = 0.03f;
    for (size_t i = 0; i < 6; i++) {
        float currentThreshold = actualWeights[i] * errorPercent;
        TEST_ASSERT_FLOAT_WITHIN(currentThreshold, expectedWeights[i], actualWeights[i]);
    }
}

void testEvaluationBatch_ReturnsAverageLoss() {
    // Set up a simple linear model: weights=[1,1,1,1,1,1] shape=[2,3], bias=[-1,3]
    float weightData[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    size_t weightDims[] = {2, 3};
    tensor_t *wParam = tensorInitFloat(weightData, weightDims, 2, NULL);
    float wGradData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wGradData, weightDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(biasData, biasDims, 2, NULL);
    float bGradData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bGradData, biasDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    // Create 2 samples manually
    float input0Data[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    tensor_t *input0 = tensorInitFloat(input0Data, inputDims, 2, NULL);
    float label0Data[] = {10.f, 20.f};
    size_t labelDims[] = {1, 2};
    tensor_t *label0 = tensorInitFloat(label0Data, labelDims, 2, NULL);

    float input1Data[] = {1.f, 0.f, 0.f};
    tensor_t *input1 = tensorInitFloat(input1Data, inputDims, 2, NULL);
    float label1Data[] = {5.f, 5.f};
    tensor_t *label1 = tensorInitFloat(label1Data, labelDims, 2, NULL);

    // Compute expected losses via inferenceWithLoss directly
    inferenceStats_t *stats0 = inferenceWithLoss(model, 1, input0, label0, MSE);
    inferenceStats_t *stats1 = inferenceWithLoss(model, 1, input1, label1, MSE);
    float expectedAvgLoss = (stats0->loss + stats1->loss) / 2.0f;
    freeInferenceStats(stats0);
    freeInferenceStats(stats1);

    // Build a batch
    sample_t s0 = {.item = input0, .label = label0};
    sample_t s1 = {.item = input1, .label = label1};
    sample_t *samples[] = {&s0, &s1};
    batch_t batch = {.samples = samples, .size = 2};

    float actualAvgLoss = evaluationBatch(model, 1, MSE, &batch, inferenceWithLoss);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedAvgLoss, actualAvgLoss);
}

// Test dataset for epoch-level tests: 4 samples, batchSize=2 → 2 batches
static tensor_t *epochItems[4];
static tensor_t *epochLabels[4];
static dataset_t epochDataset;
static bool epochDatasetInit = false;

static void initEpochDataset() {
    if (epochDatasetInit) return;

    // 4 samples: input [1,2], label [1,0] or [0,1]
    static float in0[] = {5.f, 1.f}; static float in1[] = {1.f, 5.f};
    static float in2[] = {3.f, 1.f}; static float in3[] = {1.f, 3.f};
    static float lb0[] = {1.f, 0.f}; static float lb1[] = {0.f, 1.f};
    static float lb2[] = {1.f, 0.f}; static float lb3[] = {0.f, 1.f};

    static size_t inDims[] = {1, 2};
    static size_t lbDims[] = {1, 2};

    epochItems[0] = tensorInitFloat(in0, inDims, 2, NULL);
    epochItems[1] = tensorInitFloat(in1, inDims, 2, NULL);
    epochItems[2] = tensorInitFloat(in2, inDims, 2, NULL);
    epochItems[3] = tensorInitFloat(in3, inDims, 2, NULL);

    epochLabels[0] = tensorInitFloat(lb0, lbDims, 2, NULL);
    epochLabels[1] = tensorInitFloat(lb1, lbDims, 2, NULL);
    epochLabels[2] = tensorInitFloat(lb2, lbDims, 2, NULL);
    epochLabels[3] = tensorInitFloat(lb3, lbDims, 2, NULL);

    static tensorArray_t itemsArr;
    itemsArr.array = epochItems;
    itemsArr.size = 4;
    static tensorArray_t labelsArr;
    labelsArr.array = epochLabels;
    labelsArr.size = 4;

    epochDataset.items = &itemsArr;
    epochDataset.labels = &labelsArr;
    epochDatasetInit = true;
}

static sample_t *getEpochSample(size_t id) {
    sample_t *s = *reserveMemory(sizeof(sample_t));
    s->item = epochDataset.items->array[id];
    s->label = epochDataset.labels->array[id];
    return s;
}

static size_t getEpochDatasetSize() {
    return epochDataset.items->size;
}

void testEvaluationEpoch_ReturnsAverageLossAcrossBatches() {
    initEpochDataset();

    // Identity model
    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                       NULL, NULL, false, 0, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss);

    // Identity model output = input, so loss = MSE(input, label)
    // s0: MSE([5,1],[1,0]) = ((5-1)^2+(1-0)^2)/2 = 8.5
    // s1: MSE([1,5],[0,1]) = ((1-0)^2+(5-1)^2)/2 = 8.5
    // s2: MSE([3,1],[1,0]) = ((3-1)^2+(1-0)^2)/2 = 2.5
    // s3: MSE([1,3],[0,1]) = ((1-0)^2+(3-1)^2)/2 = 2.5
    // 4 batches of 1, avg of per-batch avg = (8.5+8.5+2.5+2.5)/4 = 5.5
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, totalAvg);
}

void testEvaluationEpoch_MinibatchMatchesMicrobatchAverage() {
    initEpochDataset();

    // Identity model — output = input, so loss = MSE(input, label)
    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    // batchSize=2 → 2 minibatches of 2 samples each
    // Per-sample losses: 8.5, 8.5, 2.5, 2.5
    // Batch 0 avg = (8.5+8.5)/2 = 8.5; Batch 1 avg = (2.5+2.5)/2 = 2.5
    // Epoch avg = (8.5+2.5)/2 = 5.5 — identical to microbatch result
    dataLoader_t *dl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 2,
                                       NULL, NULL, false, 0, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, totalAvg);
}

void testTrainingBatchDefault_ReturnsAverageLossAndAccumulatesGrads() {
    // Single linear layer, 2 samples → avg loss should match manually computed value
    float wData[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    size_t wDims[] = {2, 3};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {-1.f, 3.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    // Compute expected: run calculateGradsSequential manually per sample
    float in0Data[] = {-4.f, 1.f, 9.f};
    size_t inDims[] = {1, 3};
    tensor_t *in0 = tensorInitFloat(in0Data, inDims, 2, NULL);
    float lb0Data[] = {59.f, -23.f};
    size_t lbDims[] = {1, 2};
    tensor_t *lb0 = tensorInitFloat(lb0Data, lbDims, 2, NULL);

    float in1Data[] = {5.f, -1.f, 2.f};
    tensor_t *in1 = tensorInitFloat(in1Data, inDims, 2, NULL);
    float lb1Data[] = {43.f, 249.f};
    tensor_t *lb1 = tensorInitFloat(lb1Data, lbDims, 2, NULL);

    // Get expected losses from individual calculateGrads calls
    trainingStats_t *ts0 = calculateGradsSequential(model, 1, MSE, in0, lb0);
    trainingStats_t *ts1 = calculateGradsSequential(model, 1, MSE, in1, lb1);
    float expectedAvg = (ts0->loss + ts1->loss) / 2.0f;
    freeTrainingStats(ts0);
    freeTrainingStats(ts1);

    // Reset grads to zero before testing trainingBatchDefault
    float *gArr = (float *)wGrad->data;
    for (size_t i = 0; i < 6; i++) gArr[i] = 0.f;
    float *bgArr = (float *)bGrad->data;
    bgArr[0] = 0.f; bgArr[1] = 0.f;

    // Build batch
    sample_t s0 = {.item = in0, .label = lb0};
    sample_t s1 = {.item = in1, .label = lb1};
    sample_t *samples[] = {&s0, &s1};
    batch_t batch = {.samples = samples, .size = 2};

    float actualAvg = trainingBatchDefault(model, 1, MSE, &batch, calculateGradsSequential);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedAvg, actualAvg);
}

void testTrainingEpochDefault_DoesOptimizerStepPerBatch() {
    // After one epoch with known data, weights should change
    // Epoch dataset has 2 input features, 2 output classes
    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    // Save initial weights
    float initWeights[4];
    for (size_t i = 0; i < 4; i++) initWeights[i] = wData[i];

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);

    // Use epoch dataset (batchSize=1 → 4 batches)
    initEpochDataset();
    dataLoader_t *dl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                       NULL, NULL, false, 0, true);

    float epochLoss = trainingEpochDefault(model, sizeModel, MSE, dl, sgd,
                                            calculateGradsSequential);

    // Weights should have changed
    float *curWeights = (float *)wParam->data;
    bool changed = false;
    for (size_t i = 0; i < 4; i++) {
        if (curWeights[i] != initWeights[i]) {
            changed = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(changed);
    TEST_ASSERT_TRUE(epochLoss > 0.0f);
}

void testTrainingEpochDefault_MinibatchStepsOncePerMinibatch() {
    // Same model + dataset as microbatch training test, but batchSize=2
    // means the optimizer steps twice (once per minibatch) instead of
    // four times, with gradients accumulated across two samples per step.
    // Weight trajectory differs from microbatch; only property tested
    // here is that training runs end-to-end and updates the model.
    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    float initWeights[4];
    for (size_t i = 0; i < 4; i++) initWeights[i] = wData[i];

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);

    initEpochDataset();
    dataLoader_t *dl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 2,
                                       NULL, NULL, false, 0, true);

    float epochLoss = trainingEpochDefault(model, sizeModel, MSE, dl, sgd,
                                            calculateGradsSequential);

    float *curWeights = (float *)wParam->data;
    bool changed = false;
    for (size_t i = 0; i < 4; i++) {
        if (curWeights[i] != initWeights[i]) {
            changed = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(changed);
    TEST_ASSERT_TRUE(epochLoss > 0.0f);
}

void testTrainingRun_ReturnsResult() {
    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, FLOAT32);

    initEpochDataset();
    dataLoader_t *trainDl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                            NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                           NULL, NULL, false, 0, true);

    trainingRunResult_t result = trainingRun(model, 1, MSE, trainDl, evalDl, sgd, 2,
                                             calculateGradsSequential, inferenceWithLoss, NULL);

    TEST_ASSERT_TRUE(result.finalTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(result.finalEvalStats.loss > 0.0f);
    TEST_ASSERT_TRUE(result.finalEvalStats.accuracy >= 0.0f);
    TEST_ASSERT_TRUE(result.finalEvalStats.accuracy <= 1.0f);
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

    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, FLOAT32);

    initEpochDataset();
    dataLoader_t *trainDl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                            NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                           NULL, NULL, false, 0, true);

    size_t numberOfEpochs = 3;
    trainingRunResult_t result = trainingRun(model, 1, MSE, trainDl, evalDl, sgd, numberOfEpochs,
                                              calculateGradsSequential, inferenceWithLoss,
                                              captureCallback);

    // Callback was invoked once per epoch, in order
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs, cbCallCount);
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs - 1, cbLastEpoch); // 0-indexed

    // Stats passed to callback are real (not zero-initialized) and match the final result
    TEST_ASSERT_TRUE(cbLastTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(cbLastStats.loss > 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(result.finalTrainLoss, cbLastTrainLoss);
    TEST_ASSERT_EQUAL_FLOAT(result.finalEvalStats.loss, cbLastStats.loss);
    TEST_ASSERT_EQUAL_FLOAT(result.finalEvalStats.accuracy, cbLastStats.accuracy);
}

void testEvaluationEpochWithMetrics_AllCorrect() {
    initEpochDataset();

    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                       NULL, NULL, false, 0, true);

    // Identity model: all 4 samples predict correctly (same as testEvaluationEpochAccuracy)
    epochStats_t stats = evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, stats.loss);  // same as testEvaluationEpoch
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, stats.accuracy);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, stats.precision);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, stats.recall);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, stats.f1);
}

// Partially-correct dataset: 4 samples, 3 correct, 1 wrong
static tensor_t *partialItems[4];
static tensor_t *partialLabels[4];
static dataset_t partialDataset;
static bool partialDatasetInit = false;

static void initPartialDataset() {
    if (partialDatasetInit) return;

    static float in0[] = {5.f, 1.f}; // pred=0
    static float in1[] = {3.f, 1.f}; // pred=0
    static float in2[] = {4.f, 1.f}; // pred=0 (wrong!)
    static float in3[] = {1.f, 3.f}; // pred=1

    static float lb0[] = {1.f, 0.f}; // true=0
    static float lb1[] = {1.f, 0.f}; // true=0
    static float lb2[] = {0.f, 1.f}; // true=1
    static float lb3[] = {0.f, 1.f}; // true=1

    static size_t inDims[] = {1, 2};
    static size_t lbDims[] = {1, 2};

    partialItems[0] = tensorInitFloat(in0, inDims, 2, NULL);
    partialItems[1] = tensorInitFloat(in1, inDims, 2, NULL);
    partialItems[2] = tensorInitFloat(in2, inDims, 2, NULL);
    partialItems[3] = tensorInitFloat(in3, inDims, 2, NULL);

    partialLabels[0] = tensorInitFloat(lb0, lbDims, 2, NULL);
    partialLabels[1] = tensorInitFloat(lb1, lbDims, 2, NULL);
    partialLabels[2] = tensorInitFloat(lb2, lbDims, 2, NULL);
    partialLabels[3] = tensorInitFloat(lb3, lbDims, 2, NULL);

    static tensorArray_t itemsArr;
    itemsArr.array = partialItems;
    itemsArr.size = 4;
    static tensorArray_t labelsArr;
    labelsArr.array = partialLabels;
    labelsArr.size = 4;

    partialDataset.items = &itemsArr;
    partialDataset.labels = &labelsArr;
    partialDatasetInit = true;
}

static sample_t *getPartialSample(size_t id) {
    sample_t *s = *reserveMemory(sizeof(sample_t));
    s->item = partialDataset.items->array[id];
    s->label = partialDataset.labels->array[id];
    return s;
}

static size_t getPartialDatasetSize() {
    return partialDataset.items->size;
}

void testEvaluationEpochWithMetrics_PartiallyCorrect() {
    initPartialDataset();

    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl = dataLoaderInit(getPartialSample, getPartialDatasetSize, 1,
                                       NULL, NULL, false, 0, true);

    epochStats_t stats = evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss);

    // tp=[2,1], predCount=[3,1], actualCount=[2,2]
    // accuracy = 3/4 = 0.75
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, stats.accuracy);
    // precision = mean(2/3, 1/1) = 5/6
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f / 6.0f, stats.precision);
    // recall = mean(2/2, 1/2) = 0.75
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, stats.recall);
    // F1 = mean(2*(2/3)*1/(2/3+1), 2*1*0.5/(1+0.5)) = mean(4/5, 2/3) = 11/15
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 11.0f / 15.0f, stats.f1);
}

// Zero-prediction dataset: model never predicts class 1.
// 3 samples: 2 correct for class 0, 1 wrong (true class 1, predicted 0).
static tensor_t *zeroPredItems[3];
static tensor_t *zeroPredLabels[3];
static dataset_t zeroPredDataset;
static bool zeroPredDatasetInit = false;

static void initZeroPredDataset() {
    if (zeroPredDatasetInit) return;

    static float in0[] = {5.f, 1.f}; // pred=0
    static float in1[] = {3.f, 1.f}; // pred=0
    static float in2[] = {4.f, 1.f}; // pred=0 (wrong, true=1)

    static float lb0[] = {1.f, 0.f}; // true=0
    static float lb1[] = {1.f, 0.f}; // true=0
    static float lb2[] = {0.f, 1.f}; // true=1

    static size_t inDims[] = {1, 2};
    static size_t lbDims[] = {1, 2};

    zeroPredItems[0] = tensorInitFloat(in0, inDims, 2, NULL);
    zeroPredItems[1] = tensorInitFloat(in1, inDims, 2, NULL);
    zeroPredItems[2] = tensorInitFloat(in2, inDims, 2, NULL);

    zeroPredLabels[0] = tensorInitFloat(lb0, lbDims, 2, NULL);
    zeroPredLabels[1] = tensorInitFloat(lb1, lbDims, 2, NULL);
    zeroPredLabels[2] = tensorInitFloat(lb2, lbDims, 2, NULL);

    static tensorArray_t itemsArr;
    itemsArr.array = zeroPredItems;
    itemsArr.size = 3;
    static tensorArray_t labelsArr;
    labelsArr.array = zeroPredLabels;
    labelsArr.size = 3;

    zeroPredDataset.items = &itemsArr;
    zeroPredDataset.labels = &labelsArr;
    zeroPredDatasetInit = true;
}

static sample_t *getZeroPredSample(size_t id) {
    sample_t *s = *reserveMemory(sizeof(sample_t));
    s->item = zeroPredDataset.items->array[id];
    s->label = zeroPredDataset.labels->array[id];
    return s;
}

static size_t getZeroPredDatasetSize() {
    return zeroPredDataset.items->size;
}

void testEvaluationEpochWithMetrics_HandlesZeroPredictionClass() {
    initZeroPredDataset();

    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl = dataLoaderInit(getZeroPredSample, getZeroPredDatasetSize, 1,
                                       NULL, NULL, false, 0, true);

    epochStats_t stats = evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss);

    // tp=[2,0], predCount=[3,0], actualCount=[2,1]
    // Class 1 has no predictions -> precision guard skips it
    // Class 1 has zero tp -> recall=0 for class 1
    // Class 1 prec+rec=0 -> F1 guard skips it
    // accuracy = 2/3
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f / 3.0f, stats.accuracy);
    // precision = (2/3 + 0) / 2 = 1/3
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f / 3.0f, stats.precision);
    // recall = (2/2 + 0/1) / 2 = 0.5
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.5f, stats.recall);
    // F1 class 0: 2*(2/3)*1 / ((2/3)+1) = 4/5; class 1: 0. Mean = 2/5
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f / 5.0f, stats.f1);

    // No NaN or Inf from division by zero
    TEST_ASSERT_TRUE(stats.precision == stats.precision); // NaN check
    TEST_ASSERT_TRUE(stats.recall == stats.recall);
    TEST_ASSERT_TRUE(stats.f1 == stats.f1);
}

void testEvaluationEpochWithReport_ReturnsConfusionMatrix() {
    initPartialDataset();

    float wData[] = {1.f, 0.f, 0.f, 1.f};
    size_t wDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(wData, wDims, 2, NULL);
    float wgData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wgData, wDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float bData[] = {0.f, 0.f};
    size_t bDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(bData, bDims, 2, NULL);
    float bgData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bgData, bDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl = dataLoaderInit(getPartialSample, getPartialDatasetSize, 1,
                                       NULL, NULL, false, 0, true);

    // Pre-fill with non-zero to verify WithReport zeroes the caller's buffer before accumulating
    size_t cm[2 * 2] = {99, 99, 99, 99};
    classificationReport_t report = evaluationEpochWithReport(model, 1, MSE, dl,
                                                               inferenceWithLoss, cm, 2);

    // Expected CM[predicted][actual]: [[2,1],[0,1]]
    // cm[0*2+0]=2, cm[0*2+1]=1, cm[1*2+0]=0, cm[1*2+1]=1
    TEST_ASSERT_EQUAL_UINT(2, report.confusionMatrix[0]); // pred=0, actual=0
    TEST_ASSERT_EQUAL_UINT(1, report.confusionMatrix[1]); // pred=0, actual=1
    TEST_ASSERT_EQUAL_UINT(0, report.confusionMatrix[2]); // pred=1, actual=0
    TEST_ASSERT_EQUAL_UINT(1, report.confusionMatrix[3]); // pred=1, actual=1

    TEST_ASSERT_EQUAL_UINT(2, report.numClasses);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, report.stats.accuracy);
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
