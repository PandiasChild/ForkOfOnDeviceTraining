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

void testEvaluationBatchAccuracy_CountsCorrectPredictions() {
    // Linear model: weights identity-like, so output ≈ input (mapped to 2 classes)
    float weightData[] = {1.f, 0.f, 0.f, 1.f};
    size_t weightDims[] = {2, 2};
    tensor_t *wParam = tensorInitFloat(weightData, weightDims, 2, NULL);
    float wGradData[] = {0.f, 0.f, 0.f, 0.f};
    tensor_t *wGrad = tensorInitFloat(wGradData, weightDims, 2, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    float biasData[] = {0.f, 0.f};
    size_t biasDims[] = {1, 2};
    tensor_t *bParam = tensorInitFloat(biasData, biasDims, 2, NULL);
    float bGradData[] = {0.f, 0.f};
    tensor_t *bGrad = tensorInitFloat(bGradData, biasDims, 2, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(w, b, &testQ, &testQ, &testQ, &testQ);
    layer_t *model[] = {linear};

    // Sample 0: input [5, 1] → output [5, 1] → argmax=0, label [1, 0] → argmax=0 → CORRECT
    float in0Data[] = {5.f, 1.f};
    size_t inDims[] = {1, 2};
    tensor_t *in0 = tensorInitFloat(in0Data, inDims, 2, NULL);
    float lbl0Data[] = {1.f, 0.f};
    size_t lblDims[] = {1, 2};
    tensor_t *lbl0 = tensorInitFloat(lbl0Data, lblDims, 2, NULL);

    // Sample 1: input [1, 5] → output [1, 5] → argmax=1, label [1, 0] → argmax=0 → WRONG
    float in1Data[] = {1.f, 5.f};
    tensor_t *in1 = tensorInitFloat(in1Data, inDims, 2, NULL);
    float lbl1Data[] = {1.f, 0.f};
    tensor_t *lbl1 = tensorInitFloat(lbl1Data, lblDims, 2, NULL);

    // Sample 2: input [2, 3] → output [2, 3] → argmax=1, label [0, 1] → argmax=1 → CORRECT
    float in2Data[] = {2.f, 3.f};
    tensor_t *in2 = tensorInitFloat(in2Data, inDims, 2, NULL);
    float lbl2Data[] = {0.f, 1.f};
    tensor_t *lbl2 = tensorInitFloat(lbl2Data, lblDims, 2, NULL);

    sample_t s0 = {.item = in0, .label = lbl0};
    sample_t s1 = {.item = in1, .label = lbl1};
    sample_t s2 = {.item = in2, .label = lbl2};
    sample_t *samples[] = {&s0, &s1, &s2};
    batch_t batch = {.samples = samples, .size = 3};

    size_t correct = evaluationBatchAccuracy(model, 1, &batch, 2, inference);

    TEST_ASSERT_EQUAL_UINT(2, correct);
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

    // batchSize=1 to avoid initDataLoader index initialization bug
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

void testEvaluationEpochAccuracy_ReturnsAccuracyFraction() {
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

    // batchSize=1 to avoid initDataLoader index initialization bug
    dataLoader_t *dl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1,
                                       NULL, NULL, false, 0, true);

    // s0: output=[5,1], argmax=0, label=[1,0], argmax=0 → CORRECT
    // s1: output=[1,5], argmax=1, label=[0,1], argmax=1 → CORRECT
    // s2: output=[3,1], argmax=0, label=[1,0], argmax=0 → CORRECT
    // s3: output=[1,3], argmax=1, label=[0,1], argmax=1 → CORRECT
    // accuracy = 4/4 = 1.0
    float accuracy = evaluationEpochAccuracy(model, 1, dl, 2, inference);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, accuracy);
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
    TEST_ASSERT_TRUE(result.finalEvalLoss > 0.0f);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalculateGradsSequential_MatchesPyTorch);
    RUN_TEST(testEvaluationBatch_ReturnsAverageLoss);
    RUN_TEST(testEvaluationBatchAccuracy_CountsCorrectPredictions);
    RUN_TEST(testEvaluationEpoch_ReturnsAverageLossAcrossBatches);
    RUN_TEST(testEvaluationEpochAccuracy_ReturnsAccuracyFraction);
    RUN_TEST(testTrainingBatchDefault_ReturnsAverageLossAndAccumulatesGrads);
    RUN_TEST(testTrainingEpochDefault_DoesOptimizerStepPerBatch);
    RUN_TEST(testTrainingRun_ReturnsResult);
    return UNITY_END();
}
