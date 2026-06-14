#define SOURCE_FILE "har_classifier_train_c"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "AvgPool1d.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "Distributions.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "NPYLoaderApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

#include "../../src/userApi/tensor/include/TensorApi.h"
#include "npy_writer.h"

#define EPOCHS 20
#define BATCH 64
#define LEARNING_RATE 0.01f
#define MOMENTUM 0.9f
#define SEED 42
#define SHUFFLE_SEED 42
#define NUM_CLASSES 6

#define IN_CHANNELS 9
#define LEN_INPUT 128


#define L1_OUT 32
#define WEIGHT_NDIM_0 2
#define WEIGHT_NDIM_1 2
#define BIAS_NDIM_0 1
#define BIAS_NDIM_1 1
// #define L2_OUT NUM_CLASSES
#define FLATTEN_SIZE (IN_CHANNELS * LEN_INPUT)
#define MODEL_SIZE 5
#define ROUNDING_MODE HALF_AWAY

// har_classifier: FINAL test_loss=0.3498 test_acc=0.9046


/* ------------------------------------------------------------------------- */
/* Datasets and dataloader thunks (mirrors example/MnistExperiment.c).       */
/* ------------------------------------------------------------------------- */

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/*! @brief Per-sample shape after npyLoad strips the leading N dim is [9, 128] (rank-2)
 * for items and rank-0 (single int32 value) for labels. The C model expects
 * rank-3 inputs [B=1, 9, 128] for Conv1d and rank-1 one-hot float labels [6]
 * for CrossEntropy. We rebuild both at load time.
 *
 * @param[in/out] items to reshape
 * */
static void reshapeItemsAddBatchDim(tensorArray_t *items) {
    /* items->array[i] currently has shape [9, 128] rank-2. Replace with
     * [1, 9, 128] rank-3. Data layout is row-major and unchanged, so we only
     * need to swap the shape header (allocate new dims/order arrays of length 3,
     * free the old ones). */
    for (size_t i = 0; i < items->size; ++i) {
        tensor_t *t = items->array[i];
        size_t oldRank = t->shape->numberOfDimensions;
        size_t newRank = oldRank + 1;

        size_t *newDims = reserveMemory(newRank * sizeof(size_t));
        size_t *newOrder = reserveMemory(newRank * sizeof(size_t));
        newDims[0] = 1;
        for (size_t d = 0; d < oldRank; ++d) {
            newDims[d + 1] = t->shape->dimensions[d];
        }
        for (size_t d = 0; d < newRank; ++d) {
            newOrder[d] = d;
        }

        freeReservedMemory(t->shape->dimensions);
        freeReservedMemory(t->shape->orderOfDimensions);
        t->shape->dimensions = newDims;
        t->shape->orderOfDimensions = newOrder;
        t->shape->numberOfDimensions = newRank;
    }
}
static tensorArray_t *buildOneHotLabels(tensorArray_t *intLabels) {
    /* intLabels->array[i] is a rank-0 int32 tensor (single class index 0..5).
     * We allocate a brand-new tensorArray_t whose entries are rank-1 float32
     * one-hot tensors of shape [NUM_CLASSES]. The original int32 array is
     * left intact (caller still owns it). */
    tensorArray_t *out = reserveMemory(sizeof(tensorArray_t));
    tensor_t **arr = reserveMemory(intLabels->size * sizeof(tensor_t *));
    out->array = arr;
    out->size = intLabels->size;

    for (size_t i = 0; i < intLabels->size; ++i) {
        size_t *dims = reserveMemory(1 * sizeof(size_t));
        size_t *order = reserveMemory(1 * sizeof(size_t));
        dims[0] = NUM_CLASSES;
        order[0] = 0;
        shape_t *shape = reserveMemory(sizeof(shape_t));
        shape->dimensions = dims;
        shape->orderOfDimensions = order;
        shape->numberOfDimensions = 1;

        quantization_t *q = quantizationInitFloat();
        tensor_t *t = initTensor(shape, q, NULL);

        int32_t cls = ((int32_t *)intLabels->array[i]->data)[0];
        float *data = (float *)t->data;
        for (size_t c = 0; c < NUM_CLASSES; ++c) {
            data[c] = (c == (size_t)cls) ? 1.0f : 0.0f;
        }
        arr[i] = t;
    }
    return out;
}



static void initDataSets(void) {
    tensorArray_t *trainItems = npyLoad("examples/har_classifier/data/train_x.npy");
    tensorArray_t *trainLabelsRaw = npyLoad("examples/har_classifier/data/train_y.npy");
    reshapeItemsAddBatchDim(trainItems);
    g_trainDataset.items = trainItems;
    g_trainDataset.labels = buildOneHotLabels(trainLabelsRaw);
    freeTensorArray(trainLabelsRaw);

    tensorArray_t *valItems = npyLoad("examples/har_classifier/data/val_x.npy");
    tensorArray_t *valLabelsRaw = npyLoad("examples/har_classifier/data/val_y.npy");
    reshapeItemsAddBatchDim(valItems);
    g_valDataset.items = valItems;
    g_valDataset.labels = buildOneHotLabels(valLabelsRaw);
    freeTensorArray(valLabelsRaw);

    tensorArray_t *testItems = npyLoad("examples/har_classifier/data/test_x.npy");
    tensorArray_t *testLabelsRaw = npyLoad("examples/har_classifier/data/test_y.npy");
    reshapeItemsAddBatchDim(testItems);
    g_testDataset.items = testItems;
    g_testDataset.labels = buildOneHotLabels(testLabelsRaw);
    freeTensorArray(testLabelsRaw);
}

static sample_t *getTrainSample(size_t id) {
    return npyGetSample(&g_trainDataset, id);
}
static sample_t *getValSample(size_t id) {
    return npyGetSample(&g_valDataset, id);
}
static sample_t *getTestSample(size_t id) {
    return npyGetSample(&g_testDataset, id);
}

static size_t getTrainSize(void) {
    return g_trainDataset.items->size;
}
static size_t getValSize(void) {
    return g_valDataset.items->size;
}
static size_t getTestSize(void) {
    return g_testDataset.items->size;
}

/* ------------------------------------------------------------------------- */
/* Model parameters (file-static — must outlive buildModel).                 */
/* ------------------------------------------------------------------------- */

static float weight0Data[L1_OUT*FLATTEN_SIZE] = {0};
static size_t weight0Dims[WEIGHT_NDIM_0] = {L1_OUT, FLATTEN_SIZE};

static float bias0Data[L1_OUT] = {0};
static size_t bias0Dims[BIAS_NDIM_0] = {L1_OUT};

static float weight1Data[NUM_CLASSES*L1_OUT] = {0};
static size_t weight1Dims[WEIGHT_NDIM_1] = {NUM_CLASSES, L1_OUT};

static float bias1Data[NUM_CLASSES] = {0};
static size_t bias1Dims[BIAS_NDIM_1] = {NUM_CLASSES};

static parameter_t *buildParam(distributionType_t dist, quantization_t *q, float *data, size_t *dims, size_t ndim) {
    size_t *order = reserveMemory(ndim * sizeof(size_t));
    size_t *orderFloat = reserveMemory(ndim * sizeof(size_t));
    size_t *dimensionsFloat = reserveMemory(ndim * sizeof(size_t));

    if( order == NULL || orderFloat == NULL || dimensionsFloat == NULL){
        PRINT_ERROR("Memory Allocation Failed");
        exit(1);
    }
    for (size_t i = 0; i < ndim; i++) {
        order[i] = i;
        orderFloat[i] = i;
        dimensionsFloat[i] = dims[i];
    }

    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape_t *shapeFloat = reserveMemory(sizeof(shape_t));
    if( shape == NULL || shapeFloat == NULL){
        PRINT_ERROR("Memory Allocation Failed");
        exit(1);
    }

    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = ndim;

    shapeFloat->dimensions = dimensionsFloat;
    shapeFloat->orderOfDimensions = orderFloat;
    shapeFloat->numberOfDimensions = ndim;

    quantization_t *floatq = quantizationInitFloat();
    tensor_t *floatTensor = initTensor(shapeFloat, floatq, NULL);
    initDistribution(floatTensor, &dist);

    tensor_t *p = initTensor(shape, q, NULL);
    convertTensor(floatTensor, p);
    freeTensor(floatTensor);
    printf("alles in ordnung\n");
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}


static void buildModel(layer_t **model) {
    uint8_t qBits = 16;
    roundingMode_t roundingMode = ROUNDING_MODE;
    uint8_t deltabits = qBits - 2;
    quantization_t *q0 = quantizationInitSymInt32(roundingMode);
    quantization_t *q1 = quantizationInitSymInt32(roundingMode);
    quantization_t *q2 = quantizationInitSymInt32(roundingMode);
    quantization_t *q3 = quantizationInitSymInt32(roundingMode);
    quantization_t *q4 = quantizationInitSymInt32(roundingMode);
    quantization_t *q5 = quantizationInitSymInt32(roundingMode);
    quantization_t *q6 = quantizationInitSymInt32(roundingMode);
    quantization_t *q7 = quantizationInitSymInt32(roundingMode);
    quantization_t *q8 = quantizationInitSymInt32(roundingMode);
    quantization_t *q9 = quantizationInitSymInt32(roundingMode);
    quantization_t *q10 = quantizationInitSymInt32(roundingMode);
    quantization_t *q11 = quantizationInitSymInt32(roundingMode);

    quantization_t *dq1 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
    quantization_t *dq2 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
    quantization_t *dq3 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
    quantization_t *dq4 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);

    quantization_t *q12 = quantizationInitSymInt32(roundingMode);
    quantization_t *q13 = quantizationInitSymInt32(roundingMode);
    quantization_t *q14 = quantizationInitSymInt32(roundingMode);
    quantization_t *q15 = quantizationInitSymInt32(roundingMode);


    // 128 -> 1152
    model[0] = flattenLayerInit();

    // Linear 128→32
    /* parameter_t *weight0 = buildParam(XAVIER_UNIFORM, dq1, weight0Data, weight0Dims, WEIGHT_NDIM_0);
    parameter_t *bias0 = buildParam(ZEROS, dq2, bias0Data, bias0Dims, BIAS_NDIM_0);
*/
    parameter_t *weight0 = buildParam(XAVIER_UNIFORM, q14, weight0Data, weight0Dims, WEIGHT_NDIM_0);
    parameter_t *bias0 = buildParam(ZEROS, q15, bias0Data, bias0Dims, BIAS_NDIM_0);
    model[1] = linearLayerInitLegacy(weight0, bias0, q0, q1, q2, q3);

    // ReLU
    model[2] = reluLayerInitLegacy(q4, q5);
    // Linear 32→6
    /* parameter_t *weight1 = buildParam(XAVIER_UNIFORM, dq3, weight1Data, weight1Dims, WEIGHT_NDIM_1);
    parameter_t *bias1 = buildParam(ZEROS, dq4, bias1Data, bias1Dims, BIAS_NDIM_1);
    */
    parameter_t *weight1 = buildParam(XAVIER_UNIFORM, q12, weight1Data, weight1Dims, WEIGHT_NDIM_1);
    parameter_t *bias1 = buildParam(ZEROS, q13, bias1Data, bias1Dims, BIAS_NDIM_1);

    model[3] = linearLayerInitLegacy(weight1, bias1, q6, q7, q8, q9);
    // Softmax
    model[4] = softmaxLayerInitLegacy(q10, q11);
}


/* ------------------------------------------------------------------------- */
/* Per-epoch JSON log writer + epoch callback.                               */
/* ------------------------------------------------------------------------- */

static FILE *g_log_file = NULL;
static int g_first_epoch = 1;
static struct timespec g_epoch_t0;

static void epochCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall_s =
        (double)(t1.tv_sec - g_epoch_t0.tv_sec) + (double)(t1.tv_nsec - g_epoch_t0.tv_nsec) * 1e-9;

    if (!g_first_epoch) {
        fprintf(g_log_file, ",\n");
    }
    fprintf(g_log_file,
            "    {\"epoch\": %zu, \"step_losses\": [], \"train_loss\": %.6f, "
            "\"val_loss\": %.6f, \"val_acc\": %.6f, \"wall_s\": %.4f}",
            epoch, (double)trainLoss, (double)evalStats.loss, (double)evalStats.accuracy, wall_s);
    fflush(g_log_file);
    g_first_epoch = 0;

    fprintf(stdout, "epoch %zu: train_loss=%.4f val_loss=%.4f val_acc=%.4f wall_s=%.2f\n", epoch,
            (double)trainLoss, (double)evalStats.loss, (double)evalStats.accuracy, wall_s);
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
}

static int ensureDir(const char *p) {
    if (mkdir(p, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
        return 0;
    }
    if (errno == EEXIST) {
        return 0;
    }
    fprintf(stderr, "ERROR: cannot create %s: %s\n", p, strerror(errno));
    return 1;
}

int main(void) {
    if (ensureDir("examples/har_classifier/logs") != 0) {
        return 1;
    }
    if (ensureDir("examples/har_classifier/outputs") != 0) {
        return 1;
    }

    initDataSets();

    dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                                               /*shuffle*/ true, /*shuffleSeed*/ SHUFFLE_SEED,
                                               /*dropLast*/ true);
    dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                             /*shuffle*/ false, /*shuffleSeed*/ 0,
                                             /*dropLast*/ true);
    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);

    layer_t *model[MODEL_SIZE];
    printf("main: start buildModel\n");
    buildModel(model);
    printf("main: start sgdMCreateOptim\n");
    optimizer_t *sgd =
        sgdMCreateOptim(LEARNING_RATE, MOMENTUM, /*weightDecay*/ 0.0f, model, MODEL_SIZE, SYM_INT32);

    g_log_file = fopen("examples/har_classifier/logs/c.json", "w");
    if (!g_log_file) {
        fprintf(stderr, "ERROR: cannot open log file for writing\n");
        return 1;
    }
    fprintf(g_log_file,
            "{\n"
            "  \"impl\": \"c\",\n"
            "  \"example\": \"har_classifier\",\n"
            "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
            "\"momentum\": %.6f, \"seed\": %d, \"shuffle_seed\": %d},\n"
            "  \"epochs\": [\n",
            EPOCHS, BATCH, (double)LEARNING_RATE, (double)MOMENTUM, SEED, SHUFFLE_SEED);
    fflush(g_log_file);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
    printf("main: start trainingRun\n");
    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd, EPOCHS, calculateGradsSequential, inferenceWithLoss,
        epochCallback);
    printf("main: trainingRun done\n");
    (void)result;

    epochStats_t testStats = evaluationEpochWithMetrics(
        model, MODEL_SIZE, CROSS_ENTROPY, testLoader, inferenceWithLoss, REDUCTION_MEAN);

    fprintf(g_log_file,
            "\n  ],\n"
            "  \"final\": {\"test_loss\": %.6f, \"test_acc\": %.6f, "
            "\"test_auc\": null}\n"
            "}\n",
            (double)testStats.loss, (double)testStats.accuracy);
    fclose(g_log_file);

    fprintf(stdout, "FINAL test_loss=%.4f test_acc=%.4f\n", (double)testStats.loss,
            (double)testStats.accuracy);

    /* Predictions: run inference on every test sample, write argmax to .npy.
     * inference() returns a fresh tensor we own; freeing every iteration via
     * freeTensor would also free its data buffer, which is what we want. */
    size_t numTest = getTestSize();
    int32_t *predictions = malloc(numTest * sizeof(int32_t));
    if (!predictions) {
        fprintf(stderr, "OOM allocating predictions\n");
        return 1;
    }

    for (size_t i = 0; i < numTest; ++i) {
        sample_t *s = getTestSample(i);
        tensor_t *out = inference(model, MODEL_SIZE, s->item);
        float *probs = (float *)out->data;
        size_t argmax = 0;
        float best = probs[0];
        for (size_t c = 1; c < NUM_CLASSES; ++c) {
            if (probs[c] > best) {
                best = probs[c];
                argmax = c;
            }
        }
        predictions[i] = (int32_t)argmax;
        freeTensor(out);
        freeSample(s);
    }

    size_t outShape[] = {numTest};
    int status = 0;
    int rc = npyWriteInt32("examples/har_classifier/outputs/c_predictions.npy", predictions,
                           outShape, 1);
    if (rc != 0) {
        fprintf(stderr, "ERROR: npyWriteInt32 failed (rc=%d)\n", rc);
        status = 1;
    }
    free(predictions);

    return status;
}
