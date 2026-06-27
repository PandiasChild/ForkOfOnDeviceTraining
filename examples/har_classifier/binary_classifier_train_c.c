#define SOURCE_FILE "binary_classifier_train_c"

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
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TrainingLoopApi.h"

#include "../../src/data_loader/include/Dataset.h"
#include "../../src/layer/include/Dropout.h"
#include "../../src/userApi/tensor/include/TensorApi.h"
#include "npy_writer.h"

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
#define MODEL_SIZE 4

typedef enum deltaStatus { WITH_DELTAS, WITHOUT_DELTAS } deltaStatus_t;

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
static tensorArray_t *buildSymInt32OneHotLabels(tensorArray_t *intLabels, roundingMode_t roundingMode) {
    /* intLabels->array[i] is a rank-0 int32 tensor (single class index 0..5).
     * We allocate a brand-new tensorArray_t whose entries are rank-1 int32
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

        quantization_t *q = quantizationInitSymInt32(roundingMode);
        tensor_t *t = initTensor(shape, q, NULL);

        int32_t cls = ((int32_t *)intLabels->array[i]->data)[0];
        int32_t *data = (int32_t *)t->data;
        for (size_t c = 0; c < NUM_CLASSES; ++c) {
            data[c] = (c == (size_t)cls) ? 1 : 0;
        }
        arr[i] = t;
    }
    return out;
}

/*
param shape: Caller-allocated shape; ownership transferred to the tensor
param quantization: Caller-allocated quantization; ownership transferred.
param sparsity: Optional; ownership transferred (NULL means no sparsity)
returns Pointer to an initialized tensor with own-allocated zero data.

tensor_t *initTensor(shape_t *shape, quantization_t *quantization, sparsity_t *sparsity);
 -----




*/

static tensorArray_t *createSymInt32TensorArrayFromFloatTensorArray(tensorArray_t *tensorArray, roundingMode_t roundingMode){
    quantization_t *q = quantizationInitSymInt32(roundingMode);
    size_t numberOfTensors = tensorArray->size;

    tensorArray_t *newTensorArray = reserveMemory(sizeof(tensorArray_t));
    tensor_t **newArray = reserveMemory(numberOfTensors * sizeof(tensor_t *));
    newTensorArray->array = newArray;
    newTensorArray->size = numberOfTensors;

    for (size_t i = 0; i < numberOfTensors; i++) {
        tensor_t *currentTensor = tensorArray->array[i];
        shape_t *currentShape = currentTensor->shape;
        size_t currentNumberOfDimensions = currentShape->numberOfDimensions;
        size_t *dims = reserveMemory(currentNumberOfDimensions * sizeof(size_t));
        if (dims == NULL) {
            PRINT_ERROR("Memory Allocation Failed");
            exit(1);
        }
        memcpy(dims, currentShape->dimensions, currentNumberOfDimensions * sizeof(size_t));
        size_t *order = reserveMemory(currentNumberOfDimensions * sizeof(size_t));
        if (order == NULL) {
            PRINT_ERROR("Memory Allocation Failed");
            exit(1);
        }
        setOrderOfDimsForNewTensor(currentNumberOfDimensions, order);
        shape_t *shape = reserveMemory(sizeof(shape_t));
        if (shape == NULL) {
            PRINT_ERROR("Memory Allocation Failed");
            exit(1);
        }
        setShape(shape, dims, currentNumberOfDimensions, order);

        /* Fresh quantization clone per tensor: every array entry now owns its
         * own quantization, so freeTensor on any one doesn't double-free a
         * shared `q`. */
        quantization_t *newQ = getQLike(q);

        tensor_t *t = initTensor(shape, newQ, NULL);
        newTensorArray->array[i] = t;
        convertTensor(currentTensor, t);
    }
    /* `q` was deep-copied per tensor via getQLike; free the original. */
    freeQuantization(q);
    return newTensorArray;
}


static void initDataSets(roundingMode_t roundingMode) {
    tensorArray_t *trainItems = npyLoad("examples/har_classifier/data/train_x.npy");
    tensorArray_t *trainLabelsRaw = npyLoad("examples/har_classifier/data/train_y.npy");
    reshapeItemsAddBatchDim(trainItems);
    tensorArray_t *trainItemsSymInt32 = createSymInt32TensorArrayFromFloatTensorArray(trainItems, roundingMode);
    g_trainDataset.items = trainItemsSymInt32;
    g_trainDataset.labels = buildSymInt32OneHotLabels(trainLabelsRaw, roundingMode);
    freeTensorArray(trainItems);
    freeTensorArray(trainLabelsRaw);

    tensorArray_t *valItems = npyLoad("examples/har_classifier/data/val_x.npy");
    tensorArray_t *valLabelsRaw = npyLoad("examples/har_classifier/data/val_y.npy");
    reshapeItemsAddBatchDim(valItems);
    tensorArray_t *valItemsSymInt32 = createSymInt32TensorArrayFromFloatTensorArray(valItems, roundingMode);
    g_valDataset.items = valItemsSymInt32;
    g_valDataset.labels = buildSymInt32OneHotLabels(valLabelsRaw, roundingMode);
    freeTensorArray(valItems);
    freeTensorArray(valLabelsRaw);

    tensorArray_t *testItems = npyLoad("examples/har_classifier/data/val_x.npy");
    tensorArray_t *testLabelsRaw = npyLoad("examples/har_classifier/data/val_y.npy");
    reshapeItemsAddBatchDim(testItems);
    tensorArray_t *testItemsSymInt32 = createSymInt32TensorArrayFromFloatTensorArray(testItems, roundingMode);
    g_testDataset.items = testItemsSymInt32;
    g_testDataset.labels = buildSymInt32OneHotLabels(testLabelsRaw, roundingMode);
    freeTensorArray(testItems);
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

static float weight0Data[L1_OUT * FLATTEN_SIZE] = {0};
static size_t weight0Dims[WEIGHT_NDIM_0] = {L1_OUT, FLATTEN_SIZE};

static float bias0Data[L1_OUT] = {0};
static size_t bias0Dims[BIAS_NDIM_0] = {L1_OUT};

static float weight1Data[NUM_CLASSES * L1_OUT] = {0};
static size_t weight1Dims[WEIGHT_NDIM_1] = {NUM_CLASSES, L1_OUT};

static float bias1Data[NUM_CLASSES] = {0};
static size_t bias1Dims[BIAS_NDIM_1] = {NUM_CLASSES};

static parameter_t *buildParam(quantization_t *q, float *data, size_t *dims, size_t ndim) {
    size_t *order = reserveMemory(ndim * sizeof(size_t));
    size_t *dimensions = reserveMemory(ndim * sizeof(size_t));
    size_t *orderFloat = reserveMemory(ndim * sizeof(size_t));
    size_t *dimensionsFloat = reserveMemory(ndim * sizeof(size_t));

    if (order == NULL || dimensions == NULL || orderFloat == NULL || dimensionsFloat == NULL) {
        PRINT_ERROR("Memory Allocation Failed");
        exit(1);
    }

    for (size_t i = 0; i < ndim; i++) {
        order[i] = i;
        orderFloat[i] = i;
        dimensions[i] = dims[i];
        dimensionsFloat[i] = dims[i];
    }

    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape_t *shapeFloat = reserveMemory(sizeof(shape_t));

    if (shape == NULL || shapeFloat == NULL) {
        PRINT_ERROR("Memory Allocation Failed");
        exit(1);
    }

    shape->dimensions = dimensions;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = ndim;

    shapeFloat->dimensions = dimensionsFloat;
    shapeFloat->orderOfDimensions = orderFloat;
    shapeFloat->numberOfDimensions = ndim;

    quantization_t *floatq = quantizationInitFloat();
    tensor_t *floatTensor = initTensor(shapeFloat, floatq, NULL);
    size_t numberOfElements = calcNumberOfElementsByTensor(floatTensor);
    memcpy(floatTensor->data, data, numberOfElements * sizeof(float));

    tensor_t *p = initTensor(shape, q, NULL);
    convertTensor(floatTensor, p);

    freeTensor(floatTensor);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

static void buildModel(layer_t **model, uint8_t delta_reduction, roundingMode_t roundingMode,
                       deltaStatus_t deltaStatus) {
    uint8_t qBits = 16;
    uint8_t deltabits = qBits - delta_reduction;

    quantization_t *q1 = quantizationInitSymInt32(roundingMode);
    quantization_t *q2 = quantizationInitSymInt32(roundingMode);
    quantization_t *q3 = quantizationInitSymInt32(roundingMode);
    quantization_t *q5 = quantizationInitSymInt32(roundingMode);
    quantization_t *q7 = quantizationInitSymInt32(roundingMode);
    quantization_t *q8 = quantizationInitSymInt32(roundingMode);
    quantization_t *q9 = quantizationInitSymInt32(roundingMode);
    quantization_t *q11 = quantizationInitSymInt32(roundingMode);

    quantization_t *q0;
    quantization_t *q4;
    quantization_t *q6;
    quantization_t *q10;
    quantization_t *q12;
    quantization_t *q13;
    quantization_t *q14;
    quantization_t *q15;

    if (deltaStatus == WITHOUT_DELTAS) {
        q0 = quantizationInitSymInt32(roundingMode);
        q4 = quantizationInitSymInt32(roundingMode);
        q6 = quantizationInitSymInt32(roundingMode);
        q10 = quantizationInitSymInt32(roundingMode);
        q12 = quantizationInitSymInt32(roundingMode);
        q13 = quantizationInitSymInt32(roundingMode);
        q14 = quantizationInitSymInt32(roundingMode);
        q15 = quantizationInitSymInt32(roundingMode);
    } else if (deltaStatus == WITH_DELTAS) {
        q0 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
        q4 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
        q6 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
        q10 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
        q12 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
        q13 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
        q14 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
        q15 = quantizationInitSymQDelta(qBits, roundingMode, deltabits);
    }

    // 128 -> 1152
    model[0] = flattenLayerInit();



    // Linear 128→32
    parameter_t *weight0 = buildParam(q14, weight0Data, weight0Dims, WEIGHT_NDIM_0);
    printf("buildModel: done buildParam\n");
    parameter_t *bias0 = buildParam(q15, bias0Data, bias0Dims, BIAS_NDIM_0);
    printf("buildModel: done buildParam\n");
    model[1] = linearLayerInitLegacy(weight0, bias0, q0, q1, q2, q3);
    printf("buildModel: done linearLayerInitLegacy\n");

    // ReLU
    model[2] = reluLayerInitLegacy(q4, q5);
    // Linear 32→6
    /* parameter_t *weight1 = buildParam(XAVIER_UNIFORM, dq3, weight1Data, weight1Dims,
    WEIGHT_NDIM_1); parameter_t *bias1 = buildParam(ZEROS, dq4, bias1Data, bias1Dims, BIAS_NDIM_1);
    */
    parameter_t *weight1 = buildParam(q12, weight1Data, weight1Dims, WEIGHT_NDIM_1);
    parameter_t *bias1 = buildParam(q13, bias1Data, bias1Dims, BIAS_NDIM_1);

    model[3] = linearLayerInitLegacy(weight1, bias1, q6, q7, q8, q9);

    // Softmax
    //model[4] = softmaxLayerInitLegacy(q10, q11);
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

int runExperiment(dataLoader_t *trainLoader, dataLoader_t *valLoader, dataLoader_t *testLoader,
                  int trial_number, uint8_t delta_reduction, double learning_rate, double momentum,
                  uint8_t rounding_mode, int epochs, int batch, deltaStatus_t deltaStatus) {
    layer_t *model[MODEL_SIZE];
    buildModel(model, delta_reduction, rounding_mode, deltaStatus);
    printf("runExperiment: done buildModel\n");
    optimizer_t *sgd = sgdMCreateOptim(learning_rate, momentum, /*weightDecay*/ 0.0f, model,
                                       MODEL_SIZE, SYM_INT32);

    printf("runExperiment: done sgdMCreateOptim\n");

    const char *prefix;
    if (deltaStatus == WITHOUT_DELTAS) {
        prefix = "examples/har_classifier/logs/without_deltas/trial";
    } else if (deltaStatus == WITH_DELTAS) {
        prefix = "examples/har_classifier/logs/with_deltas/trial";
    }

    int len = snprintf(NULL, 0, "%s_%d_.json", prefix, trial_number);

    char *filename = malloc(len + 1);
    if (filename == NULL) {
        return 1;
    }

    snprintf(filename, len + 1, "%s_%d.json", prefix, trial_number);

    g_log_file = fopen(filename, "w");
    free(filename);
    if (!g_log_file) {
        fprintf(stderr, "ERROR: cannot open log file for writing\n");
        return 1;
    }
    fprintf(g_log_file,
            "{\n"
            "  \"impl\": \"c\",\n"
            "  \"example\": \"bin_classifier_trial_number%d\",\n"
            "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
            "\"momentum\": %.6f, \"seed\": %d, \"shuffle_seed\": %d, \"rounding_mode\": %d},\n"
            "  \"epochs\": [\n",
            trial_number, epochs, batch, (double)learning_rate, (double)momentum, SEED,
            SHUFFLE_SEED);
    fflush(g_log_file);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){
            .funcType = MSE, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd, epochs, calculateGradsSequential, inferenceWithLoss,
        epochCallback);
    (void)result;

    epochStats_t testStats = evaluationEpochWithMetrics(
        model, MODEL_SIZE, MSE, testLoader, inferenceWithLoss, REDUCTION_MEAN);

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

    const char *npy_prefix;
    if (deltaStatus == WITHOUT_DELTAS) {
        prefix = "examples/har_classifier/outputs/without_deltas/c_predictions";
    } else if (deltaStatus == WITH_DELTAS) {
        prefix = "examples/har_classifier/outputs/with_deltas/c_predictions";
    }

    int npy_len = snprintf(NULL, 0, "%s_%d.npy", prefix, trial_number);

    char *npy_filename = malloc(len + 1);
    if (npy_filename == NULL) {
        return 1;
    }

    snprintf(npy_filename, len + 1, "%s_%d.npy", prefix, trial_number);

    size_t outShape[] = {numTest};
    int status = 0;
    int rc = npyWriteInt32(npy_filename, predictions, outShape, 1);
    if (rc != 0) {
        fprintf(stderr, "ERROR: npyWriteInt32 failed (rc=%d)\n", rc);
        status = 1;
    }
    free(predictions);
    free(npy_filename);

    return status;
}

int main(int argc, char *argv[]) {
    if (ensureDir("examples/har_classifier/logs/without_deltas") != 0) {
        return 1;
    }
    if (ensureDir("examples/har_classifier/outputs/without_deltas") != 0) {
        return 1;
    }
    if (ensureDir("examples/har_classifier/logs/with_deltas") != 0) {
        return 1;
    }
    if (ensureDir("examples/har_classifier/outputs/with_deltas") != 0) {
        return 1;
    }
    if (argc < 2) {
        printf("Keine (negative) trial_number angegeben\n");
        return 1;
    }

    int trial_number = atof(argv[1]);
    uint8_t delta_reduction = 2;
    double learning_rate = 0.01f;
    double momentum = 0.9f;
    roundingMode_t rounding_mode = HALF_AWAY;
    int epochs = 20;
    int batch = 64;

    if (argc > 2) {
        trial_number = atof(argv[1]);
        delta_reduction = atoi(argv[2]);
        learning_rate = atof(argv[3]);
        momentum = atof(argv[4]);
        rounding_mode = atof(argv[5]);
        epochs = atof(argv[6]);
        batch = atof(argv[7]);
    }

    initDataSets(rounding_mode);
    printf("main: start dataLoaderInit\n");

    dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, batch, NULL, NULL,
                                               /*shuffle*/ true, /*shuffleSeed*/ SHUFFLE_SEED,
                                               /*dropLast*/ true);
    dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                             /*shuffle*/ false, /*shuffleSeed*/ 0,
                                             /*dropLast*/ true);
    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, /*shuffleSeed*/ 0,
                                              /*dropLast*/ true);


    printf("main: start runExperiment\n");
    int status =
        runExperiment(trainLoader, valLoader, testLoader, trial_number, delta_reduction,
                      learning_rate, momentum, rounding_mode, epochs, batch, WITHOUT_DELTAS);
    if (status != 0) {
        return status;
    }
    status = runExperiment(trainLoader, valLoader, testLoader, trial_number, delta_reduction,
                           learning_rate, momentum, rounding_mode, epochs, batch, WITH_DELTAS);

    return status;
}
