/*! Important: This experiment expects the MNIST dataset. You can load the dataset using the python
* script, located in test/unit/data_loader/MNISTLoader.py
*
* You might have to change the defined paths below, if locations differ.
*
*/

#define SOURCE_FILE "SPEECH_COMMANDS_EXPERIMENT"
#define EPOCHS 10
#define BATCH 64
#define LEARNING_RATE 0.01f
#define MOMENTUM 0.9f
#define SEED 42
#define SHUFFLE_SEED 42
#define NUM_CLASSES 2
//train_y.npy [N_train]
#define IN_CHANNELS 1
//train_x.npy [N_train, 1, 16000]
#define LEN_INPUT 16000

#define MODEL_SIZE 5
#define LOG "../../../../examples/binary_classifier/data/log/Log.csv"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../src/arithmetic/include/Distributions.h"
#include "CSVHelper.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LinearApi.h"
#include "NPYLoaderApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

#include <unistd.h>
static dataset_t trainDataset;
static dataset_t valDataset;
static dataset_t testDataset;

static void initDataSets(void) {
    tensorArray_t *trainItems = npyLoad("../../../../examples/binary_classifier/data/SPEECH_COMMANDS_one_label/train_x.npy");
    tensorArray_t *trainLabels = npyLoad("../../../../examples/binary_classifier/data/SPEECH_COMMANDS_one_label/train_y.npy");
    trainDataset.items = trainItems;

    /*tensorArray_t *valItems = npyLoad("examples/binary_classifier/data/SPEECH_COMMANDS_one_label/val_x.npy");
    tensorArray_t *valLabels = npyLoad("examples/binary_classifier/data/SPEECH_COMMANDS_one_label/val_y.npy");
    valDataset.items = valItems;
*/
    tensorArray_t *testItems = npyLoad("-../../../../examples/binary_classifier/data/SPEECH_COMMANDS_one_label/test_x.npy");
    tensorArray_t *testLabels = npyLoad("../../../../examples/binary_classifier/data/SPEECH_COMMANDS_one_label/test_y.npy");
    testDataset.items = testItems;
}


static parameter_t *buildParam(distributionType_t dist, quantization_t *quantization, float *data, size_t *dims, size_t ndim,
                               size_t fanIn, size_t fanOut) {
    tensor_t *parameter =
        tensorInitWithDistribution(dist, data, dims, ndim, quantization, NULL, fanIn, fanOut);
    tensor_t *gradients = gradInitFloat(parameter, NULL);
    return parameterInit(parameter, gradients);
}

static sample_t *getTrainSample(size_t id) {
    return npyGetSample(&trainDataset, id);
}
static sample_t *getValSample(size_t id) {
    return npyGetSample(&valDataset, id);
}
static sample_t *getTestSample(size_t id) {
    return npyGetSample(&testDataset, id);
}

static size_t getTrainSize(void) {
    return trainDataset.items->size;
}
static size_t getValSize(void) {
    return valDataset.items->size;
}
static size_t getTestSize(void) {
    return testDataset.items->size;
}


static void epochCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
   char row[256] = {0};
   sprintf(row, "%lu, %f, %f, %f, %f, %f, %f\n", epoch, trainLoss, evalStats.loss,
           evalStats.accuracy, evalStats.precision, evalStats.recall, evalStats.f1);
   PRINT_DEBUG("%s\n", row);

   char *rows[] = {row};
   size_t entriesInRow[] = {7};
   csvData_t csvData;
   setCSVData(&csvData, rows, 1, entriesInRow);
   csvWriteRowsByBufferSize(LOG, &csvData, "a");
}

static void writeCsvHeader(char *filePath) {
   char *header =
       "epoch, train_loss, eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1\n";
   char *row[] = {header};
   size_t entriesInRow[] = {7};
   csvData_t csvData;
   setCSVData(&csvData, row, 1, entriesInRow);
   csvWriteRowsByBufferSize(filePath, &csvData, "w");
}

#define MODEL_SIZE 5

static void buildModel(layer_t **model) {
    uint8_t qBits = 16;
    roundingMode_t roundingMode = HTE;
    uint8_t deltabits = qBits - 2;
    quantization_t *symInt32Quantization = quantizationInitSymInt32();
    quantization_t *deltaQuantization = quantizationInitSymQDelta(qBits, roundingMode, deltabits);


    // Flatten [N_train, 1, 16000] -> [1,16000]
    model[0] = flattenLayerInit();

    // Linear 1600→20
    static float weight0Data[20*LEN_INPUT] = {0};
    static size_t weight0Dims[] = {20, LEN_INPUT};
    parameter_t *weight0 = buildParam(XAVIER_UNIFORM, deltaQuantization, weight0Data, weight0Dims, 2,
                                      LEN_INPUT, 20);
    static float bias0Data[20] = {0};
    static size_t bias0Dims[] = {1, 20};
    parameter_t *bias0 = buildParam(ZEROS, deltaQuantization, bias0Data, bias0Dims, 2, 1, 20);

    model[1] = linearLayerInit(weight0, bias0, deltaQuantization, symInt32Quantization, symInt32Quantization, symInt32Quantization);

    // ReLU
    model[2] = reluLayerInit(symInt32Quantization, symInt32Quantization);

    // Linear 20→1
    static float weight1Data[20] = {0};
    static size_t weight1Dims[] = {1, 20};
    parameter_t *weight1 = buildParam(XAVIER_UNIFORM, deltaQuantization, weight1Data, weight1Dims, 2,
                                      20, 1);

    static float bias1Data[1] = {0};
    static size_t bias1Dims[] = {1, 1};
    parameter_t *bias1 = buildParam(ZEROS, deltaQuantization, bias1Data, bias1Dims, 2, 1, 1);
    model[3] = linearLayerInit(weight1, bias1, symInt32Quantization, symInt32Quantization, symInt32Quantization, symInt32Quantization);

    // Softmax
    model[4] = softmaxLayerInit(symInt32Quantization, symInt32Quantization);
}


int main(void) {
    char cwd[1024];
    getcwd(cwd, sizeof(cwd));
    printf("CWD: %s\n", cwd);
    //writeCsvHeader(LOG);
   initDataSets();
    printf("done");

   dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                                              /*shuffle*/ true, /*shuffleSeed*/ SHUFFLE_SEED,
                                              /*dropLast*/ true);
   /* dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                            false, 0,
                                           true);
   */
   dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                             /*shuffle*/ false, /*shuffleSeed*/ 0,
                                             /*dropLast*/ true);

   layer_t *model[MODEL_SIZE];
   buildModel(model);

   optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, MOMENTUM, 0.f, model, MODEL_SIZE, FLOAT32);

   clock_t start = clock();

   trainingRunResult_t result =
       trainingRun(model, MODEL_SIZE,
                   (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN},
                   trainLoader, testLoader, sgd, EPOCHS, calculateGradsSequential,
                   inferenceWithLoss, epochCallback);

   clock_t end = clock();

   double duration_sec = (double)(end - start) / CLOCKS_PER_SEC;
   PRINT_INFO("Training finished in %f seconds\n", duration_sec);
   PRINT_INFO("Final train loss: %f, eval loss: %f\n", result.finalTrainLoss,
              result.finalEvalStats.loss);
   PRINT_INFO("Final accuracy: %.2f%%\n", result.finalEvalStats.accuracy * 100.0f);
}
