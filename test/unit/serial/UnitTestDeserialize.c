#include <stdio.h>
#include <stdlib.h>

#include "TensorApi.h"
#include "Tensor.h"
#include "unity.h"
#include "Serialize.h"
#include "Deserialize.h"

#include "QuantizationApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "Relu.h"

#include <Softmax.h>

#define FILE_PATH "../../../../../test/unit/serial/SerializeTestFile.bin"

void testSerializeAndDeserializeTensor() {
    size_t numberOfValues = 6;
    float data[] = {9, 9, 9, 4.5f, 2.1112f, 999.123f};
    size_t dims[] = {2, 3};
    size_t numberOfDims = 2;

    tensor_t *serialTensor = tensorInitFloat(data, dims, numberOfDims, NULL);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(serialTensor, f);
    fclose(f);

    float deserialData[6] = {0};
    size_t deserialDims[2] = {0};
    size_t deserialOrder[2] = {0};
    shape_t deserialShape = {
        .dimensions = deserialDims,
        .numberOfDimensions = 0,
        .orderOfDimensions = deserialOrder
    };
    quantization_t deserialQ;

    tensor_t deserialTensor = {
        .shape = &deserialShape,
        .data = (uint8_t *)deserialData,
        .sparsity = NULL,
        .quantization = &deserialQ
    };

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(&deserialTensor, f);
    fclose(f);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(data, deserialTensor.data, numberOfValues);
    TEST_ASSERT_EQUAL(serialTensor->quantization->type, deserialTensor.quantization->type);
    TEST_ASSERT_EQUAL(serialTensor->shape->numberOfDimensions,
                      deserialTensor.shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t_ARRAY(serialTensor->shape->dimensions,
                                   deserialTensor.shape->dimensions, numberOfDims);
    TEST_ASSERT_EQUAL_size_t_ARRAY(serialTensor->shape->orderOfDimensions,
                                   deserialTensor.shape->orderOfDimensions, numberOfDims);
}

void testSerializeAndDeserializeModel() {
    quantization_t *serialQ = quantizationInitFloat();

    float serialWeight0Data[20 * 28 * 28];
    for (size_t i = 0; i < 28 * 28 * 20; i++) {
        serialWeight0Data[i] = (float) i;
    }

    size_t serialWeight0Dims[] = {20, 28 * 28};
    size_t serialWeight0NumberOfDims = 2;
    tensor_t *serialWeight0Param = tensorInitFloat(serialWeight0Data, serialWeight0Dims,
                                                   serialWeight0NumberOfDims, NULL);
    tensor_t *serialWeight0Grad = gradInitFloat(serialWeight0Param, NULL);
    parameter_t *serialWeight0 = parameterInit(serialWeight0Param, serialWeight0Grad);

    float serialBias0Data[20];
    for (size_t i = 0; i < 20; i++) {
        serialBias0Data[i] = (float)i;
    }
    size_t serialBias0Dims[] = {1, 20};
    size_t serialBias0NumberOfDims = 2;
    tensor_t *serialBias0Param = tensorInitFloat(serialBias0Data, serialBias0Dims,
                                                 serialBias0NumberOfDims, NULL);
    tensor_t *serialBias0Grad = gradInitFloat(serialBias0Param, NULL);
    parameter_t *serialBias0 = parameterInit(serialBias0Param, serialBias0Grad);

    layer_t *serialLinear0 = linearLayerInit(serialWeight0, serialBias0, serialQ, serialQ, serialQ,
                                             serialQ);

    layer_t *serialRelu = reluLayerInit(serialQ, serialQ);

    float serialWeight1Data[10 * 20];
    for (size_t i = 0; i < 10 * 20; i++) {
        serialWeight1Data[i] = (float)i;
    }
    size_t serialWeight1Dims[] = {10, 20};
    size_t serialWeight1NumberOfDims = 2;
    tensor_t *serialWeight1Param = tensorInitFloat(serialWeight1Data, serialWeight1Dims,
                                                   serialWeight1NumberOfDims, NULL);
    tensor_t *serialWeight1Grad = gradInitFloat(serialWeight1Param, NULL);
    parameter_t *serialWeight1 = parameterInit(serialWeight1Param, serialWeight1Grad);

    float serialBias1Data[10];
    for (size_t i = 0; i < 10; i++) {
        serialBias1Data[i] = (float)i;
    }
    size_t serialBias1Dims[] = {1, 10};
    size_t serialBias1NumberOfDims = 2;
    tensor_t *serialBias1Param = tensorInitFloat(serialBias1Data, serialBias1Dims,
                                                 serialBias1NumberOfDims, NULL);
    tensor_t *serialBias1Grad = gradInitFloat(serialBias1Param, NULL);
    parameter_t *serialBias1 = parameterInit(serialBias1Param, serialBias1Grad);

    layer_t *serialLinear1 = linearLayerInit(serialWeight1, serialBias1, serialQ, serialQ, serialQ,
                                             serialQ);

    layer_t *serialSoftmax = softmaxLayerInit(serialQ, serialQ);

    layer_t *serialModel[] = {serialLinear0, serialRelu, serialLinear1, serialSoftmax};
    size_t sizeModel = 4;

    FILE *f = fopen(FILE_PATH, "w");
    serializeModel(serialModel, sizeModel, f);
    fclose(f);

    quantization_t deserialQ;

    float deserialWeight0Data[20 * 28 * 28] = {0};
    size_t deserialWeight0Dims[] = {20, 28 * 28};
    size_t deserialWeight0NumberOfDims = 2;
    tensor_t *deserialWeight0Param = tensorInitFloat(deserialWeight0Data, deserialWeight0Dims,
                                                     deserialWeight0NumberOfDims, NULL);
    tensor_t *deserialWeight0Grad = gradInitFloat(deserialWeight0Param, NULL);
    parameter_t *deserialWeight0 = parameterInit(deserialWeight0Param, deserialWeight0Grad);

    float deserialBias0Data[20] = {0};
    size_t deserialBias0Dims[] = {1, 20};
    size_t deserialBias0NumberOfDims = 2;
    tensor_t *deserialBias0Param = tensorInitFloat(deserialBias0Data, deserialBias0Dims,
                                                   deserialBias0NumberOfDims, NULL);
    tensor_t *deserialBias0Grad = gradInitFloat(deserialBias0Param, NULL);
    parameter_t *deserialBias0 = parameterInit(deserialBias0Param, deserialBias0Grad);

    layer_t *deserialLinear0 = linearLayerInit(deserialWeight0, deserialBias0, &deserialQ,
                                               &deserialQ,
                                               &deserialQ, &deserialQ);

    layer_t *deserialRelu = reluLayerInit(&deserialQ, &deserialQ);

    float deserialWeight1Data[10 * 20] = {0};
    size_t deserialWeight1Dims[] = {10, 20};
    size_t deserialWeight1NumberOfDims = 2;
    tensor_t *deserialWeight1Param = tensorInitFloat(deserialWeight1Data, deserialWeight1Dims,
                                                     deserialWeight1NumberOfDims, NULL);
    tensor_t *deserialWeight1Grad = gradInitFloat(deserialWeight1Param, NULL);
    parameter_t *deserialWeight1 = parameterInit(deserialWeight1Param, deserialWeight1Grad);

    float deserialBias1Data[10] = {0};
    size_t deserialBias1Dims[] = {1, 10};
    size_t deserialBias1NumberOfDims = 2;
    tensor_t *deserialBias1Param = tensorInitFloat(deserialBias1Data, deserialBias1Dims,
                                                   deserialBias1NumberOfDims, NULL);
    tensor_t *deserialBias1Grad = gradInitFloat(deserialBias1Param, NULL);
    parameter_t *deserialBias1 = parameterInit(deserialBias1Param, deserialBias1Grad);

    layer_t *deserialLinear1 = linearLayerInit(deserialWeight1, deserialBias1, &deserialQ,
                                               &deserialQ, &deserialQ, &deserialQ);

    layer_t *deserialSoftmax = softmaxLayerInit(&deserialQ, &deserialQ);

    layer_t *deserialModel[] = {deserialLinear0, deserialRelu, deserialLinear1, deserialSoftmax};

    f = fopen(FILE_PATH, "r");
    deserializeModel(deserialModel, sizeModel, f);
    fclose(f);

    size_t numberOfWeights0 = calcNumberOfElementsByTensor(
        serialModel[0]->config->linear->weights->param);
    float *serialLinear0Weights = (float *)serialModel[0]->config->linear->weights->param->data;
    float *deserialLinear0Weights = (float *)deserialModel[0]->config->linear->weights->param->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(serialLinear0Weights, deserialLinear0Weights, numberOfWeights0);

    size_t numberOfBiases0 = calcNumberOfElementsByTensor(
        serialModel[0]->config->linear->bias->param);
    float *serialLinear0Biases = (float *)serialModel[0]->config->linear->bias->param->data;
    float *deserialLinear0Biases = (float *)deserialModel[0]->config->linear->bias->param->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(serialLinear0Biases, deserialLinear0Biases, numberOfBiases0);

    TEST_ASSERT_EQUAL(serialModel[0]->config->linear->forwardQ->type,
                      deserialModel[0]->config->linear->forwardQ->type);
    TEST_ASSERT_EQUAL(serialModel[0]->config->linear->weightGradQ->type,
                      deserialModel[0]->config->linear->weightGradQ->type);
    TEST_ASSERT_EQUAL(serialModel[0]->config->linear->biasGradQ->type,
                      deserialModel[0]->config->linear->biasGradQ->type);
    TEST_ASSERT_EQUAL(serialModel[0]->config->linear->propLossQ->type,
                      deserialModel[0]->config->linear->propLossQ->type);

    TEST_ASSERT_EQUAL(serialModel[1]->config->relu->forwardQ->type,
                      deserialModel[1]->config->relu->forwardQ->type);
    TEST_ASSERT_EQUAL(serialModel[1]->config->relu->backwardQ->type,
                      deserialModel[1]->config->relu->backwardQ->type);

    size_t numberOfWeights1 = calcNumberOfElementsByTensor(
    serialModel[2]->config->linear->weights->param);
    float *serialLinear1Weights = (float *)serialModel[2]->config->linear->weights->param->data;
    float *deserialLinear1Weights = (float *)deserialModel[2]->config->linear->weights->param->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(serialLinear1Weights, deserialLinear1Weights, numberOfWeights1);

    size_t numberOfBiases1 = calcNumberOfElementsByTensor(
        serialModel[2]->config->linear->bias->param);
    float *serialLinear1Biases = (float *)serialModel[2]->config->linear->bias->param->data;
    float *deserialLinear1Biases = (float *)deserialModel[2]->config->linear->bias->param->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(serialLinear1Biases, deserialLinear1Biases, numberOfBiases1);

    TEST_ASSERT_EQUAL(serialModel[2]->config->linear->forwardQ->type,
                      deserialModel[2]->config->linear->forwardQ->type);
    TEST_ASSERT_EQUAL(serialModel[2]->config->linear->weightGradQ->type,
                      deserialModel[2]->config->linear->weightGradQ->type);
    TEST_ASSERT_EQUAL(serialModel[2]->config->linear->biasGradQ->type,
                      deserialModel[2]->config->linear->biasGradQ->type);
    TEST_ASSERT_EQUAL(serialModel[2]->config->linear->propLossQ->type,
                      deserialModel[2]->config->linear->propLossQ->type);


    TEST_ASSERT_EQUAL(serialModel[3]->config->softmax->forwardQ->type, deserialModel[3]->config->softmax->forwardQ->type);
    TEST_ASSERT_EQUAL(serialModel[3]->config->softmax->backwardQ->type, deserialModel[3]->config->softmax->backwardQ->type);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSerializeAndDeserializeTensor);
    RUN_TEST(testSerializeAndDeserializeModel);
    return UNITY_END();
}
