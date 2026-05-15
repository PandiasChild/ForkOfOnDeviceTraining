#include <stdio.h>
#include <stdlib.h>

#include "Deserialize.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "Relu.h"
#include "ReluApi.h"
#include "Serialize.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

/* SERIALIZE_TEST_FILE_PATH is injected by the CMake target_compile_definitions
 * in test/unit/serial/CMakeLists.txt as an absolute path so the test does not
 * depend on the working directory (which differs between host runs and Docker
 * LSan runs). */
#define FILE_PATH SERIALIZE_TEST_FILE_PATH

static tensor_t *makeFloatTensor2D(size_t d0, size_t d1, const float *src, size_t count) {
    /* Heap-tier construction per CONVENTIONS Rule 1. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, src, count);
    }
    return t;
}

void testSerializeAndDeserializeTensor() {
    size_t numberOfValues = 6;
    float data[] = {9, 9, 9, 4.5f, 2.1112f, 999.123f};

    tensor_t *serialTensor = makeFloatTensor2D(2, 3, data, numberOfValues);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(serialTensor, f);
    fclose(f);

    /* Heap-allocated zero-init buffer destination via initTensor. */
    tensor_t *deserialTensor = makeFloatTensor2D(2, 3, NULL, 0);

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(deserialTensor, f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    float capturedDeserialData[6];
    for (size_t i = 0; i < numberOfValues; i++) {
        capturedDeserialData[i] = ((float *)deserialTensor->data)[i];
    }
    qtype_t capturedSerialQType = serialTensor->quantization->type;
    qtype_t capturedDeserialQType = deserialTensor->quantization->type;
    size_t capturedSerialNumDims = serialTensor->shape->numberOfDimensions;
    size_t capturedDeserialNumDims = deserialTensor->shape->numberOfDimensions;

    size_t capturedSerialDims[2];
    size_t capturedDeserialDims[2];
    size_t capturedSerialOrder[2];
    size_t capturedDeserialOrder[2];
    for (size_t i = 0; i < 2; i++) {
        capturedSerialDims[i] = serialTensor->shape->dimensions[i];
        capturedDeserialDims[i] = deserialTensor->shape->dimensions[i];
        capturedSerialOrder[i] = serialTensor->shape->orderOfDimensions[i];
        capturedDeserialOrder[i] = deserialTensor->shape->orderOfDimensions[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(deserialTensor);
    freeTensor(serialTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(data, capturedDeserialData, numberOfValues);
    TEST_ASSERT_EQUAL(capturedSerialQType, capturedDeserialQType);
    TEST_ASSERT_EQUAL(capturedSerialNumDims, capturedDeserialNumDims);
    TEST_ASSERT_EQUAL_size_t_ARRAY(capturedSerialDims, capturedDeserialDims, 2);
    TEST_ASSERT_EQUAL_size_t_ARRAY(capturedSerialOrder, capturedDeserialOrder, 2);
}

void testSerializeAndDeserializeModel() {
    /* One shared layer-config quantization per side. Layer free-functions
     * release only the wrapper, so layerQ stays alive across the layer
     * frees and is freed once with freeQuantization at the end. */
    quantization_t *serialLayerQ = quantizationInitFloat();
    quantization_t *deserialLayerQ = quantizationInitFloat();

    /* === Serial side: Linear (20 x 28*28) -> ReLU -> Linear (10 x 20) -> Softmax === */
    float serialWeight0Data[20 * 28 * 28];
    for (size_t i = 0; i < 28 * 28 * 20; i++) {
        serialWeight0Data[i] = (float)i;
    }
    tensor_t *serialWeight0Param = makeFloatTensor2D(20, 28 * 28, serialWeight0Data, 20 * 28 * 28);
    tensor_t *serialWeight0Grad = gradInitFloat(serialWeight0Param, NULL);
    parameter_t *serialWeight0 = parameterInit(serialWeight0Param, serialWeight0Grad);

    float serialBias0Data[20];
    for (size_t i = 0; i < 20; i++) {
        serialBias0Data[i] = (float)i;
    }
    tensor_t *serialBias0Param = makeFloatTensor2D(1, 20, serialBias0Data, 20);
    tensor_t *serialBias0Grad = gradInitFloat(serialBias0Param, NULL);
    parameter_t *serialBias0 = parameterInit(serialBias0Param, serialBias0Grad);

    layer_t *serialLinear0 = linearLayerInitLegacy(serialWeight0, serialBias0, serialLayerQ,
                                                   serialLayerQ, serialLayerQ, serialLayerQ);
    layer_t *serialRelu = reluLayerInitLegacy(serialLayerQ, serialLayerQ);

    float serialWeight1Data[10 * 20];
    for (size_t i = 0; i < 10 * 20; i++) {
        serialWeight1Data[i] = (float)i;
    }
    tensor_t *serialWeight1Param = makeFloatTensor2D(10, 20, serialWeight1Data, 10 * 20);
    tensor_t *serialWeight1Grad = gradInitFloat(serialWeight1Param, NULL);
    parameter_t *serialWeight1 = parameterInit(serialWeight1Param, serialWeight1Grad);

    float serialBias1Data[10];
    for (size_t i = 0; i < 10; i++) {
        serialBias1Data[i] = (float)i;
    }
    tensor_t *serialBias1Param = makeFloatTensor2D(1, 10, serialBias1Data, 10);
    tensor_t *serialBias1Grad = gradInitFloat(serialBias1Param, NULL);
    parameter_t *serialBias1 = parameterInit(serialBias1Param, serialBias1Grad);

    layer_t *serialLinear1 = linearLayerInitLegacy(serialWeight1, serialBias1, serialLayerQ,
                                                   serialLayerQ, serialLayerQ, serialLayerQ);
    layer_t *serialSoftmax = softmaxLayerInit(serialLayerQ, serialLayerQ);

    layer_t *serialModel[] = {serialLinear0, serialRelu, serialLinear1, serialSoftmax};
    size_t sizeModel = 4;

    FILE *f = fopen(FILE_PATH, "w");
    serializeModel(serialModel, sizeModel, f);
    fclose(f);

    /* === Deserial side: zero-init mirror with the same layer topology. === */
    tensor_t *deserialWeight0Param = makeFloatTensor2D(20, 28 * 28, NULL, 0);
    tensor_t *deserialWeight0Grad = gradInitFloat(deserialWeight0Param, NULL);
    parameter_t *deserialWeight0 = parameterInit(deserialWeight0Param, deserialWeight0Grad);

    tensor_t *deserialBias0Param = makeFloatTensor2D(1, 20, NULL, 0);
    tensor_t *deserialBias0Grad = gradInitFloat(deserialBias0Param, NULL);
    parameter_t *deserialBias0 = parameterInit(deserialBias0Param, deserialBias0Grad);

    layer_t *deserialLinear0 =
        linearLayerInitLegacy(deserialWeight0, deserialBias0, deserialLayerQ, deserialLayerQ,
                              deserialLayerQ, deserialLayerQ);
    layer_t *deserialRelu = reluLayerInitLegacy(deserialLayerQ, deserialLayerQ);

    tensor_t *deserialWeight1Param = makeFloatTensor2D(10, 20, NULL, 0);
    tensor_t *deserialWeight1Grad = gradInitFloat(deserialWeight1Param, NULL);
    parameter_t *deserialWeight1 = parameterInit(deserialWeight1Param, deserialWeight1Grad);

    tensor_t *deserialBias1Param = makeFloatTensor2D(1, 10, NULL, 0);
    tensor_t *deserialBias1Grad = gradInitFloat(deserialBias1Param, NULL);
    parameter_t *deserialBias1 = parameterInit(deserialBias1Param, deserialBias1Grad);

    layer_t *deserialLinear1 =
        linearLayerInitLegacy(deserialWeight1, deserialBias1, deserialLayerQ, deserialLayerQ,
                              deserialLayerQ, deserialLayerQ);
    layer_t *deserialSoftmax = softmaxLayerInit(deserialLayerQ, deserialLayerQ);

    layer_t *deserialModel[] = {deserialLinear0, deserialRelu, deserialLinear1, deserialSoftmax};

    f = fopen(FILE_PATH, "r");
    deserializeModel(deserialModel, sizeModel, f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    size_t numberOfWeights0 =
        calcNumberOfElementsByTensor(serialModel[0]->config->linear->weights->param);
    size_t numberOfBiases0 =
        calcNumberOfElementsByTensor(serialModel[0]->config->linear->bias->param);
    size_t numberOfWeights1 =
        calcNumberOfElementsByTensor(serialModel[2]->config->linear->weights->param);
    size_t numberOfBiases1 =
        calcNumberOfElementsByTensor(serialModel[2]->config->linear->bias->param);

    /* Capture weight/bias arrays into heap buffers (sizes vary per layer). */
    float *capturedSerialW0 = reserveMemory(numberOfWeights0 * sizeof(float));
    float *capturedDeserialW0 = reserveMemory(numberOfWeights0 * sizeof(float));
    float *capturedSerialB0 = reserveMemory(numberOfBiases0 * sizeof(float));
    float *capturedDeserialB0 = reserveMemory(numberOfBiases0 * sizeof(float));
    float *capturedSerialW1 = reserveMemory(numberOfWeights1 * sizeof(float));
    float *capturedDeserialW1 = reserveMemory(numberOfWeights1 * sizeof(float));
    float *capturedSerialB1 = reserveMemory(numberOfBiases1 * sizeof(float));
    float *capturedDeserialB1 = reserveMemory(numberOfBiases1 * sizeof(float));

    for (size_t i = 0; i < numberOfWeights0; i++) {
        capturedSerialW0[i] = ((float *)serialModel[0]->config->linear->weights->param->data)[i];
        capturedDeserialW0[i] =
            ((float *)deserialModel[0]->config->linear->weights->param->data)[i];
    }
    for (size_t i = 0; i < numberOfBiases0; i++) {
        capturedSerialB0[i] = ((float *)serialModel[0]->config->linear->bias->param->data)[i];
        capturedDeserialB0[i] = ((float *)deserialModel[0]->config->linear->bias->param->data)[i];
    }
    for (size_t i = 0; i < numberOfWeights1; i++) {
        capturedSerialW1[i] = ((float *)serialModel[2]->config->linear->weights->param->data)[i];
        capturedDeserialW1[i] =
            ((float *)deserialModel[2]->config->linear->weights->param->data)[i];
    }
    for (size_t i = 0; i < numberOfBiases1; i++) {
        capturedSerialB1[i] = ((float *)serialModel[2]->config->linear->bias->param->data)[i];
        capturedDeserialB1[i] = ((float *)deserialModel[2]->config->linear->bias->param->data)[i];
    }

    qtype_t capturedSerialL0FwdQ = serialModel[0]->config->linear->forwardQ->type;
    qtype_t capturedDeserialL0FwdQ = deserialModel[0]->config->linear->forwardQ->type;
    qtype_t capturedSerialL0WGQ = serialModel[0]->config->linear->weightGradQ->type;
    qtype_t capturedDeserialL0WGQ = deserialModel[0]->config->linear->weightGradQ->type;
    qtype_t capturedSerialL0BGQ = serialModel[0]->config->linear->biasGradQ->type;
    qtype_t capturedDeserialL0BGQ = deserialModel[0]->config->linear->biasGradQ->type;
    qtype_t capturedSerialL0PLQ = serialModel[0]->config->linear->propLossQ->type;
    qtype_t capturedDeserialL0PLQ = deserialModel[0]->config->linear->propLossQ->type;

    qtype_t capturedSerialReluFwdQ = serialModel[1]->config->relu->forwardQ->type;
    qtype_t capturedDeserialReluFwdQ = deserialModel[1]->config->relu->forwardQ->type;
    qtype_t capturedSerialReluBwdQ = serialModel[1]->config->relu->backwardQ->type;
    qtype_t capturedDeserialReluBwdQ = deserialModel[1]->config->relu->backwardQ->type;

    qtype_t capturedSerialL1FwdQ = serialModel[2]->config->linear->forwardQ->type;
    qtype_t capturedDeserialL1FwdQ = deserialModel[2]->config->linear->forwardQ->type;
    qtype_t capturedSerialL1WGQ = serialModel[2]->config->linear->weightGradQ->type;
    qtype_t capturedDeserialL1WGQ = deserialModel[2]->config->linear->weightGradQ->type;
    qtype_t capturedSerialL1BGQ = serialModel[2]->config->linear->biasGradQ->type;
    qtype_t capturedDeserialL1BGQ = deserialModel[2]->config->linear->biasGradQ->type;
    qtype_t capturedSerialL1PLQ = serialModel[2]->config->linear->propLossQ->type;
    qtype_t capturedDeserialL1PLQ = deserialModel[2]->config->linear->propLossQ->type;

    qtype_t capturedSerialSoftFwdQ = serialModel[3]->config->softmax->forwardQ->type;
    qtype_t capturedDeserialSoftFwdQ = deserialModel[3]->config->softmax->forwardQ->type;
    qtype_t capturedSerialSoftBwdQ = serialModel[3]->config->softmax->backwardQ->type;
    qtype_t capturedDeserialSoftBwdQ = deserialModel[3]->config->softmax->backwardQ->type;

    /* FREE in reverse-init order. Layer free-functions release only the
     * wrapper; parameters and the shared layerQ are caller-managed (per
     * docs/CONVENTIONS.md "Test memory discipline"). */
    freeSoftmaxLayer(deserialSoftmax);
    freeLinearLayerLegacy(deserialLinear1);
    freeParameter(deserialBias1);
    freeParameter(deserialWeight1);
    freeReluLayerLegacy(deserialRelu);
    freeLinearLayerLegacy(deserialLinear0);
    freeParameter(deserialBias0);
    freeParameter(deserialWeight0);
    freeQuantization(deserialLayerQ);

    freeSoftmaxLayer(serialSoftmax);
    freeLinearLayerLegacy(serialLinear1);
    freeParameter(serialBias1);
    freeParameter(serialWeight1);
    freeReluLayerLegacy(serialRelu);
    freeLinearLayerLegacy(serialLinear0);
    freeParameter(serialBias0);
    freeParameter(serialWeight0);
    freeQuantization(serialLayerQ);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW0, capturedDeserialW0, numberOfWeights0);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB0, capturedDeserialB0, numberOfBiases0);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW1, capturedDeserialW1, numberOfWeights1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB1, capturedDeserialB1, numberOfBiases1);

    TEST_ASSERT_EQUAL(capturedSerialL0FwdQ, capturedDeserialL0FwdQ);
    TEST_ASSERT_EQUAL(capturedSerialL0WGQ, capturedDeserialL0WGQ);
    TEST_ASSERT_EQUAL(capturedSerialL0BGQ, capturedDeserialL0BGQ);
    TEST_ASSERT_EQUAL(capturedSerialL0PLQ, capturedDeserialL0PLQ);

    TEST_ASSERT_EQUAL(capturedSerialReluFwdQ, capturedDeserialReluFwdQ);
    TEST_ASSERT_EQUAL(capturedSerialReluBwdQ, capturedDeserialReluBwdQ);

    TEST_ASSERT_EQUAL(capturedSerialL1FwdQ, capturedDeserialL1FwdQ);
    TEST_ASSERT_EQUAL(capturedSerialL1WGQ, capturedDeserialL1WGQ);
    TEST_ASSERT_EQUAL(capturedSerialL1BGQ, capturedDeserialL1BGQ);
    TEST_ASSERT_EQUAL(capturedSerialL1PLQ, capturedDeserialL1PLQ);

    TEST_ASSERT_EQUAL(capturedSerialSoftFwdQ, capturedDeserialSoftFwdQ);
    TEST_ASSERT_EQUAL(capturedSerialSoftBwdQ, capturedDeserialSoftBwdQ);

    /* Release the assertion-buffer scratch space last. */
    freeReservedMemory(capturedSerialW0);
    freeReservedMemory(capturedDeserialW0);
    freeReservedMemory(capturedSerialB0);
    freeReservedMemory(capturedDeserialB0);
    freeReservedMemory(capturedSerialW1);
    freeReservedMemory(capturedDeserialW1);
    freeReservedMemory(capturedSerialB1);
    freeReservedMemory(capturedDeserialB1);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSerializeAndDeserializeTensor);
    RUN_TEST(testSerializeAndDeserializeModel);
    return UNITY_END();
}
