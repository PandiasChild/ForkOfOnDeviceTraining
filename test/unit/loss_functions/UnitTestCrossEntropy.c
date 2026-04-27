#include "CrossEntropy.h"
#include "QuantizationApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "TensorApi.h"
#include "unity.h"

void unitTestCrossEntropyForward() {
    tensor_t logits;
    float logitData[] = {2.f, 1.f, 0.1f};
    size_t logitDims[] = {1, 3};
    size_t logitNumberOfDims = 2;
    size_t logitOrder[] = {0, 1};
    shape_t logitShape;
    setShape(&logitShape, logitDims, logitNumberOfDims, logitOrder);
    quantization_t logitQ;
    initFloat32Quantization(&logitQ);
    setTensorValues(&logits, (uint8_t *)logitData, &logitShape, &logitQ, NULL);

    tensor_t softmaxOutput;
    float outputData[3];
    size_t outputDims[] = {1, 3};
    size_t outputNumberOfDims = 2;
    size_t outputOrder[] = {0, 1};
    shape_t outputShape;
    setShape(&outputShape, outputDims, outputNumberOfDims, outputOrder);
    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    setTensorValues(&softmaxOutput, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layer_t *softmaxLayer = softmaxLayerInit(floatQ, floatQ);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, &logits, &softmaxOutput);

    tensor_t distribution;
    float distData[] = {0.9f, 0.1f, 0.0f};
    size_t distDims[] = {1, 3};
    size_t distNumberOfDims = 2;
    size_t distOrderOfDims[] = {0, 1};
    shape_t distShape;
    setShape(&distShape, distDims, distNumberOfDims, distOrderOfDims);
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &distShape, &distQ, NULL);

    /* CAPTURE. */
    float capturedActual = crossEntropyForwardFloat(&softmaxOutput, &distribution);

    /* FREE. freeSoftmaxLayer releases the layer config wrapper only; it does
     * NOT touch the stored forwardQ/backwardQ pointer (which is floatQ). So
     * floatQ is safe to free after. */
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);

    /* ASSERT. */
    float expected = 0.5170299410820007f;
    TEST_ASSERT_EQUAL_FLOAT(expected, capturedActual);
}

void unitTestCrossEntropySoftmaxBackward() {
    size_t inputSize = 3;

    tensor_t logits;
    float logitsData[] = {2.f, 1.f, 0.1f};
    size_t logitsDims[] = {1, inputSize};
    size_t logitsNumberOfDims = 2;
    size_t logitsOrder[] = {0, 1};
    shape_t logitsShape;
    setShape(&logitsShape, logitsDims, logitsNumberOfDims, logitsOrder);
    quantization_t logitsQ;
    initFloat32Quantization(&logitsQ);
    setTensorValues(&logits, (uint8_t *)logitsData, &logitsShape, &logitsQ, NULL);

    tensor_t softmaxOutput;
    float softmaxOutputData[inputSize];
    size_t softmaxOutputDims[] = {1, inputSize};
    size_t softmaxOutputNumberOfDims = 2;
    size_t softmaxOutputOrder[] = {0, 1};
    shape_t softmaxOutputShape;
    setShape(&softmaxOutputShape, softmaxOutputDims, softmaxOutputNumberOfDims, softmaxOutputOrder);
    quantization_t softmaxOutputQ;
    initFloat32Quantization(&softmaxOutputQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxOutputData, &softmaxOutputShape,
                    &softmaxOutputQ, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layer_t *softmaxLayer = softmaxLayerInit(floatQ, floatQ);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, &logits, &softmaxOutput);

    tensor_t distribution;
    float distData[] = {0.9f, 0.1f, 0.0f};
    size_t distDims[] = {1, inputSize};
    size_t distNumberOfDims = 2;
    size_t distOrderOfDims[] = {0, 1};
    shape_t distShape;
    setShape(&distShape, distDims, distNumberOfDims, distOrderOfDims);
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &distShape, &distQ, NULL);

    tensor_t propLoss;
    float propLossData[] = {2.f, 1.f, 0.1f};
    size_t propLossDims[] = {1, inputSize};
    size_t propLossNumberOfDims = 2;
    size_t propLossOrder[] = {0, 1};
    shape_t propLossShape;
    setShape(&propLossShape, propLossDims, propLossNumberOfDims, propLossOrder);
    quantization_t propLossQ;
    initFloat32Quantization(&propLossQ);
    setTensorValues(&propLoss, (uint8_t *)propLossData, &propLossShape, &propLossQ, NULL);

    crossEntropySoftmaxBackward(&softmaxOutput, &distribution, &propLoss, /* batchSize */ 1,
                                REDUCTION_SUM);

    /* CAPTURE. propLoss.data is stack memory (propLossData[]); copy into a
     * dedicated capture array so the assertions read from a buffer that
     * doesn't depend on freeing softmaxLayer/floatQ first. */
    float capturedActual[3];
    for (size_t i = 0; i < inputSize; i++) {
        capturedActual[i] = ((float *)propLoss.data)[i];
    }

    /* FREE. */
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);

    /* ASSERT. */
    float expected[] = {-0.2410f, 0.1424f, 0.0986f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, expected[i], capturedActual[i]);
    }
}

void testCrossEntropyBackward_MeanDividesByBatchSize(void) {
    size_t inputSize = 3;

    tensor_t softmaxOutput;
    float softmaxData[] = {0.7f, 0.2f, 0.1f};
    size_t dims[] = {1, inputSize};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t softmaxQ;
    initFloat32Quantization(&softmaxQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &softmaxQ, NULL);

    tensor_t distribution;
    float distData[] = {1.f, 0.f, 0.f};
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &distQ, NULL);

    tensor_t lossGrad;
    float lossGradData[3] = {0};
    quantization_t lossGradQ;
    initFloat32Quantization(&lossGradQ);
    setTensorValues(&lossGrad, (uint8_t *)lossGradData, &shape, &lossGradQ, NULL);

    crossEntropySoftmaxBackward(&softmaxOutput, &distribution, &lossGrad,
                                /* batchSize */ 4, REDUCTION_MEAN);

    float expected[3] = {(0.7f - 1.f) / 4.f, (0.2f - 0.f) / 4.f, (0.1f - 0.f) / 4.f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expected[i], ((float *)lossGrad.data)[i]);
    }
}

void testCrossEntropyBackward_SumPreservesMagnitude(void) {
    size_t inputSize = 3;

    tensor_t softmaxOutput;
    float softmaxData[] = {0.7f, 0.2f, 0.1f};
    size_t dims[] = {1, inputSize};
    size_t order[] = {0, 1};
    shape_t shape;
    setShape(&shape, dims, 2, order);
    quantization_t softmaxQ;
    initFloat32Quantization(&softmaxQ);
    setTensorValues(&softmaxOutput, (uint8_t *)softmaxData, &shape, &softmaxQ, NULL);

    tensor_t distribution;
    float distData[] = {1.f, 0.f, 0.f};
    quantization_t distQ;
    initFloat32Quantization(&distQ);
    setTensorValues(&distribution, (uint8_t *)distData, &shape, &distQ, NULL);

    tensor_t lossGrad;
    float lossGradData[3] = {0};
    quantization_t lossGradQ;
    initFloat32Quantization(&lossGradQ);
    setTensorValues(&lossGrad, (uint8_t *)lossGradData, &shape, &lossGradQ, NULL);

    crossEntropySoftmaxBackward(&softmaxOutput, &distribution, &lossGrad,
                                /* batchSize */ 4, REDUCTION_SUM);

    float expected[3] = {0.7f - 1.f, 0.2f - 0.f, 0.1f - 0.f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expected[i], ((float *)lossGrad.data)[i]);
    }
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
    RUN_TEST(unitTestCrossEntropyForward);
    RUN_TEST(unitTestCrossEntropySoftmaxBackward);
    RUN_TEST(testCrossEntropyBackward_MeanDividesByBatchSize);
    RUN_TEST(testCrossEntropyBackward_SumPreservesMagnitude);
    return UNITY_END();
}
