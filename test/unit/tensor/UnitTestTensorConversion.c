#include "DTypes.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_requant.h"
#include "unity.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* PR-C pack tests verify the packed SYM output by unpacking it here in-test:
 * byteConversion zero-fills on widen, so sign-extend each value from qBits.
 * (The SYM->* unpack cells live on the parallel PR-B branch, not here.) */
static void symTestUnpackSignExtend(const uint8_t *packed, size_t qBits, int32_t *out, size_t n) {
    byteConversion((uint8_t *)packed, qBits, (uint8_t *)out, 32, n);
    const int32_t signBit = (int32_t)1 << (qBits - 1);
    const int32_t mask = (int32_t)(((uint32_t)1 << qBits) - 1u);
    for (size_t i = 0; i < n; i++) {
        int32_t v = out[i] & mask;
        out[i] = (v ^ signBit) - signBit;
    }
}

void testConversionIntFloat() {
    uint8_t numValues = 6;

    size_t dims[] = {6};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {1, 2, 3, 4, -1, -2};
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatData[numValues];

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&intTensor, &floatTensor);
    float actual[numValues];
    readBytesAsFloatArray(numValues, (uint8_t *)floatData, actual);

    float expected[] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, numValues);
}

void testConversionIntSymInt32() {
    uint8_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {1, 2, 3, 4, -1, -2};

    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[numValues];

    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    convertTensor(&intTensor, &symInt32Tensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(intTensor.data, symInt32Tensor.data, numValues);
}

void testConversionIntAsym() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};
    int32_t intData[] = {1, 2, 3, 4, -1, -2};

    quantization_t intQ;
    initInt32Quantization(&intQ);

    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    asymQConfig_t asymQConfig;
    initAsymQConfig(5, HALF_AWAY, &asymQConfig);
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);
    uint8_t asymData[numValues * calcBytesPerElement(&asymQ)];

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);
    convertTensor(&intTensor, &asymTensor);

    uint8_t flattenedAsymData[numValues];
    byteConversion(asymTensor.data, asymQConfig.qBits, flattenedAsymData, 8, numValues);

    /*uint8_t expectedAsym[] = {16, 22, 25, 31, 6, 0};
    int32_t expectedZeroPoint = -11;
    float expectedScale = 0.1875f;*/

    uint8_t expectedAsym[] = {15, 20, 26, 30, 5, 0};
    int32_t expectedZeroPoint = -10;
    float expectedScale = 0.1935484f;

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedAsym, flattenedAsymData, numValues);
    TEST_ASSERT_EQUAL_INT32(expectedZeroPoint, asymQConfig.zeroPoint);
    TEST_ASSERT_EQUAL_FLOAT(expectedScale, asymQConfig.scale);
}

void testConversionFloatInt() {
    uint8_t numValues = 6;

    float floatData[] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    quantization_t intQ;
    initInt32Quantization(&intQ);
    int32_t intData[numValues];
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);
    convertTensor(&floatTensor, &intTensor);

    int32_t actual[numValues];
    readBytesAsInt32Array(6, (uint8_t *)intData, actual);

    int32_t expected[] = {1, 2, 3, 4, -1, -2};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, actual, numValues);
}

void testConversionFloatSymInt32() {
    uint8_t numValues = 6;

    float floatData[] = {1.5f, 2.9f, 3.2f, 4.5f, -1.2f, -6.7f};
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[numValues];
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);
    convertTensor(&floatTensor, &symInt32Tensor);

    /* absmax = 6.7; int12 scale = 6.7/2047 ≈ 0.003273083.
     * Quantized values: round(v / scale): 458, 886, 978, 1375, -367, -2047. */
    float expectedScale = 6.7f / 2047.0f;
    int32_t expectedData[] = {458, 886, 978, 1375, -367, -2047};

    symInt32QConfig_t *outputSymInt32QC = symInt32Tensor.quantization->qConfig;
    TEST_ASSERT_FLOAT_WITHIN(0.000001f, expectedScale, outputSymInt32QC->scale);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedData, symInt32Tensor.data, numValues);

    convertTensor(&symInt32Tensor, &floatTensor);
    float expectedFloat[] = {1.5f, 2.9f, 3.2f, 4.5f, -1.2f, -6.7f};
    float *actualFloat = (float *)floatTensor.data;
    /* int12 quantisation step = scale ≈ 0.00327; worst-case round-trip
     * error is scale/2 ≈ 0.00164, so tolerance 0.002 is sufficient. */
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.002f, expectedFloat[i], actualFloat[i]);
    }
}

void testConversionFloatAsym() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    float floatData[] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    asymQConfig_t asymQConfig;
    initAsymQConfig(5, HALF_AWAY, &asymQConfig);
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[numValues * calcBytesPerElement(&asymQ)];

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    uint8_t flattenedAsymData[numValues];
    byteConversion(asymTensor.data, asymQConfig.qBits, flattenedAsymData, 8, numValues);

    uint8_t expectedAsym[] = {16, 22, 27, 31, 6, 0};
    int32_t expectedZeroPoint = -11;
    float expectedScale = 0.1875f;

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedAsym, flattenedAsymData, numValues);
    TEST_ASSERT_EQUAL_INT32(expectedZeroPoint, asymQConfig.zeroPoint);
    TEST_ASSERT_EQUAL_FLOAT(expectedScale, asymQConfig.scale);
}

void testConversionSymInt32Int() {
    size_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[] = {1, 2, 3, 4, -1, -2};
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    int32_t intData[numValues];
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    convertTensor(&symInt32Tensor, &intTensor);

    int32_t expected[] = {1, 2, 3, 4, -1, -2};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, intTensor.data, numValues);
}

void testConversionSymInt32Float() {
    uint8_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    symInt32QConfig.scale = 1.f;
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[] = {1, 2, 3, 4, -1, -2};
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    float floatData[numValues];

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&symInt32Tensor, &floatTensor);

    float expected[] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};

    /*float actual[numValues];
    readBytesAsFloatArray(numValues, floatTensor.data, actual);
    for(size_t i = 0; i < numValues; i++) {
        printf("%f\n", actual[i]);
    }*/

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, floatTensor.data, numValues);
}

void testConversionSymInt32Asym() {
    uint8_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    symInt32QConfig.scale = 1.f;
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[] = {1, 2, 3, 4, -1, -2};
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    asymQConfig_t asymQConfig;
    initAsymQConfig(5, HALF_AWAY, &asymQConfig);
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    size_t outputBitsPerElement = calcBitsPerElement(&asymQ);
    size_t outputTotalNumberOfBits = outputBitsPerElement * numValues;
    size_t numberOfRequiredBytes = ceil((double)outputTotalNumberOfBits / (double)8);
    uint8_t asymData[numberOfRequiredBytes];

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    convertTensor(&symInt32Tensor, &asymTensor);

    uint32_t output[numValues];
    byteConversion(asymTensor.data, asymQConfig.qBits, (uint8_t *)output, 32, numValues);

    float expectedScale = 0.193548f;
    int16_t expectedZeroPoint = -10;
    uint32_t expectedValues[] = {15, 20, 26, 31, 5, 0};

    TEST_ASSERT_EQUAL_FLOAT(expectedScale, asymQConfig.scale);
    TEST_ASSERT_EQUAL_INT16(expectedZeroPoint, asymQConfig.zeroPoint);
    TEST_ASSERT_EQUAL_UINT32_ARRAY(expectedValues, output, numValues);
}

void testConversionAsymInt() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    asymQConfig_t asymQConfig;
    initAsymQConfig(5, HALF_AWAY, &asymQConfig);
    asymQConfig.scale = 0.1875f;
    asymQConfig.zeroPoint = -11;

    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    quantization_t intQ;
    initInt32Quantization(&intQ);
    int32_t intData[numValues];
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    convertTensor(&asymTensor, &intTensor);

    int32_t actual[numValues];
    readBytesAsInt32Array(numValues, intTensor.data, actual);
    int32_t expectedData[] = {5, 11, 16, 20, -5, -11};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedData, actual, numValues);
}

void testConversionAsymFloat() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    asymQConfig_t asymQConfig;
    initAsymQConfig(5, HALF_AWAY, &asymQConfig);
    asymQConfig.scale = 0.1875f;
    asymQConfig.zeroPoint = -11;
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatData[numValues];

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&asymTensor, &floatTensor);

    float actual[numValues];
    readBytesAsFloatArray(numValues, floatTensor.data, actual);
    float expectedData[] = {0.9375f, 2.0625f, 3.f, 3.75f, -0.9375f, -2.0625f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedData, actual, numValues);
}

void testConversionAsymSymInt32() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    asymQConfig_t asymQConfig;
    initAsymQConfig(5, HALF_AWAY, &asymQConfig);
    asymQConfig.scale = 0.1875f;
    asymQConfig.zeroPoint = -11;
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);
    int32_t symInt32Data[numValues];

    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    convertTensor(&asymTensor, &symInt32Tensor);

    int32_t actual[numValues];
    readBytesAsInt32Array(numValues, symInt32Tensor.data, actual);

    int32_t expectedData[] = {5, 11, 16, 20, -5, -11};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedData, actual, numValues);
}

void testConversionBoolBoolCopiesOnlyPackedBytes() {
    /* N=9 BOOL elements occupy (9+7)/8 = 2 packed bytes; the same-type copy
     * must move exactly 2 bytes. Canary: the output payload sits at the start
     * of a 16-byte guard allocation whose bytes 2..15 hold sentinel 0xAA.
     * Before the fix, convertTensor memmoves N * calcBytesPerElement(BOOL)
     * = 9 bytes and clobbers the sentinels with the input buffer's 0x55
     * filler. Both buffers are oversized on purpose so the buggy 9-byte
     * memmove stays inside owned allocations and the RED run is
     * well-defined. initTensor is not used here because it allocates the
     * exact packed size (2 bytes), which would make the buggy copy run out
     * of bounds. */
    enum { N = 9, GUARD_BYTES = 16, PAYLOAD_BYTES = 2 };

    size_t dims[] = {N};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    quantization_t inQ;
    initBoolQuantization(&inQ);
    uint8_t *inBuffer = reserveMemory(GUARD_BYTES);
    memset(inBuffer, 0x55, GUARD_BYTES);
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuffer, &shape, &inQ, NULL);

    const bool pattern[N] = {true, false, true, true, false, false, true, false, true};
    tensorFillFromBoolBuffer(&inTensor, pattern, N);

    quantization_t outQ;
    initBoolQuantization(&outQ);
    uint8_t *outBuffer = reserveMemory(GUARD_BYTES);
    memset(outBuffer, 0xAA, GUARD_BYTES);
    tensor_t outTensor;
    setTensorValues(&outTensor, outBuffer, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    /* Under-copy guard: all 9 bits must arrive. */
    for (size_t i = 0; i < N; i++) {
        TEST_ASSERT_EQUAL(pattern[i], tensorBoolGet(&outTensor, i));
    }
    /* Over-copy guard: every byte after the packed payload is untouched. */
    for (size_t i = PAYLOAD_BYTES; i < GUARD_BYTES; i++) {
        TEST_ASSERT_EQUAL_UINT8(0xAA, outBuffer[i]);
    }

    freeReservedMemory(inBuffer);
    freeReservedMemory(outBuffer);
}

void testQuantTypeToStringBool() {
    TEST_ASSERT_EQUAL_STRING("BOOL", quantTypeToString(BOOL));
}

void testConversionSymInt32SameTypeCopyPropagatesScale() {
    /* Pins the same-type copy semantics PR C builds on: mantissas memmoved,
     * input scale overwrites any pre-set output scale, NO rescale happens.
     * symInt32QConfig_t is zero-initialized instead of using
     * initSymInt32QConfig so this test references no roundingMode_t
     * enumerator (PR A renames them in parallel); the same-type copy path
     * never reads roundingMode. */
    size_t numValues = 4;
    size_t dims[] = {4};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQC = {0};
    inQC.scale = 0.03125f;
    inQC.qMaxBits = 16;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {100, -200, 300, -400};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQC = {0};
    outQC.scale = 999.0f; /* pre-set garbage; the copy must overwrite it */
    outQC.qMaxBits = 16;
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[4] = {0, 0, 0, 0};
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(inData, outData, numValues);
    TEST_ASSERT_EQUAL_FLOAT(0.03125f, outQC.scale);
}

void testRequantDynamicAccumulatorRangeMatchesGold() {
    size_t dims[] = {input_requant_f1AccumRange_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f1AccumRange;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f1AccumRange_len];
    memcpy(inData, input_requant_f1AccumRange, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f1AccumRange_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f1AccumRange, outData,
                                  expected_requant_f1AccumRange_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f1AccumRange, expectedScale_requant_f1AccumRange,
                             outQc.scale);
    // out-of-place: the input tensor must be untouched (pass A reads only)
    TEST_ASSERT_EQUAL_INT32_ARRAY(input_requant_f1AccumRange, inData,
                                  input_requant_f1AccumRange_len);
    TEST_ASSERT_EQUAL_FLOAT(inputScale_requant_f1AccumRange, inQc.scale);
}

void testRequantDynamicAbsmaxZeroGivesZerosScaleOne() {
    size_t dims[] = {input_requant_f2AbsmaxZero_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f2AbsmaxZero;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f2AbsmaxZero_len];
    memcpy(inData, input_requant_f2AbsmaxZero, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f2AbsmaxZero_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f2AbsmaxZero, outData,
                                  expected_requant_f2AbsmaxZero_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f2AbsmaxZero, expectedScale_requant_f2AbsmaxZero,
                             outQc.scale);
}

void testRequantDynamicScaleTracksInputRescale() {
    // ONE output tensor + qConfig reused across BOTH calls (no re-init):
    // a kernel that fails to recompute/write the scale per call keeps the
    // stale value and fails the second assert (freeze-the-scale class).
    size_t dims[] = {input_requant_f3Rescale_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScaleA_requant_f3Rescale;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f3Rescale_len];
    memcpy(inData, input_requant_f3Rescale, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f3Rescale_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedA_requant_f3Rescale, outData,
                                  expectedA_requant_f3Rescale_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTolA_requant_f3Rescale, expectedScaleA_requant_f3Rescale,
                             outQc.scale);

    // same mantissas, input scale x10 -> fresh scale must track ~x10
    inQc.scale = inputScaleB_requant_f3Rescale;
    requantSymInt32Tensor(&inTensor, &outTensor);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedB_requant_f3Rescale, outData,
                                  expectedB_requant_f3Rescale_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTolB_requant_f3Rescale, expectedScaleB_requant_f3Rescale,
                             outQc.scale);
}

void testRequantDynamicTieRoundsHalfAwayFromZero() {
    // quotients land on exact .5 ties (gold construction: scale == 4.0f);
    // half-to-even AND floor/trunc casts produce different mantissas.
    size_t dims[] = {input_requant_f4Tie_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f4Tie;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f4Tie_len];
    memcpy(inData, input_requant_f4Tie, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f4Tie_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f4Tie, outData, expected_requant_f4Tie_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f4Tie, expectedScale_requant_f4Tie, outQc.scale);
}

void testRequantDynamicInPlaceAliasMatchesGold() {
    // In-place contract: ONE tensor_t passed as input AND output. Its single
    // qConfig is read in the input role (scale on entry) and written in the
    // output role (fresh scale on exit); qMaxBits/roundingMode of that same
    // config define the target. Pass B rewrites the mantissas index-by-index.
    // Result must be bit-identical to the out-of-place gold.
    size_t dims[] = {input_requant_f1AccumRange_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t qc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &qc, (uint8_t)qMaxBits_requant);
    qc.scale = inputScale_requant_f1AccumRange;
    quantization_t q;
    initSymInt32Quantization(&qc, &q);
    int32_t data[input_requant_f1AccumRange_len];
    memcpy(data, input_requant_f1AccumRange, sizeof(data));
    tensor_t tensor;
    setTensorValues(&tensor, (uint8_t *)data, &shape, &q, NULL);

    requantSymInt32Tensor(&tensor, &tensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f1AccumRange, data,
                                  expected_requant_f1AccumRange_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f1AccumRange, expectedScale_requant_f1AccumRange,
                             qc.scale);
}

void testRequantDynamicViaConversionMatrixDiagonal() {
    // The Quant layer (PR D) dispatches directly over the matrix; pin the
    // diagonal wiring and that it behaves identically to a direct call.
    conversionFunction_t conversionFn = conversionMatrix[SYM_INT32][SYM_INT32];
    TEST_ASSERT_NOT_NULL(conversionFn);

    size_t dims[] = {input_requant_f1AccumRange_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f1AccumRange;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f1AccumRange_len];
    memcpy(inData, input_requant_f1AccumRange, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f1AccumRange_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    conversionFn(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f1AccumRange, outData,
                                  expected_requant_f1AccumRange_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f1AccumRange, expectedScale_requant_f1AccumRange,
                             outQc.scale);
}

void testConvertTensorSymInt32SameTypeKeepsCopySemantics() {
    // Pins the spec-D1 invariant the PR-D Quant layer relies on:
    // convertTensor's same-type branch short-circuits BEFORE the matrix
    // lookup and stays memmove + scale copy — wiring the diagonal must NOT
    // change it. A requant here would yield {10922, -21845, 32767} with a
    // fresh scale 150/32767 instead of the copied mantissas + scale 0.5f.
    size_t dims[] = {3};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, 16);
    inQc.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[] = {100, -200, 300};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, 16);
    outQc.scale = 999.f;
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[3];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    int32_t expectedCopy[] = {100, -200, 300};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedCopy, outData, 3);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQc.scale);
}

void testRequantToScaleNonSaturatingMatchesGold() {
    size_t dims[] = {input_requant_f5ToScaleFit_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f5ToScaleFit;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f5ToScaleFit_len];
    memcpy(inData, input_requant_f5ToScaleFit, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    outQc.scale = targetScale_requant_f5ToScaleFit; // pre-set target (fixed-scale contract)
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f5ToScaleFit_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32TensorToScale(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f5ToScaleFit, outData,
                                  expected_requant_f5ToScaleFit_len);
    // fixed-scale contract: the pre-set target scale is NEVER modified
    TEST_ASSERT_EQUAL_FLOAT(targetScale_requant_f5ToScaleFit, outQc.scale);
}

void testRequantToScaleSaturatesAtQMinQMax() {
    // target scale deliberately too small: quotients overshoot BOTH bounds;
    // clamping at qMin/qMax is the documented Deutel-Eq.4 semantics.
    size_t dims[] = {input_requant_f6ToScaleSat_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f6ToScaleSat;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f6ToScaleSat_len];
    memcpy(inData, input_requant_f6ToScaleSat, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    outQc.scale = targetScale_requant_f6ToScaleSat;
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f6ToScaleSat_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32TensorToScale(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f6ToScaleSat, outData,
                                  expected_requant_f6ToScaleSat_len);
    TEST_ASSERT_EQUAL_FLOAT(targetScale_requant_f6ToScaleSat, outQc.scale);
}

void testRequantToScaleSharedBufferAliasMatchesGold() {
    // In-place for the fixed-scale variant = SHARED DATA BUFFER with two
    // tensor_t views, each with its OWN qConfig (input scale vs pre-set
    // target). Passing one tensor_t twice would force inScale == targetScale
    // (both roles share a single scale field) — a no-op requant — so the
    // two-view setup is the realistic aliasing mode. Pins the single
    // same-index read-then-write pass.
    size_t dims[] = {input_requant_f6ToScaleSat_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    int32_t data[input_requant_f6ToScaleSat_len];
    memcpy(data, input_requant_f6ToScaleSat, sizeof(data));

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f6ToScaleSat;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    tensor_t inView;
    setTensorValues(&inView, (uint8_t *)data, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    outQc.scale = targetScale_requant_f6ToScaleSat;
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    tensor_t outView;
    setTensorValues(&outView, (uint8_t *)data, &shape, &outQ, NULL);

    requantSymInt32TensorToScale(&inView, &outView);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f6ToScaleSat, data,
                                  expected_requant_f6ToScaleSat_len);
    TEST_ASSERT_EQUAL_FLOAT(targetScale_requant_f6ToScaleSat, outQc.scale);
}

void testConversionSymSymInt32SignExtends() {
    size_t numValues = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    /* SYM source, qBits=6, scale 0.5; mantissas {3,-3,31,-32} packed */
    symQConfig_t inQC = {0};
    inQC.scale = 0.5f;
    inQC.qBits = 6;
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t srcMant[] = {3, -3, 31, -32};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, numValues));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    /* pack the signed mantissas into the SYM bitstream */
    byteConversion((uint8_t *)srcMant, 32, inTensor.data, 6, numValues);

    symInt32QConfig_t outQC = {0};
    outQC.qMaxBits = 16;
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[4];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    int32_t expectedMant[] = {3, -3, 31, -32};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedMant, outData, numValues); /* sign preserved */
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQC.scale);                      /* scale carried */
    TEST_ASSERT_EQUAL_UINT8(6, outQC.qMaxBits);                      /* width recorded */
    freeReservedMemory(inBuf);
}

void testConversionSymFloat32Dequantizes() {
    size_t n = 3;
    size_t dims[] = {3};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    symQConfig_t inQC = {0};
    inQC.scale = 0.25f;
    inQC.qBits = 6;
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t mant[] = {4, -4, 2};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, n));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    byteConversion((uint8_t *)mant, 32, inTensor.data, 6, n);

    quantization_t outQ;
    initFloat32Quantization(&outQ);
    float outData[3];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    float expected[] = {1.0f, -1.0f, 0.5f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, outData, n);
    freeReservedMemory(inBuf);
}

void testConversionSymInt32CodesDropScale() {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    symQConfig_t inQC = {0};
    inQC.scale = 7.5f;
    inQC.qBits = 6; /* scale must be IGNORED */
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t mant[] = {5, -5, 1, -32};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, n));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    byteConversion((uint8_t *)mant, 32, inTensor.data, 6, n);

    quantization_t outQ;
    initInt32Quantization(&outQ);
    int32_t outData[4];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    int32_t expected[] = {5, -5, 1, -32};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, outData, n);
    freeReservedMemory(inBuf);
}

void testConversionSymAsymRescaleRoundTrips() {
    /* Round-trip: SYM -> ASYM -> FLOAT32 recovers dequantized SYM values.
     *
     * Fixture: n=6, SYM qBits=6, scale=0.5, mantissas {10,-8,4,-2,6,-10}.
     * Dequantized SYM: deq[i] = mant[i] * 0.5 => {5.0, -4.0, 2.0, -1.0, 3.0, -5.0}.
     *
     * ASYM qBits=5: range=10.0, qMax=32, asym scale=10/32=0.3125.
     * convertFloatTensorToAsymTensor clamps codes to [0, qMax-1=31].
     * zeroPoint = round(-5.0/0.3125) = -16; max code = clamp(32, 0, 31) = 31.
     * Recovered max = (31+(-16))*0.3125 = 4.6875; error = 0.3125 = 1 scale step.
     * Tolerance widened to 0.35 > 0.3125 to cover the clipped extremes.
     */
    size_t n = 6;
    size_t dims[] = {6};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    /* Build SYM input */
    symQConfig_t inQC = {0};
    inQC.scale = 0.5f;
    inQC.qBits = 6;
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t srcMant[] = {10, -8, 4, -2, 6, -10};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, n));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    byteConversion((uint8_t *)srcMant, 32, inTensor.data, 6, n);

    /* ASYM output tensor */
    asymQConfig_t asymQC;
    initAsymQConfig(5, HALF_AWAY, &asymQC);
    quantization_t asymQ;
    initAsymQuantization(&asymQC, &asymQ);
    uint8_t asymData[n * calcBytesPerElement(&asymQ)];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    /* FLOAT32 output tensor for round-trip verification */
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float outF[6];
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)outF, &shape, &floatQ, NULL);

    /* Convert SYM -> ASYM -> FLOAT32 */
    convertTensor(&inTensor, &asymTensor);
    convertTensor(&asymTensor, &floatTensor);

    /* Expected: dequantized SYM values */
    float expected[] = {5.0f, -4.0f, 2.0f, -1.0f, 3.0f, -5.0f};

    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.35f, expected[i], outF[i]);
    }

    freeReservedMemory(inBuf);
}

void testConversionSymInt32ToSymRescaleRoundTrips() {
    size_t n = 6;
    size_t dims[] = {6};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    /* Input: SYM_INT32 with scale=0.25, mantissas span [-40, 40].
     * Dequantized values: mantissa * 0.25 = {10, -8, 4, -2, 6, -10}; absmax = 10.0. */
    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.25f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {40, -32, 16, -8, 24, -40};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    /* Output: SYM with qBits=6.
     * Expected fresh scale = absmax / (2^(6-1) - 1) = 10.0 / 31 ≈ 0.322580645. */
    symQConfig_t outQC;
    initSymQConfig(6, HALF_AWAY, &outQC);
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &symTensor);

    /* Assert fresh output scale. */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 10.0f / 31.0f, outQC.scale);

    /* Manually unpack codes and dequantize; verify within one quant step.
     * One quant step = scale ≈ 0.323; tolerance = 0.33f. */
    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    float expectedVal[] = {10.f, -8.f, 4.f, -2.f, 6.f, -10.f};
    for (size_t i = 0; i < n; i++) {
        float rec = (float)codes[i] * outQC.scale;
        TEST_ASSERT_FLOAT_WITHIN(0.33f, expectedVal[i], rec);
    }

    /* Representative codes: 10.0 / (10.0/31) = 31; -10.0 / (10.0/31) = -31. */
    TEST_ASSERT_INT32_WITHIN(1, 31, codes[0]);
    TEST_ASSERT_INT32_WITHIN(1, -31, codes[5]);
}

void testRepackSymInt32ToSymNoRescaleFittingCarriesScale() {
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {5, -5, 31, -32, 0, 12};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symQConfig_t outQC;
    initSymQConfig(6, HALF_AWAY, &outQC);
    outQC.scale = 999.0f;
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    repackSymInt32ToSymNoRescale(&inTensor, &symTensor);

    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    TEST_ASSERT_EQUAL_INT32_ARRAY(inData, codes, 6);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQC.scale);
}

void testRepackSymInt32ToSymNoRescaleRejectsOverflow() {
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {5, 40, -5, 0, 0, 0};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symQConfig_t outQC;
    initSymQConfig(6, HALF_AWAY, &outQC);
    outQC.scale = 999.0f;
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(repackSymInt32ToSymNoRescale(&inTensor, &symTensor));
}

void testConversionFloatToSymRoundTripsSymmetric() {
    /* n=6, absMax=3.5 => scale = 3.5 / (2^(6-1) - 1) = 3.5/31 ≈ 0.112903226.
     * One quant step = scale ≈ 0.113; worst-case round-trip error = scale/2 ≈ 0.056;
     * tolerance 0.12 is > one full step to cover HALF_AWAY rounding at the boundary. */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    float floatData[] = {1.5f, -2.5f, 3.0f, -1.0f, 0.5f, -3.5f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    symQConfig_t outQC;
    initSymQConfig(6, HALF_AWAY, &outQC);
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&floatTensor, &symTensor);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.5f / 31.0f, outQC.scale);

    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    for (size_t i = 0; i < n; i++) {
        float rec = (float)codes[i] * outQC.scale;
        TEST_ASSERT_FLOAT_WITHIN(0.12f, floatData[i], rec);
    }

    /* Prove symmetric range: the OLD buggy code clamped to [0, qMax-1] so
     * negative inputs became 0; a non-zero negative code proves correct range. */
    TEST_ASSERT_TRUE(codes[1] < 0); /* floatData[1] = -2.5 */
    TEST_ASSERT_TRUE(codes[5] < 0); /* floatData[5] = -3.5 */
}

void testConversionInt32ToSymNoRescaleScale1() {
    /* INT32 codes fitting qBits=6 range [-32, 31] pack with scale=1. */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {5, -5, 31, -32, 0, 12};
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    symQConfig_t outQC;
    initSymQConfig(6, HALF_AWAY, &outQC);
    outQC.scale = 999.0f; /* garbage — proves scale=1 is written by the cell */
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&intTensor, &symTensor);

    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    TEST_ASSERT_EQUAL_INT32_ARRAY(intData, codes, 6);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, outQC.scale);
}

void testConversionInt32ToSymRejectsOutOfRange() {
    /* An INT32 code outside [-32, 31] for qBits=6 must exit(1).
     * Mutation guard: if packFitGuarded's range check is removed, the out-of-range
     * code 40 truncates silently and the child exits 0, failing this test. */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {5, 40, -5, 0, 0, 0}; /* 40 > 31: out of range for qBits=6 */
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    symQConfig_t outQC;
    initSymQConfig(6, HALF_AWAY, &outQC);
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&intTensor, &symTensor));
}

void testConversionSymInt32AsymConstantTensorNoDivByZero() {
    /* Constant tensor: min==max. The quantizeFloatToAsym degenerate branch must avoid
     * divide-by-zero and recover the constant. Before the dedup,
     * convertSymInt32TensorToAsymTensor had no min==max guard: scale=(max-min)/qMax=0,
     * so value/scale was inf and the result was garbage (UB on the float->int cast). */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {8, 8, 8, 8}; /* dequantized = 4.0 each (constant) */
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    asymQConfig_t asymQC;
    initAsymQConfig(5, HALF_AWAY, &asymQC);
    quantization_t asymQ;
    initAsymQuantization(&asymQC, &asymQ);
    uint8_t asymData[n * calcBytesPerElement(&asymQ)];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float outF[4];
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)outF, &shape, &floatQ, NULL);

    convertTensor(&inTensor, &asymTensor);    /* SYM_INT32 -> ASYM (constant) */
    convertTensor(&asymTensor, &floatTensor); /* ASYM -> FLOAT32 round-trip */

    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 4.0f, outF[i]);
    }
}

void testConversionAsymToSymRescaleOffCenterRoundTrips() {
    /* Strategy: build an ASYM input representing an off-center ALL-POSITIVE band [2, 6],
     * convert ASYM -> SYM, manually unpack the SYM output (sign-extend), dequantize with
     * the fresh SYM scale, and assert recovery within tolerance — proving the cell RESCALED
     * (a symmetric grid holds the [2,6] band only because the scale was recomputed). */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    /* Build ASYM input: qBits=5, scale=0.25, zeroPoint=-4.
     * asym codes {12,16,20,24,28,14} -> reals = (code + zeroPoint)*scale = (code-4)*0.25
     * = {2.0, 3.0, 4.0, 5.0, 6.0, 2.5} — off-center all-positive band [2, 6]. */
    asymQConfig_t inQC;
    initAsymQConfig(5, HALF_AWAY, &inQC);
    inQC.scale = 0.25f;
    inQC.zeroPoint = -4;
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);

    int32_t asymCodes[] = {12, 16, 20, 24, 28, 14};
    uint8_t asymData[calcNumberOfBytesForData(&inQ, 6)];
    byteConversion((uint8_t *)asymCodes, 32, asymData, 5, 6);
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &inQ, NULL);

    float reference[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.5f};

    /* SYM output qBits=6. */
    symQConfig_t outQC;
    initSymQConfig(6, HALF_AWAY, &outQC);
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, 6)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&asymTensor, &symTensor);

    /* Assert FRESH symmetric scale proves rescale (NOT the carried asym 0.25):
     * absMax = 6.0; scale = 6.0 / (2^(6-1) - 1) = 6.0/31. */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f / 31.0f, outQC.scale);

    /* Manual unpack + dequant + compare.
     * asym codes are exact integers so the only error is the SYM requantization step
     * ≈ scale/2 = (6/31)/2 ≈ 0.097; tolerance 0.2 is conservative (< one full step). */
    int32_t symCodes[6];
    symTestUnpackSignExtend(symTensor.data, 6, symCodes, 6);
    for (size_t i = 0; i < 6; i++) {
        float rec = (float)symCodes[i] * outQC.scale;
        TEST_ASSERT_FLOAT_WITHIN(0.2f, reference[i], rec);
    }

    /* All codes positive: off-center [2,6] band maps onto the positive half of the
     * symmetric grid after rescaling. */
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_TRUE(symCodes[i] > 0);
    }
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(testConversionIntFloat);
    RUN_TEST(testConversionIntSymInt32);
    RUN_TEST(testConversionIntAsym);

    RUN_TEST(testConversionFloatInt);
    RUN_TEST(testConversionFloatSymInt32);
    RUN_TEST(testConversionFloatAsym);

    RUN_TEST(testConversionSymInt32Int);
    RUN_TEST(testConversionSymInt32Float);
    RUN_TEST(testConversionSymInt32Asym);
    RUN_TEST(testConversionSymInt32AsymConstantTensorNoDivByZero);

    RUN_TEST(testConversionAsymInt);
    RUN_TEST(testConversionAsymFloat);
    RUN_TEST(testConversionAsymSymInt32);
    RUN_TEST(testRequantDynamicAccumulatorRangeMatchesGold);
    RUN_TEST(testRequantDynamicAbsmaxZeroGivesZerosScaleOne);
    RUN_TEST(testRequantDynamicScaleTracksInputRescale);
    RUN_TEST(testRequantDynamicTieRoundsHalfAwayFromZero);
    RUN_TEST(testRequantDynamicInPlaceAliasMatchesGold);
    RUN_TEST(testRequantDynamicViaConversionMatrixDiagonal);
    RUN_TEST(testConvertTensorSymInt32SameTypeKeepsCopySemantics);
    RUN_TEST(testRequantToScaleNonSaturatingMatchesGold);
    RUN_TEST(testRequantToScaleSaturatesAtQMinQMax);
    RUN_TEST(testRequantToScaleSharedBufferAliasMatchesGold);
    RUN_TEST(testConversionBoolBoolCopiesOnlyPackedBytes);
    RUN_TEST(testConversionSymInt32SameTypeCopyPropagatesScale);
    RUN_TEST(testQuantTypeToStringBool);
    RUN_TEST(testConversionSymSymInt32SignExtends);
    RUN_TEST(testConversionSymFloat32Dequantizes);
    RUN_TEST(testConversionSymInt32CodesDropScale);
    RUN_TEST(testConversionSymAsymRescaleRoundTrips);
    RUN_TEST(testConversionSymInt32ToSymRescaleRoundTrips);
    RUN_TEST(testRepackSymInt32ToSymNoRescaleFittingCarriesScale);
    RUN_TEST(testRepackSymInt32ToSymNoRescaleRejectsOverflow);
    RUN_TEST(testConversionFloatToSymRoundTripsSymmetric);
    RUN_TEST(testConversionInt32ToSymNoRescaleScale1);
    RUN_TEST(testConversionInt32ToSymRejectsOutOfRange);
    RUN_TEST(testConversionAsymToSymRescaleOffCenterRoundTrips);

    return UNITY_END();
}
