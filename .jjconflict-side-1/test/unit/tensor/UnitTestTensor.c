#include "DTypes.h"
#include "DeathTest.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

#include <stdlib.h>
#include <string.h>

void testGetBitmask() {
    uint8_t startbit = 1;
    uint8_t endbit = 4;
    uint8_t bitmask = getBitmask(startbit, endbit);
    uint8_t expected = 0b00001110;
    TEST_ASSERT_EQUAL(expected, bitmask);
}

void testGetBitmask2() {
    uint8_t startbit = 10;
    uint8_t endbit = 14;
    uint8_t bitmask = getBitmask(startbit, endbit);
    uint8_t expected = 0b00111100;
    TEST_ASSERT_EQUAL(expected, bitmask);
}

void testReadByte() {
    uint8_t startbit = 1;
    uint8_t endbit = 4;
    uint8_t data = 0b00101010;
    uint8_t actual = readByte(data, startbit, endbit);
    uint8_t expected = 0b00000101;
    TEST_ASSERT_EQUAL(expected, actual);
}

void testWriteByte() {
    uint8_t existing_data = 0b00000101;
    uint8_t data = 0b00000101;
    uint8_t newData = writeByte(existing_data, data, 3, 7);
    uint8_t expected = 0b00101101;
    TEST_ASSERT_EQUAL_UINT8(expected, newData);
}

void testWriteByte2() {
    uint8_t existing_data = 0b00000101;
    uint8_t data = 0b00010101;
    uint8_t newData = writeByte(existing_data, data, 3, 11);
    uint8_t expected = 0b10101101;
    TEST_ASSERT_EQUAL_UINT8(expected, newData);
}

void testByteFlattening() {
    // {1, 2, 78}
    uint8_t dataIn[] = {0b000000001, 0b00000110, 0b00111000, 0b00000001};
    size_t dataInBits = 9;

    size_t dataOutBits = 19;
    size_t numValues = 3;
    size_t numBytesDataOut = (dataOutBits * numValues - 1) / 8 + 1;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0b000000001, 0b00000000, 0b00011000, 0b00000000,
                               0b10000000,  0b00010011, 0b00000000, 0b00000000};

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening2() {
    // {1, 2, 78}
    uint8_t dataIn[] = {0b000000001, 0b00000000, 0b00011000, 0b00000000,
                        0b10000000,  0b00010011, 0b00000000, 0b00000000};
    size_t dataInBits = 19;

    size_t dataOutBits = 9;
    size_t numValues = 3;
    size_t numBytesDataOut = (dataOutBits * numValues - 1) / 8 + 1;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0b000000001, 0b00000110, 0b00111000, 0b00000001};

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening3() {
    // {1, 2, 78}
    uint8_t dataIn[] = {0b000000001, 0b00000000, 0b00001100, 0b00000000,
                        0b00000100,  0b00001011, 0b00000000, 0b00000000};
    size_t dataInBits = 8;

    size_t dataOutBits = 4;
    size_t numValues = 8;
    size_t numBytesDataOut = (dataOutBits * numValues - 1) / 8 + 1;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0b000000001, 0b00001100, 0b10110100, 0b00000000};

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening4() {
    uint8_t dataIn[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};
    size_t dataInBits = 5;
    size_t dataOutBits = 8;
    size_t numBytesDataOut = 6;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numBytesDataOut);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {16, 22, 27, 31, 6, 0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening5() {
    uint8_t dataIn[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};
    size_t dataInBits = 5;
    size_t dataOutBits = 32;
    size_t numValues = 6;
    size_t numBytesDataOut = 4 * numValues;
    uint8_t dataOut[numBytesDataOut];
    ;
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);
    uint8_t expectedBytes[] = {16, 0, 0, 0, 22, 0, 0, 0, 27, 0, 0, 0,
                               31, 0, 0, 0, 6,  0, 0, 0, 0,  0, 0, 0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, numBytesDataOut);
}

/* Regression for an ASan-detected heap-buffer-overflow surfaced by
 * testLinearBackwardFloatWithMismatchedQuantizations on the float32 -> asym8
 * conversion path. byteConversion's inner while-loop keeps iterating after
 * the output for a value is full to consume remaining input bits, accessing
 * dataOut[dataOutIndex] one byte past the end of the buffer. The OOB write
 * is a no-op on the value (writeByte with startbit==endbit), so the bug is
 * invisible to value assertions and only fires under ASan. */
void testByteConversion_NarrowingInt32ToByte_DoesNotOverreadOutputBuffer() {
    /* Two int32 values, little-endian: 0xAABBCCDD, 0xEEFF1122. */
    uint8_t dataIn[8] = {0xDD, 0xCC, 0xBB, 0xAA, 0x22, 0x11, 0xFF, 0xEE};
    size_t dataInBits = 32;
    size_t dataOutBits = 8;
    size_t numValues = 2;
    size_t numBytesDataOut = (numValues * dataOutBits - 1) / 8 + 1; /* = 2 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[2] = {0xDD, 0x22};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

/* Companion to the narrowing test above, exercising the sub-byte path of #86.
 * 32-bit input -> 4-bit output, 4 values packed into 2 bytes. The same
 * loop-control bug shows up: after the last value's bits are written, the
 * inner while-loop continues consuming input bits and accesses
 * dataOut[dataOutIndex] one byte past the end of the buffer. */
void testByteConversion_NarrowingInt32ToSubByte_DoesNotOverreadOutputBuffer() {
    /* Four int32 values, low 4 bits matter: 0xA, 0xB, 0xC, 0xD. */
    uint8_t dataIn[16] = {
        0x0A, 0, 0, 0, 0x0B, 0, 0, 0, 0x0C, 0, 0, 0, 0x0D, 0, 0, 0,
    };
    size_t dataInBits = 32;
    size_t dataOutBits = 4;
    size_t numValues = 4;
    size_t numBytesDataOut = (numValues * dataOutBits - 1) / 8 + 1; /* = 2 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    /* byte 0 = (0xB << 4) | 0xA = 0xBA; byte 1 = (0xD << 4) | 0xC = 0xDC. */
    uint8_t expectedBytes[2] = {0xBA, 0xDC};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

//companion to the test above
void testByteConversionOffsets_startbitInOffset5() {
    /* Four int32 values, (after 0xA) low 24 bits matter (numValues): 0, 0xC, 0xD.
     * 0xFF should not be read     */
    uint8_t dataIn[17] = {0xA, 0, 0, 0, 0xB0, 0, 0, 0, 0xC0, 0, 0, 0, 0xD0, 0, 0, 0, 0xFF};
    size_t dataInBits = 32;
    size_t dataOutBits = 24;
    size_t numValues = 4;
    size_t startbitInOffset = 5;
    size_t startbitOutOffset = 0;
    size_t numBytesDataOut = (numValues * dataOutBits - 1) / 8 + 1; /* = 2 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    memset(dataOut, 0, numBytesDataOut);
    byteConversionWithOffsets(dataIn, dataInBits, dataOut, startbitInOffset, dataOutBits, numValues,
                              startbitOutOffset);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);
    uint8_t expectedBytes[12] = {0, 0, 0, 0b00000101, 0, 0, 0b00000110, 0, 0, 0b00000110, 0, 0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteConversionOffsets_startbitInOffset8() {
    /* Four int32 values, (after 0xA) low 16 bits matter (numValues): 0, 0xC, 0xD.
     */
    uint8_t dataIn[17] = {0xA, 0, 0, 0, 0, 0xB, 0, 0, 0, 0xC, 0, 0, 0, 0xD, 0, 0, 0};
    size_t dataInBits = 32;
    size_t dataOutBits = 16;
    size_t numValues = 4;
    size_t startbitInOffset = 8;
    size_t startbitOutOffset = 0;
    size_t numBytesDataOut = (numValues * dataOutBits - 1) / 8 + 1; /* = 2 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    memset(dataOut, 0, numBytesDataOut);
    byteConversionWithOffsets(dataIn, dataInBits, dataOut, startbitInOffset, dataOutBits, numValues,
                              startbitOutOffset);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);
    uint8_t expectedBytes[8] = {0, 0, 0xB, 0, 0xC, 0, 0xD, 0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteConversionOffsets_startbitInOffset4() {
    /* Four int32 values, (after 0xA) low 4 bits matter (numValues): 0, 0xC, 0xD.
     */
    uint8_t dataIn[17] = {0xA, 0, 0, 0, 0xB, 0, 0, 0, 0xC0, 0, 0, 0, 0xD0, 0, 0, 0, 0};
    size_t dataInBits = 32;
    size_t dataOutBits = 4;
    size_t numValues = 4;
    size_t startbitInOffset = 4;
    size_t startbitOutOffset = 0;
    size_t numBytesDataOut = (numValues * dataOutBits - 1) / 8 + 1; /* = 2 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    memset(dataOut, 0, numBytesDataOut);
    byteConversionWithOffsets(dataIn, dataInBits, dataOut, startbitInOffset, dataOutBits, numValues,
                              startbitOutOffset);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);
    /* byte 0 = 0 | 0
     * byte 1 = (0xD0 << 4) | 0xC = 0xDC;     */
    uint8_t expectedBytes[2] = {0, 0xDC};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteConversionOffsets_startbitOutOffset4() {
    /* Four int32 values, low 4 bits matter: 0xA, 0xB, 0xC, 0xD. */
    uint8_t dataIn[16] = {
        0xA, 0, 0, 0, 0xB, 0, 0, 0, 0xC, 0, 0, 0, 0xD, 0, 0, 0,
    };
    size_t dataInBits = 32;
    size_t dataOutBits = 4;
    size_t numValues = 4;
    size_t startbitInOffset = 0;
    size_t startbitOutOffset = 4;
    size_t numBytesDataOut = (numValues * dataOutBits + startbitOutOffset - 1) / 8 + 1; /* = 3 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    memset(dataOut, 0, numBytesDataOut);
    byteConversionWithOffsets(dataIn, dataInBits, dataOut, startbitInOffset, dataOutBits, numValues,
                              startbitOutOffset);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    /* byte 0 = (0xA << 4) = 0xA0;
     * byte 1 = (0xC << 4) | 0xB = 0xCB;     * byte 2 = 0xD = 0xD;     */
    uint8_t expectedBytes[3] = {0xA0, 0xCB, 0xD};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteConversionOffsets_inAndOutOffset() {
    /* Four int32 values, low 4 bits matter: 0xA, 0xB, 0xC, 0xD. */
    uint8_t dataIn[16] = {
        0, 0xA, 0, 0, 0, 0xB, 0, 0, 0, 0xC, 0, 0, 0, 0xD, 0, 0,
    };
    size_t dataInBits = 32;
    size_t dataOutBits = 4;
    size_t numValues = 4;
    size_t startbitInOffset = 8;
    size_t startbitOutOffset = 4;
    size_t numBytesDataOut = (numValues * dataOutBits + startbitOutOffset - 1) / 8 + 1; /* = 3 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    memset(dataOut, 0, numBytesDataOut);
    byteConversionWithOffsets(dataIn, dataInBits, dataOut, startbitInOffset, dataOutBits, numValues,
                              startbitOutOffset);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[3] = {0xA0, 0xCB, 0xD};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

// use case  for Deltas
void testByteConversionTogetherWithByteConversionOffsets() {
    /* Four int32 values, low 4 bits matter: 0xA, 0xB, 0xC, 0xD. */
    uint8_t dataIn[16] = {
        0xA, 0, 0, 0, 0xB, 0, 0, 0, 0xC, 0, 0, 0, 0xD, 0, 0, 0,
    };
    size_t dataInBits = 32;
    size_t dataOutBits = 5;
    size_t numValues = 1;
    size_t deltaNumValues = 3;
    size_t deltaOutBits = 4;
    size_t startbitInOffset = dataInBits;
    size_t startbitOutOffset = dataOutBits;
    size_t numBytesDataOut =
        (numValues * dataOutBits + deltaNumValues * deltaOutBits - 1) / 8 + 1; /* = 3 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    memset(dataOut, 0, numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, 1);
    printf("done\n");
    byteConversionWithOffsets(dataIn, dataInBits, dataOut, startbitInOffset, deltaOutBits,
                              deltaNumValues, startbitOutOffset);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    /* byte 0 = (0xB << 5) | 0xA = (0b00001011 << 5) | 0b00001010 = 0b01101010 -> "remember"
     * 0b00001xxx as 0b00000001 byte 1 = ((0xD << 4) << 1) | (0xC << 1) | 0b00000001 = ((0b00001101
     * << 4) << 1) | (0b00001100 << 1) | 0b00000001 = 0b10111001 -> "remember" 0b00001xxx as
     * 0b00000001     * byte 2 = 0b00000001     */
    uint8_t expectedBytes[3] = {0b01101010, 0b10111001, 0b00000001};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testCopyTensor() {
    size_t numberOfValues = 3;
    tensor_t src;
    float data[] = {1.f, 2.f, 3.f};
    size_t dims[] = {1, numberOfValues};
    size_t numberOfDims = 2;
    size_t orderOfDims[] = {0, 1};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initFloat32Quantization(&q);

    setTensorValues(&src, (uint8_t *)data, &shape, &q, NULL);

    tensor_t dest;
    float destData[numberOfValues];
    size_t destDims[2];
    size_t destNumberOfDims;
    size_t destOrderOfDims[2];
    shape_t destShape = {.dimensions = destDims,
                         .numberOfDimensions = destNumberOfDims,
                         .orderOfDimensions = destOrderOfDims};
    setTensorValues(&dest, (uint8_t *)destData, &destShape, &q, NULL);

    copyTensor(&dest, &src);

    float expectedData[] = {1.f, 2.f, 3.f};
    size_t expectedDims[] = {1, numberOfValues};
    size_t expectedNumberOfDims = 2;
    size_t expectedOrderOfDims[] = {0, 1};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedData, dest.data, numberOfValues);
    TEST_ASSERT_EQUAL_size_t_ARRAY(expectedDims, dest.shape->dimensions, 2);
    TEST_ASSERT_EQUAL_size_t(expectedNumberOfDims, dest.shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t_ARRAY(expectedOrderOfDims, dest.shape->orderOfDimensions, 2);
}

void test_calcBitsPerElement_Sym_qBits3() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    TEST_ASSERT_EQUAL_size_t(3, calcBitsPerElement(&q));
}

void test_calcBytesPerElement_Sym_qBits3() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    /* ceil(3/8) = 1 */
    TEST_ASSERT_EQUAL_size_t(1, calcBytesPerElement(&q));
}

void test_calcNumberOfBytesForData_Sym_qBits3_N10() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    /* ceil(3*10 / 8) = ceil(30/8) = 4 */
    TEST_ASSERT_EQUAL_size_t(4, calcNumberOfBytesForData(&q, 10));
}

void test_calcNumberOfBytesForData_Delta_qBits3_N10_deltabits2() {
    symQDeltaConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY, .deltabits = 2};
    quantization_t q;
    initSymQDeltaQuantization(&cfg, &q);
    /* ceil((2*9 + 3)/ 8) = ceil(21/8) = 3 */
    TEST_ASSERT_EQUAL_size_t(3, calcNumberOfBytesForData(&q, 10));
}

void test_calcNumberOfBytesForData_Delta_qBits7_N10_deltabits5() {
    symQDeltaConfig_t cfg = {.scale = 1.0f, .qBits = 9, .roundingMode = HALF_AWAY, .deltabits = 5};
    quantization_t q;
    initSymQDeltaQuantization(&cfg, &q);
    /* ceil((9*5 + 9)/ 8) = ceil(54/8) = 7 */
    TEST_ASSERT_EQUAL_size_t(7, calcNumberOfBytesForData(&q, 10));
}

void test_calcNumberOfBytesForData_Sym_qBits5_N4() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 5, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    /* ceil(5*4 / 8) = ceil(20/8) = 3 */
    TEST_ASSERT_EQUAL_size_t(3, calcNumberOfBytesForData(&q, 4));
}

void test_calcNumberOfBytesForData_Delta_qBits5_N4_deltabits2() {
    symQDeltaConfig_t cfg = {.scale = 1.0f, .qBits = 5, .roundingMode = HALF_AWAY, .deltabits = 2};
    quantization_t q;
    initSymQDeltaQuantization(&cfg, &q);
    /* ceil((2*3 + 5)/ 8) = ceil(11/8) = 2 */
    TEST_ASSERT_EQUAL_size_t(2, calcNumberOfBytesForData(&q, 4));
}

void test_calcNumberOfBytesForData_Delta_qBits5_N4_deltabits3() {
    symQDeltaConfig_t cfg = {.scale = 1.0f, .qBits = 5, .roundingMode = HALF_AWAY, .deltabits = 3};
    quantization_t q;
    initSymQDeltaQuantization(&cfg, &q);
    /* ceil((3*3 + 5)/ 8) = ceil(14/8) = 2 */
    TEST_ASSERT_EQUAL_size_t(2, calcNumberOfBytesForData(&q, 4));
}

void test_calcBytesPerTensor_SymQBits3N10_Ceils() {
    /* 10 x 3 bits = 30 bits -> 4 bytes. Pre-fix truncation gave 30/8 = 3 (#172).
     * Mutation guard: reverting to bits/8 truncation returns 3 -> RED. */
    size_t dims[] = {1, 10};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    tensor_t t;
    setTensorValues(&t, NULL, &shape, &q, NULL);
    TEST_ASSERT_EQUAL_size_t(4, calcBytesPerTensor(&t));
}

void test_calcBytesPerTensor_BoolN3_CeilsTo1() {
    /* 3 x 1 bit -> 1 byte; pre-fix truncation gave 0 (#172). */
    size_t dims[] = {1, 3};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};
    quantization_t q;
    initBoolQuantization(&q);
    tensor_t t;
    setTensorValues(&t, NULL, &shape, &q, NULL);
    TEST_ASSERT_EQUAL_size_t(1, calcBytesPerTensor(&t));
}

void testCopyTensorInt32CarriesTypeAndData() {
    /* Mutation guard: re-removing the INT32 arm makes copyQuantization exit(1)
     * ("Unknown QType!"), killing the test run — RED. */
    int32_t srcData[] = {7, -3, 2000000000, -2000000000};
    size_t srcDims[] = {1, 4};
    size_t srcOrder[] = {0, 1};
    shape_t srcShape = {
        .dimensions = srcDims, .numberOfDimensions = 2, .orderOfDimensions = srcOrder};
    quantization_t qSrc;
    initInt32Quantization(&qSrc);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)srcData, &srcShape, &qSrc, NULL);

    int32_t dstData[4] = {0};
    size_t dstDims[2];
    size_t dstOrder[2];
    shape_t dstShape = {
        .dimensions = dstDims, .numberOfDimensions = 2, .orderOfDimensions = dstOrder};
    quantization_t qDst;
    initFloat32Quantization(&qDst); /* NULL-config family: overwritten by copyQuantization */
    tensor_t dst;
    setTensorValues(&dst, (uint8_t *)dstData, &dstShape, &qDst, NULL);

    copyTensor(&dst, &src);

    TEST_ASSERT_EQUAL_INT(INT32, dst.quantization->type);
    TEST_ASSERT_NULL(dst.quantization->qConfig);
    TEST_ASSERT_EQUAL_INT32_ARRAY(srcData, (int32_t *)dst.data, 4);
}

void testCopyTensorSymCarriesConfigAndPackedBytes() {
    /* 4 mantissas at qBits=6 -> ceil(24/8) = 3 packed bytes. Dest starts with a
     * deliberately different config (scale 1.f, SR_HALF_AWAY): every field must be
     * overwritten by the copy, INTO the caller's storage (no pointer swap).
     * Mutation guard: re-removing the SYM arm exits the run ("Unknown QType!"). */
    int32_t mantissas[] = {3, -3, 31, -32};
    uint8_t srcData[3];
    byteConversion((uint8_t *)mantissas, 32, srcData, 6, 4);
    size_t srcDims[] = {1, 4};
    size_t srcOrder[] = {0, 1};
    shape_t srcShape = {
        .dimensions = srcDims, .numberOfDimensions = 2, .orderOfDimensions = srcOrder};
    symQConfig_t srcQC = {.scale = 0.25f, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t qSrc;
    initSymQuantization(&srcQC, &qSrc);
    tensor_t src;
    setTensorValues(&src, srcData, &srcShape, &qSrc, NULL);

    uint8_t dstData[3] = {0};
    size_t dstDims[2];
    size_t dstOrder[2];
    shape_t dstShape = {
        .dimensions = dstDims, .numberOfDimensions = 2, .orderOfDimensions = dstOrder};
    symQConfig_t dstQC = {.scale = 1.f, .qBits = 6, .roundingMode = SR_HALF_AWAY};
    quantization_t qDst;
    initSymQuantization(&dstQC, &qDst);
    tensor_t dst;
    setTensorValues(&dst, dstData, &dstShape, &qDst, NULL);

    copyTensor(&dst, &src);

    TEST_ASSERT_EQUAL_INT(SYM, dst.quantization->type);
    TEST_ASSERT_EQUAL_PTR(&dstQC, dst.quantization->qConfig);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, dstQC.scale);
    TEST_ASSERT_EQUAL_UINT8(6, dstQC.qBits);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, dstQC.roundingMode);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(srcData, dstData, 3);
}

void testCopyTensorAsymCarriesConfigAndPackedBytes() {
    /* 4 codes at qBits=5 -> ceil(20/8) = 3 packed bytes. Mutation guard: re-removing
     * the ASYM arm exits the run ("Unknown QType!"). */
    int32_t codes[] = {0, 10, 20, 31};
    uint8_t srcData[3];
    byteConversion((uint8_t *)codes, 32, srcData, 5, 4);
    size_t srcDims[] = {1, 4};
    size_t srcOrder[] = {0, 1};
    shape_t srcShape = {
        .dimensions = srcDims, .numberOfDimensions = 2, .orderOfDimensions = srcOrder};
    asymQConfig_t srcQC = {.scale = 0.5f, .zeroPoint = -7, .qBits = 5, .roundingMode = HALF_AWAY};
    quantization_t qSrc;
    initAsymQuantization(&srcQC, &qSrc);
    tensor_t src;
    setTensorValues(&src, srcData, &srcShape, &qSrc, NULL);

    uint8_t dstData[3] = {0};
    size_t dstDims[2];
    size_t dstOrder[2];
    shape_t dstShape = {
        .dimensions = dstDims, .numberOfDimensions = 2, .orderOfDimensions = dstOrder};
    asymQConfig_t dstQC = {.scale = 1.f, .zeroPoint = 0, .qBits = 5, .roundingMode = SR_HALF_AWAY};
    quantization_t qDst;
    initAsymQuantization(&dstQC, &qDst);
    tensor_t dst;
    setTensorValues(&dst, dstData, &dstShape, &qDst, NULL);

    copyTensor(&dst, &src);

    TEST_ASSERT_EQUAL_INT(ASYM, dst.quantization->type);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, dstQC.scale);
    TEST_ASSERT_EQUAL_INT16(-7, dstQC.zeroPoint);
    TEST_ASSERT_EQUAL_UINT8(5, dstQC.qBits);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, dstQC.roundingMode);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(srcData, dstData, 3);
}

void testCopyTensorSymIntoNullConfigDestDies() {
    /* Config-carrying src into a dest whose qConfig is NULL (FLOAT32-initialized)
     * must fail fast, not memcpy through NULL. Mutation guard: without the NULL
     * check the child dies by SIGSEGV (signal, not exit(1)) and
     * ASSERT_EXITS_WITH_FAILURE flags the wrong termination kind. */
    int32_t mantissas[] = {1, -1};
    uint8_t srcData[2];
    byteConversion((uint8_t *)mantissas, 32, srcData, 6, 2);
    size_t srcDims[] = {1, 2};
    size_t srcOrder[] = {0, 1};
    shape_t srcShape = {
        .dimensions = srcDims, .numberOfDimensions = 2, .orderOfDimensions = srcOrder};
    symQConfig_t srcQC = {.scale = 1.f, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t qSrc;
    initSymQuantization(&srcQC, &qSrc);
    tensor_t src;
    setTensorValues(&src, srcData, &srcShape, &qSrc, NULL);

    uint8_t dstData[2] = {0};
    size_t dstDims[2];
    size_t dstOrder[2];
    shape_t dstShape = {
        .dimensions = dstDims, .numberOfDimensions = 2, .orderOfDimensions = dstOrder};
    quantization_t qDst;
    initFloat32Quantization(&qDst); /* qConfig == NULL on purpose */
    tensor_t dst;
    setTensorValues(&dst, dstData, &dstShape, &qDst, NULL);

    ASSERT_EXITS_WITH_FAILURE(copyTensor(&dst, &src));
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testByteFlattening);
    RUN_TEST(testByteFlattening2);
    RUN_TEST(testByteFlattening3);
    RUN_TEST(testByteFlattening4);
    RUN_TEST(testByteFlattening5);
    RUN_TEST(testByteConversion_NarrowingInt32ToSubByte_DoesNotOverreadOutputBuffer);
    RUN_TEST(testByteConversion_NarrowingInt32ToByte_DoesNotOverreadOutputBuffer);

    RUN_TEST(testGetBitmask);
    RUN_TEST(testGetBitmask2);
    RUN_TEST(testWriteByte);
    RUN_TEST(testWriteByte2);
    RUN_TEST(testReadByte);

    RUN_TEST(testCopyTensor);
    RUN_TEST(testCopyTensorInt32CarriesTypeAndData);
    RUN_TEST(testCopyTensorSymCarriesConfigAndPackedBytes);
    RUN_TEST(testCopyTensorAsymCarriesConfigAndPackedBytes);
    RUN_TEST(testCopyTensorSymIntoNullConfigDestDies);

    RUN_TEST(test_calcBitsPerElement_Sym_qBits3);
    RUN_TEST(test_calcBytesPerElement_Sym_qBits3);
    RUN_TEST(test_calcNumberOfBytesForData_Sym_qBits3_N10);
    RUN_TEST(test_calcNumberOfBytesForData_Sym_qBits5_N4);
    RUN_TEST(test_calcBytesPerTensor_SymQBits3N10_Ceils);
    RUN_TEST(test_calcBytesPerTensor_BoolN3_CeilsTo1);

    RUN_TEST(test_calcNumberOfBytesForData_Delta_qBits3_N10_deltabits2);
    RUN_TEST(test_calcNumberOfBytesForData_Delta_qBits7_N10_deltabits5);
    RUN_TEST(test_calcNumberOfBytesForData_Delta_qBits5_N4_deltabits2);
    RUN_TEST(test_calcNumberOfBytesForData_Delta_qBits5_N4_deltabits3);


    RUN_TEST(testByteConversionTogetherWithByteConversionOffsets);
    RUN_TEST(testByteConversionOffsets_startbitOutOffset4);
    RUN_TEST(testByteConversionOffsets_startbitInOffset4);
    RUN_TEST(testByteConversionOffsets_startbitInOffset8);
    RUN_TEST(testByteConversionOffsets_startbitInOffset5);

    RUN_TEST(testByteConversionOffsets_inAndOutOffset);

    return UNITY_END();
}
