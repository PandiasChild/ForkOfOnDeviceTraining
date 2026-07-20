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

/* numValues == 0 is a no-op: the previous memset size expression
 * (numValues*dataOutBits-1)/8+1 underflowed to ~SIZE_MAX and crashed. One
 * guard away from production: convertSymTensorToInt32Tensor passes n
 * straight through, and N=0 tensors are constructible (#160). */
void testByteConversion_ZeroValuesIsNoOp() {
    int32_t vals[1] = {0x7};
    uint8_t dataOut[2] = {0xAB, 0xCD};
    byteConversion((uint8_t *)vals, 32, dataOut, 12, 0);
    uint8_t expectedBytes[] = {0xAB, 0xCD};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, 2);
}

/* dataOutBits == 0 hit the same size underflow; with the canonical (bits+7)/8
 * ceiling it degrades to an empty write: input is consumed, nothing is
 * written. (packChunkGuarded rejects zero widths at the production entry.) */
void testByteConversion_ZeroOutBitsWritesNothing() {
    int32_t vals[2] = {0x3, 0x5};
    uint8_t dataOut[2] = {0xAB, 0xCD};
    byteConversion((uint8_t *)vals, 32, dataOut, 0, 2);
    uint8_t expectedBytes[] = {0xAB, 0xCD};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, 2);
}

/* writeByte fully defines the [startbit, endbit) range: stale 1-bits inside
 * the mask are cleared, not OR-merged; bits outside the mask are preserved.
 * byteConversion pre-zeroes via memset, so this is behavior-identical there;
 * it makes bit-offset appends robust on previously written buffers. */
void testWriteByte_ClearsStaleInMaskBits() {
    uint8_t existing_data = 0xFF;
    uint8_t data = 0b00000010;
    uint8_t newData = writeByte(existing_data, data, 1, 4);
    uint8_t expected = 0b11110101;
    TEST_ASSERT_EQUAL_UINT8(expected, newData);
}

/* A packed mixed-width stream stops mid-byte (3 x 5 bits = 15 bits) and a
 * later call continues in that same byte. byteConversion cannot express this
 * (out cursor pinned to bit 0, leading memset clobbers the shared byte);
 * byteConversionAppend seeds the out cursor from dstStartBit instead. */
void testByteConversionAppend_ContinuesMidByteAtBitOffset() {
    int32_t seg1[3] = {0x1F, 0x15, 0x0A};
    int32_t seg2[3] = {0x11, 0x1E, 0x03};
    size_t numBytesDataOut = (6 * 5 - 1) / 8 + 1; /* 30 bits -> 4 bytes */
    uint8_t *dataOut = reserveMemory(numBytesDataOut);

    byteConversionAppend((uint8_t *)seg1, 32, dataOut, 5, 3, 0);
    byteConversionAppend((uint8_t *)seg2, 32, dataOut, 5, 3, 15);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0xBF, 0xAA, 0xE8, 0x07};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

/* Delta-compression layout: 16-bit base + 5 x 3-bit deltas per group, two
 * groups back to back (62 bits) — the second group's base starts mid-byte
 * at bit 31 and spans three bytes. */
void testByteConversionAppend_MixedWidthDeltaLayout() {
    int32_t base0[1] = {0x1234};
    int32_t deltas0[5] = {5, 2, 7, 0, 3};
    int32_t base1[1] = {0xBEEF};
    int32_t deltas1[5] = {1, 6, 4, 2, 5};
    size_t numBytesDataOut = (62 - 1) / 8 + 1; /* = 8 */
    uint8_t *dataOut = reserveMemory(numBytesDataOut);

    byteConversionAppend((uint8_t *)base0, 32, dataOut, 16, 1, 0);
    byteConversionAppend((uint8_t *)deltas0, 32, dataOut, 3, 5, 16);
    byteConversionAppend((uint8_t *)base1, 32, dataOut, 16, 1, 31);
    byteConversionAppend((uint8_t *)deltas1, 32, dataOut, 3, 5, 47);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0x34, 0x12, 0xD5, 0xB1, 0x77, 0xDF, 0x98, 0x2A};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

/* Bits outside [dstStartBit, dstStartBit + numValues*dataOutBits) are
 * preserved and stale in-range bits are cleared: 2 x 4 bits at bit 4 into an
 * all-ones buffer leaves both nibble neighbors intact. */
void testByteConversionAppend_DefinesRangePreservesOutside() {
    int32_t vals[2] = {0x5, 0xA};
    uint8_t dataOut[3] = {0xFF, 0xFF, 0xFF};
    byteConversionAppend((uint8_t *)vals, 32, dataOut, 4, 2, 4);
    uint8_t expectedBytes[] = {0x5F, 0xFA, 0xFF};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, 3);
}

/* At dstStartBit == 0 append is bit-identical to byteConversion on a zeroed
 * buffer (same fixture as testByteFlattening4: packed 5-bit input widened
 * to 8-bit output — covers sub-byte packed INPUT through the append path). */
void testByteConversionAppend_OffsetZeroMatchesByteConversion() {
    uint8_t dataIn[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};
    uint8_t dataOut[6] = {0};
    byteConversionAppend(dataIn, 5, dataOut, 8, 6, 0);
    uint8_t expectedBytes[] = {16, 22, 27, 31, 6, 0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, 6);
}

/* numValues == 0 is a no-op. (byteConversion's memset-size expression
 * (numValues*dataOutBits-1)/8+1 underflows for 0; append has no memset and
 * must not touch the buffer at all.) */
void testByteConversionAppend_ZeroValuesIsNoOp() {
    int32_t vals[1] = {0x7};
    uint8_t dataOut[2] = {0xAB, 0xCD};
    byteConversionAppend((uint8_t *)vals, 32, dataOut, 12, 0, 3);
    uint8_t expectedBytes[] = {0xAB, 0xCD};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, 2);
}

/* Widening (3 -> 6 bits) from a PACKED sub-byte input stream at a mid-byte
 * dstStartBit into a stale all-ones buffer: the zero-extended upper half of
 * each widened value must CLEAR stale bits (header contract: stale in-range
 * bits are overwritten), and both cursors run mid-byte simultaneously.
 * Input {5,2,7,1} packed LSB-first at 3 bits -> dataIn {0xD5, 0x03}.
 * Output 4 x 6 bits at bits 5..28:
 *   byte 0: bits 0-4 stale 1s preserved; bits 5-7 = v0 low bits 1,0,1 -> 0xBF
 *   byte 1: bits 8-10 v0 fill 0s; bits 11-15 v1 = 0,1,0,0,0 -> 0x10
 *   byte 2: bit 16 v1 fill; bits 17-19 v2 = 1,1,1; bits 20-22 v2 fill;
 *           bit 23 v3.b0 = 1 -> 0x8E
 *   byte 3: bits 24-28 v3 rest+fill = 0; bits 29-31 stale 1s -> 0xE0 */
void testByteConversionAppend_WidensPackedSubByteInputAtBitOffsetClearsFill() {
    uint8_t dataIn[] = {0xD5, 0x03};
    uint8_t dataOut[4] = {0xFF, 0xFF, 0xFF, 0xFF};
    byteConversionAppend(dataIn, 3, dataOut, 6, 4, 5);
    uint8_t expectedBytes[] = {0xBF, 0x10, 0x8E, 0xE0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, 4);
}

/* Packing is pure low-bit truncation: -3 at 6 bits stores 0b111101. Sign
 * restoration on read-back is unpackSignExtend's job, not the packer's. */
void testByteConversionAppend_NegativeCodeStoresLowBits() {
    int32_t vals[1] = {-3};
    uint8_t dataOut[2] = {0, 0};
    byteConversionAppend((uint8_t *)vals, 32, dataOut, 6, 1, 5);
    uint8_t expectedBytes[] = {0xA0, 0x07};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, 2);
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

void test_calcNumberOfBytesForData_Sym_qBits5_N4() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 5, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    /* ceil(5*4 / 8) = ceil(20/8) = 3 */
    TEST_ASSERT_EQUAL_size_t(3, calcNumberOfBytesForData(&q, 4));
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
    TEST_ASSERT_EQUAL_INT32(-7, dstQC.zeroPoint);
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
    RUN_TEST(testByteConversion_ZeroValuesIsNoOp);
    RUN_TEST(testByteConversion_ZeroOutBitsWritesNothing);

    RUN_TEST(testByteConversionAppend_ContinuesMidByteAtBitOffset);
    RUN_TEST(testByteConversionAppend_MixedWidthDeltaLayout);
    RUN_TEST(testByteConversionAppend_DefinesRangePreservesOutside);
    RUN_TEST(testByteConversionAppend_OffsetZeroMatchesByteConversion);
    RUN_TEST(testByteConversionAppend_ZeroValuesIsNoOp);
    RUN_TEST(testByteConversionAppend_WidensPackedSubByteInputAtBitOffsetClearsFill);
    RUN_TEST(testByteConversionAppend_NegativeCodeStoresLowBits);

    RUN_TEST(testGetBitmask);
    RUN_TEST(testGetBitmask2);
    RUN_TEST(testWriteByte);
    RUN_TEST(testWriteByte2);
    RUN_TEST(testWriteByte_ClearsStaleInMaskBits);
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
    return UNITY_END();
}
