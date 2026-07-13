#include <stdint.h>
#include <stdio.h>

#include "DeathTest.h"
#include "Deserialize.h"
#include "Flatten.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "Serialize.h"
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

/* Hand-crafted malformed files exercising deserializeModel's NEW validation
 * (bad magic / wrong version / layerCount mismatch / tag mismatch). A single
 * Flatten layer is the minimal pre-built mirror model — Flatten needs no
 * quantization setup, so these tests isolate the header/tag validation from
 * any per-layer-type record decoding. */

static void testDeserializeRejectsBadMagic(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("XXXX", 1, 4, f);
    uint32_t version = 1;
    fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t layerCount = 1;
    fwrite(&layerCount, sizeof(uint32_t), 1, f);
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsWrongVersion(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    uint32_t version = 2;
    fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t layerCount = 1;
    fwrite(&layerCount, sizeof(uint32_t), 1, f);
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsLayerCountMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    uint32_t version = 1;
    fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t layerCount = 2; /* caller below passes sizeModel = 1 */
    fwrite(&layerCount, sizeof(uint32_t), 1, f);
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsTagMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    uint32_t version = 1;
    fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t layerCount = 1;
    fwrite(&layerCount, sizeof(uint32_t), 1, f);
    uint8_t tag = (uint8_t)LINEAR; /* pre-built mirror layer below is FLATTEN */
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static tensor_t *makeSymInt32Tensor2D(size_t d0, size_t d1) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
}

/* #316: a checkpoint whose per-tensor dtype differs from the pre-built skeleton
 * must be rejected before deserializeQConfig writes qConfig fields through a
 * mismatched (or NULL, for FLOAT32/INT32/BOOL) pointer, and before the payload
 * fread can overflow the skeleton's allocation. Here a FLOAT32 record is loaded
 * into a SYM_INT32-built skeleton — pre-fix it silently overwrites the dtype;
 * the reverse (SYM into a FLOAT32 skeleton) NULL-derefs. */
void testDeserializeTensorRejectsDtypeMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *floatTensor = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(floatTensor, f);
    fclose(f);

    tensor_t *symSkeleton = makeSymInt32Tensor2D(2, 3);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(symSkeleton, f));
    fclose(f);

    freeTensor(symSkeleton);
    freeTensor(floatTensor);
}

/* #316: a same-dtype record whose element count differs from the skeleton would
 * fread past tensor->data, which initTensor sized from the build-time shape. The
 * payload-size check catches size changes (shape- or packed-qBits-driven) that
 * the dtype check alone misses. */
void testDeserializeTensorRejectsPayloadSizeMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *bigTensor = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(bigTensor, f);
    fclose(f);

    /* Same dtype (FLOAT32) and rank, but a smaller allocation (2x2 = 4 elems). */
    tensor_t *smallSkeleton = makeFloatTensor2D(2, 2, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(smallSkeleton, f));
    fclose(f);

    freeTensor(smallSkeleton);
    freeTensor(bigTensor);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSerializeAndDeserializeTensor);
    RUN_TEST(testDeserializeRejectsBadMagic);
    RUN_TEST(testDeserializeRejectsWrongVersion);
    RUN_TEST(testDeserializeRejectsLayerCountMismatch);
    RUN_TEST(testDeserializeRejectsTagMismatch);
    RUN_TEST(testDeserializeTensorRejectsDtypeMismatch);
    RUN_TEST(testDeserializeTensorRejectsPayloadSizeMismatch);
    return UNITY_END();
}
