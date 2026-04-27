#include <stdbool.h>
#include <stdlib.h>

#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "NPYLoaderApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#define PROXY_DATASET_SIZE 4
#define PROXY_FEATURES 2

/* Per-test fixture handle. Each test calls makeProxyDataset() at the start
 * and freeProxyDataset() once it's captured every assertion value, in line
 * with docs/CONVENTIONS.md "Test memory discipline" (heap-tier idiom). */
typedef struct proxyFixture {
    tensor_t *items[PROXY_DATASET_SIZE];
    tensor_t *labels[PROXY_DATASET_SIZE];
    tensorArray_t itemsArr;
    tensorArray_t labelsArr;
    dataset_t dataset;
} proxyFixture_t;

static proxyFixture_t g_proxy;

static tensor_t *makeProxyTensor(const float *src, size_t count) {
    /* Heap-tier construction per CONVENTIONS Rule 1: shape via reserveMemory,
     * data populated by tensorFillFromFloatBuffer (caller-buffer-safe). */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = PROXY_FEATURES;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, src, count);
    return t;
}

static void makeProxyDataset(void) {
    static const float in0[] = {1.f, 2.f};
    static const float in1[] = {3.f, 4.f};
    static const float in2[] = {5.f, 6.f};
    static const float in3[] = {7.f, 8.f};
    static const float lb0[] = {0.f, 1.f};
    static const float lb1[] = {1.f, 0.f};
    static const float lb2[] = {0.f, 1.f};
    static const float lb3[] = {1.f, 0.f};

    g_proxy.items[0] = makeProxyTensor(in0, PROXY_FEATURES);
    g_proxy.items[1] = makeProxyTensor(in1, PROXY_FEATURES);
    g_proxy.items[2] = makeProxyTensor(in2, PROXY_FEATURES);
    g_proxy.items[3] = makeProxyTensor(in3, PROXY_FEATURES);

    g_proxy.labels[0] = makeProxyTensor(lb0, PROXY_FEATURES);
    g_proxy.labels[1] = makeProxyTensor(lb1, PROXY_FEATURES);
    g_proxy.labels[2] = makeProxyTensor(lb2, PROXY_FEATURES);
    g_proxy.labels[3] = makeProxyTensor(lb3, PROXY_FEATURES);

    g_proxy.itemsArr.array = g_proxy.items;
    g_proxy.itemsArr.size = PROXY_DATASET_SIZE;
    g_proxy.labelsArr.array = g_proxy.labels;
    g_proxy.labelsArr.size = PROXY_DATASET_SIZE;

    g_proxy.dataset.items = &g_proxy.itemsArr;
    g_proxy.dataset.labels = &g_proxy.labelsArr;
}

static void freeProxyDataset(void) {
    /* Release each heap proxy tensor; the tensorArray_t and dataset_t
     * containers themselves are stack/static (members of g_proxy) and
     * need no free. */
    for (size_t i = 0; i < PROXY_DATASET_SIZE; i++) {
        freeTensor(g_proxy.items[i]);
        freeTensor(g_proxy.labels[i]);
    }
}

static sample_t *getProxySample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = g_proxy.dataset.items->array[id];
    s->label = g_proxy.dataset.labels->array[id];
    return s;
}

static size_t getProxyDatasetSize(void) {
    return PROXY_DATASET_SIZE;
}

/* NPY-backed dataset. Each test that uses it loads at start and frees with
 * freeTensorArray (added in this batch) before exiting. */
static dataset_t g_npyDataset;

static void makeNpyDataset(void) {
    g_npyDataset.items = npyLoad(PROXY_X_PATH);
    g_npyDataset.labels = npyLoad(PROXY_Y_PATH);
}

static void freeNpyDataset(void) {
    freeTensorArray(g_npyDataset.items);
    freeTensorArray(g_npyDataset.labels);
}

static sample_t *getNpyProxySample(size_t id) {
    return npyGetSample(&g_npyDataset, id);
}

static size_t getNpyProxyDatasetSize(void) {
    return g_npyDataset.items->size;
}

void setUp() {}

void tearDown() {}

void testGetSample() {
    makeProxyDataset();

    dataLoader_t *dl =
        dataLoaderInit(getProxySample, getProxyDatasetSize, 1, NULL, NULL, false, 0, true);

    sample_t *s = dl->getSample(0);

    /* CAPTURE: copy data referenced by sample so we can assert after free. */
    float capturedItem[PROXY_FEATURES];
    float capturedLabel[PROXY_FEATURES];
    for (size_t i = 0; i < PROXY_FEATURES; i++) {
        capturedItem[i] = ((float *)s->item->data)[i];
        capturedLabel[i] = ((float *)s->label->data)[i];
    }
    float expectedItem[PROXY_FEATURES];
    float expectedLabel[PROXY_FEATURES];
    for (size_t i = 0; i < PROXY_FEATURES; i++) {
        expectedItem[i] = ((float *)g_proxy.dataset.items->array[0]->data)[i];
        expectedLabel[i] = ((float *)g_proxy.dataset.labels->array[0]->data)[i];
    }

    /* FREE in reverse-init order. */
    freeSample(s);
    freeDataLoader(dl);
    freeProxyDataset();

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedItem, capturedItem, PROXY_FEATURES);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedLabel, capturedLabel, PROXY_FEATURES);
}

void testGetBatch() {
    makeProxyDataset();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl =
        dataLoaderInit(getProxySample, getProxyDatasetSize, batchSize, NULL, NULL, false, 0, true);

    /* CAPTURE: walk every batch, copy each sample's data side-by-side with
     * the expected dataset values. Frees happen along the way (samples,
     * batch struct) but assertions wait until everything is freed. */
    float capturedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float capturedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];

    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b);

        for (size_t i = 0; i < batchSize; i++) {
            size_t k = b * batchSize + i;
            for (size_t f = 0; f < PROXY_FEATURES; f++) {
                capturedItems[k][f] = ((float *)batch->samples[i]->item->data)[f];
                capturedLabels[k][f] = ((float *)batch->samples[i]->label->data)[f];
                expectedItems[k][f] = ((float *)g_proxy.dataset.items->array[k]->data)[f];
                expectedLabels[k][f] = ((float *)g_proxy.dataset.labels->array[k]->data)[f];
            }
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    freeDataLoader(dl);
    freeProxyDataset();

    /* ASSERT on captured. */
    for (size_t k = 0; k < PROXY_DATASET_SIZE; k++) {
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedItems[k], capturedItems[k], PROXY_FEATURES);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedLabels[k], capturedLabels[k], PROXY_FEATURES);
    }
}

void testGetBatch_ShuffledMinibatchCoversAllSamples() {
    makeProxyDataset();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl =
        dataLoaderInit(getProxySample, getProxyDatasetSize, batchSize, NULL, NULL, true, 42, true);

    /* CAPTURE: collect first feature value from every sample across all
     * batches; assertions wait until everything is freed. */
    float seen[PROXY_DATASET_SIZE];
    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b);
        for (size_t i = 0; i < batchSize; i++) {
            seen[b * batchSize + i] = ((float *)batch->samples[i]->item->data)[0];
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    /* Sort the captured values so they can be compared against {1,3,5,7}. */
    for (size_t i = 0; i < PROXY_DATASET_SIZE; i++) {
        for (size_t j = i + 1; j < PROXY_DATASET_SIZE; j++) {
            if (seen[j] < seen[i]) {
                float tmp = seen[i];
                seen[i] = seen[j];
                seen[j] = tmp;
            }
        }
    }

    freeDataLoader(dl);
    freeProxyDataset();

    /* ASSERT on captured. */
    float expected[] = {1.f, 3.f, 5.f, 7.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, seen, PROXY_DATASET_SIZE);
}

void testGetBatch_NpyBackedDataset() {
    makeProxyDataset();
    makeNpyDataset();

    size_t capturedNpySize = getNpyProxyDatasetSize();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl = dataLoaderInit(getNpyProxySample, getNpyProxyDatasetSize, batchSize, NULL,
                                      NULL, false, 0, true);

    float capturedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float capturedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];

    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b);

        for (size_t i = 0; i < batchSize; i++) {
            size_t k = b * batchSize + i;
            for (size_t f = 0; f < PROXY_FEATURES; f++) {
                capturedItems[k][f] = ((float *)batch->samples[i]->item->data)[f];
                capturedLabels[k][f] = ((float *)batch->samples[i]->label->data)[f];
                expectedItems[k][f] = ((float *)g_proxy.dataset.items->array[k]->data)[f];
                expectedLabels[k][f] = ((float *)g_proxy.dataset.labels->array[k]->data)[f];
            }
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    freeDataLoader(dl);
    freeNpyDataset();
    freeProxyDataset();

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL((size_t)PROXY_DATASET_SIZE, capturedNpySize);
    for (size_t k = 0; k < PROXY_DATASET_SIZE; k++) {
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedItems[k], capturedItems[k], PROXY_FEATURES);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedLabels[k], capturedLabels[k], PROXY_FEATURES);
    }
}

void testShuffle() {
    /* Stack-only test: stays in the stack-only tier per CONVENTIONS Rule 1.
     * No heap fixtures, no frees needed. Asserts directly. */
    size_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    rngSetSeed(1);
    rngShuffleIndices(indices, 10);

    size_t expected[] = {3, 1, 0, 2, 6, 7, 8, 5, 4, 9};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected, indices, 10);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testGetSample);
    RUN_TEST(testGetBatch);
    RUN_TEST(testGetBatch_ShuffledMinibatchCoversAllSamples);
    RUN_TEST(testGetBatch_NpyBackedDataset);
    RUN_TEST(testShuffle);
    return UNITY_END();
}
