#include <stdlib.h>

#include "Tensor.h"
#include "TensorApi.h"
#include "DataLoaderApi.h"
#include "unity.h"
#include "RNG.h"
#include "DataLoader.h"
#include "StorageApi.h"
#include "Dataset.h"
#include "NPYLoaderApi.h"

#define PROXY_DATASET_SIZE 4
#define PROXY_FEATURES 2

static tensor_t *proxyItems[PROXY_DATASET_SIZE];
static tensor_t *proxyLabels[PROXY_DATASET_SIZE];
static tensorArray_t proxyItemsArr;
static tensorArray_t proxyLabelsArr;
static dataset_t proxyDataset;
static bool proxyInit = false;

static void initProxyDataset() {
    if (proxyInit) return;

    static float in0[] = {1.f, 2.f};
    static float in1[] = {3.f, 4.f};
    static float in2[] = {5.f, 6.f};
    static float in3[] = {7.f, 8.f};

    static float lb0[] = {0.f, 1.f};
    static float lb1[] = {1.f, 0.f};
    static float lb2[] = {0.f, 1.f};
    static float lb3[] = {1.f, 0.f};

    static size_t inDims[] = {1, PROXY_FEATURES};
    static size_t lbDims[] = {1, PROXY_FEATURES};

    proxyItems[0] = tensorInitFloat(in0, inDims, 2, NULL);
    proxyItems[1] = tensorInitFloat(in1, inDims, 2, NULL);
    proxyItems[2] = tensorInitFloat(in2, inDims, 2, NULL);
    proxyItems[3] = tensorInitFloat(in3, inDims, 2, NULL);

    proxyLabels[0] = tensorInitFloat(lb0, lbDims, 2, NULL);
    proxyLabels[1] = tensorInitFloat(lb1, lbDims, 2, NULL);
    proxyLabels[2] = tensorInitFloat(lb2, lbDims, 2, NULL);
    proxyLabels[3] = tensorInitFloat(lb3, lbDims, 2, NULL);

    proxyItemsArr.array = proxyItems;
    proxyItemsArr.size = PROXY_DATASET_SIZE;
    proxyLabelsArr.array = proxyLabels;
    proxyLabelsArr.size = PROXY_DATASET_SIZE;

    proxyDataset.items = &proxyItemsArr;
    proxyDataset.labels = &proxyLabelsArr;
    proxyInit = true;
}

static sample_t *getProxySample(size_t id) {
    sample_t *s = *reserveMemory(sizeof(sample_t));
    s->item = proxyDataset.items->array[id];
    s->label = proxyDataset.labels->array[id];
    return s;
}

static size_t getProxyDatasetSize() {
    return PROXY_DATASET_SIZE;
}

static dataset_t npyProxyDataset;
static bool npyProxyInit = false;

static void initNpyProxyDataset() {
    if (npyProxyInit) return;
    npyProxyDataset.items = npyLoad(PROXY_X_PATH);
    npyProxyDataset.labels = npyLoad(PROXY_Y_PATH);
    npyProxyInit = true;
}

static sample_t *getNpyProxySample(size_t id) {
    return npyGetSample(&npyProxyDataset, id);
}

static size_t getNpyProxyDatasetSize() {
    return npyProxyDataset.items->size;
}

void setUp() {}

void tearDown() {}

void testGetSample() {
    initProxyDataset();

    dataLoader_t *dl = dataLoaderInit(getProxySample, getProxyDatasetSize, 1,
                                      NULL, NULL, false, 0, true);

    sample_t *s = dl->getSample(0);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(proxyDataset.items->array[0]->data, s->item->data,
                                  PROXY_FEATURES);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(proxyDataset.labels->array[0]->data, s->label->data,
                                  PROXY_FEATURES);

    freeSample(s);
    freeDataLoader(dl);
}

void testGetBatch() {
    initProxyDataset();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl = dataLoaderInit(getProxySample, getProxyDatasetSize, batchSize,
                                      NULL, NULL, false, 0, true);

    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b);

        for (size_t i = 0; i < batchSize; i++) {
            size_t sampleIndex = b * batchSize + i;
            TEST_ASSERT_EQUAL_FLOAT_ARRAY(proxyDataset.items->array[sampleIndex]->data,
                                          batch->samples[i]->item->data, PROXY_FEATURES);
            TEST_ASSERT_EQUAL_FLOAT_ARRAY(proxyDataset.labels->array[sampleIndex]->data,
                                          batch->samples[i]->label->data, PROXY_FEATURES);
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    freeDataLoader(dl);
}

void testGetBatch_ShuffledMinibatchCoversAllSamples() {
    initProxyDataset();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl = dataLoaderInit(getProxySample, getProxyDatasetSize, batchSize,
                                      NULL, NULL, true, 42, true);

    // Collect first feature value from every sample across all batches
    float seen[PROXY_DATASET_SIZE];
    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b);
        for (size_t i = 0; i < batchSize; i++) {
            seen[b * batchSize + i] = ((float *)batch->samples[i]->item->data)[0];
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    // Every original sample has a unique first feature: 1, 3, 5, 7
    // Sort and verify all four are present (no duplicates, no zeros)
    for (size_t i = 0; i < PROXY_DATASET_SIZE; i++) {
        for (size_t j = i + 1; j < PROXY_DATASET_SIZE; j++) {
            if (seen[j] < seen[i]) {
                float tmp = seen[i];
                seen[i] = seen[j];
                seen[j] = tmp;
            }
        }
    }

    float expected[] = {1.f, 3.f, 5.f, 7.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, seen, PROXY_DATASET_SIZE);

    freeDataLoader(dl);
}

void testGetBatch_NpyBackedDataset() {
    initProxyDataset();
    initNpyProxyDataset();

    TEST_ASSERT_EQUAL(PROXY_DATASET_SIZE, getNpyProxyDatasetSize());

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl = dataLoaderInit(getNpyProxySample, getNpyProxyDatasetSize, batchSize,
                                      NULL, NULL, false, 0, true);

    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b);

        for (size_t i = 0; i < batchSize; i++) {
            size_t sampleIndex = b * batchSize + i;
            TEST_ASSERT_EQUAL_FLOAT_ARRAY(proxyDataset.items->array[sampleIndex]->data,
                                          batch->samples[i]->item->data, PROXY_FEATURES);
            TEST_ASSERT_EQUAL_FLOAT_ARRAY(proxyDataset.labels->array[sampleIndex]->data,
                                          batch->samples[i]->label->data, PROXY_FEATURES);
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    freeDataLoader(dl);
}

void testShuffle() {
    size_t numberOfIndices = 10;
    size_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    rngSetSeed(1);
    rngShuffleIndices(indices, 10);

    /*for (size_t i = 0; i < numberOfIndices; i++) {
        printf("%lu\n", indices[i]);
    }*/

    size_t expected[] = {3, 1, 0, 2, 6, 7, 8, 5, 4, 9};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected, indices, numberOfIndices);
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
