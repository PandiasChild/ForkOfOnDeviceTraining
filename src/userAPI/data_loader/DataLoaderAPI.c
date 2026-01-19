#define SOURCE_FILE "DATA_LOADER_API"

#include <stdlib.h>

#include "DataLoader.h"
#include "Common.h"
#include "StorageAPI.h"

sample_t *getSampleByIndex(dataLoader_t *dataLoader, size_t index) {
    size_t shuffledIndex = dataLoader->indices[index];
    return dataLoader->getSample(shuffledIndex);
}

batch_t *getBatch(dataLoader_t *dataLoader, size_t index) {
    batch_t *batch = *reserveMemory(sizeof(batch_t));

    size_t batchSize = dataLoader->batchSize;
    batch->size = batchSize;
    batch->samples = *reserveMemory(batchSize * sizeof(sample_t));

    size_t sampleIndex = index * batchSize;

    for (size_t i = sampleIndex; i < sampleIndex + batchSize; i++) {
        batch->samples[i] = getSampleByIndex(dataLoader, i);
    }

    return batch;
}

dataLoader_t *dataLoaderInit(getSampleFn_t getSample, getDatasetSizeFn_t getDatasetSize,
                             uint16_t batchSize, transformFn_t transform,
                             transformFn_t targetTransform, bool shuffle, uint64_t shuffleSeed,
                             bool dropLast) {

    if (dropLast == false) {
        PRINT_ERROR("dropLast == false is not supported!");
        exit(1);
    }

    dataLoader_t *dataLoader = *reserveMemory(sizeof(dataLoader_t));

    size_t numberOfIndices = getDatasetSize();
    size_t *indices = *reserveMemory(numberOfIndices * sizeof(size_t));

    initDataLoader(dataLoader, getSample, getDatasetSize, getBatch, batchSize, transform,
               targetTransform, shuffle,
               shuffleSeed, indices, dropLast);

    return dataLoader;
}


// TODO free functions
