#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdbool.h>
#include <stdio.h>

#include "Tensor.h"
#include "Dataset.h"


typedef enum
{
    FLOAT_32,
    INT_32
} dtype_t;

typedef struct dataLoader dataLoader_t;

typedef struct sample
{
    tensor_t* item;
    tensor_t* label;
} sample_t;

typedef struct batch {
    sample_t **samples;
    size_t size;
} batch_t;

typedef tensor_t* (*transformFn_t)(tensor_t* tensor);
typedef sample_t* (*getSampleFn_t)(size_t id);
typedef size_t (*getDatasetSizeFn_t)();
typedef batch_t* (*getBatchFn_t)(dataLoader_t *dataLoader, size_t index);

struct dataLoader
{
    getSampleFn_t getSample;
    getDatasetSizeFn_t getDatasetSize;

    uint16_t batchSize;
    getBatchFn_t getBatch;

    transformFn_t transform;
    transformFn_t targetTransform;

    bool shuffle;
    uint64_t shuffleSeed;
    size_t *indices;

    bool dropLast;
};

void initDataLoader(dataLoader_t *dataLoader, getSampleFn_t getSample,
                    getDatasetSizeFn_t getDatasetSize, getBatchFn_t getBatch, uint16_t batchSize,
                    transformFn_t transform, transformFn_t targetTransform, bool shuffle,
                    uint64_t shuffleSeed, size_t *indices, bool dropLast);

#endif //DATALOADER_H
