#ifndef DATALOADERAPI_H
#define DATALOADERAPI_H

#include "DataLoader.h"

dataLoader_t *dataLoaderInit(getSampleFn_t getSample, getDatasetSizeFn_t getDatasetSize, uint16_t batchSize, transformFn_t transform,
                             transformFn_t targetTransform, bool shuffle, uint64_t shuffleSeed,
                             bool dropLast);

#endif //DATALOADERAPI_H
