#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>

#include "Tensor.h"

typedef struct tensorArray
{
    tensor_t** array;
    size_t size;
} tensorArray_t;

typedef struct dataset
{
    tensorArray_t* items;
    tensorArray_t* labels;
} dataset_t;

#endif //DATASET_H
