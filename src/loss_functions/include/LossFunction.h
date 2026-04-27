#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "Tensor.h"

typedef enum lossFuncType { MSE, CROSS_ENTROPY } lossFuncType_t;

typedef enum reduction { REDUCTION_SUM, REDUCTION_MEAN } reduction_t;

typedef struct lossConfig {
    lossFuncType_t funcType;
    reduction_t reduction;
} lossConfig_t;

typedef float (*lossFwdFn_t)(tensor_t *modelOutput, tensor_t *label);
typedef void (*lossBwdFn_t)(tensor_t *modelOutput, tensor_t *label, tensor_t *result,
                            size_t batchSize, reduction_t reduction);

typedef struct lossFunctions {
    lossFwdFn_t forward;
    lossBwdFn_t backward;
} lossFunctions_t;

extern lossFunctions_t lossFunctions[];

#endif // LOSSFUNCTION_H
