#define SOURCE_FILE "DATASET_API"

#include <stdlib.h>

#include "QuantizationAPI.h"
#include "Quantization.h"
#include "DataLoader.h"
#include "Common.h"
#include "DatasetAPI.h"


quantization_t *initQByDType(dtype_t dtype) {
    switch (dtype) {
    case FLOAT_32:
        quantization_t *floatQ = quantizationInitFloat();
        return floatQ;
    case INT_32:
        quantization_t *intQ = quantizationInitInt32();
        return intQ;
    default:
        PRINT_ERROR("Unsupported DType!");
        exit(1);
    }
}

