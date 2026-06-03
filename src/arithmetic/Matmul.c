#define SOURCE_FILE "MATMUL"

#ifdef TRACK_INSTRUCTIONS
#define MATMUL_FUNC_INT matmulIntTensorsWithInstructionCounter
#define MATMUL_FUNC_FLOAT matmulFloatTensorsWithInstructionCounter
#define MATMUL_FUNC_SYM_INT32 matmulSymIntTensorsWithInstructionCounter
#else
#define MATMUL_FUNC_INT matmulIntTensors
#define MATMUL_FUNC_FLOAT matmulFloatTensors
#define MATMUL_FUNC_SYM_INT32 matmulSymIntTensors
#endif

#include <stdio.h>
#include <stdlib.h>

#include "Arithmetic.h"
#include "Common.h"
#include "DTypes.h"
#include "Matmul.h"
#include "Mul.h"
#include "Tensor.h"

size_t matmulInstructionCounter = 0;

void matmulIntTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    if (aTensor->shape->numberOfDimensions > 2 || bTensor->shape->numberOfDimensions > 2) {
        PRINT_ERROR("Matmul only supports up to 2D Tensors");
        exit(1);
    }

    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;

    size_t bNumberOfDims = bTensor->shape->numberOfDimensions;
    size_t *bDims = bTensor->shape->dimensions;

    size_t aRows, aColumns;
    if (aNumberOfDims < 2) {
        aRows = 1;
        aColumns = getDimensionsByIndex(aTensor, 0);
    } else {
        aRows = getDimensionsByIndex(aTensor, 0);
        aColumns = getDimensionsByIndex(aTensor, 1);
    }

    size_t bRows = getDimensionsByIndex(bTensor, 0);
    size_t bColumns = 0;
    if (bNumberOfDims < 2) {
        bColumns = 1;
    } else {
        bColumns = getDimensionsByIndex(bTensor, 1);
    }

    size_t resultCounter = 0;

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %lu, bRows: %lu\n", aColumns, bRows);
        exit(1);
    }

    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {

        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            int32_t result = 0;
            for (size_t i = 0; i < aColumns; i++) {
                size_t aByteIndex = 0;
                if (aNumberOfDims == 1) {
                    aByteIndex = i * sizeof(int32_t);
                } else {
                    size_t aIndices[] = {rowIndex, i};
                    size_t aValueIndex = calcElementIndexByIndices(
                        aNumberOfDims, aDims, aIndices, aTensor->shape->orderOfDimensions);
                    aByteIndex = aValueIndex * sizeof(int32_t);
                }

                int32_t aValue = readBytesAsInt32(&aTensor->data[aByteIndex]);

                size_t bByteIndex = 0;
                if (bNumberOfDims == 1) {
                    bByteIndex = i * sizeof(int32_t);
                } else {
                    size_t bIndices[] = {i, columnIndex};
                    size_t bValueIndex = calcElementIndexByIndices(
                        bNumberOfDims, bDims, bIndices, bTensor->shape->orderOfDimensions);
                    bByteIndex = bValueIndex * sizeof(int32_t);
                }

                int32_t bValue = readBytesAsInt32(&bTensor->data[bByteIndex]);

                result += mulInt32s(aValue, bValue);
            }

            size_t outputByteIndex = resultCounter * sizeof(int32_t);

            writeInt32ToByteArray(result, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
}

void matmulIntTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                            tensor_t *outputTensor) {
    if (aTensor->shape->numberOfDimensions > 2 || bTensor->shape->numberOfDimensions > 2) {
        PRINT_ERROR("Matmul only supports up to 2D Tensors");
        exit(1);
    }

    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;

    size_t bNumberOfDims = bTensor->shape->numberOfDimensions;
    size_t *bDims = bTensor->shape->dimensions;

    size_t aRows, aColumns;
    if (aNumberOfDims < 2) {
        aRows = 1;
        aColumns = getDimensionsByIndex(aTensor, 0);
    } else {
        aRows = getDimensionsByIndex(aTensor, 0);
        aColumns = getDimensionsByIndex(aTensor, 1);
    }

    size_t bRows = getDimensionsByIndex(bTensor, 0);
    size_t bColumns = 0;
    if (bNumberOfDims < 2) {
        bColumns = 1;
    } else {
        bColumns = getDimensionsByIndex(bTensor, 1);
    }

    size_t resultCounter = 0;

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %lu, bRows: %lu\n", aColumns, bRows);
        exit(1);
    }

    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {

        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            int32_t result = 0;
            for (size_t i = 0; i < aColumns; i++) {
                size_t aByteIndex = 0;
                if (aNumberOfDims == 1) {
                    aByteIndex = i * sizeof(int32_t);
                } else {
                    size_t aIndices[] = {rowIndex, i};
                    size_t aValueIndex = calcElementIndexByIndices(
                        aNumberOfDims, aDims, aIndices, aTensor->shape->orderOfDimensions);
                    aByteIndex = aValueIndex * sizeof(int32_t);
                }

                int32_t aValue = readBytesAsInt32(&aTensor->data[aByteIndex]);

                size_t bByteIndex = 0;
                if (bNumberOfDims == 1) {
                    bByteIndex = i * sizeof(int32_t);
                } else {
                    size_t bIndices[] = {i, columnIndex};
                    size_t bValueIndex = calcElementIndexByIndices(
                        bNumberOfDims, bDims, bIndices, bTensor->shape->orderOfDimensions);
                    bByteIndex = bValueIndex * sizeof(int32_t);
                }

                int32_t bValue = readBytesAsInt32(&bTensor->data[bByteIndex]);

                result += mulInt32s(aValue, bValue);
            }

            size_t outputByteIndex = resultCounter * sizeof(int32_t);

            writeInt32ToByteArray(result, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
    ++matmulInstructionCounter;
}

void matmulInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    MATMUL_FUNC_INT(aTensor, bTensor, outputTensor);
}

static void matmulFloatCore(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                            const uint8_t *biasSeed) {
    if (aTensor->shape->numberOfDimensions > 2 || bTensor->shape->numberOfDimensions > 2) {
        PRINT_ERROR("Matmul only supports up to 2D Tensors");
        exit(1);
    }

    size_t aNumberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;
    size_t bNumberOfDims = bTensor->shape->numberOfDimensions;
    size_t *bDims = bTensor->shape->dimensions;

    size_t aRows, aColumns = 0;
    if (aNumberOfDims < 2) {
        aRows = 1;
        aColumns = getDimensionsByIndex(aTensor, 0);
    } else {
        aRows = getDimensionsByIndex(aTensor, 0);
        aColumns = getDimensionsByIndex(aTensor, 1);
    }

    size_t bRows, bColumns = 0;
    if (bNumberOfDims < 2) {
        bRows = getDimensionsByIndex(bTensor, 0);
        bColumns = 1;
    } else {
        bRows = getDimensionsByIndex(bTensor, 0);
        bColumns = getDimensionsByIndex(bTensor, 1);
    }

    size_t resultCounter = 0;

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %lu, bRows: %lu\n", aColumns, bRows);
        exit(1);
    }

    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            float result = biasSeed
                               ? readBytesAsFloat((uint8_t *)&biasSeed[columnIndex * sizeof(float)])
                               : 0.0f;
            for (size_t i = 0; i < aColumns; i++) {
                size_t aByteIndex = 0;
                if (aNumberOfDims == 1) {
                    aByteIndex = i * sizeof(float);
                } else {
                    size_t aIndices[] = {rowIndex, i};
                    size_t aValueIndex = calcElementIndexByIndices(
                        aNumberOfDims, aDims, aIndices, aTensor->shape->orderOfDimensions);
                    aByteIndex = aValueIndex * sizeof(float);
                }
                float aValue = readBytesAsFloat(&aTensor->data[aByteIndex]);

                size_t bByteIndex = 0;
                if (bNumberOfDims == 1) {
                    bByteIndex = i * sizeof(float);
                } else {
                    size_t bIndices[] = {i, columnIndex};
                    size_t bValueIndex = calcElementIndexByIndices(
                        bNumberOfDims, bDims, bIndices, bTensor->shape->orderOfDimensions);
                    bByteIndex = bValueIndex * sizeof(float);
                }
                float bValue = readBytesAsFloat(&bTensor->data[bByteIndex]);
                result += mulFloat32s(aValue, bValue);
            }

            size_t outputByteIndex = resultCounter * sizeof(float);
            writeFloatToByteArray(result, &outputTensor->data[outputByteIndex]);
            resultCounter++;
        }
    }
}

void matmulFloatTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulFloatCore(aTensor, bTensor, outputTensor, NULL);
}

void matmulFloatTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                              tensor_t *outputTensor) {
    matmulFloatCore(aTensor, bTensor, outputTensor, NULL);
    ++matmulInstructionCounter;
}

void matmulFloat32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                  tensor_t *bias) {
    const uint8_t *seed = NULL;
    if (bias != NULL) {
        size_t bColumns =
            (bTensor->shape->numberOfDimensions < 2) ? 1 : getDimensionsByIndex(bTensor, 1);
        if (bias->shape->numberOfDimensions != 1) {
            PRINT_ERROR("matmulFloat32TensorsWithBias: bias must be rank-1");
            exit(1);
        }
        if (getDimensionsByIndex(bias, 0) != bColumns) {
            PRINT_ERROR("matmulFloat32TensorsWithBias: bias length != output columns");
            exit(1);
        }
        seed = bias->data;
    }
    matmulFloatCore(aTensor, bTensor, outputTensor, seed);
}

void matmulFloat32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    MATMUL_FUNC_FLOAT(aTensor, bTensor, outputTensor);
}

void matmulSymIntTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulInt32Tensors(aTensor, bTensor, outputTensor);

    symInt32QConfig_t *aSymInt32QC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bSymInt32QC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;

    outputSymInt32QC->scale = aSymInt32QC->scale * bSymInt32QC->scale;
}

void matmulSymIntTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                               tensor_t *outputTensor) {
    matmulInt32Tensors(aTensor, bTensor, outputTensor);

    symInt32QConfig_t *aSymInt32QC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bSymInt32QC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    outputSymInt32QC->scale = aSymInt32QC->scale * bSymInt32QC->scale;

    ++matmulInstructionCounter;
}

void matmulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    MATMUL_FUNC_SYM_INT32(aTensor, bTensor, outputTensor);
}

size_t getMatmulInstructionCounter() {
    return matmulInstructionCounter;
}
