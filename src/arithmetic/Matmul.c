#define SOURCE_FILE "MATMUL"
#include <math.h>

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
#include "Rounding.h"
#include "Tensor.h"

size_t matmulInstructionCounter = 0;

static void matmulIntCore(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                          const int32_t *biasSeed) {
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
    size_t bColumns = (bNumberOfDims < 2) ? 1 : getDimensionsByIndex(bTensor, 1);

    size_t resultCounter = 0;

    if (aColumns != bRows) {
        PRINT_ERROR("Rows dont match Columns");
        PRINT_DEBUG("aColumns: %lu, bRows: %lu\n", aColumns, bRows);
        exit(1);
    }

    for (size_t rowIndex = 0; rowIndex < aRows; rowIndex++) {
        for (size_t columnIndex = 0; columnIndex < bColumns; columnIndex++) {
            int32_t result = biasSeed ? biasSeed[columnIndex] : 0;
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

void matmulIntTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulIntCore(aTensor, bTensor, outputTensor, NULL);
}

void matmulIntTensorsWithInstructionCounter(tensor_t *aTensor, tensor_t *bTensor,
                                            tensor_t *outputTensor) {
    matmulIntCore(aTensor, bTensor, outputTensor, NULL);
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
        if (calcNumberOfElementsByTensor(bias) != bColumns) {
            PRINT_ERROR("matmulFloat32TensorsWithBias: bias element count != output columns");
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

static void matmulValidateSymOperand(tensor_t *t, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("matmul SYM_INT32: %s must be SYM_INT32", what);
        exit(1);
    }
    symInt32QConfig_t *qc = t->quantization->qConfig;
    if (qc->qMaxBits > ODT_SYM_OPERAND_QMAXBITS) {
        PRINT_ERROR("matmul SYM_INT32: %s qMaxBits (%u) exceeds operand contract (%u) — int32 "
                    "product accumulation would overflow (#227)",
                    what, (unsigned)qc->qMaxBits, (unsigned)ODT_SYM_OPERAND_QMAXBITS);
        exit(1);
    }
}

void matmulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    matmulValidateSymOperand(aTensor, "aTensor");
    matmulValidateSymOperand(bTensor, "bTensor");
    MATMUL_FUNC_SYM_INT32(aTensor, bTensor, outputTensor);
}

void matmulSymInt32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                   tensor_t *bias) {
    matmulValidateSymOperand(aTensor, "aTensor");
    matmulValidateSymOperand(bTensor, "bTensor");
    if (bias == NULL) {
        matmulIntCore(aTensor, bTensor, outputTensor, NULL);
    } else {
        /* Bias is a value-sum seed (not a product operand), so it is exempt from
         * the int12 operand bound but must still be SYM_INT32: the branch below
         * reads its data as int32 and its qConfig as symInt32QConfig_t (#247). */
        if (bias->quantization->type != SYM_INT32) {
            PRINT_ERROR("matmul SYM_INT32: bias must be SYM_INT32");
            exit(1);
        }
        size_t bColumns =
            (bTensor->shape->numberOfDimensions < 2) ? 1 : getDimensionsByIndex(bTensor, 1);
        if (calcNumberOfElementsByTensor(bias) != bColumns) {
            PRINT_ERROR("matmulSymInt32TensorsWithBias: bias element count != output columns");
            exit(1);
        }

        symInt32QConfig_t *biasQC = (symInt32QConfig_t *)bias->quantization->qConfig;
        float aScale = ((symInt32QConfig_t *)aTensor->quantization->qConfig)->scale;
        float bScale = ((symInt32QConfig_t *)bTensor->quantization->qConfig)->scale;
        float biasScale = biasQC->scale;
        float outputScale = aScale * bScale;
        if (!isfinite(outputScale)) {
            PRINT_ERROR("matmulSymInt32TensorsWithBias: outputScale non-finite (aScale=%f, bScale=%f)",
                        aScale, bScale);
            exit(1);
        }

        /* Rescale the bias into the accumulator's scale via the shared #189 helper
         * (guarded float->int32 cast): one fixed-point op per output column. */
        int32_t seed[bColumns];
        for (size_t c = 0; c < bColumns; c++) {
            int32_t biasIntC = readBytesAsInt32(&bias->data[c * sizeof(int32_t)]);
            seed[c] =
                rescaleIntoAccumulatorScale(biasIntC, biasScale, outputScale, biasQC->roundingMode);
        }
        matmulIntCore(aTensor, bTensor, outputTensor, seed);
    }

    symInt32QConfig_t *aQC = aTensor->quantization->qConfig;
    symInt32QConfig_t *bQC = bTensor->quantization->qConfig;
    symInt32QConfig_t *outputQC = outputTensor->quantization->qConfig;
    outputQC->scale = aQC->scale * bQC->scale;
}

size_t getMatmulInstructionCounter() {
    return matmulInstructionCounter;
}
