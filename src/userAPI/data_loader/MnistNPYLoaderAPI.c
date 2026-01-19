#define SOURCE_FILE "NPY_LOADER_API"

#include <stdio.h>
#include <stdlib.h>

#include "DataLoader.h"
#include "TensorAPI.h"
#include "Tensor.h"
#include "Common.h"
#include "StorageAPI.h"

#include "Dataset.h"
#include "MnistNPYLoader.h"
#include "DatasetAPI.h"

static void getRowShape(shape_t *totalShape, shape_t *rowShape) {
    size_t rowNumberOfDims = totalShape->numberOfDimensions - 1;

    size_t *rowDims = *reserveMemory(rowNumberOfDims * sizeof(size_t));
    size_t *rowOrderOfDims = *reserveMemory(rowNumberOfDims * sizeof(size_t));

    for (size_t i = 0; i < rowNumberOfDims; i++) {
        rowDims[i] = totalShape->dimensions[i + 1];
        rowOrderOfDims[i] = i;
    }

    rowShape->dimensions = rowDims;
    rowShape->orderOfDimensions = rowOrderOfDims;
    rowShape->numberOfDimensions = rowNumberOfDims;
}

tensorArray_t *npyLoad(char *path) {
    FILE *f = openFile(path);

    checkMagic(f);

    uint32_t headerSize = readHeaderSize(f);
    char *header = *reserveMemory(headerSize);
    readHeader(header, headerSize, f);

    dtype_t dtype = getDTypeFromHeader(header);
    quantization_t *q = initQByDType(dtype);

    size_t numberOfDims = getNumberOfDimsFromHeader(header);
    shape_t *totalShape = *reserveMemory(sizeof(shape_t));
    size_t *dims = *reserveMemory(numberOfDims * sizeof(size_t));
    size_t *orderOfDims = *reserveMemory(numberOfDims * sizeof(size_t));
    getShapeFromHeader(totalShape, dims, orderOfDims, header, numberOfDims);

    size_t numberOfRows = dims[0];

    shape_t *rowShape = *reserveMemory(sizeof(shape_t));
    getRowShape(totalShape, rowShape);

    size_t bytesPerValue = calcBytesPerElement(q);
    size_t numberOfValuesInRow = calcNumberOfElementsByShape(rowShape);

    tensorArray_t *tensorArr = *reserveMemory(sizeof(tensorArray_t));
    tensor_t **arr = *reserveMemory(numberOfRows * sizeof(tensor_t *));
    tensorArr->array = arr;
    tensorArr->size = numberOfRows;

    for (size_t i = 0; i < numberOfRows; i++) {
        float *data = *reserveMemory(numberOfValuesInRow * bytesPerValue);

        size_t n = fread(data, bytesPerValue, numberOfValuesInRow, f);
        tensorArr->array[i] = tensorInit(data, rowShape->dimensions, rowShape->numberOfDimensions,
                                         q, NULL);

        if (n != numberOfValuesInRow) {
            PRINT_ERROR("fread did not read the correct number of bytes!");
            exit(1);
        }
    }

    fclose(f);

    return tensorArr;
}

sample_t *npyGetSample(dataset_t *dataset, size_t id) {
    sample_t *sample = *reserveMemory(sizeof(sample_t));
    sample->item = dataset->items->array[id];
    sample->label = dataset->labels->array[id];

    return sample;
}

// TODO free functions