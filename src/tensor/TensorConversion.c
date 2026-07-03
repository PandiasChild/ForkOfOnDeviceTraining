#define SOURCE_FILE "TENSOR_CONVERSION"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "DTypes.h"
#include "MinMax.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "math.h"

static void packFitGuarded(const int32_t *src, size_t n, uint8_t *dst, size_t dstBits,
                           const char *what);
static void packFloatBufferAsSym(const float *values, size_t n, symQConfig_t *outQC, uint8_t *dst,
                                 const char *what);
static void quantizeFloatToAsym(const float *values, size_t n, asymQConfig_t *outQC, uint8_t *dst);

void zeroTensorData(tensor_t *tensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(tensor);
    memset(tensor->data, 0, calcNumberOfBytesForData(tensor->quantization, numberOfElements));
}

void convertInt32TensorToFloatTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    int32_t inputData[numberOfElements];
    float outputData[numberOfElements];
    readBytesAsInt32Array(numberOfElements, inputTensor->data, inputData);
    zeroTensorData(outputTensor);
    for (size_t i = 0; i < numberOfElements; i++) {
        outputData[i] = (float)inputData[i];
    }
    writeFloatArrayToByteArray(numberOfElements, outputData, outputTensor->data);
}

void convertInt32TensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    symInt32QConfig_t *outputSymInt32QConfig = outputTensor->quantization->qConfig;
    outputSymInt32QConfig->scale = 1;

    memcpy(outputTensor->data, inputTensor->data, numberOfElements * sizeof(int32_t));
}

/* Standard affine asymmetric quantization (#243). scale = (max-min)/(2^qBits-1),
 * zeroPoint = round(min/scale), code = clamp(round(v/scale - zeroPoint), 0, 2^qBits-1).
 * Dequant (elsewhere) is (code + zeroPoint)*scale. Constant tensor (min==max) uses a
 * nonzero scale to avoid divide-by-zero. The single source of truth for all four
 * *ToAsymTensor converters. */
static void quantizeFloatToAsym(const float *values, size_t n, asymQConfig_t *outQC, uint8_t *dst) {
    float mn = findMinFloat((uint8_t *)values, n);
    float mx = findMaxFloat((uint8_t *)values, n);
    const float qMax = powf(2, (float)outQC->qBits) - 1;
    float scale;
    if (mn == mx) {
        scale = (mn != 0.f) ? fabsf(mn) : 1.f;
    } else {
        scale = (mx - mn) / qMax;
    }
    int16_t zeroPoint = (int16_t)roundByMode(mn / scale, outQC->roundingMode);
    int32_t codes[n];
    for (size_t i = 0; i < n; i++) {
        codes[i] = roundByMode(clamp(values[i] / scale - (float)zeroPoint, 0.f, qMax),
                               outQC->roundingMode);
    }
    outQC->scale = scale;
    outQC->zeroPoint = zeroPoint;
    byteConversion((uint8_t *)codes, 32, dst, outQC->qBits, n);
}

void convertInt32TensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    int32_t inputAsInt32[numberOfElements];
    readBytesAsInt32Array(numberOfElements, inputTensor->data, inputAsInt32);
    float vals[numberOfElements];
    for (size_t i = 0; i < numberOfElements; i++) {
        vals[i] = (float)inputAsInt32[i];
    }
    asymQConfig_t *asymQConfig = outputTensor->quantization->qConfig;
    quantizeFloatToAsym(vals, numberOfElements, asymQConfig, outputTensor->data);
}

void convertFloatTensorToInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    float inputData[numberOfElements];
    int32_t outputData[numberOfElements];
    readBytesAsFloatArray(numberOfElements, inputTensor->data, inputData);
    zeroTensorData(outputTensor);
    for (size_t i = 0; i < numberOfElements; i++) {
        outputData[i] = (int32_t)inputData[i];
    }
    writeInt32ArrayToByteArray(numberOfElements, outputData, outputTensor->data);
}

void convertFloatTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    float absMax = findAbsMaxFloat(inputTensor->data, numberOfElements);

    symInt32QConfig_t *symInt32QC = outputTensor->quantization->qConfig;
    uint8_t qMaxBits = symInt32QC->qMaxBits;

    const float qMax = powf(2, (float)qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)qMaxBits - 1);

    float scale;
    if (absMax == 0.f) {
        scale = 1.f;
    } else {
        scale = absMax / qMax;
    }

    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    outputSymInt32QC->scale = scale;

    int32_t *outputInt32 = (int32_t *)outputTensor->data;
    float *inputFloat = (float *)inputTensor->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        outputInt32[i] =
            roundByMode(clamp(inputFloat[i] / scale, qMin, qMax), outputSymInt32QC->roundingMode);
    }
}

void convertInt32TensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    outQC->scale = 1.f;
    int32_t codes[n];
    readBytesAsInt32Array(n, inputTensor->data, codes);
    packFitGuarded(codes, n, outputTensor->data, outQC->qBits, "convertInt32TensorToSymTensor");
}

void convertFloatTensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    packFloatBufferAsSym((float *)inputTensor->data, n, outQC, outputTensor->data,
                         "convertFloatTensorToSymTensor");
}

void convertFloatTensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *asymQConfig = outputTensor->quantization->qConfig;
    quantizeFloatToAsym((float *)inputTensor->data, numberOfElements, asymQConfig,
                        outputTensor->data);
}

// Important: Scale is ignored!
void extractInt32TensorFromSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    size_t bytesPerElement = sizeof(int32_t);

    int32_t inputAsInt32[numberOfElements];
    readBytesAsInt32Array(numberOfElements, inputTensor->data, inputAsInt32);

    memcpy(outputTensor->data, inputAsInt32, numberOfElements * bytesPerElement);
}

void convertSymInt32TensorToFloat32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfValues = calcNumberOfElementsByTensor(inputTensor);
    size_t bytesPerOutputElement = sizeof(float);

    int32_t *inputAsInt32 = (int32_t *)inputTensor->data;
    float output[numberOfValues];

    symInt32QConfig_t *inputSymInt32QConfig = inputTensor->quantization->qConfig;
    float scale = inputSymInt32QConfig->scale;

    for (size_t i = 0; i < numberOfValues; i++) {
        output[i] = (float)inputAsInt32[i] * scale;
    }
    memcpy(outputTensor->data, output, numberOfValues * bytesPerOutputElement);
}

void requantSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    symInt32QConfig_t *inputSymInt32QC = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    /* latch BEFORE writing outputSymInt32QC->scale: when called in-place
     * (inputTensor == outputTensor) both pointers alias the same config */
    float inScale = inputSymInt32QC->scale;

    const float qMax = powf(2, (float)outputSymInt32QC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outputSymInt32QC->qMaxBits - 1);

    int32_t *inputInt32 = (int32_t *)inputTensor->data;
    int32_t *outputInt32 = (int32_t *)outputTensor->data;

    /* pass A: absmax over dequantized values — reads only (alias-safe) */
    float absMax = 0.f;
    for (size_t i = 0; i < numberOfElements; i++) {
        float dequant = fabsf((float)inputInt32[i] * inScale);
        if (dequant > absMax) {
            absMax = dequant;
        }
    }

    float scale;
    if (absMax == 0.f) {
        scale = 1.f;
    } else {
        scale = absMax / qMax;
    }
    outputSymInt32QC->scale = scale;

    /* pass B: same-index read-then-write — in-place safe (int32 both sides) */
    for (size_t i = 0; i < numberOfElements; i++) {
        outputInt32[i] = roundByMode(clamp(((float)inputInt32[i] * inScale) / scale, qMin, qMax),
                                     outputSymInt32QC->roundingMode);
    }
}

void requantSymInt32TensorToScale(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    symInt32QConfig_t *inputSymInt32QC = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    float inScale = inputSymInt32QC->scale;
    float targetScale = outputSymInt32QC->scale;

    /* NaN-robust: !(x > 0.f) is also true for NaN, unlike (x <= 0.f) */
    if (!(targetScale > 0.f)) {
        PRINT_ERROR("requantSymInt32TensorToScale: target scale must be pre-set and > 0 on "
                    "the output qConfig, got %f",
                    targetScale);
        exit(1);
    }

    const float qMax = powf(2, (float)outputSymInt32QC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outputSymInt32QC->qMaxBits - 1);

    int32_t *inputInt32 = (int32_t *)inputTensor->data;
    int32_t *outputInt32 = (int32_t *)outputTensor->data;

    /* single same-index read-then-write pass — shared-buffer in-place safe;
     * clamp saturates at qMin/qMax BY DESIGN (Deutel Eq. 4 analog) */
    for (size_t i = 0; i < numberOfElements; i++) {
        outputInt32[i] =
            roundByMode(clamp(((float)inputInt32[i] * inScale) / targetScale, qMin, qMax),
                        outputSymInt32QC->roundingMode);
    }
}

void convertSymInt32TensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfValues = calcNumberOfElementsByTensor(inputTensor);
    symInt32QConfig_t *inputSymInt32QConfig = inputTensor->quantization->qConfig;
    asymQConfig_t *outputAsymQConfig = outputTensor->quantization->qConfig;
    float inputScale = inputSymInt32QConfig->scale;
    int32_t *inputAsInt32 = (int32_t *)inputTensor->data;
    float inputAsFloat[numberOfValues];
    for (size_t i = 0; i < numberOfValues; i++) {
        inputAsFloat[i] = inputScale * (float)inputAsInt32[i];
    }
    quantizeFloatToAsym(inputAsFloat, numberOfValues, outputAsymQConfig, outputTensor->data);
}

void convertAsymTensorToInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    asymQConfig_t *asymQConfig = inputTensor->quantization->qConfig;
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    int16_t zeroPoint = asymQConfig->zeroPoint;
    uint8_t dataOut[numberOfElements * sizeof(int32_t)];
    memset(dataOut, 0, numberOfElements * sizeof(int32_t));
    byteConversion(inputTensor->data, asymQConfig->qBits, dataOut, 32, numberOfElements);
    int32_t outputElements[numberOfElements];
    readBytesAsInt32Array(numberOfElements, dataOut, outputElements);

    for (size_t elementIndex = 0; elementIndex < numberOfElements; elementIndex++) {
        outputElements[elementIndex] = outputElements[elementIndex] + zeroPoint;
    }
    writeInt32ArrayToByteArray(numberOfElements, outputElements, outputTensor->data);
}

void convertAsymTensorToFloatTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    zeroTensorData(outputTensor);
    asymQConfig_t *asymQConfig = inputTensor->quantization->qConfig;
    int16_t zeroPoint = asymQConfig->zeroPoint;
    int32_t inputInt[numberOfElements];
    byteConversion(inputTensor->data, asymQConfig->qBits, (uint8_t *)inputInt, 32,
                   numberOfElements);
    float *outputElements = (float *)outputTensor->data;

    for (size_t elementIndex = 0; elementIndex < numberOfElements; elementIndex++) {
        outputElements[elementIndex] =
            ((float)inputInt[elementIndex] + (float)zeroPoint) * asymQConfig->scale;
    }
}

void convertAsymTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    size_t bitsPerInputElement = calcBitsPerElement(inputTensor->quantization);
    size_t bytesPerOutputElement = sizeof(int32_t);

    asymQConfig_t *inputAsymQConfig = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QConfig = outputTensor->quantization->qConfig;

    int16_t zeroPoint = inputAsymQConfig->zeroPoint;

    int32_t inputAsInt32[numberOfElements];

    byteConversion(inputTensor->data, bitsPerInputElement, (uint8_t *)inputAsInt32, 32,
                   numberOfElements);

    for (size_t i = 0; i < numberOfElements; i++) {
        inputAsInt32[i] += zeroPoint;
    }

    memcpy(outputTensor->data, inputAsInt32, numberOfElements * bytesPerOutputElement);
    outputSymInt32QConfig->scale = inputAsymQConfig->scale;
}

static void unpackSignExtend(const uint8_t *src, size_t srcBits, int32_t *dst, size_t n) {
    if (srcBits == 0) {
        /* 1 << (srcBits - 1) below underflows size_t to SIZE_MAX -> UB shift (#247). */
        PRINT_ERROR("unpackSignExtend: srcBits must be > 0");
        exit(1);
    }
    byteConversion((uint8_t *)src, srcBits, (uint8_t *)dst, 32, n); /* zero-fills high bits */
    if (srcBits >= 32) {
        return;
    }
    const int32_t signBit = (int32_t)1 << (srcBits - 1);
    const int32_t mask = (int32_t)(((uint32_t)1 << srcBits) - 1u);
    for (size_t i = 0; i < n; i++) {
        int32_t v = dst[i] & mask;
        dst[i] = (v ^ signBit) - signBit; /* sign-extend from srcBits */
    }
}

// Important: Scale is ignored! Emits sign-extended integer codes (int_repr).
void convertSymTensorToInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    unpackSignExtend(inputTensor->data, inQC->qBits, (int32_t *)outputTensor->data, n);
}

void convertSymTensorToFloat32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    int32_t mant[n];
    unpackSignExtend(inputTensor->data, inQC->qBits, mant, n);
    float *out = (float *)outputTensor->data;
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)mant[i] * inQC->scale;
    }
}

void convertSymTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outQC = outputTensor->quantization->qConfig;

    unpackSignExtend(inputTensor->data, inQC->qBits, (int32_t *)outputTensor->data, n);
    outQC->scale = inQC->scale;
    outQC->qMaxBits = inQC->qBits;
}

void convertSymTensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    int32_t mant[n];
    unpackSignExtend(inputTensor->data, inQC->qBits, mant, n);
    float deq[n];
    for (size_t i = 0; i < n; i++) {
        deq[i] = (float)mant[i] * inQC->scale;
    }
    asymQConfig_t *outQC = outputTensor->quantization->qConfig;
    quantizeFloatToAsym(deq, n, outQC, outputTensor->data);
}

void convertAsymTensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *inQC = inputTensor->quantization->qConfig;
    size_t inBits = calcBitsPerElement(inputTensor->quantization);
    int32_t codes[n];
    byteConversion(inputTensor->data, inBits, (uint8_t *)codes, 32, n); /* asym codes >= 0 */
    float deq[n];
    for (size_t i = 0; i < n; i++) {
        deq[i] = ((float)codes[i] + (float)inQC->zeroPoint) * inQC->scale;
    }
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    packFloatBufferAsSym(deq, n, outQC, outputTensor->data, "convertAsymTensorToSymTensor");
}

char *quantTypeToString(qtype_t t) {
    switch (t) {
    case INT32:
        return "INT32";
    case FLOAT32:
        return "FLOAT32";
    case SYM_INT32:
        return "SYMINT32";
    case SYM:
        return "SYM";
    case ASYM:
        return "ASYM";
    case BOOL:
        return "BOOL";
    default:
        return "UNKNOWN";
    }
}

void unsupportedConversionTypes(tensor_t *inputTensor, tensor_t *outputTensor) {
    qtype_t inputQType = inputTensor->quantization->type;
    qtype_t outputQType = outputTensor->quantization->type;

    PRINT_ERROR("Conversion from %s to %s is not supported", quantTypeToString(inputQType),
                quantTypeToString(outputQType));
    exit(1);
}

static void packFitGuarded(const int32_t *src, size_t n, uint8_t *dst, size_t dstBits,
                           const char *what) {
    if (dstBits == 0 || dstBits > 31) {
        /* 1 << (dstBits - 1) needs dstBits in [1, 31]: 0 underflows size_t and
         * >= 32 overshoots the int32 sign bit -> UB shift (#247). */
        PRINT_ERROR("%s: dstBits (%u) must be in [1, 31]", what, (unsigned)dstBits);
        exit(1);
    }
    const int32_t hi = ((int32_t)1 << (dstBits - 1)) - 1;
    const int32_t lo = -((int32_t)1 << (dstBits - 1));
    for (size_t i = 0; i < n; i++) {
        if (src[i] < lo || src[i] > hi) {
            PRINT_ERROR("%s: value %d does not fit %u-bit SYM range [%d, %d] (#227)", what, src[i],
                        (unsigned)dstBits, lo, hi);
            exit(1);
        }
    }
    int32_t tmp[n];
    for (size_t i = 0; i < n; i++) {
        tmp[i] = src[i];
    }
    byteConversion((uint8_t *)tmp, 32, dst, dstBits, n);
}

static void packFloatBufferAsSym(const float *values, size_t n, symQConfig_t *outQC, uint8_t *dst,
                                 const char *what) {
    float absMax = findAbsMaxFloat((uint8_t *)values, n);
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    outQC->scale = scale;
    int32_t codes[n];
    for (size_t i = 0; i < n; i++) {
        codes[i] = roundByMode(clamp(values[i] / scale, qMin, qMax), outQC->roundingMode);
    }
    packFitGuarded(codes, n, dst, outQC->qBits, what);
}

void convertSymInt32TensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symInt32QConfig_t *inQC = inputTensor->quantization->qConfig;
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    float inScale = inQC->scale;
    int32_t *in = (int32_t *)inputTensor->data;
    float vals[n];
    for (size_t i = 0; i < n; i++) {
        vals[i] = (float)in[i] * inScale;
    }
    packFloatBufferAsSym(vals, n, outQC, outputTensor->data, "convertSymInt32TensorToSymTensor");
}

void repackSymInt32ToSymNoRescale(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symInt32QConfig_t *inQC = inputTensor->quantization->qConfig;
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    outQC->scale = inQC->scale;
    packFitGuarded((int32_t *)inputTensor->data, n, outputTensor->data, outQC->qBits,
                   "repackSymInt32ToSymNoRescale");
}

_Static_assert(BOOL + 1 == 6, "extend conversionMatrix when adding qtype_t entries");

conversionFunction_t conversionMatrix[6][6] = {
    [INT32] = {[INT32] = NULL,
               [FLOAT32] = convertInt32TensorToFloatTensor,
               [SYM_INT32] = convertInt32TensorToSymInt32Tensor,
               [SYM] = convertInt32TensorToSymTensor,
               [ASYM] = convertInt32TensorToAsymTensor,
               [BOOL] = unsupportedConversionTypes},
    [FLOAT32] = {[INT32] = convertFloatTensorToInt32Tensor,
                 [FLOAT32] = NULL,
                 [SYM_INT32] = convertFloatTensorToSymInt32Tensor,
                 [SYM] = convertFloatTensorToSymTensor,
                 [ASYM] = convertFloatTensorToAsymTensor,
                 [BOOL] = unsupportedConversionTypes},
    [SYM_INT32] = {[INT32] = extractInt32TensorFromSymInt32Tensor,
                   [FLOAT32] = convertSymInt32TensorToFloat32Tensor,
                   [SYM_INT32] = requantSymInt32Tensor,
                   [SYM] = convertSymInt32TensorToSymTensor,
                   [ASYM] = convertSymInt32TensorToAsymTensor,
                   [BOOL] = unsupportedConversionTypes},
    [SYM] = {[INT32] = convertSymTensorToInt32Tensor,
             [FLOAT32] = convertSymTensorToFloat32Tensor,
             [SYM_INT32] = convertSymTensorToSymInt32Tensor,
             [SYM] = NULL,
             [ASYM] = convertSymTensorToAsymTensor,
             [BOOL] = unsupportedConversionTypes},
    [ASYM] = {[INT32] = convertAsymTensorToInt32Tensor,
              [FLOAT32] = convertAsymTensorToFloatTensor,
              [SYM_INT32] = convertAsymTensorToSymInt32Tensor,
              [SYM] = convertAsymTensorToSymTensor,
              [ASYM] = NULL,
              [BOOL] = unsupportedConversionTypes},
    [BOOL] = {[INT32] = unsupportedConversionTypes,
              [FLOAT32] = unsupportedConversionTypes,
              [SYM_INT32] = unsupportedConversionTypes,
              [SYM] = unsupportedConversionTypes,
              [ASYM] = unsupportedConversionTypes,
              [BOOL] = NULL}};

static void convertTensorsWithSameType(tensor_t *inputTensor, tensor_t *outputTensor,
                                       qtype_t qType) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    size_t numberOfBytes = calcNumberOfBytesForData(inputTensor->quantization, numberOfElements);

    memmove(outputTensor->data, inputTensor->data, numberOfBytes);

    switch (qType) {
    case SYM_INT32:
        symInt32QConfig_t *inputSymIntQC = inputTensor->quantization->qConfig;
        symInt32QConfig_t *outputSymIntQC = outputTensor->quantization->qConfig;
        outputSymIntQC->scale = inputSymIntQC->scale;
        break;
    case ASYM:
        asymQConfig_t *inputAsymQC = inputTensor->quantization->qConfig;
        asymQConfig_t *outputAsymQC = outputTensor->quantization->qConfig;
        outputAsymQC->scale = inputAsymQC->scale;
        outputAsymQC->zeroPoint = inputAsymQC->zeroPoint;
        break;
    default:
        break;
    }
}

void convertTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    qtype_t inputDType = inputTensor->quantization->type;
    qtype_t outputDType = outputTensor->quantization->type;

    if (inputDType == outputDType) {
        convertTensorsWithSameType(inputTensor, outputTensor, inputDType);
    } else {
        conversionFunction_t conversionFn = conversionMatrix[inputDType][outputDType];
        if (conversionFn == NULL) {
            PRINT_ERROR("No conversion function registered for %s to %s",
                        quantTypeToString(inputDType), quantTypeToString(outputDType));
            exit(1);
        }
        conversionFn(inputTensor, outputTensor);
    }
}
