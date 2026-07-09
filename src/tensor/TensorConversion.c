#define SOURCE_FILE "TENSOR_CONVERSION"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

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

float saturatingSubstraction(float minuend, float subtrahend, float min, float max)
{
    if (subtrahend > 0) {
        if (minuend < min + subtrahend) {
            return min;
        }
    } else {
        if (minuend > max + subtrahend) {
            return max;
        }
    }
    int32_t difference = minuend - subtrahend;
    return difference;
}

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
        codes[i] =
            clampInt32(roundByMode(values[i] / scale - (float)zeroPoint, outQC->roundingMode), 0,
                       (int32_t)qMax);
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
            clampInt32(roundByMode(inputFloat[i] / scale, outputSymInt32QC->roundingMode),
                       (int32_t)qMin, (int32_t)qMax);
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
        outputInt32[i] = clampInt32(
            roundByMode((float)inputInt32[i] * (inScale / scale), outputSymInt32QC->roundingMode),
            (int32_t)qMin, (int32_t)qMax);
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

static void packFitGuardedForDelta(const int32_t *src, size_t n, uint8_t *dst, size_t qBits, size_t deltabits,
                           const char *what) {
    if (qBits == 0 || qBits > 31) {
        /* 1 << (dstBits - 1) needs dstBits in [1, 31]: 0 underflows size_t and
         * >= 32 overshoots the int32 sign bit -> UB shift (#247). */
        PRINT_ERROR("%s: qBits (%u) must be in [1, 31]", what, (unsigned)qBits);
        exit(1);
    }
    if (deltabits == 0 || deltabits > 31) {
        /* 1 << (dstBits - 1) needs dstBits in [1, 31]: 0 underflows size_t and
         * >= 32 overshoots the int32 sign bit -> UB shift (#247). */
        PRINT_ERROR("%s: deltabits (%u) must be in [1, 31]", what, (unsigned)deltabits);
        exit(1);
    }
    int32_t hi = ((int32_t)1 << (qBits - 1)) - 1;
    int32_t lo = -((int32_t)1 << (qBits - 1));
    if (src[0] < lo || src[0] > hi) {
        PRINT_ERROR("%s: value %d does not fit %u-bit SYM range [%d, %d] (#227)", what, src[0],
                    (unsigned)qBits, lo, hi);
        exit(1);
    }
    hi = ((int32_t)1 << (deltabits - 1)) - 1;
    lo = -((int32_t)1 << (deltabits - 1));
    for (size_t i = 1; i < n; i++) {
        if (src[i] < lo || src[i] > hi) {
            PRINT_ERROR("%s: value %d does not fit %u-bit SYM range [%d, %d] (#227)", what, src[i],
                        (unsigned)deltabits, lo, hi);
            exit(1);
        }
    }
    int32_t tmp[n];
    for (size_t i = 0; i < n; i++) {
        tmp[i] = src[i];
    }
    size_t totalBitAmount = ((n-1) * deltabits) + qBits;
    size_t totalByteAmount = (totalBitAmount + 7) / 8;
    memset(dst, 0, totalByteAmount);
    byteConversion((uint8_t *)tmp, 32, dst, qBits, 1);
    byteConversionWithOffsets((uint8_t *)tmp, 32, dst, 32, deltabits, (n-1), qBits);
}

void unpackSignExtendForDelta(const uint8_t *src, size_t qBits, size_t deltabits, int32_t *dst, size_t numberOfValues){
    if (qBits == 0) {
    /* 1 << (srcBits - 1) below underflows size_t to SIZE_MAX -> UB shift (#247). */
    PRINT_ERROR("unpackSignExtend: qBits must be > 0");
    exit(1);
    }
    if (deltabits == 0) {
        /* 1 << (srcBits - 1) below underflows size_t to SIZE_MAX -> UB shift (#247). */
        PRINT_ERROR("unpackSignExtend: deltabits must be > 0");
        exit(1);
    }
    uint8_t helperDst[numberOfValues * sizeof(int32_t)];
    memset(helperDst, 0, numberOfValues * 4); /* high bits will be filled with 0 */
    byteConversion((uint8_t *)src, qBits, helperDst, 32, 1);
    byteConversionWithOffsets((uint8_t *)src, deltabits, helperDst, qBits, 32, numberOfValues-1, 32);
    memcpy(dst, helperDst, numberOfValues * sizeof(int32_t));
    if (qBits >= 32) {
        return;
    }
    int32_t signBit = (int32_t)1 << (qBits - 1);
    int32_t mask = (int32_t)(((uint32_t)1 << qBits) - 1u);
    int32_t v = dst[0] & mask;
    dst[0] = (v ^ signBit) - signBit; /* sign-extend from qBits */

    if (deltabits >= 32) {
        return;
    }
    signBit = (int32_t)1 << (deltabits - 1);
    mask = (int32_t)(((uint32_t)1 << deltabits) - 1u);
    for (size_t i = 1; i < numberOfValues; i++) {
        int32_t v = dst[i] & mask;
        dst[i] = (v ^ signBit) - signBit; /* sign-extend from deltabits */
    }

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
    case DELTA:
        return "DELTA";
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



static void packFloatBufferAsSymForDelta(const float *values, size_t n, symQDeltaConfig_t *outQC, uint8_t *dst,
                                 const char *what) {
    float absMax = findAbsMaxFloat((uint8_t *)values, n);
    printf("absMax = %f\n", absMax);
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    //TODO: Qbits durch deltabits ersetzen:
    const float deltaMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float deltaMin = -powf(2, (float)outQC->qBits - 1);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    printf("scale = %f\n", scale);
    outQC->scale = scale;
    int32_t codes[n];
    codes[0] = clampInt32(roundByMode(values[0] / scale, outQC->roundingMode), (int32_t)qMin,
                              (int32_t)qMax);
    for (size_t i = 1; i < n; i++) {
        codes[i] = clampInt32(roundByMode(values[i] / scale, outQC->roundingMode), (int32_t)deltaMin,
                              (int32_t)deltaMax);
    }
    printf("Delta -----------------------------------------------------------------------------------------\n");
    /*
    packFitGuardedForDelta(codes, n, dst, outQC->qBits, outQC->qBits,what);
    */
    packFitGuarded(codes, n, dst, outQC->qBits ,what);
}

static void packFloatBufferAsSym(const float *values, size_t n, symQConfig_t *outQC, uint8_t *dst,
                                 const char *what) {
    float absMax = findAbsMaxFloat((uint8_t *)values, n);
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    outQC->scale = scale;
    printf("scale = %f\n", scale);
    int32_t codes[n];
    for (size_t i = 0; i < n; i++) {
        codes[i] = clampInt32(roundByMode(values[i] / scale, outQC->roundingMode), (int32_t)qMin,
                              (int32_t)qMax);
    }
    printf("SYM--------------------------------------------------------------------------------------------------\n");
    packFitGuarded(codes, n, dst, outQC->qBits, what);
}

static void packFloatBufferAsSym2(const float *values, size_t n, symQConfig_t *outQC, uint8_t *dst,
                                 const char *what) {
    float absMax = findAbsMaxFloat((uint8_t *)values, n);
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    outQC->scale = scale;
    printf("scale = %f\n", scale);
    int32_t codes[n];
    for (size_t i = 0; i < n; i++) {
        codes[i] = clampInt32(roundByMode(values[i] / scale, outQC->roundingMode), (int32_t)qMin,
                              (int32_t)qMax);
    }
    packFitGuardedForDelta(codes, n, dst, outQC->qBits, outQC->qBits, what);
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

void convertDeltaTensorToFloatTensor(tensor_t *inputTensor, tensor_t *outputTensor){
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQDeltaConfig_t *inQC = inputTensor->quantization->qConfig;
    int32_t mant[n];
    unpackSignExtendForDelta(inputTensor->data, inQC->qBits, inQC->deltabits, mant, n);
    float *out = (float *)outputTensor->data;
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)mant[i] * inQC->scale;
    }
    size_t firstIndex = 1;
    size_t lastIndex = n-1;
    for (int i= firstIndex; i <= lastIndex; i++){
        out[i] = out[i] + out[i-1];
    }
}
void convertDeltaTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor){
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQDeltaConfig_t *inQC = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outQC = outputTensor->quantization->qConfig;
    int32_t *out = (int32_t *)outputTensor->data;
    unpackSignExtendForDelta(inputTensor->data, inQC->qBits, inQC->deltabits, out, n);
    outQC->scale = inQC->scale;
    outQC->qMaxBits = inQC->qBits;
    size_t firstIndex = 1;
    size_t lastIndex = n-1;
    for (int i= firstIndex; i <= lastIndex; i++){
        out[i] = out[i] + out[i-1];
    }
}
/*
void convertFloatTensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    packFloatBufferAsSym((float *)inputTensor->data, n, outQC, outputTensor->data,
                         "convertFloatTensorToSymTensor");
}*/

void convertFloatTensorToDeltaTensor(tensor_t *inputTensor, tensor_t *outputTensor){
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    symQDeltaConfig_t *outQC = outputTensor->quantization->qConfig;
    float *inputData = (float *)inputTensor->data;
    float deltaData[numberOfElements];
    memset(deltaData, 0, numberOfElements * sizeof(float));
    deltaData[0] = inputData[0];
    size_t firstIndex = 1;
    size_t lastIndex = numberOfElements-1;
    for (int i= firstIndex; i <= lastIndex; i++){
        deltaData[i] = inputData[i] - inputData[i-1];
    }

    packFloatBufferAsSymForDelta(deltaData, numberOfElements, outQC, outputTensor->data,
                         "convertFloatTensorToDeltaTensor");

}


/* Grad-accumulate primitives (PR3, #261) -- see header doc comment for the
 * when-to-use contract. */
void accumulateFloatIntoSymTensorFixedGrid(tensor_t *target, const float *inc, size_t n) {
    symQConfig_t *qc = target->quantization->qConfig;
    int32_t mant[n];
    unpackSignExtend(target->data, qc->qBits, mant, n);

    bool allZero = true;
    for (size_t i = 0; i < n; i++) {
        if (mant[i] != 0) {
            allZero = false;
            break;
        }
    }
    if (allZero) {
        /* Fresh accumulator (post-initTensor zero-fill or post-sgdZeroGrad
         * memset): derive the grid from the increment (absmax/qMax; absmax
         * 0 -> scale 1.f, packFloatBufferAsSym convention). */
        float absMax = findAbsMaxFloat((uint8_t *)inc, n);
        const float qMax = powf(2, (float)qc->qBits - 1) - 1;
        qc->scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    }
    /* else: carry the grid verbatim -- no re-derivation, no renorm (D1/D2). */

    float scale = qc->scale;
    int32_t codes[n];
    for (size_t i = 0; i < n; i++) {
        codes[i] = roundByMode(((float)mant[i] * scale + inc[i]) / scale, qc->roundingMode);
    }
    /* No clamp: packFitGuarded aborts on overflow (D2, #227 discipline). */
    packFitGuarded(codes, n, target->data, qc->qBits, "accumulateFloatIntoSymTensorFixedGrid");
}

void accumulateFloatIntoSymTensorRescale(tensor_t *target, const float *inc, size_t n) {
    symQConfig_t *qc = target->quantization->qConfig;
    int32_t mant[n];
    unpackSignExtend(target->data, qc->qBits, mant, n);
    float vals[n];
    for (size_t i = 0; i < n; i++) {
        vals[i] = (float)mant[i] * qc->scale + inc[i];
    }
    /* Fresh absmax every call -- no carried grid (unlike the FixedGrid twin). */
    packFloatBufferAsSym(vals, n, qc, target->data, "accumulateFloatIntoSymTensorRescale");
}

void accumulateFloatIntoAsymTensorRescale(tensor_t *target, const float *inc, size_t n) {
    asymQConfig_t *qc = target->quantization->qConfig;
    int32_t codes[n];
    byteConversion(target->data, qc->qBits, (uint8_t *)codes, 32, n); /* asym codes >= 0 */
    float vals[n];
    for (size_t i = 0; i < n; i++) {
        vals[i] = ((float)codes[i] + (float)qc->zeroPoint) * qc->scale + inc[i];
    }
    /* Fresh affine grid every call (D4: no fit-preserving ASYM pack exists). */
    quantizeFloatToAsym(vals, n, qc, target->data);
}

_Static_assert(BOOL + 1 == 6, "extend conversionMatrix when adding qtype_t entries");

conversionFunction_t conversionMatrix[7][7] = {
    [INT32] = {[INT32] = NULL,
               [FLOAT32] = convertInt32TensorToFloatTensor,
               [SYM_INT32] = convertInt32TensorToSymInt32Tensor,
               [SYM] = convertInt32TensorToSymTensor,
               [ASYM] = convertInt32TensorToAsymTensor,
               [BOOL] = unsupportedConversionTypes,
               [DELTA] = unsupportedConversionTypes},
    [FLOAT32] = {[INT32] = convertFloatTensorToInt32Tensor,
                 [FLOAT32] = NULL,
                 [SYM_INT32] = convertFloatTensorToSymInt32Tensor,
                 [SYM] = convertFloatTensorToSymTensor,
                 [ASYM] = convertFloatTensorToAsymTensor,
                 [BOOL] = unsupportedConversionTypes,
                 [DELTA] = convertFloatTensorToDeltaTensor},
    [SYM_INT32] = {[INT32] = extractInt32TensorFromSymInt32Tensor,
                   [FLOAT32] = convertSymInt32TensorToFloat32Tensor,
                   [SYM_INT32] = requantSymInt32Tensor,
                   [SYM] = convertSymInt32TensorToSymTensor,
                   [ASYM] = convertSymInt32TensorToAsymTensor,
                   [BOOL] = unsupportedConversionTypes,
                   [DELTA] = unsupportedConversionTypes},
    [SYM] = {[INT32] = convertSymTensorToInt32Tensor,
             [FLOAT32] = convertSymTensorToFloat32Tensor,
             [SYM_INT32] = convertSymTensorToSymInt32Tensor,
             [SYM] = NULL,
             [ASYM] = convertSymTensorToAsymTensor,
             [BOOL] = unsupportedConversionTypes,
             [DELTA] = unsupportedConversionTypes},
    [ASYM] = {[INT32] = convertAsymTensorToInt32Tensor,
              [FLOAT32] = convertAsymTensorToFloatTensor,
              [SYM_INT32] = convertAsymTensorToSymInt32Tensor,
              [SYM] = convertAsymTensorToSymTensor,
              [ASYM] = NULL,
              [BOOL] = unsupportedConversionTypes,
              [DELTA] = unsupportedConversionTypes},
    [BOOL] = {[INT32] = unsupportedConversionTypes,
              [FLOAT32] = unsupportedConversionTypes,
              [SYM_INT32] = unsupportedConversionTypes,
              [SYM] = unsupportedConversionTypes,
              [ASYM] = unsupportedConversionTypes,
              [BOOL] = NULL,
              [DELTA] = unsupportedConversionTypes},
    [DELTA] = {[INT32] = unsupportedConversionTypes,
               [FLOAT32] = convertDeltaTensorToFloatTensor,
               [SYM_INT32] = convertDeltaTensorToSymInt32Tensor,
               [SYM] = unsupportedConversionTypes,
               [ASYM] = unsupportedConversionTypes,
               [BOOL] = unsupportedConversionTypes,
               [DELTA]= NULL}};

static void convertTensorsWithSameType(tensor_t *inputTensor, tensor_t *outputTensor,
                                       qtype_t qType) {
    size_t inputBits = calcBitsPerElement(inputTensor->quantization);
    size_t outputBits = calcBitsPerElement(outputTensor->quantization);
    if (inputBits != outputBits) {
        /* Same-type conversion is a verbatim packed-byte copy; differing widths
         * would reinterpret the packing (and overflow the output for wider inputs).
         * Width-changing SYM/ASYM rewrites are real conversions (repack policy:
         * PR3, #261). */
        PRINT_ERROR("Same-type conversion requires equal element widths (%zu vs %zu bits)",
                    inputBits, outputBits);
        exit(1);
    }
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    size_t numberOfBytes = calcNumberOfBytesForData(inputTensor->quantization, numberOfElements);

    memmove(outputTensor->data, inputTensor->data, numberOfBytes);

    switch (qType) {
    case SYM_INT32: {
        symInt32QConfig_t *inputSymIntQC = inputTensor->quantization->qConfig;
        symInt32QConfig_t *outputSymIntQC = outputTensor->quantization->qConfig;
        outputSymIntQC->scale = inputSymIntQC->scale;
        break;
    }
    case SYM: {
        symQConfig_t *inputSymQC = inputTensor->quantization->qConfig;
        symQConfig_t *outputSymQC = outputTensor->quantization->qConfig;
        outputSymQC->scale = inputSymQC->scale;
        break;
    }
    case ASYM: {
        asymQConfig_t *inputAsymQC = inputTensor->quantization->qConfig;
        asymQConfig_t *outputAsymQC = outputTensor->quantization->qConfig;
        outputAsymQC->scale = inputAsymQC->scale;
        outputAsymQC->zeroPoint = inputAsymQC->zeroPoint;
        break;
    }
    case DELTA: {
        symQConfig_t *inputDeltaQC = inputTensor->quantization->qConfig;
        symQConfig_t *outputDeltaQC = outputTensor->quantization->qConfig;
        outputDeltaQC->scale = inputDeltaQC->scale;
        break;
    }
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
