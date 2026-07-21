
der fehler entsteht hier
```
  if (dataInStartbit < dataInEndbit) {
      data = readByte(dataIn[dataInIndex], dataInStartbit, dataInEndbit);
  }
```
so gesehen gibt es das problem dann auch ausschließlich im ersten byte. im ersten byte wird zwar vom offset aus an gelesen, allerdings wird dann auf data in readByte eine Mask angewandt, dass dann nur die zu lesenden bytes in data geändert werden. Allerdings ist data dann im ersten durchlauf bei 0 und kennt die

//TODO: besserer kommentar
/* will not be filled with 0 prior writing. caller needs to ensure that dataOut is clear*/
//if no offset: dataInOffsetBits = dataInBits
//dataOutOffsetBits = dataOutBits
void byteConversionWithOffsets(uint8_t *dataIn, size_t dataInBits, size_t dataInOffsetBits, uint8_t *dataOut,
                               size_t dataOutBits, size_t dataOutOffsetBits, size_t numValues, size_t numValuesOffset){
    memset(dataOut, 0, ((numValues-numValuesOffset) * dataOutBits + numValuesOffset * dataOutOffsetBits - 1) / 8 + 1);
    int inBits = (int)dataInOffsetBits;
    int outBits = (int)dataOutOffsetBits;
    size_t dataOutIndex = 0;
    size_t dataInIndex = 0;
    int dataOutStartbit = 0;
    int dataInStartbit = 0;
    int dataInEndbit = inBits;
    int dataOutEndbit = outBits;
    for (size_t i = 0; i < numValues; i++) {
        /*
        printf("\n");
        printf("\n");
        printf("Value %i\n", i);
        */
        while ((dataInStartbit < dataInEndbit) | (dataOutStartbit < dataOutEndbit)) {
            /* Guard each side: input may exhaust before output (widening) or
             * output may fill before input (narrowing); skipping the out-of-range
             * access avoids OOB while preserving zero-fill semantics. */
            uint8_t data = 0;
            if (dataInStartbit < dataInEndbit) {
                data = readByte(dataIn[dataInIndex], dataInStartbit, dataInEndbit);
            }
            if (dataOutStartbit < dataOutEndbit) {
                dataOut[dataOutIndex] =
                    writeByte(dataOut[dataOutIndex], data, dataOutStartbit, dataOutEndbit);
            }

            /*
            printf("dataInStartbit %d\n", dataInStartbit);
            printf("dataInEndbit %d\n", dataInEndbit);
            printf("dataOutStartbit %d\n", dataOutStartbit);
            printf("dataOutEndbit %d\n", dataOutEndbit);
            printf("dataInIndex %d\n", dataInIndex);
            printf("dataOutIndex %d\n", dataOutIndex);
            printf("data");
            print_binary_uint8(data);
            printf("dataOut[dataOutIndex]");
            print_binary_uint8(dataOut[dataOutIndex]);
            */
            int valuesRead = min(dataInEndbit - dataInStartbit, 8 - dataInStartbit % 8);
            int valuesWritten = min(dataOutEndbit - dataOutStartbit, 8 - dataOutStartbit % 8);
            int minValue = min(valuesRead, valuesWritten);

            /*
            printf("valuesRead %d\n", valuesRead);
            printf("valuesWritten %d\n", valuesWritten);
            printf("minValue %d\n", minValue);
            */
            uint8_t deltaIn = minValue;
            uint8_t deltaOut = minValue;
            if (dataInStartbit == dataInEndbit) {
                dataOutStartbit += valuesWritten;
                deltaOut = valuesWritten;

            } else {
                dataOutStartbit += minValue;
            }
            if (dataOutStartbit == dataOutEndbit) {
                dataInStartbit += valuesRead;
                deltaIn = valuesRead;
            } else {
                dataInStartbit += minValue;
            }

            if (dataInStartbit / 8 > (dataInStartbit - deltaIn) / 8) {
                dataInIndex += 1;
            }
            if (dataOutStartbit / 8 > (dataOutStartbit - deltaOut) / 8) {
                dataOutIndex += 1;
            }
            //printf("\n");
        }
        if (numValuesOffset > 0)
        {
            numValuesOffset = numValuesOffset-1;
        }
        if (numValuesOffset == 0)
        {
            inBits = dataInBits;
            outBits = dataOutBits;
        }
        dataInStartbit = dataInEndbit % 8;
        dataInEndbit = dataInStartbit + inBits;
        dataOutStartbit = dataOutEndbit % 8;
        dataOutEndbit = dataOutStartbit + outBits;
    }
}

void byteConversion(uint8_t *dataIn, size_t dataInBits, uint8_t *dataOut, size_t dataOutBits,
                    size_t numValues) {
    memset(dataOut, 0, (numValues * dataOutBits - 1) / 8 + 1);
    byteConversionWithOffsets(dataIn, dataInBits, dataInBits, dataOut, dataOutBits, dataOutBits, numValues, 0);
}


//--------------------------------------------------------------------------------------------------------------


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
    float difference = minuend - subtrahend;
    return difference;
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
    byteConversionWithOffsets((uint8_t *)tmp, 32, 32, dst, deltabits, qBits, n, 1);
}

static void packFloatBufferAsSymForDelta(const float *values, size_t n, symQDeltaConfig_t *outQC, uint8_t *dst,
                                 const char *what) {
    float absMax = findAbsMaxFloat((uint8_t *)values, n);
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    const float deltaMax = powf(2, (float)outQC->deltabits - 1) - 1;
    const float deltaMin = -powf(2, (float)outQC->deltabits - 1);
    //printf("qMax = %f, deltaMax = %f\n", qMax, deltaMax);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;

    outQC->scale = scale;
    ---------------
    int32_t codes[n];
    //printf("roundedVal[%ld]= rounded(values[%ld]/scale) = %.6f/%.6f = %d\n", 0, 0, values[0], scale, roundByMode(values[0] / scale, outQC->roundingMode));
    codes[0] = clampInt32(roundByMode(values[0] / scale, outQC->roundingMode), (int32_t)qMin,
                              (int32_t)qMax);
    //printf("clampedVal[%ld] = codes[%ld] = %d\n", 0, 0, codes[0]);
    for (int i = 1; i < n; i++) {
        //printf("roundedVal[%ld]= rounded(values[%ld]/scale)\n = %.6f/%.6f = %d\n", i, i, values[i], scale, roundByMode(values[i] / scale, outQC->roundingMode));
        codes[i] = clampInt32(roundByMode(values[i] / scale, outQC->roundingMode), (int32_t)deltaMin,
                              (int32_t)deltaMax);
        //printf("clampedVal[%ld] = codes[%ld] = %d\n", i, i, codes[i]);
    }

    packFitGuardedForDelta(codes, n, dst, outQC->qBits, outQC->deltabits,what);

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

void convertFloatTensorToDeltaTensor(tensor_t *inputTensor, tensor_t *outputTensor){
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    symQDeltaConfig_t *outQC = outputTensor->quantization->qConfig;
    float *inputData = (float *)inputTensor->data;
    float deltaData[numberOfElements];
    memset(deltaData, 0, numberOfElements * sizeof(float));
    deltaData[0] = inputData[0];
    printf("deltaData[%d] = %f\n", 0, deltaData[0]);
    size_t firstIndex = 1;
    size_t lastIndex = numberOfElements-1;
    for (int i= firstIndex; i <= lastIndex; i++){
        deltaData[i] = inputData[i] - inputData[i-1];
        printf("deltaData[%d] = %f\n", i, deltaData[i]);
    }

    packFloatBufferAsSymForDelta(deltaData, numberOfElements, outQC, outputTensor->data,
                         "convertFloatTensorToDeltaTensor");

}
