#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Sgd.h"
#include "Tensor.h"
#include "TensorConversion.h"

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay) {
    sgd->learningRate = learningRate;
    sgd->momentumFactor = momentumFactor;
    sgd->weightDecay = weightDecay;
}

static void sgdStepFloatCore(sgd_t *sgd, const float *gradArr, float *dataArr,
                             size_t numberOfValues) {
    for (size_t elementIndex = 0; elementIndex < numberOfValues; ++elementIndex) {
        float grad = gradArr[elementIndex] + sgd->weightDecay * dataArr[elementIndex];
        dataArr[elementIndex] -= sgd->learningRate * grad;
    }
}

static void sgdStepFloat(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;

    for (size_t stateIndex = 0; stateIndex < optim->sizeStates; stateIndex++) {
        parameter_t *param = optim->parameter[stateIndex];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);
        float *dataArr = (float *)param->param->data;

        /* Grad-dtype-generic read (PR3, #261): FLOAT32 keeps the raw-cast fast
         * path (no conversion, no scratch); any other admitted grad dtype
         * (SYM_INT32/SYM/ASYM) is dequantized into a float VLA scoped to this
         * branch only, mirroring the SYM_INT32 step arms below. */
        if (param->grad->quantization->type == FLOAT32) {
            sgdStepFloatCore(sgd, (float *)param->grad->data, dataArr, numberOfValues);
        } else {
            tensor_t gradFloat;
            quantization_t gradFloatQ;
            initFloat32Quantization(&gradFloatQ);
            float gradFloatData[numberOfValues];
            uint8_t *gradFloatDataBytes = (uint8_t *)gradFloatData;
            setTensorValuesForConversion(gradFloatDataBytes, &gradFloatQ, param->grad, &gradFloat);
            convertTensor(param->grad, &gradFloat);
            sgdStepFloatCore(sgd, (float *)gradFloat.data, dataArr, numberOfValues);
        }
    }
}

static void sgdStepSymInt32(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;

    for (size_t stateIndex = 0; stateIndex < optim->sizeStates; stateIndex++) {
        parameter_t *param = optim->parameter[stateIndex];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);

        tensor_t paramFloat;
        quantization_t paramFloatQ;
        initFloat32Quantization(&paramFloatQ);
        uint8_t paramFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(paramFloatData, &paramFloatQ, param->param, &paramFloat);
        convertTensor(param->param, &paramFloat);

        float *paramFloatArr = (float *)paramFloat.data;

        tensor_t gradFloat;
        quantization_t gradFloatQ;
        initFloat32Quantization(&gradFloatQ);
        float gradFloatData[numberOfValues];
        uint8_t *gradFloatDataBytes = (uint8_t *)gradFloatData;
        setTensorValuesForConversion(gradFloatDataBytes, &gradFloatQ, param->grad, &gradFloat);
        convertTensor(param->grad, &gradFloat);

        float *gradFloatArr = (float *)gradFloat.data;

        for (size_t j = 0; j < numberOfValues; ++j) {
            float grad = gradFloatArr[j] + sgd->weightDecay * paramFloatArr[j];
            paramFloatArr[j] -= sgd->learningRate * grad;
        }

        convertTensor(&paramFloat, param->param);
    }
}

void sgdStep(optimizer_t *optimizer) {
    switch (optimizer->qtype) {
    case FLOAT32:
        sgdStepFloat(optimizer);
        break;
    case SYM_INT32:
        sgdStepSymInt32(optimizer);
        break;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

static void sgdStepMFloatCore(sgd_t *sgd, const float *gradArr, float *paramArr, float *stateArr,
                              size_t numberOfValues) {
    for (size_t elementIndex = 0; elementIndex < numberOfValues; ++elementIndex) {
        float grad = gradArr[elementIndex] + sgd->weightDecay * paramArr[elementIndex];
        stateArr[elementIndex] = sgd->momentumFactor * stateArr[elementIndex] + grad;
        paramArr[elementIndex] -= sgd->learningRate * stateArr[elementIndex];
    }
}

static void sgdStepMFloat(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *param = optim->parameter[i];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);
        float *paramArr = (float *)param->param->data;

        states_t *states = optim->states[i];
        tensor_t *state = states->stateBuffers[0];
        float *stateArr = (float *)state->data;

        /* Grad-dtype-generic read (PR3, #261) - see sgdStepFloat for the
         * fast-path/VLA rationale. */
        if (param->grad->quantization->type == FLOAT32) {
            sgdStepMFloatCore(sgd, (float *)param->grad->data, paramArr, stateArr, numberOfValues);
        } else {
            tensor_t gradFloat;
            quantization_t gradFloatQ;
            initFloat32Quantization(&gradFloatQ);
            float gradFloatData[numberOfValues];
            uint8_t *gradFloatDataBytes = (uint8_t *)gradFloatData;
            setTensorValuesForConversion(gradFloatDataBytes, &gradFloatQ, param->grad, &gradFloat);
            convertTensor(param->grad, &gradFloat);
            sgdStepMFloatCore(sgd, (float *)gradFloat.data, paramArr, stateArr, numberOfValues);
        }
    }
}

static void sgdStepMSymInt32(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;

    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *param = optim->parameter[i];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);

        tensor_t paramFloat;
        quantization_t paramFloatQ;
        initFloat32Quantization(&paramFloatQ);
        uint8_t paramFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(paramFloatData, &paramFloatQ, param->param, &paramFloat);
        convertTensor(param->param, &paramFloat);
        float *paramFloatArr = (float *)paramFloat.data;

        tensor_t gradFloat;
        quantization_t gradFloatQ;
        initFloat32Quantization(&gradFloatQ);
        float gradFloatData[numberOfValues];
        uint8_t *gradFloatDataBytes = (uint8_t *)gradFloatData;
        setTensorValuesForConversion(gradFloatDataBytes, &gradFloatQ, param->grad, &gradFloat);
        convertTensor(param->grad, &gradFloat);
        float *gradFloatArr = (float *)gradFloat.data;

        states_t *states = optim->states[i];

        tensor_t *state = states->stateBuffers[0];

        tensor_t stateFloat;
        quantization_t stateFloatQ;
        initFloat32Quantization(&stateFloatQ);
        uint8_t stateFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(stateFloatData, &stateFloatQ, state, &stateFloat);
        convertTensor(state, &stateFloat);
        float *stateFloatArr = (float *)stateFloat.data;

        for (size_t j = 0; j < numberOfValues; ++j) {
            float grad = gradFloatArr[j] + sgd->weightDecay * paramFloatArr[j];
            stateFloatArr[j] = sgd->momentumFactor * stateFloatArr[j] + grad;
            paramFloatArr[j] -= sgd->learningRate * stateFloatArr[j];
        }

        convertTensor(&stateFloat, state);
        convertTensor(&paramFloat, param->param);
    }
}

void sgdStepM(optimizer_t *optimizer) {
    switch (optimizer->qtype) {
    case FLOAT32:
        sgdStepMFloat(optimizer);
        break;
    case SYM_INT32:
        sgdStepMSymInt32(optimizer);
        break;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

void sgdZeroGrad(optimizer_t *optimizer) {
    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];
        size_t paramSize = calcNumberOfElementsByParameter(param);
        size_t totalNumberOfBytes = calcNumberOfBytesForData(param->grad->quantization, paramSize);

        memset(param->grad->data, 0, totalNumberOfBytes);

        /* Byte-zero the mantissa/code storage above is necessary but, for
         * SYM/ASYM, not sufficient for VALUE-zero: config-reset the grid so
         * code 0 decodes to exactly 0.0f (spec §5.3). SYM_INT32's scale reset
         * is hygiene (the first-store trigger is the all-zero mantissa state,
         * not the scale); ASYM's zeroPoint reset is load-bearing - without it,
         * code 0 would decode to zeroPoint*scale, not 0 (PR2 watch-list item). */
        switch (param->grad->quantization->type) {
        case SYM_INT32: {
            symInt32QConfig_t *symIntQ = param->grad->quantization->qConfig;
            symIntQ->scale = 1.f;
            break;
        }
        case SYM: {
            symQConfig_t *symQ = param->grad->quantization->qConfig;
            symQ->scale = 1.f;
            break;
        }
        case ASYM: {
            asymQConfig_t *asymQ = param->grad->quantization->qConfig;
            asymQ->scale = 1.f;
            asymQ->zeroPoint = 0;
            break;
        }
        default:
            break;
        }
    }
}
