#define SOURCE_FILE "SGD_API"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/*! Builds a momentum-state tensor at `param`'s shape but with its OWN
 * quantization (`momentumQuant`, deep-cloned via getQLike) -- decouples the
 * accumulator's dtype from the parameter's storage dtype (#277 Task 2). */
static tensor_t *momentumStateInit(tensor_t *param, quantization_t *momentumQuant) {
    return initTensor(getShapeLike(param->shape), getQLike(momentumQuant), NULL);
}

optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             layer_t **model, size_t sizeModel, quantization_t *momentumQuant,
                             arithmetic_t updateMath) {
    optimizer_t *optim = reserveMemory(sizeof(optimizer_t));
    optim->type = SGD_M;

    optimImpl_t *sgdImpl = reserveMemory(sizeof(optimImpl_t));
    sgd_t *sgd = reserveMemory(sizeof(sgd_t));
    sgdInit(sgd, learningRate, momentumFactor, weightDecay, updateMath);
    sgdImpl->sgd = sgd;
    optim->impl = sgdImpl;

    size_t sizeStates = calcTotalNumberOfStates(model, sizeModel);
    optim->sizeStates = sizeStates;
    parameter_t **parameter = reserveMemory(sizeStates * sizeof(parameter_t *));
    optim->parameter = parameter;

    size_t paramSlot = 0;
    for (size_t i = 0; i < sizeModel; i++) {
        layer_t *currentLayer = model[i];
        layerConfig_t *layerConfig = currentLayer->config;

        switch (currentLayer->type) {
        case LINEAR: {
            linearConfig_t *linearConfig = layerConfig->linear;

            optim->parameter[paramSlot] = linearConfig->weights;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (linearConfig->bias != NULL) {
                optim->parameter[paramSlot + 1] = linearConfig->bias;
                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case CONV1D: {
            conv1dConfig_t *conv1dCfg = layerConfig->conv1d;

            optim->parameter[paramSlot] = conv1dCfg->weights;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (conv1dCfg->bias != NULL) {
                optim->parameter[paramSlot + 1] = conv1dCfg->bias;
                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case CONV1D_TRANSPOSED: {
            conv1dTransposedConfig_t *ctCfg = layerConfig->conv1dTransposed;

            optim->parameter[paramSlot] = ctCfg->weights;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (ctCfg->bias != NULL) {
                optim->parameter[paramSlot + 1] = ctCfg->bias;
                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case LAYERNORM: {
            layerNormConfig_t *lnCfg = layerConfig->layerNorm;

            optim->parameter[paramSlot] = lnCfg->gamma;
            optim->parameter[paramSlot + 1] = lnCfg->beta;

            paramSlot += 2;
            break;
        }
        case GROUPNORM: {
            groupNormConfig_t *gnCfg = layerConfig->groupNorm;

            optim->parameter[paramSlot] = gnCfg->gamma;
            optim->parameter[paramSlot + 1] = gnCfg->beta;

            paramSlot += 2;
            break;
        }
        case RELU:
        case SOFTMAX:
        case FLATTEN:
        case MAXPOOL1D:
        case AVGPOOL1D:
        case ADAPTIVE_AVGPOOL1D:
        case DROPOUT:
        case QUANTIZATION:
            break;
        default:
            PRINT_ERROR("Unknown Layer Type");
            exit(1);
        }
    }

    /* momentumFactor == 0: momentum state is semantically nonexistent --
     * allocate none (sgdStepM's momentum==0 path never reads states, #308).
     * Every trainable parameter otherwise gets one state buffer at the
     * param's shape with the caller's momentumQuant (dtype-decoupled, #277). */
    if (momentumFactor == 0.0f) {
        optim->states = NULL;
    } else {
        states_t **states = reserveMemory(sizeStates * sizeof(states_t *));
        for (size_t s = 0; s < sizeStates; s++) {
            states_t *paramStates = reserveMemory(sizeof(states_t));
            paramStates->statesPerParameter = 1;
            paramStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            paramStates->stateBuffers[0] =
                momentumStateInit(optim->parameter[s]->param, momentumQuant);
            states[s] = paramStates;
        }
        optim->states = states;
    }

    /* #261, PR3: grads may be stored FLOAT32 (default), SYM_INT32 (explicit
     * low-level knob), or packed SYM/ASYM (explicit grad-storage knob,
     * memory-constrained targets). INT32/BOOL grad storage remains
     * unimplemented - fail fast rather than silently misread bytes in an
     * unsupported layout. Non-trainable params carry no grad: skip. */
    for (size_t s = 0; s < optim->sizeStates; s++) {
        tensor_t *grad = optim->parameter[s]->grad;
        if (grad == NULL) {
            continue;
        }
        qtype_t gradType = grad->quantization->type;
        if (gradType != FLOAT32 && gradType != SYM_INT32 && gradType != SYM && gradType != ASYM) {
            PRINT_ERROR("sgdMCreateOptim: gradient storage dtype %d not supported "
                        "(accepted: FLOAT32, SYM_INT32, SYM, ASYM; INT32/BOOL grad "
                        "storage remains unsupported, #261)",
                        (int)gradType);
            exit(1);
        }
    }
    return optim;
}

void freeState(states_t *state) {
    for (size_t i = 0; i < state->statesPerParameter; i++) {
        freeTensor(state->stateBuffers[i]);
    }
    freeReservedMemory(state->stateBuffers);
    freeReservedMemory(state);
}

void freeOptimSgdM(optimizer_t *sgdM) {
    for (size_t i = 0; i < sgdM->sizeStates; i++) {
        freeParameter(sgdM->parameter[i]);
        if (sgdM->states != NULL) {
            freeState(sgdM->states[i]);
        }
    }
    freeReservedMemory(sgdM->parameter);
    if (sgdM->states != NULL) {
        freeReservedMemory(sgdM->states);
    }
    sgd_t *sgdImpl = sgdM->impl->sgd;
    freeReservedMemory(sgdImpl);
    freeReservedMemory(sgdM->impl);
    freeReservedMemory(sgdM);
}
