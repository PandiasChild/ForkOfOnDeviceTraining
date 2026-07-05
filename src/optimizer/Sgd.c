#define SOURCE_FILE "SGD"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Sgd.h"
#include "Tensor.h"

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay) {
    sgd->learningRate = learningRate;
    sgd->momentumFactor = momentumFactor;
    sgd->weightDecay = weightDecay;
}

typedef struct {
    float lr, weightDecay;
} sgdUpdateCtx_t; /* plain SGD */
typedef struct {
    float momentum, weightDecay;
} sgdMStateCtx_t; /* momentum: state */
typedef struct {
    float lr;
} sgdMParamCtx_t; /* momentum: param */

/* executeOp update kernels (design spec 2026-07-03 PR1b.2, D1): arithmetic is
 * always ARITH_FLOAT32, so the funnel prologue dequants every operand to
 * float and the epilogue requants rawOut into the target's own dtype
 * (OUT_WRITE) -- this is what gives the SGD update per-parameter dtype
 * dispatch without a qtype switch here. */

/* operands {param, grad} -> rawOut = param - lr*(grad + wd*param) */
static void sgdUpdateKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                            const void *ctxv) {
    (void)n;
    (void)aux;
    const sgdUpdateCtx_t *ctx = ctxv;
    size_t numberOfValues = calcNumberOfElementsByTensor(rawOut);
    float *param = (float *)op[0]->data;
    float *grad = (float *)op[1]->data;
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < numberOfValues; i++) {
        float g = grad[i] + ctx->weightDecay * param[i];
        out[i] = param[i] - ctx->lr * g;
    }
}

/* operands {state, grad, param} -> rawOut = mom*state + grad + wd*param */
static void sgdMStateKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                            const void *ctxv) {
    (void)n;
    (void)aux;
    const sgdMStateCtx_t *ctx = ctxv;
    size_t numberOfValues = calcNumberOfElementsByTensor(rawOut);
    float *state = (float *)op[0]->data;
    float *grad = (float *)op[1]->data;
    float *param = (float *)op[2]->data;
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < numberOfValues; i++) {
        float g = grad[i] + ctx->weightDecay * param[i];
        out[i] = ctx->momentum * state[i] + g;
    }
}

/* operands {param, state} -> rawOut = param - lr*state */
static void sgdMParamKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                            const void *ctxv) {
    (void)n;
    (void)aux;
    const sgdMParamCtx_t *ctx = ctxv;
    size_t numberOfValues = calcNumberOfElementsByTensor(rawOut);
    float *param = (float *)op[0]->data;
    float *state = (float *)op[1]->data;
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < numberOfValues; i++) {
        out[i] = param[i] - ctx->lr * state[i];
    }
}

void sgdStep(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *p = optim->parameter[i];
        sgdUpdateCtx_t ctx = {.lr = sgd->learningRate, .weightDecay = sgd->weightDecay};
        executeOp(
            &(opSpec_t){
                .kernel = sgdUpdateKernel,
                .ctx = &ctx,
                .inputs = (tensor_t *[]){p->param, p->grad},
                .nInputs = 2,
                .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                .mode = OUT_WRITE,
                .auxOut = NULL,
            },
            p->param);
    }
}

void sgdStepM(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *p = optim->parameter[i];
        tensor_t *state = optim->states[i]->stateBuffers[0];

        sgdMStateCtx_t sc = {.momentum = sgd->momentumFactor, .weightDecay = sgd->weightDecay};
        executeOp(
            &(opSpec_t){
                .kernel = sgdMStateKernel,
                .ctx = &sc,
                .inputs = (tensor_t *[]){state, p->grad, p->param},
                .nInputs = 3,
                .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                .mode = OUT_WRITE,
                .auxOut = NULL,
            },
            state);

        sgdMParamCtx_t pc = {.lr = sgd->learningRate};
        executeOp(
            &(opSpec_t){
                .kernel = sgdMParamKernel,
                .ctx = &pc,
                .inputs = (tensor_t *[]){p->param, state},
                .nInputs = 2,
                .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                .mode = OUT_WRITE,
                .auxOut = NULL,
            },
            p->param);
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
