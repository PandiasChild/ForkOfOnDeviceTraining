#define SOURCE_FILE "EXECUTE-OP"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Add.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Quantization.h"
#include "Tensor.h"
#include "TensorConversion.h"

/* Maximum operands any in-tree op passes (matmul: 2). Bump deliberately. */
#define EXECUTE_OP_MAX_INPUTS 2

void executeOpIdentityKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut) {
    (void)nOperands;
    tensor_t *src = operands[0];
    size_t n = calcNumberOfElementsByTensor(src);
    size_t bytes = calcNumberOfBytesForData(src->quantization, n);
    memcpy(rawOut->data, src->data, bytes);
    if (src->quantization->type == SYM_INT32) {
        ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale =
            ((symInt32QConfig_t *)src->quantization->qConfig)->scale;
    }
}

/* Phase 4, OUT_WRITE: intermediate -> target in the target's dtype. Same-dtype
 * SYM->SYM must REQUANT via the conversionMatrix diagonal — convertTensor's
 * same-type branch is a memmove that would pass raw mantissas through
 * unrestored (the QuantizationLayer.c trap). */
static void writeOut(tensor_t *intermediate, tensor_t *target) {
    if (intermediate->quantization->type == SYM_INT32 && target->quantization->type == SYM_INT32) {
        conversionMatrix[SYM_INT32][SYM_INT32](intermediate, target);
        return;
    }
    convertTensor(intermediate, target);
}

/* Phase 4, ACC modes. The SYM->SYM add is Strategy A via
 * addSymInt32TensorsInplace (bit-identical to Linear.c's weight-grad
 * accumulate); the FLOAT32-intermediate -> SYM arm first quantizes the
 * increment to operand width with the TARGET's roundingMode (bit-identical
 * to LayerNorm's layerNormAccumulateGradSymInt32). Fixed-scale reproduces
 * linearCalcBiasGradsSymInt32: raw roundf, no clamp, scale never re-derived. */
static void accumulateOut(tensor_t *intermediate, tensor_t *target, outputMode_t mode) {
    size_t n = calcNumberOfElementsByTensor(target);

    switch (target->quantization->type) {
    case FLOAT32: {
        if (intermediate->quantization->type == FLOAT32) {
            addFloat32TensorsInplace(target, intermediate);
            return;
        }
        /* SYM intermediate: dequantize, then exact float add. */
        quantization_t floatQ;
        initFloat32Quantization(&floatQ);
        uint8_t incFloatData[(n > 0 ? n : 1) * sizeof(float)];
        tensor_t incFloat;
        setTensorValuesForConversion(incFloatData, &floatQ, intermediate, &incFloat);
        convertTensor(intermediate, &incFloat);
        addFloat32TensorsInplace(target, &incFloat);
        return;
    }
    case SYM_INT32: {
        symInt32QConfig_t *targetQC = target->quantization->qConfig;
        if (targetQC->qMaxBits > ODT_SYM_GRAD_QMAXBITS) {
            PRINT_ERROR("executeOp: SYM grad target qMaxBits (%u) exceeds grad contract (%u)",
                        (unsigned)targetQC->qMaxBits, (unsigned)ODT_SYM_GRAD_QMAXBITS);
            exit(1);
        }
        if (mode == OUT_ACC_FIXED_SCALE) {
            if (intermediate->quantization->type != SYM_INT32) {
                PRINT_ERROR("executeOp: OUT_ACC_FIXED_SCALE needs a SYM intermediate "
                            "for a SYM target (got dtype %d)",
                            (int)intermediate->quantization->type);
                exit(1);
            }
            float intermScale = ((symInt32QConfig_t *)intermediate->quantization->qConfig)->scale;
            float targetScale = targetQC->scale;
            int32_t *tg = (int32_t *)target->data;
            int32_t *in = (int32_t *)intermediate->data;
            for (size_t i = 0; i < n; i++) {
                tg[i] += (int32_t)roundf((float)in[i] * intermScale / targetScale);
            }
            return;
        }
        /* OUT_ACC_DYNAMIC_RESCALE */
        if (intermediate->quantization->type == SYM_INT32) {
            addSymInt32TensorsInplace(target, intermediate);
            return;
        }
        /* FLOAT32 intermediate: quantize to operand width first, roundingMode
         * from the TARGET's qConfig (LayerNorm.c:446-463 reproduction). */
        symInt32QConfig_t incQC;
        initSymInt32QConfig(targetQC->roundingMode, &incQC);
        quantization_t incQ;
        initSymInt32Quantization(&incQC, &incQ);
        uint8_t incSymData[(n > 0 ? n : 1) * sizeof(int32_t)];
        tensor_t incSym;
        setTensorValuesForConversion(incSymData, &incQ, intermediate, &incSym);
        convertTensor(intermediate, &incSym);
        addSymInt32TensorsInplace(target, &incSym);
        return;
    }
    default:
        PRINT_ERROR("executeOp: accumulate target dtype %d not supported "
                    "(FLOAT32/SYM_INT32; sub-byte arms land in PR3, #261)",
                    (int)target->quantization->type);
        exit(1);
    }
}

void executeOp(opKernelFn_t kernel, tensor_t **inputs, size_t nInputs, quantization_t *arithmeticQ,
               tensor_t *target, outputMode_t mode) {
    if (nInputs > EXECUTE_OP_MAX_INPUTS) {
        PRINT_ERROR("executeOp: %u inputs exceeds EXECUTE_OP_MAX_INPUTS (%u)", (unsigned)nInputs,
                    (unsigned)EXECUTE_OP_MAX_INPUTS);
        exit(1);
    }

    /* Phase 1 — prologue: convert mismatched operands into stack scratch.
     * Sizes are data-dependent (VLAs), so scratch lives in this frame. */
    size_t maxElems = 0;
    for (size_t i = 0; i < nInputs; i++) {
        size_t n = calcNumberOfElementsByTensor(inputs[i]);
        if (n > maxElems) {
            maxElems = n;
        }
    }
    /* One worst-case-sized backing row per operand slot; 4 bytes/element for
     * both FLOAT32 and SYM_INT32 arithmetic. */
    uint8_t scratch[EXECUTE_OP_MAX_INPUTS][(maxElems > 0 ? maxElems : 1) * sizeof(int32_t)];
    tensor_t scratchTensors[EXECUTE_OP_MAX_INPUTS];
    quantization_t scratchQ[EXECUTE_OP_MAX_INPUTS];
    symInt32QConfig_t scratchQC[EXECUTE_OP_MAX_INPUTS];
    tensor_t *ops[EXECUTE_OP_MAX_INPUTS];

    for (size_t i = 0; i < nInputs; i++) {
        if (inputs[i]->quantization->type == arithmeticQ->type) {
            ops[i] = inputs[i];
            continue;
        }
        switch (arithmeticQ->type) {
        case FLOAT32:
            initFloat32Quantization(&scratchQ[i]);
            break;
        case SYM_INT32: {
            symInt32QConfig_t *aqc = arithmeticQ->qConfig;
            initSymInt32QConfig(aqc->roundingMode, &scratchQC[i]);
            initSymInt32Quantization(&scratchQC[i], &scratchQ[i]);
            break;
        }
        default:
            PRINT_ERROR("executeOp: arithmetic dtype %d not supported (FLOAT32/SYM_INT32)",
                        (int)arithmeticQ->type);
            exit(1);
        }
        setTensorValuesForConversion(scratch[i], &scratchQ[i], inputs[i], &scratchTensors[i]);
        convertTensor(inputs[i], &scratchTensors[i]);
        ops[i] = &scratchTensors[i];
    }

    /* Phase 2 — intermediate in the arithmetic representation, target shape. */
    size_t outElems = calcNumberOfElementsByTensor(target);
    uint8_t rawData[(outElems > 0 ? outElems : 1) * sizeof(int32_t)];
    tensor_t raw;
    quantization_t rawQ;
    symInt32QConfig_t rawQC;
    switch (arithmeticQ->type) {
    case FLOAT32:
        initFloat32Quantization(&rawQ);
        break;
    case SYM_INT32:
        initSymInt32QConfig(((symInt32QConfig_t *)arithmeticQ->qConfig)->roundingMode, &rawQC);
        initSymInt32Quantization(&rawQC, &rawQ);
        break;
    default:
        PRINT_ERROR("executeOp: arithmetic dtype %d not supported (FLOAT32/SYM_INT32)",
                    (int)arithmeticQ->type);
        exit(1);
    }
    setTensorValues(&raw, rawData, target->shape, &rawQ, target->sparsity);

    /* Phase 3 — kernel emits raw. */
    kernel(ops, nInputs, &raw);

    /* Phase 4 — epilogue. */
    switch (mode) {
    case OUT_WRITE:
        writeOut(&raw, target);
        break;
    case OUT_ACC_DYNAMIC_RESCALE:
    case OUT_ACC_FIXED_SCALE:
        accumulateOut(&raw, target, mode);
        break;
    }
}
