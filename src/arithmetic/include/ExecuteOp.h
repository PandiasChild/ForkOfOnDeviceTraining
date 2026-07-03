#ifndef ENV5_RUNTIME_EXECUTE_OP_H
#define ENV5_RUNTIME_EXECUTE_OP_H

#include <stddef.h>

#include "ArithmeticType.h"
#include "Tensor.h"

/* The one conversion funnel (design spec 2026-07-02, §3-§4). Every op runs:
 *   prologue   — operands whose dtype != arithmetic are converted into
 *                transient stack scratch (sources are never mutated)
 *   kernel     — pure computation: operands (in arithmetic representation)
 *                -> raw intermediate (SYM kernels emit raw int32 mantissas)
 *   epilogue   — the intermediate is written/accumulated into the target,
 *                in the TARGET's own dtype. Persistent buffers are never
 *                pulled through the arithmetic.
 * Escape hatch policy: the prologue/epilogue helpers stay static in
 * ExecuteOp.c. Opening one for an op that does not fit the n-inputs/1-output
 * shape requires a documented exception here. */

typedef void (*opKernelFn_t)(tensor_t **operands, size_t nOperands, tensor_t *rawOut);

typedef enum {
    OUT_WRITE,               /* overwrite target (wire / forward output); SYM->SYM
                              * routes through the conversionMatrix diagonal
                              * (requant), never the same-type memmove */
    OUT_ACC_DYNAMIC_RESCALE, /* Strategy A: dequant both -> float-add -> fresh
                              * absmax scale (SYM targets); exact add (FLOAT32) */
    OUT_ACC_FIXED_SCALE      /* bias scheme: rescale increment into the target's
                              * EXISTING scale, integer add, raw roundf, no clamp
                              * (SYM targets); exact add (FLOAT32) */
} outputMode_t;

void executeOp(opKernelFn_t kernel, tensor_t **inputs, size_t nInputs, arithmetic_t arithmetic,
               tensor_t *target, outputMode_t mode);

/* Copies operand 0 into rawOut (data + SYM scale if applicable). For ops whose
 * increment is produced inline (LayerNorm dgamma/dbeta only — the Quantization
 * layer is a pure conversion node routed through executeConvert instead). */
void executeOpIdentityKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut);

/* Kernel-less funnel form: storage-to-storage conversion (1 input,
 * OUT_WRITE semantics). SYM->SYM routes through the conversionMatrix
 * diagonal (requant), never the same-type memmove. Supports every
 * populated conversionMatrix cell. */
void executeConvert(tensor_t *input, tensor_t *target);

#endif // ENV5_RUNTIME_EXECUTE_OP_H
