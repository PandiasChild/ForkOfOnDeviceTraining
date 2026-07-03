#ifndef ENV5_RUNTIME_EXECUTE_OP_H
#define ENV5_RUNTIME_EXECUTE_OP_H

#include <stddef.h>

#include "ArithmeticType.h"
#include "Tensor.h"

/* The one conversion funnel (design spec 2026-07-03 PR1b.2, D1). Every op runs:
 *   prologue   — operands whose dtype != arithmetic are converted into
 *                transient stack scratch (sources are never mutated)
 *   kernel     — pure computation: operands (in arithmetic representation)
 *                -> raw intermediate (SYM kernels emit raw int32 mantissas);
 *                also given auxOut (kernel-written verbatim, e.g. MaxPool's
 *                argmax indices) and ctx (kernel geometry/config), neither of
 *                which the funnel touches
 *   epilogue   — the intermediate is written/accumulated into the target,
 *                in the TARGET's own dtype. Persistent buffers are never
 *                pulled through the arithmetic. auxOut is NEVER funnel-
 *                converted — the kernel writes it in its own storage format.
 * Escape hatch policy: the prologue/epilogue helpers stay static in
 * ExecuteOp.c. Opening one for an op that does not fit the n-inputs/1-output
 * shape requires a documented exception here. */

typedef void (*opKernelFn_t)(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx);

typedef enum {
    OUT_WRITE,               /* overwrite target (wire / forward output); SYM->SYM
                              * routes through the conversionMatrix diagonal
                              * (requant), never the same-type memmove */
    OUT_ACC_DYNAMIC_RESCALE, /* Strategy A: dequant both -> float-add -> fresh
                              * absmax scale (SYM targets); exact add (FLOAT32) */
    OUT_ACC_FIXED_SCALE      /* bias scheme: rescale increment into the target's
                              * EXISTING scale via rescaleIntoAccumulatorScale
                              * (honors the TARGET's roundingMode), integer
                              * add, no clamp (SYM targets); exact add (FLOAT32) */
} outputMode_t;

/*! @brief Descriptor for a funnel op invocation (design spec D1).
 * ctx: kernel geometry/config, opaque to the funnel, passed straight through
 *      to the kernel; NULL for config-free kernels.
 * auxOut: kernel-written verbatim, in ITS OWN storage format — never funnel-
 *         converted; NULL for single-output ops (e.g. MaxPool1d's argmax
 *         indices live here). */
typedef struct opSpec {
    opKernelFn_t kernel;
    const void *ctx;
    tensor_t **inputs;
    size_t nInputs;
    arithmetic_t arithmetic;
    outputMode_t mode;
    tensor_t *auxOut;
} opSpec_t;

void executeOp(const opSpec_t *spec, tensor_t *target);

/* Copies operand 0 into rawOut (data + SYM scale if applicable). For ops whose
 * increment is produced inline (LayerNorm dgamma/dbeta only — the Quantization
 * layer is a pure conversion node routed through executeConvert instead).
 * Ignores auxOut/ctx. */
void executeOpIdentityKernel(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                             tensor_t *auxOut, const void *ctx);

/* Kernel-less funnel form: storage-to-storage conversion (1 input,
 * OUT_WRITE semantics). SYM->SYM routes through the conversionMatrix
 * diagonal (requant), never the same-type memmove. Supports every
 * populated conversionMatrix cell. */
void executeConvert(tensor_t *input, tensor_t *target);

#endif // ENV5_RUNTIME_EXECUTE_OP_H
