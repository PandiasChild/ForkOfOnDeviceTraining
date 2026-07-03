#ifndef ENV5_RUNTIME_EXECUTE_OP_H
#define ENV5_RUNTIME_EXECUTE_OP_H

#include <stddef.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
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

/* Fail-fast guard for the OUT_WRITE==0 hazard (PR3 spec D1, per-layer
 * accumulate-mode knob): weightGradAccMode/biasGradAccMode are by-value
 * layerQuant_t/config fields with no "unset" sentinel of their own, and
 * OUT_WRITE happens to be the zero-init value -- a hand-wired config that
 * forgets to set them would otherwise silently pass OUT_WRITE into a grad
 * accumulate call. Call at each grad executeOp call site, right before
 * reading the mode; `context` names the layer and field for the message
 * (e.g. "Linear weightGradAccMode"). Shared here (not duplicated per layer
 * file) since every grad-accumulating layer already depends on ExecuteOp for
 * the funnel call itself -- no new library dependency. */
static inline void executeOpValidateAccMode(outputMode_t mode, const char *context) {
    if (mode != OUT_ACC_DYNAMIC_RESCALE && mode != OUT_ACC_FIXED_SCALE) {
        PRINT_ERROR("%s: not a valid grad accumulate mode (got %d) -- config field never set? "
                    "(PR3 spec, #261)",
                    context, (int)mode);
        exit(1);
    }
}

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
