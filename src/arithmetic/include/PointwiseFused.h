#ifndef POINTWISE_FUSED_H
#define POINTWISE_FUSED_H

#include <stddef.h>

#include "Tensor.h"

/*! Fused multi-operand elementwise primitives with PyTorch-parity ROUNDING
 * ORDER as the contract (#328 groundwork). Unlike the Add/Mul engine family
 * these are single-pass fused loops; the TU disables FP contraction so the
 * documented separate roundings cannot be silently fused — the one
 * deliberate fusion is lerp's explicit fmaf().
 *
 * Shared operand contract (fail-fast on violation): every tensor operand is
 * FLOAT32, has identity orderOfDimensions, and matches dimensions. Under
 * that gate every read of index i precedes the single write of index i, so
 * ANY aliasing among operands (b == c, b aliasing a, ...) is safe. The
 * identity-order requirement deliberately sidesteps the permuted-Inplace
 * hazard class (#339).
 *
 * Instruction counters: one increment per ELEMENT per call, compiled in
 * only under TRACK_INSTRUCTIONS (ODT_TRACK_INSTRUCTIONS cache var). */

/*! a[i] = fmaf(weight, b[i]-a[i], a[i]) — torch Tensor.lerp_(b, weight),
 * small-weight branch. Bit-identical to ATen's FMA-fused CPU kernel for
 * |weight| < 0.5 (AdamW: weight = 1-beta1). ATen switches formula at
 * |weight| >= 0.5 — parity is not claimed there. */
void lerpFloat32TensorsInplace(tensor_t *a, tensor_t *b, float weight);

/*! a[i] += (s*b[i])*c[i] — torch Tensor.addcmul_(b, c, value=s): three
 * separate roundings, left-associated (vectorized-path parity; gold
 * fixtures are multiples of 32 to stay in that regime). */
void addcmulFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *c, float s);

/*! a[i] += (s*b[i]) / (sqrtf(v[i])/dScale + eps) — torch
 * addcdiv_(b, (v.sqrt()/dScale).add_(eps), value=s): the same six
 * per-element roundings WITHOUT materializing torch's denom tensor.
 * Rounding contract is bit-identical to torch's macOS-ARM kernel; torch's
 * Linux-AVX2 vectorization fuses the final multiply-add and may differ by
 * <=1 ulp in zero-accumulator regimes (#353), so the gold fixtures pin the
 * documented rounding order itself, not a torch capture.
 * Domain (caller-guaranteed, not runtime-checked): v[i] >= 0 (AdamW's v is
 * a sum of squares), dScale > 0, eps > 0. */
void addcdivDenomFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *v, float dScale,
                                       float eps, float s);

size_t getLerpInstructionCounter(void);
size_t getAddcmulInstructionCounter(void);
size_t getAddcdivDenomInstructionCounter(void);

#endif // POINTWISE_FUSED_H
