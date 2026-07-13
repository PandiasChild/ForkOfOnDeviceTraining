#ifndef BERNOULLI_H
#define BERNOULLI_H

#include "Tensor.h"

/*! Fills `mask` (a BOOL tensor) with independent Bernoulli draws: each bit is
 *  set (1) with probability `probTrue`, clear (0) with probability 1-probTrue.
 *  Uses the canonical RNG (`rngNextFloat`); seed via `rngSetSeed` for
 *  reproducibility. `mask` must be BOOL-quantized with its data buffer
 *  allocated. */
void bernoulliFillMask(tensor_t *mask, float probTrue);

/*! Signature of a Bernoulli mask-fill implementation, for swapping in
 *  optimized variants (SIMD, bit-parallel, hardware RNG). */
typedef void (*bernoulliFillMaskFn_t)(tensor_t *mask, float probTrue);

/*! The reference implementation (XorShift32 inverse-transform). Exported so it
 *  can be restored after swapping. */
void bernoulliFillMaskReference(tensor_t *mask, float probTrue);

/*! Install a sampler implementation. Passing NULL restores the reference. */
void bernoulliSetFillMaskFn(bernoulliFillMaskFn_t fn);

/*! Returns the currently active sampler (for save/restore in tests). */
bernoulliFillMaskFn_t bernoulliGetFillMaskFn(void);

#endif // BERNOULLI_H
