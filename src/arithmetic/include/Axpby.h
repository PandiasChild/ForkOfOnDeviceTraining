#ifndef ODT_AXPBY_H
#define ODT_AXPBY_H

#include "Tensor.h"

/* out = a*x + b*y, elementwise FLOAT32.
 * Contract (Add/Sub engine conventions): x, y, out must have identical rank
 * and logical dims (fail-fast otherwise); input reads honor each tensor's
 * orderOfDimensions for identity or disjoint axis-pair-swap views ONLY --
 * composed (non-involution) permutations are rejected fail-fast, a shared
 * index-algebra limitation (calcElementIndexByIndices, Arithmetic.c); the
 * write is flat contiguous row-major over out. out may alias x or y for
 * identity-order operands (same-index read-then-write); aliasing a
 * PERMUTED-view input is untested and outside the contract.
 * Serves the PPCA mean-merge and CCIPCA update steps (#326). */
void axpbyFloat32Tensors(float a, tensor_t *x, float b, tensor_t *y, tensor_t *out);

#endif // ODT_AXPBY_H
