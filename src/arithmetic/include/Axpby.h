#ifndef ODT_AXPBY_H
#define ODT_AXPBY_H

#include "Tensor.h"

/* out = a*x + b*y, elementwise FLOAT32.
 * Contract (Add/Sub engine conventions): x, y, out must have identical rank
 * and logical dims (fail-fast otherwise); input reads honor each tensor's
 * orderOfDimensions for ARBITRARY permutations (#344 fixed the shared index
 * algebra, calcElementIndexByIndices); the write is flat contiguous row-major,
 * so out must be identity-order (fail-fast otherwise). out may alias x or y
 * for identity-order operands (same-index read-then-write); aliasing a
 * PERMUTED-view input is untested and outside the contract.
 * Serves the PPCA mean-merge and CCIPCA update steps (#326). */
void axpbyFloat32Tensors(float a, tensor_t *x, float b, tensor_t *y, tensor_t *out);

#endif // ODT_AXPBY_H
