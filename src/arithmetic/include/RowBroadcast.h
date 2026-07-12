#ifndef ODT_ROW_BROADCAST_H
#define ODT_ROW_BROADCAST_H

#include "Tensor.h"

/* Standalone tensor-over-tensor row-broadcast primitives (#326). mat is
 * rank-2 [r,c] FLOAT32; reads are taken in FLAT storage order (mat must be
 * storage-ordered — pass materialized matrices, not transpose views);
 * writes are flat contiguous. out may alias mat (same-index
 * read-then-write). Fail-fast on dtype/rank/dims violations. */

/* out[i][j] = rowScales[i] * mat[i][j]; rowScales is [r] FLOAT32. */
void scaleRowsFloat32(tensor_t *mat, tensor_t *rowScales, tensor_t *out);

/* out[i][j] = mat[i][j] - row[j]; row is [c] FLOAT32. */
void subRowBroadcastFloat32(tensor_t *mat, tensor_t *row, tensor_t *out);

#endif // ODT_ROW_BROADCAST_H
