#ifndef ODT_JACOBI_EIG_H
#define ODT_JACOBI_EIG_H

#include <stddef.h>

#include "Tensor.h"

/* Symmetric eigendecomposition via cyclic Jacobi rotations (Golub & Van Loan,
 * Matrix Computations, §8.5). FLOAT32-only (#326 v1); integer eigensolvers
 * are literature-first future work — see the ARITH_SYM_INT32 tracking issue.
 *
 * a          symmetric [p,p] FLOAT32 — DESTROYED (in-place workspace; the
 *            off-diagonals are annihilated in place, keeping the module free
 *            of hidden scratch per the allocation-locality rule).
 * eigvalsOut [p] FLOAT32, sorted DESCENDING.
 * eigvecsOut [p,p] FLOAT32, ROW i = unit eigenvector of eigvalsOut[i]
 *            (rows-convention: matches PPCA basis storage; a rotate-back is
 *            then a plain matmul).
 * maxSweeps  hard sweep cap (deterministic termination).
 * tol        stop when off(A) = sqrt(sum of squared off-diagonals) < tol.
 *
 * Rotation math uses plain float expressions (self-contained kernel, the
 * Matmul precedent) — not the scalar wrappers.
 * Fail-fast: non-FLOAT32 operands, non-square a, output dims mismatch. */
void jacobiEigSymFloat32(tensor_t *a, tensor_t *eigvalsOut, tensor_t *eigvecsOut, size_t maxSweeps,
                         float tol);

#endif // ODT_JACOBI_EIG_H
