#ifndef ENV5_RUNTIME_ARITHMETIC_TYPE_H
#define ENV5_RUNTIME_ARITHMETIC_TYPE_H

#include "Quantization.h"
#include "Rounding.h"

/* Declared compute representation of an op (design spec 2026-07-02
 * arithmetic-type-split, D1). BY VALUE in layer configs: no ownership,
 * no teardown. Only compute-capable representations exist here — storage
 * dtypes (SYM/ASYM/BOOL/INT32) are expressed on tensors/storage configs. */
typedef enum arithmeticType { ARITH_FLOAT32, ARITH_SYM_INT32 } arithmeticType_t;

typedef struct arithmetic {
    arithmeticType_t type;
    roundingMode_t roundingMode; /* SYM intermediates; ignored for ARITH_FLOAT32 */
} arithmetic_t;

/* Derivation rule (spec D5): FLOAT32 -> ARITH_FLOAT32; SYM_INT32 ->
 * ARITH_SYM_INT32; storage-only dtypes -> ARITH_FLOAT32 (float is the
 * universal compute bridge). roundingMode is taken from the qConfig when
 * the dtype carries one, else HALF_AWAY. */
arithmetic_t arithmeticFromQuantization(const quantization_t *q);

#endif // ENV5_RUNTIME_ARITHMETIC_TYPE_H
