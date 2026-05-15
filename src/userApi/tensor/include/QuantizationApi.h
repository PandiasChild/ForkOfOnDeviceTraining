#ifndef QUANTIZATIONAPI_H
#define QUANTIZATIONAPI_H

#include "Quantization.h"
#include "Rounding.h"

/*! Initializes float quantization.
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitFloat();

/*! Initializes int32 quantization.
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitInt32();

/*! Initializes symInt32 quantization.
 *
 * \param roundingMode: Rounding mode to be used
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitSymInt32(roundingMode_t roundingMode);

/*! SymInt32 with explicit qMaxBits.  The existing quantizationInitSymInt32(rm)
 *  hardcodes qMaxBits=16; this variant lets callers specify the active bit
 *  width for fixed-point arithmetic (e.g. 12 bits for tighter dynamic range,
 *  32 bits for full int32 range).
 *
 * \param roundingMode: Rounding mode to be used
 * \param qMaxBits: Active bit width for fixed-point arithmetic
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitSymInt32WithBits(roundingMode_t roundingMode, uint8_t qMaxBits);

/*! Sub-byte symmetric quantization with explicit bit width and rounding. */
quantization_t *quantizationInitSym(uint8_t qBits, roundingMode_t roundingMode);

/*! Initializes asym quantization.
 *
 * \param qBits: Number of bits for qMax
 * \param roundingMode: Rounding mode to be used
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitAsym(uint8_t qBits, roundingMode_t roundingMode);

/* Note: uses (void) explicitly; siblings use () for historical K&R-style; the
 * _Static_assert in UnitTestTensorApi pattern-matches (*)(void). */
/*! Initializes bool quantization.
 *
 * \returns Pointer to initialized quantization
 */
quantization_t *quantizationInitBool(void);

#endif // QUANTIZATIONAPI_H
