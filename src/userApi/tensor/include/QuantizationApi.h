#ifndef QUANTIZATIONAPI_H
#define QUANTIZATIONAPI_H

#include "Quantization.h"
#include "Rounding.h"
/*
 * @brief The QuantizationAPI is INITIALIZATION ONLY.
 * @IMPORTANT use Tensor API for any deinitialization (e.g. freeQuantization).
 *
 * This API is exclusively responsible for initializing quantization instances
 * and their associated configuration.
 *
 * Deinitialization of quantization resources is not handled here. All cleanup
 * operations, including deinitialization of quantization objects and related
 * resources, are managed by the Tensor API.
 */



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

/*!
 * @brief Creates and initializes a symmetric delta quantization instance.
 *
 * This function allocates memory for a quantization structure and its associated
 * symmetric delta configuration, initializes them with the specified parameters,
 * and returns the resulting quantization instance.
 *
 * @param qBits[in] Number of fractional bits.
 * @param roundingMode[in] Specifies the rounding mode used during quantization.
 * @param deltabits[in] Number of fractional bits used for delta encoding.
 *
 * @returns Pointer to the initialized symmetric delta quantization instance.
 */
quantization_t *quantizationInitSymQDelta(uint8_t qBits, roundingMode_t roundingMode, uint8_t deltabits);


#endif // QUANTIZATIONAPI_H
