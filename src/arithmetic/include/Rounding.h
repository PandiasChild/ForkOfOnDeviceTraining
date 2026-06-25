#ifndef ROUNDING_H
#define ROUNDING_H
#include <stdint.h>

/*! @brief Describes rounding
 * HALF_AWAY = round half away from zero (C round(), C17 7.12.9.6)
 * SR_HALF_AWAY = stochastic rounding: uniform jitter in [-0.5, 0.5) is added
 *                before rounding half away from zero
 */
typedef enum roundingMode { HALF_AWAY = 0, SR_HALF_AWAY = 1 } roundingMode_t;

/**
 * Rounds a floating-point value to the nearest integer using half-to-even rule.
 * When the fractional part is exactly 0.5, the result is rounded to the nearest even integer.
 *
 * @return The rounded integer value.
 */
int32_t roundByMode(float input, roundingMode_t roundingMode);

/*! @brief Rescale a SYM_INT32 parameter mantissa from its own scale into an
 * accumulator's product-scale, returning the int32 seed to add into the
 * integer accumulator (forward bias seed for Matmul/Conv, LayerNorm affine
 * beta seed).
 *
 * The float->int32 cast is data-dependent and is UB on overflow
 * (C17 6.3.1.4, #189). When compiled with -DODT_SEED_GUARD (unit-test/CI
 * builds; off for release/MCU) the cast is guarded: it fails fast when the
 * rescaled value leaves no int32 headroom for one worst-case int16xint16
 * product (32768*32767). NaN-robust via !(x <= T): a 0*inf=NaN from an
 * underflowed accumulator scale is caught, not bypassed. Release builds run
 * the raw cast (UBSan #204 covers occurrences there). */
int32_t rescaleIntoAccumulatorScale(int32_t paramQ, float paramScale, float accumulatorScale,
                                    roundingMode_t roundingMode);

/**
 * Saturates a floating-point value into the range [min, max].
 *
 * Unlike naive clamping, this operation performs saturation, meaning that
 * values below min are set to min and values above max are set to max,
 * without any truncation artifacts beyond range limiting.
 *
 * @param input The value to saturate.
 * @param min The lower saturation bound.
 * @param max The upper saturation bound.
 * @return The saturated value.
 */
float clamp(float input, float min, float max);

/**
 * Performs a saturating subtraction operation.
 * The result is clamped to the range [min, max] if overflow/underflow would occur.
 *
 * @param minuend Minuend value.
 * @param subtrahend Subtrahend value.
 * @param min Lower saturation bound.
 * @param max Upper saturation bound.
 * @return Saturated difference.
 */
int32_t saturatingSubstraction(int32_t minuend, int32_t subtrahend, int32_t min, int32_t int32_t);

#endif // ROUNDING_H
