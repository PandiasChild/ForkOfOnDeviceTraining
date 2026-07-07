#ifndef ODT_DISTRIBUTIONS_H
#define ODT_DISTRIBUTIONS_H

#include <stddef.h>

typedef enum {
    ZEROS,
    ONES,
    UNIFORM,
    NORMAL,
    XAVIER_UNIFORM,
    XAVIER_NORMAL,
    KAIMING_UNIFORM,
    KAIMING_NORMAL
} distributionType_t;

/*! Carries both the kind of distribution and the kind-specific parameters
 * needed to draw values. Used by initDistribution() (TensorApi.h) and by
 * future layer factories that take an initialization recipe.
 *
 * Discriminant: `type`. Read the union member matching that discriminant.
 * ZEROS and ONES need no params; their union slot is unused.
 */
typedef struct {
    distributionType_t type;
    union {
        struct {
            float min, max;
        } uniform;
        struct {
            float mean, stddev;
        } normal;
        struct {
            float gain;
            size_t fanIn, fanOut;
        } xavier;
        struct {
            float gain;
            size_t fanMode;
        } kaiming;
    } params;
} distribution_t;

/*! Gets random value from normal distribution.\n
 * Uses the Box-Muller transform.
 *
 * @param mean: Mean to be used
 * @param standardDeviation: Standard deviation to be used
 *
 * @returns Float value
 */
float randomNormal(float mean, float standardDeviation);

/*! Gets random value from uniform distribution.
 *
 * \param min: Minimum value
 * \param max: Maximum value
 *
 * \returns Float value
 */
float randomUniform(float min, float max);

/*! Gets random value from Kaiming distribution using normal distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanMode: Number of input/output features
 *
 * @returns Float value
 */
float kaimingNormal(float gain, size_t fanMode);

/*! Gets random value from Kaiming distribution using uniform distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanMode: Number of input/output features
 *
 * @returns Float value
 */
float kaimingUniform(float gain, size_t fanMode);

/*! Gets random value from Xavier distribution using the normal distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanIn: Number of input features
 * @param fanOut: Number of output features
 *
 * @returns Float value
 */
float xavierNormal(float gain, size_t fanIn, size_t fanOut);

/*! Gets random value from Xavier distribution using the uniform distribution.
 *
 * @param gain Optional scaling factor for non-linearity
 * @param fanIn: Number of input features
 * @param fanOut: Number of output features
 *
 * @returns Float value
 */
float xavierUniform(float gain, size_t fanIn, size_t fanOut);

#endif // ODT_DISTRIBUTIONS_H
