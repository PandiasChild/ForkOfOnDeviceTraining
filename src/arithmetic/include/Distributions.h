#ifndef ODT_DISTRIBUTIONS_H
#define ODT_DISTRIBUTIONS_H

#include <stddef.h>

typedef enum
{
    ZEROS,
    ONES,
    UNIFORM,
    NORMAL,
    XAVIER_UNIFORM,
    XAVIER_NORMAL,
    KAIMING_UNIFORM,
    KAIMING_NORMAL
} distributionType_t;

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
