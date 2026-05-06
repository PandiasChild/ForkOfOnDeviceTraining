#ifndef OPTIMIZER_API_H
#define OPTIMIZER_API_H

#include "Optimizer.h"

optimizer_t *optimizerInit(optimizerType_t type, void *optimizerConfig);

/* Scales every gradient field of every parameter tracked by the optimizer
 * by the given factor in-place.
 *
 * Iterates the same parameter list that sgdZeroGrad uses
 * (optimizer->parameter[0..sizeStates-1]).
 *
 * Caller is responsible for not calling with factor == 1.0 (no-op);
 * factor must be positive and finite — non-positive or non-finite values
 * are logged via PRINT_ERROR (the closest existing tool; #151 will replace
 * this with a proper PRINT_WARN). */
void scaleOptimizerGradients(optimizer_t *optimizer, float factor);

#endif // OPTIMIZER_API_H
