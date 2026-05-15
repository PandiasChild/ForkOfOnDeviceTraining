#ifndef LAYER_COMMON_H
#define LAYER_COMMON_H

#include <assert.h>

/*! Bias presence tri-state for layer init structs.
 *  BIAS_DEFAULT lands at C99 zero-init; factories resolve it to the PyTorch
 *  default for that layer type (true for Conv / Linear).  */
typedef enum {
    BIAS_DEFAULT = 0,
    BIAS_TRUE = 1,
    BIAS_FALSE = 2,
} bias_t;

_Static_assert(BIAS_DEFAULT == 0,
               "BIAS_DEFAULT must be enum value 0 so .bias zero-init defaults to PyTorch default");

#endif /* LAYER_COMMON_H */
