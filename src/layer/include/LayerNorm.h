#ifndef ODT_LAYERNORM_H
#define ODT_LAYERNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "Tensor.h"

typedef struct layer layer_t;

typedef struct layerNormConfig {
    parameter_t *gamma;        /* shape normalizedShape; init 1 */
    parameter_t *beta;         /* shape normalizedShape; init 0 */
    size_t *normalizedShape;   /* last-D logical dims */
    size_t numNormDims;        /* D */
    float eps;                 /* default 1e-5 */
    quantization_t *forwardQ;  /* forward dispatch + output quant */
    quantization_t *backwardQ; /* backward dispatch + propLoss(dx) quant */
    bool ownsQuantizations;    /* true → freeLayerNormLayer tears down forwardQ/backwardQ
                                * (Owning factory); false → caller owns them (Borrowing). */
} layerNormConfig_t;

void initLayerNormConfig(layerNormConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                         size_t *normalizedShape, size_t numNormDims, float eps,
                         quantization_t *forwardQ, quantization_t *backwardQ);

void layerNormForward(layer_t *layer, tensor_t *input, tensor_t *output);
void layerNormBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss);
void layerNormCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_LAYERNORM_H
