#ifndef SGD_API_H
#define SGD_API_H

#include "Sgd.h"

/*! Builds an SGD(+momentum) optimizer over a model's trainable parameters.
 *
 * Each parameter's momentum state buffer is allocated at the PARAMETER'S
 * SHAPE but with its OWN quantization config (`momentumQuant`, deep-cloned
 * per state via getQLike) -- decoupled from the parameter's own dtype.
 * Pass quantizationInitFloat() for the conventional FLOAT32 accumulator
 * (the sensible default for every existing caller); a caller wanting e.g. a
 * SYM momentum accumulator may pass a SYM quantization instead, subject to
 * getQLike's supported-type set (BOOL is not supported for state clones).
 *
 * \param momentumQuant: Quantization template for momentum state buffers
 *        (caller-owned; not consumed -- cloned once per state via getQLike).
 */
optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             layer_t **model, size_t sizeModel, quantization_t *momentumQuant);

void freeOptimSgdM(optimizer_t *sgdM);

#endif // SGD_API_H
