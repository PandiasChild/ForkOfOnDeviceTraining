#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include "LossFunction.h"
#include "Tensor.h"

float crossEntropyForwardFloat(tensor_t *softmaxOutput, tensor_t *distribution);

void crossEntropySoftmaxBackward(tensor_t *softmaxOutput, tensor_t *distribution, tensor_t *loss,
                                 size_t batchSize, reduction_t reduction);

#endif // CROSSENTROPY_H
