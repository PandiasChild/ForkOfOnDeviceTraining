#define SOURCE_FILE "BERNOULLI"

#include "Bernoulli.h"

#include "RNG.h"
#include "Tensor.h"

void bernoulliFillMaskReference(tensor_t *mask, float probTrue) {
    size_t numberOfElements = calcNumberOfElementsByTensor(mask);
    for (size_t i = 0; i < numberOfElements; i++) {
        tensorBoolSet(mask, i, rngNextFloat() < probTrue);
    }
}

static bernoulliFillMaskFn_t activeFn = bernoulliFillMaskReference;

void bernoulliFillMask(tensor_t *mask, float probTrue) {
    activeFn(mask, probTrue);
}

void bernoulliSetFillMaskFn(bernoulliFillMaskFn_t fn) {
    activeFn = fn ? fn : bernoulliFillMaskReference;
}

bernoulliFillMaskFn_t bernoulliGetFillMaskFn(void) {
    return activeFn;
}
