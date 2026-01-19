#ifndef RNG_H
#define RNG_H

#include <stddef.h>
#include <stdint.h>


typedef struct {
    uint32_t state;
} rng32_t;

void rngShuffleIndices(size_t *indices, size_t n, uint32_t seed);

void rngSetSeed(uint32_t seed);

uint32_t rngGetSeed();

#endif //RNG_H
