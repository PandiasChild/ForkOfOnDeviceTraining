#define SOURCE_FILE "RNG"

#include "RNG.h"


static rng32_t rng;

static uint32_t rngNext(rng32_t *rng)
{
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static size_t rngBounded(rng32_t *rng, size_t bound)
{
    uint32_t x;
    uint32_t limit = UINT32_MAX - (UINT32_MAX % bound);

    do {
        x = rngNext(rng);
    } while (x >= limit);

    return x % bound;
}

void rngShuffleIndices(size_t *indices, size_t n, uint32_t seed)
{
    if (n < 2) return;

    rng32_t rng = { .state = seed ? seed : 1 };

    for (size_t i = n - 1; i > 0; --i) {
        size_t j = rngBounded(&rng, i + 1);

        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

void rngSetSeed(uint32_t seed) {
    rng.state = seed;
}

uint32_t rngGetSeed() {
    return rng.state;
}