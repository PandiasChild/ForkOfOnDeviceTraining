#define SOURCE_FILE "RNG"

#include "RNG.h"

static rng32_t rng = {.state = 1};

static uint32_t rngNext(rng32_t *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static size_t rngBounded(rng32_t *rng, size_t bound) {
    uint32_t x;
    uint32_t limit = UINT32_MAX - (UINT32_MAX % bound);

    do {
        x = rngNext(rng);
    } while (x >= limit);

    return x % bound;
}

void rngShuffleIndices(size_t *indices, size_t n) {
    if (n < 2) {
        return;
    }

    for (size_t i = n - 1; i > 0; --i) {
        size_t j = rngBounded(&rng, i + 1);

        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

void rngSetSeed(uint32_t seed) {
    rng.state = seed ? seed : UINT32_MAX;
}

uint32_t rngGetSeed(void) {
    return rng.state;
}

float rngNextFloatCtx(rng32_t *rngCtx) {
    /* 0x1p24f == 2^24 as a float literal: value-identical to the former
     * (float)(1 << 24), minus the (16-bit-int-only) UB of a wide untyped
     * shift -- the repo's one such site (PR #366 review). */
    return (float)(rngNext(rngCtx) >> 8) / 0x1p24f;
}

float rngNextFloat(void) {
    return rngNextFloatCtx(&rng);
}
