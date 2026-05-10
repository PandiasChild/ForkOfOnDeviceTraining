#define SOURCE_FILE "verify_xorshift32_harness"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "RNG.h"

/* Compile alongside src/rng/RNG.c. Usage: ./harness <n> <seed>
 * Emits the shuffled permutation of [0, n) to stdout, space-separated,
 * one trailing newline.
 */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <n> <seed>\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)strtoull(argv[1], NULL, 10);
    uint32_t seed = (uint32_t)strtoul(argv[2], NULL, 10);

    size_t *indices = malloc(n * sizeof(*indices));
    if (!indices) {
        return 2;
    }
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    rngSetSeed(seed);
    rngShuffleIndices(indices, n);

    for (size_t i = 0; i < n; ++i) {
        printf("%zu%c", indices[i], (i + 1 == n) ? '\n' : ' ');
    }
    free(indices);
    return 0;
}
