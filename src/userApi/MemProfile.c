#define SOURCE_FILE "MEM_PROFILE"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MemProfile.h"

#define MEM_STACK_PAINT 0xA5u

typedef struct stackThunk {
    void (*fn)(void *);
    void *arg;
} stackThunk_t;

static void *stackThunkRunner(void *p) {
    stackThunk_t *t = (stackThunk_t *)p;
    t->fn(t->arg);
    return NULL;
}

size_t measurePeakStackBytes(void (*fn)(void *), void *arg, size_t stackBytes) {
    /* raw malloc: measurement apparatus, deliberately not reserveMemory */
    unsigned char *region = malloc(stackBytes);
    if (region == NULL) {
        fprintf(stderr, "MEM_PROFILE: malloc for the stack region failed — failing loud\n");
        exit(1);
    }
    memset(region, (int)MEM_STACK_PAINT, stackBytes);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    /* Fail loud on any pthread error: a rejected stack region (attr_setstack) would run the
     * workload on the DEFAULT stack, leaving `region` fully painted -> a silent used==0
     * measurement; a failed create would leave `th` uninitialized -> UB in join. Both are
     * worse than a crash for a measurement apparatus. */
    if (pthread_attr_setstack(&attr, region, stackBytes) != 0) {
        fprintf(stderr, "MEM_PROFILE: pthread_attr_setstack rejected the stack region "
                        "— failing loud (measurement would be wrong)\n");
        exit(1);
    }

    stackThunk_t thunk = {.fn = fn, .arg = arg};
    pthread_t th;
    if (pthread_create(&th, &attr, stackThunkRunner, &thunk) != 0) {
        fprintf(stderr, "MEM_PROFILE: pthread_create failed — failing loud\n");
        exit(1);
    }
    if (pthread_join(th, NULL) != 0) {
        fprintf(stderr, "MEM_PROFILE: pthread_join failed — failing loud\n");
        exit(1);
    }
    pthread_attr_destroy(&attr);

    /* Stack grows down from the high end of [region, region+stackBytes). The
     * untouched (still-painted) bytes are the low prefix; scan from the low end
     * for the first touched byte. used = stackBytes - firstTouchedIndex. */
    size_t firstTouched = 0;
    while (firstTouched < stackBytes && region[firstTouched] == MEM_STACK_PAINT) {
        firstTouched++;
    }
    size_t used = stackBytes - firstTouched;
    free(region);
    return used;
}
