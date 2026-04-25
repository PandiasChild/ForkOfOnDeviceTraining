#define SOURCE_FILE "STORAGE_API"

#include <stdlib.h>

#include "StorageApi.h"

void *reserveMemory(size_t numberOfBytes) {
    return calloc(1, numberOfBytes);
}

void freeReservedMemory(void *ptr) {
    free(ptr);
}
