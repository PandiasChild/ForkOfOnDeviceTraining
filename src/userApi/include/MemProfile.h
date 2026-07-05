#ifndef MEM_PROFILE_H
#define MEM_PROFILE_H

#include <stddef.h>

/* Runs fn(arg) on a fresh pthread whose stack is a stackBytes region painted
 * with a sentinel, then returns the peak bytes touched (stack high-water).
 * Host-only measurement apparatus; the region uses raw malloc, NOT
 * reserveMemory, so it never enters the heap counter. */
size_t measurePeakStackBytes(void (*fn)(void *), void *arg, size_t stackBytes);

#endif
