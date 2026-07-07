#ifndef EXAMPLES_SHARED_NPY_WRITER_H
#define EXAMPLES_SHARED_NPY_WRITER_H

#include <stddef.h>
#include <stdint.h>

/* Write a contiguous row-major float32 array as a .npy v1.0 file.
 *
 * shape: array of dimension lengths
 * ndim:  number of dimensions
 *
 * Returns 0 on success, non-zero on I/O failure.
 */
int npyWriteFloat32(const char *path, const float *data, const size_t *shape, size_t ndim);

/* Same, for int32. */
int npyWriteInt32(const char *path, const int32_t *data, const size_t *shape, size_t ndim);

#endif
