#ifndef NPYLOADERAPI_H
#define NPYLOADERAPI_H

#include "Dataset.h"

/*!
 * Loads a .npy file as a tensor array.
 *
 * \param path: Path to .npy file
 * \return Pointer to tensor array
 */
tensorArray_t *npyLoad(char *path);

/*!
 * Gets a sample of given dataset.
 *
 * sample = item + label.
 *
 * \param dataset: Pointer to dataset
 * \param index: Index of sample
 * \return Pointer to sample
 */
sample_t *npyGetSample(dataset_t *dataset, size_t index);

/*!
 * Frees a tensorArray_t allocated by npyLoad: walks the inner array,
 * freeTensor()s every entry, then releases the array buffer and the
 * tensorArray_t struct itself. Stack- or static-allocated tensorArray_t
 * values must not be passed here.
 *
 * \param tensorArr: Pointer returned by npyLoad
 */
void freeTensorArray(tensorArray_t *tensorArr);

#endif // NPYLOADERAPI_H
