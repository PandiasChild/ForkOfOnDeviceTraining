#ifndef NPYLOADERAPI_H
#define NPYLOADERAPI_H

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

#endif //NPYLOADERAPI_H
