#ifndef NPYLOADERAPI_H
#define NPYLOADERAPI_H

tensorArray_t *npyLoad(char *path);

sample_t *npyGetSample(dataset_t *dataset, size_t id);

#endif //NPYLOADERAPI_H
