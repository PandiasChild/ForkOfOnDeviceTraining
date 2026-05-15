#ifndef STATE_DICT_API_H
#define STATE_DICT_API_H

#include "Layer.h"
#include <stddef.h>

typedef struct stateDictEntry {
    const char *name; /* OPTIONAL — used only in error messages */
    float *weightData;
    float *biasData; /* NULL allowed iff the corresponding layer has no bias */
} stateDictEntry_t;

/*! Load weights/biases from entries[] into the parameter layers of model,
 *  in the order they appear. Param-less layers are skipped.
 *
 *  Errors (PRINT_ERROR + exit):
 *   - numEntries != count of param layers in model
 *   - any entry's weightData == NULL
 *   - bias presence in entry does not match bias presence in the corresponding layer
 *
 *  Error messages include entries[i].name if non-NULL, otherwise the
 *  param-layer index (0-based). */
void modelLoadStateDict(layer_t **model, size_t numLayers, stateDictEntry_t *entries,
                        size_t numEntries);

#endif /* STATE_DICT_API_H */
