#define SOURCE_FILE "FLATTEN_API"

#include "FlattenApi.h"
#include "StorageApi.h"
#include "Common.h"
#include "stdlib.h"

layer_t *flattenLayerInit(void) {
    layer_t *flattenLayer = reserveMemory(sizeof(layer_t));
	if(flattenLayer == NULL){
		PRINT_ERROR("Memory Allocation Failed");
		exit(1);
	}
    flattenLayer->type = FLATTEN;
    // Load-bearing: initLayerOutputs' FLATTEN case never reads config.
    flattenLayer->config = NULL;
    return flattenLayer;
}

void freeFlattenLayer(layer_t *flattenLayer) {
    freeReservedMemory(flattenLayer);
}
