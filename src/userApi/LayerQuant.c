#define SOURCE_FILE "LAYER_QUANT"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "LayerQuant.h"
#include "StorageApi.h"

void layerQuantInitUniform(layerQuant_t *lq, quantization_t *q) {
    lq->forwardMath = q;
    lq->backwardMath = q;
    lq->weightStorage = q;
    lq->biasStorage = q;
}

quantization_t *deepCopyQuantization(quantization_t *src) {
    if (src == NULL) {
        return NULL;
    }

    quantization_t *dst = reserveMemory(sizeof(quantization_t));
    dst->type = src->type;

    size_t cfgSize = 0;
    switch (src->type) {
    case FLOAT32:
        cfgSize = 0;
        break;
    case INT32:
        cfgSize = 0;
        break;
    case BOOL:
        cfgSize = 0;
        break;
    case SYM_INT32:
        cfgSize = sizeof(symInt32QConfig_t);
        break;
    case SYM:
        cfgSize = sizeof(symQConfig_t);
        break;
    case ASYM:
        cfgSize = sizeof(asymQConfig_t);
        break;
    default:
        PRINT_ERROR("deepCopyQuantization: unknown quantization type %d", (int)src->type);
        exit(1);
    }

    if (cfgSize == 0) {
        dst->qConfig = NULL;
    } else {
        dst->qConfig = reserveMemory(cfgSize);
        memcpy(dst->qConfig, src->qConfig, cfgSize);
    }
    return dst;
}
