#define SOURCE_FILE "LAYER_QUANT"

#include "LayerQuant.h"

void layerQuantInitUniform(layerQuant_t *lq, quantization_t *q) {
    lq->forwardMath = q;
    lq->backwardMath = q;
    lq->weightStorage = q;
    lq->biasStorage = q;
}
