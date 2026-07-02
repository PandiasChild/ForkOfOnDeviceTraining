#define SOURCE_FILE "MODEL_VALIDATION_API"

#include <stdbool.h>
#include <stddef.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "ModelValidationApi.h"

/* True for accumulator-range producers (layers whose SYM forward emits raw
 * matmul/post-affine mantissas beyond the int16 norm) that ALSO declared
 * ARITH_SYM_INT32 as their forward compute representation; false for every
 * other layer type or arithmetic. */
static bool isAccumulatorRangeSymProducer(layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
    case LAYERNORM:
    case CONV1D:
    case CONV1D_TRANSPOSED:
        return layerForwardMath(layer).type == ARITH_SYM_INT32;
    default:
        return false;
    }
}

bool validateModelQuantization(layer_t **model, size_t modelSize) {
    if (model == NULL) {
        PRINT_ERROR("validateModelQuantization: model is NULL");
        return false;
    }
    bool valid = true;
    for (size_t i = 0; i < modelSize; i++) {
        if (model[i] == NULL) {
            PRINT_ERROR("validateModelQuantization: model[%zu] is NULL", i);
            valid = false;
            continue;
        }
        if (!isAccumulatorRangeSymProducer(model[i])) {
            continue; /* not a SYM accumulator-range producer */
        }
        if (i + 1 == modelSize) {
            continue; /* producer in last position: loss boundary, allowed */
        }
        if (model[i + 1] == NULL || model[i + 1]->type != QUANTIZATION) {
            PRINT_ERROR("validateModelQuantization: layer %zu (type %d) emits SYM_INT32 "
                        "accumulator-range output but layer %zu (type %d) is not a "
                        "QUANTIZATION layer — insert a Quant layer to restore the int16 "
                        "inter-layer contract",
                        i, (int)model[i]->type, i + 1, model[i + 1] ? (int)model[i + 1]->type : -1);
            valid = false;
        }
    }
    return valid;
}
