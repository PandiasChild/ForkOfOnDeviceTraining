#define SOURCE_FILE "OPTIMIZER_API"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "OptimizerApi.h"
#include "Quantization.h"
#include "Tensor.h"

void scaleOptimizerGradients(optimizer_t *optimizer, float factor) {
    /* Validation: warn (currently via PRINT_ERROR — see #151 for unified
     * warn/assert macros) on non-positive or non-finite factor. */
    if (!(factor > 0.0f && isfinite(factor))) {
        PRINT_ERROR("scaleOptimizerGradients: suspicious factor %f "
                    "(expected positive, finite)",
                    (double)factor);
    }

    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];

        switch (param->grad->quantization->type) {
        case FLOAT32: {
            size_t numberOfValues = calcNumberOfElementsByParameter(param);
            float *gradArr = (float *)param->grad->data;
            for (size_t j = 0; j < numberOfValues; j++) {
                gradArr[j] *= factor;
            }
            break;
        }
        case SYM_INT32: {
            /* float_value = int32_value * scale ⇒ multiplicative scaling can
             * be absorbed into the per-tensor scale, leaving the int32 storage
             * untouched. O(1) and avoids quantization round-trip loss. */
            symInt32QConfig_t *gradQ = param->grad->quantization->qConfig;
            gradQ->scale *= factor;
            break;
        }
        case SYM: {
            /* Packed-SYM dequant (mantissa * scale) is linear in scale exactly
             * like the SYM_INT32 case above — fold the factor into the
             * per-tensor scale, packed codes untouched (O(1), exact). */
            symQConfig_t *gradQ = param->grad->quantization->qConfig;
            gradQ->scale *= factor;
            break;
        }
        case ASYM: {
            /* Packed-ASYM dequant is (code + zeroPoint) * scale: still linear
             * in scale, so the fold is exact the same way; zeroPoint is an
             * additive offset on the code axis and is untouched. */
            asymQConfig_t *gradQ = param->grad->quantization->qConfig;
            gradQ->scale *= factor;
            break;
        }
        default:
            PRINT_ERROR("scaleOptimizerGradients: unsupported gradient qtype "
                        "(accepted: FLOAT32, SYM_INT32, SYM, ASYM; INT32/BOOL "
                        "grad storage remains unsupported, #261)");
            exit(1);
        }
    }
}
