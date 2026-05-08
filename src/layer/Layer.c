#define SOURCE_FILE "LAYER"

#include "Layer.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Flatten.h"
#include "Linear.h"
#include "Relu.h"
#include "Softmax.h"

layerFunctions_t layerFunctions[] = {
    [LINEAR] = {linearForward, linearBackward, linearCalcOutputShape},
    [RELU] = {reluForward, reluBackward, reluCalcOutputShape},
    [CONV1D] = {conv1dForward, conv1dBackward, conv1dCalcOutputShape},
    [CONV1D_TRANSPOSED] = {conv1dTransposedForward, conv1dTransposedBackward,
                           conv1dTransposedCalcOutputShape},
    [SOFTMAX] = {softmaxForward, softmaxBackward, softmaxCalcOutputShape},
    [FLATTEN] = {flattenForward, flattenBackward, flattenCalcOutputShape}};

void initLayer(layer_t *layer, layerType_t type, layerConfig_t *config) {
    layer->type = type;
    layer->config = config;
}
