#ifndef TRACE_API_H
#define TRACE_API_H

#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Tensor.h"
#include "TrainingLoopApi.h"

/*! Fired at every probe point of one traced training step. The framework hands
 *  a tensor to the sink and never opens a file; the sink (above the src/
 *  boundary) decides what to do with it.
 *
 *  - layerIdx:   model index of the layer; for the loss gradient, == modelSize.
 *  - layerType:  the layer's type (for naming / dtype decisions).
 *  - phase:      "fwd" | "agrad" | "lossgrad" for tracedGrads (Task 2);
 *                "<tag>.weight" / "<tag>.bias" for traceModelWeights/Grads (Task 3).
 *  - tensor:     borrowed; valid only for the duration of the call. */
typedef void (*traceSink_t)(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                            tensor_t *tensor);

/*! Same forward+backward as calculateGradsSequential, but fires `sink` after
 *  each layer's forward ("fwd"), after the loss backward ("lossgrad",
 *  layerIdx == modelSize), and after each layer's backward ("agrad"). */
trainingStats_t *tracedGrads(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                             reduction_t forwardReduction, tensor_t *input, tensor_t *label,
                             traceSink_t sink, void *ctx);

/* traceModelWeights(), traceModelGrads() declarations are added here in Task 3
 * — each declaration lands together with its failing test
 * (TDD-strict: no decl/stub before the test that drives it). */

#endif /* TRACE_API_H */
