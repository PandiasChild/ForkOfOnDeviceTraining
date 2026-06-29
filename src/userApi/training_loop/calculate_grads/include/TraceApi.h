#ifndef TRACE_API_H
#define TRACE_API_H

#include <stddef.h>

#include "Layer.h"
#include "Tensor.h"

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

/* tracedGrads(), traceModelWeights(), traceModelGrads() declarations are added
 * here in Tasks 2 and 3 — each declaration lands together with its failing test
 * (TDD-strict: no decl/stub before the test that drives it). */

#endif /* TRACE_API_H */
