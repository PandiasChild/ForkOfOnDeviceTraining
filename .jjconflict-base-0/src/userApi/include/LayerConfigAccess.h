#ifndef ODT_LAYER_CONFIG_ACCESS_H
#define ODT_LAYER_CONFIG_ACCESS_H

#include "ArithmeticType.h"
#include "Layer.h"

/* Uniform per-layer-type accessors (design spec 2026-07-02
 * arithmetic-type-split, D3/D4). The 12-case layerType_t switch lives here
 * and nowhere else — every consumer that needs a layer's produced-wire
 * storage config or declared forward arithmetic goes through these three
 * functions instead of re-deriving its own switch. */

/* Produced forward-wire storage config (dtype + qConfig for the layer's
 * output tensor). NULL for Flatten — it has no per-layer quantization;
 * callers fall back to the upstream tensor's own quantization (passthrough),
 * exactly as the pre-existing FLATTEN handling already did. */
quantization_t *layerOutputQ(layer_t *layer);

/* Producer's declared backward config for the dx wire it emits (#221). NULL
 * for Flatten -> passthrough of the upstream dtype (callers already fall
 * back to the upstream tensor's quantization, e.g. initGradTensor). */
quantization_t *backwardWireQ(layer_t *layer);

/* Declared forward compute representation. Flatten and Quantization have no
 * consumed arithmetic (D4 — Quantization is a pure conversion node) ->
 * {ARITH_FLOAT32, HALF_AWAY}, matching arithmeticFromQuantizationOrDefault(NULL). */
arithmetic_t layerForwardMath(layer_t *layer);

#endif // ODT_LAYER_CONFIG_ACCESS_H
