#ifndef DESERIALIZEINTERNAL_H
#define DESERIALIZEINTERNAL_H

#include <stdio.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

/*! Deserializes shape of tensor from given file (u32 LE fields, #370).
 *  Fails fast if the file rank does not match the skeleton's rank — the
 *  skeleton's dimension arrays were sized by the build-time rank.
 *
 * \param shape: Pointer to shape to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeShape(shape_t *shape, FILE *f);

/*! Deserializes quantization of tensor from given file.
 *
 * \param q: Pointer to quantization to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeQuantization(quantization_t *q, FILE *f);

/*! Deserializes a declared compute representation from given file: u8 type +
 *  u8 roundingMode.
 *
 * \param arithmetic: Pointer to arithmetic_t to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeArithmetic(arithmetic_t *arithmetic, FILE *f);

/*! Deserializes kernel geometry (all kernel_t fields) from given file.
 *  Mirrors serializeKernel.
 *
 * \param kernel: Pointer to kernel_t to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeKernel(kernel_t *kernel, FILE *f);

/*! Deserializes quantization config of tensor from given file.
 *
 * \param q: Pointer to quantization to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeQConfig(quantization_t *q, FILE *f);

/*! Not implemented yet!
 */
static void deserializeSparsity();

/*! Deserializes layer from given file.
 *
 * \param layer: Pointer to layer to deserialize into
 * \param f: Pointer of file to deserialize from
 */
static void deserializeLayer(layer_t *layer, FILE *f);

#endif // DESERIALIZEINTERNAL_H
