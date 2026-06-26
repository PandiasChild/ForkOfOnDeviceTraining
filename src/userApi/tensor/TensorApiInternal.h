#ifndef TENSORAPIINTERNAL_H
#define TENSORAPIINTERNAL_H

/*! Initializes tensor with given int32 quantization.
 *
 * Used for tensorInitInt32, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to int32 data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQInt32(int32_t *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity);
/*! Initializes tensor with given float quantization.
 *
 * Used for tensorInitFloat, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to float data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQFloat(float *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity);
/*! Initializes tensor with given symInt32 quantization.
 *
 * Used for tensorInitSymInt32, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to float data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQSymInt32(float *data, size_t *dims, size_t numberOfDims,
                                         quantization_t *quantization, sparsity_t *sparsity);

/*! Initializes tensor with given asym quantization.
 *
 * Used for tensorInitAsym, tensorInit and tensorInitWithDistribution.
 *
 * \param data: Pointer to float data array
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Pointer to quantization
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer initialized tensor
 */
static tensor_t *initTensorWithQAsym(float *data, size_t *dims, size_t numberOfDims,
                                     quantization_t *quantization, sparsity_t *sparsity);
/*! Initializes tensor with symmetric delta quantization.
 *
 * Used for tensorInitSymQDelta, tensorInit and tensorInitWithDistribution.
 *
 * This function creates an intermediate float32 tensor, converts it to a symmetric
 * int32 representation, and finally encodes it into a delta tensor.
 * Memory for the final representation is allocated based on the number
 * of elements and delta-bit width.
 *
 * @param data[in] Pointer to float data array
 * @param dims[in] Dimensions of tensor
 * @param numberOfDims[in] Number of dimensions
 * @param quantization[in] Pointer to symmetric delta quantization configuration
 * @param sparsity[in] Pointer to sparsity configuration
 *
 * @returns Pointer to initialized delta tensor
 */
static tensor_t *initTensorWithSymQDelta(float *data, size_t *dims, size_t numberOfDims,
                                         quantization_t *quantization, sparsity_t *sparsity);

/*! Frees only the pointer of a tensor.
 *
 * Only used in freeTensor.
 *
 * \param tensor: Pointer to tensor
 */
static void freeTensorPointer(tensor_t *tensor);

#endif // TENSORAPIINTERNAL_H
