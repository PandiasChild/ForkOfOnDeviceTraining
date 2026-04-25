#ifndef TENSOR_H
#define TENSOR_H

#include "Distributions.h"
#include "Tensor.h"

/*! Initializes int32 tensor with given data and shape.
 *
 * \param data: Data of tensor
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized tensor
 */
tensor_t *tensorInitInt32(int32_t *data, size_t *dims, size_t numberOfDims, sparsity_t *sparsity);
/*! Initializes float tensor with given data and shape.
 *
 * \param data: Data of tensor
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized tensor
 */
tensor_t *tensorInitFloat(float *data, size_t *dims, size_t numberOfDims, sparsity_t *sparsity);
/*! Initializes symInt32 tensor with given data and shape.
 *
 * \param data: Data of tensor
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param roundingMode: Rounding mode to be used
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized tensor
 */
tensor_t *tensorInitSymInt32(float *data, size_t *dims, size_t numberOfDims,
                             roundingMode_t roundingMode, sparsity_t *sparsity);
/*! Initializes asym tensor with given data and shape.
 *
 * \param data: Data of tensor
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param qBits: Number of bits for qMax
 * \param roundingMode: Rounding mode to be used
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized tensor
 */
tensor_t *tensorInitAsym(float *data, size_t *dims, size_t numberOfDims, uint8_t qBits,
                         roundingMode_t roundingMode, sparsity_t *sparsity);
/*! Initializes tensor to match given quantization.
 *
 * \param data: Data of tensor
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Quantization to be used
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized tensor
 */
tensor_t *tensorInit(float *data, size_t *dims, size_t numberOfDims, quantization_t *quantization,
                     sparsity_t *sparsity);
/*! Initializes tensor with distribution to match given quantization.
 *
 * \param distributionType: Type of distribution to be used
 * \param data: Data of tensor
 * \param dims: Dimensions of tensor
 * \param numberOfDims: Number of dimensions
 * \param quantization: Quantization to be used
 * \param sparsity: sparsity_t of tensor
 * \param inputFeatures: Number of input features
 * \param outputFeatures: Number of output features
 *
 * \returns Pointer to initialized tensor
 */
tensor_t *tensorInitWithDistribution(distributionType_t distributionType, float *data, size_t *dims,
                                     size_t numberOfDims, quantization_t *quantization,
                                     sparsity_t *sparsity, size_t inputFeatures,
                                     size_t outputFeatures);

/*! Initializes a tensor that owns its data buffer.
 *
 * The tensor takes ownership of `shape`, `quantization`, and `sparsity` (if non-NULL).
 * The data buffer is allocated by initTensor itself (size derived from shape and
 * quantization) and is zero-initialized.
 *
 * After this call, freeTensor(t) is unconditionally safe and frees all four
 * ownership domains (data, shape, quantization, sparsity).
 *
 * \param shape: Caller-allocated shape; ownership transferred to the tensor.
 * \param quantization: Caller-allocated quantization; ownership transferred.
 * \param sparsity: Optional; ownership transferred (NULL means no sparsity).
 *
 * \returns Pointer to an initialized tensor with own-allocated zero data.
 */
tensor_t *initTensor(shape_t *shape, quantization_t *quantization, sparsity_t *sparsity);

/*! Initializes int32 gradient tensor to match given param tensor.
 *
 * \param param: Pointer to param tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitInt32(tensor_t *param, sparsity_t *sparsity);
/*! Initializes float gradient tensor to match given param tensor.
 *
 * \param param: Pointer to param tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitFloat(tensor_t *param, sparsity_t *sparsity);
/*! Initializes symInt32 gradient tensor to match given param tensor.
 *
 * \param param: Pointer to param tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitSymInt32(tensor_t *param, roundingMode_t roundingMode, sparsity_t *sparsity);
/*! Initializes asym gradient tensor to match given param tensor.
 *
 * \param param: Pointer to param tensor
 * \param sparsity: sparsity_t of tensor
 *
 * \returns Pointer to initialized gradient tensor
 */
tensor_t *gradInitAsym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                       sparsity_t *sparsity);

/*! Initializes parameter with given param and graident tensor.
 *
 * \param param: Pointer to param tensor
 * \param grad: Pointer to gradient tensor
 *
 * \returns Pointer to initialized parameter
 */
parameter_t *parameterInit(tensor_t *param, tensor_t *grad);

/*! Gets quantization that matches a given quantization.
 *
 * \param quantization: Pointer to quantization
 *
 * \returns Pointer to quantization with matching type and config
 */
quantization_t *getQLike(quantization_t *quantization);
/*! Gets tensor that matches a given tensor.
 *
 * \param tensor: Pointer to tensor
 *
 * \returns Pointer to tensor with matching shape, sparsity and quantization
 */
tensor_t *getTensorLike(tensor_t *tensor);
/*! Gets shape that matches a given shape.
 *
 * \param shape: Pointer to shape
 *
 * \returns Pointer to shape with matching numberOfDims, dims, and orderOfDims
 */
shape_t *getShapeLike(shape_t *shape);
/*! Gets data array by quantization and number of values.
 *
 * \param quantization: Pointer to quantization
 * \param numberOfValues: Number of values
 *
 * \returns Pointer to data array with matching size
 */
uint8_t *getDataLike(quantization_t *quantization, size_t numberOfValues);
/*! Gets sparsity that matches a given sparsity.
 *
 * \param sparsity: Pointer to sparsity
 *
 * \returns Pointer to sparsity
 */
sparsity_t *getSparsityLike(sparsity_t *sparsity);

// IMPORTANT: these are needed for trainingAPI.c to free gradTensors without freeing the actual
// Pointer
/*! Frees data of tensor.
 *
 * \param tensor: Pointer to tensor
 */
void freeData(tensor_t *tensor);
/*! Frees shape of tensor.
 *
 * \param shape: Pointer to shape
 */
void freeShape(shape_t *shape);
/*! Frees quantization of tensor.
 *
 * \param quantization: Pointer to quantization
 */
void freeQuantization(quantization_t *quantization);

/*! Frees tensor.
 *
 * \param tensor: Pointer to tensor
 */
void freeTensor(tensor_t *tensor);
/*! Frees parameter.
 *
 * \param parameter: Pointer to parameter
 */
void freeParameter(parameter_t *parameter);

#endif // TENSOR_H
