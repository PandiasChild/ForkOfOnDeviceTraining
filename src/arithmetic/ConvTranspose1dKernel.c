#define SOURCE_FILE "ODT_CONV_TRANSPOSE_1D_KERNEL"

#include "ConvTranspose1dKernel.h"

#include "Common.h"

void convTranspose1dKernelFloat32(tensor_t const *input, tensor_t const *weight,
                                  tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                  size_t outputPadding, tensor_t *output) {
    if (kernel->paddingType != VALID) {
        PRINT_ERROR("convTranspose1dKernelFloat32: only VALID paddingType "
                    "supported in Phase 1");
        exit(1);
    }

    size_t batch = input->shape->dimensions[0];
    size_t inChannels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outChannels = output->shape->dimensions[1];
    size_t outputLength = output->shape->dimensions[2];
    size_t kernelSize = weight->shape->dimensions[2];

    if (inChannels % groups != 0 || outChannels % groups != 0) {
        PRINT_ERROR("convTranspose1dKernelFloat32: groups (%zu) must divide "
                    "in_channels (%zu) and out_channels (%zu)",
                    groups, inChannels, outChannels);
        exit(1);
    }

    size_t expectedOutLen = (inputLength - 1) * kernel->stride +
                            kernel->dilation * (kernelSize - 1) + outputPadding + 1;
    if (expectedOutLen != outputLength) {
        PRINT_ERROR("convTranspose1dKernelFloat32: output_length mismatch "
                    "(expected=%zu, got=%zu)",
                    expectedOutLen, outputLength);
        exit(1);
    }

    if (outputPadding != 0 &&
        outputPadding >=
            ((kernel->stride > kernel->dilation) ? kernel->stride : kernel->dilation)) {
        PRINT_ERROR("convTranspose1dKernelFloat32: outputPadding (%zu) must be "
                    "< max(stride=%zu, dilation=%zu)",
                    outputPadding, kernel->stride, kernel->dilation);
        exit(1);
    }

    size_t inChPerGroup = inChannels / groups;
    size_t outChPerGroup = outChannels / groups;

    float const *xArr = (float const *)input->data;
    float const *wArr = (float const *)weight->data;
    float const *bArr = bias ? (float const *)bias->data : NULL;
    float *yArr = (float *)output->data;

    // zero output (scatter loop accumulates with +=)
    size_t totalOut = batch * outChannels * outputLength;
    for (size_t i = 0; i < totalOut; i++) {
        yArr[i] = 0.0f;
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            size_t inLo = g * inChPerGroup;
            size_t outLo = g * outChPerGroup;

            for (size_t icOffset = 0; icOffset < inChPerGroup; icOffset++) {
                size_t ic = inLo + icOffset;
                for (size_t inPos = 0; inPos < inputLength; inPos++) {
                    float xv = xArr[(b * inChannels + ic) * inputLength + inPos];
                    size_t outBase = inPos * kernel->stride;

                    for (size_t ocOffset = 0; ocOffset < outChPerGroup; ocOffset++) {
                        size_t oc = outLo + ocOffset;
                        for (size_t k = 0; k < kernelSize; k++) {
                            size_t outIdx = outBase + k * kernel->dilation;
                            if (outIdx >= outputLength) {
                                continue;
                            }

                            float wv = wArr[(ic * outChPerGroup + ocOffset) * kernelSize + k];
                            yArr[(b * outChannels + oc) * outputLength + outIdx] += xv * wv;
                        }
                    }
                }
            }
        }
    }

    // bias add (separate pass; keeps the scatter loop a pure +=)
    if (bArr) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t oc = 0; oc < outChannels; oc++) {
                for (size_t l = 0; l < outputLength; l++) {
                    yArr[(b * outChannels + oc) * outputLength + l] += bArr[oc];
                }
            }
        }
    }
}
