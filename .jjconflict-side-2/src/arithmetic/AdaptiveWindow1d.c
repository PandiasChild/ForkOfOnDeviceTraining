#define SOURCE_FILE "ODT_ADAPTIVE_WINDOW_1D"

#include "AdaptiveWindow1d.h"

#include "Common.h"

adaptiveWindow1d_t adaptiveWindow1dAt(size_t inputLength, size_t outputLength, size_t outputPos) {
    if (inputLength == 0 || outputLength == 0) {
        PRINT_ERROR("AdaptiveWindow1d: inputLength and outputLength must be >= 1");
        exit(1);
    }
    if (outputPos >= outputLength) {
        PRINT_ERROR("AdaptiveWindow1d: outputPos (%zu) out of range [0,%zu)", outputPos,
                    outputLength);
        exit(1);
    }
    size_t start = (outputPos * inputLength) / outputLength;
    // end = ceil((outputPos+1) * inputLength / outputLength), computed with the
    // +outputLength-1 integer-ceil idiom. No overflow guard: with realistic
    // tensor lengths (well under 2^16) the numerator stays far below SIZE_MAX
    // even on 32-bit MCUs. count = end - start is always >= 1 given outputPos <
    // outputLength (guarded above), so the caller never divides by zero.
    size_t end = ((outputPos + 1) * inputLength + outputLength - 1) / outputLength;
    return (adaptiveWindow1d_t){.start = start, .count = end - start};
}
