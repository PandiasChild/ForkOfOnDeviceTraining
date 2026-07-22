#define SOURCE_FILE "ROUNDING"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "RNG.h"
#include "Rounding.h"

// C round(): rounds half away from zero (C17 7.12.9.6)
int32_t roundHalfAway(float input) {
    return round(input);
}

float randfloat() {
    return rngNextFloat();
}

int32_t roundSRHalfAway(const float input) {
    return roundHalfAway(input + randfloat() - 0.5f);
}

int32_t roundByMode(const float input, const roundingMode_t roundingMode) {
    switch (roundingMode) {
    case HALF_AWAY:
        return roundHalfAway(input);
    case SR_HALF_AWAY:
        return roundSRHalfAway(input);
    }
    return 0;
}
//TODO: Jasmin - ggf rückgängig machen!
int32_t rescaleIntoAccumulatorScale(int32_t paramQ, float paramScale, float accumulatorScale,
                                    roundingMode_t roundingMode) {
    float rescaled = (float)paramQ * paramScale / accumulatorScale;
#ifdef ODT_SEED_GUARD
    /* NaN can only occur here from 0*inf (accumulatorScale underflowed to 0
     * combined with paramQ == 0) or inf/inf — this is a genuine error, not
     * a saturable overflow, so it is still fatal. */
    if (isnan(rescaled)) {
        PRINT_ERROR("rescaleIntoAccumulatorScale: param scale incompatible with accumulator "
                "scale — result is NaN (#189) [paramQ=%d, paramScale=%f, accumulatorScale=%f]",
                paramQ, paramScale, accumulatorScale);
        exit(1);
    }

    /* Reserve the real worst int16 product 32768*32767 (the -32768 case),
     * not qMax*qMax, for whatever gets added to this seed afterwards. */
    const float kSeedLimit = 2147483647.0f - 32768.0f * 32767.0f;

    if (rescaled > kSeedLimit) {
        return (int32_t)kSeedLimit;
    }
    if (rescaled < -kSeedLimit) {
        return -(int32_t)kSeedLimit;
    }
#endif
    return roundByMode(rescaled, roundingMode);
}



int32_t clampInt32(int32_t input, int32_t min, int32_t max) {
    if (input < min) {
        return min;
    }
    if (input > max) {
        return max;
    }
    return input;
}

float clamp(float input, float min, float max) {
    if (input < min) {
        return min;
    }
    if (input > max) {
        return max;
    }
    return input;
}
