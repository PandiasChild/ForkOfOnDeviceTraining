#ifndef ENV5_RUNTIME_QUANTIZATION_H
#define ENV5_RUNTIME_QUANTIZATION_H

#include "Rounding.h"

typedef enum qtype { INT32, FLOAT32, SYM_INT32, SYM, ASYM, BOOL, DELTA } qtype_t;

typedef struct symInt32QConfig {
    float scale;
    roundingMode_t roundingMode;
    uint8_t qMaxBits;
} symInt32QConfig_t;

/* SYM_INT32 operand bit-width contract (#227). Operands feeding product
 * accumulators are int12 so int12*int12 products stay within an int32
 * accumulator (no int64). Sound for reductions N <= 511 (512*2^22 > INT32_MAX);
 * narrow the knob for wider layers. Override with -DODT_SYM_OPERAND_QMAXBITS=N. */
#ifndef ODT_SYM_OPERAND_QMAXBITS
#define ODT_SYM_OPERAND_QMAXBITS 12
#endif
#ifndef ODT_SYM_GRAD_QMAXBITS
#define ODT_SYM_GRAD_QMAXBITS 16
#endif

typedef struct symQConfig {
    float scale;
    uint8_t qBits;
    roundingMode_t roundingMode;
} symQConfig_t;

typedef struct asymQConfig {
    float scale;
    int16_t zeroPoint;
    uint8_t qBits;
    roundingMode_t roundingMode;
} asymQConfig_t;

typedef struct symQDeltaConfig {
    float scale;
    uint8_t qBits;
    roundingMode_t roundingMode;
    uint8_t deltabits;
} symQDeltaConfig_t;

typedef struct quantization {
    qtype_t type;
    void *qConfig;
} quantization_t;

// Important: This sets qMaxBits to ODT_SYM_OPERAND_QMAXBITS (12)
void initSymInt32QConfig(roundingMode_t roundingMode, symInt32QConfig_t *symInt32QConfig);
void initSymInt32QConfigWithQMaxBits(roundingMode_t roundingMode,
                                     symInt32QConfig_t *symInt32QConfig, uint8_t qMaxBits);
void initSymQConfig(uint8_t qBits, roundingMode_t roundingMode, symQConfig_t *symQConfig);
void initAsymQConfig(uint8_t qBits, roundingMode_t roundingMode, asymQConfig_t *asymQConfig);
/**
 * @brief Initializes a symmetric delta quantization configuration.
 *
 * This function sets up a symQDeltaConfig_t structure used for delta-based quantization.
 * It configures the number of Q-format fractional bits, the rounding mode, and the number
 * of delta bits. The scale is initialized to 1.0f by default.
 *
 * @param qBits[in]            Number of fractional bits used in the Q-format representation.
 * @param roundingMode[in]     Specifies the rounding mode used during quantization.
 * @param deltabits[in]        Number of bits used for delta encoding.
 * @param symQDeltaConfig[out] Pointer to the configuration structure to be initialized.
 */
void initSymQDeltaConfig(uint8_t qBits, roundingMode_t roundingMode, uint8_t deltabits, symQDeltaConfig_t *symQDeltaConfig);

void initInt32Quantization(quantization_t *quantization);
void initFloat32Quantization(quantization_t *quantization);
void initBoolQuantization(quantization_t *quantization);

void initSymInt32Quantization(symInt32QConfig_t *symInt32QConfig, quantization_t *quantization);
void initSymQuantization(symQConfig_t *symQConfig, quantization_t *quantization);
void initAsymQuantization(asymQConfig_t *asymQConfig, quantization_t *quantization);

/**
 * @brief Initializes a delta quantization instance which can be used for delta compression
 * @param symQDeltaConfig[in] Input configuration.
 * @param quantization[out] Output quantization structure.
 */
void initSymQDeltaQuantization(symQDeltaConfig_t *symQDeltaConfig, quantization_t *quantization);

#endif // ENV5_RUNTIME_QUANTIZATION_H
