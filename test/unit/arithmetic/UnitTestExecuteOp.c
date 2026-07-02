#define SOURCE_FILE "EXECUTE-OP-UTEST"
#include <stdint.h>
#include <string.h>

#include "Add.h"
#include "Common.h"
#include "DeathTest.h"
#include "ExecuteOp.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* ---- helpers -------------------------------------------------------- */

static tensor_t *buildSym(size_t n, const int32_t *mantissas, float scale) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)t->data)[i] = mantissas[i];
    }
    ((symInt32QConfig_t *)t->quantization->qConfig)->scale = scale;
    return t;
}

static tensor_t *buildFloat(size_t n, const float *values) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, n);
    return t;
}

/* ---- tests ---------------------------------------------------------- */

/* Prologue no-op: dtype matches the arithmetic -> the source is passed
 * through untouched (data AND scale unchanged after the call). */
void testProloguePassesMatchingOperandThroughUntouched(void) {
    tensor_t *in = buildSym(3, (int32_t[]){100, -200, 300}, 0.01f);
    tensor_t *out = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){in}, 1, &arith, out, OUT_WRITE);

    int32_t srcM0 = ((int32_t *)in->data)[0];
    float srcScale = ((symInt32QConfig_t *)in->quantization->qConfig)->scale;
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT(100, srcM0);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.01f, srcScale);
}

/* Prologue conversion: a FLOAT32 operand entering SYM arithmetic is
 * quantized into transient scratch; the result equals running the same
 * kernel on a pre-converted operand, and the float source is untouched. */
void testPrologueConvertsFloatOperandIntoSymArithmetic(void) {
    tensor_t *in = buildFloat(3, (float[]){0.5f, -1.0f, 1.5f});
    tensor_t *out = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){in}, 1, &arith, out, OUT_WRITE);

    /* Reference: quantize the float input to int12 SYM directly, then requant
     * to the target exactly as the epilogue's OUT_WRITE diagonal does. */
    tensor_t *ref = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    tensor_t *inSym = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    convertTensor(in, inSym);
    conversionMatrix[SYM_INT32][SYM_INT32](inSym, ref);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)out->data)[i];
        want[i] = ((int32_t *)ref->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)out->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)ref->quantization->qConfig)->scale;
    float srcV0 = ((float *)in->data)[0];
    freeTensor(inSym);
    freeTensor(ref);
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, srcV0); /* source untouched */
}

/* OUT_WRITE SYM->SYM must be bit-identical to requantSymInt32Tensor (the
 * conversionMatrix diagonal) — NOT convertTensor's same-type memmove. Uses
 * accumulator-range mantissas so the memmove-vs-requant difference is huge. */
void testOutWriteSymToSymBitEqualsDiagonalRequant(void) {
    tensor_t *in = buildSym(3, (int32_t[]){1500000, -750000, 250000}, 1e-4f);
    tensor_t *out = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){in}, 1, &arith, out, OUT_WRITE);

    tensor_t *ref = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f);
    conversionMatrix[SYM_INT32][SYM_INT32](in, ref);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)out->data)[i];
        want[i] = ((int32_t *)ref->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)out->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)ref->quantization->qConfig)->scale;
    freeTensor(ref);
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
}

/* OUT_WRITE SYM->FLOAT32 equals convertTensor (dequantization). */
void testOutWriteSymToFloatEqualsConvertTensor(void) {
    tensor_t *in = buildSym(2, (int32_t[]){12000, -6000}, 0.5f);
    tensor_t *out = buildFloat(2, (float[]){0.f, 0.f});
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){in}, 1, &arith, out, OUT_WRITE);

    float got0 = ((float *)out->data)[0];
    float got1 = ((float *)out->data)[1];
    freeTensor(out);
    freeTensor(in);
    TEST_ASSERT_EQUAL_FLOAT(6000.0f, got0); /* 12000 * 0.5 — anti-vacuity */
    TEST_ASSERT_EQUAL_FLOAT(-3000.0f, got1);
}

/* Exact float accumulation: pre-seeded target, accumulate twice, exact sums.
 * Catches overwrite-instead-of-add and wrong-target mutations. */
void testAccDynamicIntoFloatIsExactAndAccumulates(void) {
    tensor_t *inc = buildFloat(3, (float[]){0.25f, -0.5f, 1.0f});
    tensor_t *grad = buildFloat(3, (float[]){1.0f, 1.0f, 1.0f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1, &floatArith, grad,
              OUT_ACC_DYNAMIC_RESCALE);
    executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1, &floatArith, grad,
              OUT_ACC_DYNAMIC_RESCALE);

    float g0 = ((float *)grad->data)[0];
    float g1 = ((float *)grad->data)[1];
    float g2 = ((float *)grad->data)[2];
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_FLOAT(1.5f, g0); /* 1.0 + 2*0.25 — exact */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, g1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, g2);
}

/* OUT_ACC_FIXED_SCALE on a FLOAT32 target collapses to the same exact add. */
void testAccFixedIntoFloatCollapsesToExactAdd(void) {
    tensor_t *inc = buildFloat(2, (float[]){0.5f, -0.25f});
    tensor_t *grad = buildFloat(2, (float[]){2.0f, 2.0f});
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1, &floatArith, grad,
              OUT_ACC_FIXED_SCALE);

    float g0 = ((float *)grad->data)[0];
    float g1 = ((float *)grad->data)[1];
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_FLOAT(2.5f, g0);
    TEST_ASSERT_EQUAL_FLOAT(1.75f, g1);
}

/* Dynamic SYM+SYM must be bit-identical to addSymInt32TensorsInplace. */
void testAccDynamicSymIntoSymBitEqualsAddSymInplace(void) {
    tensor_t *inc = buildSym(3, (int32_t[]){1000, -2000, 500}, 0.001f);
    tensor_t *grad = buildSym(3, (int32_t[]){10, 20, 30}, 0.05f);
    tensor_t *refInc = buildSym(3, (int32_t[]){1000, -2000, 500}, 0.001f);
    tensor_t *refGrad = buildSym(3, (int32_t[]){10, 20, 30}, 0.05f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1, &arith, grad,
              OUT_ACC_DYNAMIC_RESCALE);
    addSymInt32TensorsInplace(refGrad, refInc);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)grad->data)[i];
        want[i] = ((int32_t *)refGrad->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)grad->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)refGrad->quantization->qConfig)->scale;
    freeTensor(refGrad);
    freeTensor(refInc);
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
}

/* Dynamic FLOAT32 intermediate + SYM target: the increment is quantized to
 * operand-width SYM (roundingMode from the TARGET) then Strategy-A-added —
 * bit-identical to the LayerNorm.c:446-463 reference emulated here. */
void testAccDynamicFloatIncIntoSymTargetMatchesLayerNormReference(void) {
    float incVals[3] = {0.3f, -0.7f, 0.1f};
    tensor_t *inc = buildFloat(3, incVals);
    tensor_t *grad = buildSym(3, (int32_t[]){40, -80, 120}, 0.02f);
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1, &floatArith, grad,
              OUT_ACC_DYNAMIC_RESCALE);

    /* Reference: exact LayerNorm.c:446-463 sequence on a twin. */
    tensor_t *refGrad = buildSym(3, (int32_t[]){40, -80, 120}, 0.02f);
    tensor_t *refIncFloat = buildFloat(3, incVals);
    tensor_t *refIncSym = buildSym(3, (int32_t[]){0, 0, 0}, 1.0f); /* int12 shell, HALF_AWAY */
    convertTensor(refIncFloat, refIncSym);
    addSymInt32TensorsInplace(refGrad, refIncSym);

    int32_t got[3];
    int32_t want[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = ((int32_t *)grad->data)[i];
        want[i] = ((int32_t *)refGrad->data)[i];
    }
    float gotScale = ((symInt32QConfig_t *)grad->quantization->qConfig)->scale;
    float wantScale = ((symInt32QConfig_t *)refGrad->quantization->qConfig)->scale;
    freeTensor(refIncSym);
    freeTensor(refIncFloat);
    freeTensor(refGrad);
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 3);
    TEST_ASSERT_EQUAL_FLOAT(wantScale, gotScale);
}

/* Fixed-scale SYM+SYM reproduces linearCalcBiasGradsSymInt32's semantics:
 * target[i] += roundf(interm[i] * intermScale / targetScale), NO clamp.
 * Hand-computed: interm {700, -300} @ 0.01 into target {5, -5} @ 0.02:
 * roundf(700*0.01/0.02) = 350 -> 355; roundf(-300*0.5) = -150 -> -155.
 * Target scale must stay EXACTLY 0.02 (never re-derived). */
void testAccFixedSymIntoSymRescalesIntoExistingScale(void) {
    tensor_t *inc = buildSym(2, (int32_t[]){700, -300}, 0.01f);
    tensor_t *grad = buildSym(2, (int32_t[]){5, -5}, 0.02f);
    quantization_t arith;
    symInt32QConfig_t arithQC;
    initSymInt32QConfig(HALF_AWAY, &arithQC);
    initSymInt32Quantization(&arithQC, &arith);

    executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1, &arith, grad, OUT_ACC_FIXED_SCALE);

    int32_t g0 = ((int32_t *)grad->data)[0];
    int32_t g1 = ((int32_t *)grad->data)[1];
    float gScale = ((symInt32QConfig_t *)grad->quantization->qConfig)->scale;
    freeTensor(grad);
    freeTensor(inc);
    TEST_ASSERT_EQUAL_INT(355, g0);
    TEST_ASSERT_EQUAL_INT(-155, g1);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.02f, gScale);
}

/* Sub-byte / unsupported targets abort until PR3. */
void testAccIntoSubByteTargetAborts(void) {
    tensor_t *inc = buildFloat(2, (float[]){1.f, 2.f});
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = 2;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *grad = initTensor(shape, quantizationInitSym(8, HALF_AWAY), NULL);
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    ASSERT_EXITS_WITH_FAILURE(executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1,
                                        &floatArith, grad, OUT_ACC_DYNAMIC_RESCALE));

    freeTensor(grad);
    freeTensor(inc);
}

/* Grad-width contract moved from layerNormValidateSymGrad: SYM targets wider
 * than ODT_SYM_GRAD_QMAXBITS abort. */
void testAccIntoTooWideSymTargetAborts(void) {
    tensor_t *inc = buildFloat(2, (float[]){1.f, 2.f});
    tensor_t *grad = buildSym(2, (int32_t[]){0, 0}, 1.0f);
    ((symInt32QConfig_t *)grad->quantization->qConfig)->qMaxBits = 24; /* > 16 */
    quantization_t floatArith;
    initFloat32Quantization(&floatArith);

    ASSERT_EXITS_WITH_FAILURE(executeOp(executeOpIdentityKernel, (tensor_t *[]){inc}, 1,
                                        &floatArith, grad, OUT_ACC_DYNAMIC_RESCALE));

    freeTensor(grad);
    freeTensor(inc);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testProloguePassesMatchingOperandThroughUntouched);
    RUN_TEST(testPrologueConvertsFloatOperandIntoSymArithmetic);
    RUN_TEST(testOutWriteSymToSymBitEqualsDiagonalRequant);
    RUN_TEST(testOutWriteSymToFloatEqualsConvertTensor);
    RUN_TEST(testAccDynamicIntoFloatIsExactAndAccumulates);
    RUN_TEST(testAccFixedIntoFloatCollapsesToExactAdd);
    RUN_TEST(testAccDynamicSymIntoSymBitEqualsAddSymInplace);
    RUN_TEST(testAccDynamicFloatIncIntoSymTargetMatchesLayerNormReference);
    RUN_TEST(testAccFixedSymIntoSymRescalesIntoExistingScale);
    RUN_TEST(testAccIntoSubByteTargetAborts);
    RUN_TEST(testAccIntoTooWideSymTargetAborts);
    return UNITY_END();
}
