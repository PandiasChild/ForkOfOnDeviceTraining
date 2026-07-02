#define SOURCE_FILE "UnitTestCalculateGradsSequential"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* Build a [1,2] float32 tensor from a stack buffer (data is copied into the tensor). */
static tensor_t *makeRowVec2(float a, float b) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    size_t *order = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 2;
    order[0] = 0;
    order[1] = 1;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 2;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    float vals[2] = {a, b};
    tensorFillFromFloatBuffer(t, vals, 2);
    return t;
}

/* Structural note: tracedGrads and calculateGradsSequential both call calculateGradsImpl
 * internally; npyDumpSink (and any other sink) observes tensors but does not mutate them.
 * This means the closed-form characterisation test pins both paths simultaneously. */
void testCalculateGradsSequentialClosedForm() {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    layer_t *model[2];
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
    model[1] = softmaxLayerInit(&lq);

    /* Set known weights/bias: W = {{0.1,0.2},{0.3,0.4}}, b = {0,0}. */
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float B[2] = {0.0f, 0.0f};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);

    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f); /* one-hot class 0 */

    trainingStats_t *stats = calculateGradsSequential(
        model, 2,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        REDUCTION_MEAN, x, label);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.91300f, stats->loss);

    float *wg = (float *)getGradFromParameter(model[0]->config->linear->weights)->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, wg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, wg[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, wg[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, wg[3]);

    float *bg = (float *)getGradFromParameter(model[0]->config->linear->bias)->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, bg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, bg[1]);

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
}

#define MAX_EVENTS 64
typedef struct {
    size_t idx;
    char phase[32];
    size_t ndim;
} traceEvent_t;
static traceEvent_t g_events[MAX_EVENTS];
static size_t g_eventCount;

static void recordingSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                          tensor_t *t) {
    (void)ctx;
    (void)type;
    if (g_eventCount >= MAX_EVENTS) {
        return;
    }
    g_events[g_eventCount].idx = layerIdx;
    snprintf(g_events[g_eventCount].phase, sizeof(g_events[g_eventCount].phase), "%s", phase);
    g_events[g_eventCount].ndim = t->shape->numberOfDimensions;
    g_eventCount++;
}

void testTracedGradsFiresInOrder() {
    g_eventCount = 0;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    layer_t *model[2];
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
    model[1] = softmaxLayerInit(&lq);
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f}, B[2] = {0};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);

    trainingStats_t *stats = tracedGrads(model, 2,
                                         (lossConfig_t){.funcType = CROSS_ENTROPY,
                                                        .backwardReduction = REDUCTION_MEAN,
                                                        .classWeights = NULL},
                                         REDUCTION_MEAN, x, label, recordingSink, NULL);

    /* fwd L0, fwd L1, lossgrad@2, agrad L0  (Softmax skipped under CE) */
    TEST_ASSERT_EQUAL_size_t(4, g_eventCount);
    TEST_ASSERT_EQUAL_size_t(0, g_events[0].idx);
    TEST_ASSERT_EQUAL_STRING("fwd", g_events[0].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[0].ndim);
    TEST_ASSERT_EQUAL_size_t(1, g_events[1].idx);
    TEST_ASSERT_EQUAL_STRING("fwd", g_events[1].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[1].ndim);
    TEST_ASSERT_EQUAL_size_t(2, g_events[2].idx);
    TEST_ASSERT_EQUAL_STRING("lossgrad", g_events[2].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[2].ndim);
    TEST_ASSERT_EQUAL_size_t(0, g_events[3].idx);
    TEST_ASSERT_EQUAL_STRING("agrad", g_events[3].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[3].ndim);

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
}

void testTraceModelParamsFiresPerTrainableParam() {
    g_eventCount = 0;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    layer_t *model[2];
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
    model[1] = softmaxLayerInit(&lq);
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f}, B[2] = {0};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);
    tensor_t *x = makeRowVec2(1.0f, 1.0f), *label = makeRowVec2(1.0f, 0.0f);
    trainingStats_t *stats = calculateGradsSequential(
        model, 2,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        REDUCTION_MEAN, x, label);

    traceModelWeights(model, 2, "w_before", recordingSink, NULL);
    traceModelGrads(model, 2, "grad_raw", recordingSink, NULL);

    /* weight+bias for the one Linear, then wgrad+bgrad */
    TEST_ASSERT_EQUAL_size_t(4, g_eventCount);
    TEST_ASSERT_EQUAL_STRING("w_before.weight", g_events[0].phase);
    TEST_ASSERT_EQUAL_STRING("w_before.bias", g_events[1].phase);
    TEST_ASSERT_EQUAL_STRING("grad_raw.weight", g_events[2].phase);
    TEST_ASSERT_EQUAL_STRING("grad_raw.bias", g_events[3].phase);

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
}

/* ── #221 regression: dx wire must honor the producer's declared propLossQ ── */

typedef struct {
    roundingMode_t agradRoundingMode;
    bool capturedAgrad;
} agradCapCtx_t;

static void agradCaptureSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                             tensor_t *t) {
    (void)type;
    agradCapCtx_t *c = (agradCapCtx_t *)ctx;
    /* agrad@1 = dx wire produced by Linear1 (idx 2) via backwardWireQ → propLossQ (SR_HALF_AWAY) */
    if (layerIdx == 1 && strcmp(phase, "agrad") == 0) {
        c->agradRoundingMode = ((symInt32QConfig_t *)t->quantization->qConfig)->roundingMode;
        c->capturedAgrad = true;
    }
}

/* Create a SYM_INT32 parameter (param tensor + optional grad tensor) from float values.
 * dims[0..ndim-1] describe the shape; vals[0..n-1] are loaded via tensorFillFromFloatBuffer. */
static parameter_t *makeSymParam(float *vals, size_t n, size_t *dims, size_t ndim,
                                 roundingMode_t rm, bool needsGrad) {
    size_t *d = reserveMemory(ndim * sizeof(size_t));
    size_t *o = reserveMemory(ndim * sizeof(size_t));
    for (size_t i = 0; i < ndim; i++) {
        d[i] = dims[i];
    }
    setOrderOfDimsForNewTensor(ndim, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, ndim, o);
    tensor_t *param = initTensor(s, quantizationInitSymInt32(rm), NULL);
    tensorFillFromFloatBuffer(param, vals, n);
    tensor_t *grad = needsGrad ? gradInitSymInt32(param, rm, NULL) : NULL;
    return parameterInit(param, grad);
}

void testDxWireHonorsProducerPropLossQ(void) {
    /* 4-layer SYM_INT32 chain: [Linear0 (idx 0), Quant0 (idx 1), Linear1 (idx 2), Quant1 (idx 3)].
     * All forward/backwardMath use HALF_AWAY; Linear1's propLossQ = SR_HALF_AWAY.
     * Quantization layers are mandatory between SYM producers: the forward wire carries raw
     * accumulator-range mantissas until a Quant layer restores int12 range (a direct
     * Linear-Linear chain overflows int32 under UBSan).
     *
     * Pre-fix: the agrad@1 buffer carried HALF_AWAY (derived from Quant0's forward-output
     * quantization via initGradTensor, ignoring Linear1's propLossQ).
     * Post-fix: it carries SR_HALF_AWAY (from Linear1's propLossQ). */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *symQSr = quantizationInitSymInt32(SR_HALF_AWAY);

    float wVals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float bVals[2] = {0.0f, 0.0f};
    size_t wDims[2] = {2, 2};
    size_t bDims[1] = {2};
    parameter_t *w0 = makeSymParam(wVals, 4, wDims, 2, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *b0 = makeSymParam(bVals, 2, bDims, 1, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *w1 = makeSymParam(wVals, 4, wDims, 2, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *b1 = makeSymParam(bVals, 2, bDims, 1, HALF_AWAY, /*needsGrad=*/true);

    layer_t *linear0 = linearLayerInitLegacy(w0, b0, symQ, symQ, symQ, symQ);
    /* linear1: forwardQ/backwardMath = HALF_AWAY, propLossQ = SR_HALF_AWAY */
    layer_t *linear1 = linearLayerInitLegacy(w1, b1, symQ, symQ, symQ, symQSr);

    /* Borrowing Quantization layers between SYM producers (ownsQuantizations=false). */
    quantizationConfig_t *qCfg0 = reserveMemory(sizeof(quantizationConfig_t));
    qCfg0->forwardQ = symQ;
    qCfg0->backwardQ = symQ;
    qCfg0->ownsQuantizations = false;
    layerConfig_t *lc0 = reserveMemory(sizeof(layerConfig_t));
    lc0->quantization = qCfg0;
    layer_t *quant0 = reserveMemory(sizeof(layer_t));
    initLayer(quant0, QUANTIZATION, lc0);

    quantizationConfig_t *qCfg1 = reserveMemory(sizeof(quantizationConfig_t));
    qCfg1->forwardQ = symQ;
    qCfg1->backwardQ = symQ;
    qCfg1->ownsQuantizations = false;
    layerConfig_t *lc1 = reserveMemory(sizeof(layerConfig_t));
    lc1->quantization = qCfg1;
    layer_t *quant1 = reserveMemory(sizeof(layer_t));
    initLayer(quant1, QUANTIZATION, lc1);

    layer_t *model[4] = {linear0, quant0, linear1, quant1};

    size_t *xd = reserveMemory(2 * sizeof(size_t));
    size_t *xo = reserveMemory(2 * sizeof(size_t));
    xd[0] = 1;
    xd[1] = 2;
    setOrderOfDimsForNewTensor(2, xo);
    shape_t *xShape = reserveMemory(sizeof(shape_t));
    setShape(xShape, xd, 2, xo);
    tensor_t *x = initTensor(xShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(x, (float[]){1.0f, 2.0f}, 2);

    size_t *ld = reserveMemory(2 * sizeof(size_t));
    size_t *lo = reserveMemory(2 * sizeof(size_t));
    ld[0] = 1;
    ld[1] = 2;
    setOrderOfDimsForNewTensor(2, lo);
    shape_t *lShape = reserveMemory(sizeof(shape_t));
    setShape(lShape, ld, 2, lo);
    tensor_t *label = initTensor(lShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(label, (float[]){0.0f, 0.0f}, 2);

    agradCapCtx_t ctx = {.capturedAgrad = false};
    trainingStats_t *stats = tracedGrads(
        model, 4,
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL},
        REDUCTION_SUM, x, label, agradCaptureSink, &ctx);

    /* Capture before teardown (ASSERT LAST convention). */
    bool captured = ctx.capturedAgrad;
    roundingMode_t mode = ctx.agradRoundingMode;

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayerLegacy(linear0);
    freeLinearLayerLegacy(linear1);
    freeReservedMemory(qCfg0);
    freeReservedMemory(lc0);
    freeReservedMemory(quant0);
    freeReservedMemory(qCfg1);
    freeReservedMemory(lc1);
    freeReservedMemory(quant1);
    freeParameter(w0);
    freeParameter(b0);
    freeParameter(w1);
    freeParameter(b1);
    freeQuantization(symQ);
    freeQuantization(symQSr);

    TEST_ASSERT_TRUE_MESSAGE(captured, "sink never captured agrad for layer 1");
    TEST_ASSERT_EQUAL_INT_MESSAGE(SR_HALF_AWAY, mode,
                                  "dx wire must carry the PRODUCER's declared propLossQ "
                                  "roundingMode, not the upstream forward roundingMode (#221)");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalculateGradsSequentialClosedForm);
    RUN_TEST(testTracedGradsFiresInOrder);
    RUN_TEST(testTraceModelParamsFiresPerTrainableParam);
    RUN_TEST(testDxWireHonorsProducerPropLossQ);
    return UNITY_END();
}
