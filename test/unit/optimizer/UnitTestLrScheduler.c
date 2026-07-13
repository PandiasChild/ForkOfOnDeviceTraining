#define SOURCE_FILE "LR-SCHED-UTEST"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ArithmeticType.h"
#include "DeathTest.h"
#include "LrScheduler.h"
#include "Optimizer.h"
#include "Sgd.h"
#include "unity.h"

#include "expected_lr_scheduler.h"

/* #327 signature contracts: init fns take (sched, optimizer, <params...>),
 * step takes the scheduler only. Compile-time pins. */
_Static_assert(_Generic(&stepLrInit,
                   void (*)(lrScheduler_t *, optimizer_t *, size_t, float): 1,
                   default: 0),
               "#327: stepLrInit must be (sched, optimizer, stepSize, gamma)");
_Static_assert(_Generic(&exponentialLrInit,
                   void (*)(lrScheduler_t *, optimizer_t *, float): 1,
                   default: 0),
               "#327: exponentialLrInit must be (sched, optimizer, gamma)");
_Static_assert(_Generic(&cosineAnnealingLrInit,
                   void (*)(lrScheduler_t *, optimizer_t *, size_t, float): 1,
                   default: 0),
               "#327: cosineAnnealingLrInit must be (sched, optimizer, tMax, etaMin)");
_Static_assert(_Generic(&lrSchedulerStep, void (*)(lrScheduler_t *): 1, default: 0),
               "#327: lrSchedulerStep must be (sched)");

void setUp() {}
void tearDown() {}

/* Hand-assembled SGD optimizer on the stack — the scheduler needs only the
 * vtable LR accessors, no parameters/states (same idiom as UnitTestSgd's
 * hand-assembly tests). */
static sgd_t g_sgd;
static optimImpl_t g_impl;

static optimizer_t makeSgdOptimizer(float lr) {
    sgdInit(&g_sgd, lr, 0.9f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    g_impl.sgd = &g_sgd;
    return (optimizer_t){
        .type = SGD_M, .impl = &g_impl, .parameter = NULL, .states = NULL, .sizeStates = 0};
}

/* Gold comparisons are EXACT (bit equality): the gold header is generated at
 * build time on this host, C computes the same double closed form and casts
 * to float at setLr — any difference is a real semantics bug, not noise. */
static void assertLrBitsEqual(float expected, float actual, size_t epoch) {
    uint32_t e, a;
    memcpy(&e, &expected, sizeof e);
    memcpy(&a, &actual, sizeof a);
    char msg[64];
    snprintf(msg, sizeof msg, "LR mismatch at epoch %zu", epoch);
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(e, a, msg);
}

static void runGoldSequence(lrScheduler_t *sched, optimizer_t *optim, const float *gold,
                            size_t goldLen) {
    assertLrBitsEqual(gold[0], optimizerFunctions[optim->type].getLr(optim), 0);
    for (size_t epoch = 1; epoch < goldLen; epoch++) {
        lrSchedulerStep(sched);
        assertLrBitsEqual(gold[epoch], optimizerFunctions[optim->type].getLr(optim), epoch);
    }
}

void testStepLrMatchesTorchGold(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    stepLrInit(&sched, &optim, 5, 0.5f);
    runGoldSequence(&sched, &optim, lr_gold_step_b01_s5_g05, lr_gold_step_b01_s5_g05_len);
}

void testStepLrEveryEpochMatchesTorchGold(void) {
    optimizer_t optim = makeSgdOptimizer(0.01f);
    lrScheduler_t sched;
    stepLrInit(&sched, &optim, 1, 0.9f);
    runGoldSequence(&sched, &optim, lr_gold_step_b001_s1_g09, lr_gold_step_b001_s1_g09_len);
}

void testStepLrCoarseMatchesTorchGold(void) {
    optimizer_t optim = makeSgdOptimizer(0.05f);
    lrScheduler_t sched;
    stepLrInit(&sched, &optim, 7, 0.1f);
    runGoldSequence(&sched, &optim, lr_gold_step_b005_s7_g01, lr_gold_step_b005_s7_g01_len);
}

void testExponentialLrMatchesTorchGold(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    exponentialLrInit(&sched, &optim, 0.95f);
    runGoldSequence(&sched, &optim, lr_gold_exp_b01_g095, lr_gold_exp_b01_g095_len);
}

void testExponentialLrFastDecayMatchesTorchGold(void) {
    optimizer_t optim = makeSgdOptimizer(0.01f);
    lrScheduler_t sched;
    exponentialLrInit(&sched, &optim, 0.5f);
    runGoldSequence(&sched, &optim, lr_gold_exp_b001_g05, lr_gold_exp_b001_g05_len);
}

void testCosineAnnealingLrMatchesTorchGold(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    cosineAnnealingLrInit(&sched, &optim, 20, 0.0f);
    runGoldSequence(&sched, &optim, lr_gold_cos_b01_t20_e0, lr_gold_cos_b01_t20_e0_len);
}

void testCosineAnnealingLrPastTMaxMatchesTorchGold(void) {
    /* 20 steps with tMax=10: the closed form is periodic and the LR comes
     * back UP after tMax — pins that C matches torch there too. */
    optimizer_t optim = makeSgdOptimizer(0.01f);
    lrScheduler_t sched;
    cosineAnnealingLrInit(&sched, &optim, 10, 0.001f);
    runGoldSequence(&sched, &optim, lr_gold_cos_b001_t10_e0001, lr_gold_cos_b001_t10_e0001_len);
}

void testCosineAnnealingLrAssociationDiscriminator(void) {
    /* The two fixtures above are bit-identical under BOTH float64
     * associations of pi*e/T_max — (pi*e)/T_max (torch's, correct) vs
     * pi*(e/T_max) (the bug fixed alongside this test) — so neither
     * actually enforces the expression order. This combo diverges at
     * epoch 13 (0.029999999329447746f vs 0.030000001192092896f); see
     * generate_expected_lr_scheduler.py and #327. */
    optimizer_t optim = makeSgdOptimizer(0.05f);
    lrScheduler_t sched;
    cosineAnnealingLrInit(&sched, &optim, 26, 0.01f);
    runGoldSequence(&sched, &optim, lr_gold_cos_discrim_b005_t26_e001,
                    lr_gold_cos_discrim_b005_t26_e001_len);
}

void testSchedulerOverwritesExternalLrMutationAbsolutely(void) {
    /* baseLr is captured ONCE at init; each step computes the closed form
     * from baseLr and OVERWRITES the optimizer LR. External mutation between
     * steps must not leak into subsequent values (anti-compounding pin). */
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    stepLrInit(&sched, &optim, 5, 0.5f);
    lrSchedulerStep(&sched);
    optimizerFunctions[SGD_M].setLr(&optim, 0.999f); /* sabotage */
    lrSchedulerStep(&sched);
    assertLrBitsEqual(lr_gold_step_b01_s5_g05[2], optimizerFunctions[SGD_M].getLr(&optim), 2);
}

void testStepLrInitRejectsZeroStepSize(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(stepLrInit(&sched, &optim, 0, 0.5f));
}

void testCosineAnnealingLrInitRejectsZeroTMax(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(cosineAnnealingLrInit(&sched, &optim, 0, 0.0f));
}

void testExponentialLrInitRejectsNonFiniteGamma(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(exponentialLrInit(&sched, &optim, NAN));
}

void testStepLrInitRejectsNonFiniteGamma(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(stepLrInit(&sched, &optim, 5, INFINITY));
}

void testCosineAnnealingLrInitRejectsNonFiniteEtaMin(void) {
    optimizer_t optim = makeSgdOptimizer(0.1f);
    lrScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(cosineAnnealingLrInit(&sched, &optim, 10, INFINITY));
}

void testInitRejectsNullOptimizer(void) {
    lrScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(stepLrInit(&sched, NULL, 5, 0.5f));
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testStepLrMatchesTorchGold);
    RUN_TEST(testStepLrEveryEpochMatchesTorchGold);
    RUN_TEST(testStepLrCoarseMatchesTorchGold);
    RUN_TEST(testExponentialLrMatchesTorchGold);
    RUN_TEST(testExponentialLrFastDecayMatchesTorchGold);
    RUN_TEST(testCosineAnnealingLrMatchesTorchGold);
    RUN_TEST(testCosineAnnealingLrPastTMaxMatchesTorchGold);
    RUN_TEST(testCosineAnnealingLrAssociationDiscriminator);
    RUN_TEST(testSchedulerOverwritesExternalLrMutationAbsolutely);
    RUN_TEST(testStepLrInitRejectsZeroStepSize);
    RUN_TEST(testCosineAnnealingLrInitRejectsZeroTMax);
    RUN_TEST(testExponentialLrInitRejectsNonFiniteGamma);
    RUN_TEST(testStepLrInitRejectsNonFiniteGamma);
    RUN_TEST(testCosineAnnealingLrInitRejectsNonFiniteEtaMin);
    RUN_TEST(testInitRejectsNullOptimizer);
    return UNITY_END();
}
