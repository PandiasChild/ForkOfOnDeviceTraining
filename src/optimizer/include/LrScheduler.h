#ifndef LR_SCHEDULER_H
#define LR_SCHEDULER_H

#include <stddef.h>

#include "Optimizer.h"

/*! Learning-rate schedulers with PyTorch-parity semantics (#327).
 *
 * Caller-owned struct, zero allocation. `baseLr` is captured ONCE at init
 * (via the optimizer vtable's getLr); every lrSchedulerStep() computes the
 * closed form FROM baseLr and overwrites the optimizer LR absolutely —
 * never compounding on the possibly-mutated current value.
 *
 * Stepping is boundary-agnostic: trainingRun steps once per epoch (after
 * the epoch callback); hand-rolled loops may call lrSchedulerStep() at any
 * boundary (per-batch included). `lastEpoch` counts step() calls, exactly
 * like PyTorch's last_epoch. After init (lastEpoch == 0) the LR equals
 * baseLr for all three types, matching PyTorch after construction.
 *
 * Schedule scalars are computed in double (pow/cos) and cast to float only
 * when written through setLr — mirroring PyTorch's float64 scheduler math
 * feeding float32 kernels. */

typedef enum { STEP_LR, EXPONENTIAL_LR, COSINE_ANNEALING_LR } lrSchedulerType_t;

typedef struct lrScheduler {
    lrSchedulerType_t type;
    optimizer_t *optimizer;
    float baseLr;
    size_t lastEpoch;
    union {
        struct {
            size_t stepSize;
            float gamma;
        } stepLr;
        struct {
            float gamma;
        } exponentialLr;
        struct {
            size_t tMax;
            float etaMin;
        } cosineAnnealingLr;
    } params;
} lrScheduler_t;

typedef float (*computeLrFn_t)(const lrScheduler_t *sched);

typedef struct lrSchedulerFunctions {
    computeLrFn_t computeLr;
} lrSchedulerFunctions_t;

extern lrSchedulerFunctions_t lrSchedulerFunctions[];

/*! lr = baseLr * gamma^floor(lastEpoch / stepSize)  (torch StepLR) */
void stepLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t stepSize, float gamma);

/*! lr = baseLr * gamma^lastEpoch  (torch ExponentialLR) */
void exponentialLrInit(lrScheduler_t *sched, optimizer_t *optimizer, float gamma);

/*! lr = etaMin + (baseLr - etaMin) * (1 + cos(pi * lastEpoch / tMax)) / 2
 *  (torch CosineAnnealingLR closed form; periodic past tMax) */
void cosineAnnealingLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t tMax, float etaMin);

/*! lastEpoch++ -> computeLr -> setLr through the optimizer vtable. */
void lrSchedulerStep(lrScheduler_t *sched);

#endif // LR_SCHEDULER_H
