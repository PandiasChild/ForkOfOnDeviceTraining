#ifndef ODT_PPCA_REPLAY_API_H
#define ODT_PPCA_REPLAY_API_H

#include "DataLoader.h"
#include "PpcaReplay.h"

ppcaReplay_t *ppcaReplayCreate(const ppcaReplayConfig_t *cfg);
void freePpcaReplay(ppcaReplay_t *g);

ppcaWorkspace_t *ppcaWorkspaceCreate(size_t dim, size_t rank, size_t maxSessionSamples);
void freePpcaWorkspace(ppcaWorkspace_t *ws);

ppcaReplaySet_t *ppcaReplaySetCreate(size_t numClasses, const ppcaReplayConfig_t *cfg);
void freePpcaReplaySet(ppcaReplaySet_t *set);

typedef struct {
    ppcaReplaySet_t *set;
    size_t samplesPerClass; /* r */
    uint32_t minCount;      /* class eligible once generator count >= minCount */
    rng32_t *stream;        /* caller-owned sampling stream */
} replayLoaderConfig_t;

/* Wraps base: getBatch appends r synthetic samples per eligible class
 * AFTER the real samples (sample 0 stays real -> labelRef/meanScale
 * contracts hold; the optimizer's MEAN divisor uses batch->size and thus
 * scales automatically — the replay layer adds NO second divisor).
 * Pools are lazily initialized from the first real batch (item + label
 * shape/dtype/qconfig mirrored via getShapeLike/getQLike, labels must be
 * FLOAT32 one-hot [numClasses]). The wrapper BORROWS base's fields; free
 * it ONLY with freeReplayDataLoader (freeDataLoader on the wrapper would
 * free base's indices). Base stays owned by the caller. */
dataLoader_t *replayDataLoaderWrap(dataLoader_t *base, const replayLoaderConfig_t *cfg);
void freeReplayDataLoader(dataLoader_t *wrapped);

#endif // ODT_PPCA_REPLAY_API_H
