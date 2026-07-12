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

/* Task 13 adds the replay data loader here. */

#endif // ODT_PPCA_REPLAY_API_H
