#define SOURCE_FILE "npy_dump_sink"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Quantization.h"
#include "Tensor.h"
#include "npy_dump_sink.h"
#include "npy_writer.h"

void npyDumpSink(void *ctxV, size_t layerIdx, layerType_t layerType, const char *phase,
                 tensor_t *tensor) {
    (void)layerType;
    npyDumpCtx_t *ctx = (npyDumpCtx_t *)ctxV;

    if (tensor->quantization->type != FLOAT32) {
        fprintf(stderr, "npyDumpSink: only FLOAT32 supported (probe %zu, phase %s)\n", layerIdx,
                phase);
        exit(1);
    }

    const char *probe = (layerIdx < ctx->numProbes) ? ctx->probeNames[layerIdx] : "loss";

    char path[512];
    if (ctx->sampleIdx == NPY_DUMP_NO_SAMPLE) {
        snprintf(path, sizeof(path), "%s/%s.%s.npy", ctx->dir, probe, phase);
    } else {
        snprintf(path, sizeof(path), "%s/%s.%s.s%03zu.npy", ctx->dir, probe, phase, ctx->sampleIdx);
    }

    int rc = npyWriteFloat32(path, (float *)tensor->data, tensor->shape->dimensions,
                             tensor->shape->numberOfDimensions);
    if (rc != 0) {
        fprintf(stderr, "npyDumpSink: write failed for %s (rc=%d)\n", path, rc);
        exit(1);
    }
}
