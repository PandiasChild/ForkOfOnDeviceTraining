#define SOURCE_FILE "PPCA_REPLAY_SERIALIZE"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Deserialize.h"
#include "PpcaReplaySerialize.h"
#include "Serialize.h"
#include "Tensor.h"

#define PPCA_SERIALIZE_MAGIC "ODTR"
#define PPCA_SERIALIZE_FORMAT_VERSION 1u
/* PPCA state tensors are rank 1 or 2 by construction (mean/eigvals, basis). */
#define PPCA_MAX_TENSOR_RANK 2

static void writeOrDie(const void *src, size_t bytes, FILE *f) {
    if (fwrite(src, 1, bytes, f) != bytes) {
        PRINT_ERROR("ppcaReplaySetSerialize: short write");
        exit(1);
    }
}

static void readOrDie(void *dst, size_t bytes, FILE *f) {
    if (fread(dst, 1, bytes, f) != bytes) {
        PRINT_ERROR("ppcaReplaySetDeserialize: short read / truncated file");
        exit(1);
    }
}

void ppcaReplaySetSerialize(const ppcaReplaySet_t *set, FILE *f) {
    writeOrDie(PPCA_SERIALIZE_MAGIC, 4, f);
    uint32_t version = PPCA_SERIALIZE_FORMAT_VERSION;
    writeOrDie(&version, sizeof(uint32_t), f);
    uint32_t numClasses = (uint32_t)set->numClasses;
    writeOrDie(&numClasses, sizeof(uint32_t), f);
    for (size_t c = 0; c < set->numClasses; c++) {
        const ppcaReplay_t *g = set->generators[c];
        uint32_t dim = (uint32_t)g->dim;
        uint32_t rank = (uint32_t)g->rank;
        writeOrDie(&dim, sizeof(uint32_t), f);
        writeOrDie(&rank, sizeof(uint32_t), f);
        writeOrDie(&g->count, sizeof(uint32_t), f);
        writeOrDie(&g->sigma2, sizeof(float), f);
        writeOrDie(&g->totalVar, sizeof(float), f);
        serializeTensor(g->mean, f);
        serializeTensor(g->basis, f);
        serializeTensor(g->eigvals, f);
    }
}

/* Peek one tensor record's header (shape + quantization) into locals,
 * validate against the skeleton BEFORE any overwrite, rewind, then let the
 * public deserializeTensor consume the (now trusted) record. Mirrors the
 * Serialize.c record layout field-for-field — testRoundTripPacked is the
 * drift alarm for this coupling. */
static void peekValidateThenDeserializeTensor(tensor_t *skeleton, FILE *f, const char *what) {
    long recordStart = ftell(f);
    if (recordStart < 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: stream not seekable (%s)", what);
        exit(1);
    }

    size_t nd;
    readOrDie(&nd, sizeof(size_t), f);
    if (nd != skeleton->shape->numberOfDimensions || nd > PPCA_MAX_TENSOR_RANK) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s rank mismatch (file %zu, skeleton %zu)", what, nd,
                    skeleton->shape->numberOfDimensions);
        exit(1);
    }
    size_t dims[PPCA_MAX_TENSOR_RANK];
    size_t order[PPCA_MAX_TENSOR_RANK];
    readOrDie(dims, nd * sizeof(size_t), f);
    readOrDie(order, nd * sizeof(size_t), f);
    for (size_t i = 0; i < nd; i++) {
        if (dims[i] != skeleton->shape->dimensions[i]) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s dim[%zu] mismatch (file %zu, skeleton %zu)",
                        what, i, dims[i], skeleton->shape->dimensions[i]);
            exit(1);
        }
    }

    uint8_t fileType;
    readOrDie(&fileType, sizeof(uint8_t), f);
    if ((qtype_t)fileType != skeleton->quantization->type) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s dtype mismatch (file %u, skeleton %u) — "
                    "#316 guard: rebuild the skeleton with the checkpoint's storage config",
                    what, (unsigned)fileType, (unsigned)skeleton->quantization->type);
        exit(1);
    }
    switch ((qtype_t)fileType) {
    case FLOAT32:
    case INT32:
    case BOOL:
        break;
    case SYM: {
        float scale;
        uint8_t qBits;
        roundingMode_t rm;
        readOrDie(&scale, sizeof(float), f);
        readOrDie(&qBits, sizeof(uint8_t), f);
        readOrDie(&rm, sizeof(roundingMode_t), f);
        symQConfig_t *skelQc = skeleton->quantization->qConfig;
        if (qBits != skelQc->qBits) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s SYM qBits mismatch (file %u, skeleton "
                        "%u) — same-dtype width mismatch is the #316 2x-overflow case",
                        what, (unsigned)qBits, (unsigned)skelQc->qBits);
            exit(1);
        }
        break;
    }
    case ASYM: {
        float scale;
        uint8_t qBits;
        roundingMode_t rm;
        int16_t zeroPoint;
        readOrDie(&scale, sizeof(float), f);
        readOrDie(&qBits, sizeof(uint8_t), f);
        readOrDie(&rm, sizeof(roundingMode_t), f);
        readOrDie(&zeroPoint, sizeof(int16_t), f);
        asymQConfig_t *skelQc = skeleton->quantization->qConfig;
        if (qBits != skelQc->qBits) {
            PRINT_ERROR("ppcaReplaySetDeserialize: %s ASYM qBits mismatch (file %u, skeleton %u)",
                        what, (unsigned)qBits, (unsigned)skelQc->qBits);
            exit(1);
        }
        break;
    }
    default:
        /* SYM_INT32 etc.: create() forbids these as PPCA state, so a file
         * claiming them is corrupt or foreign. */
        PRINT_ERROR("ppcaReplaySetDeserialize: %s carries non-PPCA storage dtype %u", what,
                    (unsigned)fileType);
        exit(1);
    }

    /* Post-read record-length check, sized from the PRE-overwrite skeleton:
     * deserializeTensor's payload fread is UNCHECKED, so a file cut inside
     * the payload region reads short silently — the position arithmetic
     * below catches that, and doubles as a wire-drift alarm if the peek
     * ever stops mirroring the record layout deserializeTensor consumes. */
    size_t elems = calcNumberOfElementsByShape(skeleton->shape);
    size_t expectedBytes = calcNumberOfBytesForData(skeleton->quantization, elems);
    long headerBytes = ftell(f) - recordStart;

    if (fseek(f, recordStart, SEEK_SET) != 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: rewind failed (%s)", what);
        exit(1);
    }
    deserializeTensor(skeleton, f);
    if (ftell(f) - recordStart != headerBytes + (long)expectedBytes) {
        PRINT_ERROR("ppcaReplaySetDeserialize: %s record length mismatch — truncated payload "
                    "or wire-layout drift",
                    what);
        exit(1);
    }
}

void ppcaReplaySetDeserialize(ppcaReplaySet_t *skeleton, FILE *f) {
    char magic[4];
    readOrDie(magic, 4, f);
    if (memcmp(magic, PPCA_SERIALIZE_MAGIC, 4) != 0) {
        PRINT_ERROR("ppcaReplaySetDeserialize: bad magic (not an ODTR checkpoint)");
        exit(1);
    }
    uint32_t version;
    readOrDie(&version, sizeof(uint32_t), f);
    if (version != PPCA_SERIALIZE_FORMAT_VERSION) {
        PRINT_ERROR("ppcaReplaySetDeserialize: unsupported version %u", version);
        exit(1);
    }
    uint32_t numClasses;
    readOrDie(&numClasses, sizeof(uint32_t), f);
    if (numClasses != (uint32_t)skeleton->numClasses) {
        PRINT_ERROR("ppcaReplaySetDeserialize: numClasses mismatch (file %u, skeleton %zu)",
                    numClasses, skeleton->numClasses);
        exit(1);
    }
    for (size_t c = 0; c < skeleton->numClasses; c++) {
        ppcaReplay_t *g = skeleton->generators[c];
        uint32_t dim, rank, count;
        float sigma2, totalVar;
        readOrDie(&dim, sizeof(uint32_t), f);
        readOrDie(&rank, sizeof(uint32_t), f);
        readOrDie(&count, sizeof(uint32_t), f);
        readOrDie(&sigma2, sizeof(float), f);
        readOrDie(&totalVar, sizeof(float), f);
        if (dim != (uint32_t)g->dim || rank != (uint32_t)g->rank) {
            PRINT_ERROR("ppcaReplaySetDeserialize: class %zu dim/rank mismatch "
                        "(file %u/%u, skeleton %zu/%zu)",
                        c, dim, rank, g->dim, g->rank);
            exit(1);
        }
        peekValidateThenDeserializeTensor(g->mean, f, "mean");
        peekValidateThenDeserializeTensor(g->basis, f, "basis");
        peekValidateThenDeserializeTensor(g->eigvals, f, "eigvals");
        /* Scalars land only after every guard for this class has passed. */
        g->count = count;
        g->sigma2 = sigma2;
        g->totalVar = totalVar;
    }
}
