#define SOURCE_FILE "UnitTestStorageApi"

#include <stddef.h>
#include <stdint.h>

#include "StorageApi.h"
#include "unity.h"

/* Compile-time contract: reserveMemory returns the allocated payload
 * directly (void *), and freeReservedMemory accepts that same payload.
 * The old void ** handle-indirection is removed; see issue #99 comment
 * for the design rationale (a real handle-table allocator will reintroduce
 * a typed opaque handle type later, not void **). */
_Static_assert(_Generic((&reserveMemory), void *(*)(size_t): 1, default: 0),
               "reserveMemory must return void * (payload), not void ** (handle)");
_Static_assert(_Generic((&freeReservedMemory), void (*)(void *): 1, default: 0),
               "freeReservedMemory must take void * (payload)");

void testReserveMemoryReturnsZeroedWritablePayload(void) {
    uint8_t *payload = reserveMemory(64);
    TEST_ASSERT_NOT_NULL(payload);

    for (size_t i = 0; i < 64; ++i) {
        TEST_ASSERT_EQUAL_UINT8(0, payload[i]);
    }
    payload[0] = 0xAA;
    payload[63] = 0x55;
    TEST_ASSERT_EQUAL_UINT8(0xAA, payload[0]);
    TEST_ASSERT_EQUAL_UINT8(0x55, payload[63]);

    freeReservedMemory(payload);
}

void testReserveAndFreeRoundTrip(void) {
    int *payload = reserveMemory(sizeof(int));
    TEST_ASSERT_NOT_NULL(payload);

    *payload = 42;
    TEST_ASSERT_EQUAL_INT(42, *payload);

    freeReservedMemory(payload);
}

void testFreeReservedMemoryIsNullSafe(void) {
    freeReservedMemory(NULL);
}

#ifdef ODT_MEM_PROFILE
/* GT-1: exact-integer heap accounting. No tolerance exists to fudge.
 * Guarded by ODT_MEM_PROFILE: the counter is a no-op when the macro is
 * undefined, so these tests only compile/run in profiling builds (the
 * dedicated c-memprofile CI job). */
void testHeapCounterExactCurrentAndPeak(void) {
    memProfileReset();
    TEST_ASSERT_EQUAL_size_t(0, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(0, memProfilePeakBytes());

    void *a = reserveMemory(1000);
    TEST_ASSERT_EQUAL_size_t(1000, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(1000, memProfilePeakBytes());

    void *b = reserveMemory(500);
    TEST_ASSERT_EQUAL_size_t(1500, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(1500, memProfilePeakBytes());

    freeReservedMemory(a);
    TEST_ASSERT_EQUAL_size_t(500, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(1500, memProfilePeakBytes()); /* peak holds */

    void *c = reserveMemory(2000);
    TEST_ASSERT_EQUAL_size_t(2500, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(2500, memProfilePeakBytes()); /* new peak */

    freeReservedMemory(b);
    freeReservedMemory(c);
    TEST_ASSERT_EQUAL_size_t(0, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(2500, memProfilePeakBytes());
}

/* The size-header must preserve alignment + writability of the full request. */
void testReservedRegionUsableAndAligned(void) {
    memProfileReset();
    size_t n = 777;
    unsigned char *p = reserveMemory(n);
    TEST_ASSERT_EQUAL_UINT(0, (uintptr_t)p % _Alignof(max_align_t));
    for (size_t i = 0; i < n; i++) {
        p[i] = (unsigned char)(i & 0xFF); /* full-range write; ASan guards OOB */
    }
    TEST_ASSERT_EQUAL_size_t(n, memProfileCurrentBytes());
    freeReservedMemory(p);
    TEST_ASSERT_EQUAL_size_t(0, memProfileCurrentBytes());
}
#endif /* ODT_MEM_PROFILE */

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReserveMemoryReturnsZeroedWritablePayload);
    RUN_TEST(testReserveAndFreeRoundTrip);
    RUN_TEST(testFreeReservedMemoryIsNullSafe);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testHeapCounterExactCurrentAndPeak);
    RUN_TEST(testReservedRegionUsableAndAligned);
#endif
    return UNITY_END();
}
