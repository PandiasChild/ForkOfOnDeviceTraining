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

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReserveMemoryReturnsZeroedWritablePayload);
    RUN_TEST(testReserveAndFreeRoundTrip);
    RUN_TEST(testFreeReservedMemoryIsNullSafe);
    return UNITY_END();
}
