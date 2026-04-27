#define SOURCE_FILE "UnitTestCommon"

#include "Common.h"
#include "unity.h"

_Static_assert(
    DLEVEL >= 1,
    "DEBUG_MODE_ERROR must be defined by default — PRINT_ERROR would be silently compiled out");

void setUp(void) {}
void tearDown(void) {}

void testDLevelDefaultIsErrorOrHigher(void) {
    TEST_ASSERT_GREATER_OR_EQUAL_INT(1, DLEVEL);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testDLevelDefaultIsErrorOrHigher);
    return UNITY_END();
}
