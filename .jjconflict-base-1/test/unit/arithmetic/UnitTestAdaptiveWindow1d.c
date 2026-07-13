#include "AdaptiveWindow1d.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

// L_in=4, L_out=2 -> (0,2),(2,2): divides evenly, non-overlapping
void testDividesEvenly(void) {
    adaptiveWindow1d_t w0 = adaptiveWindow1dAt(4, 2, 0);
    TEST_ASSERT_EQUAL_size_t(0, w0.start);
    TEST_ASSERT_EQUAL_size_t(2, w0.count);
    adaptiveWindow1d_t w1 = adaptiveWindow1dAt(4, 2, 1);
    TEST_ASSERT_EQUAL_size_t(2, w1.start);
    TEST_ASSERT_EQUAL_size_t(2, w1.count);
}

// L_in=5, L_out=2 -> (0,3),(2,3): overlap at index 2
void testOverlap(void) {
    adaptiveWindow1d_t w0 = adaptiveWindow1dAt(5, 2, 0);
    TEST_ASSERT_EQUAL_size_t(0, w0.start);
    TEST_ASSERT_EQUAL_size_t(3, w0.count);
    adaptiveWindow1d_t w1 = adaptiveWindow1dAt(5, 2, 1);
    TEST_ASSERT_EQUAL_size_t(2, w1.start);
    TEST_ASSERT_EQUAL_size_t(3, w1.count);
}

// L_in=6, L_out=1 -> (0,6): global average over whole length
void testGlobal(void) {
    adaptiveWindow1d_t w = adaptiveWindow1dAt(6, 1, 0);
    TEST_ASSERT_EQUAL_size_t(0, w.start);
    TEST_ASSERT_EQUAL_size_t(6, w.count);
}

// L_in=4, L_out=4 -> (0,1),(1,1),(2,1),(3,1): identity
void testIdentity(void) {
    for (size_t o = 0; o < 4; o++) {
        adaptiveWindow1d_t w = adaptiveWindow1dAt(4, 4, o);
        TEST_ASSERT_EQUAL_size_t(o, w.start);
        TEST_ASSERT_EQUAL_size_t(1, w.count);
    }
}

// L_in=3, L_out=5 -> (0,1),(0,2),(1,1),(1,2),(2,1): upsample, variable windows
void testUpsample(void) {
    size_t expStart[5] = {0, 0, 1, 1, 2};
    size_t expCount[5] = {1, 2, 1, 2, 1};
    for (size_t o = 0; o < 5; o++) {
        adaptiveWindow1d_t w = adaptiveWindow1dAt(3, 5, o);
        TEST_ASSERT_EQUAL_size_t(expStart[o], w.start);
        TEST_ASSERT_EQUAL_size_t(expCount[o], w.count);
    }
}

// L_in=1, L_out=1 -> (0,1): degenerate single element
void testSingle(void) {
    adaptiveWindow1d_t w = adaptiveWindow1dAt(1, 1, 0);
    TEST_ASSERT_EQUAL_size_t(0, w.start);
    TEST_ASSERT_EQUAL_size_t(1, w.count);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testDividesEvenly);
    RUN_TEST(testOverlap);
    RUN_TEST(testGlobal);
    RUN_TEST(testIdentity);
    RUN_TEST(testUpsample);
    RUN_TEST(testSingle);
    return UNITY_END();
}
