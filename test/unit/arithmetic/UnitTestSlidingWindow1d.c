#include "SlidingWindow1d.h"
#include "unity.h"

void testGeometryValidNoPadding() {
    kernel_t kernel = {.size = 2, .stride = 1, .dilation = 1, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(4, &kernel);

    TEST_ASSERT_EQUAL_size_t(4, g.inputLength);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(2, g.kernelSize);
    TEST_ASSERT_EQUAL_size_t(1, g.stride);
    TEST_ASSERT_EQUAL_size_t(1, g.dilation);
    TEST_ASSERT_EQUAL_size_t(0, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(0, g.padRight);
}

void testGeometryValidWithStride() {
    kernel_t kernel = {.size = 2, .stride = 2, .dilation = 1, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(6, &kernel);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(0, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(0, g.padRight);
}

void testGeometryValidWithDilation() {
    kernel_t kernel = {.size = 2, .stride = 3, .dilation = 2, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(9, &kernel);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
}

void testGeometrySameSymmetricPadding() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    TEST_ASSERT_EQUAL_size_t(5, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(1, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(1, g.padRight);
}

void testGeometrySameAsymmetricPadding() {
    kernel_t kernel = {.size = 4, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    TEST_ASSERT_EQUAL_size_t(5, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(1, g.padLeft);  // total pad = 3, left gets floor(3/2)=1
    TEST_ASSERT_EQUAL_size_t(2, g.padRight); // right gets ceil(3/2)=2
}

void testGeometryKernelLargerThanInput() {
    kernel_t kernel = {.size = 5, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(2, &kernel);
    TEST_ASSERT_EQUAL_size_t(2, g.outputLength);
    // Total pad = 5 + (2-1)*1 - 2 = 4; padLeft = 2, padRight = 2
    TEST_ASSERT_EQUAL_size_t(2, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(2, g.padRight);
}

void testGeometryValidKernelLargerThanInput() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = VALID};
    windowGeometry1d_t g = windowGeometry1dCalc(2, &kernel);
    TEST_ASSERT_EQUAL_size_t(0, g.outputLength);
    TEST_ASSERT_EQUAL_size_t(0, g.padLeft);
    TEST_ASSERT_EQUAL_size_t(0, g.padRight);
}

void testSliceCenterFullWindow() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    // outputPos=2 is in the center: window covers inputs [1, 2, 3] — all valid
    windowSlice1d_t s = windowSlice1dAt(&g, 2);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(3, s.validCount);
}

void testSliceLeftEdgeWithPadding() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    // outputPos=0: window starts at inputStart = 0 - 1 = -1; kernel pos 0 -> -1 (OOB)
    //                                                         kernel pos 1 -> 0  (OK)
    //                                                         kernel pos 2 -> 1  (OK)
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(2, s.validCount);
}

void testSliceRightEdgeWithPadding() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(5, &kernel);
    // outputPos=4: window starts at inputStart = 4 - 1 = 3; kernel pos 0 -> 3 (OK)
    //                                                       kernel pos 1 -> 4 (OK)
    //                                                       kernel pos 2 -> 5 (OOB)
    windowSlice1d_t s = windowSlice1dAt(&g, 4);
    TEST_ASSERT_EQUAL_size_t(3, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(2, s.validCount);
}

void testSliceWithDilation() {
    kernel_t kernel = {.size = 3, .stride = 1, .dilation = 2, .paddingType = VALID};
    // effective kernel = 5; inputLen=7 -> outputLen=3
    windowGeometry1d_t g = windowGeometry1dCalc(7, &kernel);
    TEST_ASSERT_EQUAL_size_t(3, g.outputLength);
    // outputPos=0: kernel positions map to input indices [0, 2, 4] — all valid
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(3, s.validCount);
}

void testSlicePartialWindowBothSides() {
    // Both edges OOB but middle is valid (NOT a true empty window — see testSliceTrulyEmpty)
    // pathological: kernel=5, dilation=1, SAME on input of length 2 (padLeft=padRight=2)
    kernel_t kernel = {.size = 5, .stride = 1, .dilation = 1, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(2, &kernel);
    // outputPos=0: inputStart = 0 - 2 = -2;
    //   kernel pos 0 -> -2  (OOB)
    //   kernel pos 1 -> -1  (OOB)
    //   kernel pos 2 -> 0   (OK)
    //   kernel pos 3 -> 1   (OK)
    //   kernel pos 4 -> 2   (OOB, inputLength=2)
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(2, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(2, s.validCount);
}

void testSliceWithDilationAndPadding() {
    // kernel=2, dilation=2, SAME on inputLen=4:
    //   effectiveKernel = 3; outputLength = 4;
    //   totalPad = 3 + 3 - 4 = 2; padLeft=1, padRight=1
    // outputPos=0: inputStart = 0 - 1 = -1
    //   kernel pos 0 -> -1 + 0*2 = -1 (OOB)
    //   kernel pos 1 -> -1 + 1*2 =  1 (OK)
    // ceil((-(-1))/2) = ceil(1/2) = 1; floor would give 0 — this distinguishes ceil from floor.
    kernel_t kernel = {.size = 2, .stride = 1, .dilation = 2, .paddingType = SAME};
    windowGeometry1d_t g = windowGeometry1dCalc(4, &kernel);
    TEST_ASSERT_EQUAL_size_t(1, g.padLeft);  // sanity
    TEST_ASSERT_EQUAL_size_t(1, g.padRight); // sanity

    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidKernelOffset);
    TEST_ASSERT_EQUAL_size_t(1, s.validCount);
}

void testSliceTrulyEmpty() {
    // Pathological geometry constructed by hand — exercise the firstK > lastK sentinel.
    // Window entirely on left padding: padLeft=100, kernel=3, dilation=1 -> all 3 kernel
    // positions land at inputStart..inputStart+2 = -100..-98, all OOB.
    windowGeometry1d_t g = {
        .inputLength = 5,
        .outputLength = 1,
        .kernelSize = 3,
        .stride = 1,
        .dilation = 1,
        .padLeft = 100,
        .padRight = 0,
    };
    windowSlice1d_t s = windowSlice1dAt(&g, 0);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(3, s.firstValidKernelOffset); // sentinel = kernelSize
    TEST_ASSERT_EQUAL_size_t(0, s.validCount);
}

void testSliceTrulyEmptyOnRightEdge() {
    // Symmetric counterpart to testSliceTrulyEmpty. Hand-built geometry where
    // outputPos pushes inputStart past the end of input.
    // padRight=10 + outputLength=10 + stride=1 - padLeft=0 = inputStart up to 9
    // for outputPos=2: inputStart = 2*1 - 0 = 2; inputLength = 2 -> inputStart >= inputLength
    windowGeometry1d_t g = {
        .inputLength = 2,
        .outputLength = 10,
        .kernelSize = 1,
        .stride = 1,
        .dilation = 2,
        .padLeft = 0,
        .padRight = 10,
    };
    windowSlice1d_t s = windowSlice1dAt(&g, 2);
    TEST_ASSERT_EQUAL_size_t(0, s.firstValidInputIdx);
    TEST_ASSERT_EQUAL_size_t(1, s.firstValidKernelOffset); // sentinel = kernelSize
    TEST_ASSERT_EQUAL_size_t(0, s.validCount);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testGeometryValidNoPadding);
    RUN_TEST(testGeometryValidWithStride);
    RUN_TEST(testGeometryValidWithDilation);
    RUN_TEST(testGeometrySameSymmetricPadding);
    RUN_TEST(testGeometrySameAsymmetricPadding);
    RUN_TEST(testGeometryKernelLargerThanInput);
    RUN_TEST(testGeometryValidKernelLargerThanInput);
    RUN_TEST(testSliceCenterFullWindow);
    RUN_TEST(testSliceLeftEdgeWithPadding);
    RUN_TEST(testSliceRightEdgeWithPadding);
    RUN_TEST(testSliceWithDilation);
    RUN_TEST(testSlicePartialWindowBothSides);
    RUN_TEST(testSliceWithDilationAndPadding);
    RUN_TEST(testSliceTrulyEmpty);
    RUN_TEST(testSliceTrulyEmptyOnRightEdge);
    return UNITY_END();
}
