#include "pch.h"

// Sample test case
// TEST(SampleTest, SimpleTest) {
//     EXPECT_EQ(1, 1); // This should pass
// }

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "PerformanceTest.ImageClassification";

    ::testing::GTEST_FLAG(filter) = "PoolingLayerTests/*";

    return RUN_ALL_TESTS();
}
