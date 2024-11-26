#include "pch.h"

// Sample test case
// TEST(SampleTest, SimpleTest) {
//     EXPECT_EQ(1, 1); // This should pass
// }

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "PerformanceTest.ImageClassification";

<<<<<<< HEAD
    //::testing::GTEST_FLAG(filter) = "PoolingLayerTests/*";
=======
    ::testing::GTEST_FLAG(filter) = "ConvolutionalLayerTests/*";
>>>>>>> acde2b25554c3cb1fe0ef447c58dfba1588857fa

    return RUN_ALL_TESTS();
}
