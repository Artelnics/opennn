#include "pch.h"

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "StatisticsTest.*";
    //::testing::GTEST_FLAG(filter) = "GeneticAlgorithmTest";
    //::testing::GTEST_FLAG(filter) = "PoolingLayerTests/*";
    //::testing::GTEST_FLAG(filter) = "ConvolutionalLayerTests/*";

    // ::testing::GTEST_FLAG(filter) = "TestingAnalysis.*";
    // ::testing::GTEST_FLAG(filter) = "CorrelationsTest.*";

    return RUN_ALL_TESTS();
}
