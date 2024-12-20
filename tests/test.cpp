#include "pch.h"

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FlattenLayerTest";
    //::testing::GTEST_FLAG(filter) = "PoolingLayerTests/*";
    //::testing::GTEST_FLAG(filter) = "ConvolutionalLayerTests/*";

    return RUN_ALL_TESTS();
}
