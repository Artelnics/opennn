#include "pch.h"

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "RecurrentLayerTest.*";

    return RUN_ALL_TESTS();
}
