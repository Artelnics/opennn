#include <gtest/gtest.h>
#include <exception>
#include <iostream>
#include <cstdlib>

using namespace std;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Disable Google Test's own exception catching
    ::testing::GTEST_FLAG(catch_exceptions) = false;

    //::testing::GTEST_FLAG(filter) = "RecurrentLayerTest.*";

    try {
        return RUN_ALL_TESTS();
    } catch (const exception& e) {
        cerr << "\nFATAL: Unhandled exception caught in test: " << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "\nFATAL: Unknown exception caught in test." << endl;
        return EXIT_FAILURE;
    }
}
