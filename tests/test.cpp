#include "pch.h"
#include <gtest/gtest.h>
#include <exception>
#include <iostream>
#include <cstdlib>

#include "../opennn/random_utilities.h"

using namespace std;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    opennn::set_seed(20260707);

    //::testing::GTEST_FLAG(filter) = "*";

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
