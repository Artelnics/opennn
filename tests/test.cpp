#include "pch.h"
#include <gtest/gtest.h>
#include <exception>
#include <iostream>
#include <cstdlib>

#include "../opennn/configuration.h"

using namespace std;
using namespace opennn;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    Configuration::instance().set(Device::CPU, Type::FP32);

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
