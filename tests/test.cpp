#include "pch.h"
#include <gtest/gtest.h>
#include <exception>
#include <iostream>
#include <cstdlib>

#include "opennn/configuration.h"
#include "opennn/device_backend.h"

using namespace std;
using namespace opennn;

namespace
{

class CpuConfigurationListener : public ::testing::EmptyTestEventListener
{
public:
    void OnTestStart(const ::testing::TestInfo&) override
    {
        Configuration::instance().set(Device::CPU, Type::FP32);

        device::reset_last_error();
    }
};

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()->listeners().Append(new CpuConfigurationListener);

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
