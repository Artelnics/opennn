#include "tests/pch.h"
#include <gtest/gtest.h>
#include <exception>
#include <iostream>
#include <cstdlib>

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"

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

        set_seed(1);

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
        const char* require_cuda = std::getenv("OPENNN_TEST_REQUIRE_CUDA");
        if (require_cuda && string_view(require_cuda) == "1" && !device::has_cuda_device())
        {
            cerr << "CUDA verification requires a CUDA build and an available GPU.\n";
            return EXIT_FAILURE;
        }
        return RUN_ALL_TESTS();
    } catch (const exception& e) {
        cerr << "\nFATAL: Unhandled exception caught in test: " << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "\nFATAL: Unknown exception caught in test." << endl;
        return EXIT_FAILURE;
    }
}
