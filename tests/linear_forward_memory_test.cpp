
#include "pch.h"

#include <cmath>

#include "opennn/tensor_types.h"
#include "opennn/configuration.h"
#include "opennn/dense_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/neural_network.h"

#ifdef __linux__
#include <fstream>
#include <string>
#include <sys/resource.h>
#endif

using namespace opennn;

#ifdef __linux__
namespace
{

long vm_data_bytes()
{
    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line))
        if (line.rfind("VmData:", 0) == 0)
            return stol(line.substr(7)) * 1024L;
    return -1;
}

}
#endif


TEST(LinearForwardMemoryTest, SteadyStateForwardAllocatesNoLargeTemporaries)
{
#ifndef __linux__
    GTEST_SKIP() << "needs Linux RLIMIT_DATA accounting";
#else
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch = 32768;

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(Shape{28}, Shape{1024}, "ReLU"));
    network.add_layer(make_unique<opennn::Dense>(Shape{1024}, Shape{1024}, "ReLU"));
    network.add_layer(make_unique<opennn::Dense>(Shape{1024}, Shape{1}, "Sigmoid"));
    network.compile();
    network.set_parameters_glorot();

    ForwardPropagation forward_propagation(batch, &network);

    const MatrixR inputs_host = MatrixR::Random(batch, 28);
    const TensorView input_view(const_cast<float*>(inputs_host.data()),
                                Shape{batch, 28}, Type::FP32);
    const vector<TensorView> inputs = {input_view};

    network.forward_propagate(inputs, forward_propagation, false);
    network.forward_propagate(inputs, forward_propagation, false);

    const long baseline = vm_data_bytes();
    ASSERT_GT(baseline, 0);

    const rlim_t slack = 64L * 1024 * 1024;

    rlimit old_limit {};
    ASSERT_EQ(getrlimit(RLIMIT_DATA, &old_limit), 0);

    rlimit capped = old_limit;
    capped.rlim_cur = rlim_t(baseline) + slack;
    if (old_limit.rlim_max != RLIM_INFINITY && capped.rlim_cur > old_limit.rlim_max)
        capped.rlim_cur = old_limit.rlim_max;
    ASSERT_EQ(setrlimit(RLIMIT_DATA, &capped), 0);

    EXPECT_NO_THROW(network.forward_propagate(inputs, forward_propagation, false))
        << "the steady-state dense forward allocated a large block: "
           "look for an Eigen expression that materializes a product "
           "temporary (e.g. (input * weights).rowwise() + bias) in "
           "linear_forward_cpu or another operator";

    setrlimit(RLIMIT_DATA, &old_limit);

    const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();
    EXPECT_TRUE(isfinite(outputs(0, 0)));
#endif
}
