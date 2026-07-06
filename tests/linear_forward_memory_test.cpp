//   Regression test for hidden Eigen temporaries in the CPU dense forward.
//
//   Every ForwardPropagation buffer is pre-allocated, so a steady-state
//   forward pass must not allocate any large block. The 2026-07 bug --
//   linear_forward_cpu written as "output = (input * weights).rowwise() +
//   bias", which makes Eigen materialize the whole product in a heap
//   temporary -- cost an extra batch x outputs allocation per dense layer per
//   pass and nearly halved the CPU inference max batch
//   (docs/benchmarks/higgs-max-batch/).
//
//   The test warms the pass up (twice, so glibc's arena and Eigen's GEMM
//   scratch reach steady state), clamps RLIMIT_DATA to the process's current
//   data usage plus a small slack, and runs the pass again: a reintroduced
//   batch-sized temporary (128 MiB here, twice the slack) blows the cap and
//   throws bad_alloc. Linux-only; skipped elsewhere.

#include "pch.h"

#include <cmath>

#include "../opennn/tensor_types.h"
#include "../opennn/configuration.h"
#include "../opennn/dense_layer.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/neural_network.h"

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
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind("VmData:", 0) == 0)
            return std::stol(line.substr(7)) * 1024L;
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

    // 32768 x (28 -> 1024 -> 1024 -> 1): a reintroduced product temporary is
    // batch x 1024 floats = 128 MiB, twice the slack below.
    const Index batch = 32768;

    NeuralNetwork network;
    network.add_layer(std::make_unique<opennn::Dense>(Shape{28}, Shape{1024}, "ReLU"));
    network.add_layer(std::make_unique<opennn::Dense>(Shape{1024}, Shape{1024}, "ReLU"));
    network.add_layer(std::make_unique<opennn::Dense>(Shape{1024}, Shape{1}, "Sigmoid"));
    network.compile();
    network.set_parameters_glorot();

    ForwardPropagation forward_propagation(batch, &network);

    const MatrixR inputs_host = MatrixR::Random(batch, 28);
    const TensorView input_view(const_cast<float*>(inputs_host.data()),
                                Shape{batch, 28}, Type::FP32);
    const std::vector<TensorView> inputs = {input_view};

    // Two warmup passes: the first faults the pre-allocated buffers in and
    // creates Eigen/OpenMP scratch (mmap'd, then released, bumping glibc's
    // mmap threshold); the second re-allocates that scratch from the arena,
    // which caches it -- after this the pass is allocation-steady.
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
    EXPECT_TRUE(std::isfinite(outputs(0, 0)));
#endif
}
