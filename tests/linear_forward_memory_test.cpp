
#include "pch.h"

// Before any OpenNN header: those pull "using namespace std" into scope, and
// windows.h declares its own global `byte`, which std::byte would then clash
// with inside the Windows headers themselves.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

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

namespace
{

#ifdef __linux__

long long committed_bytes()
{
    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line))
        if (line.rfind("VmData:", 0) == 0)
            return stoll(line.substr(7)) * 1024LL;
    return -1;
}

#elif defined(_WIN32)

// Private committed bytes - what VmData measures on Linux.
long long committed_bytes()
{
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);

    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                              sizeof(counters)))
        return -1;

    return static_cast<long long>(counters.PrivateUsage);
}

// Windows has no RLIMIT_DATA, but a job object caps committed memory the same
// way: past the cap an allocation fails, so a materialized temporary throws
// bad_alloc exactly as it would on Linux.
//
// The cap has to come back off whatever happens - a process cannot leave a job
// once assigned, and every later test allocates.
class CommitCap
{
public:

    explicit CommitCap(long long bytes)
    {
        job = CreateJobObjectW(nullptr, nullptr);
        if (!job) return;

        if (!AssignProcessToJobObject(job, GetCurrentProcess()))
        {
            CloseHandle(job);
            job = nullptr;
            return;
        }

        assigned = set_limit(bytes);
    }

    ~CommitCap()
    {
        if (job)
        {
            if (assigned) set_limit(0);   // 0 clears the limit flag
            CloseHandle(job);
        }
    }

    bool active() const { return assigned; }

    CommitCap(const CommitCap&) = delete;
    CommitCap& operator=(const CommitCap&) = delete;

private:

    bool set_limit(long long bytes)
    {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};

        if (bytes > 0)
        {
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            limits.ProcessMemoryLimit = static_cast<SIZE_T>(bytes);
        }

        return SetInformationJobObject(job,
                                       JobObjectExtendedLimitInformation,
                                       &limits,
                                       sizeof(limits)) != FALSE;
    }

    HANDLE job = nullptr;
    bool assigned = false;
};

#endif

}

TEST(LinearForwardMemoryTest, SteadyStateForwardAllocatesNoLargeTemporaries)
{
#if !defined(__linux__) && !defined(_WIN32)
    GTEST_SKIP() << "needs per-process memory accounting";
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

    // Both warmups are done, so the arena and the allocator are settled: any
    // growth from here is the forward itself asking for new memory.
    const long long baseline = committed_bytes();
    ASSERT_GT(baseline, 0);

    const long long slack = 64LL * 1024 * 1024;

    const char* const diagnosis =
        "the steady-state dense forward allocated a large block: "
        "look for an Eigen expression that materializes a product "
        "temporary (e.g. (input * weights).rowwise() + bias) in "
        "linear_forward_cpu or another operator";

    // A cap that quietly failed to apply would let this test pass whatever the
    // forward does, so ask for far more than the slack and require a refusal.
    const auto oversized_allocation = [batch]
    {
        MatrixR block(batch, 4096);   // ~512 MB, against 64 MB of slack
        block.setZero();              // touch it, so the commit is real
    };

    const char* const instrument_broken =
        "the memory cap is not in force, so this test could not detect a "
        "large temporary even if the forward allocated one";

#ifdef __linux__

    rlimit old_limit {};
    ASSERT_EQ(getrlimit(RLIMIT_DATA, &old_limit), 0);

    rlimit capped = old_limit;
    capped.rlim_cur = rlim_t(baseline) + rlim_t(slack);
    if (old_limit.rlim_max != RLIM_INFINITY && capped.rlim_cur > old_limit.rlim_max)
        capped.rlim_cur = old_limit.rlim_max;
    ASSERT_EQ(setrlimit(RLIMIT_DATA, &capped), 0);

    EXPECT_THROW(oversized_allocation(), bad_alloc) << instrument_broken;

    EXPECT_NO_THROW(network.forward_propagate(inputs, forward_propagation, false))
        << diagnosis;

    setrlimit(RLIMIT_DATA, &old_limit);

#else

    {
        const CommitCap cap(baseline + slack);

        if (!cap.active())
            GTEST_SKIP() << "could not cap this process's committed memory";

        EXPECT_THROW(oversized_allocation(), bad_alloc) << instrument_broken;

        EXPECT_NO_THROW(network.forward_propagate(inputs, forward_propagation, false))
            << diagnosis;
    }

#endif

    const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();
    EXPECT_TRUE(isfinite(outputs(0, 0)));
#endif
}
