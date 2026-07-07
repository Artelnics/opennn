// OpenNN baseline-memory benchmark.
//
// Construct the minimum OpenNN object graph used by a training application
// (empty TabularDataset, tiny NeuralNetwork, TrainingStrategy), then report:
//   - baseline_ram_mb: current resident set size for the process
//   - gpu_ready_vram_mb: nvidia-smi-reported GPU memory for this process after
//     one tiny CUDA matrix multiply, or NA when unavailable

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <process.h>
#include <psapi.h>
#include <windows.h>
#define popen _popen
#define pclose _pclose
static int current_pid() { return _getpid(); }
#else
#include <sys/resource.h>
#include <unistd.h>
static int current_pid() { return getpid(); }
#endif

#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/configuration.h"
#include "../../../opennn/device_backend.h"
#include "../../../opennn/standard_networks.h"
#include "../../../opennn/tabular_dataset.h"
#include "../../../opennn/training_strategy.h"

using namespace opennn;

static double current_rss_mb()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return double(counters.WorkingSetSize) / (1024.0 * 1024.0);
    return 0.0;
#else
    std::ifstream status("/proc/self/status");
    std::string line;

    while (std::getline(status, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
        {
            std::istringstream stream(line.substr(6));
            double kb = 0.0;
            stream >> kb;
            return kb / 1024.0;
        }
    }

    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
#endif
}

static std::optional<double> current_process_vram_mb()
{
    const int pid = current_pid();
#ifdef _WIN32
    const char* command = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>NUL";
#else
    const char* command = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null";
#endif

    FILE* pipe = popen(command, "r");
    if (!pipe) return std::nullopt;

    std::optional<double> total;
    char buffer[256];

    while (fgets(buffer, sizeof(buffer), pipe))
    {
        std::string line(buffer);
        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream stream(line);
        int row_pid = 0;
        double used_mb = 0.0;

        if (stream >> row_pid >> used_mb && row_pid == pid)
            total = total.value_or(0.0) + used_mb;
    }

    pclose(pipe);
    return total;
}

static std::optional<double> gpu_used_memory_mb()
{
#ifdef _WIN32
    const char* command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>NUL";
#else
    const char* command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null";
#endif

    FILE* pipe = popen(command, "r");
    if (!pipe) return std::nullopt;

    char buffer[256];
    std::optional<double> used;

    while (fgets(buffer, sizeof(buffer), pipe))
    {
        std::istringstream stream(buffer);
        double value = 0.0;
        if (stream >> value)
        {
            used = value;
            break;
        }
    }

    pclose(pipe);
    return used;
}

static bool run_tiny_gpu_matmul()
{
    if (!device::is_cuda_build() || !device::has_cuda_device())
        return false;

    Configuration::instance().set(Device::CUDA, Type::FP32);

#ifdef OPENNN_HAS_CUDA
    constexpr int n = 32;
    constexpr Index bytes = Index(n * n * sizeof(float));

    auto* a = static_cast<float*>(device::allocate(Device::CUDA, bytes));
    auto* b = static_cast<float*>(device::allocate(Device::CUDA, bytes));
    auto* c = static_cast<float*>(device::allocate(Device::CUDA, bytes));

    cudaStream_t stream = Backend::get_compute_stream();
    device::set_zero_async(a, bytes, stream);
    device::set_zero_async(b, bytes, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(Backend::get_cublas_handle(),
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             a, n,
                             b, n,
                             &beta,
                             c, n));
    device::synchronize(stream);

    device::deallocate(Device::CUDA, a, bytes);
    device::deallocate(Device::CUDA, b, bytes);
    device::deallocate(Device::CUDA, c, bytes);
#endif

    return true;
}

int main()
{
    const auto vram_before_mb = gpu_used_memory_mb();

    Configuration::instance().set(Device::CPU, Type::FP32);

    TabularDataset dataset(0, Shape{1}, Shape{1});
    ApproximationNetwork network(Shape{1}, Shape{}, Shape{1});
    TrainingStrategy training_strategy(&network, &dataset);
    training_strategy.set_loss("MeanSquaredError");
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    const double baseline_ram_mb = current_rss_mb();

    run_tiny_gpu_matmul();

    printf("baseline_ram_mb %.1f\n", baseline_ram_mb);

    auto vram_mb = current_process_vram_mb();
    if (!vram_mb)
    {
        const auto vram_after_mb = gpu_used_memory_mb();
        if (vram_before_mb && vram_after_mb)
            vram_mb = std::max(0.0, *vram_after_mb - *vram_before_mb);
    }

    if (vram_mb)
        printf("gpu_ready_vram_mb %.1f\n", *vram_mb);
    else
        printf("gpu_ready_vram_mb NA\n");

    return 0;
}
