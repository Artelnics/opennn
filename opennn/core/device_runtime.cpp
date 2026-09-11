// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/device_backend.h"
#include "opennn/core/log.h"
#include "opennn/core/string_utilities.h"

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_service.h>
#endif

#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>

#ifdef __linux__
#include <sched.h>
#endif

namespace opennn
{

class Backend
{
public:
    static Backend& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int);

    static BlasHandle get_cublas_handle() { return instance().cublas(device::active_lane()); }
    static void apply_math_mode() { instance().set_math_mode_on_handles(); }
    static BlasLtHandle get_cublas_lt_handle()
    {
        Backend& backend = instance();
        backend.ensure_cuda();
        return backend.cublas_lt_handle;
    }
    static DnnHandle get_cudnn_handle() { return instance().cudnn(device::active_lane()); }
    static DnnOpTensorDescriptor get_op_tensor_add_descriptor()
    {
        Backend& backend = instance();
        backend.cudnn(0);
        return backend.op_tensor_add_descriptor;
    }

private:
    Backend();
    ~Backend();

    BlasHandle cublas(int lane);
    DnnHandle cudnn(int lane);
    DeviceStream stream(int lane);
    void set_math_mode_on_handles();
    void ensure_cuda();
    void release_cuda();
    static void register_cuda_cleanup();
    std::once_flag cuda_once;

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;
    BlasLtHandle cublas_lt_handle = nullptr;
    DnnOpTensorDescriptor op_tensor_add_descriptor = nullptr;
    std::mutex lane_mutex;
    std::array<DeviceStream, device::MAX_LANES> lane_streams{};
    std::array<BlasHandle, device::MAX_LANES> cublas_handles{};
    std::array<DnnHandle, device::MAX_LANES> cudnn_handles{};
    DeviceStream transfer_stream = nullptr;

    friend DeviceStream device::get_compute_stream();
    friend DeviceStream device::get_transfer_stream();
    friend DeviceStream device::lane_stream(int);
};

}

namespace opennn::device
{

#ifdef OPENNN_HAS_CUDA
CublasPointerModeGuard::CublasPointerModeGuard(
    const cublasHandle_t new_handle,
    const cublasPointerMode_t mode)
    : handle(new_handle)
{
    CHECK_CUBLAS(cublasGetPointerMode(handle, &previous_mode));
    CHECK_CUBLAS(cublasSetPointerMode(handle, mode));
}

CublasPointerModeGuard::~CublasPointerModeGuard() noexcept
{
    if (handle) cublasSetPointerMode(handle, previous_mode);
}

CublasMathModeGuard::CublasMathModeGuard(
    const cublasHandle_t new_handle,
    const cublasMath_t mode)
    : handle(new_handle)
{
    CHECK_CUBLAS(cublasGetMathMode(handle, &previous_mode));
    CHECK_CUBLAS(cublasSetMathMode(handle, mode));
}

CublasMathModeGuard::~CublasMathModeGuard() noexcept
{
    if (handle) cublasSetMathMode(handle, previous_mode);
}
#endif

namespace
{

DeviceStream create_stream_handle(unsigned flags)
{
#ifdef OPENNN_HAS_CUDA
    DeviceStream stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, flags));
    return stream;
#else
    (void)flags;
    return nullptr;
#endif
}

void destroy_stream_handle(DeviceStream stream) noexcept
{
    if (!stream) return;

#ifdef OPENNN_HAS_CUDA
    cudaStreamDestroy(stream);
#endif
}


}


namespace
{
thread_local int active_lane_index = 0;
}

int lanes_available() noexcept
{
    static const int lanes =
        int(clamp(env_int_or("OPENNN_LANES", 1), 1LL, static_cast<long long>(MAX_LANES)));

    return lanes;
}

int active_lane() noexcept
{
    return active_lane_index;
}

void set_active_lane(int lane)
{
    throw_if(lane < 0 || lane >= lanes_available(),
             "set_active_lane: lane {} outside the {} configured (OPENNN_LANES).", lane, lanes_available());
    active_lane_index = lane;
}

DeviceStream lane_stream(int lane)
{
    return Backend::instance().stream(lane);
}

DeviceStream get_compute_stream()
{
    return Backend::instance().stream(active_lane_index);
}

DeviceStream get_transfer_stream()
{
    Backend& backend = Backend::instance();
    backend.ensure_cuda();
    return backend.transfer_stream;
}

BlasHandle get_cublas_handle()
{
    return Backend::get_cublas_handle();
}

BlasLtHandle get_cublas_lt_handle()
{
    return Backend::get_cublas_lt_handle();
}

DnnHandle get_cudnn_handle()
{
    return Backend::get_cudnn_handle();
}

DnnOpTensorDescriptor get_op_tensor_add_descriptor()
{
    return Backend::get_op_tensor_add_descriptor();
}


void refresh_blas_math_mode()
{
    Backend::apply_math_mode();
}

}

namespace opennn
{

Backend::Backend()
{
    const char* const threads_env = getenv("OPENNN_THREADS");
    set_threads_number(threads_env ? atoi(threads_env) : 0);
}

void Backend::ensure_cuda()
{
#ifdef OPENNN_HAS_CUDA
    std::call_once(cuda_once, [this]
    {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count == 0)
        {
            cudaGetLastError();
            logging::warning() << "OpenNN: no CUDA device available (" << cudaGetErrorString(status)
                 << "); running on CPU.\n";
            return;
        }

        lane_streams[0] = device::create_stream_handle(cudaStreamNonBlocking);
        transfer_stream = device::create_stream_handle(cudaStreamNonBlocking);

        CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
        CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&op_tensor_add_descriptor));
        CHECK_CUDNN(cudnnSetOpTensorDescriptor(op_tensor_add_descriptor,
                                               CUDNN_OP_TENSOR_ADD,
                                               CUDNN_DATA_FLOAT,
                                               CUDNN_NOT_PROPAGATE_NAN));
        register_cuda_cleanup();
    });
#endif
}

DeviceStream Backend::stream(int lane)
{
#ifdef OPENNN_HAS_CUDA
    ensure_cuda();
    if (!lane_streams[0]) return nullptr;
    if (lane == 0) return lane_streams[0];
    std::lock_guard<std::mutex> lock(lane_mutex);
    if (!lane_streams[lane])
        lane_streams[lane] = device::create_stream_handle(cudaStreamNonBlocking);
    return lane_streams[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

cublasHandle_t Backend::cublas(int lane)
{
#ifdef OPENNN_HAS_CUDA
    DeviceStream lane_stream = stream(lane);
    if (!lane_stream) return nullptr;
    std::lock_guard<std::mutex> lock(lane_mutex);
    if (!cublas_handles[lane])
    {
        CHECK_CUBLAS(cublasCreate(&cublas_handles[lane]));
        CHECK_CUBLAS(cublasSetMathMode(cublas_handles[lane],
                                       device::allow_tf32() ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
        CHECK_CUBLAS(cublasSetStream(cublas_handles[lane], lane_stream));
        register_cuda_cleanup();
    }
    return cublas_handles[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

void Backend::set_math_mode_on_handles()
{
#ifdef OPENNN_HAS_CUDA
    std::lock_guard<std::mutex> lock(lane_mutex);
    for (cublasHandle_t handle : cublas_handles)
        if (handle)
            CHECK_CUBLAS(cublasSetMathMode(handle, device::allow_tf32() ? CUBLAS_TF32_TENSOR_OP_MATH
                                                                       : CUBLAS_DEFAULT_MATH));
#endif
}

cudnnHandle_t Backend::cudnn(int lane)
{
#ifdef OPENNN_HAS_CUDA
    DeviceStream lane_stream = stream(lane);
    if (!lane_stream) return nullptr;
    std::lock_guard<std::mutex> lock(lane_mutex);
    if (!cudnn_handles[lane])
    {
        CHECK_CUDNN(cudnnCreate(&cudnn_handles[lane]));
        CHECK_CUDNN(cudnnSetStream(cudnn_handles[lane], lane_stream));
        register_cuda_cleanup();
    }
    return cudnn_handles[lane];
#else
    (void)lane;
    return nullptr;
#endif
}

Backend::~Backend()
{
    release_cuda();
}

void Backend::register_cuda_cleanup()
{
#ifdef OPENNN_HAS_CUDA
    // The backend can predate CUDA: CPU work constructs it without a context.
    // Register after each lazy library initialization so our handles are freed
    // before that library's own process-exit callbacks tear down its runtime.
    std::atexit([] { Backend::instance().release_cuda(); });
#endif
}

void Backend::release_cuda()
{
#ifdef OPENNN_HAS_CUDA
    device::begin_cuda_shutdown();

    if (op_tensor_add_descriptor)
        cudnnDestroyOpTensorDescriptor(std::exchange(op_tensor_add_descriptor, nullptr));

    if (cublas_lt_handle)
        cublasLtDestroy(std::exchange(cublas_lt_handle, nullptr));

    for (int lane = 0; lane < device::MAX_LANES; ++lane)
    {
        if (cublas_handles[lane]) cublasDestroy(std::exchange(cublas_handles[lane], nullptr));
        if (cudnn_handles[lane])  cudnnDestroy(std::exchange(cudnn_handles[lane], nullptr));

        device::destroy_stream_handle(std::exchange(lane_streams[lane], nullptr));
    }

    device::destroy_stream_handle(std::exchange(transfer_stream, nullptr));
#endif
}

void Backend::set_threads_number(int num_threads)
{
    if (num_threads <= 0)
    {
        // Affinity first, machine size second. `hardware_concurrency` counts
        // the cores the machine has, not the ones this process may run on, so
        // under `taskset` or a cgroup CPU limit it oversubscribes -- 28
        // threads onto 16 permitted CPUs in the benchmark harness, which also
        // makes every per-thread sizing decision downstream come out wrong.
#ifdef __linux__
        cpu_set_t permitted;

        if (sched_getaffinity(0, sizeof(permitted), &permitted) == 0)
            num_threads = CPU_COUNT(&permitted);
#endif
        if (num_threads <= 0) num_threads = thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads <= 0) num_threads = 1;
    }

    thread_pool = make_unique<ThreadPool>(num_threads);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), num_threads);

    Eigen::setNbThreads(num_threads);
    omp_set_num_threads(num_threads);

    // Every parallel region must ask for the same team. libgomp keeps exactly
    // one pool of workers, sized to the last region: a region that wants fewer
    // threads makes the surplus exit, and the next full-size region creates
    // them again with `pthread_create` -- fresh stacks, cold caches, and a
    // barrier that waits for the kernel to schedule them. The LSTM forward
    // pass was paying six thread births and deaths per batch, 10% of its
    // throughput, from two sources that each looked harmless alone:
    //
    //  - `omp_set_dynamic(1)`, which lets libgomp size a team as the CPU
    //    count minus the fifteen-minute load average, so a desktop with a
    //    browser open gave OpenNN's own regions 10 of 16 threads while
    //    oneDNN, which pins dynamic off, asked for all 16;
    //  - MKL's own thread heuristic (`MKL_DYNAMIC`), which chose 10 threads
    //    for a 256x128 `sgemv` between two 16-thread oneDNN regions.
    //
    // Dynamic teams are opt-in (`OPENNN_OMP_DYNAMIC=1`) and MKL is told to use
    // this team, exactly as PyTorch's ATen does for the same reason.
    const char* const omp_dynamic = getenv("OPENNN_OMP_DYNAMIC");
    omp_set_dynamic(omp_dynamic ? atoi(omp_dynamic) : 0);
#if defined(_OPENMP) && _OPENMP >= 200805
    omp_set_max_active_levels(1);
#endif
#ifdef EIGEN_USE_MKL_ALL
    mkl_set_dynamic(0);
    mkl_set_num_threads(num_threads);
#endif
}

Backend& Backend::instance()
{
    static Backend backend;
    return backend;
}

ThreadPoolDevice* Backend::get_thread_pool_device()
{
    return thread_pool_device.get();
}

ThreadPoolDevice& get_device()
{
    return *Backend::instance().get_thread_pool_device();
}

void set_threads_number(const int threads_number)
{
    Backend::instance().set_threads_number(threads_number);
}

}
