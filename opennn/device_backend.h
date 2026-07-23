//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "configuration.h"

namespace opennn::device
{

enum class CopyKind
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

constexpr bool is_cuda_build() noexcept
{
#ifdef OPENNN_HAS_CUDA
    return true;
#else
    return false;
#endif
}
bool has_cuda_device() noexcept;
int cuda_compute_capability() noexcept;
size_t available_memory();
string gpu_info_string() noexcept;
bool cuda_allocation_growth_forbidden() noexcept;

int64_t conv_workspace_limit_bytes() noexcept;
void    set_conv_workspace_cap(int64_t mode) noexcept;
void    set_conv_workspace_auto_limit_bytes(int64_t) noexcept;

bool conv_autotune_enabled() noexcept;
void set_conv_autotune(bool) noexcept;


class CudaAllocationGrowthGuard
{
public:
    explicit CudaAllocationGrowthGuard(bool);
    ~CudaAllocationGrowthGuard() noexcept;

    CudaAllocationGrowthGuard(const CudaAllocationGrowthGuard&) = delete;
    CudaAllocationGrowthGuard& operator=(const CudaAllocationGrowthGuard&) = delete;

private:
    bool active = false;
    bool previous = false;
};

void* allocate(Device, Index);
void deallocate(Device, void*, Index);

void set_zero(void*, Index, Device);
void set_zero_async(void*, Index, cudaStream_t = nullptr);

void copy_async(void*, const void*, Index, CopyKind, cudaStream_t = nullptr);
void copy_async(void*, const void*, Index, Device, Device, cudaStream_t = nullptr);
void synchronize(cudaStream_t = nullptr);
void check_last_error();
void reset_last_error() noexcept;

#ifdef OPENNN_HAS_CUDA
struct CublasPointerModeGuard
{
    cublasHandle_t handle = nullptr;
    cublasPointerMode_t previous_mode = CUBLAS_POINTER_MODE_HOST;

    CublasPointerModeGuard(cublasHandle_t new_handle, cublasPointerMode_t mode)
        : handle(new_handle)
    {
        CHECK_CUBLAS(cublasGetPointerMode(handle, &previous_mode));
        CHECK_CUBLAS(cublasSetPointerMode(handle, mode));
    }

    CublasPointerModeGuard(const CublasPointerModeGuard&) = delete;
    CublasPointerModeGuard& operator=(const CublasPointerModeGuard&) = delete;

    ~CublasPointerModeGuard() noexcept
    {
        if (handle) cublasSetPointerMode(handle, previous_mode);
    }
};
#endif

void* allocate_pinned_host(Index);
void deallocate_pinned_host(void*);

cudaEvent_t create_event(unsigned);
cudaEvent_t create_event();
void destroy_event(cudaEvent_t);
void record_event(cudaEvent_t, cudaStream_t);
void synchronize_event(cudaEvent_t);
void stream_wait_event(cudaStream_t, cudaEvent_t);

void destroy_graph(cudaGraph_t) noexcept;
void destroy_graph_exec(cudaGraphExec_t) noexcept;

struct GraphDeleter     { void operator()(remove_pointer_t<cudaGraph_t>* graph)    const noexcept { destroy_graph(graph); } };
struct GraphExecDeleter { void operator()(remove_pointer_t<cudaGraphExec_t>* exec) const noexcept { destroy_graph_exec(exec); } };

using GraphHandle     = unique_ptr<remove_pointer_t<cudaGraph_t>,     GraphDeleter>;
using GraphExecHandle = unique_ptr<remove_pointer_t<cudaGraphExec_t>, GraphExecDeleter>;

class StreamCapture
{
public:
    explicit StreamCapture(cudaStream_t);
    ~StreamCapture() noexcept;

    StreamCapture(const StreamCapture&) = delete;
    StreamCapture& operator=(const StreamCapture&) = delete;

    GraphHandle end();

private:
    cudaStream_t stream = nullptr;
    bool finished = false;
};

void instantiate_or_update(GraphExecHandle&, cudaGraph_t);
void launch_graph(const GraphExecHandle&, cudaStream_t);

cudaStream_t get_compute_stream();

}

namespace opennn
{

struct CudaEvent
{
    cudaEvent_t handle = nullptr;

    CudaEvent() = default;
    explicit CudaEvent(unsigned flags) { handle = device::create_event(flags); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : handle(other.handle) { other.handle = nullptr; }
    CudaEvent& operator=(CudaEvent&& other) noexcept
    {
        if (this != &other) { destroy(); handle = other.handle; other.handle = nullptr; }
        return *this;
    }

    ~CudaEvent() { destroy(); }

    void create()
    {
        destroy();
        handle = device::create_event();
    }

    void create(unsigned flags)
    {
        destroy();
        handle = device::create_event(flags);
    }

    void destroy() noexcept
    {
        device::destroy_event(handle);
        handle = nullptr;
    }

    operator cudaEvent_t() const noexcept { return handle; }
    explicit operator bool() const noexcept { return handle != nullptr; }
};

class Backend
{
public:

    static Backend& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int);

    static cublasHandle_t get_cublas_handle()                      { return instance().cublas(); }
    static cublasLtHandle_t get_cublas_lt_handle()                 { return instance().cuda.cublas_lt_handle; }
    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn(); }
    static cudaStream_t get_transfer_stream()                      { return instance().cuda.transfer_stream; }
    static cudnnOpTensorDescriptor_t get_operator_sum_descriptor() { instance().cudnn(); return instance().cuda.operator_sum_descriptor; }

private:
    Backend();
    ~Backend();

    cublasHandle_t cublas();
    cudnnHandle_t cudnn();
    friend cudaStream_t device::get_compute_stream();

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    struct CudaResources
    {
        cublasHandle_t cublas_handle = nullptr;
        cublasLtHandle_t cublas_lt_handle = nullptr;
        cudnnHandle_t cudnn_handle = nullptr;
        cudaStream_t compute_stream = nullptr;
        cudaStream_t transfer_stream = nullptr;
        cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
        once_flag cublas_init_once;
        once_flag cudnn_init_once;
    };

    CudaResources cuda;
};

inline ThreadPoolDevice& get_device()
{
    return *Backend::instance().get_thread_pool_device();
}

struct TensorView;

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
