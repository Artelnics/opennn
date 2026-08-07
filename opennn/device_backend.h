//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>

#include "opennn_types.h"
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
std::string gpu_info_string() noexcept;
bool cuda_allocation_growth_forbidden() noexcept;
void set_cuda_allocation_growth_forbidden(bool) noexcept;
bool cuda_matmul_plan_creation_forbidden() noexcept;

enum class GraphWorkspaceKind
{
    SharedScratch,
    Bf16Input,
    Bf16Gradient,
    Bf16ToFp32,
    Int8Dequant,
    Count
};

// Ledger names, indexed by kind. They live next to the enum so that adding a
// workspace means editing one place, not two files.
inline constexpr std::array<const char*, size_t(GraphWorkspaceKind::Count)>
graph_workspace_labels = {"shared_scratch", "bf16_input", "bf16_gradient",
                          "bf16_to_fp32", "int8_dequant"};

using GraphWorkspaceRequirements = std::array<Index, size_t(GraphWorkspaceKind::Count)>;

struct GraphWorkspaceView
{
    void* data = nullptr;
    Index bytes = 0;
};

using GraphWorkspaceViews = std::array<GraphWorkspaceView, size_t(GraphWorkspaceKind::Count)>;

class CudaGraphWorkspaceScope
{
public:
    explicit CudaGraphWorkspaceScope(GraphWorkspaceRequirements&,
                                     const GraphWorkspaceViews* = nullptr);
    ~CudaGraphWorkspaceScope() noexcept;

    CudaGraphWorkspaceScope(const CudaGraphWorkspaceScope&) = delete;
    CudaGraphWorkspaceScope& operator=(const CudaGraphWorkspaceScope&) = delete;

private:
    GraphWorkspaceRequirements* previous_requirements = nullptr;
    const GraphWorkspaceViews* previous_views = nullptr;
    GraphWorkspaceViews owned_views;
};

int64_t conv_workspace_limit_bytes() noexcept;
void    set_conv_workspace_cap(int64_t mode) noexcept;
void    set_conv_workspace_auto_limit_bytes(int64_t) noexcept;

bool conv_autotune_enabled() noexcept;
void set_conv_autotune(bool) noexcept;

class CudaAllocationGrowthGuard
{
public:
    // CUDA graph capture also forbids creating new cuBLASLt host plans.
    // Tests that only assert stable device buffers can leave plan creation
    // enabled with the second argument.
    explicit CudaAllocationGrowthGuard(bool,
                                       bool forbid_matmul_plan_creation = true);
    ~CudaAllocationGrowthGuard() noexcept;

    CudaAllocationGrowthGuard(const CudaAllocationGrowthGuard&) = delete;
    CudaAllocationGrowthGuard& operator=(const CudaAllocationGrowthGuard&) = delete;

private:
    bool active = false;
    bool previous = false;
    bool guard_matmul_plans = false;
    bool previous_matmul_plan_guard = false;
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

cudaStream_t create_stream(unsigned);
void destroy_stream(cudaStream_t);

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

struct GraphDeleter     { void operator()(std::remove_pointer_t<cudaGraph_t>* graph)    const noexcept { destroy_graph(graph); } };
struct GraphExecDeleter { void operator()(std::remove_pointer_t<cudaGraphExec_t>* exec) const noexcept { destroy_graph_exec(exec); } };

using GraphHandle     = std::unique_ptr<std::remove_pointer_t<cudaGraph_t>,     GraphDeleter>;
using GraphExecHandle = std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, GraphExecDeleter>;

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
    static cublasLtHandle_t get_cublas_lt_handle()                 { return instance().cublas_lt_handle; }

    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn(); }
    static cudaStream_t get_compute_stream()                       { return instance().compute_stream; }
    static cudaStream_t get_transfer_stream()                      { return instance().transfer_stream; }
    static cudnnOpTensorDescriptor_t get_operator_sum_descriptor() { instance().cudnn(); return instance().operator_sum_descriptor; }

private:
    Backend();
    ~Backend();

    cublasHandle_t cublas();
    cudnnHandle_t cudnn();

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    cublasHandle_t cublas_handle = nullptr;
    cublasLtHandle_t cublas_lt_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
    once_flag cublas_init_once;
    once_flag cudnn_init_once;
};

inline ThreadPoolDevice& get_device()
{
    return *Backend::instance().get_thread_pool_device();
}

struct TensorView;

bfloat16* ensure_bf16_gradient_workspace(Index);

bfloat16* ensure_int8_dequant_workspace(Index);

float* ensure_bf16_to_fp32_workspace(Index);

void* ensure_cudnn_conv_workspace(size_t);

// After a CUDA-graph capture the graph owns same-sized workspace buffers, so
// the eager thread-local set is redundant until the graph is invalidated (the
// ensure_* helpers regrow it on demand).
void release_matmul_thread_workspaces();

const void* data_for_gemm_dtype(const TensorView&, Type);

const void* bias_for_gemm_bf16(const TensorView&);

void run_lt_matmul_cached(
    int, int, int,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    const void*, const void*, void*,
    const void*,
    cudaDataType_t io_dtype  = CUDA_R_32F,
    cudaDataType_t out_dtype = CUDA_R_32F,
    const void* aux_pointer  = nullptr);

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int, int, int,
                               const void*, cudaDataType_t Atype, int, long long stride_a,
                               const void*, cudaDataType_t Btype, int, long long stride_b,
                               void*, cudaDataType_t Ctype, int, long long stride_c,
                               int,
                               float alpha = 1.0f, float beta = 0.0f);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
