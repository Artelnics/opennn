//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>
#include <mutex>

#include "opennn/core/opennn_types.h"
#include "opennn/core/configuration.h"

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
void set_cuda_allocation_growth_forbidden(bool) noexcept;

enum class GraphWorkspaceKind
{
    SharedScratch,
    Bf16Input,
    Bf16Gradient,
    Bf16ToFp32,
    Int8Dequant,
    PoolingMask,
    NormPartials,
    GradientPartials,
    Count
};

inline constexpr std::array<const char*, static_cast<size_t>(GraphWorkspaceKind::Count)>
graph_workspace_labels = {"shared_scratch", "bf16_input", "bf16_gradient",
                          "bf16_to_fp32", "int8_dequant", "pooling_mask", "norm_partials",
                          "gradient_partials"};

struct GraphWorkspaceView
{
    void* data = nullptr;
    Index bytes = 0;
};

using GraphWorkspaceRequirements = std::array<Index, static_cast<size_t>(GraphWorkspaceKind::Count)>;
using GraphWorkspaceViews = std::array<GraphWorkspaceView, static_cast<size_t>(GraphWorkspaceKind::Count)>;

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

// Diagnostic kernel choices ("rungs"): Auto in production; the other values
// pin one path so a gradient or an output can be checked against the reference
// on purpose (a shape may never take a rung by itself on a given GPU), and so
// the benchmark harness can A/B them. Read with rung<R>(), set with set_rung().
//
// Batch-norm backward: Auto takes cuDNN's fully fused engine when the shape
// has one and the library's own fused kernel (batchnorm_backward_fused_cuda)
// otherwise.
enum class BatchNormBackwardRung { Auto, StagedFp32, PlainNative, OwnKernel };
// Batch-norm training forward: Auto takes the library's own kernel
// (batchnorm_forward_fused_cuda) wherever it can leave the packed ReLU mask
// the backward reads in place of Y - a ReLU output with a channel count that
// is a multiple of 8, i.e. every BN of a ResNet - and cuDNN's fused graph
// elsewhere.
enum class BatchNormForwardRung { Auto, CudnnGraph, OwnKernel };
// Max pooling on CUDA: Auto takes the library's own forward + argmax-mask
// backward (max_pooling_forward_cuda / max_pooling_backward_cuda) in training,
// where the mask slot exists, and cuDNN's pooling elsewhere (inference,
// average pooling, windows above 255 elements).
enum class MaxPoolingRung { Auto, Cudnn, OwnKernel };

template<typename Rung> Rung rung() noexcept;
template<typename Rung> void set_rung(Rung) noexcept;

class CudaAllocationGrowthGuard
{
public:

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

#ifdef OPENNN_HAS_CUDA
struct GraphExecDeleter { void operator()(cudaGraphExec_t exec) const noexcept { cudaGraphExecDestroy(exec); } };
#else
struct GraphExecDeleter { void operator()(void*) const noexcept {} };
#endif

using GraphExecHandle = unique_ptr<remove_pointer_t<cudaGraphExec_t>, GraphExecDeleter>;

class StreamCapture
{
public:
    explicit StreamCapture(cudaStream_t);
    ~StreamCapture() noexcept;

    StreamCapture(const StreamCapture&) = delete;
    StreamCapture& operator=(const StreamCapture&) = delete;

    void end(GraphExecHandle&);

private:
#ifdef OPENNN_HAS_CUDA
    cudaStream_t stream = nullptr;
    bool finished = false;
#endif
};

void launch_graph(const GraphExecHandle&, cudaStream_t);

// Execution lanes. Lane 0 is the compute stream every kernel and cuDNN/cuBLAS
// call has always run on; lanes 1.. are extra streams, each with its own
// cuDNN/cuBLAS handles and thread scratch, for work the schedule forks off -
// kernels on different lanes run concurrently, inside a captured graph too
// (a forked lane joins the capture through the fork event and must be joined
// back before the capture ends). The active lane is a thread-local index the
// accessors below read; a scheduler sets it around the ops it forks and
// restores it. lanes_available() is the configured count (OPENNN_LANES,
// default 1: no forking anywhere).
constexpr int MAX_LANES = 4;
int lanes_available() noexcept;
int active_lane() noexcept;
void set_active_lane(int lane);
cudaStream_t lane_stream(int lane);

// The active lane's stream; host<->device staging copies use the transfer
// stream, joined to compute with events (see Batch::wait_h2d_on_compute_stream).
cudaStream_t get_compute_stream();
cudaStream_t get_transfer_stream();

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

    ~CudaEvent() { destroy(); }

    void create()
    {
        destroy();
        handle = device::create_event();
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

    // Handles of the active lane (see device::active_lane).
    static cublasHandle_t get_cublas_handle()                      { return instance().cublas(device::active_lane()); }
    static cublasLtHandle_t get_cublas_lt_handle()                 { return instance().cublas_lt_handle; }

    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn(device::active_lane()); }
    static cudnnOpTensorDescriptor_t get_op_tensor_add_descriptor()
    {
        Backend& backend = instance();
        backend.cudnn(0);
        return backend.op_tensor_add_descriptor;
    }

private:
    Backend();
    ~Backend();

    cublasHandle_t cublas(int lane);
    cudnnHandle_t cudnn(int lane);
    cudaStream_t stream(int lane);

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    cublasLtHandle_t cublas_lt_handle = nullptr;
    cudnnOpTensorDescriptor_t op_tensor_add_descriptor = nullptr;

    // Per lane; lane 0 is created with the backend, the others on first use.
    std::mutex lane_mutex;
    std::array<cudaStream_t, device::MAX_LANES>   lane_streams{};
    std::array<cublasHandle_t, device::MAX_LANES> cublas_handles{};
    std::array<cudnnHandle_t, device::MAX_LANES>  cudnn_handles{};

    cudaStream_t transfer_stream = nullptr;

    friend cudaStream_t device::get_compute_stream();
    friend cudaStream_t device::get_transfer_stream();
    friend cudaStream_t device::lane_stream(int);
};

inline ThreadPoolDevice& get_device()
{
    return *Backend::instance().get_thread_pool_device();
}

struct TensorView;

// Per-thread device scratch, one growable buffer per GraphWorkspaceKind
// (a captured inference graph pins them through CudaGraphWorkspaceScope).
// Growth is forbidden while a CudaAllocationGrowthGuard is active - warm-up
// must have sized them.
void* ensure_workspace_bytes(device::GraphWorkspaceKind, Index bytes);
template<typename T>
T* ensure_workspace(device::GraphWorkspaceKind kind, Index count)
{
    return static_cast<T*>(ensure_workspace_bytes(kind, count * Index(sizeof(T))));
}
inline bfloat16* ensure_bf16_gradient_workspace(Index n) { return ensure_workspace<bfloat16>(device::GraphWorkspaceKind::Bf16Gradient, n); }
inline bfloat16* ensure_int8_dequant_workspace(Index n)  { return ensure_workspace<bfloat16>(device::GraphWorkspaceKind::Int8Dequant, n); }
inline float*    ensure_bf16_to_fp32_workspace(Index n)  { return ensure_workspace<float>(device::GraphWorkspaceKind::Bf16ToFp32, n); }
inline void*     ensure_shared_scratch(size_t bytes)     { return ensure_workspace_bytes(device::GraphWorkspaceKind::SharedScratch, Index(bytes)); }

void release_thread_workspaces();

const void* data_for_gemm_dtype(const TensorView&, Type);

const void* bias_for_gemm_bf16(const TensorView&);

// D = epilogue(A * B [+ addend]): with `addend` the matmul reads it as C with
// beta = 1 (same layout as D), so a sum that would otherwise be a separate
// pass costs one read inside the epilogue.
void run_lt_matmul_cached(
    int, int, int,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    const void*, const void*, void*,
    const void*,
    cudaDataType_t io_dtype  = CUDA_R_32F,
    cudaDataType_t out_dtype = CUDA_R_32F,
    const void* aux_pointer  = nullptr,
    const void* addend       = nullptr);

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
