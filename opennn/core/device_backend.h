//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>

#include "opennn/core/opennn_types.h"
#include "opennn/core/configuration.h"

namespace opennn
{

template<typename Handle>
struct CudnnDescriptor
{
    Handle handle = nullptr;
#ifdef OPENNN_HAS_CUDA
    cudnnStatus_t (*deleter)(Handle) = nullptr;
#else
    void (*deleter)(Handle) = nullptr;
#endif

    CudnnDescriptor() = default;

    CudnnDescriptor(CudnnDescriptor&& other) noexcept
        : handle(other.handle), deleter(other.deleter)
    {
        other.handle = nullptr;
        other.deleter = nullptr;
    }

    CudnnDescriptor& operator=(CudnnDescriptor&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle = other.handle;
            deleter = other.deleter;
            other.handle = nullptr;
            other.deleter = nullptr;
        }
        return *this;
    }

    CudnnDescriptor(const CudnnDescriptor&) = delete;
    CudnnDescriptor& operator=(const CudnnDescriptor&) = delete;

    ~CudnnDescriptor() { reset(); }

    void reset()
    {
        if (handle && deleter) deleter(handle);
        handle = nullptr;
        deleter = nullptr;
    }

    Handle get() const noexcept { return handle; }
    operator Handle() const noexcept { return handle; }
    explicit operator bool() const noexcept { return handle != nullptr; }
};

}

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
    FlashAttention,
    Count
};

inline constexpr std::array<const char*, static_cast<size_t>(GraphWorkspaceKind::Count)>
graph_workspace_labels = {"shared_scratch", "bf16_input", "bf16_gradient",
                          "bf16_to_fp32", "int8_dequant", "pooling_mask", "norm_partials",
                          "gradient_partials", "flash_attention"};

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

    CudaGraphWorkspaceScope(const CudaGraphWorkspaceScope&) = delete;
    CudaGraphWorkspaceScope& operator=(const CudaGraphWorkspaceScope&) = delete;

    ~CudaGraphWorkspaceScope() noexcept;

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
// Scaled dot-product attention: Auto takes FlashAttention-2 wherever the build
// has a kernel for the shape and the mask lets it (core/cuda/flash_attention.cuh
// says which those are) and cuDNN's fused graph everywhere else, which is
// everything on a build without FA2 kernels. CudnnGraph pins cuDNN, which is
// the other half of an A/B; FlashAttention asks for FA2 but cannot promise it,
// since a layer it does not cover - a causal mask over a padded batch, say -
// still has to compute, so a measurement under it should read the call count
// (flash_attention::call_count) rather than assume.
enum class AttentionRung { Auto, CudnnGraph, FlashAttention };

template<typename Rung> Rung rung() noexcept;
template<typename Rung> void set_rung(Rung) noexcept;

class CudaAllocationGrowthGuard
{
public:

    explicit CudaAllocationGrowthGuard(bool,
                                       bool forbid_matmul_plan_creation = true);

    CudaAllocationGrowthGuard(const CudaAllocationGrowthGuard&) = delete;
    CudaAllocationGrowthGuard& operator=(const CudaAllocationGrowthGuard&) = delete;

    ~CudaAllocationGrowthGuard() noexcept;

private:
    bool active = false;
    bool previous = false;
    bool guard_matmul_plans = false;
    bool previous_matmul_plan_guard = false;
};

void* allocate(Device, Index);
// noexcept: called from Buffer/PinnedBuffer destructors, where a throw would
// terminate instead of reporting the CUDA error that caused it.
void deallocate(Device, void*, Index) noexcept;

void set_zero(void*, Index, Device);
void set_zero_async(void*, Index, cudaStream_t = nullptr);

void copy_async(void*, const void*, Index, CopyKind, cudaStream_t = nullptr);
void copy_async(void*, const void*, Index, Device, Device, cudaStream_t = nullptr);
void synchronize(cudaStream_t = nullptr);
void check_last_error();
void reset_last_error() noexcept;

#ifdef OPENNN_HAS_CUDA
class CublasPointerModeGuard
{
public:

    CublasPointerModeGuard(cublasHandle_t, cublasPointerMode_t);

    CublasPointerModeGuard(const CublasPointerModeGuard&) = delete;
    CublasPointerModeGuard& operator=(const CublasPointerModeGuard&) = delete;

    ~CublasPointerModeGuard() noexcept;

private:

    cublasHandle_t handle = nullptr;
    cublasPointerMode_t previous_mode = CUBLAS_POINTER_MODE_HOST;
};

class CublasMathModeGuard
{
public:

    CublasMathModeGuard(cublasHandle_t, cublasMath_t);

    CublasMathModeGuard(const CublasMathModeGuard&) = delete;
    CublasMathModeGuard& operator=(const CublasMathModeGuard&) = delete;

    ~CublasMathModeGuard() noexcept;

private:

    cublasHandle_t handle = nullptr;
    cublasMath_t previous_mode = CUBLAS_DEFAULT_MATH;
};
#endif

class PinnedBuffer
{
public:

    PinnedBuffer() = default;
    explicit PinnedBuffer(Index byte_count);

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&&) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&&) noexcept;

    ~PinnedBuffer() noexcept;

    void resize_bytes(Index);
    void grow_to(Index);

    void* data() noexcept { return pointer; }
    const void* data() const noexcept { return pointer; }

    template<typename T>
    T* as() noexcept { return static_cast<T*>(pointer); }

    template<typename T>
    const T* as() const noexcept { return static_cast<const T*>(pointer); }

    Index byte_size() const noexcept { return allocated_bytes; }
    bool empty() const noexcept { return allocated_bytes == 0; }
    explicit operator bool() const noexcept { return pointer != nullptr; }

private:

    void reset() noexcept;
    void swap(PinnedBuffer&) noexcept;

    void* pointer = nullptr;
    Index allocated_bytes = 0;
};

void record_event(cudaEvent_t, cudaStream_t);
void synchronize_event(cudaEvent_t);
void stream_wait_event(cudaStream_t, cudaEvent_t);

class CudaEvent
{
public:

    CudaEvent() = default;
    explicit CudaEvent(unsigned flags);

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&&) noexcept;
    CudaEvent& operator=(CudaEvent&&) noexcept;

    ~CudaEvent() noexcept;

    void create();

    cudaEvent_t get() const noexcept { return handle; }
    explicit operator bool() const noexcept { return handle != nullptr; }

private:

    void reset() noexcept;

    cudaEvent_t handle = nullptr;
};

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

    StreamCapture(const StreamCapture&) = delete;
    StreamCapture& operator=(const StreamCapture&) = delete;

    ~StreamCapture() noexcept;

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

cublasHandle_t get_cublas_handle();
cublasLtHandle_t get_cublas_lt_handle();
cudnnHandle_t get_cudnn_handle();
cudnnOpTensorDescriptor_t get_op_tensor_add_descriptor();

}

namespace opennn
{

ThreadPoolDevice& get_device();
void set_threads_number(int);

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
// The query-delta accumulator and the softmax delta sums FlashAttention-2's
// backward writes; one buffer for both, and one buffer for every attention
// layer, since only one of them is inside its backward at a time.
inline float*    ensure_flash_attention_workspace(Index n) { return ensure_workspace<float>(device::GraphWorkspaceKind::FlashAttention, n); }

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
