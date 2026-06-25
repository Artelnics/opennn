//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "types.h"

namespace opennn::device
{

enum class CopyKind
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

bool is_cuda_build() noexcept;
bool has_cuda_device() noexcept;
int cuda_compute_capability() noexcept;
size_t available_memory();
bool cuda_allocation_growth_forbidden() noexcept;
void set_cuda_allocation_growth_forbidden(bool) noexcept;

// GPU backend toggles, all controlled from code (no environment variables).
// cuBLASLt scratch (workspace) growth lock, independent of the allocation lock.
bool cuda_scratch_growth_forbidden() noexcept;
void set_cuda_scratch_growth_forbidden(bool) noexcept;

// GEMM (cuBLASLt) algorithm autotuning. Off by default: keep cuBLASLt's first
// heuristic. Enable to time several candidates on the first matmul.
bool gemm_autotune_enabled() noexcept;
void set_gemm_autotune(bool) noexcept;

// bf16 cuBLAS compute type. Off by default uses the fast tensor-core path
// (COMPUTE_32F_FAST_16BF); enable "plain" to fall back to COMPUTE_32F.
bool bf16_compute_plain() noexcept;
void set_bf16_compute_plain(bool) noexcept;

// cuDNN convolution/batchnorm engine selection (read by the cudnn-frontend path).
// Autotune is on by default; conv_legacy forces the legacy v7 API.
bool conv_autotune_enabled() noexcept;
void set_conv_autotune(bool) noexcept;
bool conv_legacy_forced() noexcept;
void set_conv_legacy(bool) noexcept;

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

// RAII over a stream-capture window: if the captured body throws before end(),
// the destructor closes the capture and discards the orphaned graph instead of
// leaving the stream stuck in capture mode.
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

// Refreshes an instantiated graph in place via cudaGraphExecUpdate when the
// topology is unchanged; falls back to a full re-instantiation otherwise.
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

    static cublasHandle_t get_cublas_handle()                      { return instance().cublas_handle; }
    static cublasLtHandle_t get_cublas_lt_handle()                 { return instance().cublas_lt_handle; }
    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn_handle; }
    static cudaStream_t get_compute_stream()                       { return instance().compute_stream; }
    static cudaStream_t get_transfer_stream()                      { return instance().transfer_stream; }
    static cudnnOpTensorDescriptor_t get_operator_sum_descriptor() { return instance().operator_sum_descriptor; }

private:
    Backend();
    ~Backend();

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    cublasHandle_t cublas_handle = nullptr;
    cublasLtHandle_t cublas_lt_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
};

inline ThreadPoolDevice& get_device()
{
    return *Backend::instance().get_thread_pool_device();
}

struct TensorView;

bfloat16* ensure_bf16_gradient_workspace(Index);

float* ensure_bf16_to_fp32_workspace(Index);

void* ensure_cudnn_conv_workspace(size_t);

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
    cudaDataType_t out_dtype = CUDA_R_32F);

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
