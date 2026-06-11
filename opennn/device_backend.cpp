//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E V I C E   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "device_backend.h"
#include "tensor_types.h"
#include "string_utilities.h"

#include <atomic>

namespace opennn::device
{

namespace
{

std::atomic_bool cuda_allocation_growth_forbidden_runtime{false};

void throw_if_auto(Device device_type)
{
    throw_if(device_type == Device::Auto,
             "device backend expects a resolved device.");
}

CopyKind copy_kind(Device source, Device target)
{
    throw_if_auto(source);
    throw_if_auto(target);

    if (source == Device::CUDA && target == Device::CUDA) return CopyKind::DeviceToDevice;
    if (source == Device::CUDA) return CopyKind::DeviceToHost;
    if (target == Device::CUDA) return CopyKind::HostToDevice;

    return CopyKind::HostToHost;
}

#ifndef OPENNN_HAS_CUDA
[[noreturn]] void throw_cuda_unavailable()
{
    throw runtime_error("CUDA support is not compiled in.");
}
#endif

#ifdef OPENNN_HAS_CUDA

cudaMemcpyKind to_cuda_copy_kind(CopyKind kind)
{
    switch (kind)
    {
        case CopyKind::HostToHost:     return cudaMemcpyHostToHost;
        case CopyKind::HostToDevice:   return cudaMemcpyHostToDevice;
        case CopyKind::DeviceToHost:   return cudaMemcpyDeviceToHost;
        case CopyKind::DeviceToDevice: return cudaMemcpyDeviceToDevice;
    }

    throw runtime_error("unsupported CUDA copy kind.");
}

#endif

}

bool is_cuda_build() noexcept
{
#ifdef OPENNN_HAS_CUDA
    return true;
#else
    return false;
#endif
}

bool has_cuda_device() noexcept
{
#ifdef OPENNN_HAS_CUDA
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }

    return count > 0;
#else
    return false;
#endif
}

int cuda_compute_capability() noexcept
{
#ifdef OPENNN_HAS_CUDA
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
    {
        cudaGetLastError();
        return -1;
    }

    return properties.major * 10 + properties.minor;
#else
    return -1;
#endif
}

size_t available_memory()
{
#ifdef OPENNN_HAS_CUDA
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
#else
    throw_cuda_unavailable();
#endif
}

bool cuda_allocation_growth_forbidden() noexcept
{
    return cuda_allocation_growth_forbidden_runtime.load(std::memory_order_relaxed)
        || env_flag_enabled("OPENNN_CUDA_NO_ALLOC_GROWTH");
}

void set_cuda_allocation_growth_forbidden(bool forbidden) noexcept
{
    cuda_allocation_growth_forbidden_runtime.store(forbidden, std::memory_order_relaxed);
}

CudaAllocationGrowthGuard::CudaAllocationGrowthGuard(bool enabled)
    : active(enabled && is_cuda_build())
{
    if (active)
    {
        previous = cuda_allocation_growth_forbidden();
        set_cuda_allocation_growth_forbidden(true);
    }
}

CudaAllocationGrowthGuard::~CudaAllocationGrowthGuard() noexcept
{
    if (active)
        set_cuda_allocation_growth_forbidden(previous);
}

void* allocate(Device device_type, Index byte_count)
{
    throw_if_auto(device_type);
    throw_if(byte_count < 0, "device allocation size cannot be negative.");

    if (byte_count == 0) return nullptr;

    if (device_type == Device::CUDA)
    {
#ifdef OPENNN_HAS_CUDA
        throw_if(cuda_allocation_growth_forbidden(),
                 format("CUDA allocation of {} bytes while CUDA allocation growth is forbidden "
                        "(warmup incomplete before CUDA graph capture).",
                        byte_count));
        void* device_pointer = nullptr;
        CHECK_CUDA(cudaMalloc(&device_pointer, static_cast<size_t>(byte_count)));
        return device_pointer;
#else
        throw_cuda_unavailable();
#endif
    }

    return Eigen::aligned_allocator<uint8_t>{}.allocate(static_cast<size_t>(byte_count));
}

void deallocate(Device device_type, void* pointer, Index byte_count)
{
    if (!pointer) return;

    throw_if_auto(device_type);

    if (device_type == Device::CUDA)
    {
#ifdef OPENNN_HAS_CUDA
        cudaFree(pointer);
#endif
        return;
    }

    Eigen::aligned_allocator<uint8_t>{}.deallocate(static_cast<uint8_t*>(pointer),
                                                   static_cast<size_t>(byte_count));
}

void set_zero(void* data, Index byte_count, Device device_type)
{
    throw_if_auto(device_type);
    throw_if(byte_count < 0, "device memset size cannot be negative.");

    if (!data || byte_count == 0) return;

    if (device_type == Device::CUDA)
    {
#ifdef OPENNN_HAS_CUDA
        CHECK_CUDA(cudaMemset(data, 0, static_cast<size_t>(byte_count)));
#else
        throw_cuda_unavailable();
#endif
        return;
    }

    memset(data, 0, static_cast<size_t>(byte_count));
}

void set_zero_async(void* data, Index byte_count, cudaStream_t stream)
{
    throw_if(byte_count < 0, "device async memset size cannot be negative.");

    if (!data || byte_count == 0) return;

#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(stream ? cudaMemsetAsync(data, 0, static_cast<size_t>(byte_count), stream)
                      : cudaMemset(data, 0, static_cast<size_t>(byte_count)));
#else
    (void)stream;
    memset(data, 0, static_cast<size_t>(byte_count));
#endif
}

void copy_async(void* destination,
                const void* source,
                Index byte_count,
                CopyKind kind,
                cudaStream_t stream)
{
    throw_if(byte_count < 0, "device copy size cannot be negative.");

    if (byte_count == 0 || !destination || !source) return;

#ifdef OPENNN_HAS_CUDA
    const cudaMemcpyKind cuda_kind = to_cuda_copy_kind(kind);

    if (stream)
        CHECK_CUDA(cudaMemcpyAsync(destination, source,
                                   static_cast<size_t>(byte_count),
                                   cuda_kind,
                                   stream));
    else
        CHECK_CUDA(cudaMemcpy(destination, source,
                              static_cast<size_t>(byte_count),
                              cuda_kind));
#else
    (void)stream;
    if (kind != CopyKind::HostToHost) throw_cuda_unavailable();
    memcpy(destination, source, static_cast<size_t>(byte_count));
#endif
}

void copy_async(void* destination,
                const void* source,
                Index byte_count,
                Device source_device,
                Device target_device,
                cudaStream_t stream)
{
    copy_async(destination, source, byte_count, copy_kind(source_device, target_device), stream);
}

void synchronize(cudaStream_t stream)
{
#ifdef OPENNN_HAS_CUDA
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());
#else
    (void)stream;
#endif
}

void check_last_error()
{
#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(cudaPeekAtLastError());
#endif
}

cudaStream_t create_stream(unsigned flags)
{
#ifdef OPENNN_HAS_CUDA
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, flags));
    return stream;
#else
    (void)flags;
    return nullptr;
#endif
}

void destroy_stream(cudaStream_t stream)
{
    if (!stream) return;

#ifdef OPENNN_HAS_CUDA
    cudaStreamDestroy(stream);
#endif
}

void* allocate_pinned_host(Index byte_count)
{
    throw_if(byte_count < 0, "pinned host allocation size cannot be negative.");

    if (byte_count == 0) return nullptr;

#ifdef OPENNN_HAS_CUDA
    void* host_pointer = nullptr;
    CHECK_CUDA(cudaMallocHost(&host_pointer, static_cast<size_t>(byte_count)));
    return host_pointer;
#else
    void* host_pointer = malloc(static_cast<size_t>(byte_count));
    if (!host_pointer) throw bad_alloc();
    return host_pointer;
#endif
}

void deallocate_pinned_host(void* pointer)
{
    if (!pointer) return;

#ifdef OPENNN_HAS_CUDA
    cudaFreeHost(pointer);
#else
    free(pointer);
#endif
}

cudaEvent_t create_event(unsigned flags)
{
#ifdef OPENNN_HAS_CUDA
    cudaEvent_t event = nullptr;
    CHECK_CUDA(cudaEventCreateWithFlags(&event, flags));
    return event;
#else
    (void)flags;
    return nullptr;
#endif
}

cudaEvent_t create_event()
{
#ifdef OPENNN_HAS_CUDA
    constexpr unsigned default_flags = cudaEventDisableTiming;
#else
    constexpr unsigned default_flags = 0;
#endif
    return create_event(default_flags);
}

void destroy_event(cudaEvent_t event)
{
    if (!event) return;

#ifdef OPENNN_HAS_CUDA
    cudaEventDestroy(event);
#endif
}

void record_event(cudaEvent_t event, cudaStream_t stream)
{
#ifdef OPENNN_HAS_CUDA
    throw_if(!event, "cannot record a null CUDA event.");
    CHECK_CUDA(cudaEventRecord(event, stream));
#else
    (void)event;
    (void)stream;
#endif
}

void synchronize_event(cudaEvent_t event)
{
    if (!event) return;

#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(cudaEventSynchronize(event));
#endif
}

void stream_wait_event(cudaStream_t stream, cudaEvent_t event)
{
    if (!event) return;

#ifdef OPENNN_HAS_CUDA
    CHECK_CUDA(cudaStreamWaitEvent(stream, event, 0));
#else
    (void)stream;
#endif
}

#ifdef OPENNN_HAS_CUDA

StreamCapture::StreamCapture(cudaStream_t new_stream)
    : stream(new_stream)
{
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

GraphHandle StreamCapture::end()
{
    cudaGraph_t graph = nullptr;
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    finished = true;
    return GraphHandle(graph);
}

StreamCapture::~StreamCapture() noexcept
{
    if (finished) return;

    cudaGraph_t orphan = nullptr;
    cudaStreamEndCapture(stream, &orphan);
    if (orphan) cudaGraphDestroy(orphan);
    cudaGetLastError();
}

void instantiate_or_update(GraphExecHandle& exec, cudaGraph_t graph)
{
    if (exec)
    {
        cudaGraphExecUpdateResultInfo update_info{};
        if (cudaGraphExecUpdate(exec.get(), graph, &update_info) == cudaSuccess)
            return;

        cudaGetLastError();
        exec.reset();
    }

    cudaGraphExec_t raw = nullptr;
    CHECK_CUDA(cudaGraphInstantiate(&raw, graph, nullptr, nullptr, 0));
    exec.reset(raw);
}

void launch_graph(const GraphExecHandle& exec, cudaStream_t stream)
{
    CHECK_CUDA(cudaGraphLaunch(exec.get(), stream));
}

void destroy_graph(cudaGraph_t graph) noexcept
{
    if (graph) cudaGraphDestroy(graph);
}

void destroy_graph_exec(cudaGraphExec_t exec) noexcept
{
    if (exec) cudaGraphExecDestroy(exec);
}

#else

StreamCapture::StreamCapture(cudaStream_t) { throw_cuda_unavailable(); }
StreamCapture::~StreamCapture() noexcept {}
GraphHandle StreamCapture::end() { throw_cuda_unavailable(); }
void instantiate_or_update(GraphExecHandle&, cudaGraph_t) { throw_cuda_unavailable(); }
void launch_graph(const GraphExecHandle&, cudaStream_t) { throw_cuda_unavailable(); }
void destroy_graph(cudaGraph_t) noexcept {}
void destroy_graph_exec(cudaGraphExec_t) noexcept {}

#endif

cudaStream_t get_compute_stream()
{
    return Backend::get_compute_stream();
}

}

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void initialize_cuda_backend(cudaStream_t& compute_stream,
                                    cudaStream_t& transfer_stream,
                                    cublasHandle_t& cublas_handle,
                                    cublasLtHandle_t& cublas_lt_handle,
                                    cudnnHandle_t& cudnn_handle,
                                    cudnnOpTensorDescriptor_t& operator_sum_descriptor)
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        cerr << "OpenNN: no CUDA device available (" << cudaGetErrorString(status)
             << "); running on CPU.\n";
        return;
    }

    // Default (blocking) stream: must serialize with legacy stream 0, else recurrent/LSTM training races and diverges.
    compute_stream = device::create_stream(cudaStreamDefault);
    transfer_stream = device::create_stream(cudaStreamNonBlocking);

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, compute_stream));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor,
                                           CUDNN_OP_TENSOR_ADD,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_NOT_PROPAGATE_NAN));
}

static void destroy_cuda_backend(cudaStream_t& compute_stream,
                                 cudaStream_t& transfer_stream,
                                 cublasHandle_t& cublas_handle,
                                 cublasLtHandle_t& cublas_lt_handle,
                                 cudnnHandle_t& cudnn_handle,
                                 cudnnOpTensorDescriptor_t& operator_sum_descriptor)
{
    if (operator_sum_descriptor)
    {
        cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
        operator_sum_descriptor = nullptr;
    }

    if (cublas_lt_handle)
    {
        cublasLtDestroy(cublas_lt_handle);
        cublas_lt_handle = nullptr;
    }

    if (cublas_handle)
    {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }

    if (cudnn_handle)
    {
        cudnnDestroy(cudnn_handle);
        cudnn_handle = nullptr;
    }

    device::destroy_stream(compute_stream);
    compute_stream = nullptr;

    device::destroy_stream(transfer_stream);
    transfer_stream = nullptr;
}

#else

static void initialize_cuda_backend(cudaStream_t&,
                                    cudaStream_t&,
                                    cublasHandle_t&,
                                    cublasLtHandle_t&,
                                    cudnnHandle_t&,
                                    cudnnOpTensorDescriptor_t&)
{
}

static void destroy_cuda_backend(cudaStream_t&,
                                 cudaStream_t&,
                                 cublasHandle_t&,
                                 cublasLtHandle_t&,
                                 cudnnHandle_t&,
                                 cudnnOpTensorDescriptor_t&)
{
}

#endif

Backend::Backend()
{
    set_threads_number(0);
    initialize_cuda_backend(compute_stream,
                            transfer_stream,
                            cublas_handle,
                            cublas_lt_handle,
                            cudnn_handle,
                            operator_sum_descriptor);
}

Backend::~Backend()
{
    destroy_cuda_backend(compute_stream,
                         transfer_stream,
                         cublas_handle,
                         cublas_lt_handle,
                         cudnn_handle,
                         operator_sum_descriptor);
}

void Backend::set_threads_number(int num_threads)
{
    if (num_threads <= 0)
    {
        num_threads = thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads <= 0) num_threads = 1;
    }

    thread_pool = make_unique<ThreadPool>(num_threads);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), num_threads);

    Eigen::setNbThreads(num_threads);
    omp_set_num_threads(num_threads);
    omp_set_dynamic(1);
#if defined(_OPENMP) && _OPENMP >= 200805
    omp_set_max_active_levels(1);
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

}

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

namespace
{
    struct LtMatmulPlan
    {
        cublasLtMatmulDesc_t   matmul_descriptor = nullptr;
        cublasLtMatrixLayout_t a_matrix_layout = nullptr;
        cublasLtMatrixLayout_t b_matrix_layout = nullptr;
        cublasLtMatrixLayout_t output_matrix_layout = nullptr;
        cublasLtMatmulAlgo_t   algorithm{};
        bool                   has_algorithm = false;
        size_t                 workspace_bytes = 0;

        LtMatmulPlan() = default;
        LtMatmulPlan(const LtMatmulPlan&) = delete;
        LtMatmulPlan& operator=(const LtMatmulPlan&) = delete;
        LtMatmulPlan(LtMatmulPlan&& other) noexcept { swap_with(other); }
        LtMatmulPlan& operator=(LtMatmulPlan&& other) noexcept { swap_with(other); return *this; }

        void swap_with(LtMatmulPlan& other) noexcept
        {
            swap(matmul_descriptor, other.matmul_descriptor);
            swap(a_matrix_layout, other.a_matrix_layout);
            swap(b_matrix_layout, other.b_matrix_layout);
            swap(output_matrix_layout, other.output_matrix_layout);
            swap(algorithm, other.algorithm);
            swap(has_algorithm, other.has_algorithm);
            swap(workspace_bytes, other.workspace_bytes);
        }

        ~LtMatmulPlan()
        {
            cublasLtMatrixLayoutDestroy(output_matrix_layout);
            cublasLtMatrixLayoutDestroy(b_matrix_layout);
            cublasLtMatrixLayoutDestroy(a_matrix_layout);
            cublasLtMatmulDescDestroy(matmul_descriptor);
        }
    };

    struct LtMatmulPlanKey
    {
        int m;
        int n;
        int k;
        int transA;
        int transB;
        int epilogue;   // cublasLtEpilogue_t cast to int (e.g. BIAS, RELU_BIAS, BGRADA)
        int io_dtype;   // cudaDataType_t for A and B (inputs)
        int out_dtype;  // cudaDataType_t for C and D (outputs)

        bool operator==(const LtMatmulPlanKey&) const noexcept = default;
    };

    struct LtMatmulPlanKeyHash
    {
        size_t operator()(const LtMatmulPlanKey& key) const noexcept
        {
            return hash_combine(key.m, key.n, key.k,
                                key.transA, key.transB, key.epilogue,
                                key.io_dtype, key.out_dtype);
        }
    };

    struct CudaGemmThreadState
    {
        Buffer workspace{Device::CUDA};

        Buffer bf16_input{Device::CUDA};
        Buffer bf16_gradient{Device::CUDA};
        Buffer bf16_to_fp32{Device::CUDA};

        unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_gemm_plans;
    };

    CudaGemmThreadState& thread_state()
    {
        thread_local CudaGemmThreadState state;
        return state;
    }

    constexpr size_t cublas_lt_workspace_search_bytes = 32ull * 1024 * 1024;

    cublasComputeType_t gemm_compute_type(cudaDataType_t a_type, cudaDataType_t b_type = CUDA_R_32F)
    {
        return (a_type == CUDA_R_16BF || b_type == CUDA_R_16BF)
            ? CUBLAS_COMPUTE_32F
            : CUBLAS_COMPUTE_DTYPE;
    }

    struct LtMatmulPreferenceGuard
    {
        cublasLtMatmulPreference_t pref = nullptr;
        LtMatmulPreferenceGuard() { CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref)); }
        ~LtMatmulPreferenceGuard() { cublasLtMatmulPreferenceDestroy(pref); }
    };

    bool workspace_growth_forbidden() noexcept
    {
        return device::cuda_allocation_growth_forbidden()
            || env_flag_enabled("OPENNN_CUDA_NO_SCRATCH_GROWTH");
    }

    template <typename T>
    T* ensure_workspace(Buffer& workspace_buffer, Index n)
    {
        if (n * Index(sizeof(T)) > workspace_buffer.bytes && workspace_buffer.data)
        {
            throw_if(workspace_growth_forbidden(),
                     "ensure_workspace: workspace allocation growth is forbidden "
                     "(warmup incomplete before CUDA graph capture).");
            device::synchronize(Backend::get_compute_stream());
        }
        
        return workspace_buffer.ensure<T>(n);
    }

    void* ensure_cublas_lt_workspace(size_t min_bytes)
    {
        return ensure_workspace<uint8_t>(thread_state().workspace, Index(min_bytes));
    }

    bfloat16* ensure_bf16_input_workspace(Index n)
    {
        return ensure_workspace<bfloat16>(thread_state().bf16_input, n);
    }
}

bfloat16* ensure_bf16_gradient_workspace(Index n)
{
    return ensure_workspace<bfloat16>(thread_state().bf16_gradient, n);
}

float* ensure_bf16_to_fp32_workspace(Index n)
{
    return ensure_workspace<float>(thread_state().bf16_to_fp32, n);
}

void* ensure_cudnn_conv_workspace(size_t min_bytes)
{
    return ensure_workspace<uint8_t>(thread_state().workspace, Index(min_bytes));
}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type)
{
    if (input.type == target_type) return input.data;

    if (input.type == Type::FP32 && target_type == Type::BF16)
    {
        bfloat16* dst = ensure_bf16_input_workspace(input.size());
        cast_fp32_to_bf16_cuda(input.size(), input.as<float>(), dst);
        return dst;
    }

    if (input.type == Type::BF16 && target_type == Type::FP32)
    {
        float* dst = ensure_bf16_to_fp32_workspace(input.size());
        cast_bf16_to_fp32_cuda(input.size(), input.as<bfloat16>(), dst);
        return dst;
    }

    throw runtime_error("data_for_gemm_dtype: unsupported type pair");
}

namespace
{
const LtMatmulPlan& get_lt_gemm_plan(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    cudaDataType_t io_dtype,
    cudaDataType_t out_dtype)
{
    const LtMatmulPlanKey key{m, n, k,
                              int(transA), int(transB), int(epilogue),
                              int(io_dtype), int(out_dtype)};
    auto& plans = thread_state().lt_gemm_plans;
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;

    throw_if(workspace_growth_forbidden(),
             "get_lt_gemm_plan: new GEMM plan requested while workspace growth is forbidden "
             "(unseen shape during CUDA graph capture; warmup incomplete).");

    LtMatmulPlan plan;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.matmul_descriptor, gemm_compute_type(io_dtype), CUDA_R_32F));

    auto set_desc = [&](cublasLtMatmulDescAttributes_t attr, const auto& value)
    {
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor, attr, &value, sizeof(value)));
    };

    set_desc(CUBLASLT_MATMUL_DESC_TRANSA,   transA);
    set_desc(CUBLASLT_MATMUL_DESC_TRANSB,   transB);
    set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    set_desc(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_dtype);

    const int a_rows = (transA == CUBLAS_OP_N) ? m : k;
    const int a_cols = (transA == CUBLAS_OP_N) ? k : m;
    const int b_rows = (transB == CUBLAS_OP_N) ? k : n;
    const int b_cols = (transB == CUBLAS_OP_N) ? n : k;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_matrix_layout,  io_dtype,  a_rows, a_cols, a_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_matrix_layout,  io_dtype,  b_rows, b_cols, b_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.output_matrix_layout, out_dtype, m, n, m));

    LtMatmulPreferenceGuard pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref.pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublas_lt_workspace_search_bytes, sizeof(cublas_lt_workspace_search_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned_results = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(Backend::get_cublas_lt_handle(),
                                                plan.matmul_descriptor,
                                                plan.a_matrix_layout,
                                                plan.b_matrix_layout,
                                                plan.output_matrix_layout,
                                                plan.output_matrix_layout,
                                                pref.pref, 1, &heuristic, &returned_results));

    if (returned_results > 0)
    {
        plan.algorithm = heuristic.algo;
        plan.has_algorithm = true;
        plan.workspace_bytes = heuristic.workspaceSize;

        // Grow the global workspace to fit this plan's chosen algorithm.
        ensure_cublas_lt_workspace(plan.workspace_bytes);
    }

    return plans.emplace(key, move(plan)).first->second;
}
}

void run_lt_matmul_cached(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    const void* a_data, const void* b_data, void* c_data,
    const void* bias_pointer,
    cudaDataType_t io_dtype,
    cudaDataType_t out_dtype)
{
    const LtMatmulPlan& plan = get_lt_gemm_plan(m, n, k, transA, transB, epilogue, io_dtype, out_dtype);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.matmul_descriptor,
                                &one,
                                a_data, plan.a_matrix_layout,
                                b_data, plan.b_matrix_layout,
                                &zero,
                                c_data, plan.output_matrix_layout,
                                c_data, plan.output_matrix_layout,
                                plan.has_algorithm ? &plan.algorithm : nullptr,
                                ensure_cublas_lt_workspace(plan.workspace_bytes), plan.workspace_bytes,
                                Backend::get_compute_stream()));
}

void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
               int m, int n, int k,
               const void* A, cudaDataType_t Atype, int lda,
               const void* B, cudaDataType_t Btype, int ldb,
               void* C, cudaDataType_t Ctype, int ldc,
               float alpha, float beta)
{
    const cublasComputeType_t compute = gemm_compute_type(Atype, Btype);
    CHECK_CUBLAS(cublasGemmEx(Backend::get_cublas_handle(),
                              transa, transb,
                              m, n, k,
                              &alpha,
                              A, Atype, lda,
                              B, Btype, ldb,
                              &beta,
                              C, Ctype, ldc,
                              compute,
                              CUBLAS_GEMM_DEFAULT));
}

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void* A, cudaDataType_t Atype, int lda, long long stride_a,
                               const void* B, cudaDataType_t Btype, int ldb, long long stride_b,
                               void* C, cudaDataType_t Ctype, int ldc, long long stride_c,
                               int batch_count,
                               float alpha, float beta)
{
    const cublasComputeType_t compute = gemm_compute_type(Atype, Btype);
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(Backend::get_cublas_handle(),
                                            transa, transb,
                                            m, n, k,
                                            &alpha,
                                            A, Atype, lda, stride_a,
                                            B, Btype, ldb, stride_b,
                                            &beta,
                                            C, Ctype, ldc, stride_c,
                                            batch_count,
                                            compute,
                                            CUBLAS_GEMM_DEFAULT));
}

}

#else

namespace opennn
{

bfloat16* ensure_bf16_gradient_workspace(Index)
{
    throw runtime_error("ensure_bf16_gradient_workspace requires CUDA support.");
}

float* ensure_bf16_to_fp32_workspace(Index)
{
    throw runtime_error("ensure_bf16_to_fp32_workspace requires CUDA support.");
}

void* ensure_cudnn_conv_workspace(size_t)
{
    throw runtime_error("ensure_cudnn_conv_workspace requires CUDA support.");
}

const void* data_for_gemm_dtype(const TensorView&, Type)
{
    throw runtime_error("data_for_gemm_dtype requires CUDA support.");
}

void run_lt_matmul_cached(int, int, int,
                          cublasOperation_t,
                          cublasOperation_t,
                          cublasLtEpilogue_t,
                          const void*, const void*, void*,
                          const void*,
                          cudaDataType_t,
                          cudaDataType_t)
{
    throw runtime_error("run_lt_matmul_cached requires CUDA support.");
}

void gemm_cuda(cublasOperation_t, cublasOperation_t,
               int, int, int,
               const void*, cudaDataType_t, int,
               const void*, cudaDataType_t, int,
               void*, cudaDataType_t, int,
               float, float)
{
    throw runtime_error("gemm_cuda requires CUDA support.");
}

void gemm_strided_batched_cuda(cublasOperation_t, cublasOperation_t,
                               int, int, int,
                               const void*, cudaDataType_t, int, long long,
                               const void*, cudaDataType_t, int, long long,
                               void*, cudaDataType_t, int, long long,
                               int,
                               float, float)
{
    throw runtime_error("gemm_strided_batched_cuda requires CUDA support.");
}

}

#endif // OPENNN_HAS_CUDA

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
