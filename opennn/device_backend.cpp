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
#include "memory_debug.h"

#include <atomic>
#include <cstdlib>

namespace opennn::device
{

namespace
{

atomic_bool cuda_allocation_growth_forbidden_runtime{false};

// Conv workspace cap state. mode: 0 = off (uncapped/autotune), >0 = explicit
// bytes, <0 = AUTO (use the per-network auto value). auto value = largest single
// -layer activation, set by ForwardPropagation, clamped to conv_workspace_auto_ceiling.
//
// Defaults: AUTO cap + autotune off. Uncapped autotune lets the cuDNN-frontend
// consider (and, on an allocation failure it silently swallows, select) plans whose
// workspace scales with batch size into GBs — which OOMs on large batches for no
// speed gain (autotune probing is a net slowdown on small/medium convs). A bounded
// AUTO cap forces a small-workspace plan: same result, less memory, faster here.
constexpr int64_t conv_workspace_auto_ceiling = int64_t(256) * 1024 * 1024;  // 256 MiB
atomic<int64_t> conv_workspace_cap_mode{-1};                          // AUTO
atomic<int64_t> conv_workspace_auto_bytes{conv_workspace_auto_ceiling};
atomic_bool conv_autotune_enabled_flag{false};

void throw_if_auto(Device device_type)
{
    throw_if(device_type == Device::Auto,
             "device backend expects a resolved device.");
}


#ifndef OPENNN_HAS_CUDA
[[noreturn]] void throw_cuda_unavailable()
{
    throw runtime_error("CUDA support is not compiled in.");
}
#endif

void* allocate_cuda(Index byte_count)
{
#ifdef OPENNN_HAS_CUDA
    throw_if(cuda_allocation_growth_forbidden(),
             format("CUDA alloc of {} bytes forbidden (warmup incomplete).", byte_count));
    void* device_pointer = nullptr;
    CHECK_CUDA(cudaMalloc(&device_pointer, static_cast<size_t>(byte_count)));
    return device_pointer;
#else
    (void)byte_count;
    throw_cuda_unavailable();
#endif
}

}


bool has_cuda_device() noexcept
{
#ifdef OPENNN_HAS_CUDA
    // The visible device count cannot change within a process, so probe once.
    static const bool available = []() noexcept
    {
        int count = 0;
        const cudaError_t error = cudaGetDeviceCount(&count);
        if (error != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }

        return count > 0;
    }();

    return available;
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

std::string gpu_info_string() noexcept
{
#ifdef OPENNN_HAS_CUDA
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "GPU info unavailable";
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    int ver = 0;
    cudaRuntimeGetVersion(&ver);
    return std::format("{:<32s}  {:.0f} MB total / {:.0f} MB free  CC {}.{}  CUDA {:d}.{:d}",
                       p.name,
                       total_b / 1048576.0,
                       free_b  / 1048576.0,
                       p.major, p.minor,
                       ver / 1000, (ver % 1000) / 10);
#else
    return "CPU only";
#endif
}

bool cuda_allocation_growth_forbidden() noexcept
{
    return cuda_allocation_growth_forbidden_runtime.load(memory_order_relaxed);
}

void set_cuda_allocation_growth_forbidden(bool forbidden) noexcept
{
    cuda_allocation_growth_forbidden_runtime.store(forbidden, memory_order_relaxed);
}

int64_t conv_workspace_limit_bytes() noexcept
{
    const int64_t mode = conv_workspace_cap_mode.load(memory_order_relaxed);
    if (mode == 0) return 0;                                       // off (autotune)
    if (mode > 0)  return mode;                                    // explicit bytes
    return conv_workspace_auto_bytes.load(memory_order_relaxed);   // AUTO
}

void set_conv_workspace_cap(int64_t mode) noexcept
{
    conv_workspace_cap_mode.store(mode, memory_order_relaxed);
}

void set_conv_workspace_auto_limit_bytes(int64_t bytes) noexcept
{
    if (bytes > 0)
        conv_workspace_auto_bytes.store(bytes < conv_workspace_auto_ceiling ? bytes : conv_workspace_auto_ceiling,
                                        memory_order_relaxed);
}

bool conv_autotune_enabled() noexcept
{
    return conv_autotune_enabled_flag.load(memory_order_relaxed);
}

void set_conv_autotune(bool enabled) noexcept
{
    conv_autotune_enabled_flag.store(enabled, memory_order_relaxed);
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
        return allocate_cuda(byte_count);

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
    cudaMemcpyKind cuda_kind = cudaMemcpyHostToHost;
    switch (kind)
    {
        case CopyKind::HostToHost:     cuda_kind = cudaMemcpyHostToHost;     break;
        case CopyKind::HostToDevice:   cuda_kind = cudaMemcpyHostToDevice;   break;
        case CopyKind::DeviceToHost:   cuda_kind = cudaMemcpyDeviceToHost;   break;
        case CopyKind::DeviceToDevice: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default: throw runtime_error("Invalid device copy kind.");
    }

    if (stream)
        CHECK_CUDA(cudaMemcpyAsync(destination, source,
                                   static_cast<size_t>(byte_count),
                                   cuda_kind, stream));
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
    throw_if_auto(source_device);
    throw_if_auto(target_device);

    CopyKind kind = CopyKind::HostToHost;
    if (source_device == Device::CUDA && target_device == Device::CUDA) kind = CopyKind::DeviceToDevice;
    else if (source_device == Device::CUDA)                             kind = CopyKind::DeviceToHost;
    else if (target_device == Device::CUDA)                             kind = CopyKind::HostToDevice;

    copy_async(destination, source, byte_count, kind, stream);
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

void reset_last_error() noexcept
{
#ifdef OPENNN_HAS_CUDA
    cudaGetLastError();
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
    return create_event(cudaEventDisableTiming);
#else
    return nullptr;
#endif
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

Backend::Backend()
{
    // OPENNN_THREADS caps the host thread pool (Eigen + OpenMP GEMM paths);
    // unset or <= 0 keeps the hardware concurrency default. Benchmarks use it
    // to run every engine at the same thread count.
    const char* threads_env = std::getenv("OPENNN_THREADS");
    set_threads_number(threads_env ? std::atoi(threads_env) : 0);

#ifdef OPENNN_HAS_CUDA
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

    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    // The legacy cuBLAS handle and cuDNN are NOT created here -- see
    // Backend::cublas() and Backend::cudnn().
#endif
}

// Deferred from the constructor (see get_cublas_handle() in the header):
// creating the legacy handle reserves a device workspace that the
// cuBLASLt-plus-custom-kernels inference path never uses.
cublasHandle_t Backend::cublas()
{
#ifdef OPENNN_HAS_CUDA
    call_once(cublas_init_once, [this]
    {
        if (!compute_stream) return;   // no CUDA device: stay null, as before

        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    });
#endif
    return cublas_handle;
}

// Deferred from the constructor (see get_cudnn_handle() in the header): a
// cuDNN handle costs fixed VRAM whether or not any cuDNN op ever runs.
cudnnHandle_t Backend::cudnn()
{
#ifdef OPENNN_HAS_CUDA
    call_once(cudnn_init_once, [this]
    {
        if (!compute_stream) return;   // no CUDA device: stay null, as before

        CHECK_CUDNN(cudnnCreate(&cudnn_handle));
        CHECK_CUDNN(cudnnSetStream(cudnn_handle, compute_stream));

        CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
        CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor,
                                               CUDNN_OP_TENSOR_ADD,
                                               CUDNN_DATA_FLOAT,
                                               CUDNN_NOT_PROPAGATE_NAN));
    });
#endif
    return cudnn_handle;
}

Backend::~Backend()
{
#ifdef OPENNN_HAS_CUDA
    if (operator_sum_descriptor) { cudnnDestroyOpTensorDescriptor(operator_sum_descriptor); operator_sum_descriptor = nullptr; }
    if (cublas_lt_handle)        { cublasLtDestroy(cublas_lt_handle);                       cublas_lt_handle = nullptr; }
    if (cublas_handle)           { cublasDestroy(cublas_handle);                             cublas_handle = nullptr; }
    if (cudnn_handle)            { cudnnDestroy(cudnn_handle);                               cudnn_handle = nullptr; }
    device::destroy_stream(compute_stream);  compute_stream = nullptr;
    device::destroy_stream(transfer_stream); transfer_stream = nullptr;
#endif
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

    struct CudaMatmulThreadState
    {
        Buffer workspace{Device::CUDA};

        Buffer bf16_input{Device::CUDA};
        Buffer bf16_gradient{Device::CUDA};
        Buffer bf16_to_fp32{Device::CUDA};

        unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_matmul_plans;
    };

    CudaMatmulThreadState& thread_state()
    {
        thread_local CudaMatmulThreadState state;
        return state;
    }

    constexpr size_t cublas_lt_workspace_search_bytes = 32ull * 1024 * 1024;

    cublasComputeType_t matmul_compute_type(cudaDataType_t a_type, cudaDataType_t b_type = CUDA_R_32F)
    {
        if (a_type == CUDA_R_16BF || b_type == CUDA_R_16BF)
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        return CUBLAS_COMPUTE_DTYPE;
    }

    template <typename T>
    T* ensure_workspace(Buffer& workspace_buffer, Index n)
    {
        if (n * Index(sizeof(T)) > workspace_buffer.bytes && workspace_buffer.data)
        {
            throw_if(device::cuda_allocation_growth_forbidden(),
                     "workspace growth forbidden (warmup incomplete).");
            device::synchronize(Backend::get_compute_stream());
        }
        
        return workspace_buffer.ensure<T>(n);
    }

    // cublasLt and the cuDNN-frontend graphs all draw scratch from this one
    // buffer (ops run serially on the compute stream, so the live peak is the
    // max single workspace, not the sum). Record each growth delta so
    // memory_debug telescopes to the final buffer size, the achieved floor.
    void* ensure_shared_scratch(size_t min_bytes)
    {
        Buffer& buffer = thread_state().workspace;
        const Index before = buffer.bytes;
        void* pointer = ensure_workspace<uint8_t>(buffer, Index(min_bytes));
        if (buffer.bytes > before)
            memory_debug::record("workspace.cudnn_frontend", "shared_scratch",
                                 buffer.bytes - before, "high_water");
        return pointer;
    }

    bfloat16* ensure_bf16_input_workspace(Index n)
    {
        return ensure_workspace<bfloat16>(thread_state().bf16_input, n);
    }

    LtMatmulPlan& get_lt_matmul_plan(
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
        auto& plans = thread_state().lt_matmul_plans;
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;

        throw_if(device::cuda_allocation_growth_forbidden(), "matmul plan forbidden (warmup incomplete).");

        LtMatmulPlan plan;

        CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.matmul_descriptor, matmul_compute_type(io_dtype), CUDA_R_32F));

        auto set_desc = [&](cublasLtMatmulDescAttributes_t attr, const auto& value)
        {
            CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor, attr, &value, sizeof(value)));
        };

        set_desc(CUBLASLT_MATMUL_DESC_TRANSA,   transA);
        set_desc(CUBLASLT_MATMUL_DESC_TRANSB,   transB);
        set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
        set_desc(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_dtype);

        if (epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS)
        {
            const int64_t aux_ld = m;
            set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, aux_ld);
        }

        const int a_rows = (transA == CUBLAS_OP_N) ? m : k;
        const int a_cols = (transA == CUBLAS_OP_N) ? k : m;
        const int b_rows = (transB == CUBLAS_OP_N) ? k : n;
        const int b_cols = (transB == CUBLAS_OP_N) ? n : k;

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_matrix_layout,  io_dtype,  a_rows, a_cols, a_rows));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_matrix_layout,  io_dtype,  b_rows, b_cols, b_rows));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.output_matrix_layout, out_dtype, m, n, m));

        cublasLtMatmulPreference_t pref = nullptr;
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &cublas_lt_workspace_search_bytes, sizeof(cublas_lt_workspace_search_bytes)));

        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned_results = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(Backend::get_cublas_lt_handle(),
                                                    plan.matmul_descriptor,
                                                    plan.a_matrix_layout,
                                                    plan.b_matrix_layout,
                                                    plan.output_matrix_layout,
                                                    plan.output_matrix_layout,
                                                    pref, 1,
                                                    &heuristic, &returned_results));
        cublasLtMatmulPreferenceDestroy(pref);

        if (returned_results > 0)
        {
            plan.algorithm = heuristic.algo;
            plan.has_algorithm = true;
            plan.workspace_bytes = heuristic.workspaceSize;
            ensure_shared_scratch(plan.workspace_bytes);
        }

        return plans.emplace(key, move(plan)).first->second;
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
    return ensure_shared_scratch(min_bytes);
}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type)
{
    if (input.type == target_type) return input.data;

    if (input.is_fp32() && target_type == Type::BF16)
    {
        bfloat16* dst = ensure_bf16_input_workspace(input.size());
        cast_fp32_to_bf16(input.size(), input.as<float>(), dst);
        return dst;
    }

    if (input.is_bf16() && target_type == Type::FP32)
    {
        float* dst = ensure_bf16_to_fp32_workspace(input.size());
        cast_bf16_to_fp32(input.size(), input.as<bfloat16>(), dst);
        return dst;
    }

    throw runtime_error("data_for_gemm_dtype: unsupported type pair");
}

// Cast an fp32 bias to bf16 for a fused bf16 cuBLASLt BIAS epilogue (which
// rejects an fp32 bias). Uses the gradient workspace, which is unused during
// forward propagation, so it does not clobber the input cast.
const void* bias_for_gemm_bf16(const TensorView& bias)
{
    bfloat16* dst = ensure_bf16_gradient_workspace(bias.size());
    cast_fp32_to_bf16(bias.size(), bias.as<float>(), dst);
    return dst;
}

void run_lt_matmul_cached(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    const void* a_data, const void* b_data, void* c_data,
    const void* bias_pointer,
    cudaDataType_t io_dtype,
    cudaDataType_t out_dtype,
    const void* aux_pointer)
{
    LtMatmulPlan& plan = get_lt_matmul_plan(m, n, k, transA, transB, epilogue, io_dtype, out_dtype);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));

    if (aux_pointer)
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_pointer, sizeof(aux_pointer)));

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.matmul_descriptor,
                                &one,
                                a_data, plan.a_matrix_layout,
                                b_data, plan.b_matrix_layout,
                                &zero,
                                c_data, plan.output_matrix_layout,
                                c_data, plan.output_matrix_layout,
                                plan.has_algorithm ? &plan.algorithm : nullptr,
                                ensure_shared_scratch(plan.workspace_bytes), plan.workspace_bytes,
                                Backend::get_compute_stream()));
}

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void* A, cudaDataType_t Atype, int lda, long long stride_a,
                               const void* B, cudaDataType_t Btype, int ldb, long long stride_b,
                               void* C, cudaDataType_t Ctype, int ldc, long long stride_c,
                               int batch_count,
                               float alpha, float beta)
{
    const cublasComputeType_t compute = matmul_compute_type(Atype, Btype);
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

const void* bias_for_gemm_bf16(const TensorView&)
{
    throw runtime_error("bias_for_gemm_bf16 requires CUDA support.");
}

void run_lt_matmul_cached(int, int, int,
                          cublasOperation_t,
                          cublasOperation_t,
                          cublasLtEpilogue_t,
                          const void*, const void*, void*,
                          const void*,
                          cudaDataType_t,
                          cudaDataType_t,
                          const void*)
{
    throw runtime_error("run_lt_matmul_cached requires CUDA support.");
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
