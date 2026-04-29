//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "configuration.h"   // Buffer references DeviceType; existing callers expect Configuration available transitively.

namespace opennn
{

static constexpr Index ALIGN_BYTES = EIGEN_MAX_ALIGN_BYTES;
static constexpr Index ALIGN_ELEMENTS = ALIGN_BYTES / sizeof(type);

inline int to_int(Index value) { return static_cast<int>(value); }
inline type to_type(Index value) { return static_cast<type>(value); }

inline Index align_up(Index n, Index alignment)
{
    return n == 0 ? 0 : (n + alignment - 1) & ~(alignment - 1);
}

inline Index get_aligned_size(Index size)     { return align_up(size,    ALIGN_ELEMENTS); }
inline Index get_aligned_bytes(Index n_bytes) { return align_up(n_bytes, ALIGN_BYTES); }

template<typename Container>
inline Index ssize(const Container& c) noexcept
{
    return static_cast<Index>(c.size());
}

inline bool is_aligned(const void* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % ALIGN_BYTES == 0;
}

// EnumMap<Enum> moved to enum_map.h.

constexpr cudaDataType_t      CUDA_REDUCTION_DTYPE   = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE_DTYPE   = CUBLAS_COMPUTE_32F_FAST_TF32;

// Activation dtype used to be a compile-time constant gated by
// `OPENNN_USE_BF16_ACTIVATIONS`. It is now a runtime field on every Layer
// (`activation_dtype` / `cuda_activation_dtype`), populated by
// NeuralNetwork::compile() from Configuration::resolve(). Anything outside a
// network (TensorView default ctor, generic GEMM helpers, etc.) defaults to
// FP32 — call sites that need BF16 must pass the dtype explicitly.

inline Index dtype_bytes(cudnnDataType_t t)
{
    switch (t) {
        case CUDNN_DATA_FLOAT:    return 4;
        case CUDNN_DATA_INT32:    return 4;
        case CUDNN_DATA_BFLOAT16: return 2;
        case CUDNN_DATA_HALF:     return 2;
        case CUDNN_DATA_INT8:     return 1;
        default:                  return 4;
    }
}

inline cudaDataType_t cudnn_to_cuda_dtype(cudnnDataType_t t)
{
    switch (t) {
        case CUDNN_DATA_FLOAT:    return CUDA_R_32F;
        case CUDNN_DATA_INT32:    return CUDA_R_32I;
        case CUDNN_DATA_BFLOAT16: return CUDA_R_16BF;
        case CUDNN_DATA_HALF:     return CUDA_R_16F;
        case CUDNN_DATA_INT8:     return CUDA_R_8I;
        default:                  return CUDA_R_32F;
    }
}

// ThreadSafeQueue<T> moved to thread_safe_queue.h.

struct Shape
{
    static constexpr size_t MaxRank = 4;

    Index  dims[MaxRank] = {0};
    size_t rank          = 0;

    Shape() noexcept = default;

    Shape(size_t n, Index value) : rank(min(n, MaxRank))
    { std::fill_n(dims, rank, value); }

    Shape(initializer_list<Index> list) : rank(min(list.size(), MaxRank))
    { std::copy_n(list.begin(), rank, dims); }

    const Index* begin() const noexcept { return dims; }
    const Index* end()   const noexcept { return dims + rank; }
    const Index& operator[](size_t i) const noexcept { return dims[i]; }
    Index&       operator[](size_t i)       noexcept { return dims[i]; }

    Index& back()             { if (rank == 0) throw runtime_error("Shape::back() on empty"); return dims[rank - 1]; }
    const Index& back() const { if (rank == 0) throw runtime_error("Shape::back() on empty"); return dims[rank - 1]; }

    bool empty() const noexcept { return rank == 0; }

    Index size() const noexcept
    {
        return rank == 0 ? 0 : std::accumulate(begin(), end(), Index(1), std::multiplies<>{});
    }

    void clear() noexcept { rank = 0; }
    void push_back(Index v) noexcept { if (rank < MaxRank) dims[rank++] = v; }

    friend ostream& operator<<(ostream& os, const Shape& s)
    {
        os << "[";
        for (size_t i = 0; i < s.rank; ++i) os << (i ? ", " : " ") << s.dims[i];
        os << " ]";
        return os;
    }

    bool operator==(const Shape& other) const noexcept
    {
        return rank == other.rank && std::equal(begin(), end(), other.begin());
    }

    bool operator!=(const Shape& other) const noexcept { return !(*this == other); }

    Shape& append(const Shape& other)
    {
        const size_t n = min(other.rank, MaxRank - rank);
        std::copy_n(other.dims, n, dims + rank);
        rank += n;
        return *this;
    }
};

// DeviceType / TrainingPrecision / InferencePrecision live in configuration.h
// (which is included from this header for Buffer's DeviceType reference).

struct Buffer
{
    void* data = nullptr;
    Index bytes = 0;
    DeviceType device_type = DeviceType::CPU;

    template<typename T> T*       as()       { return static_cast<T*>(data); }
    template<typename T> const T* as() const { return static_cast<const T*>(data); }

    Index size()  const { return bytes / Index(sizeof(type)); }   // legacy: assumes T = type
    bool  empty() const { return bytes == 0; }

    void resize_bytes(Index n_bytes, DeviceType t)
    {
        if(n_bytes == bytes && device_type == t) return;
        free_buffer();
        device_type = t;
        if(n_bytes > 0) data = alloc(t, n_bytes);
        bytes = n_bytes;
    }

    void setZero()
    {
        if(!data) return;
#ifdef OPENNN_WITH_CUDA
        if(device_type == DeviceType::CUDA) CHECK_CUDA(cudaMemset(data, 0, bytes));
        else
#endif
            std::memset(data, 0, static_cast<size_t>(bytes));
    }

#ifdef OPENNN_WITH_CUDA
    // Migrate the buffer to the target side: alloc on target, copy, free source.
    void migrate_to(DeviceType target)
    {
        if(device_type == target || !data) return;
        void* fresh = alloc(target, bytes);
        const cudaMemcpyKind kind = (target == DeviceType::CUDA)
            ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        CHECK_CUDA(cudaMemcpy(fresh, data, bytes, kind));
        dealloc(device_type, data, bytes);
        data = fresh;
        device_type = target;
    }
#endif

    explicit Buffer(DeviceType t = DeviceType::CPU) noexcept : device_type(t) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& o) noexcept : Buffer() { swap(o); }
    Buffer& operator=(Buffer&& o) noexcept { swap(o); return *this; }

    ~Buffer() { free_buffer(); }

    void swap(Buffer& o) noexcept
    {
        std::swap(data, o.data);
        std::swap(bytes, o.bytes);
        std::swap(device_type, o.device_type);
    }

private:
    static void* alloc(DeviceType t, Index n)
    {
#ifdef OPENNN_WITH_CUDA
        if(t == DeviceType::CUDA) { void* p = nullptr; CHECK_CUDA(cudaMalloc(&p, n)); return p; }
#endif
        return Eigen::aligned_allocator<uint8_t>{}.allocate(static_cast<size_t>(n));
    }

    static void dealloc(DeviceType t, void* p, Index n)
    {
#ifdef OPENNN_WITH_CUDA
        if(t == DeviceType::CUDA) { cudaFree(p); return; }
#endif
        Eigen::aligned_allocator<uint8_t>{}.deallocate(static_cast<uint8_t*>(p), static_cast<size_t>(n));
    }

    void free_buffer()
    {
        if(data) dealloc(device_type, data, bytes);
        data = nullptr;
        bytes = 0;
    }
};

struct TensorView
{
    void* data = nullptr;

    Shape shape;

    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;

    TensorView(void* new_data = nullptr, const Shape& new_shape = {},
               cudnnDataType_t new_dtype = CUDNN_DATA_FLOAT) noexcept
        : data(new_data), shape(new_shape), dtype(new_dtype) {}

    Index get_rank() const noexcept { return shape.rank; }

    Index size() const noexcept { return shape.size(); }

    Index byte_size() const noexcept { return size() * dtype_bytes(dtype); }

    bool empty() const noexcept { return shape.empty(); }

    // Typed reinterpretation of `data`. We deliberately do NOT assert that the
    // requested type matches `dtype`: the codebase has several call sites that
    // pass raw byte pointers to APIs (cuBLASLt, cuDNN) which interpret the
    // bytes via plan/descriptor metadata, so a `bias.as<float>()` on a BF16
    // bias is legal — cuBLASLt reads the underlying bytes as BF16 because the
    // plan was built with bias_dtype = BF16. Inlining + a single primary
    // template avoids the "specialization after instantiation" ordering trap
    // with the inline as_matrix/as_vector helpers below.
    template<typename T>
    T* as() const noexcept
    {
        assert(data);
        return reinterpret_cast<T*>(data);
    }

    // Float-typed view of `data`. Used at CUDA dispatch sites where kernels expect
    // raw float* regardless of the project's `type` alias (e.g. descriptive stats,
    // scaler tables, masks). Mirrors as<T>() but skips the dtype check.
    float* as_float() const noexcept
    {
        return reinterpret_cast<float*>(data);
    }

    cudaDataType_t cuda_dtype() const noexcept { return cudnn_to_cuda_dtype(dtype); }

    // FP16 is intentionally absent: no kernel in this project is instantiated
    // for __half. Adding it here would generate unresolved symbols for every
    // dispatch site (scale, unscale, bounding, layernorm, pooling, ...).
    template<typename F>
    void dispatch(F&& fn) const
    {
        if (dtype == CUDNN_DATA_BFLOAT16) fn(__nv_bfloat16{});
        else                              fn(float{});
    }

    TensorView reshape(const Shape& new_shape) const
    { return TensorView(data, new_shape, dtype); }

    MatrixMap as_matrix() const
    {
        assert(shape.rank >= 2);
        return MatrixMap(as<float>(), shape[0], shape.size() / shape[0]);
    }

    MatrixMap as_matrix(Index batch_index) const
    {
        assert(shape.rank >= 2);
        const Index rows = shape[shape.rank - 2];
        const Index cols = shape[shape.rank - 1];
        return MatrixMap(as<float>() + batch_index * rows * cols, rows, cols);
    }

    MatrixMap as_flat_matrix() const
    {
        assert(shape.rank >= 1);
        const Index cols = shape[shape.rank - 1];
        return MatrixMap(as<float>(), shape.size() / cols, cols);
    }

    MatrixMap as_flat_matrix(Index batch_index) const
    {
        assert(shape.rank >= 2);
        const Index cols = shape[shape.rank - 1];
        const Index rows = shape.size() / (shape[0] * cols);
        return MatrixMap(as<float>() + batch_index * rows * cols, rows, cols);
    }

    VectorMap as_vector() const
    {
        return VectorMap(as<float>(), shape.size());
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor() const
    {
        assert(shape.rank == Rank);
        Eigen::array<Index, Rank> dims;
        std::copy_n(shape.dims, Rank, dims.begin());
        return TensorMapR<Rank>(as<float>(), dims);
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor(Index batch_index) const
    {
        assert(shape.rank == Rank + 1);
        Eigen::array<Index, Rank> dims;
        for(int i = 0; i < Rank; ++i) dims[i] = shape[i + 1];
        const Index slice_size = shape.size() / shape[0];
        return TensorMapR<Rank>(as<float>() + batch_index * slice_size, dims);
    }

    void fill(float value);

#ifdef OPENNN_WITH_CUDA

    mutable shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    cudnnTensorDescriptor_t get_descriptor() const
    {
        if (!descriptor_handle && !shape.empty())
            set_descriptor(shape);
        return descriptor_handle.get();
    }

private:
    void set_descriptor(const Shape& s) const
    {
        // NHWC layout: N first, then H, W, C trailing. For rank < 4 the
        // missing leading dims default to 1.
        int n = 1, c = 1, h = 1, w = 1;
        const size_t r = s.rank;
        if (r >= 1) c = static_cast<int>(s[r - 1]);
        if (r >= 2) n = static_cast<int>(s[0]);
        if (r >= 3) w = static_cast<int>(s[r - 2]);
        if (r >= 4) h = static_cast<int>(s[r - 3]);

        if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
            return;

        if (!descriptor_handle)
        {
            cudnnTensorDescriptor_t raw_desc;
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&raw_desc));

            descriptor_handle = std::shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t p) {
                cudnnDestroyTensorDescriptor(p);
            });
        }

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NHWC, dtype, n, c, h, w));
    }

#endif

};

// Eigen Matrix/Vector data-manipulation helpers (slice_rows, filter_missing_values,
// shuffle_rows, is_binary, is_constant, etc.) live in statistics.h now — they
// operate on Eigen types, not TensorView, and conceptually belong with the
// statistical/data-prep code there.

template<typename T, size_t N>
using array = Eigen::array<T, N>;

string shape_to_string(const Shape&, const string& = " ");
Shape string_to_shape(const string&, const string& = " ");


// get_maximum_size was inlined into its single caller (language_dataset.cpp).
// operator<< for vector<T> moved to pch.h so it's available everywhere.
// LtMatmulPlan / LtMatmulPlanKey / LtMatmulPlanKeyHash live in cuda_gemm.h.

// Configuration class (singleton runtime configuration) moved to configuration.h.

// Container for CUDA/cuBLAS/cuDNN handles, streams and lazily-allocated workspaces.
// It is *infrastructure* — owning these resources for the life of the program. The
// "are we on GPU?" question is answered by Configuration, not here, so this class
// no longer carries a DeviceType: that state would just duplicate Configuration's.
class Device
{
public:

    static Device& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int num_threads);

    static cublasHandle_t get_cublas_handle()                      { return instance().cublas_handle; }
    static cublasLtHandle_t get_cublas_lt_handle()                 { return instance().cublas_lt_handle; }
    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn_handle; }
    static cudaStream_t get_compute_stream()                       { return instance().compute_stream; }
    static cudnnOpTensorDescriptor_t get_operator_sum_descriptor() { return instance().operator_sum_descriptor; }
    static cudnnOpTensorDescriptor_t get_operator_multiplication_descriptor() { return instance().operator_multiplication_descriptor; }

    // cuBLASLt workspace, BF16 input scratch, and the matmul plan cache live
    // in cuda_gemm.h/.cpp now. Device only owns generic runtime context.

private:
    Device();
    ~Device();

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    cublasHandle_t cublas_handle = nullptr;
    cublasLtHandle_t cublas_lt_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
};

inline ThreadPoolDevice& get_device()
{
    return *Device::instance().get_thread_pool_device();
}

inline void TensorView::fill(float value)
{
    if(!data) return;

#ifdef OPENNN_WITH_CUDA
    // Decide CPU vs GPU per buffer, not per-Device. set_parameters_random()
    // calls fill() on TensorViews that still point at host memory even when
    // Device is already set to Gpu (params are migrated only later via
    // copy_parameters_device). Probing the pointer keeps this dispatch
    // independent of the global Device state.
    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, data);
    const bool gpu_data = (err == cudaSuccess) && (attr.type == cudaMemoryTypeDevice);
    if (err != cudaSuccess) cudaGetLastError();   // clear sticky error from CPU pointer probe

    if (gpu_data)
    {
        if(value == 0.0f)
        {
            CHECK_CUDA(cudaMemset(data, 0, byte_size()));
            return;
        }

        CHECK_CUDNN(cudnnSetTensor(Device::get_cudnn_handle(),
                                   get_descriptor(), data, &value));
        return;
    }
#endif

    assert(dtype == CUDNN_DATA_FLOAT);
    float* p = static_cast<float*>(data);
    std::fill(p, p + size(), value);
}

#ifdef OPENNN_WITH_CUDA

// Scalar pointer-args used by cuDNN op-tensor calls (cudnnOpTensor sees these
// as host scalars). Kept here because they're shared across cuDNN op sites
// (sum, multiplication) — not GEMM-specific.
inline const float one = 1.0f;
inline const float zero = 0.0f;
inline const float minus_one = -1.0f;

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
