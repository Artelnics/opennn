//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include <queue>
#include <mutex>
#include <condition_variable>

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

template <typename Enum>
struct EnumMap
{
    using Entry = pair<Enum, string>;

    const vector<Entry>& entries;

    const string& to_string(Enum value) const
    {
        for(const auto& [e, s] : entries)
            if(e == value)
                return s;
        throw runtime_error("Unknown enum value");
    }

    Enum from_string(const string& name) const
    {
        for(const auto& [e, s] : entries)
            if(s == name)
                return e;
        throw runtime_error("Unknown enum string: " + name);
    }

    Enum from_string(const string& name, Enum fallback) const
    {
        for(const auto& [e, s] : entries)
            if(s == name)
                return e;
        return fallback;
    }
};

constexpr cudaDataType_t      CUDA_REDUCTION_DTYPE   = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE_DTYPE   = CUBLAS_COMPUTE_32F_FAST_TF32;

#if defined(OPENNN_BF16_ACTIVATIONS) && defined(OPENNN_WITH_CUDA)
constexpr cudnnDataType_t     CUDNN_ACTIVATION_DTYPE = CUDNN_DATA_BFLOAT16;
constexpr cudaDataType_t      CUDA_ACTIVATION_DTYPE  = CUDA_R_16BF;
#else
constexpr cudnnDataType_t     CUDNN_ACTIVATION_DTYPE = CUDNN_DATA_FLOAT;
constexpr cudaDataType_t      CUDA_ACTIVATION_DTYPE  = CUDA_R_32F;
#endif

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

template <typename T>
class ThreadSafeQueue
{
public:

    void push(T item)
    {
        { lock_guard<mutex> lock(mutex_); queue_.push(std::move(item)); }
        cond_.notify_one();
    }

    T pop()
    {
        unique_lock<mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    bool empty() const
    {
        lock_guard<mutex> lock(mutex_);
        return queue_.empty();
    }

private:

    queue<T> queue_;
    mutable mutex mutex_;
    condition_variable cond_;
};

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

enum class DeviceType { Cpu, Gpu };

struct Buffer
{
    void* data = nullptr;
    Index bytes = 0;
    DeviceType device_type = DeviceType::Cpu;

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
        if(device_type == DeviceType::Gpu) CHECK_CUDA(cudaMemset(data, 0, bytes));
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
        const cudaMemcpyKind kind = (target == DeviceType::Gpu)
            ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        CHECK_CUDA(cudaMemcpy(fresh, data, bytes, kind));
        dealloc(device_type, data, bytes);
        data = fresh;
        device_type = target;
    }
#endif

    explicit Buffer(DeviceType t = DeviceType::Cpu) noexcept : device_type(t) {}
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
        if(t == DeviceType::Gpu) { void* p = nullptr; CHECK_CUDA(cudaMalloc(&p, n)); return p; }
#endif
        return Eigen::aligned_allocator<uint8_t>{}.allocate(static_cast<size_t>(n));
    }

    static void dealloc(DeviceType t, void* p, Index n)
    {
#ifdef OPENNN_WITH_CUDA
        if(t == DeviceType::Gpu) { cudaFree(p); return; }
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

    cudnnDataType_t dtype = CUDNN_ACTIVATION_DTYPE;

    TensorView(void* new_data = nullptr, const Shape& new_shape = {},
               cudnnDataType_t new_dtype = CUDNN_ACTIVATION_DTYPE) noexcept
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

inline bool row_finite(const VectorR& v, Index i) { return isfinite(v(i)); }
inline bool row_finite(const MatrixR& m, Index i) { return m.row(i).array().isFinite().all(); }

inline VectorR slice_rows(const VectorR& v, const vector<Index>& idx)
{
    VectorR r(idx.size());
    for (Index i = 0; i < Index(idx.size()); ++i) r(i) = v(idx[i]);
    return r;
}

inline MatrixR slice_rows(const MatrixR& m, const vector<Index>& idx)
{
    MatrixR r(idx.size(), m.cols());
    for (Index i = 0; i < Index(idx.size()); ++i) r.row(i) = m.row(idx[i]);
    return r;
}

VectorR filter_missing_values(const VectorR&);

template<typename X, typename Y>
pair<X, Y> filter_missing_values(const X& x, const Y& y)
{
    if (x.rows() != y.rows())
        throw runtime_error("filter_missing_values: row count mismatch");

    vector<Index> valid;
    valid.reserve(x.rows());

    for (Index i = 0; i < x.rows(); ++i)
        if (row_finite(x, i) && row_finite(y, i))
            valid.push_back(i);

    return { slice_rows(x, valid), slice_rows(y, valid) };
}

void shuffle_rows(MatrixR& matrix);

template<typename T, size_t N>
using array = Eigen::array<T, N>;

inline bool is_contiguous(const vector<Index>& v)
{
    return std::adjacent_find(v.begin(), v.end(),
        [](Index a, Index b) { return b != a + 1; }) == v.end();
}

template <typename T>
inline bool is_binary(const T& tensor)
{
    return all_of(tensor.data(), tensor.data() + tensor.size(),
                  [](type v) { return v == type(0) || v == type(1) || isnan(v); });
}

MatrixR append_rows(const MatrixR& , const MatrixR&);

template<typename T>
vector<T> gather_by_index(const vector<T>& data, const vector<Index>& indices)
{
    vector<T> result;
    result.reserve(indices.size());

    transform(indices.begin(), indices.end(), back_inserter(result),
              [&data](Index i) { return data[i]; });

    return result;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums);

template <typename T>
inline bool is_constant(const T& tensor)
{
    const type* data = tensor.data();
    const type* end = data + tensor.size();

    const type* first = find_if(data, end, [](type v) { return !isnan(v); });

    if (first == end)
        return true;

    const type val = *first;

    return all_of(first + 1, end,
                  [val](type v) { return isnan(v) || abs(val - v) <= numeric_limits<float>::min(); });
}

inline vector<Index> get_true_indices(const VectorB& v)
{
    vector<Index> indices;
    indices.reserve(v.size());

    for(Index i = 0; i < v.size(); ++i)
        if (v(i))
            indices.push_back(i);

    return indices;
}

VectorI calculate_rank(const VectorR&, bool ascending = true);

vector<Index> get_elements_greater_than(const vector<Index>&, Index);

VectorI get_nearest_points(const MatrixR& ,const VectorR& , int = 1);

void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&, type*, bool = true, int contiguous = -1);

VectorR perform_Householder_QR_decomposition(const MatrixR&, const VectorR&);

string shape_to_string(const Shape&, const string& = " ");
Shape string_to_shape(const string&, const string& = " ");

VectorMap vector_map(const MatrixR&, Index);


template <typename T>
size_t get_maximum_size(const vector<vector<T>>& v)
{
    size_t maximum_size = 0;

    for(const auto& inner : v)
        if (inner.size() > maximum_size)
            maximum_size = inner.size();

    return maximum_size;
}

template <typename T>
ostream& operator << (ostream& os, const vector<T>& vec)
{
    os << "[ ";

    for(size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i + 1 < vec.size())
            os << "; ";
    }

    os << " ]";
    return os;
}

#ifdef OPENNN_WITH_CUDA

// Cached cuBLASLt plan: 1 op desc + 4 layout descs + a heuristic-selected algo.
// Built once per (m, n, k, transA, transB) shape; bias *pointer* is set per call
// because it varies per layer/iteration.
struct LtMatmulPlan
{
    cublasLtMatmulDesc_t   op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc  = nullptr;
    cublasLtMatrixLayout_t b_desc  = nullptr;
    cublasLtMatrixLayout_t c_desc  = nullptr;
    cublasLtMatrixLayout_t d_desc  = nullptr;
    cublasLtMatmulAlgo_t   algo{};
    bool                   algo_valid = false;

    LtMatmulPlan() = default;
    LtMatmulPlan(const LtMatmulPlan&) = delete;
    LtMatmulPlan& operator=(const LtMatmulPlan&) = delete;
    LtMatmulPlan(LtMatmulPlan&& o) noexcept { *this = std::move(o); }
    LtMatmulPlan& operator=(LtMatmulPlan&& o) noexcept
    {
        std::swap(op_desc, o.op_desc);
        std::swap(a_desc,  o.a_desc);
        std::swap(b_desc,  o.b_desc);
        std::swap(c_desc,  o.c_desc);
        std::swap(d_desc,  o.d_desc);
        std::swap(algo,    o.algo);
        std::swap(algo_valid, o.algo_valid);
        return *this;
    }
    ~LtMatmulPlan()
    {
        cublasLtMatrixLayoutDestroy(d_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(op_desc);
    }
};

struct LtMatmulPlanKey
{
    int m;
    int n;
    int k;
    int transA;
    int transB;
    int epilogue;   // cublasLtEpilogue_t cast to int (e.g. BIAS, RELU_BIAS)
    int io_dtype;   // cudaDataType_t for A and B (inputs)
    int out_dtype;  // cudaDataType_t for C and D (outputs)

    bool operator==(const LtMatmulPlanKey& o) const noexcept
    {
        return m == o.m && n == o.n && k == o.k
            && transA == o.transA && transB == o.transB
            && epilogue == o.epilogue
            && io_dtype == o.io_dtype && out_dtype == o.out_dtype;
    }
};

struct LtMatmulPlanKeyHash
{
    size_t operator()(const LtMatmulPlanKey& k) const noexcept
    {
        // Standard mix; keys are int-tuples, modest cardinality.
        size_t h = std::hash<int>{}(k.m);
        const auto mix = [](size_t& acc, int v) {
            acc ^= std::hash<int>{}(v) + 0x9e3779b9 + (acc << 6) + (acc >> 2);
        };
        mix(h, k.n);
        mix(h, k.k);
        mix(h, k.transA);
        mix(h, k.transB);
        mix(h, k.epilogue);
        mix(h, k.io_dtype);
        mix(h, k.out_dtype);
        return h;
    }
};

#endif // OPENNN_WITH_CUDA

class Device
{
public:
    static Device& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int num_threads);

    void set(DeviceType type);
    bool is_gpu() const { return device_type == DeviceType::Gpu; }
    bool is_cpu() const { return device_type == DeviceType::Cpu; }

    static cublasHandle_t get_cublas_handle()                      { return instance().cublas_handle; }
    static cublasLtHandle_t get_cublas_lt_handle()                 { return instance().cublas_lt_handle; }
    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn_handle; }
    static cudaStream_t get_compute_stream()                       { return instance().compute_stream; }
    static cudnnOpTensorDescriptor_t get_operator_sum_descriptor() { return instance().operator_sum_descriptor; }
    static cudnnOpTensorDescriptor_t get_operator_multiplication_descriptor() { return instance().operator_multiplication_descriptor; }

#ifdef OPENNN_WITH_CUDA
    static float* get_ones(int n)
    {
        auto& self = instance();
        if(n > self.ones_size)
        {
            if(self.ones_device) CHECK_CUDA(cudaFree(self.ones_device));
            const size_t elem_bytes = dtype_bytes(CUDNN_ACTIVATION_DTYPE);
            CHECK_CUDA(cudaMalloc(&self.ones_device, n * elem_bytes));

            if (CUDNN_ACTIVATION_DTYPE == CUDNN_DATA_FLOAT) {
                vector<float> h(n, 1.0f);
                CHECK_CUDA(cudaMemcpy(self.ones_device, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            } else {
                vector<__nv_bfloat16> h(n, __float2bfloat16(1.0f));
                CHECK_CUDA(cudaMemcpy(self.ones_device, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            }
            self.ones_size = n;
        }
        return self.ones_device;
    }

    // 32 MB matches NVIDIA's cuBLASLt sample default. Lazy-allocated on first use,
    // shared across all cublasLtMatmul call sites.
    static constexpr size_t cublas_lt_workspace_bytes() { return 32ull * 1024 * 1024; }

    static void* get_cublas_lt_workspace()
    {
        auto& self = instance();
        if(!self.cublas_lt_workspace)
            CHECK_CUDA(cudaMalloc(&self.cublas_lt_workspace, cublas_lt_workspace_bytes()));
        return self.cublas_lt_workspace;
    }

    // BF16 scratch buffer used when a layer (typically the first trainable Dense)
    // receives an FP32 input but its weights live in the BF16 working copy.
    // cuBLASLt requires A and B to share dtype, so we down-cast the input to BF16
    // here and feed the scratch into the matmul. Lazy-grown to fit the largest
    // request seen so far; callers must not retain the pointer across batches.
    static __nv_bfloat16* get_bf16_input_scratch(Index n_elements)
    {
        auto& self = instance();
        const size_t needed_bytes = size_t(n_elements) * sizeof(__nv_bfloat16);
        if(needed_bytes > self.bf16_input_scratch_bytes)
        {
            if(self.bf16_input_scratch) cudaFree(self.bf16_input_scratch);
            CHECK_CUDA(cudaMalloc(&self.bf16_input_scratch, needed_bytes));
            self.bf16_input_scratch_bytes = needed_bytes;
        }
        return reinterpret_cast<__nv_bfloat16*>(self.bf16_input_scratch);
    }

    // Returns a cached cuBLASLt plan for a GEMM with the requested epilogue.
    // Built (descriptors + heuristic algo) once per unique key and reused.
    // The bias *pointer* is set on the returned op_desc by the caller per
    // call — it is not part of the cache key.
    //
    // `io_dtype` is the dtype of the GEMM inputs (A and B); `out_dtype` is the
    // dtype of the outputs (C and D). For pure FP32 they're both CUDA_R_32F;
    // for fully-bf16 forward (bias_add path) they're both CUDA_R_16BF; for
    // mixed-precision backward (BGRADA, weight gradient) inputs are bf16 but
    // weight_grad output stays FP32. Defaults preserve legacy behaviour.
    //
    // Supported epilogues: CUBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_RELU_BIAS,
    // CUBLASLT_EPILOGUE_BGRADA.
    static const LtMatmulPlan& get_lt_gemm_plan(
        int m, int n, int k,
        cublasOperation_t transA,
        cublasOperation_t transB,
        cublasLtEpilogue_t epilogue,
        cudaDataType_t io_dtype  = CUDA_ACTIVATION_DTYPE,
        cudaDataType_t out_dtype = CUDA_ACTIVATION_DTYPE);
#endif

private:
    Device();
    ~Device();

    DeviceType device_type = DeviceType::Cpu;

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    cublasHandle_t cublas_handle = nullptr;
    cublasLtHandle_t cublas_lt_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
    float* ones_device = nullptr;
    int ones_size = 0;
    void* cublas_lt_workspace = nullptr;
    void* bf16_input_scratch = nullptr;
    size_t bf16_input_scratch_bytes = 0;

#ifdef OPENNN_WITH_CUDA
    std::unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_gemm_plans;
#endif
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

inline const float one = 1.0f;
inline const float zero = 0.0f;
inline const float minus_one = -1.0f;

inline void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k,
                      const void* A, cudaDataType_t Atype, int lda,
                      const void* B, cudaDataType_t Btype, int ldb,
                      void* C, cudaDataType_t Ctype, int ldc,
                      float alpha = 1.0f, float beta = 0.0f)
{
    // CUBLAS_COMPUTE_32F_FAST_TF32 only triggers TF32 rounding for FP32 inputs;
    // for BF16 inputs cuBLAS rejects it (NOT_SUPPORTED) and we want
    // CUBLAS_COMPUTE_32F (FP32 accumulator over BF16 Tensor Cores).
    const cublasComputeType_t compute = (Atype == CUDA_R_16BF || Btype == CUDA_R_16BF)
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_DTYPE;
    CHECK_CUBLAS(cublasGemmEx(Device::get_cublas_handle(),
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

inline void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k,
                                      const float* A, int lda, long long stride_a,
                                      const float* B, int ldb, long long stride_b,
                                      float* C, int ldc, long long stride_c,
                                      int batch_count,
                                      float alpha = 1.0f, float beta = 0.0f)
{
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(Device::get_cublas_handle(),
                                            transa, transb,
                                            m, n, k,
                                            &alpha,
                                            A, CUDA_ACTIVATION_DTYPE, lda, stride_a,
                                            B, CUDA_ACTIVATION_DTYPE, ldb, stride_b,
                                            &beta,
                                            C, CUDA_ACTIVATION_DTYPE, ldc, stride_c,
                                            batch_count,
                                            CUBLAS_COMPUTE_DTYPE,
                                            CUBLAS_GEMM_DEFAULT));
}

// Fused GEMM + (optionally activation +) bias via cuBLASLt epilogue: one launch
// for the whole gemm-bias[-relu] sequence, no intermediate write-then-read of
// the output tensor between stages.
//
// `epilogue` selects the post-matmul fusion. Currently supported:
//   - CUBLASLT_EPILOGUE_BIAS       : D = α(A·B) + bias                (default)
//   - CUBLASLT_EPILOGUE_RELU_BIAS  : D = max(0, α(A·B) + bias)
//
// Layout assumptions (encoded in the cached plan):
//   - All operands use CUDA_ACTIVATION_DTYPE.
//   - Bias is FP32, length = m, broadcast along D's columns
//     (i.e. one bias element per output feature row).
//   - Tightly packed: lda = (transa==N ? m : k), ldb = (transb==N ? k : n), ldc = ldd = m.
//     If a caller needs different strides, it should not use this wrapper.
//
// Not thread-safe: the bias pointer is set on the cached op_desc per call, so concurrent
// invocations on the same shape would race. Matches the rest of this codebase's
// single-stream GPU usage assumption.
inline void gemm_bias_cuda(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const void* A,
                           const void* B,
                           void* C,
                           const float* bias,
                           cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS,
                           float alpha = 1.0f, float beta = 0.0f)
{
    const LtMatmulPlan& plan = Device::get_lt_gemm_plan(m, n, k, transa, transb, epilogue);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    void* workspace = Device::get_cublas_lt_workspace();
    const size_t workspace_size = Device::cublas_lt_workspace_bytes();

    CHECK_CUBLAS(cublasLtMatmul(Device::get_cublas_lt_handle(),
                                plan.op_desc,
                                &alpha,
                                A, plan.a_desc,
                                B, plan.b_desc,
                                &beta,
                                C, plan.c_desc,   // C (read when beta != 0)
                                C, plan.d_desc,   // D (write); aliasing C is supported
                                plan.algo_valid ? &plan.algo : nullptr,
                                workspace, workspace_size,
                                Device::get_compute_stream()));
}

// Fused matmul + bias-gradient via cuBLASLt's BGRADA epilogue. Computes the
// matmul C = α(A·B) + βC and, as a side product, writes the row-wise reduction
// of A (in cuBLAS column-major view) into `bias_grad`. For Dense backward,
// pass A = output_delta with transA = N to get bias_grad = sum_rows(dY) — i.e.
// the bias gradient — for free, replacing a separate sum() reduction kernel.
//
// `bias_grad` length must be m (the matmul's M dim, == D rows in column-major).
// Same threading caveat as gemm_bias_cuda.
inline void gemm_bgrad_cuda(cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void* A,
                            const void* B,
                            void* C,
                            float* bias_grad,
                            float alpha = 1.0f, float beta = 0.0f)
{
    // Inputs (output_delta, input) follow the activation dtype; the weight
    // gradient (D / C) is always FP32 so Adam can accumulate without precision
    // loss. cuBLASLt mixed-dtype matmul handles bf16 in × fp32 out natively.
    const LtMatmulPlan& plan = Device::get_lt_gemm_plan(m, n, k, transa, transb,
                                                        CUBLASLT_EPILOGUE_BGRADA,
                                                        CUDA_ACTIVATION_DTYPE,
                                                        CUDA_R_32F);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_grad, sizeof(bias_grad)));

    void* workspace = Device::get_cublas_lt_workspace();
    const size_t workspace_size = Device::cublas_lt_workspace_bytes();

    CHECK_CUBLAS(cublasLtMatmul(Device::get_cublas_lt_handle(),
                                plan.op_desc,
                                &alpha,
                                A, plan.a_desc,
                                B, plan.b_desc,
                                &beta,
                                C, plan.c_desc,
                                C, plan.d_desc,
                                plan.algo_valid ? &plan.algo : nullptr,
                                workspace, workspace_size,
                                Device::get_compute_stream()));
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
