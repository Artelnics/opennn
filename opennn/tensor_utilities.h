//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "configuration.h"

namespace opennn
{

static constexpr Index ALIGN_BYTES = EIGEN_MAX_ALIGN_BYTES;
static constexpr Index ALIGN_ELEMENTS = ALIGN_BYTES / sizeof(float);

inline int to_int(Index value) { return static_cast<int>(value); }
inline float to_type(Index value) { return static_cast<float>(value); }

inline Index align_up(Index value, Index alignment)
{
    return value == 0 ? 0 : (value + alignment - 1) & ~(alignment - 1);
}

inline Index get_aligned_size(Index size)     { return align_up(size,    ALIGN_ELEMENTS); }
inline Index get_aligned_bytes(Index n_bytes) { return align_up(n_bytes, ALIGN_BYTES); }
inline Index get_aligned_bytes(Index count, Type dtype) { return get_aligned_bytes(count * type_bytes(dtype)); }

template<typename Container>
inline Index ssize(const Container& container) noexcept
{
    return static_cast<Index>(container.size());
}

inline bool is_aligned(const void* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % ALIGN_BYTES == 0;
}

constexpr cudaDataType_t      CUDA_REDUCTION_DTYPE   = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE_DTYPE   = CUBLAS_COMPUTE_32F_FAST_TF32;

struct Shape
{
    static constexpr size_t MaxRank = 4;

    Index  dims[MaxRank] = {0};
    size_t rank          = 0;

    Shape() noexcept = default;

    Shape(size_t new_rank, Index value) : rank(new_rank)
    {
        if (new_rank > MaxRank)
            throw runtime_error("Shape: rank " + to_string(new_rank)
                                + " exceeds MaxRank=" + to_string(MaxRank) + ".");
        fill_n(dims, rank, value);
    }

    Shape(initializer_list<Index> list) : rank(list.size())
    {
        if (list.size() > MaxRank)
            throw runtime_error("Shape: initializer rank " + to_string(list.size())
                                + " exceeds MaxRank=" + to_string(MaxRank) + ".");
        copy_n(list.begin(), rank, dims);
    }

    const Index* begin() const noexcept { return dims; }
    const Index* end()   const noexcept { return dims + rank; }
    const Index& operator[](size_t i) const noexcept { return dims[i]; }
    Index&       operator[](size_t i)       noexcept { return dims[i]; }

    Index& back()             { if (rank == 0) throw runtime_error("Shape::back() on empty"); return dims[rank - 1]; }
    const Index& back() const { if (rank == 0) throw runtime_error("Shape::back() on empty"); return dims[rank - 1]; }

    bool empty() const noexcept { return rank == 0; }

    Index dim_or_zero(size_t i) const noexcept { return i < rank ? dims[i] : Index(0); }

    Index size() const noexcept
    {
        return rank == 0 ? 0 : accumulate(begin(), end(), Index(1), multiplies<>{});
    }

    void clear() noexcept { rank = 0; }
    void push_back(Index value) noexcept { if (rank < MaxRank) dims[rank++] = value; }

    friend ostream& operator<<(ostream& os, const Shape& shape)
    {
        os << "[";
        for (size_t i = 0; i < shape.rank; ++i) os << (i ? ", " : " ") << shape.dims[i];
        os << " ]";
        return os;
    }

    bool operator==(const Shape& other) const noexcept
    {
        return rank == other.rank && equal(begin(), end(), other.begin());
    }

    bool operator!=(const Shape& other) const noexcept { return !(*this == other); }

    Shape& append(const Shape& other)
    {
        const size_t copy_count = min(other.rank, MaxRank - rank);
        copy_n(other.dims, copy_count, dims + rank);
        rank += copy_count;
        return *this;
    }
};

inline Index get_aligned_size(const vector<Shape>& shapes)
{
    return transform_reduce(shapes.begin(), shapes.end(), Index(0), plus<>{},
        [](const Shape& s) { return get_aligned_size(s.size()); });
}

inline Index get_aligned_size(const vector<vector<Shape>>& shapes)
{
    return transform_reduce(shapes.begin(), shapes.end(), Index(0), plus<>{},
        [](const vector<Shape>& s) { return get_aligned_size(s); });
}

inline Index get_aligned_bytes(const vector<Shape>& shapes, const vector<Type>& dtypes)
{
    return transform_reduce(shapes.begin(), shapes.end(), dtypes.begin(), Index(0), plus<>{},
        [](const Shape& s, Type t) { return get_aligned_bytes(s.size(), t); });
}

inline Index get_aligned_bytes(const vector<vector<Shape>>& shapes,
                               const vector<vector<Type>>& dtypes)
{
    return transform_reduce(shapes.begin(), shapes.end(), dtypes.begin(), Index(0), plus<>{},
        [](const vector<Shape>& s, const vector<Type>& t) { return get_aligned_bytes(s, t); });
}

inline Index get_aligned_bytes(const vector<Shape>& shapes, Type dtype)
{
    return transform_reduce(shapes.begin(), shapes.end(), Index(0), plus<>{},
        [dtype](const Shape& s) { return get_aligned_bytes(s.size(), dtype); });
}

inline Index get_aligned_bytes(const vector<vector<Shape>>& shapes, Type dtype)
{
    return transform_reduce(shapes.begin(), shapes.end(), Index(0), plus<>{},
        [dtype](const vector<Shape>& s) { return get_aligned_bytes(s, dtype); });
}

struct Buffer
{
    void* data = nullptr;
    Index bytes = 0;
    Device device_type = Device::CPU;

    template<typename T> T*       as()       { return static_cast<T*>(data); }
    template<typename T> const T* as() const { return static_cast<const T*>(data); }

    Index size_in_floats() const { return bytes / Index(sizeof(float)); }
    bool  empty() const { return bytes == 0; }

    void resize_bytes(Index n_bytes, Device new_device_type)
    {
        if (n_bytes == bytes && device_type == new_device_type) return;
        free_buffer();
        device_type = new_device_type;
        if (n_bytes > 0) data = alloc(new_device_type, n_bytes);
        bytes = n_bytes;
    }

    void grow_to(Index n_bytes)
    {
        if (n_bytes > bytes)
            resize_bytes(n_bytes, device_type);
    }

    template<typename T>
    T* ensure(Index n_elements)
    {
        grow_to(n_elements * Index(sizeof(T)));
        return as<T>();
    }

    void* ensure_bytes(size_t min_bytes)
    {
        if (min_bytes == 0) return nullptr;
        grow_to(Index(min_bytes));
        return data;
    }

    void setZero()
    {
        if (!data) return;
#ifdef OPENNN_HAS_CUDA
        if (device_type == Device::CUDA) CHECK_CUDA(cudaMemset(data, 0, bytes));
        else
#endif
            memset(data, 0, static_cast<size_t>(bytes));
    }

#ifdef OPENNN_HAS_CUDA
    void migrate_to(Device target)
    {
        if (device_type == target || !data) return;
        void* fresh = alloc(target, bytes);
        const cudaMemcpyKind kind = (target == Device::CUDA)
            ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        CHECK_CUDA(cudaMemcpy(fresh, data, bytes, kind));
        dealloc(device_type, data, bytes);
        data = fresh;
        device_type = target;
    }

    void migrate_to(Device target, cudaStream_t stream)
    {
        if (device_type == target || !data) return;
        if (!stream) { migrate_to(target); return; }

        void* fresh = alloc(target, bytes);
        const cudaMemcpyKind kind = (target == Device::CUDA)
            ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        CHECK_CUDA(cudaMemcpyAsync(fresh, data, bytes, kind, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        dealloc(device_type, data, bytes);
        data = fresh;
        device_type = target;
    }
#endif

    explicit Buffer(Device new_device_type = Device::CPU) noexcept : device_type(new_device_type) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept : Buffer() { swap(other); }
    Buffer& operator=(Buffer&& other) noexcept { swap(other); return *this; }

    ~Buffer() { free_buffer(); }

    void swap(Buffer& other) noexcept
    {
        std::swap(data, other.data);
        std::swap(bytes, other.bytes);
        std::swap(device_type, other.device_type);
    }

private:
    static void* alloc([[maybe_unused]] Device device_type, Index byte_count)
    {
#ifdef OPENNN_HAS_CUDA
        if (device_type == Device::CUDA) { void* device_pointer = nullptr; CHECK_CUDA(cudaMalloc(&device_pointer, byte_count)); return device_pointer; }
#endif
        return Eigen::aligned_allocator<uint8_t>{}.allocate(static_cast<size_t>(byte_count));
    }

    static void dealloc([[maybe_unused]] Device device_type, void* pointer, Index byte_count)
    {
#ifdef OPENNN_HAS_CUDA
        if (device_type == Device::CUDA) { cudaFree(pointer); return; }
#endif
        Eigen::aligned_allocator<uint8_t>{}.deallocate(static_cast<uint8_t*>(pointer), static_cast<size_t>(byte_count));
    }

    void free_buffer()
    {
        if (data) dealloc(device_type, data, bytes);
        data = nullptr;
        bytes = 0;
    }
};

struct TensorView
{
    void* data = nullptr;

    Shape shape;

    Type type = Type::FP32;

    TensorView(void* new_data = nullptr, const Shape& new_shape = {},
               Type new_dtype = Type::FP32) noexcept
        : data(new_data), shape(new_shape), type(new_dtype) {}

    Index get_rank() const noexcept { return shape.rank; }

    Index size() const noexcept { return shape.size(); }

    Index byte_size() const noexcept { return size() * type_bytes(type); }

    bool empty() const noexcept { return shape.empty(); }

    template<typename T>
    T* as() const noexcept
    {
        assert(data);
        return reinterpret_cast<T*>(data);
    }

    float* as_float() const noexcept
    {
        return reinterpret_cast<float*>(data);
    }

    cudaDataType_t cuda_dtype() const noexcept { return to_cuda(type); }

    template<typename F>
    void dispatch(F&& fn) const
    {
        visit_type<Type::FP32, Type::BF16>(type, [&](auto info)
        {
            fn(typename decltype(info)::type{});
        });
    }

    TensorView reshape(const Shape& new_shape) const
    { return TensorView(data, new_shape, type); }

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
        copy_n(shape.dims, Rank, dims.begin());
        return TensorMapR<Rank>(as<float>(), dims);
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor(Index batch_index) const
    {
        assert(shape.rank == Rank + 1);
        Eigen::array<Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) dims[i] = shape[i + 1];
        const Index slice_size = shape.size() / shape[0];
        return TensorMapR<Rank>(as<float>() + batch_index * slice_size, dims);
    }

    void fill(float value);
    void setZero() { fill(0.0f); }

#ifdef OPENNN_HAS_CUDA
    void set_zero_async() const;

    mutable shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    cudnnTensorDescriptor_t get_descriptor() const
    {
        if (!descriptor_handle && !shape.empty())
            set_descriptor(shape);
        return descriptor_handle.get();
    }

private:
    void set_descriptor(const Shape& shape) const
    {
        // NHWC layout: rank < 4 leading dims default to 1.
        int batch_count = 1, channels = 1, height = 1, width = 1;
        const size_t rank = shape.rank;
        if (rank >= 1) channels    = static_cast<int>(shape[rank - 1]);
        if (rank >= 2) batch_count = static_cast<int>(shape[0]);
        if (rank >= 3) width       = static_cast<int>(shape[rank - 2]);
        if (rank >= 4) height      = static_cast<int>(shape[rank - 3]);

        if (batch_count <= 0 || channels <= 0 || height <= 0 || width <= 0)
            return;

        if (!descriptor_handle)
        {
            cudnnTensorDescriptor_t raw_desc;
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&raw_desc));

            descriptor_handle = shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t descriptor) {
                cudnnDestroyTensorDescriptor(descriptor);
            });
        }

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NHWC, to_cudnn(type), batch_count, channels, height, width));
    }

#endif

};

inline TensorView& view_at_slot_or(vector<TensorView>& views,
                                   const vector<size_t>& slots, size_t i,
                                   TensorView& fallback)
{
    return i < slots.size() ? views[slots[i]] : fallback;
}

inline TensorView& view_at_slot_or(vector<vector<TensorView>>& views,
                                   const vector<size_t>& slots, size_t i,
                                   TensorView& fallback)
{
    return i < slots.size() ? views[slots[i]][0] : fallback;
}

template<typename T, size_t N>
using array = Eigen::array<T, N>;

string shape_to_string(const Shape&, const string& = " ");
Shape string_to_shape(const string&, const string& = " ");

// Boost-style hash combine. Mixes one or more values into a single size_t. Used
// for plan/graph cache keys (cuBLASLt, cuDNN SDPA).
template<typename... Vs>
size_t hash_combine(const Vs&... values)
{
    size_t h = 0;
    ((h ^= hash<Vs>{}(values) + 0x9e3779b9 + (h << 6) + (h >> 2)), ...);
    return h;
}

class Backend
{
public:

    static Backend& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int num_threads);

    static cublasHandle_t get_cublas_handle()                      { return instance().cublas_handle; }
    static cublasLtHandle_t get_cublas_lt_handle()                 { return instance().cublas_lt_handle; }
    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn_handle; }
    static cudaStream_t get_compute_stream()                       { return instance().compute_stream; }
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
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
};

inline ThreadPoolDevice& get_device()
{
    return *Backend::instance().get_thread_pool_device();
}

inline void TensorView::fill(float value)
{
    if (!data) return;

#ifdef OPENNN_HAS_CUDA
    // Probe the pointer: set_parameters_random() may call fill() on host-resident
    // TensorViews even when Device is already GPU.
    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, data);
    const bool gpu_data = (err == cudaSuccess) && (attr.type == cudaMemoryTypeDevice);
    if (err != cudaSuccess) cudaGetLastError();   // clear sticky error from CPU pointer probe

    if (gpu_data)
    {
        if (value == 0.0f)
        {
            CHECK_CUDA(cudaMemset(data, 0, byte_size()));
            return;
        }

        CHECK_CUDNN(cudnnSetTensor(Backend::get_cudnn_handle(),
                                   get_descriptor(), data, &value));
        return;
    }
#endif

    assert(type == Type::FP32);
    float* data_pointer = static_cast<float*>(data);
    std::fill(data_pointer, data_pointer + size(), value);
}

#ifdef OPENNN_HAS_CUDA

inline void TensorView::set_zero_async() const
{
    if (!data || byte_size() == 0) return;
    CHECK_CUDA(cudaMemsetAsync(data, 0, byte_size(), Backend::get_compute_stream()));
}

inline const float one = 1.0f;
inline const float zero = 0.0f;

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
