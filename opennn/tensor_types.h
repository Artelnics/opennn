//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   T Y P E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "configuration.h"
#include "device_backend.h"

namespace opennn
{

template<Type T> struct TypeInfo;

template<> struct TypeInfo<Type::FP32>
{
    using type = float;
    static constexpr cudnnDataType_t cudnn = CUDNN_DATA_FLOAT;
    static constexpr cudaDataType_t  cuda  = CUDA_R_32F;
    static constexpr Index           bytes = Index(sizeof(float));
    static constexpr const char*     name  = "FP32";
};

template<> struct TypeInfo<Type::BF16>
{
    using type = bfloat16;
    static constexpr cudnnDataType_t cudnn = CUDNN_DATA_BFLOAT16;
    static constexpr cudaDataType_t  cuda  = CUDA_R_16BF;
    static constexpr Index           bytes = Index(sizeof(bfloat16));
    static constexpr const char*     name  = "BF16";
};

template<Type... Supported, typename F>
void visit_type(Type t, F&& f)
{
    bool matched = false;
    ([&]
    {
        if (!matched && t == Supported)
        {
            f(TypeInfo<Supported>{});
            matched = true;
        }
    }(), ...);
    throw_if(!matched, "visit_type: unsupported Type value");
}

template<Type... Supported, typename F>
void visit_type_pair(Type t_in, Type t_out, F&& f)
{
    visit_type<Supported...>(t_in, [&](auto in_info)
    {
        visit_type<Supported...>(t_out, [&](auto out_info)
        {
            f(in_info, out_info);
        });
    });
}

inline cudnnDataType_t to_cudnn(Type type)
{
    switch (type)
    {
    case Type::FP32: return TypeInfo<Type::FP32>::cudnn;
    case Type::BF16: return TypeInfo<Type::BF16>::cudnn;
    case Type::Auto: break;
    }

    throw runtime_error("to_cudnn: Type::Auto must be resolved before tensor use.");
}

inline cudaDataType_t to_cuda(Type type)
{
    switch (type)
    {
    case Type::FP32: return TypeInfo<Type::FP32>::cuda;
    case Type::BF16: return TypeInfo<Type::BF16>::cuda;
    case Type::Auto: break;
    }

    throw runtime_error("to_cuda: Type::Auto must be resolved before tensor use.");
}

inline Index type_bytes(Type type)
{
    switch (type)
    {
    case Type::FP32: return TypeInfo<Type::FP32>::bytes;
    case Type::BF16: return TypeInfo<Type::BF16>::bytes;
    case Type::Auto: break;
    }

    throw runtime_error("type_bytes: Type::Auto must be resolved before tensor use.");
}

static constexpr Index ALIGN_BYTES = EIGEN_MAX_ALIGN_BYTES;
static constexpr Index ALIGN_ELEMENTS = ALIGN_BYTES / sizeof(float);

inline int to_int(Index value)
{
    throw_if(value > Index(numeric_limits<int>::max()) || value < Index(numeric_limits<int>::min()),
             format("to_int: value {} exceeds int range.", value));
    return static_cast<int>(value);
}
inline float to_type(Index value) { return static_cast<float>(value); }

inline Index align_up(Index value, Index alignment)
{
    return value == 0 ? 0 : (value + alignment - 1) & ~(alignment - 1);
}

inline Index get_aligned_size(Index size)     { return align_up(size,    ALIGN_ELEMENTS); }
inline Index get_aligned_bytes(Index n_bytes) { return align_up(n_bytes, ALIGN_BYTES); }
inline Index get_aligned_bytes(Index count, Type dtype) { return get_aligned_bytes(count * type_bytes(dtype)); }

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
        throw_if(new_rank > MaxRank,
                 format("Shape: rank {} exceeds MaxRank={}.",
                        new_rank, MaxRank));
        fill_n(dims, rank, value);
    }

    Shape(initializer_list<Index> list) : rank(list.size())
    {
        throw_if(list.size() > MaxRank,
                 format("Shape: initializer rank {} exceeds MaxRank={}.",
                        list.size(), MaxRank));
        copy_n(list.begin(), rank, dims);
    }

    template<typename It>
    Shape(It first, It last) : rank(size_t(distance(first, last)))
    {
        throw_if(rank > MaxRank,
                 format("Shape: iterator-pair rank {} exceeds MaxRank={}.",
                        rank, MaxRank));
        copy_n(first, rank, dims);
    }

    const Index* begin() const noexcept { return dims; }
    const Index* end()   const noexcept { return dims + rank; }
    const Index& operator[](size_t i) const noexcept { return dims[i]; }
    Index&                     operator[](size_t i)       noexcept { return dims[i]; }

    Index&                     back()       { throw_if(rank == 0, "Shape::back() on empty"); return dims[rank - 1]; }
    const Index& back() const { throw_if(rank == 0, "Shape::back() on empty"); return dims[rank - 1]; }

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

    Shape& append(const Shape& other)
    {
        const size_t copy_count = min(other.rank, MaxRank - rank);
        copy_n(other.dims, copy_count, dims + rank);
        rank += copy_count;
        return *this;
    }
};

struct TensorSpec
{
    Shape shape;
    Type  dtype = Type::FP32;
};

inline Index get_aligned_size(const vector<TensorSpec>& specs)
{
    return transform_reduce(specs.begin(), specs.end(), Index(0), plus<>{},
        [](const auto& spec) { return get_aligned_size(spec.shape.size()); });
}

inline Index get_aligned_size(const vector<vector<TensorSpec>>& specs)
{
    return transform_reduce(specs.begin(), specs.end(), Index(0), plus<>{},
        [](const auto& s) { return get_aligned_size(s); });
}

inline Index get_aligned_bytes(const TensorSpec& spec) { return get_aligned_bytes(spec.shape.size(), spec.dtype); }

inline Index get_aligned_bytes(const vector<TensorSpec>& specs)
{
    return transform_reduce(specs.begin(), specs.end(), Index(0), plus<>{},
        [](const TensorSpec& spec) { return get_aligned_bytes(spec); });
}

inline Index get_aligned_bytes(const vector<vector<TensorSpec>>& specs)
{
    return transform_reduce(specs.begin(), specs.end(), Index(0), plus<>{},
        [](const auto& s) { return get_aligned_bytes(s); });
}

inline Index get_aligned_bytes(const vector<Shape>& shapes, Type dtype)
{
    return transform_reduce(shapes.begin(), shapes.end(), Index(0), plus<>{},
        [dtype](const Shape& s) { return get_aligned_bytes(s.size(), dtype); });
}

inline Index get_aligned_bytes(const vector<TensorSpec>& specs, Type dtype)
{
    return transform_reduce(specs.begin(), specs.end(), Index(0), plus<>{},
        [dtype](const auto& spec) { return get_aligned_bytes(spec.shape.size(), dtype); });
}

inline Index get_aligned_bytes(const vector<vector<TensorSpec>>& specs, Type dtype)
{
    return transform_reduce(specs.begin(), specs.end(), Index(0), plus<>{},
        [dtype](const auto& s) { return get_aligned_bytes(s, dtype); });
}

struct Buffer
{
    void* data = nullptr;
    Index bytes = 0;
    Device device_type = Device::CPU;
    bool owns = true;   // false => non-owning view; the memory is freed by its owner.

    template<typename T> T*       as()       { return static_cast<T*>(data); }
    template<typename T> const T* as() const { return static_cast<const T*>(data); }

    Index size_in_floats() const noexcept { return bytes / Index(sizeof(float)); }
    bool  empty() const noexcept { return bytes == 0; }

    void resize_bytes(Index byte_count, Device allocation_device)
    {
        if (byte_count == bytes && device_type == allocation_device) return;

        const bool changes_cuda_allocation =
            (device_type == Device::CUDA && data)
            || (allocation_device == Device::CUDA && byte_count > 0);
        throw_if(changes_cuda_allocation && device::cuda_allocation_growth_forbidden(),
                 format("CUDA buffer resize from {} to {} bytes while CUDA allocation growth is forbidden "
                        "(warmup incomplete before CUDA graph capture).",
                        bytes,
                        byte_count));

        free_buffer();
        device_type = allocation_device;
        if (byte_count == 0) return;

        data = device::allocate(allocation_device, byte_count);
        bytes = byte_count;
    }

    // Point at memory owned by another Buffer (non-owning view): the viewed
    // memory must outlive this Buffer and is never freed or resized through it.
    // Used to overlay a smaller, temporally-disjoint buffer onto a larger one.
    void set_view(void* external_data, Index byte_count, Device view_device) noexcept
    {
        free_buffer();
        data = external_data;
        bytes = byte_count;
        device_type = view_device;
        owns = false;
    }

    void grow_to(Index minimum_bytes)
    {
        if (minimum_bytes > bytes)
            resize_bytes(minimum_bytes, device_type);
    }

    template<typename T>
    T* ensure(Index element_count)
    {
        grow_to(element_count * Index(sizeof(T)));
        return as<T>();
    }

    void setZero()
    {
        device::set_zero(data, bytes, device_type);
    }

    void migrate_to(Device target_device, cudaStream_t stream = nullptr)
    {
        if (device_type == target_device || !data) return;

        Buffer target_buffer(target_device);
        target_buffer.resize_bytes(bytes, target_device);
        device::copy_async(target_buffer.data, data, bytes, device_type, target_device, stream);
        if (stream) device::synchronize(stream);

        swap(target_buffer);
    }

    explicit Buffer(Device initial_device = Device::CPU) noexcept : device_type(initial_device) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept : Buffer() { swap(other); }
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this == &other) return *this;

        free_buffer();
        data = other.data;
        bytes = other.bytes;
        device_type = other.device_type;
        owns = other.owns;

        other.data = nullptr;
        other.bytes = 0;
        other.device_type = Device::CPU;
        other.owns = true;

        return *this;
    }

    ~Buffer() { free_buffer(); }

    void swap(Buffer& other) noexcept
    {
        std::swap(data, other.data);
        std::swap(bytes, other.bytes);
        std::swap(device_type, other.device_type);
        std::swap(owns, other.owns);
    }

private:
    void free_buffer()
    {
        if (data && owns) device::deallocate(device_type, data, bytes);
        data = nullptr;
        bytes = 0;
        owns = true;
    }
};

struct TensorView
{
    void* data = nullptr;

    Shape shape;

    Type type = Type::FP32;
    Device device = Device::CPU;

    TensorView(void* new_data = nullptr, const Shape& new_shape = {},
               Type new_dtype = Type::FP32,
               Device new_device = Device::CPU) noexcept
        : data(new_data), shape(new_shape), type(new_dtype), device(new_device) {}

    Index get_rank() const noexcept { return shape.rank; }

    Index size() const noexcept { return shape.size(); }

    Index byte_size() const { return size() * type_bytes(type); }

    bool empty() const noexcept { return shape.empty(); }
    bool is_cuda() const noexcept { return device == Device::CUDA; }
    bool is_fp32() const noexcept { return type == Type::FP32; }
    bool is_bf16() const noexcept { return type == Type::BF16; }

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

    cudaDataType_t cuda_dtype() const { return to_cuda(type); }

    template<typename F>
    void dispatch(F&& fn) const
    {
        visit_type<Type::FP32, Type::BF16>(type, [&](auto info)
        {
            fn(typename decltype(info)::type{});
        });
    }

    TensorView reshape(const Shape& new_shape) const
    { return TensorView(data, new_shape, type, device); }

    MatrixMap as_matrix() const
    {
        throw_if(shape.rank < 2, "TensorView::as_matrix requires rank >= 2.");
        throw_if(shape.size() > 0 && !data, "TensorView::as_matrix requires non-null data.");

        const Index row_count = shape[0];
        const Index column_count = row_count == 0 ? 0 : shape.size() / row_count;
        return MatrixMap(reinterpret_cast<float*>(data), row_count, column_count);
    }

    MatrixMap as_matrix(Index matrix_index) const
    {
        throw_if(shape.rank < 2, "TensorView::as_matrix(matrix_index) requires rank >= 2.");
        throw_if(shape.size() > 0 && !data, "TensorView::as_matrix(matrix_index) requires non-null data.");

        const Index row_count = shape[shape.rank - 2];
        const Index column_count = shape[shape.rank - 1];
        const Index matrix_element_count = row_count * column_count;
        const Index matrix_count = matrix_element_count == 0 ? 0 : shape.size() / matrix_element_count;

        throw_if(matrix_index < 0 || matrix_index >= matrix_count,
                 format("TensorView::as_matrix(matrix_index): matrix index {} out of range [0, {}).",
                        matrix_index, matrix_count));

        return MatrixMap(reinterpret_cast<float*>(data) + matrix_index * matrix_element_count,
                         row_count,
                         column_count);
    }

    MatrixMap as_flat_matrix() const
    {
        throw_if(shape.rank < 1, "TensorView::as_flat_matrix requires rank >= 1.");
        throw_if(shape.size() > 0 && !data, "TensorView::as_flat_matrix requires non-null data.");

        const Index column_count = shape[shape.rank - 1];
        const Index row_count = column_count == 0 ? 0 : shape.size() / column_count;
        return MatrixMap(reinterpret_cast<float*>(data), row_count, column_count);
    }

    VectorMap as_vector() const
    {
        throw_if(shape.size() > 0 && !data, "TensorView::as_vector requires non-null data.");
        return VectorMap(reinterpret_cast<float*>(data), shape.size());
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor() const
    {
        throw_if(shape.rank != Rank,
                 format("TensorView::as_tensor requires rank {}, got {}.", Rank, shape.rank));
        throw_if(shape.size() > 0 && !data, "TensorView::as_tensor requires non-null data.");

        Eigen::array<Index, Rank> dims;
        copy_n(shape.dims, Rank, dims.begin());
        return TensorMapR<Rank>(reinterpret_cast<float*>(data), dims);
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor(Index batch_index) const
    {
        throw_if(shape.rank != Rank + 1,
                 format("TensorView::as_tensor(batch_index) requires rank {}, got {}.",
                        Rank + 1, shape.rank));
        throw_if(batch_index < 0 || batch_index >= shape[0],
                 format("TensorView::as_tensor(batch_index): batch index {} out of range [0, {}).",
                        batch_index, shape[0]));
        throw_if(shape.size() > 0 && !data, "TensorView::as_tensor(batch_index) requires non-null data.");

        Eigen::array<Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) dims[i] = shape[i + 1];
        const Index slice_element_count = shape.size() / shape[0];
        return TensorMapR<Rank>(reinterpret_cast<float*>(data) + batch_index * slice_element_count, dims);
    }

    void fill(float);
    void setZero() { fill(0.0f); }
    void set_zero_async() const;

    mutable shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    cudnnTensorDescriptor_t get_descriptor() const;

private:
    void set_descriptor(const Shape&) const;

};

inline TensorView& slot_or(vector<TensorView>& views, const vector<size_t>& slot_indices, size_t i)
{
    static TensorView empty;
    return i < slot_indices.size() ? views[slot_indices[i]] : empty;
}

template<typename T, size_t N>
using array = Eigen::array<T, N>;

string shape_to_string(const Shape&, const string& = " ");
Shape string_to_shape(const string&, const string& = " ");

inline bool is_contiguous(const vector<Index>& indices)
{
    return ranges::adjacent_find(indices,
        [](Index a, Index b) { return b != a + 1; }) == indices.end();
}

void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&, float*, int contiguous = -1);

template<typename... Vs>
size_t hash_combine(const Vs&... values)
{
    size_t h = 0;
    ((h ^= hash<Vs>{}(values) + 0x9e3779b9 + (h << 6) + (h >> 2)), ...);
    return h;
}

inline void TensorView::set_zero_async() const
{
    if (!data || byte_size() == 0) return;
    device::set_zero_async(data, byte_size(), Backend::get_compute_stream());
}

inline const float one = 1.0f;
inline const float zero = 0.0f;

void copy_device_to_host_float(const void*, Type,
                               Index, float*,
                               cudaStream_t stream);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
