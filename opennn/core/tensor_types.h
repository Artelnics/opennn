//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   T Y P E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"

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

template<> struct TypeInfo<Type::INT8>
{
    using type = int8_t;
    static constexpr cudnnDataType_t cudnn = CUDNN_DATA_INT8;
    static constexpr cudaDataType_t  cuda  = CUDA_R_8I;
    static constexpr Index           bytes = Index(sizeof(int8_t));
    static constexpr const char*     name  = "INT8";
};

template<Type... Supported, typename F>
void visit_type(Type t, F&& f)
{
    bool matched = false;
    ([&]
    {
        if (!matched && t == Supported)
        {
            f.template operator()<typename TypeInfo<Supported>::type>();
            matched = true;
        }
    }(), ...);
    throw_if(!matched, "visit_type: unsupported Type value");
}

template<Type... Supported, typename F>
void visit_type_pair(Type t_in, Type t_out, F&& f)
{
    visit_type<Supported...>(t_in, [&]<typename TIn>()
    {
        visit_type<Supported...>(t_out, [&]<typename TOut>()
        {
            f.template operator()<TIn, TOut>();
        });
    });
}

template<typename F>
inline auto with_type_info(Type type, const char* caller, F&& f)
{
    switch (type)
    {
    case Type::FP32: return f(TypeInfo<Type::FP32>{});
    case Type::BF16: return f(TypeInfo<Type::BF16>{});
    case Type::INT8: return f(TypeInfo<Type::INT8>{});
    case Type::Auto: break;
    }

    throw runtime_error(string(caller) + ": Type::Auto must be resolved before tensor use.");
}

inline cudnnDataType_t to_cudnn(Type type)
{
    return with_type_info(type, "to_cudnn", [](auto info) { return info.cudnn; });
}

inline cudaDataType_t to_cuda(Type type)
{
    return with_type_info(type, "to_cuda", [](auto info) { return info.cuda; });
}

inline Index type_bytes(Type type)
{
    return with_type_info(type, "type_bytes", [](auto info) { return info.bytes; });
}

static constexpr Index ALIGN_BYTES = EIGEN_MAX_ALIGN_BYTES;
static constexpr Index ALIGN_ELEMENTS = ALIGN_BYTES / sizeof(float);

inline int to_int(Index value)
{
    throw_if(value > Index(numeric_limits<int>::max()) || value < Index(numeric_limits<int>::min()),
             "to_int: value {} exceeds int range.", value);
    return static_cast<int>(value);
}
inline float to_type(Index value) { return static_cast<float>(value); }

namespace detail
{

inline Index checked_index_add(Index left, Index right, string_view operation)
{
    throw_if(left < 0 || right < 0, "{}: values cannot be negative.", operation);
    throw_if(left > numeric_limits<Index>::max() - right,
             "{}: Index addition overflow.", operation);
    return left + right;
}

inline Index checked_index_multiply(Index left, Index right, string_view operation)
{
    throw_if(left < 0 || right < 0, "{}: values cannot be negative.", operation);
    if (left == 0 || right == 0) return 0;
    throw_if(left > numeric_limits<Index>::max() / right,
             "{}: Index multiplication overflow.", operation);
    return left * right;
}

template<typename Range, typename Projection>
Index checked_index_sum(const Range& values, Projection projection, string_view operation)
{
    Index total = 0;
    for (const auto& value : values)
        total = checked_index_add(total, projection(value), operation);
    return total;
}

}

inline Index align_up(Index value, Index alignment)
{
    throw_if(alignment <= 0 || (alignment & (alignment - 1)) != 0,
             "align_up: alignment must be a positive power of two.");
    if (value == 0) return 0;
    return detail::checked_index_add(value, alignment - 1, "align_up") & ~(alignment - 1);
}

inline Index ceil_div(Index value, Index divisor)
{
    throw_if(value < 0 || divisor <= 0,
             "ceil_div: value must be non-negative and divisor must be positive.");
    return value / divisor + Index(value % divisor != 0);
}

inline Index get_aligned_size(Index size)     { return align_up(size,    ALIGN_ELEMENTS); }
inline Index get_aligned_bytes(Index n_bytes) { return align_up(n_bytes, ALIGN_BYTES); }
inline Index get_aligned_bytes(Index count, Type dtype)
{
    return get_aligned_bytes(detail::checked_index_multiply(count, type_bytes(dtype),
                                                            "get_aligned_bytes"));
}

inline bool is_aligned(const void* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % ALIGN_BYTES == 0;
}

constexpr cublasComputeType_t CUBLAS_COMPUTE_DTYPE   = CUBLAS_COMPUTE_32F_FAST_TF32;

struct Shape
{
    static constexpr size_t MaxRank = 4;

    Shape() noexcept = default;

    static Shape filled(size_t rank, Index value)
    {
        throw_if(rank > MaxRank, "Shape::filled: rank {} exceeds MaxRank={}.", rank, MaxRank);
        throw_if(value < 0, "Shape::filled: dimensions cannot be negative.");

        Shape shape;
        shape.rank = rank;
        fill_n(shape.dims, rank, value);

        return shape;
    }

    Shape(initializer_list<Index> list) : rank(list.size())
    {
        throw_if(list.size() > MaxRank,
                 "Shape: initializer rank {} exceeds MaxRank={}.",
                        list.size(), MaxRank);
        throw_if(ranges::any_of(list, [](Index value) { return value < 0; }),
                 "Shape: dimensions cannot be negative.");
        copy_n(list.begin(), rank, dims);
    }

    template<typename It>
    Shape(It first, It last) : rank(static_cast<size_t>(distance(first, last)))
    {
        throw_if(rank > MaxRank,
                 "Shape: iterator-pair rank {} exceeds MaxRank={}.",
                        rank, MaxRank);
        throw_if(any_of(first, last, [](Index value) { return value < 0; }),
                 "Shape: dimensions cannot be negative.");
        copy_n(first, rank, dims);
    }

    size_t get_rank() const noexcept { return rank; }

    const Index* begin() const noexcept { return dims; }
    const Index* end() const noexcept { return dims + rank; }
    Index operator[](size_t i) const noexcept { return dims[i]; }
    Index back() const { throw_if(rank == 0, "Shape::back() on empty"); return dims[rank - 1]; }

    bool empty() const noexcept { return rank == 0; }

    Index dim_or_zero(size_t i) const noexcept { return i < rank ? dims[i] : Index(0); }

    Index size() const
    {
        throw_if(rank > MaxRank, "Shape::size: rank exceeds MaxRank={}.", MaxRank);
        if (rank == 0) return 0;

        Index element_count = 1;
        for (const Index dimension : *this)
            element_count = detail::checked_index_multiply(element_count, dimension,
                                                           "Shape::size");
        return element_count;
    }

    void clear() noexcept { rank = 0; }

    void set_dimension(size_t index, Index value)
    {
        throw_if(index >= rank,
                 "Shape::set_dimension: index {} is out of range for rank {}.",
                 index, rank);
        throw_if(value < 0, "Shape::set_dimension: dimensions cannot be negative.");
        dims[index] = value;
    }

    void push_back(Index value)
    {
        throw_if(rank >= MaxRank, "Shape::push_back: rank exceeds MaxRank={}.", MaxRank);
        throw_if(value < 0, "Shape::push_back: dimensions cannot be negative.");
        dims[rank++] = value;
    }

    friend ostream& operator<<(ostream& os, const Shape& shape)
    {
        os << "[";
        for (size_t i = 0; i < shape.get_rank(); ++i) os << (i ? ", " : " ") << shape.dims[i];
        os << " ]";
        return os;
    }

    bool operator==(const Shape& other) const noexcept
    {
        return rank == other.get_rank() && equal(begin(), end(), other.begin());
    }

    Shape& append(const Shape& other)
    {
        throw_if(rank > MaxRank || other.get_rank() > MaxRank - rank,
                 "Shape::append: combined rank exceeds MaxRank={}.", MaxRank);
        throw_if(any_of(other.begin(), other.end(), [](Index value) { return value < 0; }),
                 "Shape::append: dimensions cannot be negative.");
        copy_n(other.dims, other.get_rank(), dims + rank);
        rank += other.get_rank();
        return *this;
    }

private:
    Index dims[MaxRank] = {0};
    size_t rank = 0;
};

struct TensorSpec
{
    Shape shape;
    Type  dtype = Type::FP32;

    bool operator==(const TensorSpec&) const noexcept = default;
};

inline Index get_aligned_size(const vector<TensorSpec>& specs)
{
    return detail::checked_index_sum(specs,
        [](const auto& spec) { return get_aligned_size(spec.shape.size()); },
        "get_aligned_size(TensorSpec list)");
}

inline Index get_aligned_size(const vector<vector<TensorSpec>>& specs)
{
    return detail::checked_index_sum(specs,
        [](const auto& group) { return get_aligned_size(group); },
        "get_aligned_size(TensorSpec groups)");
}

inline Index get_aligned_bytes(const TensorSpec& spec) { return get_aligned_bytes(spec.shape.size(), spec.dtype); }

inline Index get_aligned_bytes(const vector<TensorSpec>& specs)
{
    return detail::checked_index_sum(specs,
        [](const TensorSpec& spec) { return get_aligned_bytes(spec); },
        "get_aligned_bytes(TensorSpec list)");
}

inline Index get_aligned_bytes(const vector<vector<TensorSpec>>& specs)
{
    return detail::checked_index_sum(specs,
        [](const auto& group) { return get_aligned_bytes(group); },
        "get_aligned_bytes(TensorSpec groups)");
}

inline Index get_aligned_bytes(const vector<Shape>& shapes, Type dtype)
{
    return detail::checked_index_sum(shapes,
        [dtype](const Shape& shape) { return get_aligned_bytes(shape.size(), dtype); },
        "get_aligned_bytes(Shape list)");
}

inline Index get_aligned_bytes(const vector<TensorSpec>& specs, Type dtype)
{
    return detail::checked_index_sum(specs,
        [dtype](const auto& spec) { return get_aligned_bytes(spec.shape.size(), dtype); },
        "get_aligned_bytes(TensorSpec list, dtype)");
}

inline Index get_aligned_bytes(const vector<vector<TensorSpec>>& specs, Type dtype)
{
    return detail::checked_index_sum(specs,
        [dtype](const auto& group) { return get_aligned_bytes(group, dtype); },
        "get_aligned_bytes(TensorSpec groups, dtype)");
}

struct Buffer
{
    explicit Buffer(Device initial_device = Device::CPU) : allocation_device(initial_device)
    {
        validate_device(initial_device, "Buffer");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept : Buffer() { swap(other); }
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this == &other) return *this;

        free_buffer();
        pointer = other.pointer;
        allocated_bytes = other.allocated_bytes;
        allocation_device = other.allocation_device;
        owns_allocation = other.owns_allocation;

        other.pointer = nullptr;
        other.allocated_bytes = 0;
        other.allocation_device = Device::CPU;
        other.owns_allocation = true;

        return *this;
    }

    ~Buffer() { free_buffer(); }

    template<typename T>
    T* as()
    {
        validate_state("Buffer::as");
        return static_cast<T*>(pointer);
    }

    template<typename T>
    const T* as() const
    {
        validate_state("Buffer::as");
        return static_cast<const T*>(pointer);
    }

    void* data() const noexcept { return pointer; }
    Index byte_size() const noexcept { return allocated_bytes; }
    Device get_device() const noexcept { return allocation_device; }
    bool owns_memory() const noexcept { return owns_allocation; }
    Index size_in_floats() const noexcept { return allocated_bytes / Index(sizeof(float)); }
    bool  empty() const noexcept { return allocated_bytes == 0; }

    VectorMap as_vector() &
    {
        validate_state("Buffer::as_vector");
        throw_if(allocation_device != Device::CPU,
                 "Buffer::as_vector requires host storage.");
        throw_if(allocated_bytes % Index(sizeof(float)) != 0,
                 "Buffer::as_vector requires a whole number of floats.");
        return VectorMap(static_cast<float*>(pointer), size_in_floats());
    }

    ConstVectorMap as_vector() const &
    {
        validate_state("Buffer::as_vector");
        throw_if(allocation_device != Device::CPU,
                 "Buffer::as_vector requires host storage.");
        throw_if(allocated_bytes % Index(sizeof(float)) != 0,
                 "Buffer::as_vector requires a whole number of floats.");
        return ConstVectorMap(static_cast<const float*>(pointer), size_in_floats());
    }

    void resize_bytes(Index byte_count, Device new_device)
    {
        validate_state("Buffer::resize_bytes");
        validate_size_and_device(byte_count, new_device, "Buffer::resize_bytes");
        if (owns_allocation
            && byte_count == allocated_bytes
            && allocation_device == new_device) return;

        const bool changes_cuda_allocation =
            (allocation_device == Device::CUDA && pointer)
            || (new_device == Device::CUDA && byte_count > 0);
        throw_if(changes_cuda_allocation && device::cuda_allocation_growth_forbidden(),
                 "CUDA buffer resize from {} to {} bytes while CUDA allocation growth is forbidden "
                        "(warmup incomplete before CUDA graph capture).",
                        allocated_bytes,
                        byte_count);

        Buffer replacement(new_device);
        if (byte_count > 0)
        {
            replacement.pointer = device::allocate(new_device, byte_count);
            replacement.allocated_bytes = byte_count;
        }

        swap(replacement);
    }

    void set_view(void* external_data, Index byte_count, Device view_device)
    {
        validate_state("Buffer::set_view");
        validate_size_and_device(byte_count, view_device, "Buffer::set_view");
        throw_if((external_data == nullptr) != (byte_count == 0),
                 "Buffer::set_view requires null data exactly when the view is empty.");
        throw_if(owns_allocation && pointer && external_data == pointer,
                 "Buffer::set_view cannot alias the buffer's owned allocation.");

        Buffer replacement(view_device);
        replacement.pointer = external_data;
        replacement.allocated_bytes = byte_count;
        replacement.owns_allocation = false;
        swap(replacement);
    }

    void grow_to(Index minimum_bytes)
    {
        validate_state("Buffer::grow_to");
        throw_if(minimum_bytes < 0, "Buffer::grow_to size cannot be negative.");
        if (minimum_bytes > allocated_bytes)
            resize_bytes(minimum_bytes, allocation_device);
    }

    template<typename T>
    T* ensure(Index element_count)
    {
        grow_to(detail::checked_index_multiply(element_count, Index(sizeof(T)),
                                               "Buffer::ensure"));
        return as<T>();
    }

    void setZero()
    {
        validate_state("Buffer::setZero");
        device::set_zero(pointer, allocated_bytes, allocation_device);
    }

    void migrate_to(Device target_device, cudaStream_t stream = nullptr)
    {
        validate_state("Buffer::migrate_to");
        validate_device(target_device, "Buffer::migrate_to");
        if (allocation_device == target_device) return;

        Buffer target_buffer(target_device);
        if (!pointer)
            return swap(target_buffer);

        target_buffer.resize_bytes(allocated_bytes, target_device);
        device::copy_async(target_buffer.pointer, pointer, allocated_bytes,
                           allocation_device, target_device, stream);
        if (stream) device::synchronize(stream);

        swap(target_buffer);
    }

    void swap(Buffer& other) noexcept
    {
        std::swap(pointer, other.pointer);
        std::swap(allocated_bytes, other.allocated_bytes);
        std::swap(allocation_device, other.allocation_device);
        std::swap(owns_allocation, other.owns_allocation);
    }

private:

    static void validate_device(Device device, 
                                string_view operation)
    {
        throw_if(device == Device::Auto, "{} requires a concrete device.", operation);
    }

    static void validate_size_and_device(Index byte_count,
                                         Device device,
                                         string_view operation)
    {
        throw_if(byte_count < 0, "{} size cannot be negative.", operation);
        validate_device(device, operation);
    }

    void validate_state(string_view operation) const
    {
        validate_size_and_device(allocated_bytes, allocation_device, operation);
        throw_if((pointer == nullptr) != (allocated_bytes == 0),
                 "{} found inconsistent data and byte count.", operation);
    }

    void free_buffer()
    {
        if (pointer && owns_allocation)
            device::deallocate(allocation_device, pointer, allocated_bytes);
        pointer = nullptr;
        allocated_bytes = 0;
        owns_allocation = true;
    }

    void* pointer = nullptr;
    Index allocated_bytes = 0;
    Device allocation_device = Device::CPU;
    bool owns_allocation = true;
};

struct TensorView
{
    TensorView(void* new_data = nullptr,
               const Shape& new_shape = {},
               Type new_type = Type::FP32,
               Device new_device = Device::CPU) noexcept
        : data(new_data),
          shape(new_shape),
          type(new_type),
          device(new_device) {}

    void* get_data() const noexcept { return data; }
    const Shape& get_shape() const noexcept { return shape; }
    Type get_type() const noexcept { return type; }
    Device get_device() const noexcept { return device; }
    Index get_rank() const noexcept { return Index(shape.get_rank()); }

    Index size() const { return shape.size(); }

    Index byte_size() const
    {
        return detail::checked_index_multiply(size(), type_bytes(type),
                                              "TensorView::byte_size");
    }

    Index flat_columns() const noexcept { return shape.get_rank() == 0 ? 0 : shape[shape.get_rank() - 1]; }
    Index flat_rows() const noexcept
    {
        const Index columns = flat_columns();
        return columns == 0 ? 0 : shape.size() / columns;
    }

    bool empty() const noexcept { return shape.empty(); }
    bool is_cuda() const noexcept { return device == Device::CUDA; }
    bool is_fp32() const noexcept { return type == Type::FP32; }
    bool is_bf16() const noexcept { return type == Type::BF16; }
    bool is_int8() const noexcept { return type == Type::INT8; }

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
        visit_type<Type::FP32, Type::BF16>(type, fn);
    }

    TensorView reshape(const Shape& new_shape) const
    {
        const Index current_size = size();
        throw_if(new_shape.size() != current_size,
                 "TensorView::reshape cannot change the element count from {} to {}.",
                 current_size, new_shape.size());
        return TensorView(data, new_shape, type, device);
    }

    TensorView reshape_prefix(const Shape& new_shape) const
    {
        const Index current_size = size();
        throw_if(new_shape.size() > current_size,
                 "TensorView::reshape_prefix cannot grow the element count from {} to {}.",
                 current_size, new_shape.size());
        return TensorView(data, new_shape, type, device);
    }

    MatrixMap as_matrix() const
    {
        require_host_fp32_data("TensorView::as_matrix");
        throw_if(shape.get_rank() < 2, "TensorView::as_matrix requires rank >= 2.");

        const Index row_count = shape[0];
        const Index column_count = row_count == 0 ? 0 : shape.size() / row_count;
        return MatrixMap(reinterpret_cast<float*>(data), row_count, column_count);
    }

    MatrixMap as_matrix(Index matrix_index) const
    {
        require_host_fp32_data("TensorView::as_matrix(matrix_index)");
        throw_if(shape.get_rank() < 2,
                 "TensorView::as_matrix(matrix_index) requires rank >= 2.");

        const Index row_count = shape[shape.get_rank() - 2];
        const Index column_count = shape[shape.get_rank() - 1];
        const Index matrix_element_count = row_count * column_count;
        const Index matrix_count = matrix_element_count == 0
            ? 0
            : shape.size() / matrix_element_count;

        throw_if(matrix_index < 0 || matrix_index >= matrix_count,
                 "TensorView::as_matrix(matrix_index): matrix index {} out of range [0, {}).",
                        matrix_index, matrix_count);

        return MatrixMap(reinterpret_cast<float*>(data) + matrix_index * matrix_element_count,
                         row_count,
                         column_count);
    }

    MatrixMap as_flat_matrix() const
    {
        require_host_fp32_data("TensorView::as_flat_matrix");
        throw_if(shape.get_rank() < 1, "TensorView::as_flat_matrix requires rank >= 1.");

        return MatrixMap(reinterpret_cast<float*>(data), flat_rows(), flat_columns());
    }

    VectorMap as_vector() const
    {
        require_host_fp32_data("TensorView::as_vector");
        return VectorMap(reinterpret_cast<float*>(data), shape.size());
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor() const
    {
        require_host_fp32_data("TensorView::as_tensor");
        throw_if(shape.get_rank() != Rank,
                 "TensorView::as_tensor requires rank {}, got {}.", Rank, shape.get_rank());

        Eigen::array<Index, Rank> dims;
        copy_n(shape.begin(), Rank, dims.begin());
        return TensorMapR<Rank>(reinterpret_cast<float*>(data), dims);
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor(Index batch_index) const
    {
        require_host_fp32_data("TensorView::as_tensor(batch_index)");
        throw_if(shape.get_rank() != Rank + 1,
                 "TensorView::as_tensor(batch_index) requires rank {}, got {}.",
                        Rank + 1, shape.get_rank());
        throw_if(batch_index < 0 || batch_index >= shape[0],
                 "TensorView::as_tensor(batch_index): batch index {} out of range [0, {}).",
                        batch_index, shape[0]);

        Eigen::array<Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) dims[i] = shape[i + 1];
        const Index slice_element_count = shape.size() / shape[0];
        return TensorMapR<Rank>(reinterpret_cast<float*>(data)
                               + batch_index * slice_element_count,
                               dims);
    }

    void fill(float) const;
    void setZero() const { fill(0.0f); }
    void set_zero_async() const;

    CudnnDescriptor<cudnnTensorDescriptor_t> get_descriptor() const;

private:
    void require_host_fp32(string_view accessor) const
    {
        throw_if(device != Device::CPU || type != Type::FP32,
                 "{} requires CPU FP32 storage.", accessor);
    }

    void require_host_fp32_data(string_view accessor) const
    {
        require_host_fp32(accessor);
        throw_if(shape.size() > 0 && !data, "{} requires non-null data.", accessor);
    }

    void* data = nullptr;
    Shape shape;
    Type type = Type::FP32;
    Device device = Device::CPU;
};

inline TensorView& slot_or(vector<TensorView>& views,
                           const vector<size_t>& slot_indices,
                           size_t i,
                           TensorView& fallback)
{
    if (i >= slot_indices.size()) return fallback;

    throw_if(slot_indices[i] >= views.size(),
             "slot_or: slot index {} is out of range for {} views.",
             slot_indices[i], views.size());
    return views[slot_indices[i]];
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

enum class ColumnContiguity { Unknown, NonContiguous, Contiguous };

inline ColumnContiguity classify_column_contiguity(const vector<Index>& indices)
{
    return is_contiguous(indices) ? ColumnContiguity::Contiguous
                                  : ColumnContiguity::NonContiguous;
}

inline bool resolve_column_contiguity(ColumnContiguity column_contiguity,
                                      const vector<Index>& indices)
{
    return column_contiguity == ColumnContiguity::Unknown
        ? is_contiguous(indices)
        : column_contiguity == ColumnContiguity::Contiguous;
}

void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&,
                      span<float>, ColumnContiguity = ColumnContiguity::Unknown);

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
    opennn::device::set_zero_async(data, byte_size(), device::get_compute_stream());
}

inline const float one = 1.0f;
inline const float zero = 0.0f;

void copy_device_to_host_float(const void*, Type,
                               Index, float*,
                               cudaStream_t stream);
void copy_device_to_host_float(const void*, Type,
                               Index, float*,
                               cudaStream_t stream,
                               vector<uint16_t>& bf16_staging);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
