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
static constexpr Index ALIGN_MASK = ~(ALIGN_ELEMENTS - 1);

inline Index get_aligned_size(Index size)
{
    if (size == 0) return 0;
    return (size + ALIGN_ELEMENTS - 1) & ALIGN_MASK;
}

inline Index get_aligned_bytes(Index n_bytes)
{
    if (n_bytes == 0) return 0;
    return (n_bytes + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1);
}

template<typename Container>
inline Index ssize(const Container& c) noexcept
{
    return static_cast<Index>(c.size());
}

inline bool is_aligned(const void* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % ALIGN_BYTES == 0;
}

template<typename Base, typename T>
inline bool is_instance_of(const T* ptr)
{
    return dynamic_cast<const Base*>(ptr);
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

constexpr cudnnDataType_t     CUDNN_WEIGHT_DTYPE     = CUDNN_DATA_FLOAT;
constexpr cudnnDataType_t     CUDNN_ACTIVATION_DTYPE = CUDNN_DATA_FLOAT;
constexpr cudaDataType_t      CUDA_WEIGHT_DTYPE      = CUDA_R_32F;
constexpr cudaDataType_t      CUDA_ACTIVATION_DTYPE  = CUDA_R_32F;
constexpr cudaDataType_t      CUDA_REDUCTION_DTYPE   = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE_DTYPE   = CUBLAS_COMPUTE_32F;

inline Index dtype_bytes(cudnnDataType_t t)
{
    switch (t) {
        case CUDNN_DATA_FLOAT:    return 4;
        case CUDNN_DATA_BFLOAT16: return 2;
        case CUDNN_DATA_HALF:     return 2;
        default:                  return 4;
    }
}

inline cudaDataType_t cudnn_to_cuda_dtype(cudnnDataType_t t)
{
    switch (t) {
        case CUDNN_DATA_FLOAT:    return CUDA_R_32F;
        case CUDNN_DATA_BFLOAT16: return CUDA_R_16BF;
        case CUDNN_DATA_HALF:     return CUDA_R_16F;
        default:                  return CUDA_R_32F;
    }
}

template<cudnnDataType_t D> struct MapDtype;
template<> struct MapDtype<CUDNN_DATA_FLOAT>    { using type = float; };
template<> struct MapDtype<CUDNN_DATA_BFLOAT16> { using type = __nv_bfloat16; };

template <typename T>
class ThreadSafeQueue
{
private:

    queue<T> queue_;
    mutex mutex_;
    condition_variable cond_;

public:

    void push(T item)
    {
        {
            lock_guard<mutex> lock(mutex_);
            queue_.push(move(item));
        }
        cond_.notify_one();
    }

    T pop()
    {
        unique_lock<mutex> lock(mutex_);
        cond_.wait(lock, [this]() { return !queue_.empty(); });
        T item = queue_.front();
        queue_.pop();
        return item;
    }

    bool empty()
    {
        lock_guard<mutex> lock(mutex_);
        return queue_.empty();
    }
};

class Shape
{
public:

    static constexpr size_t MaxRank = 4;

    Shape() noexcept = default;

    Shape(size_t n, Index value)
    {
        rank_ = (n > MaxRank) ? MaxRank : n;

        for (size_t i = 0; i < rank_; ++i)
            shape_[i] = value;
    }

    Shape(initializer_list<Index> list)
    {
        rank_ = min(list.size(), MaxRank);
        size_t i = 0;
        for (Index d : list)
            if (i < rank_)
                shape_[i++] = d;
    }

    size_t rank() const noexcept
    {
        return rank_;
    }

    const Index* begin() const noexcept
    {
        return shape_;
    }

    const Index* end() const noexcept
    {
        return shape_ + rank_;
    }

    const Index& operator[](size_t i) const noexcept
    {
        return shape_[i];
    }

    Index& operator[](size_t i) noexcept
    {
        return shape_[i];
    }

    Index& back()
    {
        if(rank_ == 0) throw runtime_error("Shape::back() on empty shape");
        return shape_[rank_ - 1];
    }

    const Index& back() const
    {
        if(rank_ == 0) throw runtime_error("Shape::back() on empty shape");
        return shape_[rank_ - 1];
    }

    bool empty() const noexcept
    {
        return rank_ == 0;
    }

    Index size() const noexcept
    {
        if (rank_ == 0) return 0;

        Index total = shape_[0];

        for (size_t i = 1; i < rank_; ++i)
            total *= shape_[i];

        return total;
    }

    void clear() noexcept
    {
        rank_ = 0;
    }

    void push_back(Index value) noexcept
    {
        if (rank_ < MaxRank)
            shape_[rank_++] = value;
    }

    friend ostream& operator<<(ostream& os, const Shape& s)
    {
        os << "[ ";

        for (size_t i = 0; i < s.rank_; ++i)
            os << s.shape_[i] << (i < s.rank_ - 1 ? ", " : " ");

        os << "]";
        return os;
    }

    bool operator == (const Shape& other) const noexcept
    {
        if (rank_ != other.rank_) return false;

        for (size_t i = 0; i < rank_; ++i)
            if (shape_[i] != other.shape_[i]) return false;

        return true;
    }

    bool operator != (const Shape& other) const noexcept
    {
        return !(*this == other);
    }

    Shape& append(const Shape& other)
    {
        for(size_t i = 0; i < other.rank_ && rank_ < MaxRank; ++i)
            shape_[rank_++] = other.shape_[i];

        return *this;
    }

    template<int Rank>
    Eigen::array<Index, Rank> get_eigen_dims() const {
        if (Rank != rank_) 
            throw std::runtime_error("Shape Error: Requested Rank (" + std::to_string(Rank) +
                                     ") does not match Shape rank (" + std::to_string(rank_) + ").");
        
        Eigen::array<Index, Rank> dims;

        for (size_t i = 0; i < Rank; ++i) 
            dims[i] = shape_[i];
        
        return dims;
    }

private:

    Index shape_[MaxRank] = {0};
    size_t rank_ = 0;
};

struct Memory
{
    VectorR vector;

    type* data() { return vector.data(); }
    const type* data() const { return vector.data(); }
    Index size() const { return vector.size(); }
    bool empty() const { return vector.size() == 0; }

    void resize(Index n) { vector.resize(n); }
    void setZero() { vector.setZero(); }
    void setZero(Index n) { vector = VectorR::Zero(n); }

    // Device-aware: zeroes host or device storage depending on Device::is_gpu().
    // Defined out-of-class below because Device is declared later in this header.
    void setZero_active();

    uint8_t* device_data = nullptr;
    Index    allocated_bytes = 0;

    Memory() = default;
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    Memory(Memory&& other) noexcept
        : vector(std::move(other.vector))
        , device_data(other.device_data)
        , allocated_bytes(other.allocated_bytes)
    {
        other.device_data = nullptr;
        other.allocated_bytes = 0;
    }

    Memory& operator=(Memory&& other) noexcept
    {
        if(this != &other)
        {
            free_device();
            vector = std::move(other.vector);
            device_data = other.device_data;
            allocated_bytes = other.allocated_bytes;
            other.device_data = nullptr;
            other.allocated_bytes = 0;
        }
        return *this;
    }

    ~Memory() { free_device(); }

    type*          device()             { return reinterpret_cast<type*>(device_data); }
    const type*    device()       const { return reinterpret_cast<const type*>(device_data); }
    uint8_t*       device_bytes()       { return device_data; }
    const uint8_t* device_bytes() const { return device_data; }

#ifdef OPENNN_WITH_CUDA
    void resize_device(Index n_floats) { resize_device_bytes(n_floats * Index(sizeof(float))); }

    void resize_device_bytes(Index n_bytes)
    {
        if(n_bytes == allocated_bytes) return;
        free_device();
        allocated_bytes = n_bytes;
        if(n_bytes > 0) CHECK_CUDA(cudaMalloc(&device_data, n_bytes));
    }

    void setZero_device()
    {
        if(device_data && allocated_bytes > 0)
            CHECK_CUDA(cudaMemset(device_data, 0, allocated_bytes));
    }

private:
    void free_device()
    {
        if(device_data) { cudaFree(device_data); device_data = nullptr; allocated_bytes = 0; }
    }
#else
private:
    void free_device() {}
#endif
};

struct TensorView
{
    TensorView() noexcept = default;

    type* data = nullptr;

    Shape shape;

    cudnnDataType_t dtype = CUDNN_ACTIVATION_DTYPE;

    TensorView(type* new_data, const Shape& new_shape,
               cudnnDataType_t new_dtype = CUDNN_ACTIVATION_DTYPE) noexcept
        : data(new_data), shape(new_shape), dtype(new_dtype) {}

    Index get_rank() const noexcept { return shape.rank(); }

    Index size() const noexcept { return shape.size(); }

    Index byte_size() const noexcept { return size() * dtype_bytes(dtype); }

    bool empty() const noexcept { return shape.empty(); }

    template<typename T>       T* as()       noexcept;
    template<typename T> const T* as() const noexcept;

    // Float-typed view of `data`. Used at CUDA dispatch sites where kernels expect
    // raw float* regardless of the project's `type` alias (e.g. descriptive stats,
    // scaler tables, masks). Mirrors as<T>() but skips the dtype check.
    float*       as_float()       noexcept { return reinterpret_cast<float*>(data); }
    const float* as_float() const noexcept { return reinterpret_cast<const float*>(data); }

    cudaDataType_t cuda_dtype() const noexcept { return cudnn_to_cuda_dtype(dtype); }

    template<typename F>
    void dispatch(F&& fn) const
    {
        if (dtype == CUDNN_DATA_BFLOAT16) fn(__nv_bfloat16{});
        else                              fn(float{});
    }

    TensorView reshape(const Shape& new_shape) const
    { return TensorView(data, new_shape, dtype); }

    void print() const
    {
        if(!data || shape.empty())
        {
            cout << "TensorView: Empty or Null" << "\n";
            return;
        }

        cout << "Shape: " << shape << "\n";

        const Index total_size = size();
        const Index last_dim_stride = shape.back();

        for(Index i = 0; i < total_size; ++i)
        {
            cout << data[i] << " ";

            if (shape.rank() > 1 && (i + 1) % last_dim_stride == 0)
                cout << "\n";
        }

        if (shape.rank() == 1 || total_size % last_dim_stride != 0)
            cout << "\n";
    }

    MatrixMap as_matrix() const
    {
        assert(data && shape.rank() >= 2);
        return MatrixMap(data, shape[0], shape.size() / shape[0]);
    }

    MatrixMap as_matrix(Index batch_index) const
    {
        assert(data && shape.rank() >= 2);
        const Index rows = shape[shape.rank() - 2];
        const Index cols = shape[shape.rank() - 1];
        return MatrixMap(data + batch_index * rows * cols, rows, cols);
    }

    MatrixMap as_flat_matrix() const
    {
        assert(data && shape.rank() >= 1);
        const Index cols = shape[shape.rank() - 1];
        return MatrixMap(data, shape.size() / cols, cols);
    }

    MatrixMap as_flat_matrix(Index batch_index) const
    {
        assert(data && shape.rank() >= 2);
        const Index cols = shape[shape.rank() - 1];
        Index rows = 1;
        for (size_t i = 1; i + 1 < shape.rank(); ++i)
            rows *= shape[i];
        return MatrixMap(data + batch_index * rows * cols, rows, cols);
    }

    VectorMap as_vector() const
    {
        assert(data);
        return VectorMap(data, shape.size());
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor() const
    {
        assert(data && shape.rank() == Rank);
        return TensorMapR<Rank>(data, shape.template get_eigen_dims<Rank>());
    }

    template<int Rank>
    TensorMapR<Rank> as_tensor(Index batch_index) const
    {
        assert(data && shape.rank() == Rank + 1);
        Eigen::array<Index, Rank> dims;
        Index slice_size = 1;
        for(int i = 0; i < Rank; ++i)
        {
            dims[i] = shape[i + 1];
            slice_size *= shape[i + 1];
        }
        return TensorMapR<Rank>(data + batch_index * slice_size, dims);
    }

    void fill(float value);

#ifdef OPENNN_WITH_CUDA

    float* device = nullptr;

    mutable shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    TensorView(float* new_data, std::shared_ptr<cudnnTensorStruct> handle)
        : data(new_data), descriptor_handle(handle) {}

    explicit TensorView(const Shape& new_shape)
        : shape(new_shape)
    {
        set_descriptor(new_shape);
    }

    cudnnTensorDescriptor_t get_descriptor() const
    {
        if (!descriptor_handle && !shape.empty())
            set_descriptor(shape);
        return descriptor_handle ? descriptor_handle.get() : nullptr;
    }

    void set_descriptor(const Shape& desc_shape) const
    {
        int n = 1, c = 1, h = 1, w = 1;
        if (desc_shape.rank() == 4) {
            n = static_cast<int>(desc_shape[0]);
            h = static_cast<int>(desc_shape[1]);
            w = static_cast<int>(desc_shape[2]);
            c = static_cast<int>(desc_shape[3]);
        }
        else if (desc_shape.rank() == 3) {
            n = static_cast<int>(desc_shape[0]);
            w = static_cast<int>(desc_shape[1]);
            c = static_cast<int>(desc_shape[2]);
        }
        else if (desc_shape.rank() == 2) {
            n = static_cast<int>(desc_shape[0]);
            c = static_cast<int>(desc_shape[1]);
        }
        else if (desc_shape.rank() == 1) {
            c = static_cast<int>(desc_shape[0]);
        }

        if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
            return;

        if (!descriptor_handle)
        {
            cudnnTensorDescriptor_t raw_desc;
            if (cudnnCreateTensorDescriptor(&raw_desc) != CUDNN_STATUS_SUCCESS)
                throw runtime_error("TensorView: Failed to create descriptor.");

            descriptor_handle = std::shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t p) {
                if (p) cudnnDestroyTensorDescriptor(p);
            });
        }

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NHWC, dtype, n, c, h, w));
    }

    Index device_size() const
    {
        if (!descriptor_handle) return 0;

        constexpr int REQUESTED_DIMS = CUDNN_DIM_MAX;
        cudnnDataType_t dataType;
        int nbDims = 0, dimA[REQUESTED_DIMS], strideA[REQUESTED_DIMS];

        CHECK_CUDNN(cudnnGetTensorNdDescriptor(descriptor_handle.get(), REQUESTED_DIMS, &dataType, &nbDims, dimA, strideA));

        Index total_elements = 1;
        for (int i = 0; i < nbDims; ++i)
            total_elements *= static_cast<Index>(dimA[i]);
        return total_elements;
    }

#endif

};

template<> inline       float*         TensorView::as<float>()               noexcept { assert(dtype == CUDNN_DATA_FLOAT);    return reinterpret_cast<float*>(data); }
template<> inline const float*         TensorView::as<float>()         const noexcept { assert(dtype == CUDNN_DATA_FLOAT);    return reinterpret_cast<const float*>(data); }
template<> inline       __nv_bfloat16* TensorView::as<__nv_bfloat16>()       noexcept { assert(dtype == CUDNN_DATA_BFLOAT16); return reinterpret_cast<__nv_bfloat16*>(data); }
template<> inline const __nv_bfloat16* TensorView::as<__nv_bfloat16>() const noexcept { assert(dtype == CUDNN_DATA_BFLOAT16); return reinterpret_cast<const __nv_bfloat16*>(data); }

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

type* link(type*, const vector<TensorView*>&);
void link(type*, const vector<vector<TensorView*>>&);

inline Index get_size(const vector<Shape>& shapes)
{
    Index total = 0;

    for(const Shape& s : shapes)
    {
        const Index n = s.size();
        if(n > 0)
            total += (n + ALIGN_ELEMENTS - 1) & ALIGN_MASK;
    }

    return total;
}

inline Index get_size(const vector<vector<Shape>>& shapes)
{
    Index total = 0;

    for(const auto& layer_shapes : shapes)
        total += get_size(layer_shapes);

    return total;
}

Index get_size(const vector<TensorView*>&);
Index get_size(const vector<vector<TensorView*>>&);

template<typename T, size_t N>
using array = Eigen::array<T, N>;

template <typename Index>
Eigen::array<IndexPair<Index>, 1> axes(const Index a, Index b)
{
    return Eigen::array<IndexPair<Index>, 1>({IndexPair<Index>(a, b)});
}

template <typename Index>
Eigen::array<IndexPair<Index>, 2> axes(const Index a1, Index b1, Index a2, Index b2)
{
    return Eigen::array<IndexPair<Index>, 2>({IndexPair<Index>(a1, b1), IndexPair<Index>(a2, b2)});
}

inline Eigen::array<Index, 1> array_1(const Index a)
{
    return Eigen::array<Index, 1>({a});
}

inline Eigen::array<Index, 2> array_2(const Index a, Index b)
{
    return Eigen::array<Index, 2>({a, b});
}

inline Eigen::array<Index, 3> array_3(const Index a, Index b, Index c)
{
    return Eigen::array<Index, 3>({a, b, c});
}

inline Eigen::array<Index, 4> array_4(const Index a, Index b, Index c, Index d)
{
    return Eigen::array<Index, 4>({a, b, c, d});
}

inline bool is_contiguous(const vector<Index>& v)
{
    const type first = v[0];

    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] != first + Index(i))
            return false;

    return true;
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

template<int rank>
Index count_NAN(const TensorR<rank>& x)
{
    return count_if(x.data(), x.data() + x.size(), [](type value) {return std::isnan(value); });
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

Index count_greater_than(const vector<Index>&, Index);

VectorI calculate_rank(const VectorR&, bool ascending = true);

vector<Index> get_elements_greater_than(const vector<Index>&, Index);
vector<Index> get_elements_greater_than(const vector<vector<Index>>&, Index);

VectorI get_nearest_points(const MatrixR& ,const VectorR& , int = 1);

void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&, type*, bool = true, int contiguous = -1);

template <typename Type, int Rank>
bool contains(const TensorR<Rank>& tensor, const Type& value)
{
    return find(tensor.data(), tensor.data() + tensor.size(), value) != (tensor.data() + tensor.size());
}

template <typename Derived, typename T>
bool contains(const Eigen::MatrixBase<Derived>& vector, const T& value)
{
    const auto* begin = vector.derived().data();
    const auto* end = begin + vector.size();

    return find(begin, end, value) != end;
}


VectorR perform_Householder_QR_decomposition(const MatrixR&, const VectorR&);

template <typename T>
void push_back(Tensor<T, 1, AlignedMax>& tensor, const T& value)
{
    const Index old_size = tensor.dimension(0);

    Tensor<T, 1> new_tensor(old_size + 1);

    memcpy(new_tensor.data(), tensor.data(), old_size * sizeof(T));

    new_tensor(old_size) = value;

    tensor = new_tensor;
}

string shape_to_string(const Shape&, const string& = " ");
Shape string_to_shape(const string&, const string& = " ");

VectorMap vector_map(const MatrixR&, Index);

MatrixMap matrix_map(const Tensor3&, Index);
TensorMap3 tensor_map(const Tensor4&, Index);
MatrixMap matrix_map(const Tensor4&, Index, Index);

inline VectorMap vector_map(const TensorView& tensor_view)
{
    assert(tensor_view.data);
    assert(reinterpret_cast<uintptr_t>(tensor_view.data) % EIGEN_MAX_ALIGN_BYTES == 0);

    return VectorMap(tensor_view.data, tensor_view.size());
}

inline MatrixMap matrix_map(const TensorView& tensor_view)
{
    assert(tensor_view.data);
    assert(reinterpret_cast<uintptr_t>(tensor_view.data) % EIGEN_MAX_ALIGN_BYTES == 0);

    return MatrixMap(tensor_view.data, tensor_view.shape[0], tensor_view.size() / tensor_view.shape[0]);
}

template <Index rank>
TensorMapR<rank> tensor_map(const TensorView& tensor_view)
{
    assert(tensor_view.data);
    assert(reinterpret_cast<uintptr_t>(tensor_view.data) % EIGEN_MAX_ALIGN_BYTES == 0);

    if (tensor_view.get_rank() != rank)
        throw runtime_error("Dimensions is " + to_string(tensor_view.get_rank()) + " and must be " + to_string(rank));

    return TensorMapR<rank>(tensor_view.data, tensor_view.shape.get_eigen_dims<rank>());
}

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

template<class T, int n>
VectorI get_shape(const Tensor<T, n, AlignedMax>& tensor)
{
    return Eigen::Map<const VectorI>(tensor.dimensions().data(), n);
}

template <typename T, typename Value>
inline bool is_equal(const T& tensor,
                     const Value& value,
                     const Value& tolerance = Value(1.0e-3))
{
    const auto* data = tensor.data();
    const Index size = tensor.size();

    if constexpr (is_same_v<Value, bool>)
    {
        for(Index i = 0; i < size; ++i)
            if (data[i] != value)
                return false;
    }
    else
    {
        for(Index i = 0; i < size; ++i)
            if(std::abs(data[i] - value) > tolerance)
                return false;
    }

    return true;
}

template <typename T, typename U>
inline bool are_equal(const T& A, const U& B, type tolerance = type(1.0e-3))
{
    if(A.size() != B.size())
        throw runtime_error("are_equal: sizes are different.");

    const type* a = A.data();
    const type* b = B.data();

    for(Index i = 0; i < A.size(); ++i)
        if(abs(a[i] - b[i]) > tolerance)
            return false;

    return true;
}

enum class DeviceType { Cpu, Gpu };

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
        if(d_desc)  cublasLtMatrixLayoutDestroy(d_desc);
        if(c_desc)  cublasLtMatrixLayoutDestroy(c_desc);
        if(b_desc)  cublasLtMatrixLayoutDestroy(b_desc);
        if(a_desc)  cublasLtMatrixLayoutDestroy(a_desc);
        if(op_desc) cublasLtMatmulDescDestroy(op_desc);
    }
};

struct LtMatmulPlanKey
{
    int m;
    int n;
    int k;
    int transA;
    int transB;

    bool operator==(const LtMatmulPlanKey& o) const noexcept
    {
        return m == o.m && n == o.n && k == o.k && transA == o.transA && transB == o.transB;
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
    DeviceType get_type() const { return device_type; }
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

    // Returns a cached cuBLASLt plan for a GEMM with bias-add epilogue. The plan
    // is built (descriptors + heuristic algo) once per unique shape and reused
    // thereafter. Per-call bias pointer is set on the returned op_desc by the
    // caller — it is intentionally not part of the cache key.
    static const LtMatmulPlan& get_lt_gemm_bias_plan(
        int m, int n, int k,
        cublasOperation_t transA,
        cublasOperation_t transB);
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

#ifdef OPENNN_WITH_CUDA
    std::unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_gemm_bias_plans;
#endif
};

inline ThreadPoolDevice& get_device()
{
    return *Device::instance().get_thread_pool_device();
}

inline void Memory::setZero_active()
{
#ifdef OPENNN_WITH_CUDA
    if(Device::instance().is_gpu()) { setZero_device(); return; }
#endif
    vector.setZero();
}

inline void TensorView::fill(float value)
{
    if(!data) return;

#ifdef OPENNN_WITH_CUDA
    if(Device::instance().is_gpu())
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
    std::fill(data, data + size(), value);
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
    CHECK_CUBLAS(cublasGemmEx(Device::get_cublas_handle(),
                              transa, transb,
                              m, n, k,
                              &alpha,
                              A, Atype, lda,
                              B, Btype, ldb,
                              &beta,
                              C, Ctype, ldc,
                              CUBLAS_COMPUTE_DTYPE,
                              CUBLAS_GEMM_DEFAULT));
}

inline void gemv_cuda(cublasOperation_t transa,
                      int m, int n,
                      const void* A, cudaDataType_t Atype, int lda,
                      const void* x, cudaDataType_t Xtype,
                      void* y, cudaDataType_t Ytype,
                      float alpha = 1.0f, float beta = 0.0f)
{
    const int m_out = (transa == CUBLAS_OP_N) ? m : n;
    const int k_dim = (transa == CUBLAS_OP_N) ? n : m;

    gemm_cuda(transa, CUBLAS_OP_N,
              m_out, 1, k_dim,
              A, Atype, lda,
              x, Xtype, k_dim,
              y, Ytype, m_out,
              alpha, beta);
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

// Fused GEMM + bias-add via cuBLASLt epilogue: one launch, no intermediate
// write-then-read of the output tensor.
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
                           float alpha = 1.0f, float beta = 0.0f)
{
    const LtMatmulPlan& plan = Device::get_lt_gemm_bias_plan(m, n, k, transa, transb);

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

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
