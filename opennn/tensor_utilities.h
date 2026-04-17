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

static constexpr Index ALIGN_BYTES = EIGEN_MAX_ALIGN_BYTES; // usually 32
static constexpr Index ALIGN_ELEMENTS = ALIGN_BYTES / sizeof(type);

inline int to_int(Index value) { return static_cast<int>(value); }
inline type to_type(Index value) { return static_cast<type>(value); }
static constexpr Index ALIGN_MASK = ~(ALIGN_ELEMENTS - 1);

inline Index get_aligned_size(Index size)
{
    if (size == 0) return 0;
    return (size + ALIGN_ELEMENTS - 1) & ALIGN_MASK;
}

// Signed size helper: returns STL container .size() as Index (avoids static_cast<Index>(x.size()) noise).
template<typename Container>
inline Index ssize(const Container& c) noexcept
{
    return static_cast<Index>(c.size());
}

inline bool is_aligned(const void* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % ALIGN_BYTES == 0;
}

#ifdef OPENNN_WITH_CUDA

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

#endif

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

    Index& back() noexcept
    {
        return shape_[rank_ - 1];
    }

    const Index& back() const
    {
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
        if (Rank != rank_) {
            throw std::runtime_error("Shape Error: Requested Rank (" + std::to_string(Rank) +
                                     ") does not match Shape rank (" + std::to_string(rank_) + ").");
        }
        Eigen::array<Index, Rank> dims;
        for (size_t i = 0; i < Rank; ++i) {
            dims[i] = shape_[i];
        }
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

#ifdef OPENNN_WITH_CUDA
    float* device_data = nullptr;
    Index allocated_size = 0;

    Memory() = default;
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    Memory(Memory&& other) noexcept
        : vector(std::move(other.vector)),
          device_data(other.device_data),
          allocated_size(other.allocated_size)
    {
        other.device_data = nullptr;
        other.allocated_size = 0;
    }

    Memory& operator=(Memory&& other) noexcept
    {
        if(this != &other)
        {
            if(device_data) cudaFree(device_data);
            vector = std::move(other.vector);
            device_data = other.device_data;
            allocated_size = other.allocated_size;
            other.device_data = nullptr;
            other.allocated_size = 0;
        }
        return *this;
    }

    ~Memory()
    {
        if(device_data) cudaFree(device_data);
    }

    void resize_device(Index n)
    {
        if(n == allocated_size) return;
        if(device_data) cudaFree(device_data);
        allocated_size = n;
        if(n > 0) CHECK_CUDA(cudaMalloc(&device_data, n * sizeof(float)));
    }

    void setZero_device()
    {
        if(device_data && allocated_size > 0)
            CHECK_CUDA(cudaMemset(device_data, 0, allocated_size * sizeof(float)));
    }

    type* device() { return device_data; }
    const type* device() const { return device_data; }
#endif
};


struct TensorView
{
    TensorView() noexcept = default;

    type* data = nullptr;

    Shape shape;

    TensorView(type* new_data, const Shape& new_shape) noexcept
        : data(new_data), shape(new_shape) {}

    Index get_rank() const noexcept { return shape.rank(); }

    Index size() const noexcept { return shape.size(); }

    bool empty() const noexcept { return shape.empty(); }

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

#ifdef OPENNN_WITH_CUDA

    float* device = nullptr;

    shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    TensorView(float* new_data, std::shared_ptr<cudnnTensorStruct> handle)
        : data(new_data), descriptor_handle(handle) {}

    explicit TensorView(const Shape& new_shape)
        : shape(new_shape)
    {
        set_descriptor(new_shape);
    }

    cudnnTensorDescriptor_t get_descriptor() const
    {
        return descriptor_handle ? descriptor_handle.get() : nullptr;
    }

    void set_descriptor(const Shape& desc_shape)
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

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, c, h, w));
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

    void fill(float value)
    {
        if (!data) return;

        if (value == 0.0f)
            CHECK_CUDA(cudaMemset(data, 0, device_size() * sizeof(float)));
        // @todo non-zero fill needs get_cudnn_handle()
    }

#endif

};

VectorR filter_missing_values(const VectorR& input);

pair<VectorR, VectorR> filter_missing_values(const VectorR&, const VectorR&);
pair<VectorR, MatrixR> filter_missing_values(const VectorR&, const MatrixR&);
pair<MatrixR, MatrixR> filter_missing_values(const MatrixR&, const MatrixR&);

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

    for (size_t i = 0; i < v.size(); i++)
        if (v[i] != first + Index(i))
            return false;

    return true;
}

inline bool is_binary(const VectorR& tensor)
{
    return all_of(tensor.data(), tensor.data() + tensor.size(),
                  [](type v) { return v == type(0) || v == type(1) || isnan(v); });
}

template <int Rank>
bool is_binary(const TensorR<Rank>& tensor)
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

inline bool is_constant(const VectorR& tensor)
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

template <int Rank>
bool is_constant(const TensorR<Rank>& tensor)
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

bool contains(const vector<string>&, const string&);

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

template <typename T>
string vector_to_string(const vector<T>& x, const string& separator = " ")
{
    ostringstream buffer;

    for(size_t i = 0; i < x.size(); i++)
    {
        buffer << x[i];
        if (i < x.size() - 1)
            buffer << separator;
    }

    return buffer.str();
}

string vector_to_string(const VectorI& x, const string& separator = " ");
string vector_to_string(const VectorR& x, const string& separator = " ");
string vector_to_string(const VectorMap& x, const string& separator = " ");

template <typename T, size_t Rank>
string tensor_to_string(const TensorR<Rank>& x, const string& separator = " ")
{
    ostringstream buffer;

    for(Index i = 0; i < x.size(); i++)
        buffer << x(i) << separator;

    return buffer.str();
}

template <typename T, size_t Rank>
void string_to_tensor(const string& input, TensorR<Rank>& x)
{
    istringstream stream(input);
    T value;
    Index i = 0;

    while (stream >> value)
        x(i++) = value;
}

void string_to_vector(const string& input, VectorR& x);

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

    if constexpr (rank == 2) // @todo For what is this? Can we simplify?
        if (tensor_view.get_rank() == 4)
            return TensorMap2(tensor_view.data,
                              tensor_view.shape[0],
                              tensor_view.size() / tensor_view.shape[0]);

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
    VectorI shape(n);

    memcpy(shape.data(), tensor.dimensions().data(), static_cast<size_t>(n)*sizeof(Index));

    return shape;
}

template <typename Type, int Rank>
bool is_equal(const Tensor<Type, Rank, AlignedMax>& tensor,
              const Type& value,
              const Type& tolerance = 0.001)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
    {
        if constexpr (is_same_v<Type, bool>)
        {
            if (tensor(i) != value)
                return false;
        }
        else
        {
            if (std::abs(tensor(i) - value) > tolerance)
                return false;
        }
    }

    return true;
}

inline bool is_equal(const MatrixR& matrix,
                     const type& value,
                     const type& tolerance = type(1.0e-3))
{
    const type* data = matrix.data();
    const Index size = matrix.size();

    for(Index i = 0; i < size; ++i)
        if(std::abs(data[i] - value) > tolerance)
            return false;

    return true;
}

inline bool is_equal(const VectorR& vector,
                     const type& value,
                     const type& tolerance = type(1.0e-3))
{
    const type* data = vector.data();
    const Index size = vector.size();

    for(Index i = 0; i < size; ++i)
        if(std::abs(data[i] - value) > tolerance)
            return false;

    return true;
}

template <int Rank>
bool are_equal(const TensorR<Rank>& A,
               const TensorR<Rank>& B,
               type tolerance = type(1.0e-3))
{
    if(A.size() != B.size())
        throw runtime_error("are_equal: Tensor sizes are different.");

    const type* a = A.data();
    const type* b = B.data();

    for(Index i = 0; i < A.size(); ++i)
        if(abs(a[i] - b[i]) > tolerance)
            return false;

    return true;
}

inline bool are_equal(const MatrixR& A,
                      const MatrixR& B,
                      type tolerance = type(1.0e-3))
{
    if(A.rows() != B.rows() || A.cols() != B.cols())
        throw runtime_error("are_equal: Matrix sizes are different.");

    const type* a = A.data();
    const type* b = B.data();

    for(Index i = 0; i < A.size(); ++i)
        if(abs(a[i] - b[i]) > tolerance)
            return false;

    return true;
}

inline bool are_equal(const VectorR& A,
                      const VectorR& B,
                      type tolerance = type(1.0e-3))
{
    if(A.size() != B.size())
        throw runtime_error("are_equal: Vector sizes are different.");

    const type* a = A.data();
    const type* b = B.data();

    for(Index i = 0; i < A.size(); ++i)
        if(abs(a[i] - b[i]) > tolerance)
            return false;

    return true;
}

enum class DeviceType { Cpu, Gpu };

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

#ifdef OPENNN_WITH_CUDA
    static cublasHandle_t get_cublas_handle()                      { return instance().cublas_handle; }
    static cudnnHandle_t get_cudnn_handle()                        { return instance().cudnn_handle; }
    static cudnnOpTensorDescriptor_t get_operator_sum_descriptor() { return instance().operator_sum_descriptor; }
    static cudnnOpTensorDescriptor_t get_operator_multiplication_descriptor() { return instance().operator_multiplication_descriptor; }
    static cudnnReduceTensorDescriptor_t get_reduce_add_descriptor();
    static void* get_reduction_workspace();
    static size_t get_reduction_workspace_size();

    static float* get_ones(int n)
    {
        auto& self = instance();
        if(n > self.ones_size)
        {
            if(self.ones_device) cudaFree(self.ones_device);
            cudaMalloc(&self.ones_device, n * sizeof(float));
            vector<float> h(n, 1.0f);
            cudaMemcpy(self.ones_device, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            self.ones_size = n;
        }
        return self.ones_device;
    }
#endif

private:
    Device();
    ~Device();

    DeviceType device_type = DeviceType::Cpu;

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

#ifdef OPENNN_WITH_CUDA
    cublasHandle_t cublas_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
    cudnnReduceTensorDescriptor_t reduce_add_descriptor = nullptr;
    void* reduction_workspace = nullptr;
    size_t reduction_workspace_size = 0;
    float* ones_device = nullptr;
    int ones_size = 0;
#endif
};

inline ThreadPoolDevice& get_device()
{
    return *Device::instance().get_thread_pool_device();
}

void set_threads_number(int num_threads);

#ifdef OPENNN_WITH_CUDA

inline const float one = 1.0f;
inline const float zero = 0.0f;
inline const float minus_one = -1.0f;

VectorR vector_from_device(const type*, size_t);
MatrixR matrix_from_device(const type*, size_t, size_t);
Tensor3 tensor3_from_device(const type*, size_t, size_t, size_t);
Tensor4 tensor4_from_device(const type*, size_t, size_t, size_t, size_t);

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
