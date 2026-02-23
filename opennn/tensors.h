//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

struct Shape
{
    static constexpr size_t MaxRank = 6;
    Index shape[MaxRank] = {0};
    size_t rank = 0;

    Shape() noexcept = default;

    Shape(initializer_list<Index> list)
    {
        rank = min(list.size(), MaxRank);
        size_t i = 0;
        for (Index d : list)
            if (i < rank)
                shape[i++] = d;
    }

    Shape(size_t n, Index value)
    {
        rank = (n > MaxRank) ? MaxRank : n;

        for (size_t i = 0; i < rank; ++i)
            shape[i] = value;
    }

    template<typename InputIt, typename = typename enable_if<!is_integral<InputIt>::value>::type>
    Shape(InputIt first, InputIt last) {
        rank = 0;
        while (first != last && rank < MaxRank) {
            shape[rank++] = static_cast<Index>(*first);
            ++first;
        }
    }

    const Index& operator[](size_t i) const noexcept
    {
        return shape[i];
    }

    Index& operator[](size_t i) noexcept
    {
        return shape[i];
    }

    Index& back() noexcept
    {
        return shape[rank - 1];
    }

    const Index& back() const
    {
        return shape[rank - 1];
    }

    size_t size() const noexcept
    {
        return rank;
    }

    bool empty() const noexcept { return rank == 0; }

    Index* begin() noexcept
    {
        return shape;
    }

    Index* end() noexcept
    {
        return shape + rank;
    }

    const Index* begin() const noexcept
    {
        return shape;
    }

    const Index* end() const noexcept
    {
        return shape + rank;
    }

    void push_back(Index d)
    {
        if (rank < MaxRank)
            shape[rank++] = d;
    }


    void insert(const Index* /*pos*/, const Index* first, const Index* last)
    {
        while (first != last)
        {
            this->push_back(*first);
            ++first;
        }
    }


    Index count() const noexcept
    {
        if (rank == 0) return 0;

        Index total = 1;

        for (size_t i = 0; i < rank; ++i)
            total *= shape[i];

        return total;
    }

    void clear() noexcept
    {
        rank = 0;
    }

    void resize(size_t n)
    {
        if (n > MaxRank)
            throw std::out_of_range("Shape::resize: rank exceeds MaxRank (8)");

        rank = n;
    }


    void resize(size_t n, Index value)
    {
        resize(n);

        for (size_t i = 0; i < rank; ++i)
            shape[i] = value;
    }

    friend ostream& operator<<(ostream& os, const Shape& s)
    {
        os << "[ ";

        for (size_t i = 0; i < s.rank; ++i)
            os << s.shape[i] << (i < s.rank - 1 ? ", " : " ");

        os << "]";
        return os;
    }

    bool operator==(const Shape& other) const noexcept
    {
        if (rank != other.rank) return false;

        for (size_t i = 0; i < rank; ++i)
            if (shape[i] != other.shape[i]) return false;

        return true;
    }

    bool operator!=(const Shape& other) const noexcept
    {
        return !(*this == other);
    }
};


struct TensorView
{
    type* data = nullptr;
    Shape shape;

    TensorView() noexcept = default;

    TensorView(type* new_data, const Shape& new_shape) noexcept
    {
        data = new_data;
        shape = new_shape;
    }

    Index rank() const noexcept
    {
        return shape.size();
    }

    Index size() const noexcept
    {
        return shape.count();
    }

    void print() const
    {
        if(!data || shape.empty())
        {
            cout << "TensorView: Empty or Null" << endl;
            return;
        }

        cout << "Shape: " << shape << endl;

        const Index total_size = size();
        const Index last_dim_stride = shape.back();

        for(Index i = 0; i < total_size; ++i)
        {
            cout << data[i] << " ";

            if (shape.size() > 1 && (i + 1) % last_dim_stride == 0)
                cout << endl;
        }

        if (shape.size() == 1 || total_size % last_dim_stride != 0)
            cout << endl;
    }
};


void shuffle_rows(MatrixR& matrix);

type* link(type*, vector<TensorView*>);
void link(type*, vector<vector<TensorView*>>);

Index get_size(const vector<TensorView*>);
Index get_size(vector<vector<TensorView*>>);


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


inline array<Index, 5> array_5(const Index a, Index b, Index c, Index d, Index e)
{
    return array<Index, 5>({a, b, c, d, e});
}

void set_row(MatrixR&, const VectorR&, Index);

void sum_matrices(const VectorR&, Tensor3&);

void multiply_matrices(Tensor3&, const VectorR&);
void multiply_matrices(Tensor3&, const Tensor2&);

void set_identity(MatrixR&);

//Tensor2 self_kronecker_product(const VectorR&);

inline bool is_binary(const VectorR& tensor)
{

    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
        if (tensor(i) != type(0) && tensor(i) != type(1) && !isnan(tensor(i)))
            return false;

    return true;
}


template <int Rank>
bool is_binary(const TensorR<Rank>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
        if (tensor(i) != type(0) && tensor(i) != type(1) && !isnan(tensor(i)))
            return false;

    return true;
}

MatrixR append_rows(const MatrixR& , const MatrixR&);

template<typename T>
vector<T> gather_by_index(const vector<T>& data, const vector<Index>& indices)
{
    vector<T> result;
    result.reserve(indices.size());

    for(Index i : indices)
        result.push_back(data[i]);

    return result;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums);

inline bool is_constant(const VectorR& tensor)
{
    const Index size = tensor.size();

    Index first_non_nan_index = 0;

    while (first_non_nan_index < size && isnan(tensor(first_non_nan_index)))
        first_non_nan_index++;

    if (first_non_nan_index == size)
        return true;

    const type first_not_nan_element = tensor(first_non_nan_index);

    for(Index i = first_non_nan_index + 1; i < size; ++i)
        if(!isnan(tensor(i)) && abs(first_not_nan_element - tensor(i)) > numeric_limits<float>::min())
            return false;

    return true;
}


template <int Rank>
bool is_constant(const TensorR<Rank>& tensor)
{
    const Index size = tensor.size();

    Index first_non_nan_index = 0;

    while (first_non_nan_index < size && isnan(tensor(first_non_nan_index)))
        first_non_nan_index++;

    if (first_non_nan_index == size)
        return true;

    const type first_not_nan_element = tensor(first_non_nan_index);

    for(Index i = first_non_nan_index + 1; i < size; ++i)
        if(!isnan(tensor(i)) && abs(first_not_nan_element - tensor(i)) > numeric_limits<float>::min())
            return false;

    return true;
}

void save_csv(const Tensor2&, const filesystem::path&);


inline Index count_NAN(const MatrixR& x)
{
    return count_if(x.data(), x.data() + x.size(), [](type value) {return std::isnan(value); });
}


template<int rank>
Index count_NAN(const TensorR<rank>& x)
{
    return count_if(x.data(), x.data() + x.size(), [](type value) {return std::isnan(value); });
}

Index count_between(const VectorR&, type, type);

Index count_greater_than(const vector<Index>&, Index);

VectorI calculate_rank_greater(const VectorR&);
VectorI calculate_rank_less(const VectorR&);

vector<Index> get_elements_greater_than(const vector<Index>&, Index);
vector<Index> get_elements_greater_than(const vector<vector<Index>>&, Index);

MatrixR filter_column_minimum_maximum(const MatrixR&, Index, type, type);

VectorI get_nearest_points(const MatrixR& ,const VectorR& , int );

void fill_tensor_data_row_major(const MatrixR&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&, type*);

//void fill_tensor_sequence(const Tensor2&, const vector<Index>&, const vector<Index>&, Index, type*);

template <typename Type, int Rank>
bool contains(const TensorR<Rank>& vector, const Type& value)
{
    Tensor<Type, 1> copy(vector);

    const Type* it = find(copy.data(), copy.data() + copy.size(), value);

    return it != (copy.data() + copy.size());
}


bool contains(const vector<string>&, const string&);

VectorR perform_Householder_QR_decomposition(const MatrixR&, const VectorR&);

vector<Index> join_vector_vector(const vector<Index>&, const vector<Index>&);

template <typename T>
void push_back(Tensor<T, 1, AlignedMax>& tensor, const T& value)
{
    const int new_size = tensor.dimension(0) + 1;

    Tensor<T, 1> new_tensor(new_size);

    for(int i = 0; i < tensor.dimension(0); i++)
        new_tensor(i) = tensor(i);

    new_tensor(new_size - 1) = value;

    tensor = new_tensor;
}

string shape_to_string(const Shape&, const string& = " ");
Shape string_to_shape(const string&, const string& = " ");

Shape prepend(const Index&, const Shape&);

Index get_size(const Shape&);

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


inline string vector_to_string(const VectorI& x, const string& separator = " ")
{
    ostringstream buffer;

    for(Index i = 0; i < x.size(); i++)
        buffer << x(i) << separator;

    return buffer.str();
}


inline string vector_to_string(const VectorR& x, const string& separator = " ")
{
    ostringstream buffer;

    for(Index i = 0; i < x.size(); i++)
        buffer << x(i) << separator;

    return buffer.str();
}


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


inline void string_to_vector(const string& input, VectorR& x)
{
    istringstream stream(input);
    type value;
    vector<type> buffer;

    while (stream >> value)
        buffer.push_back(value);

    x.resize(static_cast<Index>(buffer.size()));

    for (Index i = 0; i < x.size(); ++i)
        x(i) = buffer[i];
}

type round_to_precision(type, const int&);

VectorMap vector_map(const MatrixR&, Index);

VectorMap tensor_map(const Tensor2&, Index);

MatrixMap tensor_map(const Tensor3&, Index);
TensorMap3 tensor_map(const Tensor4&, Index);
MatrixMap tensor_map(const Tensor4&, Index, Index);

TensorMap3 tensor_map_(const TensorMap4, Index);
//VectorMap tensor_map_(const MatrixMap&, Index);

inline VectorMap vector_map(const TensorView& tensor_view)
{
    if(!tensor_view.data)
        throw runtime_error("tensor_map: Null pointer in pair.");

    if (reinterpret_cast<uintptr_t>(tensor_view.data) % EIGEN_MAX_ALIGN_BYTES != 0)
        throw runtime_error("tensor_map alignment error: Pointer is not aligned. "
                            "This will cause a crash with AlignedMax TensorMaps.");

    return VectorMap(tensor_view.data, tensor_view.size() / tensor_view.shape[0]);
}


inline MatrixMap matrix_map(const TensorView& tensor_view)
{
    if(!tensor_view.data)
        throw runtime_error("tensor_map: Null pointer in pair.");

    if (reinterpret_cast<uintptr_t>(tensor_view.data) % EIGEN_MAX_ALIGN_BYTES != 0)
        throw runtime_error("tensor_map alignment error: Pointer is not aligned. "
                            "This will cause a crash with AlignedMax TensorMaps.");

    return MatrixMap(tensor_view.data, tensor_view.shape[0], tensor_view.size() / tensor_view.shape[0]);
}


template <Index rank>
TensorMapR<rank> tensor_map(const TensorView& tensor_view)
{
    if(!tensor_view.data)
        throw runtime_error("tensor_map: Null pointer in pair.");

    if (reinterpret_cast<uintptr_t>(tensor_view.data) % EIGEN_MAX_ALIGN_BYTES != 0)
        throw runtime_error("tensor_map alignment error: Pointer is not aligned. "
                            "This will cause a crash with AlignedMax TensorMaps.");

    if constexpr (rank == 2)
        if (tensor_view.rank() == 4)
            return MatrixMap(tensor_view.data,
                              tensor_view.shape[0],
                              tensor_view.size() / tensor_view.shape[0]);

    if (tensor_view.rank() != rank)
        throw runtime_error("Dimensions is " + to_string(tensor_view.rank()) + " and must be " + to_string(rank));

    if constexpr (rank == 1)
        return VectorMap(tensor_view.data, tensor_view.shape[0]);
    else if constexpr (rank == 2)
        return MatrixMap(tensor_view.data,
                          tensor_view.shape[0],
                          tensor_view.shape[1]);
    else if constexpr (rank == 3)
        return TensorMap3(tensor_view.data,
                          tensor_view.shape[0],
                          tensor_view.shape[1],
                          tensor_view.shape[2]);
    else if constexpr (rank == 4)
        return TensorMap4(tensor_view.data,
                          tensor_view.shape[0],
                          tensor_view.shape[1],
                          tensor_view.shape[2],
                          tensor_view.shape[3]);
    else if constexpr (rank == 5)
        return TensorMap5(tensor_view.data,
                          tensor_view.shape[0],
                          tensor_view.shape[1],
                          tensor_view.shape[2],
                          tensor_view.shape[3],
                          tensor_view.shape[4]);

    else
        static_assert(rank >= 1 && rank <= 5, "Unsupported tensor rank");
}


template <typename T>
size_t get_maximum_size(const vector<vector<T>>& v)
{
    size_t maximum_size = 0;

    for(size_t i = 0; i < v.size(); i++)
        if (v[i].size() > maximum_size)
            maximum_size = v[i].size();

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

    memcpy(shape.data(), tensor.dimensions().data(), size_t(n)*sizeof(Index));

    return shape;
}


template <typename Type, int Rank>
bool is_equal(const Tensor<Type, Rank, AlignedMax>& tensor,
              const Type& value,
              const Type& tolerance = 0.001)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
        if constexpr (is_same_v<Type, bool>)
        {
            if (tensor(i) != value)
                return false;
            else
                if (std::abs(tensor(i) - value) > tolerance)
                    return false;
        }

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

#ifdef OPENNN_CUDA

struct TensorViewCuda
{
    float* data = nullptr;
    shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    TensorViewCuda() = default;

    TensorViewCuda(float* new_data, std::shared_ptr<cudnnTensorStruct> handle)
        : data(new_data), descriptor_handle(handle) {}

    cudnnTensorDescriptor_t get_descriptor() const 
    {
        return descriptor_handle ? descriptor_handle.get() : nullptr;
    }

    void set_descriptor(const Shape& shape)
    {
        if (descriptor_handle == nullptr)
        {
            cudnnTensorDescriptor_t raw_desc;
            if (cudnnCreateTensorDescriptor(&raw_desc) != CUDNN_STATUS_SUCCESS)
                throw std::runtime_error("TensorViewCuda: Failed to create descriptor.");

            descriptor_handle = std::shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t p) {
                if (p) cudnnDestroyTensorDescriptor(p);
                });
        }

        int n = 1, c = 1, h = 1, w = 1;
        if (shape.size() > 0) n = static_cast<int>(shape[0]);
        if (shape.size() > 1) c = static_cast<int>(shape[1]);
        if (shape.size() > 2) h = static_cast<int>(shape[2]);
        if (shape.size() > 3) w = static_cast<int>(shape[3]);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    }

    Index size() const
    {
        if (descriptor_handle == nullptr) return 0;

        constexpr int REQUESTED_DIMS = CUDNN_DIM_MAX;
        cudnnDataType_t dataType;
        int nbDims = 0, dimA[REQUESTED_DIMS], strideA[REQUESTED_DIMS];

        CHECK_CUDNN(cudnnGetTensorNdDescriptor(descriptor_handle.get(), REQUESTED_DIMS, &dataType, &nbDims, dimA, strideA));

        Index total_elements = 1;
        for (int i = 0; i < nbDims; ++i)
            total_elements *= static_cast<Index>(dimA[i]);
        return total_elements;
    }
};


struct TensorCuda
{
    float* data = nullptr;

    shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    TensorCuda() = default;
    explicit TensorCuda(const Shape& shape) { resize(shape); }

    ~TensorCuda() { if (data) cudaFree(data); }

    TensorCuda(const TensorCuda&) = delete;
    TensorCuda& operator=(const TensorCuda&) = delete;

    TensorCuda(TensorCuda&& other) noexcept
        : data(other.data), descriptor_handle(std::move(other.descriptor_handle))
    {
        other.data = nullptr;
    }

    TensorCuda& operator = (TensorCuda&& other) noexcept
    {
        if (this != &other)
        {
            free();
            data = other.data;
            descriptor_handle = std::move(other.descriptor_handle);
            other.data = nullptr;
        }

        return *this;
    }

    cudnnTensorDescriptor_t get_descriptor() const
    {
        return descriptor_handle ? descriptor_handle.get() : nullptr;
    }

    void resize(const Shape& shape)
    {
        set_descriptor(shape);
        const size_t total_elements = size();
        const size_t bytes = total_elements * sizeof(float);
        if (data) cudaFree(data);
        CHECK_CUDA(cudaMalloc(&data, bytes));
        CHECK_CUDA(cudaMemset(data, 0, bytes));
    }

    void set_descriptor(const Shape& shape)
    {
        if (descriptor_handle == nullptr)
        {
            cudnnTensorDescriptor_t raw_desc;
            if (cudnnCreateTensorDescriptor(&raw_desc) != CUDNN_STATUS_SUCCESS)
                throw std::runtime_error("TensorCuda: Failed to create descriptor.");

            descriptor_handle = std::shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t p) {
                if (p) cudnnDestroyTensorDescriptor(p);
                });
        }

        int n = 1, c = 1, h = 1, w = 1;
        if (shape.size() > 0) n = static_cast<int>(shape[0]);
        if (shape.size() > 1) c = static_cast<int>(shape[1]);
        if (shape.size() > 2) h = static_cast<int>(shape[2]);
        if (shape.size() > 3) w = static_cast<int>(shape[3]);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    }

    Index size() const
    {
        if (descriptor_handle == nullptr) return 0;
        constexpr int REQUESTED_DIMS = CUDNN_DIM_MAX;
        cudnnDataType_t dataType;
        int nbDims = 0, dimA[REQUESTED_DIMS], strideA[REQUESTED_DIMS];

        CHECK_CUDNN(cudnnGetTensorNdDescriptor(descriptor_handle.get(), REQUESTED_DIMS, &dataType, &nbDims, dimA, strideA));

        Index total_elements = 1;
        for (int i = 0; i < nbDims; ++i)
            total_elements *= static_cast<Index>(dimA[i]);
        return total_elements;
    }

    void free()
    {
        if (data) { cudaFree(data); data = nullptr; }
        descriptor_handle.reset();
    }

    TensorViewCuda view() const
    {
        return TensorViewCuda(data, descriptor_handle);
    }
};


type* link(type*, vector<TensorViewCuda*>);
void link(type*, vector<vector<TensorViewCuda*>>);

Index get_size(const vector<TensorViewCuda*>);
Index get_size(vector<vector<TensorViewCuda*>>);

#endif

class Device
{
public:
    static Device& instance();
    ThreadPoolDevice* get_thread_pool_device();
    void set_threads_number(int num_threads);

#ifdef OPENNN_CUDA
    cublasHandle_t get_cublas_handle();
    cudnnHandle_t get_cudnn_handle();
    cudnnOpTensorDescriptor_t get_operator_sum_descriptor();
    cudnnOpTensorDescriptor_t get_operator_multiplication_descriptor();
#endif

private:
    Device();
    ~Device();

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

#ifdef OPENNN_CUDA
    cublasHandle_t cublas_handle = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;
    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
#endif
};

inline ThreadPoolDevice& get_device()
{
    return *Device::instance().get_thread_pool_device();
}

void set_threads_number(int num_threads);

#ifdef OPENNN_CUDA
    inline cublasHandle_t get_cublas_handle()
    {
        return Device::instance().get_cublas_handle();
    }


    inline cudnnHandle_t get_cudnn_handle()
    {
        return Device::instance().get_cudnn_handle();
    }


    inline cudnnOpTensorDescriptor_t get_operator_sum_descriptor()
    {
        return Device::instance().get_operator_sum_descriptor();
    }


    inline cudnnOpTensorDescriptor_t get_operator_multiplication_descriptor()
    {
        return Device::instance().get_operator_multiplication_descriptor();
    }

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
