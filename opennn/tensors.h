#ifndef TENSORS_H
#define TENSORS_H

#include "pch.h"

namespace opennn
{

struct ParameterView
{
    type* data = nullptr;
    Index size;

    ParameterView() noexcept = default;
    ParameterView(type* new_data, Index new_size) noexcept : data(new_data), size(new_size)
    {}
};


struct TensorView
{
    type* data = nullptr;
    dimensions dims;

    TensorView() noexcept = default;
    TensorView(type* new_data, dimensions new_dims) noexcept : data(new_data), dims(std::move(new_dims))
    {}

    Index rank() const { return dims.size(); }
    //Index size() const { return get_size(dims); }

    template <int Rank>
    auto to_tensor_map() const
    {
        if (dims.size() != Rank)
            throw runtime_error("TensorView Error: Requested TensorMap rank " + to_string(Rank) +
                                " does not match stored dimensions size " + to_string(dims.size()));
        if constexpr (Rank == 1)
            return TensorMap<const Tensor<const type, 1>>(data, dims[0]);
        else if constexpr (Rank == 2)
            return TensorMap<const Tensor<const type, 2>>(data, dims[0], dims[1]);
        else if constexpr (Rank == 3)
            return TensorMap<const Tensor<const type, 3>>(data, dims[0], dims[1], dims[2]);
        else if constexpr (Rank == 4)
            return TensorMap<const Tensor<const type, 4>>(data, dims[0], dims[1], dims[2], dims[3]);
        else
            static_assert(Rank >= 1 && Rank <= 4, "Unsupported TensorMap Rank requested.");
    }

    template <int Rank>
    auto to_tensor_map()
    {
        if (dims.size() != Rank)
            throw runtime_error("TensorView Error: Requested TensorMap rank " + to_string(Rank) +
                                " does not match stored dimensions size " + to_string(dims.size()));
        if constexpr (Rank == 1)
            return TensorMap<const Tensor<const type, 1>>(data, dims[0]);
        else if constexpr (Rank == 2)
            return TensorMap<const Tensor<const type, 2>>(data, dims[0], dims[1]);
        else if constexpr (Rank == 3)
            return TensorMap<const Tensor<const type, 3>>(data, dims[0], dims[1], dims[2]);
        else if constexpr (Rank == 4)
            return TensorMap<const Tensor<const type, 4>>(data, dims[0], dims[1], dims[2], dims[3]);
        else
            static_assert(Rank >= 1 && Rank <= 4, "Unsupported TensorMap Rank requested.");
    }

};


template<typename T, size_t N>
using array = Eigen::array<T, N>;

template <typename Index>
Eigen::array<IndexPair<Index>, 1> axes(const Index& a, const Index& b)
{
    return Eigen::array<IndexPair<Index>, 1>({IndexPair<Index>(a, b)});
}


template <typename Index>
Eigen::array<IndexPair<Index>, 2> axes(const Index& a1, const Index& b1, const Index& a2, const Index& b2)
{
    return Eigen::array<IndexPair<Index>, 2>({IndexPair<Index>(a1, b1), IndexPair<Index>(a2, b2)});
}


inline Eigen::array<Index, 1> array_1(const Index& a)
{
    return Eigen::array<Index, 1>({a});
}


inline Eigen::array<Index, 2> array_2(const Index& a, const Index& b)
{
    return Eigen::array<Index, 2>({a, b});
}


inline Eigen::array<Index, 3> array_3(const Index& a, const Index& b, const Index& c)
{
    return Eigen::array<Index, 3>({a, b, c});
}


inline Eigen::array<Index, 4> array_4(const Index& a, const Index& b, const Index& c, const Index& d)
{
    return Eigen::array<Index, 4>({a, b, c, d});
}


inline array<Index, 5> array_5(const Index& a, const Index& b, const Index& c, const Index& d, const Index& e)
{
    return array<Index, 5>({a, b, c, d, e});
}


Index get_random_index(const Index&, const Index&);

type get_random_type(const type& = type(-1), const type& = type(1));

bool get_random_bool();

template<int rank>
void set_random(Tensor<type, rank>& tensor, const type& minimum = -0.1, const type& maximum = 0.1)
{
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<type> distribution(minimum, maximum);

    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(gen);
}


template<int rank>
void set_random(TensorMap<Tensor<type, rank>>& tensor, const type& minimum = -0.1, const type& maximum = 0.1)
{
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<type> distribution(minimum, maximum);

    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(gen);
}

template<int rank>
void set_random_integers(Tensor<type, rank>& tensor, const Index& minimum, const Index& maximum)
{
    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<Index> distribution(minimum, maximum);

    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = static_cast<type>(distribution(gen));
}

template<int rank>
void set_random_integers(TensorMap<Tensor<type, rank>>& tensor, const Index& minimum, const Index& maximum)
{
    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<Index> distribution(minimum, maximum);

    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = static_cast<type>(distribution(gen));
}


type bound(const type& value, const type& minimum, const type& maximum);

void set_row(Tensor<type, 2>&, const Tensor<type, 1>&, const Index&);

void sum_matrices(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 3>&);

void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 1>&);
void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 2>&);

void set_identity(Tensor<type, 2>&);

void sum_diagonal(Tensor<type, 2>&, const type&);


// template <typename T, Index Rank, typename CTensor>
// void batch_matrix_multiplication(const ThreadPoolDevice* device,
//                                  const Tensor<T, Rank>& A,
//                                  const Tensor<T, Rank>& B,
//                                  CTensor& C,
//                                  const Eigen::array<IndexPair<Index>, 1>& contraction_axes)
// {
//     static_assert(Rank >= 2 && Rank <= 4, "Tensor rank isn't supported");

//     if constexpr (Rank == 2)
//     {
//         C.device(*device) = A.contract(B, contraction_axes);
//         return;
//     }

//     const Index A_rows = A.dimension(0);
//     const Index A_columns = A.dimension(1);

//     const Index B_rows = B.dimension(0);
//     const Index B_columns = B.dimension(1);

//     const Index C_rows = contraction_axes[0].first == 0 ? A_columns : A_rows;
//     const Index C_columns = contraction_axes[0].second == 1 ? B_rows : B_columns;

//     // const Index batch_number = (Rank == 3) ? A.dimension(2) : A.dimension(2) * A.dimension(3);
//     Index batch_number = 1;
//     for (Index rank_index = 2; rank_index < Rank; ++rank_index)
//         batch_number *= A.dimension(rank_index);

//     const Index A_matrix_size = A_rows * A_columns;
//     const Index B_matrix_size = B_rows * B_columns;
//     const Index C_matrix_size = C_rows * C_columns;

//     const T* A_data = A.data();
//     const T* B_data = B.data();
//     T* C_data = C.data();

// #pragma omp parallel for
//     for (Index batch_index = 0; batch_index < batch_number; ++batch_index)
//     {
//         const TensorMap<const Tensor<T, 2>> A_mat(A_data + batch_index * A_matrix_size, A_rows, A_columns);
//         const TensorMap<const Tensor<T, 2>> B_mat(B_data + batch_index * B_matrix_size, B_rows, B_columns);

//         TensorMap<Tensor<T, 2>> C_mat(C_data + batch_index * C_matrix_size, C_rows, C_columns);

//         C_mat = A_mat.contract(B_mat, contraction_axes);
//     }
// }

template <typename T, Index Rank, typename CTensor>
void batch_matrix_multiplication(const ThreadPoolDevice* device,
                                 const Tensor<T, Rank>& A,
                                 const Tensor<T, Rank>& B,
                                 CTensor& C,
                                 const Eigen::array<IndexPair<Index>, 1>& contraction_axes)
{
    static_assert(Rank >= 2 && Rank <= 4, "Tensor rank isn't supported");

    if constexpr (Rank == 2)
    {
        C.device(*device) = A.contract(B, contraction_axes);
        return;
    }

    const Index A_rows = A.dimension(Rank - 2);
    const Index A_cols = A.dimension(Rank - 1);

    const Index B_rows = B.dimension(Rank - 2);
    const Index B_cols = B.dimension(Rank - 1);

    const Index C_rows = A_rows;
    const Index C_cols = B_rows;

    Index batch_number = 1;
    for (Index rank_index = 0; rank_index < Rank - 2; ++rank_index)
    {
        batch_number *= A.dimension(rank_index);
    }


    cout << batch_number << endl;

    const Index A_matrix_size = A_rows * A_cols;
    const Index B_matrix_size = B_rows * B_cols;
    const Index C_matrix_size = C_rows * C_cols;

    const T* A_data = A.data();
    const T* B_data = B.data();
    T* C_data = C.data();

    cout << "C_rows: " << C_rows << endl;
    cout << "C_cols: " << C_cols << endl;

#pragma omp parallel for
    for (Index batch_index = 0; batch_index < batch_number; ++batch_index)
    {
        const TensorMap<const Tensor<T, 2>> A_mat(A_data + batch_index * A_matrix_size, A_rows, A_cols);
        const TensorMap<const Tensor<T, 2>> B_mat(B_data + batch_index * B_matrix_size, B_rows, B_cols);

        cout << "A_mat" << A_mat.dimensions() << endl;
        cout << "B_mat" << B_mat.dimensions() << endl;

        TensorMap<Tensor<T, 2>> C_mat(C_data + batch_index * C_matrix_size, C_rows, C_cols);

        // Realiza la contracción para el par de matrices actual
        C_mat.device(*device) = A_mat.contract(B_mat, contraction_axes);
    }
}

Tensor<type, 2> self_kronecker_product(const ThreadPoolDevice*, const Tensor<type, 1>&);

void divide_columns(const ThreadPoolDevice*, TensorMap<Tensor<type, 2>>&, const Tensor<type, 1>&);

template <int Rank>
bool is_binary(const Tensor<type, Rank>& tensor)
{
    const Index size = tensor.size();

    for (Index i = 0; i < size; i++)
        if (tensor(i) != type(0) && tensor(i) != type(1) && !isnan(tensor(i)))
            return false;

    return true;
}


template <int Rank>
bool is_constant(const Tensor<type, Rank>& tensor)
{
    const Index size = tensor.size();

    Index first_non_nan_index = 0;

    while (first_non_nan_index < size && isnan(tensor(first_non_nan_index)))
        first_non_nan_index++;

    if (first_non_nan_index == size)
        return true;

    const type first_not_nan_element = tensor(first_non_nan_index);

    for (Index i = first_non_nan_index + 1; i < size; ++i)
        if (!isnan(tensor(i)) && abs(first_not_nan_element - tensor(i)) > numeric_limits<float>::min())
            return false;

    return true;
}

void save_csv(const Tensor<type,2>&, const filesystem::path&);

template<int rank>
Index count_NAN(const Tensor<type, rank>& x)
{
    return count_if(x.data(), x.data() + x.size(), [](type value) {return std::isnan(value); });
}

Index count_between(Tensor<type, 1>&, const type&, const type&);

Index count_greater_than(const vector<Index>&, const Index&);

Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>&);
Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>&);

vector<Index> get_elements_greater_than(const vector<Index>&, const Index&);
vector<Index> get_elements_greater_than(const vector<vector<Index>>&, const Index&);

Tensor<type,2> filter_column_minimum_maximum(const Tensor<type,2>&, const Index&, const type&, const type&);

//type l2_distance(const type&, const TensorMap<Tensor<type, 0> > &);
type l2_distance(const Tensor<type, 1>&, const Tensor<type, 1>&);

void fill_tensor_data_row_major(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_data(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_sequence(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, const Index&, type*);

template <typename Type, int Rank>
bool contains(const Tensor<Type, Rank>& vector, const Type& value)
{
    Tensor<Type, 1> copy(vector);

    const Type* it = find(copy.data(), copy.data() + copy.size(), value);

    return it != (copy.data() + copy.size());
}


bool contains(const vector<string>&, const string&);

Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>&, const Tensor<type, 1>&);

vector<Index> join_vector_vector(const vector<Index>&, const vector<Index>&);

Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_vector_matrix(const Tensor<type, 1>&, const Tensor<type, 2>&);
Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>&, const Tensor<type, 2>&);

template <typename T>
void push_back(Tensor<T, 1>& tensor, const T& value)
{
    const int new_size = tensor.dimension(0) + 1;

    Tensor<T, 1> new_tensor(new_size);

    for (int i = 0; i < tensor.dimension(0); i++)
        new_tensor(i) = tensor(i);

    new_tensor(new_size - 1) = value;

    tensor = new_tensor;
}

string dimensions_to_string(const dimensions&, const string& = " ");
dimensions string_to_dimensions(const string&, const string& = " ");

dimensions prepend(const Index& x, const dimensions& d);

Index get_size(const dimensions& d);

template <typename T>
string vector_to_string(const vector<T>& x, const string& separator = " ")
{
    ostringstream buffer;

    for (size_t i = 0; i < x.size(); i++)
    {
        buffer << x[i];
        if (i < x.size() - 1)
            buffer << separator;
    }

    return buffer.str();
}


template <typename T, size_t Rank>
string tensor_to_string(const Tensor<T, Rank>& x, const string& separator = " ")
{
    ostringstream buffer;

    for(Index i = 0; i < x.size(); i++)
        buffer << x(i) << separator;

    return buffer.str();
}


template <typename T, size_t Rank>
void string_to_tensor(const string& input, Tensor<T, Rank>& x)
{
    istringstream stream(input);
    T value;
    Index i = 0;

    while (stream >> value)
        x(i++) = value;
}


type round_to_precision(type, const int&);

TensorMap<Tensor<type, 1>> tensor_map(const Tensor<type, 2>&, const Index&);

TensorMap<Tensor<type, 2>> tensor_map(const Tensor<type, 3>&, const Index&);
TensorMap<Tensor<type, 3>> tensor_map(const Tensor<type, 4>&, const Index&);
TensorMap<Tensor<type, 2>> tensor_map(const Tensor<type, 4>&, const Index&, const Index&);

TensorMap<Tensor<type, 3>> tensor_map_(const TensorMap<Tensor<type, 4>>&, const Index&);
TensorMap<Tensor<type, 1>> tensor_map_(const TensorMap<Tensor<type, 2>>&, const Index&);


template <Index rank>
TensorMap<Tensor<type, rank>> tensor_map(const TensorView& x_pair)
{
    if (!x_pair.data)
        throw runtime_error("tensor_map: Null pointer in pair.");

    if (x_pair.rank() != rank)
        throw runtime_error("Dimensions is " + to_string(x_pair.rank()) + " and must be " + to_string(rank));

    if constexpr (rank == 1)
        return TensorMap<Tensor<type, 1>>(x_pair.data, x_pair.dims[0]);
    else if constexpr (rank == 2)
        return TensorMap<Tensor<type, 2>>(x_pair.data,
                                          x_pair.dims[0],
                                          x_pair.dims[1]);
    else if constexpr (rank == 3)
        return TensorMap<Tensor<type, 3>>(x_pair.data,
                                          x_pair.dims[0],
                                          x_pair.dims[1],
                                          x_pair.dims[2]);
    else if constexpr (rank == 4)
        return TensorMap<Tensor<type, 4>>(x_pair.data,
                                          x_pair.dims[0],
                                          x_pair.dims[1],
                                          x_pair.dims[2],
                                          x_pair.dims[3]);
    else
        static_assert(rank >= 1 && rank <= 4, "Unsupported tensor rank");
}


template <typename T>
size_t get_maximum_size(const vector<vector<T>>& v)
{
    size_t maximum_size = 0;

    //#pragma omp parallel for reduction(max : maximum_size)
    for (size_t i = 0; i < v.size(); i++)
        if (v[i].size() > maximum_size)
            maximum_size = v[i].size();

    return maximum_size;
}


template <typename T>
void print_vector(const vector<T>& vec)
{
    cout << "[ ";

    for (size_t i = 0; i < vec.size(); ++i) {
        cout << vec[i];
        if (i < vec.size() - 1)
            cout << ";";
    }

    cout << " ]\n";
}


template <typename T>
void print_vector(const vector<vector<T>>& vec)
{
    cout << "[ ";

    for (size_t i = 0; i < vec.size(); ++i)
    {
        print_vector(vec[i]);
        if (i < vec.size() - 1)
            cout << ";";
    }

    cout << " ]\n";
}

void print_pairs(const vector<pair<string, Index>>&);


template<class T, int n>
Tensor<Index, 1> get_dimensions(const Tensor<T, n>& tensor)
{
    Tensor<Index, 1> dimensions(n);

    memcpy(dimensions.data(), tensor.dimensions().data(), size_t(n)*sizeof(Index));

    return dimensions;
}


template <int Rank>
bool is_equal(const Tensor<bool, Rank>& tensor,
    const bool& value)
{
    const Index size = tensor.size();

    for (Index i = 0; i < size; i++)
    {
        if (tensor(i) != value)
            return false;
    }

    return true;
}


template <typename Type, int Rank>
bool is_equal(const Tensor<Type, Rank>& tensor,
    const Type& value,
    const Type& tolerance = 0.001)
{
    const Index size = tensor.size();

    for (Index i = 0; i < size; i++)
    {
        if (std::abs(tensor(i) - value) > tolerance)
            return false;
    }

    return true;
}


template <typename Type, int Rank>
bool are_equal(const Tensor<Type, Rank>& tensor_1,
               const Tensor<Type, Rank>& tensor_2,
               const Type& tolerance = 0.001)
{
    if (tensor_1.size() != tensor_2.size())
        throw runtime_error("Tensor sizes are different");

    const Index size = tensor_1.size();

    for (Index i = 0; i < size; i++)
        if constexpr (is_same_v<Type, bool>)
        {
            if (tensor_1(i) != tensor_2(i))
                return false;
            else
                if (abs(tensor_1(i) - tensor_2(i)) > tolerance)
                    return false;
        }

    return true;
}

}

#endif
