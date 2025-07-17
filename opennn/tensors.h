#ifndef TENSORS_H
#define TENSORS_H

#include "pch.h"

namespace opennn
{

template<typename T, std::size_t N>
using array = Eigen::array<T, N>;

template <typename Index>
array<IndexPair<Index>, 1> axes(const Index& a, const Index& b)
{
    return array<IndexPair<Index>, 1>({IndexPair<Index>(a, b)});
}


template <typename Index>
array<IndexPair<Index>, 2> axes(const Index& a1, const Index& b1, const Index& a2, const Index& b2)
{
    const array<IndexPair<Index>, 2> indices
        = { IndexPair<Index>(a1, b1), IndexPair<Index>(a2, b2) };

    return indices;
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



type bound(const type& value, const type& minimum, const type& maximum);

void set_row(Tensor<type, 2>&, const Tensor<type, 1>&, const Index&);

void set_row(Tensor<type, 2, RowMajor>&, const Tensor<type, 1>&, const Index&);

Tensor<type, 2> delete_row(const Tensor<type, 2>&, const Index&);

void sum_matrices(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 3>&);

//void substract_matrices(const ThreadPoolDevice*, const Tensor<type, 2>&, Tensor<type, 3>&);

void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 1>&);
void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 2>&);

void set_identity(Tensor<type, 2>&);

void sum_diagonal(Tensor<type, 2>&, const type&);


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

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);

    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = contraction_axes[0].first == 0 ? A_columns : A_rows;
    const Index C_columns = contraction_axes[0].second == 1 ? B_rows : B_columns;

    // const Index batch_number = (Rank == 3) ? A.dimension(2) : A.dimension(2) * A.dimension(3);
    Index batch_number = 1;
    for (Index rank_index = 2; rank_index < Rank; ++rank_index)
        batch_number *= A.dimension(rank_index);

    const Index A_matrix_size = A_rows * A_columns;
    const Index B_matrix_size = B_rows * B_columns;
    const Index C_matrix_size = C_rows * C_columns;

    const T* A_data = A.data();
    const T* B_data = B.data();
    T* C_data = C.data();

#pragma omp parallel for
    for (Index batch_index = 0; batch_index < batch_number; ++batch_index)
    {
        const TensorMap<const Tensor<T, 2>> A_mat(A_data + batch_index * A_matrix_size, A_rows, A_columns);
        const TensorMap<const Tensor<T, 2>> B_mat(B_data + batch_index * B_matrix_size, B_rows, B_columns);
        TensorMap<Tensor<T, 2>> C_mat(C_data + batch_index * C_matrix_size, C_rows, C_columns);

        C_mat = A_mat.contract(B_mat, contraction_axes);
    }
}

Tensor<type, 2> self_kronecker_product(const ThreadPoolDevice*, const Tensor<type, 1>&);

//void divide_columns(const ThreadPoolDevice*, Tensor<type, 2>&, const Tensor<type, 1>&);
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

type l1_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l1_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l1_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

type l2_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l2_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l2_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

type l2_distance(const type&, const TensorMap<Tensor<type, 0> > &);
type l2_distance(const Tensor<type, 1>&, const Tensor<type, 1>&);
type l2_distance(const type&, const type&);
type l2_distance(const Tensor<type, 2>&, const Tensor<type, 2>&);
Tensor<type, 1> l2_distance(const Tensor<type, 2>&, const Tensor<type, 2>&, const Index&);

void fill_tensor_data_row_major(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_data(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, type*);
void fill_tensor_sequence(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, type*);

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
Tensor<type, 1> string_to_tensor(const string&, const string & = " ");

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


template <typename T>
string tensor_to_string(const Tensor<T, 1>& x, const string& separator = " ")
{
    const Index size = x.size();

    ostringstream buffer;

    if(x.size() == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < size; i++)
        buffer << x[i] << separator;

    return buffer.str();
}


template <typename T>
string tensor_2_to_string(const Tensor<T, 2>& x, const string& separator = " ")
{
    ostringstream buffer;

    if(x.size() == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < x.dimension(0); i++)
        for(Index j = 0; j < x.dimension(1); j++)
            buffer << x(i,j) << separator;

    return buffer.str();
}


template <typename T>
string tensor_4_to_string(const Tensor<T, 4>& x, const string& separator = " ")
{
    ostringstream buffer;

    if(x.size() == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < x.dimension(0); i++)
        for(Index j = 0; j < x.dimension(1); j++)
            for(Index k = 0; k < x.dimension(2); k++)
                for(Index l = 0; l < x.dimension(3); l++)
                    buffer << x(i,j,k,l) << separator;

    return buffer.str();
}


type round_to_precision(type, const int&);

TensorMap<Tensor<type, 1>> tensor_map(const Tensor<type, 2>&, const Index&);

TensorMap<Tensor<type, 2>> tensor_map(const Tensor<type, 3>&, const Index&);
TensorMap<Tensor<type, 3>> tensor_map(const Tensor<type, 4>&, const Index&);
TensorMap<Tensor<type, 2>> tensor_map(const Tensor<type, 4>&, const Index&, const Index&);

TensorMap<Tensor<type, 3>> tensor_map_(const TensorMap<Tensor<type, 4>>&, const Index&);
TensorMap<Tensor<type, 1>> tensor_map_(const TensorMap<Tensor<type, 2>>&, const Index&);


template <Index rank>
TensorMap<Tensor<type, rank>> tensor_map(const pair<type*, dimensions>& x_pair)
{
    if (!x_pair.first)
        throw runtime_error("tensor_map: Null pointer in pair.");

    if (x_pair.second.size() != rank)
        throw runtime_error("Dimensions is " + to_string(x_pair.second.size()) + " and must be " + to_string(rank));

    if constexpr (rank == 1)
        return TensorMap<Tensor<type, 1>>(x_pair.first, x_pair.second[0]);
    else if constexpr (rank == 2)
        return TensorMap<Tensor<type, 2>>(x_pair.first,
                                          x_pair.second[0],
                                          x_pair.second[1]);
    else if constexpr (rank == 3)
        return TensorMap<Tensor<type, 3>>(x_pair.first,
                                          x_pair.second[0],
                                          x_pair.second[1],
                                          x_pair.second[2]);
    else if constexpr (rank == 4)
        return TensorMap<Tensor<type, 4>>(x_pair.first,
                                          x_pair.second[0],
                                          x_pair.second[1],
                                          x_pair.second[2],
                                          x_pair.second[3]);
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


template <typename Type, int Rank>
bool is_equal(const Tensor<Type, Rank>& tensor,
              const Type& value,
              const Type& tolerance = 0.001)
{
    const Index size = tensor.size();

    for (Index i = 0; i < size; i++)
    {
        if constexpr (is_same_v<Type, bool>)
        {
            if (tensor(i) != value)
                return false;
        }
        else
        {
            if (abs(tensor(i) - value) > tolerance)
                return false;
        }
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
        if constexpr (std::is_same_v<Type, bool>)
        {
            if (tensor_1(i) != tensor_2(i))
                return false;
        }
        else
        {
            if (abs(tensor_1(i) - tensor_2(i)) > tolerance)
                return false;
        }

    return true;
}

/*
template <int Rank>
void copy_from_vector(Tensor<type, Rank>& destination, const Tensor<type, 1>& source, Index& index) 
{
    memcpy(destination.data(), source.data() + index, destination.size() * sizeof(type));

    index += destination.size();
}


template <int Rank>
void copy_to_vector(Tensor<type, 1>& destination, const Tensor<type, Rank>& source, Index& index)
{
    memcpy(destination.data() + index, source.data(), source.size() * sizeof(type));

    index += source.size();
}
*/
}

#endif
