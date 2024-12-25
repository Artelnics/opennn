#ifndef TENSORS_H
#define TENSORS_H

#include "pch.h"

namespace opennn
{

const Eigen::array<IndexPair<Index>, 1> A_B = { IndexPair<Index>(1, 0) };

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

type bound(const type& value, const type& minimum, const type& maximum);

//void get_row(Tensor<type, 1>&, const Tensor<type, 2, RowMajor>&, const Index&);

void set_row(Tensor<type, 2>&, const Tensor<type, 1>&, const Index&);

void set_row(Tensor<type, 2, RowMajor>&, const Tensor<type, 1>&, const Index&);

Tensor<type, 2> delete_row(const Tensor<type, 2>&, const Index&);

void sum_columns(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);
void sum_columns(const ThreadPoolDevice*, const Tensor<type, 1>&, TensorMap<Tensor<type, 2>>&);
void sum_matrices(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 3>&);
//void sum_matrices(const ThreadPoolDevice*, const TensorMap<Tensor<type, 1>>&, Tensor<type, 3>&);
//void sum_matrices(const ThreadPoolDevice*, const Tensor<type, 2>&, Tensor<type, 3>&);

void substract_columns(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);
void substract_matrices(const ThreadPoolDevice*, const Tensor<type, 2>&, Tensor<type, 3>&);

void set_identity(Tensor<type, 2>&);

void sum_diagonal(Tensor<type, 2>&, const type&);
//void sum_diagonal(Tensor<type, 2>&, const Tensor<type, 1>&);

//void substract_diagonal(Tensor<type, 2>&, const Tensor<type, 1>&);

void multiply_rows(const Tensor<type, 2>&, const Tensor<type, 1>&);
void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 1>&);
void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 2>&);

void batch_matrix_multiplication(const ThreadPoolDevice*, const TensorMap<Tensor<type, 3>>&, TensorMap<Tensor<type, 3>>&, TensorMap<Tensor<type, 3>>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
void batch_matrix_multiplication(const ThreadPoolDevice*, TensorMap<Tensor<type, 3>>&, const TensorMap<Tensor<type, 3>>&, TensorMap<Tensor<type, 3>>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
//void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 3>&, Tensor<type, 4>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
//void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 3>&, Tensor<type, 3>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
//void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, const Eigen::array<IndexPair<Index>, 1> = A_B);

Tensor<type, 2> self_kronecker_product(const ThreadPoolDevice*, const Tensor<type, 1>&);

//void divide_columns(const ThreadPoolDevice*, Tensor<type, 2>&, const Tensor<type, 1>&);
void divide_columns(const ThreadPoolDevice*, TensorMap<Tensor<type, 2>>&, const Tensor<type, 1>&);

bool is_binary_vector(const Tensor<type, 1>&);
bool is_binary_matrix(const Tensor<type, 2>&);

bool is_constant_vector(const Tensor<type, 1>&);
bool is_constant_matrix(const Tensor<type, 2>&);

Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&);
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

Tensor<type,2> filter_column_minimum_maximum(Tensor<type,2>&, const Index&, const type&, const type&);

type l1_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l1_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l1_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

type l2_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l2_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l2_norm_hessian(const ThreadPoolDevice*, Tensor<type, 1>&, Tensor<type, 2>&);

type l2_distance(const type&, const TensorMap<Tensor<type, 0> > &);
type l2_distance(const Tensor<type, 1>&, const Tensor<type, 1>&);
type l2_distance(const type&, const type&);
type l2_distance(const Tensor<type, 2>&, const Tensor<type, 2>&);
Tensor<type, 1> l2_distance(const Tensor<type, 2>&, const Tensor<type, 2>&, const Index&);

void fill_tensor_data(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, type*);
void fill_tensor_data_row_major(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&, type*);


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


template <typename T>
string vector_to_string(const vector<T>& x, const string& separator = " ")
{
    ostringstream buffer;

    for(size_t i = 0; i < x.size(); i++)
        buffer << x[i] << separator;

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


type round_to_precision(type, const int&);
//Tensor<type,2> round_to_precision_matrix(Tensor<type,2>, const int&);
//Tensor<type, 1> round_to_precision_tensor(Tensor<type, 1> tensor, const int& precision);

TensorMap<Tensor<type, 1>> tensor_map(const Tensor<type, 2>&, const Index&);

TensorMap<Tensor<type, 1>> tensor_map_1(const pair<type*, dimensions>& x_pair);
TensorMap<Tensor<type, 2>> tensor_map_2(const pair<type*, dimensions>& x_pair);
TensorMap<Tensor<type, 3>> tensor_map_3(const pair<type*, dimensions>& x_pair);
TensorMap<Tensor<type, 4>> tensor_map_4(const pair<type*, dimensions>& x_pair);

template <typename T>
size_t get_maximum_size(const vector<vector<T>>& v)
{
    size_t maximum_size = 0;

    for (size_t i = 0; i < v.size(); i++)
        if (v[i].size() > maximum_size)
            maximum_size = v[i].size();

    return maximum_size;
}


template <typename T>
void print_vector(const vector<T>& vec) 
{
    cout << "[ ";

    for (const auto& element : vec) 
        cout << element << " ";
   
    cout << "]\n";
}

void print_pairs(const vector<pair<string, Index>>&);


template<class T, int n>
Tensor<Index, 1> get_dimensions(const Tensor<T, n>& tensor)
{
    Tensor<Index, 1> dimensions(n);

    memcpy(dimensions.data(), tensor.dimensions().data(), size_t(n)*sizeof(Index));

    return dimensions;
}


template<class T>
Tensor<T, 1> tensor_wrapper(T obj)
{
    Tensor<T, 1> wrapper(1);
    wrapper.setValues({obj});

    return wrapper;
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
            else if (abs(tensor(i) - value) > tolerance)
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

}

#endif
