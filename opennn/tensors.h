#ifndef TENSORS_H
#define TENSORS_H

#include <iostream>
#include <stdio.h>

#include "config.h"

namespace opennn
{

const Eigen::array<IndexPair<Index>, 1> A_B = { IndexPair<Index>(1, 0) };

type calculate_random_uniform(const type& = type(0), const type& = type(1));

bool calculate_random_bool();

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

void initialize_sequential(Tensor<type, 1>&);
void initialize_sequential(Tensor<Index, 1>&);

void initialize_sequential(Tensor<Index, 1>&, const Index&, const Index&, const Index&);

void get_row(Tensor<type, 1>&, const Tensor<type, 2, RowMajor>&, const Index&);

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
void sum_diagonal(Tensor<type, 2>&, const Tensor<type, 1>&);
void sum_diagonal(TensorMap<Tensor<type, 2>>&, const Tensor<type, 1>&);

void substract_diagonal(Tensor<type, 2>&, const Tensor<type, 1>&);

void multiply_rows(const Tensor<type, 2>&, const Tensor<type, 1>&);
void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 1>&);
void multiply_matrices(const ThreadPoolDevice*, Tensor<type, 3>&, const Tensor<type, 2>&);

void batch_matrix_multiplication(const ThreadPoolDevice*, const TensorMap<Tensor<type, 3>>&, TensorMap<Tensor<type, 3>>&, TensorMap<Tensor<type, 3>>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
void batch_matrix_multiplication(const ThreadPoolDevice*, TensorMap<Tensor<type, 3>>&, const TensorMap<Tensor<type, 3>>&, TensorMap<Tensor<type, 3>>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 4>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
//void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 3>&, Tensor<type, 4>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
//void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 3>&, Tensor<type, 3>&, const Eigen::array<IndexPair<Index>, 1> = A_B);
void batch_matrix_multiplication(const ThreadPoolDevice*, const Tensor<type, 4>&, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, const Eigen::array<IndexPair<Index>, 1> = A_B);

Tensor<type, 2> self_kronecker_product(const ThreadPoolDevice*, const Tensor<type, 1>&);

void divide_columns(const ThreadPoolDevice*, Tensor<type, 2>&, const Tensor<type, 1>&);
void divide_columns(const ThreadPoolDevice*, TensorMap<Tensor<type, 2>>&, const Tensor<type, 1>&);

bool is_zero(const Tensor<type, 1>&, const type& = type(NUMERIC_LIMITS_MIN));

bool is_binary_vector(const Tensor<type, 1>&);
bool is_binary_matrix(const Tensor<type, 2>&);

bool is_constant_vector(const Tensor<type, 1>&);
bool is_constant_matrix(const Tensor<type, 2>&);

bool is_false(const Tensor<bool, 1>&);

bool is_equal(const Tensor<type, 2>&, const type&, const type& = type(0));

bool are_equal(const Tensor<type, 1>&, const Tensor<type, 1>&, const type& = type(0));
bool are_equal(const Tensor<bool, 1>&, const Tensor<bool, 1>&);
bool are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&, const type& = type(0));
bool are_equal(const Tensor<bool, 2>&, const Tensor<bool, 2>&);

Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&);

Index count_NAN(const Tensor<type, 1>&);
Index count_NAN(const Tensor<type, 2>&);

Index count_empty(const vector<string>&);
Index count_not_empty(const vector<string>&);

Index count_less_than(const Tensor<Index, 1>&, const Index&);
Index count_between(Tensor<type, 1>&, const type&, const type&);

Index count_less_than(const Tensor<double, 1>&, const double&);
Index count_greater_than(const vector<Index>&, const Index&);

//void save_csv(const Tensor<type,2>&, const string&);

Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>&);
Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>&);

vector<string> sort_by_rank(const vector<string>&, const Tensor<Index,1>&);
Tensor<Index, 1> sort_by_rank(const Tensor<Index,1>&, const Tensor<Index,1>&);

Tensor<Index, 1> get_indices_less_than(const Tensor<Index,1>&, const Index&);

Tensor<Index, 1> get_indices_less_than(const Tensor<double,1>&, const double&);

vector<Index> get_elements_greater_than(const vector<Index>&, const Index&);
vector<Index> get_elements_greater_than(const vector<vector<Index>>&, const Index&);

void delete_indices(Tensor<Index,1>&, const Tensor<Index,1>&);
void delete_indices(vector<string>&, const Tensor<Index,1>&);
void delete_indices(Tensor<double,1>&, const Tensor<Index,1>&);

//vector<string> get_first(const vector<string>&, const Index&);
//Tensor<Index, 1> get_first(const Tensor<Index,1>&, const Index&);

//void scrub_missing_values(Tensor<type, 2>&, const type&);

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

bool contains(const Tensor<size_t, 1>&, const size_t&);
bool contains(const Tensor<type, 1>&, const type&);
bool contains(const vector<string>&, const string&);
bool contains(const Tensor<Index, 1>&, const Index&);

Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>&, const Tensor<type, 1>&);

vector<Index> join_vector_vector(const vector<Index>&, const vector<Index>&);
vector<string> assemble_text_vector_vector(const vector<string>&, const vector<string>&);

Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_vector_matrix(const Tensor<type, 1>&, const Tensor<type, 2>&);
Tensor<type, 2> assemble_matrix_vector(const Tensor<type, 2>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>&, const Tensor<type, 2>&);

template <typename T>
void push_back(Tensor<T, 1>& tensor, const T& value) 
{
    int new_size = tensor.dimension(0) + 1;

    Tensor<T, 1> new_tensor(new_size);

    for (int i = 0; i < tensor.dimension(0); i++)
        new_tensor(i) = tensor(i);

    new_tensor(new_size - 1) = value;

    tensor = new_tensor;
}

//Tensor<Tensor<Index, 1>, 1> push_back(const Tensor<Tensor<Index, 1>&, 1>, const Tensor<Index, 1>&);

string dimensions_to_string(const dimensions&, const string& = " ");
dimensions string_to_dimensions(const string&, const string& = " ");
Tensor<type, 1> string_to_tensor(const string&, const string & = " ");

string tensor_to_string(const Tensor<type, 1>&, const string& = " ");
string tensor_to_string(const Tensor<Index, 1>&, const string& = " ");
string string_tensor_to_string(const vector<string>&, const string& = " ");

vector<string> to_string_tensor(const Tensor<type, 1>&);

type round_to_precision(type, const int&);
//Tensor<type,2> round_to_precision_matrix(Tensor<type,2>, const int&);
//Tensor<type, 1> round_to_precision_tensor(Tensor<type, 1> tensor, const int& precision);

TensorMap<Tensor<type, 1>> tensor_map(const Tensor<type, 2>&, const Index&);

TensorMap<Tensor<type, 1>> tensor_map_1(const pair<type*, dimensions>& x_pair);
TensorMap<Tensor<type, 2>> tensor_map_2(const pair<type*, dimensions>& x_pair);
TensorMap<Tensor<type, 3>> tensor_map_3(const pair<type*, dimensions>& x_pair);
TensorMap<Tensor<type, 4>> tensor_map_4(const pair<type*, dimensions>& x_pair);

template <typename T>
void print_vector(const vector<T>& vec) 
{
    cout << "[ ";

    for (const auto& element : vec) 
        cout << element << " ";
   
    cout << "]\n";
}

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

}

#endif
