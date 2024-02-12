
#ifndef TENSORUTILITIES_H
#define TENSORUTILITIES_H

// System includes

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <vector>
#include <numeric>
#include <stdio.h>

// OpenNN includes

#include "config.h"
#include "opennn_strings.h"
#include "statistics.h"


#include "../eigen/unsupported/Eigen/KroneckerProduct"

#include "../eigen/Eigen/Dense"

using Eigen::MatrixXd;

namespace opennn
{
// Random

type calculate_random_uniform(const type& = type(0), const type& = type(1));

bool calculate_random_bool();

// Initialization

void initialize_sequential(Tensor<type, 1>&);
void initialize_sequential(Tensor<Index, 1>&);

void initialize_sequential(Tensor<Index, 1>&, const Index&, const Index&, const Index&);

// Rows

void get_row(Tensor<type, 1>&, const Tensor<type, 2, RowMajor>&, const Index&);

void set_row(Tensor<type, 2>&, const Tensor<type, 1>&, const Index&);

void set_row(Tensor<type, 2, RowMajor>&, const Tensor<type, 1>&, const Index&);


Tensor<type, 2> delete_row(const Tensor<type, 2>&, const Index&);

// Columns

// Sum

void sum_columns(ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);
void sum_matrices(ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 3>&);

void sum_diagonal(Tensor<type, 2>&, const type&);
void sum_diagonal(Tensor<type, 2>&, const Tensor<type, 1>&);

// Multiplication

void multiply_rows(Tensor<type, 2>&, const Tensor<type, 1>&);

// Division

void divide_raw_variables(ThreadPoolDevice*, Tensor<type, 2>&, const Tensor<type, 1>&);

// Checking

bool has_NAN(const Tensor<type, 1>&);
bool has_NAN(Tensor<type, 2>&);

bool is_zero(const Tensor<type, 1>&);
bool is_zero(const Tensor<type,1>&, const type&);

bool is_binary(const Tensor<type, 2>&);

bool is_constant(const Tensor<type, 1>&);

bool is_false(const Tensor<bool, 1>&);

bool is_equal(const Tensor<type, 2>&, const type&, const type& = type(0));

bool are_equal(const Tensor<type, 1>&, const Tensor<type, 1>&, const type& = type(0));
bool are_equal(const Tensor<bool, 1>&, const Tensor<bool, 1>&);
bool are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&, const type& = type(0));
bool are_equal(const Tensor<bool, 2>&, const Tensor<bool, 2>&);

bool is_less_than(const Tensor<type, 1>&, const type&);

Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&);

// Count

Index count_NAN(const Tensor<type, 1>&);
Index count_NAN(const Tensor<type, 2>&);

Index count_true(const Tensor<bool, 1>&);

Index count_empty(const Tensor<string, 1>&);

Index count_less_than(const Tensor<Index, 1>&, const Index&);
Index count_between(Tensor<type, 1>&, const type&, const type&);

Index count_less_than(const Tensor<double, 1>&, const double&);
Index count_greater_than(const Tensor<Index, 1>&, const Index&);

// Serialization

void save_csv(const Tensor<type,2>&, const string&);

Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>&);
Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>&);

Tensor<string, 1> sort_by_rank(const Tensor<string,1>&, const Tensor<Index,1>&);
Tensor<Index, 1> sort_by_rank(const Tensor<Index,1>&, const Tensor<Index,1>&);

Tensor<Index, 1> get_indices_less_than(const Tensor<Index,1>&, const Index&);

Tensor<Index, 1> get_indices_less_than(const Tensor<double,1>&, const double&);

Tensor<Index, 1> get_elements_greater_than(const Tensor<Index, 1>&, const Index&);
Tensor<Index, 1> get_elements_greater_than(const Tensor<Tensor<Index, 1>,1>&, const Index&);

void delete_indices(Tensor<Index,1>&, const Tensor<Index,1>&);
void delete_indices(Tensor<string,1>&, const Tensor<Index,1>&);
void delete_indices(Tensor<double,1>&, const Tensor<Index,1>&);

Tensor<string, 1> get_first(const Tensor<string,1>&, const Index&);
Tensor<Index, 1> get_first(const Tensor<Index,1>&, const Index&);

void scrub_missing_values(Tensor<type, 2>&, const type&);

Tensor<type,2> filter_raw_variable_minimum_maximum(Tensor<type,2>&, const Index&, const type&, const type&);

// Kronecker product

Tensor<type, 2> kronecker_product(const Tensor<type, 1>&, const Tensor<type, 1>&);

void kronecker_product(const Tensor<type, 1>&, Tensor<type, 2>&);

// L1 norm

type l1_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l1_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l1_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

// L2 norm

type l2_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l2_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l2_norm_hessian(const ThreadPoolDevice*, Tensor<type, 1>&, Tensor<type, 2>&);

// L2 distance

type l2_distance(const type&, const TensorMap<Tensor<type, 0> > &);
type l2_distance(const Tensor<type, 1>&, const Tensor<type, 1>&);
type l2_distance(const type&, const type&);
type l2_distance(const Tensor<type, 2>&, const Tensor<type, 2>&);
Tensor<type, 1> l2_distance(const Tensor<type, 2>&, const Tensor<type, 2>&, const Index&);

// Check

void check_size(const Tensor<type, 1>&, const Index&, const string&);

void check_dimensions(const Tensor<type, 2>&, const Index&, const Index&, const string&);

void check_raw_variables_number(const Tensor<type, 2>&, const Index&, const string&);
void check_rows_number(const Tensor<type, 2>&, const Index&, const string&);

// Fill

void fill_submatrix(const Tensor<type, 2>&, const Tensor<Index, 1>& rows_indices, const Tensor<Index, 1>&, type*);
void fill_submatrix(const Tensor<type, 2>&, const Tensor<Index, 1>&, const Tensor<Index, 1>&, Tensor<type, 2>&);

// Contain

bool contains(const Tensor<size_t, 1>&, const size_t&);
bool contains(const Tensor<type, 1>&, const type&);
bool contains(const Tensor<string, 1>&, const string&);
bool contains(const Tensor<Index, 1>&, const Index&);

Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>&, const Tensor<type, 1>&);

// Assemble 

Tensor<Index, 1> join_vector_vector(const Tensor<Index, 1>&, const Tensor<Index, 1>&);
Tensor<string, 1> assemble_text_vector_vector(const Tensor<string, 1>&, const Tensor<string, 1>&);

Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_vector_matrix(const Tensor<type, 1>&, const Tensor<type, 2>&);
Tensor<type, 2> assemble_matrix_vector(const Tensor<type, 2>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>&, const Tensor<type, 2>&);

// Push back

void push_back_index(Tensor<Index, 1>&, const Index&);
void push_back_string(Tensor<string, 1>&, const string&);
void push_back_type(Tensor<type, 1>&, const type&);

Tensor<Tensor<Index, 1>, 1> push_back(const Tensor<Tensor<Index, 1>&, 1>, const Tensor<Index, 1>&);

// Conversion

string string_tensor_to_string(const Tensor<string, 1>&, const string&);

Tensor<string, 1> to_string_tensor(const Tensor<type, 1>&);



Index partition(Tensor<type, 2>&, Index, Index, Index);
Tensor<Index, 1> intersection(const Tensor<Index, 1>&, const Tensor<Index, 1>&);
void swap_rows(Tensor<type, 2>&, Index, Index);
void quick_sort(Tensor<type, 2>&, Index, Index, Index);
void quicksort_by_column(Tensor<type, 2>&, Index);

Tensor<type, 1> calculate_delta(const Tensor<type, 1>&);
Tensor<type, 1> fill_gaps_by_value(Tensor<type, 1>&, Tensor<type, 1>&, const type&);
Tensor<type, 1> mode(Tensor<type, 1>&);

void print_tensor(const float* vector, const int dimensions[]);


template<class T, int n>
Tensor<Index, 1> get_dimensions(const Tensor<T, n>& tensor)
{
    Tensor<Index, 1> dimensions(n);

    memcpy(dimensions.data(), tensor.dimensions().data(), size_t(n)*sizeof(Index));

    return dimensions;
}

/*
template<int rank>
pair<type*, dimensions> get_pair(const Tensor<type, rank>& tensor)
{
    type* data = const_cast<type*>(tensor.data());
    
    dimensions dimensions(1);

    for (int i = 0; i < rank; i++)
        dimensions[0].push_back(tensor.dimension(i));

    return pair<type*, dimensions>(data, dimensions);
}
*/
}

#endif
