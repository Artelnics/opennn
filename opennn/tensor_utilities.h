
#ifndef TENSORUTILITIES_H
#define TENSORUTILITIES_H

// System includes

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>
#include <numeric>
#include <stdio.h>

// OpenNN includes

#include "config.h"
#include "opennn_strings.h"

#include "../eigen/unsupported/Eigen/KroneckerProduct"

#include "../eigen/Eigen/Dense"

using Eigen::MatrixXd;

namespace opennn
{

void initialize_sequential(Tensor<type, 1>&);
void initialize_sequential(Tensor<Index, 1>&);

void initialize_sequential(Tensor<Index, 1>&, const Index&, const Index&, const Index&);

void multiply_rows(Tensor<type, 2>&, const Tensor<type, 1>&);

void divide_columns(ThreadPoolDevice*, Tensor<type, 2>&, const Tensor<type, 1>&);
void divide_columns(ThreadPoolDevice*, TensorMap<Tensor<type, 2>>&, const Tensor<type, 1>&);

bool is_zero(const Tensor<type, 1>&);
bool is_zero(const Tensor<type,1>&, const type&);

bool is_nan(const Tensor<type,1>&);
bool is_nan(const type&);

bool is_constant(const Tensor<type, 1>&);

bool is_equal(const Tensor<type, 2>&, const type&, const type& = type(0));

bool are_equal(const Tensor<type, 1>&, const Tensor<type, 1>&, const type& = type(0));
bool are_equal(const Tensor<bool, 1>&, const Tensor<bool, 1>&);
bool are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&, const type& = type(0));
bool are_equal(const Tensor<bool, 2>&, const Tensor<bool, 2>&);

Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&);

bool is_false(const Tensor<bool, 1>&);
Index true_count(const Tensor<bool, 1>&);

bool is_binary(const Tensor<type, 2>&);

void save_csv(const Tensor<type,2>&, const string&);

Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>&);
Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>&);

Tensor<string, 1> sort_by_rank(const Tensor<string,1>&, const Tensor<Index,1>&);
Tensor<Index, 1> sort_by_rank(const Tensor<Index,1>&, const Tensor<Index,1>&);

Index count_elements_less_than(const Tensor<Index,1>&, const Index&);

bool is_less_than(const Tensor<type, 1>&, const type&);
Tensor<Index, 1> get_indices_less_than(const Tensor<Index,1>&, const Index&);

Index count_elements_less_than(const Tensor<double,1>&, const double&);
Tensor<Index, 1> get_indices_less_than(const Tensor<double,1>&, const double&);

Index count_elements_greater_than(const Tensor<Index,1>&, const Index&);
Tensor<Index, 1> get_elements_greater_than(const Tensor<Index, 1>&, const Index&);
Tensor<Index, 1> get_elements_greater_than(const Tensor<Tensor<Index, 1>,1>&, const Index&);

void delete_indices(Tensor<Index,1>&, const Tensor<Index,1>&);
void delete_indices(Tensor<string,1>&, const Tensor<Index,1>&);
void delete_indices(Tensor<double,1>&, const Tensor<Index,1>&);

Tensor<string, 1> get_first(const Tensor<string,1>&, const Index&);
Tensor<Index, 1> get_first(const Tensor<Index,1>&, const Index&);

void scrub_missing_values(Tensor<type, 2>&, const type&);

Index count_between(Tensor<type,1>&, const type&, const type&);
void set_row(Tensor<type,2>&, const Tensor<type,1>&, const Index&);
Tensor<type,2> filter_column_minimum_maximum(Tensor<type,2>&, const Index&, const type&, const type&);

Tensor<type, 2> kronecker_product(Tensor<type, 1>&, Tensor<type, 1>&);

void kronecker_product_void(TensorMap<Tensor<type, 1>>&, TensorMap<Tensor<type, 2>>&);

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

void sum_diagonal(Tensor<type, 2>&, const type&);

Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>&, const Tensor<type, 1>&);

void fill_submatrix(const Tensor<type, 2>&, const Tensor<Index, 1>& rows_indices, const Tensor<Index, 1>&, type*);
void fill_submatrix(const Tensor<type, 2>&, const Tensor<Index, 1>&, const Tensor<Index, 1>&, Tensor<type, 2>&);

Index count_NAN(const Tensor<type, 1>&);
Index count_NAN(const Tensor<type, 2>&);

bool has_NAN(const Tensor<type, 1>&);
bool has_NAN(Tensor<type, 2>&);

Index count_empty_values(const Tensor<string, 1>&);

void check_size(const Tensor<type, 1>&, const Index&, const string&);

void check_dimensions(const Tensor<type, 2>&, const Index&, const Index&, const string&);

void check_columns_number(const Tensor<type, 2>&, const Index&, const string&);
void check_rows_number(const Tensor<type, 2>&, const Index&, const string& );

bool contains(const Tensor<size_t,1>&, const size_t&);
bool contains(const Tensor<type,1>&, const type&);
bool contains(const Tensor<string,1>&, const string&);
bool contains(const Tensor<Index,1>&, const Index&);

// Assemble methods

Tensor<Index, 1> join_vector_vector(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_vector_matrix(const Tensor<type, 1>&, const Tensor<type, 2>&);
Tensor<type, 2> assemble_matrix_vector(const Tensor<type, 2>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>&, const Tensor<type, 2>&);

Tensor<string, 1> assemble_text_vector_vector(const Tensor<string, 1>&, const Tensor<string, 1>&);

void push_back_index(Tensor<Index, 1>&, const Index&);
void push_back_string(Tensor<string, 1>&, const string&);
void push_back_type(Tensor<type, 1>&, const type&);

Index partition(Tensor<type, 2>&, Index, Index, Index);
void swap_rows(Tensor<type, 2>&, Index, Index);
void quick_sort(Tensor<type, 2>&, Index, Index, Index);
void quicksort_by_column(Tensor<type, 2>&, Index);

Tensor<type, 1> compute_elementwise_difference(Tensor<type, 1>&);
Tensor<type, 1> fill_gaps_by_value(Tensor<type, 1>&, Tensor<type, 1>&, type);
Tensor<type, 1> compute_mode(Tensor<type, 1>&);

Tensor<Tensor<Index, 1>, 1> push_back(const Tensor<Tensor<Index, 1>&, 1>, const Tensor<Index, 1>&);

Tensor<type, 2> delete_row(const Tensor<type, 2>&, const Index&);

string tensor_string_to_text(const Tensor<string,1>&, string&);
Tensor<string, 1> to_string_tensor(const Tensor<type,1>&);

template<class T, int n>
Tensor<Index, 1> get_dimensions(const Tensor<T, n>& tensor)
{
    Tensor<Index, 1> dimensions(n);

    memcpy(dimensions.data(), tensor.dimensions().data(), static_cast<size_t>(n)*sizeof(Index));

    return dimensions;
}


void print_tensor(const float* vector, const int dimensions[]);

}

#endif
