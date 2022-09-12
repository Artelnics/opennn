
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

#include "../eigen/Eigen/Dense"


namespace opennn
{

void initialize_sequential(Tensor<type, 1>&);
void initialize_sequential(Tensor<Index, 1>&);

void multiply_rows(Tensor<type, 2>&, const Tensor<type, 1>&);
void divide_columns(Tensor<type, 2>&, const Tensor<type, 1>&);

bool is_zero(const Tensor<type, 1>&);
bool is_zero(const Tensor<type,1>&, const type&);
bool is_nan(const Tensor<type,1>&);
bool is_constant(const Tensor<type, 1>&);

bool is_equal(const Tensor<type, 2>&, const type&, const type& = type(0));

bool are_equal(const Tensor<type, 1>&, const Tensor<type, 1>&, const type& = type(0));
bool are_equal(const Tensor<bool, 1>&, const Tensor<bool, 1>&);
bool are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&, const type& = type(0));
bool are_equal(const Tensor<bool, 2>&, const Tensor<bool, 2>&);

Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>&, const Tensor<type, 2>&);

bool is_false(const Tensor<bool, 1>&);

bool is_binary(const Tensor<type, 2>&);

void save_csv(const Tensor<type,2>&, const string&);

// Rank and indices methods

Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>&);
Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>&);
Tensor<string, 1> sort_by_rank(const Tensor<string,1>&, const Tensor<Index,1>&);
Tensor<Index, 1> sort_by_rank(const Tensor<Index,1>&, const Tensor<Index,1>&);

Index count_elements_less_than(const Tensor<Index,1>&, const Index&);
Tensor<Index, 1> get_indices_less_than(const Tensor<Index,1>&, const Index&);

Index count_elements_less_than(const Tensor<double,1>&, const double&);
Tensor<Index, 1> get_indices_less_than(const Tensor<double,1>&, const double&);

void delete_indices(Tensor<string,1>&, const Tensor<Index,1>&);
void delete_indices(Tensor<Index,1>&, const Tensor<Index,1>&);
void delete_indices(Tensor<double,1>&, const Tensor<Index,1>&);

Tensor<string, 1> get_first(const Tensor<string,1>&, const Index& );
Tensor<Index, 1> get_first(const Tensor<Index,1>&, const Index& );

void scrub_missing_values(Tensor<type, 2>&, const type&);

Index count_between(Tensor<type,1>&, const type&, const type&);
void set_row(Tensor<type,2>&, const Tensor<type,1>&, const Index&);
Tensor<type,2> filter_column_minimum_maximum(Tensor<type,2>&, const Index&, const type&, const type&);

Tensor<type, 2> kronecker_product(const Tensor<type, 1>&, const Tensor<type, 1>&);

type l1_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l1_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l1_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

type l2_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l2_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l2_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

void sum_diagonal(Tensor<type, 2>&, const type&);

Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>&, const Tensor<type, 1>&);

void fill_submatrix(const Tensor<type, 2>&, const Tensor<Index, 1>& rows_indices, const Tensor<Index, 1>&, type*);
void fill_submatrix(const Tensor<type, 2>&, const Tensor<Index, 1>&, const Tensor<Index, 1>&, Tensor<type, 2>&);

Index count_NAN(const Tensor<type, 1>&);
Index count_NAN(const Tensor<type, 2>&);

Index count_empty_values(const Tensor<string, 1>&);

void check_size(const Tensor<type, 1>&, const Index&, const string&);

void check_dimensions(const Tensor<type, 2>&, const Index&, const Index&, const string&);

void check_columns_number(const Tensor<type, 2>&, const Index&, const string&);
void check_rows_number(const Tensor<type, 2>&, const Index&, const string& );

bool is_less_than(const Tensor<type, 1>&, const type&);
bool contains(const Tensor<type,1>&, const type&);
bool contains(const Tensor<string,1>&, const string&);
bool contains(const Tensor<Index,1>&, const Index&);

Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_vector_matrix(const Tensor<type, 1>&, const Tensor<type, 2>&);
Tensor<type, 2> assemble_matrix_vector(const Tensor<type, 2>&, const Tensor<type, 1>&);
Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>&, const Tensor<type, 2>&);

Tensor<string, 1> assemble_text_vector_vector(const Tensor<string, 1>&, const Tensor<string, 1>&);
string tensor_string_to_text(const Tensor<string,1>&, string&);

Tensor<type, 2> delete_row(const Tensor<type, 2>&, const Index&);

Tensor<Index, 1> push_back(const Tensor<Index, 1>&, const Index&);
Tensor<string, 1> push_back(const Tensor<string, 1>&, const string&);
Tensor<type, 1> push_back(const Tensor<type, 1>&, const type&);

Tensor<string, 1> to_string_tensor(const Tensor<type,1>&);

template<class T, int N>
Tensor<Index, 1> get_dimensions(const Tensor<T, N>&tensor)
{
    Tensor<Index, 1> dims(N);
    memcpy(dims.data(), tensor.dimensions().data(), static_cast<size_t>(N)*sizeof(Index));
    return dims;
}

void print_tensor(const float* vector, const int dims[]);

}

#endif
