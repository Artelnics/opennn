
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
#include "4d_dimensions.h"

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
bool is_nan(const type&);
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

Tensor<type, 2> kronecker_product(const Tensor<type, 1>&, const Tensor<type, 1>&);

type l1_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l1_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l1_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

type l2_norm(const ThreadPoolDevice*, const Tensor<type, 1>&);
void l2_norm_gradient(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 1>&);
void l2_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>&);

type l2_distance(const TensorMap<Tensor<type, 0>>&, const TensorMap<Tensor<type, 0>>&);
type l2_distance(const Tensor<type, 1>&, const Tensor<type, 1>&);
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

Tensor<Index, 1> push_back(const Tensor<Index, 1>&, const Index&);
Tensor<string, 1> push_back(const Tensor<string, 1>&, const string&);
Tensor<type, 1> push_back(const Tensor<type, 1>&, const type&);

Tensor<type, 2> delete_row(const Tensor<type, 2>&, const Index&);

string tensor_string_to_text(const Tensor<string,1>&, string&);
Tensor<string, 1> to_string_tensor(const Tensor<type,1>&);

template<class T, int N>
Tensor<Index, 1> get_dimensions(const Tensor<T, N>&tensor)
{
    Tensor<Index, 1> dims(N);
    memcpy(dims.data(), tensor.dimensions().data(), static_cast<size_t>(N)*sizeof(Index));
    return dims;
}

void print_tensor(const float* vector, const int dims[]);

template<typename InputTensorType, typename KernelTensorType, typename ConvolutionalDimensionType = Eigen::array<Index, 3>>
auto perform_convolution(const InputTensorType& input, const KernelTensorType& kernel, const Index row_stride = 1, const Index column_stride = 1, const ConvolutionalDimensionType convolution_dimension = Eigen::array{Convolutional4dDimensions::channel_index, Convolutional4dDimensions::column_index, Convolutional4dDimensions::row_index})
{
    Eigen::array<Index, 4> strides;
    strides[Convolutional4dDimensions::sample_index] = 1;
    strides[Convolutional4dDimensions::channel_index] = 1;
    strides[Convolutional4dDimensions::row_index] = row_stride;
    strides[Convolutional4dDimensions::column_index] = column_stride;

    return input.convolve(kernel, convolution_dimension).stride(strides);
}

}

#endif
