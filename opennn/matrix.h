/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T R I X   C O N T A I N E R                                                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques, SL                                                                     */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#pragma once

#ifndef __MATRIX_H__
#define __MATRIX_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// OpenNN includes

#include "vector.h"
#include "sparse_matrix.h"

using namespace std;

namespace OpenNN
{


/// This template class defines a matrix for general purpose use.
/// This matrix also implements some mathematical methods which can be useful. 

template <class T>
class Matrix : public vector<T>
{


public:

    // CONSTRUCTORS

    explicit Matrix();

    explicit Matrix(const size_t&, const size_t&);

    explicit Matrix(const size_t&, const size_t&, const T&);

    explicit Matrix(const string&);

    explicit Matrix(const string&, const char&, const bool&);

    Matrix(const Matrix&);

    Matrix(const initializer_list< Vector<T> >&);

    Matrix(const initializer_list< Vector<T> >&, const initializer_list<string>&);

    // DESTRUCTOR

    virtual ~Matrix();

    // ASSIGNMENT OPERATORS

    inline Matrix<T>& operator = (const Matrix<T>&);

    // REFERENCE OPERATORS

    inline T& operator()(const size_t&, const size_t&);    

    inline const T& operator()(const size_t&, const size_t&) const;

    inline T& operator()(const size_t&, const string&);

    inline const T& operator()(const size_t&, const string&) const;

    bool operator == (const Matrix<T>&) const;

    bool operator == (const T&) const;

    bool operator != (const Matrix<T>&) const;

    bool operator != (const T& value) const;

    bool operator >(const Matrix<T>&) const;

    bool operator >(const T& value) const;

    bool operator <(const Matrix<T>&) const;

    bool operator <(const T& value) const;

    bool operator >= (const Matrix<T>&) const;

    bool operator >= (const T&) const;

    bool operator <= (const Matrix<T>&) const;

    bool operator <= (const T&) const;

    bool compare_rows(const size_t&, const Matrix<T>&, const size_t&) const;

    // METHODS

    // Get methods

    const size_t& get_rows_number() const;

    const size_t& get_columns_number() const;

    const Vector<string> get_header() const;
    const string get_header(const size_t&) const;

    size_t get_column_index(const string&) const;
    Vector<size_t> get_column_indices(const Vector<string>&) const;

    Vector<size_t> get_binary_column_indices() const;

    size_t get_empty_row_index() const;

    // Set methods

    void set();

    void set(const size_t&, const size_t&);

    void set(const size_t&, const size_t&, const T&);

    void set(const Matrix<T>&);

    void set(const string&);

    void set(const initializer_list< Vector<T> >&);

    void set_identity(const size_t&);

    void set_rows_number(const size_t&);

    void set_columns_number(const size_t&);

    void set_header(const Vector<string>&);
    void set_header(const size_t&, const string&);

    void append_header(const string&);

    void tuck_in(const size_t&, const size_t&, const Matrix<T>&);
    void tuck_in(const size_t&, const size_t&, const Vector<T>&);

    size_t count_diagonal_elements() const;

    size_t count_off_diagonal_elements() const;

    // Count methods

    size_t count_equal_to(const T&) const;

    size_t count_equal_to(const size_t&, const T&) const;
    size_t count_equal_to(const size_t&, const Vector<T>&) const;

    size_t count_equal_to(const size_t&, const Vector<T>&,
                          const size_t&, const T&) const;

    size_t count_equal_to(const size_t&, const Vector<T>&,
                          const size_t&, const T&,
                          const size_t&, const T&,
                          const size_t&, const T&) const;

    size_t count_equal_to(const size_t&, const T&, const size_t&, const T&) const;
    size_t count_equal_to(const size_t&, const T&, const size_t&, const T&, const size_t&, const T&, const size_t&, const T&) const;

    size_t count_equal_to(const string&, const T&) const;
    size_t count_equal_to(const string&, const Vector<T>&) const;
    size_t count_equal_to(const string&, const T&, const string&, const T&) const;

    Vector<size_t> count_equal_to_by_rows(const T&) const;
    Vector<double> count_equal_to_by_rows(const T&, const Vector<double>&) const;

    size_t count_not_equal_to(const T&) const;

    size_t count_not_equal_to(const size_t&, const T&) const;
    size_t count_not_equal_to(const size_t&, const Vector<T>&) const;

    size_t count_not_equal_to(const size_t&, const T&, const size_t&, const T&) const;

    size_t count_not_equal_to(const string&, const T&) const;
    size_t count_not_equal_to(const string&, const T&, const string&, const T&) const;

    size_t count_rows_equal_to(const T&) const;
    size_t count_rows_not_equal_to(const T&) const;
    Vector<size_t> get_rows_equal_to(const T&) const;

    size_t count_rows_equal_to(const Vector<size_t>&, const T&) const;

    bool is_row_equal_to(const size_t&, const Vector<size_t>&, const T&) const;

    Matrix<T> get_submatrix(const Vector<size_t>&, const Vector<size_t>&) const;

    Matrix<T> get_submatrix_rows(const Vector<size_t>&) const;
    Matrix<T> get_submatrix_rows(const size_t&, const size_t&) const;

    Matrix<T> get_submatrix_columns(const Vector<size_t>&) const;

    Vector<T> get_row(const size_t&) const;

    Vector<T> get_rows(const size_t&, const size_t&) const;

    Vector<T> get_row(const size_t&, const Vector<size_t>&) const;

    Vector<T> get_column(const size_t&) const;
    Vector<T> get_column(const string&) const;

    Matrix<T> get_columns(const Vector<string>&) const;

    Vector<T> get_column(const size_t&, const Vector<size_t>&) const;

    Matrix<T> get_columns_contain(const string&) const;

    Matrix<T> filter_column_equal_to(const size_t&, const T&) const;
    Matrix<T> filter_column_equal_to(const string&, const T&) const;

    Matrix<T> filter_column_equal_to(const size_t&, const Vector<T>&) const;
    Matrix<T> filter_column_equal_to(const string&, const Vector<T>&) const;

    Matrix<T> filter_column_equal_to(const size_t&, const Vector<T>&,
                                      const size_t&, const T&) const;

    Matrix<T> filter_column_equal_to(const string&, const Vector<T>&,
                                      const string&, const T&) const;

    Matrix<T> filter_column_equal_to(const size_t&, const T&,
                                      const size_t&, const T&,
                                      const size_t&, const T&,
                                      const size_t&, const T&) const;

    Matrix<T> filter_column_equal_to(const string&, const T&,
                                      const string&, const T&,
                                      const string&, const T&,
                                      const string&, const T&) const;

    Matrix<T> filter_column_equal_to(const size_t&, const Vector<T>&,
                                      const size_t&, const T&,
                                      const size_t&, const T&,
                                      const size_t&, const T&) const;

    Matrix<T> filter_column_equal_to(const string&, const Vector<T>&,
                                      const string&, const T&,
                                      const string&, const T&,
                                      const string&, const T&) const;

    Matrix<T> filter_column_not_equal_to(const size_t&, const T&) const;
    Matrix<T> filter_column_not_equal_to(const string&, const T&) const;

    Matrix<T> filter_column_not_equal_to(const string&, const Vector<T>&) const;
    Matrix<T> filter_column_not_equal_to(const size_t&, const Vector<T>&) const;

    Matrix<T> filter_column_less_than(const size_t&, const T&) const;
    Matrix<T> filter_column_less_than(const string&, const T&) const;

    Matrix<T> filter_column_greater_than(const size_t&, const T&) const;
    Matrix<T> filter_column_greater_than(const string&, const T&) const;

    Matrix<T> filter_column_less_than_string(const string&, const double&) const;
    Matrix<T> filter_column_greater_than_string(const string&, const double&) const;

    Matrix<T> filter_minimum_maximum(const size_t&, const T&, const T&) const;
    Matrix<T> filter_minimum_maximum(const string&, const T&, const T&) const;

    Matrix<T> filter_extreme_values(const size_t&, const double&, const double&) const;
    Matrix<T> filter_extreme_values(const string&, const double&, const double&) const;

    size_t count_dates(const string&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;
    size_t count_dates(const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;

    Matrix<T> filter_dates(const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;
    Matrix<T> filter_dates(const string&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;

    size_t count_dates_string(const string&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;
    size_t count_dates_string(const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;

    Matrix<T> filter_dates_string(const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;
    Matrix<T> filter_dates_string(const string&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;

    Matrix<T> delete_Tukey_outliers_string(const size_t&, const double&) const;
    Matrix<T> delete_Tukey_outliers_string(const string&, const double&) const;

    Matrix<T> delete_histogram_outliers(const size_t&, const size_t&, const size_t&) const;
    Matrix<T> delete_histogram_outliers(const string&, const size_t&, const size_t&) const;

    Matrix<T> fill_missing_dates_dd_mm_yyyy(const string&, const char&) const;

    size_t get_previous_row_index(const size_t&, const size_t&) const;
    size_t get_previous_row_index(const size_t&, const string&) const;

    size_t count_previous_values(const size_t&, const size_t&) const;

    Matrix<T> get_unique_elements_rows() const;
    Matrix<T> get_unique_elements_rows_frequency() const;

    size_t count_unique(const string&, const T&, const string&) const;

    Vector<size_t> count_unique_elements() const;

    Vector<T> get_diagonal() const;

    void set_row(const size_t&, const Vector<T>&);

    void set_row(const size_t&, const T&);

    void set_submatrix_rows(const size_t&, const Matrix<T>&);

    void set_column(const size_t&, const Vector<T>&, const string& = "");
    void set_column(const string&, const Vector<T>&, const string& = "");

    void set_column(const string&, const T&, const string& = "");
    void set_column(const size_t&, const T&, const string& = "");

    void set_diagonal(const T&);

    void set_diagonal(const Vector<T>&);

    void initialize_diagonal(const size_t&, const T&);

    void initialize_diagonal(const size_t&, const Vector<T>&);

    void sum_diagonal(const T&);

    void sum_diagonal(const Vector<T>&);

    Matrix<T> append_row(const Vector<T>&) const;

    Matrix<T> append_column(const Vector<T>&, const string& = "") const;

    Matrix<T> insert_row(const size_t&, const Vector<T>&) const;
    void insert_row_values(const size_t&, const size_t&, const Vector<T>&);

    Matrix<T> insert_column(const size_t&, const Vector<T>&, const string& = "") const;
    Matrix<T> insert_column(const string&, const Vector<T>&, const string& = "") const;

    Matrix<T> add_columns(const size_t&) const;
    Matrix<T> add_columns_first(const size_t&) const;

    void split_column(const string&, const Vector<string>&, const char& = ',', const string& = "NA");
    
    void split_column(const string&, const string&, const string&, const size_t&, const size_t&);

    void swap_columns(const size_t&, const size_t&);
    void swap_columns(const string&, const string&);

    void merge_columns(const string&, const string&, const string&, const char&);
    void merge_columns(const size_t&, const size_t&, const char&);

    Matrix<T> merge_matrices(const Matrix<T>&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> merge_matrices(const Matrix<T>&, const size_t&, const size_t&) const;

    Matrix<T> right_join(const Matrix<T>&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> right_join(const Matrix<T>&, const size_t&, const size_t&) const;

    Matrix<T> left_join(const Matrix<T>&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> left_join(const Matrix<T>&, const string&, const string&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> left_join(const Matrix<T>&, const size_t&, const size_t&) const;

    T get_first(const size_t&) const;
    T get_first(const string&) const;

    T get_last(const size_t&) const;
    T get_last(const string&) const;

    Matrix<T> delete_row(const size_t&) const;
    Matrix<T> delete_rows(const Vector<size_t>&) const;

    Matrix<T> delete_rows_with_value(const T&) const;

    Matrix<T> delete_rows_equal_to(const T&) const;

    Matrix<T> delete_first_rows(const size_t&) const;
    Matrix<T> get_first_rows(const size_t&) const;

    Matrix<T> delete_last_rows(const size_t&) const;
    Matrix<T> get_last_rows(const size_t&) const;

    Matrix<T> delete_last_columns(const size_t&) const;

    Matrix<T> delete_column(const size_t&) const;
    Matrix<T> delete_columns(const Vector<size_t>&) const;

    Matrix<T> delete_first_columns(const size_t&) const;

    Matrix<T> delete_column(const string&) const;
    Matrix<T> delete_columns(const Vector<string>&) const;

    Matrix<T> delete_columns_name_contains(const Vector<string>&) const;

    Vector<size_t> get_constant_columns_indices() const;

    Matrix<T> delete_constant_rows() const;
    Matrix<T> delete_constant_columns() const;

    Matrix<T> delete_binary_columns() const;

    Matrix<T> delete_binary_columns(const double&) const;

    Matrix<T> assemble_rows(const Matrix<T>&) const;

    Matrix<T> assemble_columns(const Matrix<T>&) const;

    Matrix<T> sort_ascending(const size_t&) const;
    Matrix<T> sort_descending(const size_t&) const;

    Matrix<T> sort_ascending_strings(const size_t&) const;
    Matrix<T> sort_descending_strings(const size_t&) const;

    Matrix<T> sort_ascending_strings_absolute_value(const size_t&) const;
    Matrix<T> sort_descending_strings_absolute_value(const size_t&) const;

    Matrix<T> sort_rank_rows(const Vector<size_t>&) const;

    Matrix<T> sort_columns(const Vector<size_t>&) const;
    Matrix<T> sort_columns(const Vector<string>&) const;

    void initialize(const T&);
    void initialize(const Vector<T>&);
    void replace(const T&, const T&);

    void replace(const string&, const T&, const string&, const T&, const T&);
    void replace(const string&, const T&, const T&, const string&, const T&, const T&);

    void replace_header(const string&, const string&);

    void replace_in_row(const size_t&, const T&, const T&);
    void replace_in_column(const size_t&, const T&, const T&);
    void replace_in_column(const string&, const T&, const T&);

    void replace_substring(const string&, const string&);
    void replace_substring(const size_t&, const string&, const string&);
    void replace_substring(const string&, const string&, const string&);

    void replace_contains(const string&, const string&);
    void replace_contains_in_row(const size_t&, const string&, const string&);

    void replace_special_characters();

    void replace_column_equal_to(const string&, const T&, const T&);
    void replace_column_not_equal_to(const string&, const T&, const T&);
    void replace_column_not_equal_to(const string&, const Vector<T>&, const T&);

    void replace_column_less_than_string(const string&, const double&, const T&);

    void replace_column_contains(const string&, const string&, const string&);
    size_t count_column_contains(const string&, const string&) const;

    Vector<size_t> count_column_occurrences(const T&) const;

    bool has_column_value(const size_t&, const T&) const;

    void randomize_uniform(const double& = -1.0, const double& = 1.0);
    void randomize_uniform(const Vector<double>&, const Vector<double>&);
    void randomize_uniform(const Matrix<double>&, const Matrix<double>&);

    void randomize_normal(const double& = 0.0, const double& = 1.0);
    void randomize_normal(const Vector<double>&, const Vector<double>&);
    void randomize_normal(const Matrix<double>&, const Matrix<double>&);

    void initialize_identity();

    void initialize_diagonal(const T&);

    void update_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&);
    void update_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&);

    // Mathematical methods

    T calculate_sum() const;

    Vector<int> calculate_rows_sum_int() const;
    Vector<T> calculate_rows_sum() const;
    Vector<T> calculate_columns_sum() const;
    T calculate_column_sum(const size_t&) const;

    Vector<T> calculate_columns_mean() const;

    void sum_row(const size_t&, const Vector<T>&);
    Matrix<T> sum_rows(const Vector<T>&) const;
    Matrix<T> subtract_rows(const Vector<T>&) const;
    Matrix<T> multiply_rows(const Vector<T>&) const;
    Vector<Matrix<T>> multiply_rows(const Matrix<T>&) const;

    double calculate_trace() const;

    Matrix<double> calculate_softmax() const;

    Vector<double> calculate_mean() const;
    double calculate_mean(const size_t&) const;

    Vector<double> calculate_mean(const Vector<size_t>&) const;

    Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector<double> calculate_mean_monthly_occurrences() const;

    void check_time_column(const size_t&, const size_t&) const;
    void check_time_column(const string&, const size_t&) const;

    Vector<double> calculate_mean_missing_values(const Vector< Vector<size_t> >&) const;
    Matrix<double> calculate_time_series_mean_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const;
    Matrix<double> calculate_time_series_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Vector<double> > calculate_mean_standard_deviation() const;

    Vector< Vector<double> > calculate_mean_standard_deviation(const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector<double> calculate_median() const;
    double calculate_median(const size_t&) const;

    Vector<double> calculate_median(const Vector<size_t>&) const;

    Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector<double> calculate_median_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< double > calculate_missing_values_percentage() const;

    Vector<double> calculate_LP_norm(const double& p) const;
    Matrix<double> calculate_LP_norm_gradient(const double& p) const;

    T calculate_minimum() const;

    T calculate_maximum() const;

    T calculate_column_minimum(const size_t&) const;

    T calculate_column_maximum(const size_t&) const;

    Vector<T> calculate_means_integers() const;
    Vector<T> calculate_means_integers(const Vector<size_t>&) const;

    Vector<T> calculate_means_integers_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<T> calculate_means_integers_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector<T> calculate_means_binary() const;
    Vector<T> calculate_means_binary_column() const;
    Vector<T> calculate_means_binary_columns() const;

    Vector<T> calculate_means_binary_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<T> calculate_means_binary_column_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<T> calculate_means_binary_columns_missing_values(const Vector< Vector<size_t> >&) const;

    Vector<T> calculate_means_continiuous(const size_t& = 10) const;
    Vector<T> calculate_means_continiuous_missing_values(const Vector< Vector<size_t> >&, const size_t& = 10) const;

    Vector< Vector<T> > calculate_minimum_maximum() const;

    Vector< Vector<T> > calculate_minimum_maximum(const Vector<size_t>&) const;

    Vector< Vector<T> > calculate_minimum_maximum(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_statistics() const;

    Vector< Statistics<T> > calculate_statistics(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_statistics(const Vector< Vector<size_t> >&, const Vector<size_t>&) const;

    Vector< Vector<T> > calculate_columns_minimums_maximums(const Vector< Vector<size_t> >&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const;

    Vector< Statistics<T> > calculate_columns_statistics_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >) const;

    Vector< Statistics<T> > calculate_rows_statistics(const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_rows_statistics_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Statistics<T> > calculate_columns_statistics(const Vector<size_t>&) const;

    Vector<T> calculate_rows_means(const Vector<size_t>& = Vector<size_t>()) const;

    Vector<T> calculate_columns_minimums(const Vector<size_t>& = Vector<size_t>()) const;

    Vector<T> calculate_columns_maximums(const Vector<size_t>& = Vector<size_t>()) const;

    Vector< Vector<double> > calculate_shape_parameters() const;

    Vector< Vector<double> > calculate_shape_parameters_missing_values(const Vector<Vector<size_t> > &) const;

    Vector< Vector<double> > calculate_shape_parameters(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_rows_shape_parameters(const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_rows_shape_parameters_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Vector<double> > calculate_columns_shape_parameters(const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_columns_shape_parameters_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Vector<double> > calculate_box_plots(const Vector< Vector<size_t> >&, const Vector<size_t>&) const;

    Matrix<double> calculate_covariance_matrix() const;

    Vector< Histogram<T> > calculate_histograms(const size_t& = 10) const;

    Vector< Histogram<T> > calculate_histograms_missing_values(const Vector< Vector<size_t> >&, const size_t& = 10) const;

    Matrix<size_t> calculate_less_than_indices(const T&) const;

    Matrix<size_t> calculate_greater_than_indices(const T&) const;

    Matrix<T> calculate_competitive() const;
    Matrix<T> calculate_softmax_rows() const;
    Matrix<T> calculate_softmax_columns() const;
    Matrix<T> calculate_normalized_columns() const;
    Matrix<T> calculate_scaled_minimum_maximum_0_1_columns() const;
    Matrix<T> calculate_scaled_mean_standard_deviation_columns() const;

    Matrix<T> calculate_reverse_columns() const;

    void delete_trends(const Vector< LinearRegressionParameters<double> >&, const size_t&);

    void delete_trends_missing_values(const Vector< LinearRegressionParameters<double> >&, const size_t&, const Vector< Vector<size_t> >&);

    void delete_inputs_trends_missing_values(const Vector< LinearRegressionParameters<double> >&, const size_t&,
                                            const Vector<size_t>&, const Vector< Vector<size_t> >&);

    void delete_outputs_trends_missing_values(const Vector< LinearRegressionParameters<double> >&, const size_t&,
                                             const Vector<size_t>&, const Vector< Vector<size_t> >&);

    void scale_mean_standard_deviation(const Vector< Statistics<T> >&);

    Vector< Statistics<T> > scale_mean_standard_deviation();

    void scale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_minimum_maximum(const Vector< Statistics<T> >&);
    void scale_range(const Vector< Statistics<T> >&, const T& minimum, const T& maximum);

    Vector< Statistics<T> > scale_minimum_maximum();
    Vector< Statistics<T> > scale_range(const T&, const T&);

    void scale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_logarithmic(const Vector< Statistics<T> >&);

    Vector< Statistics<T> > scale_logarithmic();

    void scale_rows_logarithmic(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_columns_logarithmic(const Vector<Statistics<T> >&, const Vector<size_t>&);

    void unscale_mean_standard_deviation(const Vector< Statistics<T> >&);

    void unscale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_minimum_maximum(const Vector< Statistics<T> >&);

    void unscale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_logarithmic(const Vector< Statistics<T> >&);

    void unscale_rows_logarithmic(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_columns_logarithmic(const Vector< Statistics<T> >&, const Vector<size_t>&);

    Vector<size_t> calculate_minimal_indices() const;

    Vector<size_t> calculate_minimal_indices_omit(const T&) const;

    Vector<size_t> calculate_maximal_indices() const;

    Vector<size_t> calculate_maximal_indices_omit(const T&) const;

    Vector< Vector<size_t> > calculate_minimal_maximal_indices() const;

    double calculate_sum_squared_error(const Matrix<double>&) const;
    Vector<double> calculate_error_rows(const Matrix<double>&) const;
    Vector<double> calculate_weighted_error_rows(const Matrix<double>&, const double&, const double&) const;
    Vector<double> calculate_sum_squared_error_rows(const Matrix<double>&) const;

    double calculate_cross_entropy_error(const Matrix<double>&) const;

    double calculate_minkowski_error(const Matrix<double>&, const double&) const;

    double calculate_weighted_sum_squared_error(const Matrix<double>&, const double&, const double&) const;

    double calculate_sum_squared_error(const Vector<double>&) const;

    Vector<double> calculate_rows_L2_norm() const;

    Matrix<T> calculate_absolute_value() const;

    Matrix<T> calculate_transpose() const;

    T calculate_determinant() const;

    Matrix<T> calculate_cofactor() const;

    Matrix<T> calculate_inverse() const;

    Matrix<T> calculate_LU_inverse() const;

    Vector<T> solve_LDLT(const Vector<double>&) const;

    double calculate_euclidean_distance(const size_t&, const size_t&) const;

    Vector<double> calculate_euclidean_distance(const Vector<T>&) const;

    Vector<double> calculate_euclidean_distance(const Matrix<T>&) const;

    Vector<double> calculate_euclidean_weighted_distance(const Vector<T>&, const Vector<double>&) const;
    Matrix<double> calculate_euclidean_weighted_distance_matrix(const Vector<T>&, const Vector<double>&) const;

    double calculate_manhattan_distance(const size_t&, const size_t&) const;

    Vector<double> calculate_manhattan_distance(const Vector<T>&) const;

    Vector<double> calculate_manhattan_weighted_distance(const Vector<T>&, const Vector<double>&) const;
    Matrix<double> calculate_manhattan_weighted_distance_matrix(const Vector<T>&, const Vector<double>&) const;

    void divide_by_rows(const Vector<T>&);

    void filter(const T&, const T&);

    Matrix<T> operator + (const T&) const;

    Matrix<T> operator + (const Matrix<T>&) const;

    Matrix<T> operator - (const T& scalar) const;

    Matrix<T> operator - (const Matrix<T>&) const;

    Matrix<T> operator * (const T&) const;

    Matrix<T> operator * (const Matrix<T>&) const;

    Matrix<T> operator / (const T&) const;

    Matrix<T> operator / (const Vector<T>&) const;

    Matrix<T> operator / (const Matrix<T>&) const;

    void operator += (const T& value);

    void operator += (const Matrix<T>& other_matrix);

    void operator -= (const T&);

    void operator -= (const Matrix<T>&);

    void operator *= (const T&);

    void operator *= (const Matrix<T>&);

    void operator /= (const T&);

    void operator /= (const Matrix<T>&);

    Vector<double> dot(const Vector<double>&) const;

    Matrix<double> dot(const Matrix<double>&) const;

    void dot(const Matrix<double>&, const Matrix<double>&);
    void sum_dot(const Matrix<double>&, const Matrix<double>&);

    Matrix<double> calculate_linear_combinations(const Matrix<double>&, const Vector<double>&) const;

    Matrix<double> calculate_eigenvalues() const;
    Matrix<double> calculate_eigenvectors() const;

    void direct(const Vector<T>&, const Vector<T>&);

    Matrix<T> direct(const Matrix<T>&) const;

    bool empty() const;

    bool is_square() const;

    bool is_symmetric() const;

    bool is_antisymmetric() const;

    bool is_diagonal() const;

    bool is_scalar() const;

    bool is_identity() const;

    bool is_binary() const;

    bool is_column_binary(const size_t&) const;

    bool is_column_constant(const size_t&) const;

    bool is_positive() const;

    void convert_time_column_dd_mm_dd_yyyy_hh(const string&, const char&);

    void convert_time_column_yyyy_MM(const string&, const char&);

    void convert_time_column_yyyy_MM_dd_hh_mm_ss(const string&, const char&);

    void convert_time_column_dd_mm_yyyy(const string&, const char&);
    void convert_time_column_mm_dd_yyyy(const string&, const char&);

    void timestamp_to_date(const string&);

    void convert_time_series(const size_t&, const size_t&, const size_t&);
    Matrix<T> get_time_series(const size_t&, const size_t&);

    Matrix<T> calculate_lag_plot_matrix(const size_t&, const size_t&);

    void convert_association();

    void convert_angular_variables_degrees(const size_t&);

    void convert_angular_variables_radians(const size_t&);

    KMeansResults<T> calculate_k_means(const size_t&) const;

    // Bounding methods

    Matrix<T> calculate_lower_bounded(const T&) const;
    Matrix<T> calculate_upper_bounded(const T&) const;

    Matrix<T> calculate_lower_upper_bounded(const Vector<T>&, const Vector<T>&) const;

    // Correlation methods

    Vector<T> calculate_multiple_linear_regression_parameters(const Vector<T>&) const;

    // Serialization methods

    void print() const;

    void load(const string&);
    void load_csv(const string&, const char& = ',', const bool& = true);
    void load_binary(const string&);

    void load_product_strings(const string&, const char& = ',');

    void save(const string&) const;

    void save_binary(const string&) const;

    void save_csv(const string&, const char& = ',',  const Vector<string>& = Vector<string>(), const string& = "Id") const;

    void save_json(const string&, const Vector<string>& = Vector<string>()) const;

    void parse(const string&);

    string matrix_to_string(const char& = ' ') const;

    Matrix<size_t> to_size_t_matrix() const;
    Matrix<double> to_double_matrix() const;
    Matrix<string> to_string_matrix(const size_t& = 3) const;
    SparseMatrix<T> to_sparse_matrix() const;

    Matrix<double> bool_to_double(const double& exception_value = -999) const;

    Matrix<size_t> string_to_size_t(const size_t& exception_value = 999) const;
    Matrix<double> string_to_double(const double& exception_value = -999) const;

    vector<T> to_std_vector() const;

    Vector<T> to_vector() const;
    Vector<T> rows_to_vector(const char&) const;

    Vector< Vector<T> > to_vector_of_vectors() const;

    void print_preview() const;

    double calculate_logistic_function(const Vector<double>&, const Vector<T>&) const;
    Vector<double> calculate_logistic_error_gradient(const Vector<double>&, const Vector<T>&) const;

    Matrix<T> impute_missing_values_time_series_value(const size_t&, const double&, const T&) const;
    Matrix<T> impute_missing_values_time_series_previous(const size_t&, const double&) const;

    static time_t to_time_t(const size_t&, const size_t&, const size_t&);
    static time_t to_time_t(const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&);

private:

    /// Number of rows in the matrix.

    size_t rows_number;

    /// Number of columns in the matrix.

    size_t columns_number;

    /// Header with names of columns.

    Vector<string> header;

};


// CONSTRUCTORS

/// Default constructor. It creates a matrix with zero rows and zero columns.

template <class T>
Matrix<T>::Matrix() : vector<T>()
{
    set();
}


/// Constructor. It creates a matrix with n rows and m columns, containing n*m copies of the default value for Type.
/// @param new_rows_number Number of rows in matrix.
/// @param new_columns_number Number of columns in matrix.

template <class T>
Matrix<T>::Matrix(const size_t& new_rows_number, const size_t& new_columns_number) : vector<T>(new_rows_number*new_columns_number)
{
   if(new_rows_number == 0 && new_columns_number == 0)
   {
        set();
   }
   else if(new_rows_number == 0)
   {
      set();
   }
   else if(new_columns_number == 0)
   {
      set();
   }
   else
   {
        set(new_rows_number,new_columns_number);
   }
}


/// Constructor. It creates a matrix with n rows and m columns, containing n*m copies of the type value of Type.
/// @param new_rows_number Number of rows in matrix.
/// @param new_columns_number Number of columns in matrix.
/// @param value Value of Type.

template <class T>
Matrix<T>::Matrix(const size_t& new_rows_number, const size_t& new_columns_number, const T& value) : vector<T>(new_rows_number*new_columns_number)
{
   if(new_rows_number == 0 && new_columns_number == 0)
   {
        set();
   }
   else if(new_rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Constructor Matrix(const size_t&, const size_t&, const T&).\n"
             << "Number of rows must be greater than zero.\n";

      throw logic_error(buffer.str());
   }
   else if(new_columns_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Constructor Matrix(const size_t&, const size_t&, const T&).\n"
             << "Number of columns must be greater than zero.\n";

      throw logic_error(buffer.str());
   }
   else
   {
      // Set sizes

      set(new_rows_number,new_columns_number,value);
   }
}


/// File constructor. It creates a matrix which members are loaded from a data file.
/// @param file_name Name of matrix data file.

template <class T>
Matrix<T>::Matrix(const string& file_name) : vector<T>()
{
   rows_number = 0;
   columns_number = 0;

   load(file_name);
}

template <class T>
Matrix<T>::Matrix(const string& file_name, const char& separator, const bool& header) : vector<T>()
{
   rows_number = 0;
   columns_number = 0;

   load_csv(file_name, separator, header);

}


/// Copy constructor. It creates a copy of an existing matrix.
/// @param other_matrix Matrix to be copied.

template <class T>
Matrix<T>::Matrix(const Matrix& other_matrix) : vector<T>(other_matrix.begin(), other_matrix.end())
{
   rows_number = other_matrix.rows_number;
   columns_number = other_matrix.columns_number;

   header = other_matrix.header;
}


template <class T>
Matrix<T>::Matrix(const initializer_list< Vector<T> >& new_columns) : vector<T>()
{
    set(new_columns);
}


template <class T>
Matrix<T>::Matrix(const initializer_list< Vector<T> >& new_columns, const initializer_list<string>& new_header) : vector<T>()
{
    set(new_columns);

    set_header(new_header);
}


/// Destructor.

template <class T>
Matrix<T>::~Matrix()
{
    Vector<string>().swap(header);
    vector<T>().swap(*this);

    rows_number = 0;
    columns_number = 0;
}


// ASSIGNMENT OPERATORS

/// Assignment operator. It assigns to self a copy of an existing matrix.
/// @param other_matrix Matrix to be assigned.

template <class T>
Matrix<T>& Matrix<T>::operator = (const Matrix<T>& other_matrix)
{
    if(other_matrix.rows_number != rows_number || other_matrix.columns_number != columns_number)
    {
        rows_number = other_matrix.rows_number;
        columns_number = other_matrix.columns_number;

        this->clear();

        this->resize(rows_number*columns_number);
    }

    copy(other_matrix.begin(), other_matrix.end(),(*this).begin());

    header = other_matrix.header;

    return(*this);
}


// REFERENCE OPERATORS

/// Reference operator.

/// Returns the element(i,j) of the matrix.
/// @param row Index of row.
/// @param column Index of column.

template <class T>
inline T& Matrix<T>::operator()(const size_t& row, const size_t& column)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator()(const size_t&, const size_t&).\n"
             << "Row index (" << row << ") must be less than number of rows (" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }
   else if(column >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator()(const size_t&, const size_t&).\n"
             << "Column index (" << column << ") must be less than number of columns (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Return matrix element

   return((*this)[rows_number*column+row]);
}


/// Reference operator.

/// Returns the element(i,j) of the matrix.
/// @param row Index of row.
/// @param column Index of column.

template <class T>
inline const T& Matrix<T>::operator()(const size_t& row, const size_t& column) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator()(const size_t&, const size_t&).\n"
             << "Row index (" << row << ") must be less than number of rows (" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }
   else if(column >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator()(const size_t&, const size_t&).\n"
             << "Column index (" << column << ") must be less than number of columns (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Return matrix element

   return((*this)[rows_number*column+row]);
}


template <class T>
inline T& Matrix<T>::operator()(const size_t& row, const string& column_name)
{
   const size_t column = get_column_index(column_name);

   return((*this)(row, column));
}


/// Reference operator.

template <class T>
inline const T& Matrix<T>::operator()(const size_t& row, const string& column_name) const
{
    const size_t column = get_column_index(column_name);

   return((*this)(row, column));
}


/// Equivalent relational operator between this matrix and other matrix.
/// It produces true if all the elements of the two matrices are equal, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator == (const Matrix<T>& other_matrix) const
{
   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
       return(false);
   }
   else if(other_columns_number != columns_number)
   {
        return(false);
   }
   else
   {
       for(size_t i = 0; i < this->size(); i++)
       {
            if((*this)[i] != other_matrix[i])
            {
                return(false);
            }
       }
   }

   return(true);
}


/// Equivalent relational operator between this matrix and a Type value.
/// It produces true if all the elements of this matrix are equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator == (const T& value) const
{
   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] != value)
         {
            return(false);
         }
   }

   return(true);
}


/// Not equivalent relational operator between this matrix and other matrix.
/// It produces true if the two matrices have any not equal element, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator != (const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator != (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator != (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] != other_matrix[i])
        {
            return(true);
        }
   }

   return(false);
}


/// Not equivalent relational operator between this matrix and a Type value.
/// It produces true if some element of this matrix is not equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator != (const T& value) const
{
   // Control sentence(if debug)

   for(size_t i = 0; i < this->size(); i++)
   {
     if((*this)[i] != value)
     {
        return(true);
     }
   }

   return(false);
}


/// Greater than relational operator between this matrix and other vector.
/// It produces true if all the elements of this matrix are greater than the corresponding elements of the other matrix,
/// and false otherwise.
/// @param other_matrix matrix to be compared with.

template <class T>
bool Matrix<T>::operator >(const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >(const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >(const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
        if((*this)[i] <= other_matrix[i])
        {
            return(false);
        }
   }

   return(true);
}


/// Greater than relational operator between this matrix and a Type value.
/// It produces true if all the elements of this matrix are greater than the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator >(const T& value) const
{
    for(size_t i = 0; i < this->size(); i++)
    {
        if((*this)[i] <= value)
        {
            return(false);
        }
    }

    return(true);
}


/// Less than relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are less than the corresponding elements of the other matrix,
/// and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator <(const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator <(const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator <(const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] >= other_matrix[i])
         {
           return(false);
         }
   }

   return(true);
}


/// Less than relational operator between this matrix and a Type value.
/// It produces true if all the elements of this matrix are less than the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator <(const T& value) const
{
   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] >= value)
         {
           return(false);
         }
   }

   return(true);
}


/// Greater than or equal to relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are greater than or equal to the corresponding elements of the
/// other matrix, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator >= (const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] < other_matrix[i])
         {
            return(false);
         }
   }

   return(true);
}


/// Greater than or equal to than relational operator between this matrix and a Type value.
/// It produces true if all the elements of this matrix are greater than or equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator >= (const T& value) const
{
   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] < value)
         {
            return(false);
         }
   }

   return(true);
}


/// Less than or equal to relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are less than or equal to the corresponding elements of the
/// other matrix, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator <= (const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] > other_matrix[i])
         {
            return(false);
         }
   }

   return(true);
}


/// Less than or equal to than relational operator between this matrix and a Type value.
/// It produces true if all the elements of this matrix are less than or equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator <= (const T& value) const
{
   for(size_t i = 0; i < this->size(); i++)
   {
        if((*this)[i] > value)
        {
            return(false);
        }
   }

   return(true);
}


// METHODS

/// Returns the number of rows in the matrix.

template <class T>
const size_t& Matrix<T>::get_rows_number() const
{
   return(rows_number);
}


/// Returns the number of columns in the matrix.

template <class T>
const size_t& Matrix<T>::get_columns_number() const
{
   return(columns_number);
}


template <class T>
const Vector<string> Matrix<T>::get_header() const
{
   return(header);
}


template <class T>
const string Matrix<T>::get_header(const size_t& index) const
{
   return(header[index]);
}


template <class T>
size_t Matrix<T>::get_column_index(const string& column_name) const
{
    if(rows_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t get_column_index(const string&) const.\n"
              << "Number of rows must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    size_t count = 0;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(header[i] == column_name)
        {
            count++;
        }
    }

    if(count == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t get_column_index(const string&) const.\n"
              << "Header does not contain " << column_name << ":\n"
              << header;

       throw logic_error(buffer.str());
    }

    if(count > 1)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t get_column_index(const string&) const.\n"
              << "Multiple occurrences of column name " << column_name << ".\n";

       throw logic_error(buffer.str());
    }

    size_t index;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(header[i] == column_name)
        {
            index = i;
        }
    }

    return(index);
}


template <class T>
Vector<size_t> Matrix<T>::get_column_indices(const Vector<string>& names) const
{
    const size_t size = names.size();

    if(header == "")
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "Vector<size_t> get_column_indices(const Vector<string>&) const.\n"
               << "Header is empty.\n";

        throw logic_error(buffer.str());
    }

    size_t count = 0;

    for(size_t i = 0; i < size; i++)
    {
        for(size_t j = 0; j < header.size(); j++)
        {
            if(names[i] == header[j])
            {
                count++;
                break;
            }
        }
    }

    if(size != count)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "Vector<size_t> get_column_indices(const Vector<string>&) const.\n"
               << "Header does not contain some name.\n";

        buffer << "Header: " << header << endl;
        buffer << "Names: " << names << endl;

        throw logic_error(buffer.str());
    }

    Vector<size_t> indices(size);

    for(size_t i = 0; i < size; i++)
    {
        indices[i] = get_column_index(names[i]);
    }

    return indices;
}


template <class T>
Vector<size_t> Matrix<T>::get_binary_column_indices() const
{
    Vector<size_t> binary_columns;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(is_column_binary(i))
        {
            binary_columns.push_back(i);
        }
    }

    return binary_columns;
}


template <class T>
size_t Matrix<T>::get_empty_row_index() const
{
    for(size_t i = 0; i < rows_number; i++)
    {
        //if((*this)(i, 0) == "")
        {
            return(i);
        }
    }

   return(rows_number-1);
}


/// This method set the numbers of rows and columns of the matrix to zero.

template <class T>
void Matrix<T>::set()
{
   rows_number = 0;
   columns_number = 0;

   Vector<string>().swap(header);

   vector<T>().swap(*this);
}


/// This method set new numbers of rows and columns in the matrix.
/// @param new_rows_number Number of rows.
/// @param new_columns_number Number of columns.

template <class T>
void Matrix<T>::set(const size_t& new_rows_number, const size_t& new_columns_number)
{
   // Control sentence(if debug)

   if(new_rows_number == 0 && new_columns_number == 0)
   {
      set();
   }
   else if(new_rows_number == 0)
   {
      set();
   }
   else if(new_columns_number == 0)
   {
      set();
   }
   else
   {
      rows_number = new_rows_number;
      columns_number = new_columns_number;

      this->resize(rows_number*columns_number);

      header.set(columns_number, "");
   }
}


/// This method set new numbers of rows and columns in the matrix.
/// It also initializes all the matrix elements to a given value.
/// @param new_rows_number Number of rows.
/// @param new_columns_number Number of columns.
/// @param value Initialization value.

template <class T>
void Matrix<T>::set(const size_t& new_rows_number, const size_t& new_columns_number, const T& value)
{
   if(new_rows_number == 0 && new_columns_number == 0)
   {
      set();
   }
   else if(new_rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void set(const size_t&, const size_t&, const T&) method.\n"
             << "Number of rows must be greater than zero.\n";

      throw logic_error(buffer.str());
   }
   else if(new_columns_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
            << "void set(const size_t&, const size_t&, const T&) method.\n"
            << "Number of columns must be greater than zero.\n";

      throw logic_error(buffer.str());
   }
   else
   {
      set(new_rows_number, new_columns_number);
      initialize(value);
   }
}


/// Sets all the members of the matrix to those of another matrix.
/// @param other_matrix Setting matrix.

template <class T>
void Matrix<T>::set(const Matrix<T>& other_matrix)
{
    rows_number = other_matrix.rows_number;
    columns_number = other_matrix.columns_number;

    set(rows_number,columns_number);

    for(size_t i = 0; i <static_cast<size_t>(this->size()); i++)
    {
       (*this)[i] = other_matrix[i];
    }
}


/// Sets the members of this object by loading them from a data file.
/// @param file_name Name of data file.

template <class T>
void Matrix<T>::set(const string& file_name)
{
   load(file_name);
}


template <class T>
void Matrix<T>::set(const initializer_list< Vector<T> >& columns)
{
    if(columns.size() == 0)
    {
        set();
    }

    const size_t new_columns_number = columns.size();

    const size_t new_rows_number = (*columns.begin()).size();

    if(new_rows_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix(const initializer_list< Vector<T> >& list) constructor.\n"
              << "Size of list vectors must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    set(new_rows_number, new_columns_number);

    for(size_t i = 0;  i < new_columns_number; i++)
    {
         const Vector<T> new_column(*(columns.begin() + i));

         if(new_column.size() != new_rows_number)
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Matrix Template.\n"
                   << "Matrix(const initializer_list< Vector<T> >& list) constructor.\n"
                   << "Size of vector " << i << " (" << new_column.size() << ") must be equal to number of rows (" << new_rows_number << ").\n";

            throw logic_error(buffer.str());
         }

         set_column(i, new_column);
    }

}


/// Sets the matrix to be squared, with elements equal one in the diagonal and zero outside the diagonal.
/// @param new_size New number of rows and columns in this matrix.

template <class T>
void Matrix<T>::set_identity(const size_t& new_size)
{
   set(new_size, new_size);
   initialize_identity();
}


/// Sets a new number of rows in the matrix.
/// @param new_rows_number Number of matrix rows.

template <class T>
void Matrix<T>::set_rows_number(const size_t& new_rows_number)
{
   if(new_rows_number != rows_number)
   {
      set(new_rows_number, columns_number);
   }
}


/// Sets a new number of columns in the matrix.
/// @param new_columns_number Number of matrix columns.

template <class T>
void Matrix<T>::set_columns_number(const size_t& new_columns_number)
{
   if(new_columns_number != columns_number)
   {
      set(rows_number, new_columns_number);
   }
}


template <class T>
void Matrix<T>::set_header(const Vector<string>& new_header)
{
   (*this).header = new_header;
}


template <class T>
void Matrix<T>::set_header(const size_t& index, const string& index_name)
{
    header[index] = index_name;
}


template <class T>
void Matrix<T>::append_header(const string& str)
{
    for(size_t i = 0; i < header.size(); i++)
    {
        header[i].append(str);
    }
}


/// Tuck in another matrix starting from a given position.
/// @param row_position Insertion row position.
/// @param column_position Insertion row position.
/// @param other_matrix Matrix to be inserted.

template <class T>
void Matrix<T>::tuck_in(const size_t& row_position, const size_t& column_position, const Matrix<T>& other_matrix)
{
   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_position + other_rows_number > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void insert(const size_t&, const size_t&, const Matrix<T>&) const method.\n"
             << "Cannot tuck in matrix.\n";

      throw logic_error(buffer.str());
   }

   if(column_position + other_columns_number > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void insert(const size_t&, const size_t&, const Matrix<T>&) const method.\n"
             << "Cannot tuck in matrix.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < other_rows_number; i++)
   {
      for(size_t j = 0; j < other_columns_number; j++)
      {
        (*this)(row_position+i,column_position+j) = other_matrix(i,j);
      }
   }
}


/// Tuck in a vector to a given row of the matrix
/// @param row_position Insertion row position.
/// @param column_position Insertion row position.
/// @param other_vector Vector to be inserted.

template <class T>
void Matrix<T>::tuck_in(const size_t& row_position, const size_t& column_position, const Vector<T>& other_vector)
{
//   const size_t other_rows_number = other_vector.get_rows_number();
   const size_t other_columns_number = other_vector.size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_position  > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void insert(const size_t&, const size_t&, const Vector<T>&) const method.\n"
             << "Cannot tuck in vector.\n";

      throw logic_error(buffer.str());
   }

   if(column_position + other_columns_number > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void insert(const size_t&, const size_t&, const Vector<T>&) const method.\n"
             << "Cannot tuck in vector.\n";

      throw logic_error(buffer.str());
   }

   #endif

//   for(size_t i = 0; i < other_rows_number; i++)
//   {
     for(size_t j = 0; j < other_columns_number; j++)
     {
       (*this)(row_position,column_position+j) = other_vector[j];
     }
//   }
}

/// Returns the number of elements in the diagonal which are not zero.
/// This method is only defined for square matrices.

template <class T>
size_t Matrix<T>::count_diagonal_elements() const
{
    #ifdef __OPENNN_DEBUG__

    if(!is_square())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t count_diagonal_elements() const method.\n"
              << "The matrix is not square.\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,i) != 0)
        {
            count++;
        }
    }

    return(count);
}


/// Returns the number of elements outside the diagonal which are not zero.
/// This method is only defined for square matrices.

template <class T>
size_t Matrix<T>::count_off_diagonal_elements() const
{
    #ifdef __OPENNN_DEBUG__

    if(!is_square())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t count_off_diagonal_elements() const method.\n"
              << "The matrix is not square.\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(i != j &&(*this)(i,j) != 0)
            {
                count++;
            }
        }
    }

    return(count);
}


template <class T>
size_t Matrix<T>::count_equal_to(const T& value) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i, j) == value)
            {
                count++;
            }
        }
    }

    return count;
}


/// Returns the number of elements in a given column that are equal to a given value.
/// @param column_index Index of column.
/// @param value Value to find.

template <class T>
size_t Matrix<T>::count_equal_to(const size_t& column_index, const T& value) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_index) == value)
        {
            count++;
        }
    }

    return(count);
}


/// Returns the number of elements in a given column that are equal to a given set of values.
/// @param column_index Index of column.
/// @param values Vector of values.

template <class T>
size_t Matrix<T>::count_equal_to(const size_t& column_index, const Vector<T>& values) const
{
    const size_t values_size = values.size();

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < values_size; j++)
        {
            if((*this)(i, column_index) == values[j])
            {
                count++;

                break;
            }
        }
    }

    return(count);
}


/// Looks for elements that have a given set of values in one column and a a given value in another column,
/// and returns the number of elements found.
/// @param column_index Index of a column.
/// @param values Vector of values to be found in the column above.
/// @param column_index Index of another column.
/// @param value Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_equal_to(const size_t& column_1_index, const Vector<T>& values_1,
                                 const size_t& column_2_index, const T& value_2) const
{
    const size_t values_1_size = values_1.size();

    size_t count = 0;

    T matrix_element;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_2_index) == value_2)
        {
            matrix_element = (*this)(i, column_1_index);

            for(size_t j = 0; j < values_1_size; j++)
            {
                if(matrix_element == values_1[j])
                {
                    count++;

                    break;
                }
            }
        }
   }

    return(count);
}


/// Looks for elements that have a given set of values in one column,
/// and given values in another three columns.
/// It returns the number of elements found.
/// @param column_1_index Index of a column.
/// @param values_1 Vector of values to be found in the column above.
/// @param column_2_index Index of another column.
/// @param value_2 Value to be found in the column above.
/// @param column_3_index Index of another column.
/// @param value_3 Value to be found in the column above.
/// @param column_4_index Index of another column.
/// @param value_4 Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_equal_to(const size_t& column_1_index, const Vector<T>& values_1,
                                 const size_t& column_2_index, const T& value_2,
                                 const size_t& column_3_index, const T& value_3,
                                 const size_t& column_4_index, const T& value_4) const
{
    const size_t values_1_size = values_1.size();

    size_t count = 0;

    T matrix_element;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_2_index) == value_2
        &&(*this)(i, column_3_index) == value_3
        &&(*this)(i, column_4_index) == value_4)
        {
            matrix_element = (*this)(i, column_1_index);

            for(size_t j = 0; j < values_1_size; j++)
            {
                if(matrix_element == values_1[j])
                {
                    count++;

                    break;
                }
            }
        }
   }

    return(count);
}


/// Looks for elements that have a given value in one column,
/// and another given value in another column.
/// It returns the number of elements found.
/// @param column_1_index Index of a column.
/// @param value_1 Value to be found in the column above.
/// @param column_2_index Index of another column.
/// @param value_2 Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_equal_to(const size_t& column_1_index, const T& value_1,
                                 const size_t& column_2_index, const T& value_2) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_1_index) == value_1
        &&(*this)(i, column_2_index) == value_2)
        {
            count++;
        }
    }

    return(count);
}


/// Looks for elements that have given values in 4 different columns.
/// It returns the number of elements found.
/// @param column_1_index Index of a column.
/// @param value_1 Value to be found in the column above.
/// @param column_2_index Index of another column.
/// @param value_2 Value to be found in the column above.
/// @param column_3_index Index of another column.
/// @param value_3 Value to be found in the column above.
/// @param column_4_index Index of another column.
/// @param value_4 Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_equal_to(const size_t& column_1_index, const T& value_1,
                                 const size_t& column_2_index, const T& value_2,
                                 const size_t& column_3_index, const T& value_3,
                                 const size_t& column_4_index, const T& value_4) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_1_index) == value_1
        &&(*this)(i, column_2_index) == value_2
        &&(*this)(i, column_3_index) == value_3
        &&(*this)(i, column_4_index) == value_4)
        {
            count++;
        }
    }

    return(count);
}


template <class T>
size_t Matrix<T>::count_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(count_equal_to(column_index, value));
}


template <class T>
size_t Matrix<T>::count_equal_to(const string& column_name, const Vector<T>& values) const
{
    const size_t column_index = get_column_index(column_name);

    return(count_equal_to(column_index, values));
}


template <class T>
size_t Matrix<T>::count_equal_to(const string& column_1_name, const T& value_1,
                                 const string& column_2_name, const T& value_2) const
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);

    return(count_equal_to(column_1_index, value_1, column_2_index, value_2));
}


template <class T>
Vector<size_t> Matrix<T>::count_equal_to_by_rows(const T& value) const
{
    Vector<size_t> count_by_rows(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        count_by_rows[i] = this->get_row(i).count_equal_to(value);
    }

    return count_by_rows;
}


template <class T>
Vector<double> Matrix<T>::count_equal_to_by_rows(const T& value, const Vector<double>& weights) const
{
    Vector<double> count_by_rows(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        count_by_rows[i] = this->get_row(i).count_equal_to(value, weights);
    }

    return count_by_rows;
}


template <class T>
size_t Matrix<T>::count_not_equal_to(const T& value) const
{
    const size_t this_size = this->size();

    const size_t count = count(this->begin(), this->end(), value);

    return(this_size-count);
}


template <class T>
size_t Matrix<T>::count_not_equal_to(const size_t& column_index, const T& value) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_index) != value)
        {
            count++;
        }
    }

    return(count);
}


template <class T>
size_t Matrix<T>::count_not_equal_to(const size_t& column_index, const Vector<T>& values) const
{
    const size_t values_size = values.size();

    size_t index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        size_t count = 0;

        for(size_t j = 0; j < values_size; j++)
        {
            if((*this)(i, column_index) != values[j])
            {
                count++;
            }
        }

        if(count == values.size())
        {
            index++;
        }
    }

    return(index);
}


template <class T>
size_t Matrix<T>::count_not_equal_to(const size_t& column_1_index, const T& value_1,
                                     const size_t& column_2_index, const T& value_2) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_1_index) == value_1 &&(*this)(i, column_2_index) != value_2)
        {
            count++;
        }
    }

    return(count);
}


template <class T>
size_t Matrix<T>::count_not_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(count_not_equal_to(column_index, value));
}


template <class T>
size_t Matrix<T>::count_not_equal_to(const string& column_1_name, const T& value_1,
                                     const string& column_2_name, const T& value_2) const
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);

    return(count_not_equal_to(column_1_index, value_1, column_2_index, value_2));
}


template <class T>
size_t Matrix<T>::count_rows_equal_to(const T& value) const
{
    size_t count = rows_number;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) != value)
            {
                count--;

                break;
            }
        }
    }

    return count;
}


template <class T>
size_t Matrix<T>::count_rows_not_equal_to(const T& value) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(get_row(i) != value) count++;
    }

    return count;
}


template <class T>
Vector<size_t> Matrix<T>::get_rows_equal_to(const T& value) const
{
    Vector<size_t> indices;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(get_row(i) == value)
        {
            indices.push_back(i);
        }
    }

    return indices;
}


template <class T>
size_t Matrix<T>::count_rows_equal_to(const Vector<size_t>& column_indices, const T& value) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(is_row_equal_to(i, column_indices, value))
        {
            count++;
        }
    }

    return(count);
}


template <class T>
bool Matrix<T>::is_row_equal_to(const size_t& row_index, const Vector<size_t>& column_indices, const T& value) const
{
    const size_t column_indices_size = column_indices.size();

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if((*this)(row_index,column_indices[i]) != value)
        {
            return(false);
        }
    }

    return(true);
}


/// Returns a matrix with the values of given rows and columns from this matrix.
/// @param row_indices Indices of matrix rows.
/// @param column_indices Indices of matrix columns.

template <class T>
Matrix<T> Matrix<T>::get_submatrix(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t column_indices_size = column_indices.size();

   Matrix<T> sub_matrix(row_indices_size, column_indices_size);

   size_t row_index;
   size_t column_index;

   for(size_t i = 0; i < row_indices_size; i++)
   {
      row_index = row_indices[i];

      for(size_t j = 0; j < column_indices_size; j++)
      {
         column_index = column_indices[j];
         sub_matrix(i,j) = (*this)(row_index,column_index);
      }
   }

   return(sub_matrix);
}


/// Returns a submatrix with the values of given rows from this matrix.
/// @param row_indices Indices of matrix rows.

template <class T>
Matrix<T> Matrix<T>::get_submatrix_rows(const Vector<size_t>& row_indices) const
{
   const size_t row_indices_size = row_indices.size();

   Matrix<T> sub_matrix(row_indices_size, columns_number);

   size_t row_index;

   for(size_t i = 0; i < row_indices_size; i++)
   {
      row_index = row_indices[i];

      for(size_t j = 0; j < columns_number; j++)
      {
         sub_matrix(i,j) = (*this)(row_index,j);
      }
   }

   sub_matrix.set_header(get_header());

   return(sub_matrix);
}


template <class T>
Matrix<T> Matrix<T>::get_submatrix_rows(const size_t& first, const size_t& last) const
{
    const Vector<size_t> indices(first, 1, last);

    return get_submatrix_rows(indices);
}


/// Returns a submatrix with the values of given columns from this matrix.
/// @param column_indices Indices of matrix columns.

template <class T>
Matrix<T> Matrix<T>::get_submatrix_columns(const Vector<size_t>& column_indices) const
{
    const size_t column_indices_size = column_indices.size();

    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if(column_indices[i] >= columns_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: Matrix Template.\n"
                  << "Matrix<T> get_submatrix_columns(const Vector<size_t>&) const method.\n"
                  << "Column index(" << i << ") must be less than number of columns(" << columns_number << ").\n";

           throw logic_error(buffer.str());
        }
    }

    #endif

   Matrix<T> sub_matrix(rows_number, column_indices_size);

   size_t column_index;

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < column_indices_size; j++)
      {
         column_index = column_indices[j];

         sub_matrix(i,j) = (*this)(i,column_index);
      }
   }

   if(header != "")
   {
       const Vector<string> sub_header = header.get_subvector(column_indices);

       sub_matrix.set_header(sub_header);
   }

   return(sub_matrix);
}


/// Returns the row i of the matrix.
/// @param i Index of row.

template <class T>
Vector<T> Matrix<T>::get_row(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(i >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_row(const size_t&) const method.\n"
             << "Row index (" << i << ") must be less than number of rows (" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<T> row(columns_number);

   for(size_t j = 0; j < columns_number; j++)
   {
      row[j] = (*this)(i,j);
   }

   return(row);
}


template <class T>
Vector<T> Matrix<T>::get_rows(const size_t& first_index, const size_t& last_index) const
{
    #ifdef __OPENNN_DEBUG__

    if(last_index > rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Vector<T> get_rows(const size_t&, const size_t&) const method.\n"
              << "Last index(" << last_index << ") must be less than number of rows(" << rows_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    Vector<double> new_row;

    for(size_t i = first_index-1; i < last_index; i++)
    {
        new_row = new_row.assemble(get_row(i));
    }

    return new_row;
}


/// Returns the row i of the matrix, but only the elements specified by given indices.
/// @param row_index Index of row.
/// @param column_indices Column indices of row.

template <class T>
Vector<T> Matrix<T>::get_row(const size_t& row_index, const Vector<size_t>& column_indices) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_row(const size_t&, const Vector<size_t>&) const method.\n"
             << "Row index (" << row_index << ") must be less than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t size = column_indices.size();

   Vector<T> row(size);

   for(size_t i = 0; i < size; i++)
   {
      row[i] = (*this)(row_index,column_indices[i]);
   }

   return(row);
}


template <class T>
size_t Matrix<T>::get_previous_row_index(const size_t& row_index, const size_t& column_index) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(row_index >= rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t Matrix<T>::get_previous_row_index(const size_t&, const size_t&) const method.\n"
              << "Row index (" << row_index << ") must be less than number of rows (" << rows_number << ").\n";

       throw logic_error(buffer.str());
    }

    if(column_index >= columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t Matrix<T>::get_previous_row_index(const size_t&, const size_t&) const method.\n"
              << "Column index (" << column_index << ") must be less than number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    const T value = (*this)(row_index, column_index);

    for(int i = static_cast<int>(row_index)-1; i >= 0; --i)
    {
        if((*(this))(i, column_index) == value)
        {
            return(static_cast<size_t>(i));
        }
    }

    return(row_index);
}


template <class T>
size_t Matrix<T>::get_previous_row_index(const size_t& row_index, const string& column_name) const
{
    const size_t column_index = get_column_index(column_name);

    return(get_previous_row_index(row_index, column_index));

}


template <class T>
size_t Matrix<T>::count_previous_values(const size_t& row_index, const size_t& column_index) const
{
    const T value = (*this)(row_index, column_index);

    size_t count = 0;

    for(int i = static_cast<int>(row_index)-1; i >= 0; --i)
    {
        if((*this)(i, column_index) == value)
        {
            count++;
        }
    }

    return(count);
}


/// Returns the column j of the matrix.
/// @param j Index of column.

template <class T>
Vector<T> Matrix<T>::get_column(const size_t& j) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(j >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_column(const size_t&) const method.\n"
             << "Column index(" << j << ") must be less than number of columns(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<T> column(rows_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      column[i] = (*this)(i,j);
   }

   return(column);
}


template <class T>
Vector<T> Matrix<T>::get_column(const string& column_name) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(this->empty())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_column(const string&) const method.\n"
             << "Matrix is empty.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t column_index = get_column_index(column_name);

   return(get_column(column_index));
}


template <class T>
Matrix<T> Matrix<T>::get_columns(const Vector<string>& column_names) const
{
    const Vector<size_t> indices = get_column_indices(column_names);

    return(get_submatrix_columns(indices));
}


/// Returns the column j of the matrix, but only those elements specified by given indices.
/// @param column_index Index of column.
/// @param row_indices Row indices of column.

template <class T>
Vector<T> Matrix<T>::get_column(const size_t& column_index, const Vector<size_t>& row_indices) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_column(const size_t&) const method.\n"
             << "Column index(" << column_index << ") must be less than number of rows(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t size = row_indices.size();

   Vector<T> column(size);

   for(size_t i = 0; i < size; i++)
   {
      column[i] = (*this)(row_indices[i],column_index);
   }

   return(column);
}


template <class T>
Matrix<T> Matrix<T>::get_columns_contain(const string& find_what) const
{
    Vector<size_t> indices;

    for(size_t i = 0; i < columns_number; i++)
    {
        if((*this)(0,i).find(find_what) != string::npos)
        {
             indices.push_back(i);
        }
    }

    return get_submatrix_columns(indices);

}


// @todo Improve this method

template <class T>
Matrix<T> Matrix<T>::get_unique_elements_rows() const
{
    Matrix<string> unique(1,columns_number);

    unique.insert_row(0, get_row(0));

    #pragma omp parallel for

    for(int i = 1; i < rows_number; i++)
    {
        const Vector<string> row = (*this).get_row(i);

        size_t index = 0;

        Vector<string> unique_row;

        for(size_t j = 0; j < unique.get_rows_number(); j++)
        {
            unique_row = unique.get_row(j);

            if(unique_row == row)
            {
                break;
            }
            else
            {
                index++;
            }
        }

        if(index == unique.get_rows_number())
        {
            unique.insert_row(0,row);
        }
    }

    return(unique);
}


template <class T>
Matrix<T> Matrix<T>::get_unique_elements_rows_frequency() const
{
    Matrix<string> unique(1,columns_number);

    Vector<string> row;

    unique.set_row(0,(*this).get_row(0));

    for(size_t i = 1; i < rows_number; i++)
    {
        const size_t unique_rows_number = unique.get_rows_number();

        size_t index = 0;

        for(size_t j = 0; j < unique_rows_number; j++)
        {
            if(compare_rows(i, unique, j))
            {
                break;
            }
            else
            {
                index++;
            }
        }

        if(index == unique_rows_number)
        {
            row = get_row(i);
            unique.insert_row(0,row);
        }
    }

    Vector<string> frequency(unique.get_rows_number());

    const size_t unique_rows_number = unique.get_rows_number();

    for(size_t i = 0; i < unique_rows_number; i++)
    {
        size_t index = 0;

        for(size_t j = 0; j < rows_number; j++)
        {
            if(unique.compare_rows(i,(*this),j))
            {
                index++;
            }
        }

        frequency[i] = to_string(index);
    }

    return(unique.insert_column(0,frequency));
}


template <class T>
size_t Matrix<T>::count_unique(const string& column_1_name, const T& value_1, const string& column_2_name) const
{
    const size_t column_index_1 = get_column_index(column_1_name);
    const size_t column_index_2 = get_column_index(column_2_name);

    Vector<T> elements;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_index_1) == value_1)
        {
            if(!elements.contains((*this)(i, column_index_2)))
            {
                elements.push_back((*this)(i, column_index_2));
            }
        }
    }

    return(elements.get_unique_elements().size());
}


template <class T>
Vector<size_t> Matrix<T>::count_unique_elements() const
{
    Vector<size_t> unique_elements;

    for(size_t i = 0; i < columns_number; i++)
    {
        unique_elements.push_back((*this).get_column(i).get_unique_elements().size() - 1);
    }

    return(unique_elements);
}


/// Returns the diagonal of the matrix.

template <class T>
Vector<T> Matrix<T>::get_diagonal() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_diagonal() const method.\n"
             << "Matrix must be squared.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<T> diagonal(rows_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      diagonal[i] = (*this)(i,i);
   }

   return(diagonal);
}


/// Sets new values of a single row in the matrix.
/// @param row_index Index of row.
/// @param new_row New values of single row.

template <class T>
void Matrix<T>::set_row(const size_t& row_index, const Vector<T>& new_row)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const Vector<T>&) method.\n"
             << "Index must be less than number of rows.\n";

      throw logic_error(buffer.str());
   }

   const size_t size = new_row.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const Vector<T>&) method.\n"
             << "Size(" << size << ") must be equal to number of columns(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set new row

   for(size_t i = 0; i < columns_number; i++)
   {
     (*this)(row_index,i) = new_row[i];
   }
}


/// Sets a new value of a single row in the matrix.
/// @param row_index Index of row.
/// @param value New value of single row.

template <class T>
void Matrix<T>::set_row(const size_t& row_index, const T& value)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const T&) method.\n"
             << "Index must be less than number of rows.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set new row

   for(size_t i = 0; i < columns_number; i++)
   {
     (*this)(row_index,i) = value;
   }
}


/// Sets a new value of a single row in the matrix.
/// @param row_index Index of row.
/// @param value New value of single row.

template <class T>
void Matrix<T>::set_submatrix_rows(const size_t& row_index, const Matrix<T>& submatrix)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_index+submatrix.get_rows_number()>= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const T&) method.\n"
             << "Submatrix doesn't fix in this matrix.\n";

      throw logic_error(buffer.str());
   }
   if(submatrix.get_columns_number() != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const T&) method.\n"
             << "Submatrix columns number is different than matrix columns number.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set new row

   for(size_t i = 0; i < submatrix.get_rows_number(); i++)
   {
     (*this).set_row(row_index+i,submatrix.get_row(i));
   }

   if(header == "" && row_index == 0)
   {
       set_header(submatrix.get_header());
   }
}


template <class T>
void Matrix<T>::set_column(const size_t& column_index, const Vector<T>& new_column, const string& new_name)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_column(const size_t&, const Vector<T>&).\n"
             << "Index(" << column_index << ") must be less than number of columns(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   const size_t size = new_column.size();

   if(size != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_column(const size_t&, const Vector<T>&).\n"
             << "Size must be equal to number of rows.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
     (*this)(i,column_index) = new_column[i];
   }

   header[column_index] = new_name;
}


template <class T>
void Matrix<T>::set_column(const string& column_name, const Vector<T>& new_column, const string& new_name)
{
    const size_t column_index = get_column_index(column_name);

    this->set_column(column_index, new_column, new_name);
}


template <class T>
void Matrix<T>::set_column(const string& column_name, const T& value, const string& new_name)
{
    const size_t column_index = get_column_index(column_name);

    set_column(column_index, value, new_name);
}


template <class T>
void Matrix<T>::set_column(const size_t& column_index, const T& value, const string& new_name)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_column(const size_t&, const T&).\n"
             << "Index must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
     (*this)(i,column_index) = value;
   }

   header[column_index] = new_name;
}


/// Sets a new value for the diagonal elements in the matrix.
/// The matrix must be square.
/// @param new_diagonal New value of diagonal.

template <class T>
void Matrix<T>::set_diagonal(const T& new_diagonal)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_diagonal(const T&).\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
     (*this)(i,i) = new_diagonal;
   }
}


/// Sets new values of the diagonal in the matrix.
/// The matrix must be square.
/// @param new_diagonal New values of diagonal.

template <class T>
void Matrix<T>::set_diagonal(const Vector<T>& new_diagonal)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_diagonal(const Vector<T>&) const.\n"
             << "Matrix is not square.\n";

      throw logic_error(buffer.str());
   }

   const size_t size = new_diagonal.size();

   if(size != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_diagonal(const Vector<T>&) const.\n"
             << "Size of diagonal(" << size << ") is not equal to size of matrix (" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
     (*this)(i,i) = new_diagonal[i];
   }
}


/// Sets this matrix to be diagonal.
/// A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero.
/// It also initializes the elements on the main diagonal to a unique given value.
/// @param new_size Number of rows and colums in the matrix.
/// @param new_value Value of all the elements in the main diagonal.

template <class T>
void Matrix<T>::initialize_diagonal(const size_t& new_size, const T& new_value)
{
   set(new_size, new_size, 0.0);
   set_diagonal(new_value);
}


/// Sets this matrix to be diagonal.
/// A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero.
/// It also initializes the elements on the main diagonal to given values.
/// @param new_size Number of rows and colums in the matrix.
/// @param new_values Values of the elements in the main diagonal.

template <class T>
void Matrix<T>::initialize_diagonal(const size_t& new_size, const Vector<T>& new_values)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t new_values_size = new_values.size();

   if(new_values_size != new_size)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "initialize_diagonal(const size_t&, const size_t&) const.\n"
             << "Size of new values is not equal to size of square matrix.\n";

      throw logic_error(buffer.str());
   }

   #endif

   set(new_size, new_size, 0.0);
   set_diagonal(new_values);
}


/// This method sums a new value to the diagonal elements in the matrix.
/// The matrix must be square.
/// @param value New summing value.

template <class T>
void Matrix<T>::sum_diagonal(const T& value)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "sum_diagonal(const T&).\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,i) += value;
   }
}


/// This method sums new values to the diagonal in the matrix.
/// The matrix must be square.
/// @param new_summing_values Vector of summing values.

template <class T>
void Matrix<T>::sum_diagonal(const Vector<T>& new_summing_values)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "sum_diagonal(const Vector<T>&).\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   const size_t size = new_summing_values.size();

   if(size != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "sum_diagonal(const Vector<T>&).\n"
             << "Size must be equal to number of rows.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,i) += new_summing_values[i];
   }
}


/// This method appends a new row to the matrix.
/// The size of the row vector must be equal to the number of columns of the matrix.
/// Note that resizing is necessary here and therefore this method can be very inefficient.
/// @param new_row Row to be appended.

template <class T>
Matrix<T> Matrix<T>::append_row(const Vector<T>& new_row) const
{
    #ifdef __OPENNN_DEBUG__

    const size_t size = new_row.size();

    if(size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "append_row(const Vector<T>&) const.\n"
              << "Size(" << size << ") must be equal to number of columns(" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<T> copy;

    if(this->empty())
    {
        copy.set(1,new_row.size());
        copy.set_row(0,new_row);

        return copy;
    }

    copy.set(rows_number+1, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
           copy(i,j) = (*this)(i,j);
        }
    }

    copy.set_row(rows_number, new_row);

    if(header != "") copy.set_header(header);

    return copy;
}


template <class T>
Matrix<T> Matrix<T>::append_column(const Vector<T>& new_column, const string& new_name) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = new_column.size();

   if(size != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "append_column(const Vector<T>&) const.\n"
             << "Size(" << size << ") must be equal to number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t new_columns_number = columns_number + 1;

   Matrix<T> new_matrix;

   if(this->empty())
   {
       new_matrix.set(new_column.size(), 1);

       new_matrix.set_column(0, new_column, new_name);

       return new_matrix;
   }
   else
   {
       new_matrix.set(rows_number, new_columns_number);

       for(size_t i = 0; i < rows_number; i++)
       {
           for(size_t j = 0; j < columns_number; j++)
           {
               new_matrix(i,j) = (*this)(i,j);
           }

           new_matrix(i,columns_number) = new_column[i];
       }
   }

   if(header != "")
   {
       Vector<string> new_header = get_header();

       new_header.push_back(new_name);

       new_matrix.set_header(new_header);
   }

   return(new_matrix);
}


/// Inserts a new row in a given position.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param position Index of new row.
/// @param new_row Vector with the row contents.

template <class T>
Matrix<T> Matrix<T>::insert_row(const size_t& position, const Vector<T>& new_row) const
{
   #ifdef __OPENNN_DEBUG__

    if(position > rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "insert_row(const size_t&, const Vector<T>&) const.\n"
              << "Position must be less or equal than number of rows.\n";

       throw logic_error(buffer.str());
    }

   const size_t size = new_row.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "insert_row(const size_t&, const Vector<T>&) const.\n"
             << "Size must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t new_rows_number = rows_number + 1;

   Matrix<T> new_matrix(new_rows_number, columns_number);

   for(size_t i = 0; i < position; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
           new_matrix(i,j) = (*this)(i,j);
       }
   }

   for(size_t j = 0; j < columns_number; j++)
   {
       new_matrix(position,j) = new_row[j];
   }

   for(size_t i = position+1; i < new_rows_number; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
           new_matrix(i,j) = (*this)(i-1,j);
       }
   }

   new_matrix.set_header(header);

   return(new_matrix);
}


template <class T>
void Matrix<T>::insert_row_values(const size_t& row_index, const size_t& column_position, const Vector<T>& values)
{
    const size_t values_size = values.size();

    for(size_t i = 0; i < values_size; i++)
    {
       (*this)(row_index, column_position + i) =  values[i];
    }
}


/// Inserts a new column in a given position.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param position Index of new column.
/// @param new_column Vector with the column contents.

template <class T>
Matrix<T> Matrix<T>::insert_column(const size_t& position, const Vector<T>& new_column, const string& new_name) const
{
   #ifdef __OPENNN_DEBUG__

    if(position > columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "insert_column(const size_t&, const Vector<T>&) const.\n"
              << "Position must be less or equal than number of columns.\n";

       throw logic_error(buffer.str());
    }

   const size_t size = static_cast<size_t>(new_column.size());

   if(size != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "insert_column(const size_t, const Vector<T>&) const.\n"
             << "Size(" << size << ") must be equal to number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t new_columns_number = columns_number + 1;

   Matrix<T> new_matrix(rows_number, new_columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
       for(size_t j = 0; j < position; j++)
       {
           new_matrix(i,j) = (*this)(i,j);
       }

       new_matrix(i,position) = new_column[i];

       for(size_t j = position+1; j < new_columns_number; j++)
       {
           new_matrix(i,j) = (*this)(i,j-1);
       }
   }

   if(header != "")
   {
       Vector<string> new_header = get_header();

       new_matrix.set_header(new_header.insert_element(position, new_name));
   }

   return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::insert_column(const string& column_name, const Vector<T>& new_column, const string& new_name) const
{
    const size_t column_index = get_column_index(column_name);

    return insert_column(column_index, new_column, new_name);
}


template <class T>
Matrix<T> Matrix<T>::add_columns(const size_t& columns_to_add) const
{
    Matrix<T> new_matrix(rows_number,columns_number+columns_to_add, T());

    Vector<string> new_header(columns_number+columns_to_add, "");

    for(size_t j = 0; j < columns_number; j++)
    {
        new_header[j] = header[j];
    }

#pragma omp parallel for

    for(int i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            new_matrix(i,j) = (*this)(i,j);
        }
    }

    new_matrix.set_header(new_header);

    return new_matrix;
}


template <class T>
Matrix<T> Matrix<T>::add_columns_first(const size_t& columns_to_add) const
{
    Matrix<T> new_matrix(rows_number,columns_number+columns_to_add, T());

    Vector<string> new_header(columns_number+columns_to_add, "");

    for(size_t j = 0; j < columns_number; j++)
    {
        new_header[columns_to_add+j] = header[j];
    }

#pragma omp parallel for

    for(int i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            new_matrix(i,columns_to_add+j) = (*this)(i,j);
        }
    }

    new_matrix.set_header(new_header);

    return new_matrix;
}


template <class T>
void Matrix<T>::split_column(const string& column_name, const Vector<string>& new_columns_name, const char& delimiter, const string& missing_value_label)
{
    const size_t column_index = get_column_index(column_name);

    const size_t new_columns_name_size = new_columns_name.size();

    const size_t new_columns_number = columns_number - 1 + new_columns_name_size;

    Matrix<T> new_matrix(rows_number, new_columns_number);

    Vector<T> new_row(new_columns_number);

    Vector<string> missing_values_vector(new_columns_name_size, missing_value_label);

    new_row = get_row(0).replace_element(column_index, new_columns_name);
    new_matrix.set_row(0, new_row);

    for(size_t i = 1; i < rows_number; i++)
    {
        if((*this)(i,column_index) == missing_value_label)
        {
            new_row = get_row(i).replace_element(column_index, missing_values_vector);
        }
        else
        {
            new_row = get_row(i).split_element(column_index, delimiter);
        }

        new_matrix.set_row(i, new_row);
    }

    set(new_matrix);
}


template <class T>
void Matrix<T>::split_column(const string& column_name, const string& column_1_name, const string& column_2_name, const size_t& size_1, const size_t& size_2)
{
    const size_t column_index = get_column_index(column_name);

    const Vector<T> column = get_column(column_index);

    Vector<T> column_1(rows_number);
    Vector <T> column_2(rows_number);

    column_1[0] = column_1_name;
    column_2[0] = column_2_name;

    for(size_t i = 1; i < rows_number; i++)
    {
        column_1[i] = column[i].substr(0, size_1);
        column_2[i] = column[i].substr(size_1, size_2);
    }

    set_column(column_index, column_1, column_1_name);
   (*this) = insert_column(column_index+1, column_2, column_2_name);
}


template <class T>
void Matrix<T>::swap_columns(const size_t& column_1_index, const size_t& column_2_index)
{
    const Vector<T> column_1 = get_column(column_1_index);
    const Vector<T> column_2 = get_column(column_2_index);

    const string header_1 = header[column_1_index];
    const string header_2 = header[column_2_index];

    set_column(column_1_index, column_2);

    set_column(column_2_index, column_1);

    header[column_1_index] = header_2;
    header[column_2_index] = header_1;
}


template <class T>
void Matrix<T>::swap_columns(const string& column_1_name, const string& column_2_name)
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);

    swap_columns(column_1_index, column_2_index);
}


template <class T>
void Matrix<T>::merge_columns(const string& column_1_name, const string& column_2_name, const string& merged_column_name, const char& separator)
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);

    const Vector<T> column_1 = get_column(column_1_index);
    const Vector<T> column_2 = get_column(column_2_index);

    Vector<T> merged_column(column_1.size());

    for(size_t i = 0; i < column_1.size(); i++)
    {
        merged_column[i] = column_1[i] + separator + column_2[i];
    }

    set_column(column_1_index, merged_column);

    set_header(column_1_index, merged_column_name);

    delete_column(column_2_index);

}


template <class T>
void Matrix<T>::merge_columns(const size_t& column_1_index, const size_t& column_2_index, const char& separator)
{
    const size_t column_number = column_2_index-column_1_index+1;

    const Vector<size_t> indices(column_1_index,1,column_2_index);

    const Matrix<T> columns_matrix = get_submatrix_columns(indices);

    Vector<T> joined_column(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < column_number; j++)
        {
            const Vector<T> column = columns_matrix.get_column(j);

            joined_column[i] += column[i] + separator;
        }
    }

   (*this) = delete_columns(indices);

   (*this) = insert_column(indices[0],joined_column);
}


template <class T>
Matrix<T> Matrix<T>::merge_matrices(const Matrix<T>& other_matrix, const string& columns_1_name, const string& columns_2_name,
                                    const string& left_header_tag, const string& right_header_tag) const
{
    const size_t other_columns_number = other_matrix.get_columns_number();

    const size_t columns_1_index = this->get_column_index(columns_1_name);
    const size_t columns_2_index = other_matrix.get_column_index(columns_2_name);

    const Vector<T> columns_1 = this->get_column(columns_1_index);
    const Vector<T> columns_2 = other_matrix.get_column(columns_2_index);

    const size_t columns_1_size = columns_1.size();

    const Vector<T> header_1 = this->get_header();
    const Vector<T> header_2 = other_matrix.get_header();

    size_t merged_rows_number = columns_1.count_equal_to(columns_2);

    Vector<T> merged_header = header_1.delete_element(columns_1_index) + left_header_tag;

    merged_header = merged_header.assemble(header_2.delete_element(columns_2_index) + right_header_tag);

    merged_header = merged_header.insert_element(columns_1_index, columns_1_name);

    if(merged_rows_number == 0)
    {
        Matrix<T> merged_matrix;

        return merged_matrix;
    }

    Matrix<T> merged_matrix(merged_rows_number,merged_header.size());

    merged_matrix.set_header(merged_header);

    size_t current_merged_row_index = 0;

    Vector<T> columns_2_sorted_values = columns_2.sort_ascending_values();
    Vector<size_t> columns_2_sorted_indices = columns_2.sort_ascending_indices();

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const T current_index_value = columns_1[i];

        pair<typename vector<T>::iterator,typename vector<T>::iterator> bounds = equal_range(columns_2_sorted_values.begin(), columns_2_sorted_values.end(), current_index_value);

        const size_t initial_index = bounds.first - columns_2_sorted_values.begin();
        const size_t final_index = bounds.second - columns_2_sorted_values.begin();

        for(size_t j = initial_index; j < final_index; j++)
        {
            const size_t current_row_2_index = columns_2_sorted_indices[j];

            for(size_t k = 0; k < columns_number; k++)
            {
                merged_matrix(current_merged_row_index,k) = (*this)(i,k);
            }

            for(size_t k = 0; k < other_columns_number; k++)
            {
                if(k < columns_2_index)
                {
                    merged_matrix(current_merged_row_index,k+columns_number) = other_matrix(current_row_2_index,k);
                }
                else if(k > columns_2_index)
                {
                    merged_matrix(current_merged_row_index,k+columns_number-1) = other_matrix(current_row_2_index,k);
                }
            }

            current_merged_row_index++;
        }
    }

    return merged_matrix;
}


template <class T>
Matrix<T> Matrix<T>::merge_matrices(const Matrix<T>& other_matrix, const size_t& columns_1_index, const size_t& columns_2_index) const
{
    const size_t other_columns_number = other_matrix.get_columns_number();

    const Vector<T> columns_1 = this->get_column(columns_1_index);
    const Vector<T> columns_2 = other_matrix.get_column(columns_2_index);

    const size_t columns_1_size = columns_1.size();

    size_t merged_rows_number = columns_1.count_equal_to(columns_2);

    if(merged_rows_number == 0)
    {
        Matrix<T> merged_matrix;

        return merged_matrix;
    }

    Matrix<T> merged_matrix(merged_rows_number,columns_number + other_columns_number - 1);

    size_t current_merged_row_index = 0;

    Vector<T> columns_2_sorted_values = columns_2.sort_ascending_values();
    Vector<size_t> columns_2_sorted_indices = columns_2.sort_ascending_indices();

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const T current_index_value = columns_1[i];

        pair<typename vector<T>::iterator,typename vector<T>::iterator> bounds = equal_range(columns_2_sorted_values.begin(), columns_2_sorted_values.end(), current_index_value);

        const size_t initial_index = bounds.first - columns_2_sorted_values.begin();
        const size_t final_index = bounds.second - columns_2_sorted_values.begin();

        for(size_t j = initial_index; j < final_index; j++)
        {
            const size_t current_row_2_index = columns_2_sorted_indices[j];

            for(size_t k = 0; k < columns_number; k++)
            {
                merged_matrix(current_merged_row_index,k) = (*this)(i,k);
            }

            for(size_t k = 0; k < other_columns_number; k++)
            {
                if(k < columns_2_index)
                {
                    merged_matrix(current_merged_row_index,k+columns_number) = other_matrix(current_row_2_index,k);
                }
                else if(k > columns_2_index)
                {
                    merged_matrix(current_merged_row_index,k+columns_number-1) = other_matrix(current_row_2_index,k);
                }
            }

            current_merged_row_index++;
        }
    }

    return merged_matrix;
}


template <class T>
Matrix<T> Matrix<T>::right_join(const Matrix<T>& other_matrix, const string& columns_1_name, const string& columns_2_name,
                                const string& left_header_tag, const string& right_header_tag) const
{
    const size_t columns_1_index = this->get_column_index(columns_1_name);
    const size_t columns_2_index = other_matrix.get_column_index(columns_2_name);

    const Vector<T> columns_1 = this->get_column(columns_1_index);
    const Vector<T> columns_2 = other_matrix.get_column(columns_2_index);

    const size_t columns_1_size = columns_1.size();
    const size_t columns_2_size = columns_2.size();

    const Vector<T> header_1 = this->get_header();
    const Vector<T> header_2 = other_matrix.get_header();

    Vector<T> merged_header = header_1.delete_element(columns_1_index) + left_header_tag;

    merged_header = merged_header.assemble(header_2.delete_element(columns_2_index) + right_header_tag);

    merged_header = merged_header.insert_element(columns_1_index, columns_1_name);

    Matrix<T> merged_matrix = other_matrix.add_columns_first(columns_number-1);

    merged_matrix = merged_matrix.insert_column(columns_1_index, columns_2);
    merged_matrix = merged_matrix.delete_column(columns_number+columns_2_index);

    merged_matrix.set_header(merged_header);

    Vector<size_t> columns_1_sorted_indices = columns_1.sort_ascending_indices();
    Vector<size_t> columns_2_sorted_indices = columns_2.sort_ascending_indices();

    size_t columns_1_pointer = 0;

    for(size_t i = 0; i < columns_2_size; i++)
    {
        const size_t current_row_index = columns_2_sorted_indices[i];

        const T current_index_value = columns_2[current_row_index];

        if(columns_1[columns_1_sorted_indices[columns_1_pointer]] > current_index_value)
        {
            continue;
        }

        while(columns_1_pointer < columns_1_size)
        {
            const size_t current_row_1_index = columns_1_sorted_indices[columns_1_pointer];

            if(columns_1[current_row_1_index] < current_index_value)
            {
                columns_1_pointer++;

                continue;
            }
            else if(columns_1[current_row_1_index] == current_index_value)
            {
                for(size_t k = 0; k < columns_number; k++)
                {
                    if(k < columns_1_index)
                    {
                        merged_matrix(current_row_index,k) = (*this)(current_row_1_index,k);
                    }
                    else if(k > columns_1_index)
                    {
                        merged_matrix(current_row_index,k) = (*this)(current_row_1_index,k);
                    }
                }

                break;
            }
            else if(columns_1[current_row_1_index] > current_index_value)
            {
                break;
            }
        }
    }

    return merged_matrix;
}


template <class T>
Matrix<T> Matrix<T>::left_join(const Matrix<T>& other_matrix, const string& columns_1_name, const string& columns_2_name,
                                const string& left_header_tag, const string& right_header_tag) const
{
    const size_t other_columns_number = other_matrix.get_columns_number();

    const size_t columns_1_index = this->get_column_index(columns_1_name);

    const size_t columns_2_index = other_matrix.get_column_index(columns_2_name);

    const Vector<T> columns_1 = this->get_column(columns_1_index);
    const Vector<T> columns_2 = other_matrix.get_column(columns_2_index);

    const size_t columns_1_size = columns_1.size();
    const size_t columns_2_size = columns_2.size();

    const Vector<T> header_1 = this->get_header();
    const Vector<T> header_2 = other_matrix.get_header();

    Vector<T> merged_header = header_1.delete_element(columns_1_index) + left_header_tag;

    merged_header = merged_header.assemble(header_2.delete_element(columns_2_index) + right_header_tag);

    merged_header = merged_header.insert_element(columns_1_index, columns_1_name);

    Matrix<T> merged_matrix = this->add_columns(other_columns_number-1);

    merged_matrix.set_header(merged_header);

    Vector<size_t> columns_1_sorted_indices = columns_1.sort_ascending_indices();
    Vector<size_t> columns_2_sorted_indices = columns_2.sort_ascending_indices();

    size_t columns_2_pointer = 0;

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const size_t current_row_index = columns_1_sorted_indices[i];

        const T current_index_value = columns_1[current_row_index];

        if(columns_2[columns_2_sorted_indices[columns_2_pointer]] > current_index_value)
        {
            continue;
        }

        while(columns_2_pointer < columns_2_size)
        {
            const size_t current_row_2_index = columns_2_sorted_indices[columns_2_pointer];

            if(columns_2[current_row_2_index] < current_index_value)
            {
                columns_2_pointer++;

                continue;
            }
            else if(columns_2[current_row_2_index] == current_index_value)
            {
                for(size_t k = 0; k < other_columns_number; k++)
                {
                    if(k < columns_2_index)
                    {
                        merged_matrix(current_row_index,k+columns_number) = other_matrix(current_row_2_index,k);
                    }
                    else if(k > columns_2_index)
                    {
                        merged_matrix(current_row_index,k+columns_number-1) = other_matrix(current_row_2_index,k);
                    }
                }

                break;
            }
            else if(columns_2[current_row_2_index] > current_index_value)
            {
                break;
            }
        }

        if(columns_2_pointer >= columns_2_size)
        {
            break;
        }
    }

    return merged_matrix;
}


template <class T>
Matrix<T> Matrix<T>::left_join(const Matrix<T>& other_matrix, const size_t& columns_1_index, const size_t& columns_2_index) const
{
    const size_t other_columns_number = other_matrix.get_columns_number();

    const Vector<T> columns_1 = this->get_column(columns_1_index);
    const Vector<T> columns_2 = other_matrix.get_column(columns_2_index);

    const size_t columns_1_size = columns_1.size();
    const size_t columns_2_size = columns_2.size();

    Matrix<T> merged_matrix = this->add_columns(other_columns_number-1);

    Vector<size_t> columns_1_sorted_indices = columns_1.sort_ascending_indices();
    Vector<size_t> columns_2_sorted_indices = columns_2.sort_ascending_indices();

    size_t columns_2_pointer = 0;

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const size_t current_row_index = columns_1_sorted_indices[i];

        const T current_index_value = columns_1[current_row_index];

        if(columns_2[columns_2_sorted_indices[columns_2_pointer]] > current_index_value)
        {
            continue;
        }

        while(columns_2_pointer < columns_2_size)
        {
            const size_t current_row_2_index = columns_2_sorted_indices[columns_2_pointer];

            if(columns_2[current_row_2_index] < current_index_value)
            {
                columns_2_pointer++;

                continue;
            }
            else if(columns_2[current_row_2_index] == current_index_value)
            {
                for(size_t k = 0; k < other_columns_number; k++)
                {
                    if(k < columns_2_index)
                    {
                        merged_matrix(current_row_index,k+columns_number) = other_matrix(current_row_2_index,k);
                    }
                    else if(k > columns_2_index)
                    {
                        merged_matrix(current_row_index,k+columns_number-1) = other_matrix(current_row_2_index,k);
                    }
                }

                break;
            }
            else if(columns_2[current_row_2_index] > current_index_value)
            {
                break;
            }
        }

        if(columns_2_pointer >= columns_2_size)
        {
            break;
        }
    }

    return merged_matrix;
}


template <class T>
Matrix<T> Matrix<T>::left_join(const Matrix<T>& other_matrix, const string& matrix_1_name_1, const string& matrix_1_name_2,const string& matrix_2_name_1,const string& matrix_2_name_2,
                                const string& left_header_tag, const string& right_header_tag) const
{
    const size_t other_columns_number = other_matrix.get_columns_number();

    const size_t matrix_1_columns_1_index = this->get_column_index(matrix_1_name_1);
    const size_t matrix_1_columns_2_index = this->get_column_index(matrix_1_name_2);

    const size_t matrix_2_columns_1_index = other_matrix.get_column_index(matrix_2_name_1);
    const size_t matrix_2_columns_2_index = other_matrix.get_column_index(matrix_2_name_2);

    const Vector<T> matrix_1_columns_1 = this->get_column(matrix_1_columns_1_index);
    const Vector<T> matrix_1_columns_2 = this->get_column(matrix_1_columns_2_index);
    const Vector<T> matrix_2_columns_1 = other_matrix.get_column(matrix_2_columns_1_index);
    const Vector<T> matrix_2_columns_2 = other_matrix.get_column(matrix_2_columns_2_index);

    const size_t matrix_1_columns_1_size = matrix_1_columns_1.size();
    const size_t matrix_1_columns_2_size = matrix_1_columns_2.size();
    const size_t matrix_2_columns_1_size = matrix_2_columns_1.size();

    const Vector<T> header_1 = this->get_header();
    const Vector<T> header_2 = other_matrix.get_header();

    Vector<T> merged_header = header_1.delete_element(matrix_1_columns_2_index).delete_element(matrix_1_columns_1_index) + left_header_tag;

    merged_header = merged_header.assemble(header_2.delete_element(matrix_2_columns_2_index).delete_element(matrix_2_columns_1_index) + right_header_tag);

    merged_header = merged_header.insert_element(matrix_1_columns_1_index, matrix_1_name_1).insert_element(matrix_1_columns_2_index,matrix_1_name_2);

    Matrix<T> merged_matrix = this->add_columns(other_columns_number-2);

    merged_matrix.set_header(merged_header);

    Vector<size_t> columns_1_sorted_indices = matrix_1_columns_1.sort_ascending_indices();
    Vector<size_t> columns_2_sorted_indices = matrix_2_columns_1.sort_ascending_indices();

    size_t columns_2_pointer = 0;

    for(size_t i = 0; i < matrix_1_columns_1_size; i++)
    {
        const size_t current_row_index = columns_1_sorted_indices[i];

        const T current_index_1_value = matrix_1_columns_1[current_row_index];
        const T current_index_2_value = matrix_1_columns_2[current_row_index];

        if(matrix_2_columns_1[columns_2_sorted_indices[columns_2_pointer]] > current_index_1_value)
        {
            continue;
        }

        while(columns_2_pointer < matrix_1_columns_2_size)
        {
            const size_t current_row_2_index = columns_2_sorted_indices[columns_2_pointer];

            if(matrix_2_columns_1[current_row_2_index] < current_index_1_value)
            {
                columns_2_pointer++;

                continue;
            }
            else if(matrix_2_columns_1[current_row_2_index] == current_index_1_value && stod(matrix_2_columns_2[current_row_2_index]) == stod(current_index_2_value))
            {
                for(size_t k = 0; k < other_columns_number; k++)
                {
                    if(k < matrix_2_columns_1_index && k < matrix_2_columns_2_index)
                    {
                        merged_matrix(current_row_index,k+columns_number) = other_matrix(current_row_2_index,k);
                    }
                    else if(k > matrix_2_columns_1_index && k < matrix_2_columns_2_index)
                    {
                        merged_matrix(current_row_index,k+columns_number-1) = other_matrix(current_row_2_index,k);
                    }
                    else if(k > matrix_2_columns_1_index && k > matrix_2_columns_2_index)
                    {
                        merged_matrix(current_row_index,k+columns_number-2) = other_matrix(current_row_2_index,k);
                    }
                }

                break;
            }
            else if(matrix_2_columns_1[current_row_2_index] == current_index_1_value && stod(matrix_2_columns_2[current_row_2_index]) != stod(current_index_2_value))
            {
                columns_2_pointer++;

                continue;
            }
            else if(matrix_2_columns_1[current_row_2_index] > current_index_1_value)
            {
                break;
            }
        }

        if(columns_2_pointer >= matrix_2_columns_1_size) break;
    }

    return merged_matrix;
}


/// This method removes the row with given index.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param row_index Index of row to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_row(const size_t& row_index) const
{
   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> delete_row(const size_t&) const.\n"
             << "Index of row must be less than number of rows.\n";

      throw logic_error(buffer.str());
   }
   else if(rows_number < 2)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> delete_row(const size_t&) const.\n"
             << "Number of rows must be equal or greater than two.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> new_matrix(rows_number-1, columns_number);

   for(size_t i = 0; i < row_index; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
        new_matrix(i,j) = (*this)(i,j);
      }
   }

   for(size_t i = row_index+1; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         new_matrix(i-1,j) = (*this)(i,j);
      }
   }

   if(header != "")
   {
       new_matrix.set_header(header);
   }

   return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::delete_rows(const Vector<size_t>& rows_to_remove) const
{
    const size_t rows_to_delete_number = rows_to_remove.size();

    if(rows_to_delete_number == 0) return Matrix<T>(*this);

    const size_t rows_to_keep_number = rows_number - rows_to_delete_number;

    Vector<size_t> rows_to_keep(rows_to_keep_number);

    size_t index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(!rows_to_remove.contains(i))
        {
            rows_to_keep[index] = i;

            index++;
        }
    }

    return get_submatrix_rows(rows_to_keep);
}


template <class T>
Matrix<T> Matrix<T>::delete_rows_with_value(const T& value) const
{
    Vector<T> row(columns_number);

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        row = get_row(i);

        if(!row.contains(value))
        {
            count++;
        }
    }

    if(count == 0)
    {
        Matrix<T> copy_matrix(*this);

        return copy_matrix;
    }

    Vector<size_t> indices(count);

    size_t index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        row = get_row(i);

        if(!row.contains(value))
        {
            indices[index] = i;
            index++;
        }
    }

    return(get_submatrix_rows(indices));
}


template <class T>
Matrix<T> Matrix<T>::delete_rows_equal_to(const T& value) const
{
    const Vector<size_t> indices = get_rows_equal_to(value);

    return delete_rows(indices);
}


template <class T>
Matrix<T> Matrix<T>::delete_first_rows(const size_t& number) const
{

    const Vector<size_t> indices(number, 1, rows_number-1);

    return get_submatrix_rows(indices);
}


template <class T>
Matrix<T> Matrix<T>::delete_first_columns(const size_t& number) const
{
    const Vector<size_t> indices(number, 1, columns_number-1);

    return get_submatrix_columns(indices);
}


template <class T>
Matrix<T> Matrix<T>::get_first_rows(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, number-1);

    return get_submatrix_rows(indices);
}


template <class T>
T Matrix<T>::get_first(const size_t& column_index) const
{
    return(*this)(0, column_index);
}


template <class T>
T Matrix<T>::get_first(const string& column_name) const
{
    const size_t column_index = get_column_index(column_name);

    return(*this)(0, column_index);
}


template <class T>
T Matrix<T>::get_last(const size_t& column_index) const
{
    return(*this)(rows_number-1, column_index);

}


template <class T>
T Matrix<T>::get_last(const string& column_name) const
{
    const size_t column_index = get_column_index(column_name);

    return(*this)(rows_number-1, column_index);
}


template <class T>
Matrix<T> Matrix<T>::delete_last_rows(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, rows_number-number-1);

    return get_submatrix_rows(indices);
}


template <class T>
Matrix<T> Matrix<T>::delete_last_columns(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, columns_number-number-1);

    return get_submatrix_columns(indices);
}


template <class T>
Matrix<T> Matrix<T>::get_last_rows(const size_t& number) const
{
    const size_t rows_number = get_rows_number();

    const Vector<size_t> indices(rows_number-number, 1, rows_number-1);

    return get_submatrix_rows(indices);
}


/// This method removes the column with given index.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param column_index Index of column to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_column(const size_t& column_index) const
{
   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> delete_column(const size_t&) const.\n"
             << "Index of column must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }
   else if(columns_number < 2)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> delete_column(const size_t&) const.\n"
             << "Number of columns must be equal or greater than two.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> new_matrix(rows_number, columns_number-1);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < column_index; j++)
      {
        new_matrix(i,j) = (*this)(i,j);
      }
   }

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = column_index+1; j < columns_number; j++)
      {
         new_matrix(i,j-1) = (*this)(i,j);
      }
   }

   if(header != "")
   {
       const Vector<string> new_header = get_header().delete_index(column_index);

       new_matrix.set_header(new_header);
   }

   return new_matrix;
}


/// This method removes the column with given index.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param column_index Index of column to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_column(const string& column_name) const
{
    const Vector<size_t> indices = header.calculate_equal_to_indices(column_name);

    const size_t occurrences_number = indices.size();

    if(occurrences_number == 0)
    {
        return *this;
    }
    else if(occurrences_number == 1)
    {
        return delete_column(indices[0]);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "void Matrix<T>::delete_column_by_name(const string& column_name).\n"
               << "Number of columns with name " << column_name << " is " << indices.size() << ").\n";

        throw logic_error(buffer.str());
    }
}


template <class T>
Matrix<T> Matrix<T>::delete_columns(const Vector<size_t>& delete_indices) const
{
    if(delete_indices.empty())
    {
        return Matrix<T>(*this);
    }

    Matrix<T> new_data;

    Vector<size_t> keep_indices;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(!delete_indices.contains(i))
        {
            keep_indices.push_back(i);
        }
    }

    const size_t keep_indices_size = keep_indices.size();

    if(keep_indices_size != columns_number)
    {
        new_data = get_submatrix_columns(keep_indices);

        if(header != "") new_data.set_header(header.get_subvector(keep_indices));
    }

    return new_data;
}


template <class T>
Matrix<T> Matrix<T>::delete_columns(const Vector<string>& delete_names) const
{
    const Vector<size_t> indices = get_column_indices(delete_names);

    return delete_columns(indices);
}


template <class T>
Matrix<T> Matrix<T>::delete_columns_name_contains(const Vector<string>& substrings) const
{
    Vector<size_t> indices;

    for(size_t i = 0; i < columns_number; i++)
    {
        for(size_t j = 0; j < substrings.size(); j++)
        {
            if(header[i].find(substrings[j]) != string::npos)
            {
                indices.push_back(i);

                break;
            }
        }
    }

    return delete_columns(indices);
}


template <class T>
Vector<size_t> Matrix<T>::get_constant_columns_indices() const
{
    Vector<size_t> constant_columns;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(is_column_constant(i))
        {
            constant_columns.push_back(i);
        }
    }

    return constant_columns;
}


template <class T>
Matrix<T> Matrix<T>::delete_constant_columns() const
{
    return delete_columns(get_constant_columns_indices());
}


template <class T>
Matrix<T> Matrix<T>::delete_binary_columns() const
{
    const Vector<size_t> binary_columns = get_binary_column_indices();

    return delete_columns(binary_columns);
}


template <class T>
Matrix<T> Matrix<T>::delete_binary_columns(const double& minimum_support) const
{
    const Vector<size_t> binary_columns = get_binary_column_indices();

    const size_t binary_columns_number = binary_columns.size();

    Vector<size_t> columns_to_remove;

    for(size_t i = 0; i < binary_columns_number; i++)
    {
        const double support = get_column(binary_columns[i]).calculate_sum()/static_cast<double>(rows_number);

        if(support < minimum_support) columns_to_remove.push_back(binary_columns[i]);
    }

    return delete_columns(columns_to_remove);
}


template <class T>
Matrix<T> Matrix<T>::delete_constant_rows() const
{
    Vector<size_t> constant_rows;

    Vector<T> row;

    for(size_t i = 0; i < rows_number; i++)
    {
        row = get_row(i);

        if(row.is_constant())
        {
            constant_rows.push_back(i);
        }
    }

    return delete_rows(constant_rows);
}


/// Assemble two matrices.
/// @param other_matrix matrix to be get_assembled to this matrix.

template <class T>
Matrix<T> Matrix<T>::assemble_rows(const Matrix<T>& other_matrix) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> assemble_rows(const Matrix<T>&) const method.\n"
             << "Number of columns of other matrix (" << other_columns_number << ") must be equal to number of columns of this matrix (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t other_rows_number = other_matrix.get_rows_number();

   if (rows_number == 0 && other_rows_number == 0)
   {
       return Matrix<T>();
   }
   else if (rows_number == 0)
   {
       return other_matrix;
   }

   Matrix<T> assembly(rows_number + other_rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         assembly(i,j) = (*this)(i,j);
      }
   }

   for(size_t i = 0; i < other_rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         assembly(rows_number+i,j) = other_matrix(i,j);
      }
   }

   if(header != "")
   {
       assembly.set_header(header);
   }
   else if(other_matrix.get_header() != "")
   {
       assembly.set_header(other_matrix.get_header());
   }

   return(assembly);
}


/// Sorts the rows of the matrix in descending order attending to the values of the column with given index.
/// It returns a new sorted matrix, it does not change the original one.
/// @param column_index Index of column to sort.

template <class T>
Matrix<T> Matrix<T>::sort_ascending(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<T> column = get_column(column_index);

    const Vector<size_t> indices = column.sort_ascending_indices();

    size_t index;

    for(size_t i = 0; i < rows_number; i++)
    {
        index = indices[i];

        for(size_t j = 0; j < columns_number; j++)
        {
            sorted(i,j) = (*this)(index, j);
        }
    }

    return(sorted);
}


template <class T>
Matrix<T> Matrix<T>::sort_ascending_strings(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<double> column = get_column(column_index).string_to_double();

    const Vector<size_t> indices = column.sort_ascending_indices();

    size_t index;

    for(size_t i = 0; i < rows_number; i++)
    {
        index = indices[i];

        for(size_t j = 0; j < columns_number; j++)
        {
            sorted(i,j) = (*this)(index, j);
        }
    }

    if(header != "") sorted.set_header(header);

    return(sorted);
}


template <class T>
Matrix<T> Matrix<T>::sort_descending_strings(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<double> column = get_column(column_index).string_to_double();

    const Vector<size_t> indices = column.sort_descending_indices();

    size_t index;

    for(size_t i = 0; i < rows_number; i++)
    {
        index = indices[i];

        for(size_t j = 0; j < columns_number; j++)
        {
            sorted(i,j) = (*this)(index, j);
        }
    }

    if(header != "") sorted.set_header(header);

    return(sorted);
}


template <class T>
Matrix<T> Matrix<T>::sort_ascending_strings_absolute_value(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<double> column = get_column(column_index).string_to_double().calculate_absolute_value();

    const Vector<size_t> indices = column.sort_ascending_indices();

    size_t index;

    for(size_t i = 0; i < rows_number; i++)
    {
        index = indices[i];

        for(size_t j = 0; j < columns_number; j++)
        {
            sorted(i,j) = (*this)(index, j);
        }
    }

    return(sorted);
}


template <class T>
Matrix<T> Matrix<T>::sort_descending_strings_absolute_value(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<double> column = get_column(column_index).string_to_double().calculate_absolute_value();

    const Vector<size_t> indices = column.sort_descending_indices();

    size_t index;

    for(size_t i = 0; i < rows_number; i++)
    {
        index = indices[i];

        for(size_t j = 0; j < columns_number; j++)
        {
            sorted(i,j) = (*this)(index, j);
        }
    }

    return(sorted);
}


template <class T>
Matrix<T> Matrix<T>::sort_rank_rows(const Vector<size_t>& rank) const
{
    #ifdef __OPENNN_DEBUG__
    const size_t rank_size = rank.size();

      if(rows_number != rank_size) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "Matrix<T> sort_rank_rows(const Vector<size_t>&) const.\n"
               << "Matrix number of rows is " << rows_number << " and rank size is " << rank_size
               << " and they must be the same.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Matrix<T> sorted_matrix(rows_number,columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        sorted_matrix.set_row(i,(*this).get_row(rank[i]));
    }

    if(header != "") sorted_matrix.set_header(header);

    return sorted_matrix;
}


template <class T>
Matrix<T> Matrix<T>::sort_columns(const Vector<size_t>& rank) const
{
    #ifdef __OPENNN_DEBUG__
    const size_t columns_size = rank.size();

      if(rows_number != rank_size) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "Matrix<T> sort_rank_rows(const Vector<size_t>&) const.\n"
               << "Matrix number of rows is " << rows_number << " and rank size is " << rank_size
               << " and they must be the same.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Matrix<T> sorted_matrix(rows_number,columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        sorted_matrix.set_column(i,(*this).get_column(rank[i]),(*this).get_header()[rank[i]]);
    }

//    if(header != "") sorted_matrix.set_header(header);

    return sorted_matrix;
}


template <class T>
Matrix<T> Matrix<T>::sort_columns(const Vector<string>& new_header) const
{
    #ifdef __OPENNN_DEBUG__

    if(columns_number != new_header.size()) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> sort_columns(const Vector<string>& new_header) const.\n"
             << "New header size doesn't match with columns number.\n";

      throw logic_error(buffer.str());
    }

      const size_t count = new_header.count_equal_to((*this).get_header());

      if(count != new_header.size()) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "Matrix<T> sort_columns(const Vector<string>& new_header) const.\n"
               << "Occurrences number doesn't match with columns number\n";

        throw logic_error(buffer.str());
      }

    #endif

    Matrix<T> sorted_matrix(rows_number,columns_number);

    for(size_t i = 0; i < new_header.size(); i++)
    {
        const string current_variable = new_header[i];

        for (size_t j = 0; j < columns_number; j++)
        {
            const string current_column_name = (*this).get_header()[j];

            if(current_variable == current_column_name)
            {
                sorted_matrix.set_column(i,(*this).get_column(j),current_variable);

                break;
            }
        }
    }

//    if(header != "") sorted_matrix.set_header(header);

    return sorted_matrix;
}


template <class T>
bool compare(size_t a, size_t b, const Vector<T>& data)
{
    return data[a]<data[b];
}


/// Sorts the rows of the matrix in ascending order attending to the values of the column with given index.
/// It returns a new sorted matrix, it does not change the original one.
/// @param column_index Index of column to sort.

template <class T>
Matrix<T> Matrix<T>::sort_descending(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<T> column = get_column(column_index);

    const Vector<size_t> indices = column.sort_descending_indices();

    size_t index;

    for(size_t i = 0; i < rows_number; i++)
    {
        index = indices[i];

        for(size_t j = 0; j < columns_number; j++)
        {
            sorted(i,j) = (*this)(index, j);
        }
    }

    return(sorted);
}


/// @param other_matrix matrix to be get_assemblyd to this matrix.

template <class T>
Matrix<T> Matrix<T>::assemble_columns(const Matrix<T>& other_matrix) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> assemble_columns(const Matrix<T>&) const method.\n"
             << "Number of rows of other matrix (" << other_rows_number << ") must be equal to number of rows of this matrix (" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t other_columns_number = other_matrix.get_columns_number();

   Matrix<T> assembly(rows_number, columns_number + other_columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         assembly(i,j) = (*this)(i,j);
      }
      for(size_t j = 0; j < other_columns_number; j++)
      {
         assembly(i,columns_number+j) = other_matrix(i,j);
      }
   }

   if(header != "" && other_matrix.get_header() != "") assembly.set_header(header.assemble(other_matrix.get_header()));

   return(assembly);
}


/// Initializes all the elements of the matrix with a given value.
/// @param value Type value.

template <class T>
void Matrix<T>::initialize(const T& value)
{
    fill((*this).begin(),(*this).end(), value);
}


template <class T>
void Matrix<T>::initialize(const Vector<T>& v)
{
    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            (*this)(i,j)  = v[j];
        }
    }
}


template <class T>
void Matrix<T>::replace(const T& find_what, const T& replace_with)
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
    {
        if((*this)[i] == find_what)
        {
           (*this)[i] = replace_with;
        }
    }
}


template <class T>
void Matrix<T>::replace(const string& column_1, const T& value_1, const string& column_2, const T& value_2_find, const T& value_2_replace)
{
    const size_t column_1_index = get_column_index(column_1);
    const size_t column_2_index = get_column_index(column_2);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_1_index) == value_1 && (*this)(i, column_2_index) == value_2_find)
        {
            (*this)(i, column_2_index) = value_2_replace;
        }
    }
}


template <class T>
void Matrix<T>::replace(const string& column_1, const T& value_1_find, const T& value_1_replace, const string& column_2, const T& value_2_find, const T& value_2_replace)
{
    const size_t column_1_index = get_column_index(column_1);
    const size_t column_2_index = get_column_index(column_2);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_1_index) == value_1_find && (*this)(i, column_2_index) == value_2_find)
        {
            (*this)(i, column_1_index) = value_1_replace;
            (*this)(i, column_2_index) = value_2_replace;
        }
    }
}


template <class T>
void Matrix<T>::replace_header(const string& find_what, const string& replace_with)
{
    for(size_t i = 0; i < columns_number; i++)
        if(header[i] == find_what)
            header[i] = replace_with;
}


template <class T>
void Matrix<T>::replace_in_row(const size_t& row_index, const T& find_what, const T& replace_with)
{
    for(size_t i = 0; i < columns_number; i++)
    {
        if((*this)(row_index,i) == find_what)
        {
           (*this)(row_index,i) = replace_with;
        }
    }
}


template <class T>
void Matrix<T>::replace_in_column(const size_t& column_index, const T& find_what, const T& replace_with)
{
    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_index) == find_what)
        {
           (*this)(i, column_index) = replace_with;
        }
    }
}


template <class T>
void Matrix<T>::replace_in_column(const string& column_name, const T& find_what, const T& replace_with)
{
    const size_t column_index = get_column_index(column_name);

    replace_in_column(column_index, find_what, replace_with);
}


template <class T>
void Matrix<T>::replace_substring(const string& find_what, const string& replace_with)
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
    {
        size_t position = 0;

        while((position = (*this)[i].find(find_what, position)) != string::npos)
        {
            (*this)[i].replace(position, find_what.length(), replace_with);

             position += replace_with.length();
        }
    }
}


template <class T>
void Matrix<T>::replace_substring(const size_t& column_index, const string& find_what, const string& replace_with)
{
    for(size_t i = 0; i < rows_number; i++)
    {
        size_t position = 0;

        while((position = (*this)(i, column_index).find(find_what, position)) != string::npos)
        {
            (*this)(i, column_index).replace(position, find_what.length(), replace_with);

             position += replace_with.length();
        }
    }
}


template <class T>
void Matrix<T>::replace_substring(const string& column_name, const string& find_what, const string& replace_with)
{
    const size_t column_index = get_column_index(column_name);

    replace_substring(column_index, find_what, replace_with);
}


template <class T>
void Matrix<T>::replace_contains(const string& find_what, const string& replace_with)
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
    {
        if((*this)[i].find(find_what) != string::npos)
        {
            (*this)[i] = replace_with;
        }
    }
}


template <class T>
void Matrix<T>::replace_contains_in_row(const size_t& row_index, const string& find_what, const string& replace_with)
{
    for(size_t i = 0; i < columns_number; i++)
    {
        if((*this)(row_index, i).find(find_what) != string::npos)
        {
            (*this)(row_index, i) = replace_with;
        }
    }
}


/// @todo

template <class T>
void Matrix<T>::replace_special_characters()
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
    {
//        if((*this)[i].find(find_what) != string::npos)
        {
//            (*this)[i] = replace_with;
        }
    }
}


template <class T>
void Matrix<T>::replace_column_equal_to(const string& column_name, const T& find_value, const T& replace_value)
{
    const size_t column_index = get_column_index(column_name);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == find_value)
        {
          (*this)(i,column_index) = replace_value;
        }
    }
}


template <class T>
void Matrix<T>::replace_column_not_equal_to(const string& column_name, const T& find_value, const T& replace_value)
{
    const size_t column_index = get_column_index(column_name);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) != find_value)
        {
          (*this)(i,column_index) = replace_value;
        }
    }
}


template <class T>
void Matrix<T>::replace_column_not_equal_to(const string& column_name, const Vector<T>& find_values, const T& replace_value)
{
    const size_t column_index = get_column_index(column_name);

    for(size_t i = 0; i < rows_number; i++)
    {
        if(!find_values.contains((*this)(i,column_index)))
        {
          (*this)(i,column_index) = replace_value;
        }
    }
}


template <class T>
void Matrix<T>::replace_column_less_than_string(const string& name, const double& value, const T& replace)
{
    const size_t column_index = get_column_index(name);

    const Vector<size_t> row_indices = get_column(name).string_to_double().get_indices_less_than(value);

    for(size_t i = 0; i < row_indices.size(); i++)
    {
        (*this)(row_indices[i], column_index) = replace;
    }
}


template <class T>
void Matrix<T>::replace_column_contains(const string& column_name, const string& find_what, const string& replace_with)
{
    const size_t column_index = get_column_index(column_name);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index).find(find_what) != string::npos)
        {
            (*this)(i,column_index) = replace_with;
        }
    }
}


template <class T>
size_t Matrix<T>::count_column_contains(const string& column_name, const string& find_what) const
{
    const size_t column_index = get_column_index(column_name);

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index).find(find_what) != string::npos)
        {
             count++;
        }
    }

    return(count);
}


template <class T>
Vector<size_t> Matrix<T>::count_column_occurrences(const T& value) const
{
    Vector<size_t> occurrences(columns_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) == value) occurrences[j]++;
        }
    }

    return occurrences;
}


template <class T>
bool Matrix<T>::has_column_value(const size_t& column_index, const T& value) const
{
    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == value)
        {
             return(true);
        }
    }

    return(false);
}


/// Initializes all the elements in the matrix with random values comprised between a minimum and a maximum
/// values.
/// @param minimum Minimum possible value.
/// @param maximum Maximum possible value.

template <class T>
void Matrix<T>::randomize_uniform(const double& minimum, const double& maximum)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(minimum > maximum)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const double&, const double&) const method.\n"
             << "Minimum value must be less or equal than maximum value.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
     (*this)[i] = static_cast<T>(calculate_random_uniform(minimum, maximum));
   }
}


/// Initializes all the elements in the matrix with random values comprised between a minimum and a maximum
/// values for each element.
/// @param minimums Minimum possible values.
/// @param maximums Maximum possible values.

template <class T>
void Matrix<T>::randomize_uniform(const Vector<double>& minimums, const Vector<double>& maximums)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(minimums.size() != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of minimums must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   if(maximums.size() != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of maximums must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   if(minimums > maximums)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Minimums must be less or equal than maximums.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<double> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
        column.randomize_uniform(minimums[i], maximums[i]);

        set_column(i, column);
   }
}


/// Initializes all the elements in the matrix with random values comprised between a minimum and a maximum
/// values for each element.
/// @param minimum Minimum possible values.
/// @param maximum Maximum possible values.

template <class T>
void Matrix<T>::randomize_uniform(const Matrix<double>& minimum, const Matrix<double>& maximum)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(minimum > maximum)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Matrix<double>&, const Matrix<double>&) const method.\n"
             << "Minimum values must be less or equal than their respective maximum values.\n";

      throw logic_error(buffer.str());
   }

   #endif


   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] = calculate_random_uniform(minimum[i], maximum[i]);
   }
}


/// Assigns random values to each element in the matrix, taken from a normal distribution with
/// a given mean and a given standard deviation.
/// @param mean Mean value of uniform distribution.
/// @param standard_deviation Standard deviation value of uniform distribution.

template <class T>
void Matrix<T>::randomize_normal(const double& mean, const double& standard_deviation)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(standard_deviation < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const double&, const double&) method.\n"
             << "Standard deviation must be equal or greater than zero.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
     (*this)[i] = calculate_random_normal(mean, standard_deviation);
   }
}


/// Assigns random values to each element in the matrix, taken from a normal distribution with
/// a given mean and a given standard deviation.
/// @param means Means values of uniform distribution.
/// @param standard_deviations Standard deviations values of uniform distribution.

template <class T>
void Matrix<T>::randomize_normal(const Vector<double>& means, const Vector<double>& standard_deviations)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(means.size() != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of means must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   if(standard_deviations.size() != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of standard deviations must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   if(means < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Means must be less or equal than zero.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<double> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
        column.randomize_normal(means[i], standard_deviations[i]);

        set_column(i, column);
   }
}


/// Assigns random values to each element in the vector, taken from normal distributions with
/// given means and standard deviations for each element.
/// @param mean Mean values of uniform distributions.
/// @param standard_deviation Standard deviation values of uniform distributions.

template <class T>
void Matrix<T>::randomize_normal(const Matrix<double>& mean, const Matrix<double>& standard_deviation)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(standard_deviation < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Matrix<double>&, const Matrix<double>&) const method.\n"
             << "Standard deviations must be equal or greater than zero.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] = calculate_random_uniform(mean[i], standard_deviation[i]);
   }
}


/// Sets the diagonal elements in the matrix with ones and the rest elements with zeros. The matrix
/// must be square.

template <class T>
void Matrix<T>::initialize_identity()
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      cout << "OpenNN Exception: Matrix Template.\n"
                << "initialize_identity() const method.\n"
                << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

  (*this).initialize(0);

   for(size_t i = 0; i < rows_number; i++)
   {
     (*this)(i,i) = 1;
   }
}


/// Sets the diagonal elements in the matrix with a given value and the rest elements with zeros.
/// The matrix must be square.

template <class T>
void Matrix<T>::initialize_diagonal(const T& value)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      cout << "OpenNN Exception: Matrix Template.\n"
                << "initialize_diagonal(const T&) const method.\n"
                << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(i==j)
         {
           (*this)(i,j) = value;
         }
         else
         {
           (*this)(i,j) = 0;
         }
      }
   }
}


/// Returns an approximation of the inverse Hessian matrix according to the Davidon-Fletcher-Powel
///(DFP) algorithm.
/// @param old_parameters A previous set of parameters.
/// @param old_gradient The gradient of the error function for that previous set of parameters.
/// @param old_inverse_Hessian The Hessian of the error function for that previous set of parameters.
/// @param parameters Actual set of parameters.
/// @param gradient The gradient of the error function for the actual set of parameters.

template <class T>
void Matrix<T>::update_DFP_inverse_Hessian(const Vector<double>& old_parameters, const Vector<double>& parameters,
                                           const Vector<double>& old_gradient, const Vector<double>& gradient)
{
   ostringstream buffer;

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

//   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

//   const size_t parameters_number = neural_network_pointer->get_parameters_number();

//   const size_t old_parameters_size = old_parameters.size();
//   const size_t parameters_size = parameters.size();

//   if(old_parameters_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Size of old parameters vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }
//   else if(parameters_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Size of parameters vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

//   const size_t old_gradient_size = old_gradient.size();
//   const size_t gradient_size = gradient.size();

//   if(old_gradient_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Size of old gradient vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }
//   else if(gradient_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Size of gradient vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

//   const size_t rows_number = old_inverse_Hessian.get_rows_number();
//   const size_t columns_number = old_inverse_Hessian.get_columns_number();

//   if(rows_number != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Number of rows in old inverse Hessian must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }
//   else if(columns_number != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Number of columns in old inverse Hessian must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

   #endif

   // Parameters difference Vector

   const Vector<double> parameters_difference = parameters - old_parameters;

   // Control sentence(if debug)

   if(parameters_difference.calculate_absolute_value() < numeric_limits<double>::min())
   {
      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
             << "Parameters difference vector is zero.\n";

      throw logic_error(buffer.str());
   }

   // Gradient difference Vector

   const Vector<double> gradient_difference = gradient - old_gradient;

   if(gradient_difference.calculate_absolute_value() < 1.0e-50)
   {
      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
             << "Gradient difference vector is zero.\n";

      throw logic_error(buffer.str());
   }

//   if(this->calculate_absolute_value() < 1.0e-50)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Old inverse Hessian matrix is zero.\n";

//      throw logic_error(buffer.str());
//   }

   const double parameters_dot_gradient = parameters_difference.dot(gradient_difference);

//   if(fabs(parameters_dot_gradient) < 1.0e-50)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Denominator of first term is zero.\n";

//      throw logic_error(buffer.str());
//   }
//   else if(fabs(gradient_difference.dot(old_inverse_Hessian).dot(gradient_difference)) < 1.0e-50)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_DFP_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Denominator of second term is zero.\n";

//      throw logic_error(buffer.str());
//   }

   const Vector<double> Hessian_dot_gradient_difference = this->dot(gradient_difference);

   Matrix<double> term_1;
   term_1.direct(parameters_difference,parameters_difference);
   term_1 /= parameters_dot_gradient;

   (*this) += term_1;

   vector<T>().swap(term_1);

   Matrix<double> term_2;
   term_2.direct(Hessian_dot_gradient_difference, Hessian_dot_gradient_difference);
   term_2 /= gradient_difference.dot(Hessian_dot_gradient_difference);

   (*this) -= term_2;

   vector<T>().swap(term_2);
}


/// Returns an approximation of the inverse Hessian matrix according to the
/// Broyden-Fletcher-Goldfarb-Shanno(BGFS) algorithm.
/// @param old_parameters A previous set of parameters.
/// @param old_gradient The gradient of the error function for that previous set of parameters.
/// @param old_inverse_Hessian The Hessian of the error function for that previous set of parameters.
/// @param parameters Actual set of parameters.
/// @param gradient The gradient of the error function for the actual set of parameters.

template <class T>
void Matrix<T>::update_BFGS_inverse_Hessian(const Vector<double>& old_parameters, const Vector<double>& parameters,
const Vector<double>& old_gradient, const Vector<double>& gradient)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

//   ostringstream buffer;

//   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

//   const size_t parameters_number = neural_network_pointer->get_parameters_number();

//   const size_t old_parameters_size = old_parameters.size();
//   const size_t parameters_size = parameters.size();

//   if(old_parameters_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Size of old parameters vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }
//   else if(parameters_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Size of parameters vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

//   const size_t old_gradient_size = old_gradient.size();

//   if(old_gradient_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method."
//             << endl
//             << "Size of old gradient vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

//   const size_t gradient_size = gradient.size();

//   if(gradient_size != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method."
//             << endl
//             << "Size of gradient vector must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

//   const size_t rows_number = old_inverse_Hessian.get_rows_number();
//   const size_t columns_number = old_inverse_Hessian.get_columns_number();

//   if(rows_number != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Number of rows in old inverse Hessian must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

//   if(columns_number != parameters_number)
//   {
//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Number of columns in old inverse Hessian must be equal to number of parameters.\n";

//      throw logic_error(buffer.str());
//   }

   #endif

   // Parameters difference Vector

   const Vector<double> parameters_difference = parameters - old_parameters;

//   if(parameters_difference.calculate_absolute_value() < 1.0e-50)
//   {
//       ostringstream buffer;

//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Parameters difference vector is zero.\n";

//      throw logic_error(buffer.str());
//   }

   // Gradient difference Vector

   const Vector<double> gradient_difference = gradient - old_gradient;

//   if(gradient_difference.calculate_absolute_value() < numeric_limits<double>::min())
//   {
//       ostringstream buffer;

//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Gradient difference vector is zero.\n";

//      throw logic_error(buffer.str());
//   }

//   if(old_inverse_Hessian.calculate_absolute_value() < 1.0e-50)
//   {
//       ostringstream buffer;

//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Matrix<double> calculate_BFGS_inverse_Hessian(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) method.\n"
//             << "Old inverse Hessian matrix is zero.\n";

//	  throw logic_error(buffer.str());
//   }

   // BGFS Vector

   const double parameters_dot_gradient = parameters_difference.dot(gradient_difference);
   const Vector<double> Hessian_dot_gradient = this->dot(gradient_difference);
   const double gradient_dot_Hessian_dot_gradient = gradient_difference.dot(Hessian_dot_gradient);

   const Vector<double> BFGS = parameters_difference/parameters_dot_gradient
   - Hessian_dot_gradient/gradient_dot_Hessian_dot_gradient;

   // Calculate inverse Hessian approximation

   Matrix<double> term_1;
   term_1.direct(parameters_difference, parameters_difference);
   term_1 /= parameters_dot_gradient;

   (*this) += term_1;

   vector<T>().swap(term_1);

   Matrix<double> term_2;
   term_2.direct(Hessian_dot_gradient, Hessian_dot_gradient);
   term_2 /= gradient_dot_Hessian_dot_gradient;

   (*this) -= term_2;

   vector<T>().swap(term_2);

   Matrix<double> term_3;
   term_3.direct(BFGS, BFGS);
   term_3 *= gradient_dot_Hessian_dot_gradient;

   (*this) += term_3;

   vector<T>().swap(term_3);
}



/// Returns the sum of all the elements in the matrix.

template <class T>
T Matrix<T>::calculate_sum() const
{
   T sum = 0;

   for(size_t i = 0; i < this->size(); i++)
   {
        sum += (*this)[i];
   }

   return(sum);
}


/// Returns the sum of all the rows in the matrix.

template <class T>
Vector<T> Matrix<T>::calculate_rows_sum() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(this->empty())
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "Vector<T> calculate_rows_sum() const method.\n"
                 << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

    #endif

   Vector<T> rows_sum(rows_number, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
            rows_sum[i] += (*this)(i,j);
       }
   }

   return(rows_sum);
}


template <class T>
Vector<int> Matrix<T>::calculate_rows_sum_int() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(this->empty())
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "Vector<T> calculate_rows_sum() const method.\n"
                 << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

    #endif

   Vector<int> rows_sum(rows_number, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
            rows_sum[i] += stoi((*this)(i,j));
       }
   }

   return(rows_sum);
}


template <class T>
Vector<T> Matrix<T>::calculate_columns_sum() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(this->empty())
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "Vector<T> calculate_columns_sum() const method.\n"
                 << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

    #endif

   Vector<T> columns_sum(columns_number, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
            columns_sum[j] += (*this)(i,j);
       }
   }

   return(columns_sum);
}


template <class T>
T Matrix<T>::calculate_column_sum(const size_t& column_index) const
{
   T column_sum = 0.0;

   for(size_t i = 0; i < rows_number; i++)
   {
       column_sum += (*this)(i,column_index);
   }

   return(column_sum);
}




template <class T>
Vector<T> Matrix<T>::calculate_columns_mean() const
{
   return calculate_columns_sum()/static_cast<double>(rows_number);
}


/// Sums the values of a given row with the values of a given vector.
/// The size of the vector must be equal to the number of columns.

template <class T>
void Matrix<T>::sum_row(const size_t& row_index, const Vector<T>& vector)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(vector.size() != columns_number)
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "void sum_row(const size_t&, const Vector<T>&) method.\n"
                 << "Size of vector must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    for(size_t j = 0; j < columns_number; j++)
    {
       (*this)(row_index,j) += vector[j];
    }
}


/// Sums the values of all the rows of this matrix with the values of a given vector.
/// The size of the vector must be equal to the number of columns.

template <class T>
Matrix<T> Matrix<T>::sum_rows(const Vector<T>& vector) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(vector.size() != columns_number)
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "void sum_rows(const Vector<T>&) method.\n"
                 << "Size of vector must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<T> new_matrix(rows_number, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {          
           new_matrix(i,j) = (*this)(i,j) + vector[j];
        }
    }

    return new_matrix;
}


/// Subtracts the values of all the rows of this matrix with the values of a given vector.
/// The size of the vector must be equal to the number of columns.

template <class T>
Matrix<T> Matrix<T>::subtract_rows(const Vector<T>& vector) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(vector.size() != columns_number)
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "void subtract_rows(const Vector<T>&) method.\n"
                 << "Size of vector must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<T> new_matrix(rows_number, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
           new_matrix(i,j) = (*this)(i,j) - vector[j];
        }
    }

    return new_matrix;
}


template <class T>
Matrix<T> Matrix<T>::multiply_rows(const Vector<T>& vector) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(vector.size() != columns_number)
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "void multiply_rows(const Vector<T>&) method.\n"
                 << "Size of vector must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<T> new_matrix(rows_number, columns_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
           new_matrix(i,j) *= vector[j];
        }
    }

    return new_matrix;
}


template <class T>
Vector<Matrix<T>> Matrix<T>::multiply_rows(const Matrix<T>& matrix) const
{
    const size_t points_number = matrix.get_rows_number();

    Vector<Matrix<T>> new_vector_matrix(points_number);

    for(size_t point_number = 0; point_number < points_number; point_number++)
    {
        new_vector_matrix[point_number].set(rows_number, columns_number, 0.0);

        for(size_t i = 0; i < rows_number; i++)
        {
            for(size_t j = 0; j < columns_number; j++)
            {
               new_vector_matrix[point_number](i,j) = (*this)(i,j)*matrix(point_number,j);
            }
        }
    }

    return new_vector_matrix;
}


/// Returns the trace of the matrix, which is defined to be the sum of the main diagonal elements.
/// The matrix must be square.

template <class T>
double Matrix<T>::calculate_trace() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(!is_square())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_trace() const method.\n"
             << "Matrix is not square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   double trace = 0.0;

   for(size_t i = 0; i < rows_number; i++)
   {
      trace += (*this)(i,i);
   }

   return(trace);
}


/// Returns the softmax vector of this matrix,
/// whose elements sum one, and can be interpreted as probabilities.

template <class T> Matrix<double> Matrix<T>::calculate_softmax() const
{
  Matrix<T> softmax(rows_number, columns_number);

  for(size_t j = 0; j < rows_number; j++)
  {
      T sum = 0;

      for(size_t i = 0; i < columns_number; i++)
      {
        sum += exp((*this)(j,i));
      }

      for(size_t i = 0; i < columns_number; i++)
      {
        softmax(j,i) = exp((*this)(j,i)) / sum;
      }
  }

  return(softmax);
}


/// Returns a vector with the mean values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

template <class T>
Vector<double> Matrix<T>::calculate_mean() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean() const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Mean

   Vector<double> mean(columns_number, 0.0);

   for(size_t j = 0; j < columns_number; j++)
   {
      for(size_t i = 0; i < rows_number; i++)
      {
         mean[j] += (*this)(i,j);
      }

      mean[j] /= static_cast<double>(rows_number);
   }

   return(mean);
}


/// Returns a vector with the mean values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

template <class T>
double Matrix<T>::calculate_mean(const size_t& column_index) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_mean(const size_t&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_mean(const size_t&) const method.\n"
             << "Index of column must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Mean

   double mean = 0.0;

    for(size_t i = 0; i < rows_number; i++)
    {
        mean += (*this)(i,column_index);
    }

   mean /= static_cast<double>(rows_number);

   return(mean);
}


/// Returns a vector with the mean values of given columns.
/// The size of the vector is equal to the size of the column indices vector.
/// @param column_indices Indices of columns.

template <class T>
Vector<double> Matrix<T>::calculate_mean(const Vector<size_t>& column_indices) const
{
   const size_t column_indices_size = column_indices.size();

   size_t column_index;

   // Mean

   Vector<double> mean(column_indices_size, 0.0);

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      for(size_t i = 0; i < rows_number; i++)
      {
         mean[j] += (*this)(i,column_index);
      }

      mean[j] /= static_cast<double>(rows_number);
   }

   return(mean);
}


/// Returns a vector with the mean values of given columns for given rows.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.

template <class T>
Vector<double> Matrix<T>::calculate_mean(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t column_indices_size = column_indices.size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   size_t row_index;
   size_t column_index;

   // Mean

   Vector<double> mean(column_indices_size, 0.0);

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      for(size_t i = 0; i < row_indices_size; i++)
      {
         row_index = row_indices[i];

         mean[j] += (*this)(row_index,column_index);
      }

      mean[j] /= static_cast<double>(rows_number);
   }

   return(mean);
}


template <class T>
Vector<double> Matrix<T>::calculate_mean_monthly_occurrences() const
{
    Vector<double> row;

    Vector<double> month_average(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        row = (*this).get_row(i);

        row = row.filter_not_equal_to(0);

        month_average[i] = row.calculate_mean();
    }

    return(month_average);
}


template <class T>
void Matrix<T>::check_time_column(const size_t& column_index, const size_t& period) const
{
    if(columns_number <= 1) return;

    for(size_t i = 1; i < columns_number; i++)
    {
        if((*this)(i, column_index) != (*this)(i-1, column_index) + period)
        {
           const string message = "OpenNN Exception: Matrix template.\n"
                                  "void check_time_column(const size_t&, const size_t&) const method.\n"
                                  "Row " + to_string(i) + " has a wrong period.\n";

           throw logic_error(message);
        }
    }
}


template <class T>
void Matrix<T>::check_time_column(const string& column_name, const size_t& period) const
{
    const size_t column_index = get_column_index(column_name);

    check_time_column(column_index, period);
}


/// Returns a vector with the mean values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector<double> Matrix<T>::calculate_mean_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<size_t> row_indices(0, 1, rows_number-1);
    Vector<size_t> column_indices(0, 1, columns_number-1);

    return(calculate_mean_missing_values(row_indices, column_indices, missing_indices));
}


/// @todo

template <class T>
Matrix<double> Matrix<T>::calculate_time_series_mean_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<size_t> row_indices(0, 1, rows_number-1);
    Vector<size_t> column_indices(0, 1, columns_number-1);

    return(calculate_time_series_mean_missing_values(row_indices, column_indices, missing_indices));
}


/// Returns a vector with the mean values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector<double> Matrix<T>::calculate_mean_missing_values(const Vector<size_t>& row_indices,
                                                        const Vector<size_t>& column_indices,
                                                        const Vector< Vector<size_t> >& missing_indices) const
{
   const size_t column_indices_size = column_indices.size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t row_indices_size = row_indices.size();

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, Vector< Vector<size_t> >&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   // Mean

   Vector<double> mean(column_indices_size, 0.0);

   Vector< Vector<size_t> > used_rows(column_indices_size);

    #pragma omp parallel for schedule(dynamic)

   for(int i = 0; i < static_cast<int>(column_indices_size); i++)
   {
       used_rows[static_cast<size_t>(i)] = row_indices.get_difference(missing_indices[static_cast<size_t>(i)]);
   }

    #pragma omp parallel for schedule(dynamic)

   for(int j = 0; j < static_cast<int>(column_indices_size); j++)
   {
      const size_t column_index = column_indices[static_cast<size_t>(j)];

      size_t count = 0;

      const size_t current_rows_number = used_rows[static_cast<size_t>(j)].size();

      for(size_t i = 0; i < current_rows_number; i++)
      {
         const size_t row_index = used_rows[static_cast<size_t>(j)][i];

         mean[j] += (*this)(row_index,column_index);
         count++;
      }

      if(count != 0)
      {
          mean[static_cast<size_t>(j)] /= static_cast<double>(count);
      }
   }

   return(mean);
}


/// @todo

template <class T>
Matrix<double> Matrix<T>::calculate_time_series_mean_missing_values(const Vector<size_t>& row_indices,
                                                                    const Vector<size_t>& column_indices,
                                                                    const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t column_indices_size = column_indices.size();
    const size_t row_indices_size = row_indices.size();

    Matrix<double> means(row_indices_size, column_indices_size, 0.0);

    size_t column_index = 0;

    Vector<size_t> mean_indices;
    double step = 0.0;
    size_t current_row;
    double previous_value;
    double next_value;

    for(size_t j = 0; j < column_indices_size; j++)
    {
       column_index = column_indices[j];

       const size_t missing_indices_size = missing_indices[column_index].size();

       for(size_t i = 0; i < missing_indices_size; i++)
       {
           if(missing_indices[column_index][i+1] == missing_indices[column_index][i] + 1
           && i != missing_indices_size)
           {
               if(!mean_indices.contains(missing_indices[column_index][i]))
               {
                   mean_indices.push_back(missing_indices[column_index][i]);
               }

               if(!mean_indices.contains(missing_indices[column_index][i+1]))
               {
                  mean_indices.push_back(missing_indices[column_index][i+1]);
               }
           }
           else
           {
               if(!mean_indices.contains(missing_indices[column_index][i]))
               {
                   mean_indices.push_back(missing_indices[column_index][i]);
               }

               const size_t mean_indices_size = mean_indices.size();
               const size_t minimum = mean_indices.calculate_minimum();
               const size_t maximum = mean_indices.calculate_maximum();
               const size_t previous_index = minimum - 1;
               const size_t next_index = maximum + 1;

               if(minimum == 0 )
               {
                   step = (*this)(next_index + 1,column_index) -(*this)(next_index,column_index);

                   next_value = (*this)(next_index,column_index);

                   for(int k = 0; k < static_cast<int>(mean_indices_size); k++)
                   {
                       current_row = mean_indices[static_cast<size_t>(k)];

                       means(current_row,column_index) = next_value - step *(mean_indices_size - static_cast<size_t>(k));
                   }

                   mean_indices.clear();
                   step = 0.0;
               }
               else if(maximum == row_indices_size - 1)
               {
                   step = (*this)(previous_index,column_index) -(*this)(previous_index - 1,column_index);

                   previous_value = (*this)(previous_index,column_index);

                   for(int k = 0; k < static_cast<int>(mean_indices_size); k++)
                   {
                       current_row = mean_indices[static_cast<size_t>(k)];

                       means(current_row,column_index) = previous_value + step *(k+1);
                   }

                   mean_indices.clear();
                   step = 0.0;
               }
               else
               {
                   step = ((*this)(next_index,column_index) -(*this)(previous_index,column_index) ) /static_cast<double>(mean_indices_size + 1.0);

                   previous_value = (*this)(previous_index,column_index);

                   for(int k = 0; k < static_cast<int>(mean_indices_size); k++)
                   {
                       current_row = mean_indices[static_cast<size_t>(k)];

                       means(current_row,column_index) = previous_value + step *(k+1);
                   }

                   mean_indices.clear();
                   step = 0.0;
               }
           }
       }
    }

    return(means);
}


/// Returns a vector of vectors with the mean and standard deviation values of all the matrix columns.
/// The size of the vector is two.
/// The size of each element is equal to the number of columns in the matrix.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_mean_standard_deviation() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation() const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Mean

   Vector<double> mean(columns_number, 0.0);
   Vector<double> standard_deviation(columns_number, 0.0);

   for(size_t i = 0; i < columns_number; i++)
   {
      mean[i] = get_column(i).calculate_mean();
      standard_deviation[i] = get_column(i).calculate_standard_deviation();
   }

   return {mean, standard_deviation};
}


/// Returns a vector of vectors with the mean and standard deviation values of given columns.
/// The size of the vector is two.
/// The size of each element is equal to the size of the column indices vector.
/// @param column_indices Indices of columns.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_mean_standard_deviation(const Vector<size_t>& column_indices) const
{
   const size_t column_indices_size = column_indices.size();

   Vector<double> mean(column_indices_size);
   Vector<double> standard_deviation(column_indices_size);

   size_t column_index;

   Vector<double> column(rows_number);

   for(size_t i = 0; i < column_indices_size; i++)
   {
      column_index = column_indices[i];

      column = get_column(column_index);

      mean[i] = column.calculate_mean();
      standard_deviation[i] = column.calculate_standard_deviation();
   }

   return {mean, standard_deviation};
}


/// Returns a vector of vectors with the mean and standard deviation values of given columns for given rows.
/// The size of the vector is two.
/// The size of each element is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_mean_standard_deviation(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t column_indices_size = column_indices.size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Row indices size must be equal or less than rows number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   size_t row_index;
   size_t column_index;

   // Mean

   Vector<double> mean(column_indices_size, 0.0);

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      mean[j] = 0.0;

      for(size_t i = 0; i < row_indices_size; i++)
      {
         row_index = row_indices[i];

         mean[j] += (*this)(row_index,column_index);
      }

      mean[j] /= static_cast<double>(rows_number);
   }

   // Standard deviation

   Vector<double> standard_deviation(column_indices_size, 0.0);

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      standard_deviation[j] = 0.0;

      for(size_t i = 0; i < row_indices_size; i++)
      {
         row_index = row_indices[i];

         standard_deviation[j] += ((*this)(row_index,column_index) - mean[j])*((*this)(row_index,column_index) - mean[j]);
      }

      standard_deviation[j] = sqrt(standard_deviation[j]/(rows_number-1.0));
   }

   // Mean and standard deviation

   return {mean, standard_deviation};
}


/// Returns the minimum value from all elements in the matrix.

template <class T>
T Matrix<T>::calculate_minimum() const
{
   T minimum = static_cast<T>(numeric_limits<double>::max());

   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] < minimum)
         {
            minimum = (*this)[i];
         }
   }

   return(minimum);
}


/// Returns a vector with the median values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

template <class T>
Vector<double> Matrix<T>::calculate_median() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_median() const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // median

   Vector<double> median(columns_number, 0.0);

   for(size_t j = 0; j < columns_number; j++)
   {
       Vector<T> sorted_column(this->get_column(j));

       sort(sorted_column.begin(), sorted_column.end(), less<double>());

       if(rows_number % 2 == 0)
       {
         median[j] = (sorted_column[rows_number*2/4] + sorted_column[rows_number*2/4+1])/2;
       }
       else
       {
         median[j] = sorted_column[rows_number*2/4];
       }
   }

   return(median);
}


/// Returns a vector with the median values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

template <class T>
double Matrix<T>::calculate_median(const size_t& column_index) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_median(const size_t&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_median(const size_t&) const method.\n"
             << "Index of column must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // median

   double median = 0.0;

   Vector<T> sorted_column(this->get_column(column_index));

   sort(sorted_column.begin(), sorted_column.end(), less<double>());

   if(rows_number % 2 == 0)
   {
     median = (sorted_column[rows_number*2/4] + sorted_column[rows_number*2/4+1])/2;
   }
   else
   {
     median = sorted_column[rows_number*2/4];
   }

   return(median);
}


/// Returns a vector with the median values of given columns.
/// The size of the vector is equal to the size of the column indices vector.
/// @param column_indices Indices of columns.

template <class T>
Vector<double> Matrix<T>::calculate_median(const Vector<size_t>& column_indices) const
{
   const size_t column_indices_size = column_indices.size();

   size_t column_index;

   // median

   Vector<double> median(column_indices_size, 0.0);

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      Vector<T> sorted_column(this->get_column(column_index));

      sort(sorted_column.begin(), sorted_column.end(), less<double>());

      if(rows_number % 2 == 0)
      {
        median[j] = (sorted_column[rows_number * 2 / 4] + sorted_column[rows_number * 2 / 4 + 1])/2;
      }
      else
      {
        median[j] = sorted_column[rows_number * 2 / 4];
      }
   }

   return(median);
}


/// Returns a vector with the median values of given columns for given rows.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.

template <class T>
Vector<double> Matrix<T>::calculate_median(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t column_indices_size = column_indices.size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   size_t column_index;

   // median

   Vector<double> median(column_indices_size, 0.0);

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      Vector<T> sorted_column(this->get_column(column_index, row_indices));

      sort(sorted_column.begin(), sorted_column.end(), less<double>());

      if(row_indices_size % 2 == 0)
      {
        median[j] = (sorted_column[row_indices_size*2/4] + sorted_column[row_indices_size*2/4 + 1])/2;
      }
      else
      {
        median[j] = sorted_column[row_indices_size * 2 / 4];
      }
   }

   return(median);
}


/// Returns a vector with the median values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector<double> Matrix<T>::calculate_median_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<size_t> row_indices(0, 1, rows_number-1);
    Vector<size_t> column_indices(0, 1, columns_number-1);

    return(calculate_median_missing_values(row_indices, column_indices, missing_indices));
}


/// Returns a vector with the median values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector<double> Matrix<T>::calculate_median_missing_values(const Vector<size_t>& row_indices,
                                                          const Vector<size_t>& column_indices,
                                                          const Vector< Vector<size_t> >& missing_indices) const
{
   const size_t column_indices_size = column_indices.size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t row_indices_size = row_indices.size();

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, Vector< Vector<size_t> >&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   // median

   Vector<double> median(column_indices_size, 0.0);

   Vector< Vector<size_t> > used_rows(column_indices_size);

    #pragma omp parallel for schedule(dynamic)

   for(int i = 0; i < static_cast<int>(column_indices_size); i++)
   {
       used_rows[static_cast<size_t>(i)] = row_indices.get_difference(missing_indices[static_cast<size_t>(i)]);
   }

    #pragma omp parallel for schedule(dynamic)

   for(int j = 0; j < static_cast<int>(column_indices_size); j++)
   {
      const size_t column_index = column_indices[static_cast<size_t>(j)];

      const size_t current_rows_number = used_rows[static_cast<size_t>(j)].size();

      Vector<T> sorted_column(this->get_column(column_index, used_rows[j]));

      sort(sorted_column.begin(), sorted_column.end(), less<double>());

      if(current_rows_number % 2 == 0)
      {
        median[j] = (sorted_column[current_rows_number*2/4] + sorted_column[current_rows_number*2/4+1])/2;
      }
      else
      {
        median[j] = sorted_column[current_rows_number * 2 / 4];
      }
   }

   return(median);
}


///Returns the percentage of missing values in each column.

template <class T>
Vector< double > Matrix<T>::calculate_missing_values_percentage() const
{
    Vector< double > missing_values(columns_number);

    Vector<T> column(rows_number);

    for(size_t i = 0; i < columns_number; i++)
    {
       column = get_column(i);

       missing_values[i] = column.count_equal_to("NA")*100.0/static_cast<double>(rows_number-1.0);
    }

    return(missing_values);
}

/// Returns the matrix p-norm by rows.

template <class T>
Vector< double > Matrix<T>::calculate_LP_norm(const double& p) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<double> calculate_LP_norm(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

      // Control sentence(if debug)

    Vector<double> norm(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            norm[i] += pow(fabs((*this)(i,j)), p);
        }

        norm[i] = pow(norm[i], 1.0 / p);
    }

    return(norm);
}


/// Returns the gradient of the matrix norm.

template <class T>
Matrix< double > Matrix<T>::calculate_LP_norm_gradient(const double& p) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Matrix<double> calculate_p_norm_gradient(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

      Matrix<double> gradient(rows_number, columns_number);

      const Vector<double> p_norm = calculate_LP_norm(p);

      if(p_norm == 0.0) {
        gradient.initialize(0.0);
      } else {
        for(size_t i = 0; i < rows_number; i++) {
            for(size_t j = 0; j < columns_number; j++) {
                gradient(i,j) =
                 (*this)(i,j) * pow(fabs((*this)(i,j)), p - 2.0) / pow(p_norm[i], p - 1.0);
            }
        }
      }

      return(gradient);

}

/// Returns the maximum value from all elements in the matrix.

template <class T>
T Matrix<T>::calculate_maximum() const
{
    T maximum = static_cast<T>(-numeric_limits<double>::max());

    for(size_t i = 0; i < this->size(); i++)
    {
          if((*this)[i] > maximum)
          {
             maximum = (*this)[i];
          }
    }

   return(maximum);
}


/// Returns a vector of vectors with the minimum and maximum values of all the matrix columns.
/// The size of the vector is two.
/// The size of each element is equal to the number of columns in the matrix.

template <class T>
Vector< Vector<T> > Matrix<T>::calculate_minimum_maximum() const
{
   Vector<T> minimum(columns_number,static_cast<T>(numeric_limits<double>::max()));
   Vector<T> maximum(columns_number,static_cast<T>(-numeric_limits<double>::max()));

   for(size_t j = 0; j < columns_number; j++)
   {
      for(size_t i = 0; i < rows_number; i++)
      {
         if((*this)(i,j) < minimum[j])
         {
            minimum[j] = (*this)(i,j);
         }

         if((*this)(i,j) > maximum[j])
         {
            maximum[j] = (*this)(i,j);
         }
      }
   }

   return {minimum, maximum};
}


/// Returns a vector of vectors with the minimum and maximum values of given columns.
/// The size of the vector is two.
/// The size of each element is equal to the size of the column indices vector.
/// @param column_indices Indices of columns.

template <class T>
Vector< Vector<T> > Matrix<T>::calculate_minimum_maximum(const Vector<size_t>& column_indices) const
{
   const size_t column_indices_size = column_indices.size();

   #ifdef __OPENNN_DEBUG__

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template."
                << "Vector<T> calculate_minimum_maximum(const Vector<size_t>&) const method.\n"
                << "Index of column must be less than number of columns.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   size_t column_index;

   Vector<T> minimum(column_indices_size, static_cast<T>(numeric_limits<double>::max()));
   Vector<T> maximum(column_indices_size,static_cast<T>(-numeric_limits<double>::max()));

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      for(size_t i = 0; i < rows_number; i++)
      {
         if((*this)(i,column_index) < minimum[j])
         {
            minimum[j] = (*this)(i,column_index);
         }

         if((*this)(i,column_index) > maximum[j])
         {
            maximum[j] = (*this)(i,column_index);
         }
      }
   }

   return {minimum, maximum};
}


/// Returns a vector of vectors with the minimum and maximum values of given columns for given rows.
/// The size of the vector is two.
/// The size of each element is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.

template <class T>
Vector< Vector<T> > Matrix<T>::calculate_minimum_maximum(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t column_indices_size = column_indices.size();

   Vector<T> minimum(column_indices_size,static_cast<T>(numeric_limits<double>::max()));
   Vector<T> maximum(column_indices_size,static_cast<T>(-numeric_limits<double>::max()));

   size_t row_index;
   size_t column_index;

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      for(size_t i = 0; i < row_indices_size; i++)
      {
         row_index = row_indices[i];

         if((*this)(row_index,column_index) < minimum[j])
         {
            minimum[j] = (*this)(row_index,column_index);
         }

         if((*this)(row_index,column_index) > maximum[j])
         {
            maximum[j] = (*this)(row_index,column_index);
         }
      }
   }

   return {minimum, maximum};
}


/// Returns the minimum value from all elements in the matrix.

template <class T>
T Matrix<T>::calculate_column_minimum(const size_t& column_index) const
{
   T minimum = static_cast<T>(numeric_limits<double>::max());

   for(size_t i = 0; i < rows_number; i++)
   {
         if((*this)(i,column_index) < minimum)
         {
            minimum = (*this)[i];
         }
   }

   return(minimum);
}


/// Returns the maximum value from all elements in the matrix.

template <class T>
T Matrix<T>::calculate_column_maximum(const size_t& column_index) const
{
    T maximum = static_cast<T>(-numeric_limits<double>::max());

    for(size_t i = 0; i < rows_number; i++)
    {
          if((*this)(i,column_index) > maximum)
          {
             maximum = (*this)[i];
          }
    }

   return(maximum);
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the given integers. The matrix must have 2 columns, the first
/// one containing the integers and the second one the corresponding values.

template <class T>
Vector<T> Matrix<T>::calculate_means_integers() const
{
    #ifdef __OPENNN_DEBUG__

    if(columns_number != 2)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector<T> calculate_means_integers(const Vector<size_t>&) const method.\n"
              << "Number of columns must be two.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Vector<size_t> integers = this->get_column(0).get_integer_elements(get_rows_number());

    return calculate_means_integers(integers);
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the given integers. The matrix must have 2 columns, the first
/// one containing the integers and the second one the corresponding values.
/// @param integers Integers.

template <class T>
Vector<T> Matrix<T>::calculate_means_integers(const Vector<size_t>& integers) const
{
    const size_t integers_number = integers.size();

    #ifdef __OPENNN_DEBUG__

    if(integers_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector<T> calculate_means_integers(const Vector<size_t>&) const method.\n"
              << "Number of integers must be greater than 0.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = this->get_rows_number();

    Vector<T> means(integers_number);

    T sum = 0.0;
    size_t count = 0;

    for(size_t i = 0; i < integers_number; i++)
    {
        sum = 0.0;
        count = 0;

        const size_t current_integer = integers[i];

        for(unsigned j = 0; j < rows_number; j++)
        {
            if((*this)(j,0) == current_integer)
            {
                sum += (*this)(j,1);
                count++;
            }
        }

        if(count != 0)
        {
            means[i] = static_cast<T>(sum)/static_cast<T>(count);
        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the given integers. The matrix must have 2 columns, the first
/// one containing the integers and the second one the corresponding values.
/// @param missing_indices Missing values.

template <class T>
Vector<T> Matrix<T>::calculate_means_integers_missing_values(const Vector<Vector<size_t>>& missing_indices) const
{
    #ifdef __OPENNN_DEBUG__

    if(columns_number != 2)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector<T> calculate_means_integers(const Vector<size_t>&) const method.\n"
              << "Number of columns must be two.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Vector<size_t> integers = this->get_column(0).get_integer_elements(get_rows_number());

    return calculate_means_integers_missing_values(integers, missing_indices);
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the given integers. The matrix must have 2 columns, the first
/// one containing the integers and the second one the corresponding values.
/// @param integers Integers.
/// @param missing_indices Missing indices.

template <class T>
Vector<T> Matrix<T>::calculate_means_integers_missing_values(const Vector<size_t>& integers, const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t integers_number = integers.size();

    #ifdef __OPENNN_DEBUG__

    if(columns_number != 2)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector<T> calculate_means_integers(const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
              << "Number of columns must be two.\n";

       throw logic_error(buffer.str());
    }


    if(integers_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector<T> calculate_means_integers(const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
              << "Number of integers must be greater than 0.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = this->get_rows_number();

    Vector<T> means(integers_number);

    T sum = 0.0;
    size_t count = 0;

    for(size_t i = 0; i < integers_number; i++)
    {
        sum = 0.0;
        count = 0;

        const size_t current_integer = integers[i];

        for(unsigned j = 0; j < rows_number; j++)
        {
            if((*this)(j,0) == current_integer && !missing_indices[0].contains(i))
            {
                if(!missing_indices[1].contains(j))
                {
                    sum += (*this)(j,1);
                    count++;
                }
            }
        }

        if(count != 0)
        {
            means[i] = static_cast<T>(sum)/static_cast<T>(count);
        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the binary columns. The last column must contain the values
/// used to calculate the means. The rest of the columns must be binary.

template <class T>
Vector<T> Matrix<T>::calculate_means_binary() const
{
    if(columns_number == 2)
    {
        return calculate_means_binary_column();
    }
    else
    {
        return calculate_means_binary_columns();
    }
}


/// Returns a vector containing the values of the means for the 0s and 1s of a
/// binary column.
/// The matrix must have 2 columns, the first one has to be binary.

template <class T>
Vector<T> Matrix<T>::calculate_means_binary_column() const
{
    Vector<T> means(2,0.0);

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,0) == 0.0)
        {
            means[0] += (*this)(i,1);
            count++;
        }
        else if((*this)(i,0) == 1.0)
        {
            means[1] += (*this)(i,1);
            count++;
        }
    }

    if(count != 0)
    {
        means[0] = static_cast<T>(means[0])/static_cast<T>(count);
        means[1] = static_cast<T>(means[1])/static_cast<T>(count);
    }
    else
    {
        means[0] = 0.0;
        means[1] = 0.0;
    }

    return means;
}


/// Returns a vector containing the values of the means for the 1s of each
/// of the binary columns.
/// All the columns except the last one must be binary.

template <class T>
Vector<T> Matrix<T>::calculate_means_binary_columns() const
{
    Vector<T> means(columns_number-1);

    T sum = 0.0;
    size_t count = 0;

    for(size_t i = 0; i < columns_number-1; i++)
    {
        sum = 0.0;
        count = 0;

        for(size_t j = 0; j < rows_number; j++)
        {
            if((*this)(j,i) == 1.0)
            {
                sum += (*this)(j,columns_number-1);
                count++;
            }
        }

        if(count != 0)
        {
            means[i] = static_cast<T>(sum)/static_cast<T>(count);
        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;
}


/// Returns the mean for each of the positives for each of the binary columns.
/// @param missing_indices Missing indices.

template <class T>
Vector<T> Matrix<T>::calculate_means_binary_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    if(columns_number == 2)
    {
        return calculate_means_binary_column_missing_values(missing_indices);
    }
    else
    {
        return calculate_means_binary_columns_missing_values(missing_indices);
    }
}


/// Returns a vector containing the values of the means for the 0s and 1s of a
/// binary column.
/// The matrix must have 2 columns, the first one has to be binary.
/// @param missing_indices Missing values.

template <class T>
Vector<T> Matrix<T>::calculate_means_binary_column_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<T> means(2,0.0);

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(!missing_indices[0].contains(i))
        {
            if((*this)(i,0) == 0.0)
            {
                means[0] += (*this)(i,1);
                count++;
            }
            else if((*this)(i,0) == 1.0)
            {
                means[1] += (*this)(i,1);
                count++;
            }
        }
    }

    if(count != 0)
    {
        means[0] = static_cast<T>(means[0])/static_cast<T>(count);
        means[1] = static_cast<T>(means[1])/static_cast<T>(count);
    }
    else
    {
        means[0] = 0.0;
        means[1] = 0.0;
    }

    return means;
}


/// Returns a vector containing the values of the means for the 1s of each
/// of the binary columns.
/// All the columns except the last one must be binary.
/// @param missing_indices Missing values.

template <class T>
Vector<T> Matrix<T>::calculate_means_binary_columns_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<T> means(columns_number-1);

    T sum = 0.0;
    size_t count = 0;

    for(size_t i = 0; i < columns_number-2; i++)
    {
        sum = 0.0;
        count = 0;

        for(size_t j = 0; j < rows_number; j++)
        {
            if((*this)(j,i) == 1.0)
            {
                if(!missing_indices[i].contains(j))
                {
                    sum += (*this)(j,columns_number-1);
                    count++;
                }
            }
        }

        if(count != 0)
        {
            means[i] = static_cast<T>(sum)/static_cast<T>(count);
        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;
}


/// Calculates the means dividing the variable into bins.
/// @param bins_number Number of bins.

template <class T>
Vector<T> Matrix<T>::calculate_means_continiuous(const size_t& bins_number) const
{
    Vector<T> means(bins_number,0.0);

    const T minimum = this->get_column(0).calculate_minimum();
    const T maximum = this->get_column(1).calculate_maximum();

    const double step = static_cast<double>(minimum+maximum)/static_cast<double>(bins_number);

    double previous_limit = minimum;
    double current_limit = minimum + step;

    size_t count;

    for(size_t i = 0; i < bins_number-1; i++)
    {
        count = 0;

        for(size_t j = 0; j < rows_number; j++)
        {
            if(i == bins_number-2 &&(*this)(j,0) >= previous_limit &&(*this)(j,1) <= current_limit)
            {
                means[0] += (*this)(j,0);
                count++;
            }
            else if((*this)(j,0) >= previous_limit &&(*this)(j,1) < current_limit)
            {
                means[0] += (*this)(j,0);
                count++;
            }
        }

        means[i] = static_cast<T>(means[i])/static_cast<T>(count);
    }

    return means;
}


/// Calculates the means dividing the variable into bins.
/// @param bins_number Number of bins.

template <class T>
Vector<T> Matrix<T>::calculate_means_continiuous_missing_values(const Vector< Vector<size_t> >& missing_values,
                                                                const size_t & bins_number) const
{
    Vector<T> means(bins_number,0.0);

    const T minimum = this->get_column(0).calculate_minimum();
    const T maximum = this->get_column(1).calculate_maximum();

    const double step = static_cast<double>(minimum+maximum)/static_cast<double>(bins_number);

    double previous_limit = minimum;
    double current_limit = minimum + step;

    size_t count;

    for(size_t i = 0; i < bins_number-1; i++)
    {
        count = 0;

        for(size_t j = 0; j < rows_number; j++)
        {
            if(!missing_values[0].contains(j))
            {
                if(i == bins_number-2 &&(*this)(j,0) >= previous_limit &&(*this)(j,1) <= current_limit)
                {
                    means[0] += (*this)(j,0);
                    count++;
                }
                else if((*this)(j,0) >= previous_limit &&(*this)(j,1) < current_limit)
                {
                    means[0] += (*this)(j,0);
                    count++;
                }
            }
        }

        means[i] = static_cast<T>(means[i])/static_cast<T>(count);
    }

    return means;
}


/// Returns the basic statistics of the columns.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of columns in this matrix.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_statistics() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Statistics<double> > calculate_statistics() const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector< Statistics<T> > statistics(columns_number);

   Vector<T> column(rows_number);

#pragma omp parallel for private(column)

   for(int i = 0; i < static_cast<int>(columns_number); i++)
   {
      column = get_column(i);

      statistics[i] = column.calculate_statistics();

      statistics[i].name = header[i];
   }

   return(statistics);
}


/// Returns the basic statistics of the columns when the matrix has missing values.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_statistics_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Statistics<double> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(missing_indices.size() != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Statistics<double> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Size of missing indices(" << missing_indices.size() << ") must be equal to to number of columns(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector< Statistics<T> > statistics(columns_number);

   Vector<T> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = get_column(i);

      statistics[i] = column.calculate_statistics_missing_values(missing_indices[i]);
   }

   return(statistics);
}


/// Returns the basic statistics of given columns for given rows.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of given columns.
/// @param row_indices Indices of the rows for which the statistics are to be computed.
/// @param column_indices Indices of the columns for which the statistics are to be computed.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_statistics(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
    const size_t row_indices_size = row_indices.size();
    const size_t column_indices_size = column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    size_t row_index, column_index;

    Vector<T> minimums(column_indices_size, numeric_limits<T>::max());
    Vector<T> maximums;

    if(numeric_limits<T>::is_signed)
    {
      maximums.set(column_indices_size, -numeric_limits<T>::max());
    }
    else
    {
        maximums.set(column_indices_size, 0);
    }

    Vector<double> sums(column_indices_size, 0.0);
    Vector<double> squared_sums(column_indices_size, 0.0);

    for(size_t i = 0; i < row_indices_size; i++)
    {
        row_index = row_indices[i];

#pragma omp parallel for private(column_index)

        for(int j = 0; j < static_cast<int>(column_indices_size); j++)
        {
            column_index = column_indices[static_cast<size_t>(j)];

            if((*this)(row_index,column_index) < minimums[j])
            {
                minimums[j] = (*this)(row_index,column_index);
            }

            if((*this)(row_index,column_index) > maximums[j])
            {
                maximums[j] = (*this)(row_index,column_index);
            }

            sums[j] += (*this)(row_index,column_index);
            squared_sums[j] += (*this)(row_index,column_index) *(*this)(row_index,column_index);
        }
    }

    const Vector<double> mean = sums/static_cast<double>(row_indices_size);

    Vector<double> standard_deviation(column_indices_size, 0.0);

    if(row_indices_size > 1)
    {
        for(size_t i = 0; i < column_indices_size; i++)
        {
            const double numerator = squared_sums[i] -(sums[i] * sums[i]) / row_indices_size;
            const double denominator = row_indices_size - 1.0;

            standard_deviation[i] = numerator / denominator;

            standard_deviation[i] = sqrt(standard_deviation[i]);
        }
    }

    for(size_t i = 0; i < column_indices_size; i++)
    {
        statistics[i].minimum = minimums[i];
        statistics[i].maximum = maximums[i];
        statistics[i].mean = mean[i];
        statistics[i].standard_deviation = standard_deviation[i];
    }

    return statistics;
}


template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_statistics(const Vector< Vector<size_t> >& row_indices, const Vector<size_t>& column_indices) const
{
    const size_t column_indices_size = column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    Vector<T> minimums(column_indices_size, numeric_limits<T>::max());
    Vector<T> maximums;

    if(numeric_limits<T>::is_signed)
    {
      maximums.set(column_indices_size, -numeric_limits<T>::max());
    }
    else
    {
        maximums.set(column_indices_size, 0);
    }

    Vector<double> sums(column_indices_size, 0.0);
    Vector<double> squared_sums(column_indices_size, 0.0);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(column_indices_size); i++)
    {
        const size_t column_index = column_indices[static_cast<size_t>(i)];

        const size_t current_rows_number = row_indices[static_cast<size_t>(i)].size();

        for(size_t j = 0; j < current_rows_number; j++)
        {
            const size_t row_index = row_indices[static_cast<size_t>(i)][j];

            if((*this)(row_index,column_index) < minimums[i])
            {
                minimums[i] = (*this)(row_index,column_index);
            }

            if((*this)(row_index,column_index) > maximums[i])
            {
                maximums[i] = (*this)(row_index,column_index);
            }

            sums[i] += (*this)(row_index,column_index);
            squared_sums[i] += (*this)(row_index,column_index) *(*this)(row_index,column_index);
        }
    }

    Vector<double> standard_deviation(column_indices_size, 0.0);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        const size_t current_rows_number = row_indices[i].size();

        if(current_rows_number > 1)
        {
            const double numerator = squared_sums[i] -(sums[i] * sums[i]) /static_cast<double>(current_rows_number);
            const double denominator = current_rows_number - 1.0;

            standard_deviation[i] = sqrt(numerator/denominator);
        }
    }

    for(size_t i = 0; i < column_indices_size; i++)
    {
        statistics[i].minimum = minimums[i];
        statistics[i].maximum = maximums[i];
        statistics[i].mean = sums[i]/static_cast<double>(row_indices[i].size());
        statistics[i].standard_deviation = standard_deviation[i];
    }

    return statistics;
}


/// Returns a vector of vectors of size 2. The first position contains the minimums of the given columns for the given rows while the second position
/// contains the maximums of the given columns for the given rows.
/// @param row_indices Given rows
/// @param column_indices Given columns

template <class T>
Vector< Vector<T> > Matrix<T>::calculate_columns_minimums_maximums(const Vector< Vector<size_t> >& row_indices, const Vector<size_t>& column_indices) const
{
    Vector< Vector<T> > minimums_maximums(2);

    const size_t column_indices_size = column_indices.size();

    Vector<T> minimums(column_indices_size, numeric_limits<T>::max());
    Vector<T> maximums;

    if(numeric_limits<T>::is_signed)
    {
       maximums.set(column_indices_size, -numeric_limits<T>::max());
    }
    else
    {
        maximums.set(column_indices_size, 0);
    }

//    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(column_indices_size); i++)
    {
        const size_t column_index = column_indices[static_cast<size_t>(i)];

        const size_t current_rows_number = row_indices[static_cast<size_t>(i)].size();

        for(size_t j = 0; j < current_rows_number; j++)
        {
            const size_t row_index = row_indices[static_cast<size_t>(i)][j];

            if((*this)(row_index,column_index) < minimums[i])
            {
                minimums[i] = (*this)(row_index,column_index);
            }

            if((*this)(row_index,column_index) > maximums[i])
            {
                maximums[i] = (*this)(row_index,column_index);
            }
        }
    }

    minimums_maximums[0] = minimums;
    minimums_maximums[1] = maximums;

    return minimums_maximums;
}


/// Returns the basic statistics of all the columns for given rows.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param row_indices Indices of the rows for which the statistics are to be computed.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_rows_statistics(const Vector<size_t>& row_indices) const
{
    const size_t row_indices_size = row_indices.size();

    Vector< Statistics<T> > statistics(columns_number);

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i, row_indices);

        statistics[i] = column.calculate_statistics();
    }

    return statistics;
}


/// Returns the basic statistics of all the columns for given rows when the matrix has missing values.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param row_indices Indices of the rows for which the statistics are to be computed.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_rows_statistics_missing_values(const Vector<size_t>& row_indices, const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t row_indices_size = row_indices.size();

    Vector< Statistics<T> > statistics(columns_number);

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i, row_indices);

        statistics[i] = column.calculate_statistics_missing_values(missing_indices[i]);
    }

    return statistics;
}


/// Returns the basic statistics of given columns.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of given columns.
/// @param column_indices Indices of the columns for which the statistics are to be computed.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_columns_statistics(const Vector<size_t>& column_indices) const
{
    const size_t column_indices_size = column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = column_indices[i];

        column = get_column(index);

        statistics[i] = column.calculate_statistics();
    }

    return statistics;
}


template <class T>
Vector<T> Matrix<T>::calculate_rows_means(const Vector<size_t>& row_indices) const
{
    Vector<size_t> used_row_indices;

    if(row_indices.empty())
    {
        used_row_indices.set(this->get_rows_number());
        used_row_indices.initialize_sequential();
    }
    else
    {
        used_row_indices = row_indices;
    }

    const size_t row_indices_size = used_row_indices.size();

    Vector<T> means(columns_number);

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i, used_row_indices);

        means[i] = column.calculate_mean();
    }

    return means;
}


template <class T>
Vector<T> Matrix<T>::calculate_columns_minimums(const Vector<size_t>& column_indices) const
{
    Vector<size_t> used_column_indices;

    if(column_indices.empty())
    {
        used_column_indices.set(columns_number);
        used_column_indices.initialize_sequential();
    }
    else
    {
        used_column_indices = column_indices;
    }

    const size_t column_indices_size = used_column_indices.size();

    Vector<T> minimums(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = used_column_indices[i];

        column = get_column(index);

        minimums[i] = column.calculate_minimum();
    }

    return minimums;
}


template <class T>
Vector<T> Matrix<T>::calculate_columns_maximums(const Vector<size_t>& column_indices) const
{
    Vector<size_t> used_column_indices;

    if(column_indices.empty())
    {
        used_column_indices.set(columns_number);
        used_column_indices.initialize_sequential();
    }
    else
    {
        used_column_indices = column_indices;
    }

    const size_t column_indices_size = used_column_indices.size();

    Vector<T> maximums(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = used_column_indices[i];

        column = get_column(index);

        maximums[i] = column.calculate_maximum();
    }

    return maximums;
}


/// Returns the basic statistics of given columns when the matrix has missing values.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of given columns.
/// @param column_indices Indices of the columns for which the statistics are to be computed.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_columns_statistics_missing_values(const Vector<size_t>& column_indices,
                                                                               const Vector< Vector<size_t> > missing_indices) const
{
    const size_t column_indices_size = column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

#pragma omp parallel for private(index, column) schedule(dynamic)

    for(int i = 0; i < static_cast<int>(column_indices_size); i++)
    {
        index = column_indices[static_cast<size_t>(i)];

        column = get_column(index);

        statistics[i] = column.calculate_statistics_missing_values(missing_indices[index]);
    }

    return statistics;
}


/// Returns the asymmetry and the kurtosis of the columns.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of columns in this matrix.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_shape_parameters() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector< Vector<double> > calculate_shape_parameters() const method.\n"
              << "Number of rows must be greater than one.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Vector< Vector<double> > shape_parameters(columns_number);

    Vector<T> column(rows_number);

    for(size_t i = 0; i < columns_number; i++)
    {
       column = get_column(i);

       shape_parameters[i] = column.calculate_shape_parameters();
    }

    return(shape_parameters);
}


/// Returns the asymmetry and the kurtosis of the columns when the matrix has missing values.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_shape_parameters_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Vector<double> > calculate_shape_parameters_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(missing_indices.size() != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Vector<double> > calculate_shape_parameters_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Size of missing indices(" << missing_indices.size() << ") must be equal to to number of columns(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector< Vector<double> > shape_parameters(columns_number);

   Vector<T> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = get_column(i);

      shape_parameters[i] = column.calculate_shape_parameters_missing_values(missing_indices[i]);
   }

   return(shape_parameters);
}


/// Returns the asymmetry and the kurtosis of given columns for given rows.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of given columns.
/// @param row_indices Indices of the rows for which the statistics are to be computed.
/// @param column_indices Indices of the columns for which the statistics are to be computed.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_shape_parameters(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
    const size_t row_indices_size = row_indices.size();
    const size_t column_indices_size = column_indices.size();

    Vector< Vector<double> > shape_parameters(column_indices_size);

    size_t index;

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = column_indices[i];

        column = get_column(index, row_indices);

        shape_parameters[i] = column.calculate_shape_parameters();
    }

    return shape_parameters;
}


/// Returns the asymmetry and the kurtosis of all the columns for given rows.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param row_indices Indices of the rows for which the statistics are to be computed.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_rows_shape_parameters(const Vector<size_t>& row_indices) const
{
    const size_t row_indices_size = row_indices.size();

    Vector< Vector<double> > shape_parameters(columns_number);

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i, row_indices);

        shape_parameters[i] = column.calculate_shape_parameters();
    }

    return shape_parameters;
}


/// Returns the asymmetry and the kurtosis of all the columns for given rows when the matrix has missing values.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param row_indices Indices of the rows for which the statistics are to be computed.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_rows_shape_parameters_missing_values(const Vector<size_t>& row_indices, const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t row_indices_size = row_indices.size();

    Vector< Vector<double> > shape_parameters(columns_number);

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i, row_indices);

        shape_parameters[i] = column.calculate_shape_parameters_missing_values(missing_indices[i]);
    }

    return shape_parameters;
}


/// Returns the asymmetry and the kurtosis of given columns.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of given columns.
/// @param column_indices Indices of the columns for which the statistics are to be computed.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_columns_shape_parameters(const Vector<size_t>& column_indices) const
{
    const size_t column_indices_size = column_indices.size();

    Vector< Vector<double> > shape_parameters(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = column_indices[i];

        column = get_column(index);

        shape_parameters[i] = column.calculate_shape_parameters();
    }

    return shape_parameters;
}


/// Returns the asymmetry and the kurtosis of given columns when the matrix has missing values.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of given columns.
/// @param column_indices Indices of the columns for which the statistics are to be computed.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_columns_shape_parameters_missing_values(const Vector<size_t>& column_indices, const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t column_indices_size = column_indices.size();

    Vector< Vector<double> > shape_parameters(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = column_indices[i];

        column = get_column(index);

        shape_parameters[i] = column.calculate_shape_parameters_missing_values(missing_indices[index]);
    }

    return shape_parameters;
}


/// Calculates the box plots for a set of rows of each of the given columns of this matrix.
/// @param rows_indices Rows to be used for the box plot.
/// @param columns_indices Indices of the columns for which box plots are going to be calculated.

template <class T>
Vector< Vector<double> >  Matrix<T>::calculate_box_plots(const Vector<Vector<size_t> >& rows_indices, const Vector<size_t>& columns_indices) const
{
    const size_t columns_number = columns_indices.size();

    #ifdef __OPENNN_DEBUG__

    if(columns_number == rows_indices.size())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "void calculate_box_plots() const method.\n"
              << "Size of row indices must be equal to the number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Vector< Vector<double> > box_plots(columns_number);

    Vector<double> column;

    #pragma omp parallel for private(column)

    for(int i = 0; i < static_cast<int>(columns_number); i++)
    {
        box_plots[static_cast<size_t>(i)].resize(5);

        const size_t rows_number = rows_indices[static_cast<size_t>(i)].size();

        column = this->get_column(columns_indices[i]).get_subvector(rows_indices[i]);

        sort(column.begin(), column.end(), less<double>());

        // Minimum

        box_plots[static_cast<size_t>(i)][0] = column[0];

        if(rows_number % 2 == 0)
        {
            // First quartile

            box_plots[static_cast<size_t>(i)][1] = (column[rows_number / 4] + column[rows_number / 4 + 1]) / 2.0;

            // Second quartile

            box_plots[static_cast<size_t>(i)][2] = (column[rows_number * 2 / 4] +
                           column[rows_number * 2 / 4 + 1]) /
                          2.0;

            // Third quartile

            box_plots[static_cast<size_t>(i)][3] = (column[rows_number * 3 / 4] +
                           column[rows_number * 3 / 4 + 1]) /
                          2.0;
        }
        else
        {
            // First quartile

            box_plots[static_cast<size_t>(i)][1] = column[rows_number / 4];

            // Second quartile

            box_plots[static_cast<size_t>(i)][2] = column[rows_number * 2 / 4];

            //Third quartile

            box_plots[static_cast<size_t>(i)][3] = column[rows_number * 3 / 4];
        }

        // Maximum

        box_plots[static_cast<size_t>(i)][4] = column[rows_number-1];
    }

    return box_plots;
}


/// Retruns the covariance matrix of this matrix.
/// The number of columns and rows of the matrix is equal to the number of columns of this matrix.

template <class T>
Matrix<double> Matrix<T>::calculate_covariance_matrix() const
{
    const size_t size = columns_number;

    #ifdef __OPENNN_DEBUG__

    if(size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "void calculate_covariance_matrix() const method.\n"
              << "Number of columns must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<double> covariance_matrix(size, size, 0.0);

    Vector<double> first_column;
    Vector<double> second_column;

    for(size_t i = 0; i < size; i++)
    {
        first_column = (*this).get_column(i);

        for(size_t j = i; j < size; j++)
        {
            second_column = (*this).get_column(j);

            covariance_matrix(i,j) = first_column.calculate_covariance(second_column);
            covariance_matrix(j,i) = covariance_matrix(i,j);
        }
    }

    return covariance_matrix;
}


/// Calculates a histogram for each column, each having a given number of bins.
/// It returns a vector of vectors of vectors.
/// The size of the main vector is the number of columns.
/// Each subvector contains the frequencies and centers of that colums.
/// @param bins_number Number of bins for each histogram.

template <class T>
Vector< Histogram<T> > Matrix<T>::calculate_histograms(const size_t& bins_number) const
{
   Vector< Histogram<T> > histograms(columns_number);

   Vector<T> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = get_column(i);

      if(column.is_binary())
      {
          histograms[i] = column.calculate_histogram_binary();
      }
      else
      {
          histograms[i] = column.calculate_histogram(bins_number);
      }
   }

   return(histograms);
}


/// Calculates a histogram for each column, each having a given number of bins, when the data has missing values.
/// It returns a vector of vectors of vectors.
/// The size of the main vector is the number of columns.
/// Each subvector contains the frequencies and centers of that colums.
/// @param bins_number Number of bins for each histogram.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Histogram<T> > Matrix<T>::calculate_histograms_missing_values(const Vector< Vector<size_t> >& missing_indices, const size_t& bins_number) const
{
   Vector< Histogram<T> > histograms(columns_number);

   Vector<T> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = get_column(i);

      histograms[i] = column.calculate_histogram_missing_values(missing_indices[i], bins_number);
   }

   return(histograms);
}


/// Returns the matrix indices at which the elements are less than some given value.
/// @param value Value.

template <class T>
Matrix<size_t> Matrix<T>::calculate_less_than_indices(const T& value) const
{
   Matrix<size_t> indices;

   Vector<size_t> row(2);

   for(size_t i = 0; i < rows_number; i++)
   {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) < value && indices.empty())
            {
                indices.set(1, 2);

                row[0] = i;
                row[1] = j;

                indices.set_row(0, row);
            }
            else if((*this)(i,j) < value)
            {
                row[0] = i;
                row[1] = j;

                indices.append_row(row);
            }
        }
   }

   return(indices);
}


/// Returns the matrix indices at which the elements are greater than some given value.
/// @param value Value.

template <class T>
Matrix<size_t> Matrix<T>::calculate_greater_than_indices(const T& value) const
{
   Matrix<size_t> indices;

   Vector<size_t> row(2);

   for(size_t i = 0; i < rows_number; i++)
   {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) > value && indices.empty())
            {
                indices.set(1, 2);

                row[0] = i;
                row[1] = j;

                indices.set_row(0, row);
            }
            else if((*this)(i,j) > value)
            {
                row[0] = i;
                row[1] = j;

                indices.append_row(row);
            }
        }
   }

   return(indices);
}


template <class T>
Matrix<T> Matrix<T>::calculate_competitive() const
{
    Matrix<T> competitive(rows_number, columns_number, 0);

    for(size_t i = 0; i < rows_number; i++)
    {
        const size_t maximal_index = get_row(i).calculate_maximal_index();

        competitive(i, maximal_index) = 1;
    }

    return(competitive);
}


template <class T>
Matrix<T> Matrix<T>::calculate_softmax_rows() const
{
    Matrix<T> softmax(rows_number,columns_number);

    for (size_t i = 0; i < rows_number; i++)
    {
        softmax.set_row(i,(*this).get_row(i).calculate_softmax());
    }

    return softmax;
}


template <class T>
Matrix<T> Matrix<T>::calculate_softmax_columns() const
{
    Matrix<T> softmax(rows_number,columns_number);

    for (size_t i = 0; i < columns_number; i++)
    {
        softmax.set_column(i,(*this).get_column(i).calculate_softmax());
    }

    return softmax;
}


template <class T>
Matrix<T> Matrix<T>::calculate_normalized_columns() const
{
    Matrix<T> softmax(rows_number,columns_number);

    for (size_t i = 0; i < columns_number; i++)
    {
        softmax.set_column(i,(*this).get_column(i).calculate_normalized());
    }

    return softmax;
}


template <class T>
Matrix<T> Matrix<T>::calculate_scaled_minimum_maximum_0_1_columns() const
{
    Matrix<T> softmax(rows_number,columns_number);

    for (size_t i = 0; i < columns_number; i++)
    {
        softmax.set_column(i,(*this).get_column(i).calculate_scaled_minimum_maximum_0_1());
    }

    if(header != "") softmax.set_header(header);

    return softmax;
}


template <class T>
Matrix<T> Matrix<T>::calculate_scaled_mean_standard_deviation_columns() const
{
    Matrix<T> softmax(rows_number,columns_number);

    for (size_t i = 0; i < columns_number; i++)
    {
        softmax.set_column(i,(*this).get_column(i).calculate_scaled_mean_standard_deviation());
    }

    if(header != "") softmax.set_header(header);

    return softmax;
}


template <class T>
Matrix<T> Matrix<T>::calculate_reverse_columns() const
{
    Matrix<T> reverse(rows_number,columns_number);

    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < columns_number; j++)
        {
            reverse(i,j) = (*this)(i, columns_number-j-1);
        }
    }

    Vector<string> reverse_header(columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        reverse_header[i] = header[columns_number-i-1];
    }

    reverse.set_header(reverse_header);

    return reverse;
}


/// Scales the matrix elements with the mean and standard deviation method.
/// It updates the data in the matrix.
/// @param statistics Vector of statistics structures conatining the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns in this matrix.

template <class T>
void Matrix<T>::scale_mean_standard_deviation(const Vector< Statistics<T> >& statistics)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = statistics.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void scale_mean_standard_deviation(const Vector< Statistics<T> >&) const method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].standard_deviation < numeric_limits<double>::min())
      {
         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,j) = ((*this)(i,j) - statistics[j].mean)/statistics[j].standard_deviation;
         }
      }
   }
}


/// Scales the matrix elements substracting the linear trend.
/// It updates the data in the matrix.
/// @param trends Linear regression parameters for the trend.
/// @param column_index Index of the time column.

template <class T>
void Matrix<T>::delete_trends(const Vector< LinearRegressionParameters<double> >& trends, const size_t& column_index)
{
    size_t index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(j != column_index)
            {
               (*this)(i,j) = (*this)(i,j) -(trends[index].intercept + trends[index].slope *(*this)(i, column_index));

                index++;
            }
        }
    }
}


/// Scales the matrix elements substracting the linear trend.
/// This method is applied when the data set has missing values.
/// It updates the data in the matrix.
/// @param trends Linear regression parameters for the trend.
/// @param column_index Index of the time column.
/// @param missing_indices Indices of the rows with missing values.

template <class T>
void Matrix<T>::delete_trends_missing_values(const Vector< LinearRegressionParameters<double> >& trends, const size_t& column_index,
                                              const Vector< Vector<size_t> >& missing_indices)
{
    for(size_t j = 0; j < columns_number; j++)
    {
        if(j != column_index)
        {
            const Vector<size_t> used_rows = (Vector<size_t>(0, 1, rows_number - 1)).get_difference(missing_indices[j]);

            const size_t used_rows_number = used_rows.size();

            size_t index = 0;

            for(size_t i = 0; i < used_rows_number; i++)
            {
               (*this)(used_rows[i], j) = (*this)(used_rows[i], j) -(trends[index].intercept + trends[index].slope *(*this)(used_rows[i], column_index));
            }

            index++;
        }
    }
}


/// Scales the matrix elements substracting the linear trend of the inputs.
/// This method is applied when the data set has missing values.
/// It updates the data in the matrix.
/// @param trends Linear regression parameters for the inputs trend.
/// @param column_index Index of the time column.
/// @param column_indices Indices of the inputs columns.
/// @param missing_indices Indices of the rows with missing values.

template <class T>
void Matrix<T>::delete_inputs_trends_missing_values(const Vector< LinearRegressionParameters<double> >& trends, const size_t& column_index,
                                                    const Vector<size_t>& column_indices, const Vector< Vector<size_t> >& missing_indices)
{
    size_t used_columns_number = column_indices.size();

    size_t current_column = 0;

    size_t trends_index = 0;

    for(size_t j = 0; j < used_columns_number; j++)
    {
        current_column = column_indices[j];

        const Vector<size_t> used_rows = (Vector<size_t>(0, 1, rows_number - 1)).get_difference(missing_indices[current_column]);

        const size_t used_rows_number = used_rows.size();

        for(size_t i = 0; i < used_rows_number; i++)
        {
           (*this)(used_rows[i], current_column) = (*this)(used_rows[i], current_column) -
                   (trends[trends_index].intercept + trends[trends_index].slope *(*this)(used_rows[i], column_index));
        }

        trends_index++;
    }
}


/// Scales the matrix elements substracting the linear trend of the targets.
/// This method is applied when the data set has missing values.
/// It updates the data in the matrix.
/// @param trends Linear regression parameters for the targets trend.
/// @param column_index Index of the time column.
/// @param column_indices Indices of the targets columns.
/// @param missing_indices Indices of the rows with missing values.

template <class T>
void Matrix<T>::delete_outputs_trends_missing_values(const Vector< LinearRegressionParameters<double> >& trends,
                                                     const size_t& column_index,
                                                     const Vector<size_t>& column_indices,
                                                     const Vector< Vector<size_t> >& missing_indices)
{
    const size_t used_columns_number = column_indices.size();

    size_t current_column = 0;

    size_t trends_index = 0;

    for(size_t j = 0; j < used_columns_number; j++)
    {
        current_column = column_indices[j];

        const Vector<size_t> used_rows = (Vector<size_t>(0, 1, rows_number - 1)).get_difference(missing_indices[current_column]);

        const size_t used_rows_number = used_rows.size();

        for(size_t i = 0; i < used_rows_number; i++)
        {
           (*this)(used_rows[i], current_column) = (*this)(used_rows[i], current_column) -
                   (trends[trends_index].intercept + trends[trends_index].slope *(*this)(used_rows[i], column_index));
        }

        trends_index++;
    }
}


/// Scales the data using the mean and standard deviation method and
/// the mean and standard deviation values calculated from the matrix.
/// It also returns the statistics of all the columns.

template <class T>
Vector< Statistics<T> > Matrix<T>::scale_mean_standard_deviation()
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_mean_standard_deviation(statistics);

    return(statistics);
}


/// Scales given rows from the matrix using the mean and standard deviation method.
/// @param statistics Vector of statistics for all the columns.
/// @param row_indices Indices of rows to be scaled.

template <class T>
void Matrix<T>::scale_rows_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t row_index;

    // Scale columns

    for(size_t j = 0; j < columns_number; j++)
    {
       if(statistics[j].standard_deviation < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < row_indices.size(); i++)
          {
             row_index = row_indices[i];

            (*this)(row_index,j) = ((*this)(row_index,j) - statistics[j].mean)/statistics[j].standard_deviation;
          }
       }
    }
}


/// Scales given columns of this matrix with the mean and standard deviation method.
/// @param statistics Vector of statistics structure containing the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param columns_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

template <class T>
void Matrix<T>::scale_columns_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& columns_indices)
{
   const size_t columns_indices_size = columns_indices.size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t statistics_size = statistics.size();

   if(statistics_size != columns_indices_size)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector template.\n"
             << "void scale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
             << "Size of statistics must be equal to size of columns indices.\n";

      throw logic_error(buffer.str());
   }

   #endif

   size_t column_index;

   // Scale columns

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      if(statistics[j].standard_deviation < numeric_limits<double>::min())
      {
         // Do nothing
      }
      else
      {
         column_index = columns_indices[j];

#pragma omp parallel for

         for(int i = 0; i < static_cast<int>(rows_number); i++)
         {
           (*this)(i,column_index) = ((*this)(i,column_index) - statistics[j].mean)/statistics[j].standard_deviation;
         }
      }
   }
}


/// Scales the matrix columns with the minimum and maximum method.
/// It updates the data in the matrix.
/// @param statistics Vector of statistics structures containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns in this matrix.

template <class T>
void Matrix<T>::scale_minimum_maximum(const Vector< Statistics<T> >& statistics)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = statistics.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void scale_minimum_maximum(const Vector< Statistics<T> >&) method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
      {
            // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,j) = 2.0*((*this)(i,j) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)-1.0;
         }
      }
   }
}


template <class T>
void Matrix<T>::scale_range(const Vector< Statistics<T> >& statistics, const T& minimum, const T& maximum)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = statistics.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void scale_minimum_maximum(const Vector< Statistics<T> >&) method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
      {
          for(size_t i = 0; i < rows_number; i++)
          {
               (*this)(i,j) = 0.0;
          }
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,j) = (maximum-minimum)*((*this)(i,j) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)+minimum;
         }
      }
   }
}

/// Scales the data using the minimum and maximum method and
/// the minimum and maximum values calculated from the matrix.
/// It also returns the statistics of all the columns.


template <class T>
Vector< Statistics<T> > Matrix<T>::scale_minimum_maximum()
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_minimum_maximum(statistics);

    return(statistics);
}


template <class T>
Vector< Statistics<T> > Matrix<T>::scale_range(const T& minimum, const T& maximum)
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_range(statistics, minimum, maximum);

    return(statistics);
}


/// Scales given rows from the matrix using the minimum and maximum method.
/// @param statistics Vector of statistics for all the columns.
/// @param row_indices Indices of rows to be scaled.

template <class T>
void Matrix<T>::scale_rows_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    // Control sentence(if debug)

    const size_t row_indices_size = row_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Rescale data

    size_t row_index;

    for(size_t j = 0; j < columns_number; j++)
    {
       if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < row_indices_size; i++)
          {
             row_index = row_indices[i];

            (*this)(row_index,j) = 2.0*((*this)(row_index,j) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum) - 1.0;
          }
       }
    }
}


/// Scales given columns of this matrix with the minimum and maximum method.
/// @param statistics Vector of statistics structure containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param column_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

template <class T>
void Matrix<T>::scale_columns_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    // Control sentence(if debug)

    const size_t column_indices_size = column_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != column_indices_size)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to size of columns indices.\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Rescale data

    for(size_t j = 0; j < column_indices_size; j++)
    {
       column_index = column_indices[j];

       if(statistics[j].maximum - statistics[j].minimum > 0.0)
       {
//#pragma omp parallel for

          for(int i = 0; i < static_cast<int>(rows_number); i++)
          {
            (*this)(i,column_index) = 2.0*((*this)(i,column_index) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum) - 1.0;
          }
       }
    }
}


/// Scales the matrix columns with the logarithimic method.
/// It updates the data in the matrix.
/// @param statistics Vector of statistics structures containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns in this matrix.

template <class T>
void Matrix<T>::scale_logarithmic(const Vector< Statistics<T> >& statistics)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = statistics.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void scale_logarithmic(const Vector< Statistics<T> >&) method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
      {
            // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,j) = log(1.0+ (2.0*((*this)(i,j) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)));
         }
      }
   }
}


/// Scales the data using the logarithmic method and
/// the minimum and maximum values calculated from the matrix.
/// It also returns the statistics of all the columns.

template <class T>
Vector< Statistics<T> > Matrix<T>::scale_logarithmic()
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_logarithmic(statistics);

    return(statistics);
}


/// Scales given rows from the matrix using the logarithmic method.
/// @param statistics Vector of statistics for all the columns.
/// @param row_indices Indices of rows to be scaled.

template <class T>
void Matrix<T>::scale_rows_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    // Control sentence(if debug)

    const size_t row_indices_size = row_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_rows_logarithmic(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Rescale data

    size_t row_index;

    for(size_t j = 0; j < columns_number; j++)
    {
       if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < row_indices_size; i++)
          {
             row_index = row_indices[i];

            (*this)(row_index,j) = log(1.0+ (2.0*((*this)(row_index,j) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)));
          }
       }
    }
}


/// Scales given columns of this matrix with the logarithmic method.
/// @param statistics Vector of statistics structure containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param column_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

template <class T>
void Matrix<T>::scale_columns_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    // Control sentence(if debug)

    const size_t column_indices_size = column_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != column_indices_size)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_columns_logarithmic(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to size of columns indices.\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Rescale data

    for(size_t j = 0; j < column_indices_size; j++)
    {
       column_index = column_indices[j];

       if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
#pragma omp parallel for

          for(int i = 0; i < static_cast<int>(rows_number); i++)
          {
            (*this)(i,column_index) = log(1.0+ (2.0*((*this)(i,column_index) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)));
          }
       }
    }
}


/// Unscales the matrix columns with the mean and standard deviation method.
/// It updates the matrix elements.
/// @param statistics Vector of statistics structures containing the mean and standard deviations for the unscaling.
/// The size of that vector must be equal to the number of columns in this matrix.

template <class T>
void Matrix<T>::unscale_mean_standard_deviation(const Vector< Statistics<T> >& statistics)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = statistics.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void unscale_mean_standard_deviation(const Vector< Statistics<T> >&) const method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].standard_deviation < numeric_limits<double>::min())
      {
         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,j) = (*this)(i,j)*statistics[j].standard_deviation + statistics[j].mean;
         }
      }
   }
}


/// Unscales given rows using the mean and standard deviation method.
/// @param statistics Vector of statistics structures for all the columns.
/// The size of this vector must be equal to the number of columns.
/// @param row_indices Indices of rows to be unscaled.

template <class T>
void Matrix<T>::unscale_rows_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    size_t row_index;

    // Unscale columns

    for(size_t j = 0;  j < columns_number; j++)
    {
       if(statistics[j].standard_deviation < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
             row_index = row_indices[i];

            (*this)(row_index,j) = (*this)(row_index,j)*statistics[j].standard_deviation + statistics[j].mean;
          }
       }
    }
}


/// Unscales given columns of this matrix with the mean and standard deviation method.
/// @param statistics Vector of statistics structure containing the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns in the matrix.
/// @param column_indices Vector of indices with the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be scaled.

template <class T>
void Matrix<T>::unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    #ifdef __OPENNN_DEBUG__

    if(statistics.size() != column_indices.size())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "void unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) const method.\n"
              << "Size of statistics vector must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

   size_t column_index;

   // Unscale columns

   for(size_t j = 0;  j < column_indices.size(); j++)
   {
      column_index = column_indices[j];

      if(statistics[j].standard_deviation > numeric_limits<double>::min())
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,column_index) = (*this)(i,column_index)*statistics[j].standard_deviation + statistics[j].mean;
         }
      }
   }
}


/// Unscales the matrix columns with the minimum and maximum method.
/// @param statistics Vector of statistics which contains the minimum and maximum scaling values.
/// The size of that vector must be equal to the number of columns in this matrix.

template <class T>
void Matrix<T>::unscale_minimum_maximum(const Vector< Statistics<T> >& statistics)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = statistics.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void unscale_minimum_maximum(const Vector< Statistics<T> >&) method.\n"
             << "Size of minimum vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
      {
         cout << "OpenNN Warning: Matrix template.\n"
                   << "void unscale_minimum_maximum(const Vector< Statistics<T> >&) const method.\n"
                   << "Minimum and maximum values of column " << j << " are equal.\n"
                   << "Those columns won't be unscaled.\n";

         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,j) = 0.5*((*this)(i,j) + 1.0)*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
         }
      }
   }
}


/// Unscales given rows using the minimum and maximum method.
/// @param statistics Vector of statistics structures for all the columns.
/// The size of this vector must be equal to the number of columns.
/// @param row_indices Indices of rows to be unscaled.

template <class T>
void Matrix<T>::unscale_rows_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    size_t row_index;

    // Unscale rows

    for(size_t j = 0; j < columns_number; j++)
    {
       if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
              row_index = row_indices[i];

            (*this)(row_index,j) = 0.5*((*this)(row_index,j) + 1.0)*(statistics[j].maximum-statistics[j].minimum)
             + statistics[j].minimum;
          }
       }
    }
}


/// Unscales given columns in the matrix with the minimum and maximum method.
/// @param statistics Vector of statistics structures containing the minimum and maximum values for the unscaling.
/// The size of that vector must be equal to the number of columns in the matrix.
/// @param column_indices Vector of indices of the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be unscaled.

template <class T>
void Matrix<T>::unscale_columns_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    #ifdef __OPENNN_DEBUG__

    if(statistics.size() != column_indices.size())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "void unscale_columns_minimum_maximum_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) const method.\n"
              << "Size of statistics vector must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Unscale columns

    for(size_t j = 0; j < column_indices.size(); j++)
    {
       const size_t column_index = column_indices[j];

       if(statistics[column_index].maximum - statistics[column_index].minimum > 0.0)
       {
          for(size_t i = 0; i < rows_number; i++)
          {
            (*this)(i,column_index) = 0.5*((*this)(i,column_index) + 1.0)*(statistics[j].maximum-statistics[j].minimum)
                                     + statistics[j].minimum;
          }
       }
    }
}


/// Unscales the matrix columns with the logarithmic method.
/// @param statistics Vector of statistics which contains the minimum and maximum scaling values.
/// The size of that vector must be equal to the number of columns in this matrix.

template <class T>
void Matrix<T>::unscale_logarithmic(const Vector< Statistics<T> >& statistics)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = statistics.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void unscale_logarithmic(const Vector< Statistics<T> >&) method.\n"
             << "Size of minimum vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
      {
         cout << "OpenNN Warning: Matrix template.\n"
                   << "void unscale_minimum_maximum(const Vector< Statistics<T> >&) const method.\n"
                   << "Minimum and maximum values of column " << j << " are equal.\n"
                   << "Those columns won't be unscaled.\n";

         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           (*this)(i,j) = 0.5*(exp((*this)(i,j)))*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
         }
      }
   }
}


/// Unscales given rows using the logarithimic method.
/// @param statistics Vector of statistics structures for all the columns.
/// The size of this vector must be equal to the number of columns.
/// @param row_indices Indices of rows to be unscaled.

template <class T>
void Matrix<T>::unscale_rows_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    size_t row_index;

    // Unscale rows

    for(size_t j = 0; j < columns_number; j++)
    {
       if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
              row_index = row_indices[i];

            (*this)(row_index,j) = 0.5*(exp((*this)(row_index,j)))*(statistics[j].maximum-statistics[j].minimum)
             + statistics[j].minimum;
          }
       }
    }
}


/// Unscales given columns in the matrix with the logarithmic method.
/// @param statistics Vector of statistics structures containing the minimum and maximum values for the unscaling.
/// The size of that vector must be equal to the number of columns in the matrix.
/// @param column_indices Vector of indices of the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be unscaled.

template <class T>
void Matrix<T>::unscale_columns_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    #ifdef __OPENNN_DEBUG__

    if(statistics.size() != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "void unscale_columns_logarithmic(const Vector< Statistics<T> >&, const Vector<size_t>&) const method.\n"
              << "Size of statistics vector must be equal to number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Unscale columns

    for(size_t j = 0; j < column_indices.size(); j++)
    {
        column_index = column_indices[j];

       if(statistics[column_index].maximum - statistics[column_index].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
            (*this)(i,column_index) = 0.5*(exp((*this)(i,column_index)))*(statistics[column_index].maximum-statistics[column_index].minimum)
             + statistics[column_index].minimum;
          }
       }
    }
}


/// Returns the row and column indices corresponding to the entry with minimum value.

template <class T>
Vector<size_t> Matrix<T>::calculate_minimal_indices() const
{
   T minimum = (*this)(0,0);
   Vector<size_t> minimal_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if((*this)(i,j) < minimum)
         {
            minimum = (*this)(i,j);
            minimal_indices[0] = i;
            minimal_indices[1] = j;
         }
      }
   }

   return(minimal_indices);
}


template <class T>
Vector<size_t> Matrix<T>::calculate_minimal_indices_omit(const T& value_to_omit) const
{
   T minimum = std::numeric_limits<T>::max();

   Vector<size_t> minimal_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if((*this)(i,j) != value_to_omit && (*this)(i,j) < minimum)
         {
            minimum = (*this)(i,j);
            minimal_indices[0] = i;
            minimal_indices[1] = j;
         }
      }
   }

   return(minimal_indices);
}


/// Returns the row and column indices corresponding to the entry with maximum value.

template <class T>
Vector<size_t> Matrix<T>::calculate_maximal_indices() const
{
   T maximum = (*this)(0,0);

   Vector<size_t> maximal_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if((*this)(i,j) > maximum)
         {
            maximum = (*this)(i,j);
            maximal_indices[0] = i;
            maximal_indices[1] = j;
         }
      }
   }

   return(maximal_indices);
}



template <class T>
Vector<size_t> Matrix<T>::calculate_maximal_indices_omit(const T& value_to_omit) const
{
   T maximum = std::numeric_limits<T>::min();

   Vector<size_t> maximum_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if((*this)(i,j) != value_to_omit && (*this)(i,j) > maximum)
         {
            maximum = (*this)(i,j);
            maximum_indices[0] = i;
            maximum_indices[1] = j;
         }
      }
   }

   return(maximum_indices);
}

/// Returns the row and column indices corresponding to the entries with minimum and maximum values.
/// The format is a vector of two vectors.
/// Each subvector also has two elements.
/// The first vector contains the minimal indices, and the second vector contains the maximal indices.

template <class T>
Vector< Vector<size_t> > Matrix<T>::calculate_minimal_maximal_indices() const
{
   T minimum = (*this)(0,0);
   T maximum = (*this)(0,0);

   Vector<size_t> minimal_indices(2, 0);
   Vector<size_t> maximal_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if((*this)(i,j) < minimum)
         {
            minimum = (*this)(i,j);
            minimal_indices[0] = i;
            minimal_indices[1] = j;
         }

         if((*this)(i,j) > maximum)
         {
            maximum = (*this)(i,j);
            maximal_indices[0] = i;
            maximal_indices[1] = j;
         }
      }
   }

   Vector< Vector<size_t> > minimal_maximal_indices(2);
   minimal_maximal_indices[0] = minimal_indices;
   minimal_maximal_indices[1] = maximal_indices;

   return(minimal_maximal_indices);
}


/// Returns the sum squared error between the elements of this matrix and the elements of another matrix.
/// @param other_matrix Other matrix.

template <class T>
double Matrix<T>::calculate_sum_squared_error(const Matrix<double>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
             << "Other number of rows must be equal to this number of rows.\n";

      throw logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
             << "Other number of columns must be equal to this number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   double error = 0.0;

   double sum_squared_error = 0.0;

   for(int i = 0; i < this->size(); i++)
   {      
        error = (*this)[i] - other_matrix[i];

        sum_squared_error += error*error;
   }

   return(sum_squared_error);
}


template <class T>
Vector<double> Matrix<T>::calculate_error_rows(const Matrix<double>& other_matrix) const
{
    Vector<double> error_rows(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            error_rows[i] += ((*this)(i,j) - other_matrix(i,j))*((*this)(i,j) - other_matrix(i,j));
        }

        error_rows[i] = sqrt(error_rows[i]);
    }

    return error_rows;
}


template <class T>
Vector<double> Matrix<T>::calculate_weighted_error_rows(const Matrix<double>& other_matrix, const double& weight1, const double& weight2) const
{
    Vector<double> weighted_error_rows(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j =0; j < columns_number; j++)
        {
            if(other_matrix(i,j) == 1.0)
            {
                weighted_error_rows[i] += weight1*((*this)(i,j) - other_matrix(i,j))*((*this)(i,j) - other_matrix(i,j));
            }
            else
            {
                weighted_error_rows[i] += weight2*((*this)(i,j) - other_matrix(i,j))*((*this)(i,j) - other_matrix(i,j));
            }
        }

        weighted_error_rows[i] = sqrt(weighted_error_rows[i]);
    }

    return weighted_error_rows;
}


/// Returns the sum squared error between rows of this matrix and the elements of another matrix.
/// @param other_matrix Other matrix.

template <class T>
Vector<double> Matrix<T>::calculate_sum_squared_error_rows(const Matrix<double>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
             << "Other number of rows must be equal to this number of rows.\n";

      throw logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
             << "Other number of columns must be equal to this number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif


   Vector<double> sum_squared_error_rows(rows_number, 0.0);

//#pragma omp parallel for reduction(+:sum_squared_error)

   double sum_squared_error = 0.0;

   for(int i = 0; i < rows_number; i++)
   {
       sum_squared_error = 0.0;

       for(int j = 0; j < columns_number; j++)
       {
           sum_squared_error += ((*this)(i,j) - other_matrix(i,j))*((*this)(i,j) - other_matrix(i,j));
       }

       sum_squared_error_rows[static_cast<unsigned>(i)] = sum_squared_error;

   }

   return(sum_squared_error_rows);
}


template <class T>
double Matrix<T>::calculate_cross_entropy_error(const Matrix<double>& other_matrix) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "double calculate_cross_entropy_error(const Matrix<double>&) const method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "double calculate_cross_entropy_error(const Matrix<double>&) const method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    double cross_entropy_error = 0.0;

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < columns_number; j++)
        {
            if(fabs(other_matrix(static_cast<unsigned>(i), static_cast<unsigned>(j)) - 0.0) < std::numeric_limits<double>::min())
            {
                cross_entropy_error -= log(1.0 - (*this)(i,j));
            }
            else if(fabs(other_matrix(static_cast<unsigned>(i), static_cast<unsigned>(j)) - 0.0) < std::numeric_limits<double>::min())
            {
                cross_entropy_error -= log((*this)(i,j));
            }
        }
    }

    return cross_entropy_error;
}


/// Returns the minkowski error between the elements of this matrix and the elements of another matrix.
/// @param other_matrix Other matrix.x.
/// @param minkowski_parameter Minkowski exponent value.

template <class T>
double Matrix<T>::calculate_minkowski_error(const Matrix<double>& other_matrix, const double& minkowski_parameter) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "double calculate_minkowski_error(const Matrix<double>&, const double&) const method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "double calculate_minkowski_error(const Matrix<double>&, const double&) const method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    double minkowski_error = 0.0;

    for(size_t i = 0; i < this->size(); i++)
    {
         minkowski_error += pow(fabs((*this)[i] - other_matrix[i]), minkowski_parameter);
    }

    return minkowski_error;
}



template <class T>
double Matrix<T>::calculate_weighted_sum_squared_error(const Matrix<double>& other_matrix, const double& positives_weight, const double& negatives_weight) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "double calculate_minkowski_error(const Matrix<double>&, const double&) const method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "double calculate_minkowski_error(const Matrix<double>&, const double&) const method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    double weighted_sum_squared_error = 0.0;

    double error = 0.0;

    for(size_t i = 0; i < this->size(); i++)
    {
        error = (*this)[i] - other_matrix[i];

        if(other_matrix[i] == 1.0)
        {
            weighted_sum_squared_error += positives_weight*error*error;
        }
        else if(other_matrix[i] == 0.0)
        {
            weighted_sum_squared_error += negatives_weight*error*error;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: Matrix template.\n"
                   << "double calculate_error() const method.\n"
                   << "Other matrix is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }
    }

    return weighted_sum_squared_error;
}



/// This method retuns the sum squared error between the elements of this matrix and the elements of a vector, by columns.
/// The size of the vector must be equal to the number of columns of this matrix.
/// @param vector Vector to be compared to this matrix.

template <class T>
double Matrix<T>::calculate_sum_squared_error(const Vector<double>& vector) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Vector<double>&) const method.\n"
             << "Size must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   double sum_squared_error = 0.0;

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         sum_squared_error += ((*this)(i,j) - vector[j])*((*this)(i,j) - vector[j]);
      }
   }

   return(sum_squared_error);
}


/// Returns a vector with the norm of each row.
/// The size of that vector is the number of rows.

template <class T>
Vector<double> Matrix<T>::calculate_rows_L2_norm() const
{
   Vector<double> rows_norm(rows_number, 0.0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         rows_norm[i] += (*this)(i,j)*(*this)(i,j);
      }

      rows_norm[i] = sqrt(rows_norm[i]);
   }

   return(rows_norm);
}


/// Returns a matrix with the absolute values of this matrix.

template <class T>
Matrix<T> Matrix<T>::calculate_absolute_value() const
{
    Matrix<T> absolute_value(rows_number, columns_number);

    for(size_t i = 0; i < this->size(); i++)
    {
          if((*this)[i] > 0)
          {
             absolute_value[i] = (*this)[i];
          }
          else
          {
             absolute_value[i] = -(*this)[i];
          }
    }

    return(absolute_value);
}


/// Returns the transpose of the matrix.

template <class T>
Matrix<T> Matrix<T>::calculate_transpose() const
{
   Matrix<T> transpose(columns_number, rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      for(size_t j = 0; j < rows_number; j++)
      {
         transpose(i,j) = (*this)(j,i);
      }
   }

   return(transpose);
}


/// Returns the determinant of a square matrix.

template <class T>
T Matrix<T>::calculate_determinant() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "calculate_determinant() const method.\n"
              << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_determinant() const method.\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   T determinant = 0;

   if(rows_number == 1)
   {
      determinant = (*this)(0,0);
   }
   else if(rows_number == 2)
   {
      determinant = (*this)(0,0)*(*this)(1,1) -(*this)(1,0)*(*this)(0,1);
   }
   else
   {
      int sign;

      for(size_t row_index = 0; row_index < rows_number; row_index++)
      {
         // Calculate sub data

         Matrix<T> sub_matrix(rows_number-1, columns_number-1);

         for(size_t i = 1; i < rows_number; i++)
         {
            size_t j2 = 0;

            for(size_t j = 0; j < columns_number; j++)
            {
               if(j == row_index)
               {
                  continue;
               }

               sub_matrix(i-1,j2) = (*this)(i,j);

               j2++;
            }
         }

         //sign = static_cast<size_t>(pow(-1.0, row_index+2.0));

         sign = static_cast<int>((((row_index + 2) % 2) == 0) ? 1 : -1 );

         determinant += sign*(*this)(0,row_index)*sub_matrix.calculate_determinant();
      }
   }

   return(determinant);
}


/// Returns the cofactor matrix.

template <class T>
Matrix<T> Matrix<T>::calculate_cofactor() const
{
   Matrix<T> cofactor(rows_number, columns_number);

   Matrix<T> c(rows_number-1, columns_number-1);

   for(size_t j = 0; j < rows_number; j++)
   {
      for(size_t i = 0; i < rows_number; i++)
      {
         // Form the adjoint a(i,j)

         size_t i1 = 0;

         for(size_t ii = 0; ii < rows_number; ii++)
         {
            if(ii == i) continue;

            size_t j1 = 0;

            for(size_t jj = 0; jj < rows_number; jj++)
            {
               if(jj == j) continue;

               c(i1,j1) = (*this)(ii,jj);
               j1++;
            }
            i1++;
         }

         const double determinant = c.calculate_determinant();

         cofactor(i,j) = static_cast<T>((((i + j) % 2) == 0) ? 1 : -1)*determinant;
         //cofactor(i,j) = pow(-1.0, i+j+2.0)*determinant;
      }
   }

   return(cofactor);
}


/// Returns the inverse of a square matrix.
/// An error message is printed if the matrix is singular.

template <class T>
Matrix<T> Matrix<T>::calculate_inverse() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "calculate_inverse() const method.\n"
              << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_inverse() const method.\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const double determinant = calculate_determinant();

   if(determinant == 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_inverse() const method.\n"
             << "Matrix is singular.\n";

      throw logic_error(buffer.str());
   }

   if(rows_number == 1)
   {
        Matrix<T> inverse(1, 1, 1.0/determinant);

        return(inverse);
   }

   // Calculate cofactor matrix

   const Matrix<T> cofactor = calculate_cofactor();

   // Adjoint matrix is the transpose of cofactor matrix

   const Matrix<T> adjoint = cofactor.calculate_transpose();

   // Inverse matrix is adjoint matrix divided by matrix determinant

   const Matrix<T> inverse = adjoint/determinant;

   return(inverse);
}


/// Returns the inverse of a square matrix using the LU decomposition method.
/// The given matrix must be invertible.

template <class T>
Matrix<T> Matrix<T>::calculate_LU_inverse() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "calculate_LU_inverse() const method.\n"
              << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_LU_inverse() const method.\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> inverse(rows_number, columns_number);

   const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)(this->data()), rows_number, columns_number);
   Eigen::Map<Eigen::MatrixXd> inverse_eigen(inverse.data(), rows_number, columns_number);

   inverse_eigen = this_eigen.inverse();

   return(inverse);
}


/// Solve a system of the form Ax = b, using the Cholesky decomposition.
/// A is this matrix and must be positive or negative semidefinite.
/// @param b Independent term of the system.

template <class T>
Vector<T> Matrix<T>::solve_LDLT(const Vector<double>& b) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "solve_LLT(const Vector<double>&) const method.\n"
              << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

    if(rows_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "solve_LLT(const Vector<double>&) const method.\n"
              << "Matrix must be squared.\n";

       throw logic_error(buffer.str());
    }

   #endif

   Vector<T> solution(rows_number);

   const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
   const Eigen::Map<Eigen::VectorXd> b_eigen((double*)b.data(),rows_number);
   Eigen::Map<Eigen::VectorXd> solution_eigen(solution.data(), rows_number);

   solution_eigen = this_eigen.ldlt().solve(b_eigen);

   return(solution);
}


/// Calculates the distance between two rows in the matix

template <class T>
double Matrix<T>::calculate_euclidean_distance(const size_t& first_index, const size_t& second_index) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_euclidean_distance(const size_t&, const size_t&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Vector<T> first_row = get_row(first_index);
    const Vector<T> second_row = get_row(second_index);

    return(first_row.calculate_euclidean_distance(second_row));
}


template <class T>
Vector<double> Matrix<T>::calculate_euclidean_distance(const Vector<T>& instance) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_euclidean_distance(const Vector<T>&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = (*this).get_row(i).calculate_euclidean_distance(instance);
    }

    return(distances);
}


template<class T>
Vector<double> Matrix<T>::calculate_euclidean_distance(const Matrix<T>& other_matrix) const
{
    Vector<double> distances(rows_number, 0.0);
    double error;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            error = (*this)(i,j) - other_matrix(i,j);

            distances[i] += error * error;
        }

        distances[i] = sqrt(distances[i]);
    }

    return((distances));
}


template <class T>
Vector<double> Matrix<T>::calculate_euclidean_weighted_distance(const Vector<T>& instance, const Vector<double>& weights) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_euclidean_weighted_distance(const Vector<T>&, const Vector<double>&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = (*this).get_row(i).calculate_euclidean_weighted_distance(instance,weights);
    }

    return(distances);
}


template <class T>
Matrix<double> Matrix<T>::calculate_euclidean_weighted_distance_matrix(const Vector<T>& instance, const Vector<double>& weights) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_euclidean_weighted_distance(const Vector<T>&, const Vector<double>&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Matrix<double> distances(rows_number,columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances.set_row(i,(*this).get_row(i).calculate_euclidean_weighted_distance_vector(instance,weights));
    }

    return(distances);
}


/// Calculates the distance between two rows in the matix

template <class T>
double Matrix<T>::calculate_manhattan_distance(const size_t& first_index, const size_t& second_index) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_manhattan_distance(const size_t&, const size_t&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Vector<T> first_row = get_row(first_index);
    const Vector<T> second_row = get_row(second_index);

    return(first_row.calculate_manhattan_distance(second_row));
}


template <class T>
Vector<double> Matrix<T>::calculate_manhattan_distance(const Vector<T>& instance) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_manhattan_distance(const size_t&, const size_t&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = (*this).get_row(i).calculate_manhattan_distance(instance);
    }

    return(distances);
}


template <class T>
Vector<double> Matrix<T>::calculate_manhattan_weighted_distance(const Vector<T>& instance, const Vector<double>& weights) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_manhattan_weighted_distance(const Vector<T>&, const Vector<double>&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = (*this).get_row(i).calculate_manhattan_weighted_distance(instance,weights);
    }

    return(distances);
}


template <class T>
Matrix<double> Matrix<T>::calculate_manhattan_weighted_distance_matrix(const Vector<T>& instance, const Vector<double>& weights) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_manhattan_weighted_distance(const Vector<T>&, const Vector<double>&) const method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Matrix<double> distances(rows_number,columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances.set_row(i,(*this).get_row(i).calculate_manhattan_weighted_distance_vector(instance,weights));
    }

    return(distances);
}


template <class T>
void Matrix<T>::divide_by_rows(const Vector<T>& vector)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

     if(rows_number != vector.size())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "divide_by_rows(const Vector<T>&) method.\n"
               << "Size of vector (" << vector.size() << ") must be equal to number of rows (" << rows_number <<").\n";

        throw logic_error(buffer.str());
     }

     #endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            (*this)(i,j) /= vector[i];
        }
    }
}


template <class T>
void Matrix<T>::filter(const T& minimum, const T& maximum)
{
    for(size_t i = 0; i < this->size(); i++)
    {
        if((*this)[i] < minimum)(*this)[i] = minimum;

        else if((*this)[i] > maximum)(*this)[i] = maximum;
    }
}


/// Sum matrix+scalar arithmetic operator.
/// @param scalar Scalar value to be added to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator + (const T& scalar) const
{
   Matrix<T> sum(rows_number, columns_number);

   transform(this->begin(), this->end(), sum.begin(), bind2nd(plus<T>(), scalar));

   return(sum);
}


/// Sum matrix+matrix arithmetic operator.
/// @param other_matrix Matrix to be added to this vector.

template <class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator + (const Matrix<T>&) const.\n"
             << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> sum(rows_number, columns_number);

   transform(this->begin(), this->end(), other_matrix.begin(), sum.begin(), plus<T>());

   return(sum);
}


/// Difference matrix-scalar arithmetic operator.
/// @param scalar Scalar value to be subtracted to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator -(const T& scalar) const
{
   Matrix<T> difference(rows_number, columns_number);

   transform( this->begin(), this->end(), difference.begin(), bind2nd(minus<T>(), scalar));

   return(difference);
}


/// Difference matrix-matrix arithmetic operator.
/// @param other_matrix Matrix to be subtracted to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator -(const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator -(const Matrix<T>&) const method.\n"
             << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix ("<< rows_number << "," << columns_number <<").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> difference(rows_number, columns_number);

   transform( this->begin(), this->end(), other_matrix.begin(), difference.begin(), minus<T>());

   return(difference);
}


/// Product matrix*scalar arithmetic operator.
/// @param scalar Scalar value to be multiplied to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator *(const T& scalar) const
{
    Matrix<T> product(rows_number, columns_number);

    for(size_t i = 0; i < this->size(); i++)
    {
        product[i] = (*this)[i]*scalar;
    }

    return(product);
}


/// Product matrix*matrix arithmetic operator.
/// @param other_matrix Matrix to be multiplied to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator *(const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator *(const Matrix<T>&) const method.\n"
             << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> product(rows_number, columns_number);

   for(size_t i = 0; i < this->size(); i++)
   {
         product[i] = (*this)[i]*other_matrix[i];
   }

   return(product);
}


/// Cocient Matrix/scalar arithmetic operator.
/// @param scalar Value of scalar.

template <class T>
Matrix<T> Matrix<T>::operator /(const T& scalar) const
{
    Matrix<T> results(rows_number, columns_number);

    for(size_t i = 0; i < results.size(); i++)
    {
        results[i] = (*this)[i]/scalar;
    }

    return(results);
}


/// Cocient matrix/vector arithmetic operator.
/// @param vector Vector to be divided to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator /(const Vector<T>& vector) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator /(const Vector<T>&) const.\n"
             << "Size of vector must be equal to number of rows.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> cocient(rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         cocient(i,j) = (*this)(i,j)/vector[i];
      }
   }

   return(cocient);
}


/// Cocient matrix/matrix arithmetic operator.
/// @param other_matrix Matrix to be divided to this vector.

template <class T>
Matrix<T> Matrix<T>::operator /(const Matrix<T>& other_matrix) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator /(const Matrix<T>&) const method.\n"
             << "Both matrix sizes must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> cocient(rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
         cocient[i] = (*this)[i]/other_matrix[i];
   }

   return(cocient);
}


/// Scalar sum and assignment operator.
/// @param value Scalar value to be added to this matrix.

template <class T>
void Matrix<T>::operator += (const T& value)
{
   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
        (*this)(i,j) += value;
      }
   }
}


/// Matrix sum and assignment operator.
/// @param other_matrix Matrix to be added to this matrix.

template <class T>
void Matrix<T>::operator += (const Matrix<T>& other_matrix)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator += (const Matrix<T>&).\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator += (const Matrix<T>&).\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] += other_matrix[i];
   }
}


/// Scalar rest and assignment operator.
/// @param value Scalar value to be subtracted to this matrix.

template <class T>
void Matrix<T>::operator -= (const T& value)
{
   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
        (*this)(i,j) -= value;
      }
   }
}


/// Matrix rest and assignment operator.
/// @param other_matrix Matrix to be subtracted to this matrix.

template <class T>
void Matrix<T>::operator -= (const Matrix<T>& other_matrix)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator -= (const Matrix<T>&).\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator -= (const Matrix<T>&).\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] -= other_matrix[i];
   }
}


/// Scalar product and assignment operator.
/// @param value Scalar value to be multiplied to this matrix.

template <class T>
void Matrix<T>::operator *= (const T& value)
{
   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] *= value;
   }
}


/// Matrix product and assignment operator.
/// @param other_matrix Matrix to be multiplied to this matrix.

template <class T>
void Matrix<T>::operator *= (const Matrix<T>& other_matrix)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator *= (const Matrix<T>&).\n"
             << "The number of rows in the other matrix (" << other_rows_number << ")"
             << " is not equal to the number of rows in this matrix (" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] *= other_matrix[i];
   }
}


/// Scalar division and assignment operator.
/// @param value Scalar value to be divided to this matrix.

template <class T>
void Matrix<T>::operator /= (const T& value)
{
   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] /= value;
   }
}


/// Matrix division and assignment operator.
/// @param other_matrix Matrix to be divided to this matrix.

template <class T>
void Matrix<T>::operator /= (const Matrix<T>& other_matrix)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator /= (const Matrix<T>&).\n"
             << "Both numbers of rows must be the same.\n";

      throw logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator /= (const Matrix<T>&).\n"
             << "Both numbers of columns must be the same.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
        (*this)[i] /= other_matrix[i];
   }
}


/// Returns the dot product of this matrix with a vector.
/// The size of the vector must be equal to the number of columns of the matrix.
/// @param vector Vector to be multiplied to this matrix.

template <class T>
Vector<double> Matrix<T>::dot(const Vector<double>& vector) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> dot(const Vector<T>&) const method.\n"
             << "Vector size must be equal to matrix number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Calculate matrix-vector poduct

   Vector<double> product(rows_number);

//   for(int i = 0; i < static_cast<int>(rows_number); i++)
//   {
//       product[i] = 0;

//      for(size_t j = 0; j < columns_number; j++)
//      {
//         product[i] += vector[j]*(*this)(i,j);
//      }
//   }

   const Eigen::Map<Eigen::MatrixXd> matrix_eigen((double*)this->data(), rows_number, columns_number);
   const Eigen::Map<Eigen::VectorXd> vector_eigen((double*)vector.data(), columns_number);
   Eigen::Map<Eigen::VectorXd> product_eigen(product.data(), rows_number);

   product_eigen = matrix_eigen*vector_eigen;

   return(product);
}


/// Returns the dot product of this matrix with another matrix.
/// @param other_matrix Matrix to be multiplied to this matrix.

template <class T>
Matrix<double> Matrix<T>::dot(const Matrix<double>& other_matrix) const
{
   const size_t other_columns_number = other_matrix.get_columns_number();
   const size_t other_rows_number = other_matrix.get_rows_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__


   if(other_rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> dot(const Matrix<T>&) const method.\n"
             << "The number of rows of the other matrix (" << other_rows_number << ") must be equal to the number of columns of this matrix (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> product(rows_number, other_columns_number);

//   for(size_t i = 0; i < rows_number; i++)
//   {
//     for(size_t j = 0; j < other_columns_number; j++)
//     {
//       for(size_t k = 0; k < columns_number; k++)
//       {
//            product(i,j) += (*this)(i,k)*other_matrix(k,j);
//       }
//     }
//   }

   const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
   const Eigen::Map<Eigen::MatrixXd> other_eigen((double*)other_matrix.data(), other_rows_number, other_columns_number);
   Eigen::Map<Eigen::MatrixXd> product_eigen(product.data(), rows_number, other_columns_number);

   product_eigen = this_eigen*other_eigen;

   return(product);
}


template <class T>
void Matrix<T>::dot(const Matrix<double>& a, const Matrix<double>& b)
{    
    const size_t a_rows_number = a.get_rows_number();
    const size_t a_columns_number = a.get_columns_number();

    const size_t b_columns_number = b.get_columns_number();

    if(rows_number != a_rows_number || columns_number != b_columns_number)
    {
        set(a_rows_number, b_columns_number);
    }

    initialize(0.0);

   for(size_t i = 0; i < a_rows_number; i++)
   {
     for(size_t j = 0; j < b_columns_number; j++)
     {
       for(size_t k = 0; k < a_columns_number; k++)
       {
            (*this)(i,j) += a(i,k)*b(k,j);
       }
     }
   }
}


template <class T>
void Matrix<T>::sum_dot(const Matrix<double>& a, const Matrix<double>& b)
{
    const size_t a_rows_number = a.get_rows_number();
    const size_t a_columns_number = a.get_columns_number();

    const size_t b_columns_number = b.get_columns_number();

//    if(rows_number != a_rows_number || columns_number != b_columns_number)
//    {
//        set(a_rows_number, b_columns_number);
//    }

   for(size_t i = 0; i < a_rows_number; i++)
   {
     for(size_t j = 0; j < b_columns_number; j++)
     {
       for(size_t k = 0; k < a_columns_number; k++)
       {
            (*this)(i,j) += a(i,k)*b(k,j);
       }
     }
   }
}


template <class T>
Matrix<double> Matrix<T>::calculate_linear_combinations(const Matrix<double>& other_matrix, const Vector<double>& other_vector) const
{
   const size_t other_columns_number = other_matrix.get_columns_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> calculate_linear_combinations(const Matrix<T>&) const method.\n"
             << "The number of rows of the other matrix (" << other_rows_number << ") must be equal to the number of columns of this matrix (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t new_rows_number = rows_number;
   const size_t new_columns_number = other_columns_number;

   Matrix<T> new_matrix(new_rows_number, new_columns_number);

   //new_matrix.initialize(other_vector);

   double sum;

   for(size_t i = 0; i < rows_number; i++)
   {
     for(size_t j = 0; j < other_columns_number; j++)
     {
        sum = 0.0;

       for(size_t k = 0; k < columns_number; k++)
       {
            sum += (*this)(i,k)*other_matrix(k,j);
       }

       sum += other_vector[j];

       new_matrix(i,j) = sum;
     }
   }

   return(new_matrix);
}


/*
template<class T>
Matrix<double> Matrix<T>::dot(const Vector< Matrix<double> >& vector_matrix) const
{
    Matrix<double> result(rows_number, vector_matrix[0].get_columns_number());

    for(size_t i = 0; i < rows_number; i++)
    {
        result.set_row(i, this->get_row(i).dot(vector_matrix[i]));
    }

    return result;
}
*/

/// Calculates the eigen values of this matrix, which must be squared.
/// Returns a matrix with only one column and rows the same as this matrix with the eigenvalues.

template<class T>
Matrix<double> Matrix<T>::calculate_eigenvalues() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values() const method.\n"
              << "Number of columns must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_rows_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values() const method.\n"
              << "Number of rows must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() != (*this).get_rows_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values() const method.\n"
              << "The matrix must be squared.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<T> eigenvalues(rows_number, 1);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> matrix_eigen(this_eigen, Eigen::EigenvaluesOnly);
    Eigen::Map<Eigen::MatrixXd> eigenvalues_eigen(eigenvalues.data(), rows_number, 1);

    eigenvalues_eigen = matrix_eigen.eigenvalues();

    return(eigenvalues);
}


/// Calculates the eigenvectors of this matrix, which must be squared.
/// Returns a matrix whose columns are the eigenvectors.

template<class T>
Matrix<double> Matrix<T>::calculate_eigenvectors() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values() const method.\n"
              << "Number of columns must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_rows_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values() const method.\n"
              << "Number of rows must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() != (*this).get_rows_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values() const method.\n"
              << "The matrix must be squared.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<T> eigenvectors(rows_number, rows_number);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> matrix_eigen(this_eigen, Eigen::ComputeEigenvectors);
    Eigen::Map<Eigen::MatrixXd> eigenvectors_eigen(eigenvectors.data(), rows_number, rows_number);

    eigenvectors_eigen = matrix_eigen.eigenvectors();

    return(eigenvectors);
}


/// Calculates the direct product of this matrix with another matrix.
/// This product is also known as the Kronecker product.
/// @param other_matrix Second product term.

template <class T>
Matrix<T> Matrix<T>::direct(const Matrix<T>& other_matrix) const
{
   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   Matrix<T> direct(rows_number*other_rows_number, columns_number*other_columns_number);

   size_t alpha;
   size_t beta;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            for(size_t k = 0; k < other_rows_number; k++)
            {
                for(size_t l = 0; l < other_columns_number; l++)
                {
                    alpha = other_rows_number*i+k;
                    beta = other_columns_number*j+l;

                    direct(alpha,beta) = (*this)(i,j)*other_matrix(k,l);
                }
            }
        }
    }

   return(direct);
}


template <class T>
void Matrix<T>::direct(const Vector<T>& v1, const Vector<T>& v2)
{
    const size_t new_size = v1.size();

    set(new_size, new_size);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(new_size); i++)
    {
      for(size_t j = 0; j < new_size; j++)
      {
        (*this)(i, j) = v1[i] * v2[j];
      }
    }
}


/// Returns true if number of rows and columns is zero.

template <class T>
bool Matrix<T>::empty() const
{
   if(rows_number == 0 && columns_number == 0)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


/// Returns true if this matrix is square.
/// A square matrix has the same numbers of rows and columns.

template <class T>
bool Matrix<T>::is_square() const
{
   if(rows_number == columns_number)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


/// Returns true if this matrix is symmetric.
/// A symmetric matrix is a squared matrix which is equal to its transpose.

template <class T>
bool Matrix<T>::is_symmetric() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_symmetric() const method.\n"
             << "Matrix must be squared.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Matrix<T> transpose = calculate_transpose();

   if((*this) == transpose)
   {
       return(true);
   }
   else
   {
       return(false);
   }
}


/// Returns true if this matrix is antysymmetric.
/// A symmetric matrix is a squared matrix which its opposed is equal to its transpose.

template <class T>
bool Matrix<T>::is_antisymmetric() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_antisymmetric() const method.\n"
             << "Matrix must be squared.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Matrix<T> transpose = calculate_transpose();

   if((*this) == transpose*(-1))
   {
       return(true);
   }
   else
   {
       return(false);
   }
}


/// Returns true if this matrix is diagonal.
/// A diagonal matrix is which the entries outside the main diagonal are zero.

template <class T>
bool Matrix<T>::is_diagonal() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_diagonal() const method.\n"
             << "Matrix must be squared.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(i != j &&(*this)(i,j) != 0)
         {
            return(false);
         }
      }
   }

   return(true);
}


/// Returns true if this matrix is scalar.
/// A scalar matrix is a diagonal matrix whose diagonal elements all contain the same scalar.

template <class T>
bool Matrix<T>::is_scalar() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_scalar() const method.\n"
             << "Matrix must be squared.\n";

      throw logic_error(buffer.str());
   }

   #endif

   return get_diagonal().is_constant();
}


/// Returns true if this matrix is the identity.
/// The identity matrix or unit matrix is a square matrix with ones on the main diagonal and zeros elsewhere.

template <class T>
bool Matrix<T>::is_identity() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_unity() const method.\n"
             << "Matrix must be squared.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(i != j &&(*this)(i,j) != 0)
         {
            return(false);
         }
         else if(i == j &&(*this)(i,j) != 1)
         {
            return(false);
         }
      }
   }

   return(true);
}


/// Returns true if this matrix has binary values.

template <class T>
bool Matrix<T>::is_binary() const
{
   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] != 0 &&(*this)[i] != 1)
         {
            return(false);
         }
   }

   return(true);
}


/// Returns true if a column this matrix has binary values.

template <class T>
bool Matrix<T>::is_column_binary(const size_t& j) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(j >= columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "bool is_column_binary(const size_t) const method method.\n"
              << "Index of column(" << j << ") must be less than number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

   for(size_t i = 0; i < rows_number; i++)
   {
         if((*this)(i,j) != 0 &&(*this)(i,j) != 1)
         {
            return(false);
         }
   }

   return(true);
}


template <class T>
bool Matrix<T>::compare_rows(const size_t& row_index, const Matrix<T>& other_matrix, const size_t& other_row_index) const
{
    for(size_t j = 0; j < columns_number; j++)
    {
        if((*this)(row_index, j) != other_matrix(other_row_index, j))
        {
            return(false);
        }
    }

    return(true);
}


/// Returns true if all the elements have the same value within a defined
/// tolerance,
/// and false otherwise.
/// @param tolerance Tolerance value, so that if abs(max-min) <= tol, then the
/// vector is considered constant.

template <class T> bool Matrix<T>::is_column_constant(const size_t& column_index) const
{
  if(columns_number == 0)
  {
    return(false);
  }

  const T initial_value = (*this)(0, column_index);

  for(size_t i = 1; i < rows_number; i++)
  {
      if((*this)(i,column_index) != initial_value)
      {
          return(false);
      }
  }

  return(true);
}


template <class T>
bool Matrix<T>::is_positive() const
{

  for(size_t i = 0; i < this->size(); i++)
  {
      if((*this)[i] < 0)
      {
          return(false);
      }
  }

  return(true);
}


/// Returns a new matrix where a given column has been filtered.
/// @param column_index Index of column.
/// @param minimum Minimum filtering value.
/// @param maximum Maximum filtering value.

template <class T>
Matrix<T> Matrix<T>::filter_minimum_maximum(const size_t& column_index, const T& minimum, const T& maximum) const
{
    const Vector<T> column = get_column(column_index);

    const size_t new_rows_number = column.count_between(minimum, maximum);

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) >= minimum &&(*this)(i,column_index) <= maximum)
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    return new_matrix;
}


template <class T>
Matrix<T> Matrix<T>::filter_minimum_maximum(const string& column_name, const T& minimum, const T& maximum) const
{
    const size_t column_index = get_column_index(column_name);

    return filter_minimum_maximum(column_index, minimum, maximum);
}


template <class T>
Matrix<T> Matrix<T>::filter_extreme_values(const size_t& column_index, const double& lower_ratio, const double& upper_ratio) const
{
    const size_t lower_index = rows_number*lower_ratio;
    const size_t upper_index = rows_number*upper_ratio;

    const Vector<T> column = get_column(column_index).sort_ascending_values();

    const T lower_value = column[lower_index];
    const T upper_value = column[upper_index];

    return filter_minimum_maximum(column_index, lower_value, upper_value);
}


template <class T>
Matrix<T> Matrix<T>::filter_extreme_values(const string& column_name, const double& lower_ratio, const double& upper_ratio) const
{
    const size_t column_index = get_column_index(column_name);

    return filter_extreme_values(column_index, lower_ratio, upper_ratio);
}


template <class T>
size_t Matrix<T>::count_dates(const size_t& column_index,
                              const size_t& start_day, const size_t& start_month, const size_t& start_year,
                              const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{    
    const time_t start_date = to_time_t(start_day, start_month, start_year);
    const time_t end_date = to_time_t(end_day, end_month, end_year);

    size_t count = 0;

    time_t date;

    for(size_t i = 0; i < rows_number; i++)
    {
        date = static_cast<time_t>(stoi((*this)(i, column_index)));

        if(date >= start_date && date <= end_date)
        {
            count++;
        }
    }

    return(count);
}


template <class T>
size_t Matrix<T>::count_dates(const string& column_name,
                              const size_t& start_day, const size_t& start_month, const size_t& start_year,
                              const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    const size_t column_index = get_column_index(column_name);

    return count_dates(column_index, start_day, start_month, start_year, end_day, end_month, end_year);
}


template <class T>
Matrix<T> Matrix<T>::filter_dates(const size_t& column_index,
                                  const size_t& start_day, const size_t& start_month, const size_t& start_year,
                                  const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    const size_t count = count_dates(column_index, start_day, start_month, start_year, end_day, end_month, end_year);

    if(count == 0) throw logic_error("OpenNN Exception: Matrix Template.\n"
                                     "Matrix<T> filter_dates(const size_t&) method.\n"
                                     "Number of rows between start and end dates is zero.\n");



    const size_t new_rows_number = count;

    Matrix<T> new_matrix(new_rows_number, columns_number);

    const time_t start_date = to_time_t(start_day, start_month, start_year);
    const time_t end_date = to_time_t(end_day, end_month, end_year);

    size_t index = 0;

    Vector<T> row(columns_number);

    time_t date;

    for(size_t i = 0; i < rows_number; i++)
    {
        date = static_cast<time_t>(stoi((*this)(i, column_index)));

        if(date >= start_date && date <= end_date)
        {
            row = get_row(i);

            new_matrix.set_row(index, row);

            index++;
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_dates(const string& column_name,
                                  const size_t& start_day, const size_t& start_month, const size_t& start_year,
                                  const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    const size_t column_index = get_column_index(column_name);

    return filter_dates(column_index, start_day, start_month, start_year, end_day, end_month, end_year);
}


template <class T>
size_t Matrix<T>::count_dates_string(const size_t& column_index,
                              const size_t& start_day, const size_t& start_month, const size_t& start_year,
                              const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    size_t count = 0;

    const time_t start_date = to_time_t(start_day, start_month, start_year);
    const time_t end_date = to_time_t(end_day, end_month, end_year);

    time_t date;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_index) == "") continue;

        try
        {
            date = static_cast<time_t>(stoi((*this)(i, column_index)));
        }
        catch(exception& e)
        {
            cerr << e.what() << ": " << (*this)(i, column_index) << endl;
        }

        if(date >= start_date && date <= end_date)
        {
            count++;
        }
    }

    return(count);
}


template <class T>
size_t Matrix<T>::count_dates_string(const string& column_name,
                              const size_t& start_day, const size_t& start_month, const size_t& start_year,
                              const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    const size_t column_index = get_column_index(column_name);

    return count_dates_string(column_index, start_day, start_month, start_year, end_day, end_month, end_year);
}


template <class T>
Matrix<T> Matrix<T>::filter_dates_string(const size_t& column_index,
                                  const size_t& start_day, const size_t& start_month, const size_t& start_year,
                                  const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{    
    const size_t count = count_dates_string(column_index, start_day, start_month, start_year, end_day, end_month, end_year);

    if(count == 0) throw logic_error("OpenNN Exception: Matrix Template.\n"
                                     "Matrix<T> filter_dates_string(const size_t&) method.\n"
                                     "Number of rows between start and end dates is zero.\n");

    const size_t new_rows_number = count;

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t index = 0;

    Vector<T> row(columns_number);

    const time_t start_date = to_time_t(start_day, start_month, start_year);
    const time_t end_date = to_time_t(end_day, end_month, end_year);

    time_t date;

    for(size_t i = 0; i < rows_number; i++)
    {
        try
        {
            date = static_cast<time_t>(stoi((*this)(i, column_index)));
        }
        catch(exception& e)
        {
            cerr << e.what() << ": " <<(*this)(i, column_index) << endl;
        }

        if(date >= start_date && date <= end_date)
        {
            row = get_row(i);

            new_matrix.set_row(index, row);

            index++;
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_dates_string(const string& column_name,
                                  const size_t& start_day, const size_t& start_month, const size_t& start_year,
                                  const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    const size_t column_index = get_column_index(column_name);

    return filter_dates_string(column_index, start_day, start_month, start_year, end_day, end_month, end_year);
}


template <class T>
Matrix<T> Matrix<T>::fill_missing_dates_dd_mm_yyyy(const string& column_name, const char& separator) const
{
    const size_t column_index = get_column_index(column_name);

    Matrix<T> data(*this);
    data.convert_time_column_dd_mm_yyyy(column_name, separator);

    const time_t start_date = stoi(data(1, column_index));

    const time_t end_date = stoi(data(rows_number-1, column_index));

    const size_t days_number = (end_date+86400-start_date)/86400;

    const size_t new_rows_number = 1+days_number;

    Matrix<T> new_data(new_rows_number, columns_number);

    time_t date;

    Vector<T> row;

    for(size_t i = 1; i < new_rows_number; i++)
    {
        date = start_date + 86400*(i-1);

        new_data(i, column_index) = to_string(date);

        for(size_t j = 1; j < rows_number; j++)
        {
            if(stoi(data(j, column_index)) == date)
            {
                row = data.get_row(j);
                new_data.set_row(i, row);
                break;
            }
        }
    }

    new_data.set_row(0, get_row(0));

    new_data.timestamp_to_date(column_name);

    new_data.replace("", "NA");

    return new_data;
}


template <class T>
Matrix<T> Matrix<T>::delete_Tukey_outliers_string(const size_t& column_index, const double& cleaning_parameter) const
{
    const Vector<double> column = get_column(column_index).string_to_double();

    const Vector<size_t> indices = column.calculate_Tukey_outliers_iterative(cleaning_parameter);

    return delete_rows(indices);
}


template <class T>
Matrix<T> Matrix<T>::delete_Tukey_outliers_string(const string& column_name, const double& cleaning_parameter) const
{
    const size_t column_index = get_column_index(column_name);

    return delete_Tukey_outliers_string(column_index, cleaning_parameter);
}


template <class T>
Matrix<T> Matrix<T>::delete_histogram_outliers(const size_t& column_index, const size_t& bins_number, const size_t& minimum_frequency) const
{
    const Vector<double> column = get_column(column_index);

//    const Vector<size_t> indices = column.calculate_histogram_outliers_iterative(bins_number, minimum_frequency);
    const Vector<size_t> indices = column.calculate_histogram_outliers(bins_number, minimum_frequency);

    if(indices.size() > 0)
    {
//        cout << "indices: " << indices.size() << endl;
    }

    return delete_rows(indices);
}


template <class T>
Matrix<T> Matrix<T>::delete_histogram_outliers(const string& column_name, const size_t& bins_number, const size_t& minimum_frequency) const
{
    const size_t column_index = get_column_index(column_name);

    return delete_histogram_outliers(column_index, bins_number, minimum_frequency);
}


/// Returns a new matrix where a given column has been filtered.
/// @param column_index Index of column.
/// @param value Filtering value.

template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const size_t& column_index, const T& value) const
{
    const size_t count = count_equal_to(column_index, value);

    if(count == 0) return Matrix<T>();

    const size_t new_rows_number = count;

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == value)
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const size_t& column_1, const T& value_1,
                                             const size_t& column_2, const T& value_2,
                                             const size_t& column_3, const T& value_3,
                                             const size_t& column_4, const T& value_4) const
{
    const size_t count = count_equal_to(column_1, value_1, column_2, value_2, column_3, value_3, column_4, value_4);

    const size_t new_rows_number = count;

    Matrix<T> new_matrix(new_rows_number, columns_number);

    Vector<T> row(columns_number);
    size_t row_index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_1) == value_1
        &&(*this)(i,column_2) == value_2
        &&(*this)(i,column_3) == value_3
        &&(*this)(i,column_4) == value_4)
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& column_1_name, const T& value_1,
                                             const string& column_2_name, const T& value_2,
                                             const string& column_3_name, const T& value_3,
                                             const string& column_4_name, const T& value_4) const
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);
    const size_t column_3_index = get_column_index(column_3_name);
    const size_t column_4_index = get_column_index(column_4_name);

    return(filter_column_equal_to(column_1_index, value_1, column_2_index, value_2, column_3_index, value_3, column_4_index, value_4));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const size_t& column_index, const Vector<T>& values) const
{
    const size_t values_size = values.size();

    const size_t count = count_equal_to(column_index, values);

    const size_t new_rows_number = count;

    Matrix<T> new_matrix(new_rows_number, columns_number);

    Vector<T> row(columns_number);

    size_t row_index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < values_size; j++)
        {
            if((*this)(i,column_index) == values[j])
            {
                row = get_row(i);

                new_matrix.set_row(row_index, row);

                row_index++;

                break;
            }
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_not_equal_to(const size_t& column_index, const Vector<T>& values) const
{
    const size_t values_size = values.size();

    const size_t new_rows_number = count_not_equal_to(column_index, values);

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        size_t count = 0;

        for(size_t j = 0; j < values_size; j++)
        {
            if((*this)(i,column_index) != values[j])
            {
                count++;
            }
        }

        if(count == values.size())
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const size_t& index_1, const Vector<T>& values_1,
                                             const size_t& index_2, const T& value_2) const
{    
    const size_t count = count_equal_to(index_1, values_1, index_2, value_2);

    const size_t new_rows_number = count;

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    const size_t values_1_size = values_1.size();

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, index_2) == value_2)
        {            
            for(size_t j = 0; j < values_1_size; j++)
            {
                if((*this)(i, index_1) == values_1[j])
                {
                    row = get_row(i);

                    new_matrix.set_row(row_index, row);

                    row_index++;

                    break;
                }
            }
        }
    }

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const size_t& column_1_index, const Vector<T>& values_1,
                                             const size_t& column_2_index, const T& value_2,
                                             const size_t& column_3_index, const T& value_3,
                                             const size_t& column_4_index, const T& value_4) const
{
    size_t new_rows_number = count_equal_to(column_1_index, values_1,
                                            column_2_index, value_2,
                                            column_3_index, value_3,
                                            column_4_index, value_4);

    Matrix<T> new_matrix(new_rows_number, columns_number);

    const size_t values_1_size = values_1.size();

    size_t row_index = 0;

    Vector<T> row(columns_number);

    T matrix_element;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_2_index) == value_2
        &&(*this)(i,column_3_index) == value_3
        &&(*this)(i,column_4_index) == value_4)
        {
            matrix_element = (*this)(i,column_1_index);

            for(size_t j = 0; j < values_1_size; j++)
            {
                if(values_1[j] == matrix_element)
                {
                    row = get_row(i);

                    new_matrix.set_row(row_index, row);

                    row_index++;

                    break;
                }
            }
        }
    }

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& column_1, const Vector<T>& values_1,
                                             const string& column_2, const T& value_2,
                                             const string& column_3, const T& value_3,
                                             const string& column_4, const T& value_4) const
{
    const size_t column_1_index = get_column_index(column_1);
    const size_t column_2_index = get_column_index(column_2);
    const size_t column_3_index = get_column_index(column_3);
    const size_t column_4_index = get_column_index(column_4);

    return(filter_column_equal_to(column_1_index, values_1, column_2_index, value_2, column_3_index, value_3, column_4_index, value_4));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_equal_to(column_index, value));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& column_name, const Vector<T>& values) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_equal_to(column_index, values));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_not_equal_to(const string& column_name, const Vector<T>& values) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_not_equal_to(column_index, values));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& name_1, const Vector<T>& values_1,
                                             const string& name_2, const T& value_2) const
{
    const size_t index_1 = get_column_index(name_1);
    const size_t index_2 = get_column_index(name_2);

    return(filter_column_equal_to(index_1, values_1, index_2, value_2));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_not_equal_to(const size_t& column_index, const T& value) const
{
    const size_t new_rows_number = count_not_equal_to(column_index, value);

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) != value)
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_not_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_not_equal_to(column_index, value));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_less_than(const size_t& column_index, const T& value) const
{
    const Vector<T> column = get_column(column_index);

    size_t new_rows_number = column.count_less_than(value);

    if(new_rows_number == 0)
    {
        Matrix<T> new_matrix;

        return new_matrix;
    }

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) < value)
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_less_than(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_less_than(column_index, value));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_greater_than(const size_t& column_index, const T& value) const
{
    const Vector<T> column = get_column(column_index);

    const size_t new_rows_number = column.count_greater_than(value);

    if(new_rows_number == 0)
    {
        Matrix<T> new_matrix;

        return new_matrix;
    }

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) > value)
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    new_matrix.set_header(header);

    return(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::filter_column_greater_than(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_greater_than(column_index, value));
}


template <class T>
Matrix<T> Matrix<T>::filter_column_less_than_string(const string& name, const double& value) const
{
    const Vector<size_t> indices = get_column(name).string_to_double().get_indices_less_than(value);

    return delete_rows(indices);
}



template <class T>
Matrix<T> Matrix<T>::filter_column_greater_than_string(const string& name, const double& value) const
{
    const Vector<size_t> indices = get_column(name).string_to_double().get_indices_greater_than(value);

    return delete_rows(indices);
}


template <class T>
void Matrix<T>::convert_time_column_dd_mm_dd_yyyy_hh(const string& column_name, const char& separator)
{
    // Mon Jan 30 2017 12:52:24

    const size_t column_index = get_column_index(column_name);

    vector<string> date_elements;
    vector<string> time_elements;

    int year;
    int month;
    int month_day;
    int hours;
    int minutes;
    int seconds;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == "")
        {
            continue;
        }

        date_elements = split_string((*this)(i,column_index), separator);

        if(date_elements.size() != 5)
        {
            continue;
        }

        // Month

        if(date_elements[1] == "Jan")
        {
            month = 1;
        }
        else if(date_elements[1] == "Feb")
        {
            month = 2;
        }
        else if(date_elements[1] == "Mar")
        {
            month = 3;
        }
        else if(date_elements[1] == "Apr")
        {
            month = 4;
        }
        else if(date_elements[1] == "May")
        {
            month = 5;
        }
        else if(date_elements[1] == "Jun")
        {
            month = 6;
        }
        else if(date_elements[1] == "Jul")
        {
            month = 7;
        }
        else if(date_elements[1] == "Aug")
        {
            month = 8;
        }
        else if(date_elements[1] == "Sep")
        {
            month = 9;
        }
        else if(date_elements[1] == "Oct")
        {
            month = 10;
        }
        else if(date_elements[1] == "Nov")
        {
            month = 11;
        }
        else if(date_elements[1] == "Dec")
        {
            month = 12;
        }
        else
        {
            cout << "Unknown month: " << month << endl;
        }

        // Month day

        month_day = stoi(date_elements[2]);

        // Year

        year = stoi(date_elements[3]);

        // Time

        time_elements = split_string(date_elements[4], ':');

        hours = stoi(time_elements[0]);
        minutes = stoi(time_elements[1]);
        seconds = stoi(time_elements[2]);

        struct tm timeinfo;
        timeinfo.tm_year = year - 1900;
        timeinfo.tm_mon = month - 1;
        timeinfo.tm_mday = month_day;

        timeinfo.tm_hour = hours;
        timeinfo.tm_min = minutes;
        timeinfo.tm_sec = seconds;

       (*this)(i,column_index) = to_string(mktime(&timeinfo));
    }
}


template <class T>
void Matrix<T>::convert_time_column_yyyy_MM(const string& column_name, const char& separator)
{
    // data format -> 1985-4
    const size_t column_index = get_column_index(column_name);
    vector<string> date_elements;
    int year;
    int month;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == "")
        {
            continue;
        }
        date_elements = split_string((*this)(i,column_index), separator);
        if(date_elements.size() != 2)
        {
            continue;
        }
        // Month
        if(date_elements[1] == "1")
        {
            month = 1;
        }
        else if(date_elements[1] == "2")
        {
            month = 2;
        }
        else if(date_elements[1] == "3")
        {
            month = 3;
        }
        else if(date_elements[1] == "4")
        {
            month = 4;
        }
        else if(date_elements[1] == "5")
        {
            month = 5;
        }
        else if(date_elements[1] == "6")
        {
            month = 6;
        }
        else if(date_elements[1] == "7")
        {
            month = 7;
        }
        else if(date_elements[1] == "8")
        {
            month = 8;
        }
        else if(date_elements[1] == "9")
        {
            month = 9;
        }
        else if(date_elements[1] == "10")
        {
            month = 10;
        }
        else if(date_elements[1] == "11")
        {
            month = 11;
        }
        else if(date_elements[1] == "12")
        {
            month = 12;
        }
        else
        {
            cout << "Unknown month: " << month << endl;
        }
        // Year
        year = stoi(date_elements[0]);
        struct tm timeinfo;
        timeinfo.tm_year = year - 1900;
        timeinfo.tm_mon = month - 1;

       (*this)(i,column_index) = to_string(mktime(&timeinfo));
    }
}


template <class T>
void Matrix<T>::convert_time_column_dd_mm_yyyy(const string& column_name, const char& separator)
{
    // data format -> 13/04/1873

    const size_t column_index = get_column_index(column_name);

    Vector<string> date_elements;

    int year;
    int month;
    int day;

    string day_string;
    string month_string;
    string year_string;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == "")
        {
            cout << "Empty element" << endl;
            continue;
        }

        date_elements = split_string((*this)(i, column_index), separator);

        if(date_elements.size() != 3)
        {
            cout << "Number of elements must be 3: " <<(*this)(i,column_index) << endl;
            continue;
        }

        day_string = date_elements[0];
        month_string = date_elements[1];
        year_string = date_elements[2];

        day = stoi(day_string);

        month = stoi(month_string);

        year = stoi(year_string);

        const time_t time = to_time_t(day, month, year);

       (*this)(i, column_index) = to_string(time);
    }
}


template <class T>
void Matrix<T>::convert_time_column_mm_dd_yyyy(const string& column_name, const char& separator)
{
    // date format: 01/31/2017

    const size_t column_index = get_column_index(column_name);

    vector<string> date_elements;

    int year;
    int month;
    int day;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == "")
        {
            continue;
        }

        date_elements = split_string((*this)(i,column_index), separator);

        if(date_elements.size() != 3)
        {
            continue;
        }

        // Month

        month = stoi(date_elements[0]);

        if(month < 1 || month > 12)
        {
            cout << "Unknown month: " << month << endl;
        }

        // Day

        day = stoi(date_elements[1]);

        if(day < 1 || day > 31)
        {
            cout << "Unknown day: " << day << endl;
        }

        // Year

        year = stoi(date_elements[2]);

        const time_t time = to_time_t(day, month, year);

       (*this)(i,column_index) = to_string(time);
    }
}


template <class T>
void Matrix<T>::convert_time_column_yyyy_MM_dd_hh_mm_ss(const string& column_name, const char& separator)
{
    // date format: 01/31/2017/01/00/00

    const size_t column_index = get_column_index(column_name);

    vector<string> date_elements;

    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == "") continue;

        date_elements = split_string((*this)(i,column_index), separator);

        if(date_elements.size() != 6) continue;

        // Year

        year = stoi(date_elements[0]);

        // Month

        month = stoi(date_elements[1]);

        if(month < 1 || month > 12)
        {
            cout << "Unknown month: " << month << endl;
        }

        // Day

        day = stoi(date_elements[2]);

        if(day < 1 || day > 31)
        {
            cout << "Unknown day: " << day << endl;
        }

        // Hour

        hour = stoi(date_elements[3]);

        if(hour < 0 || hour > 23)
        {
            cout << "Unknown hour: " << hour << endl;
        }

        // Minute

        minute = stoi(date_elements[4]);

        if(minute < 0 || minute > 59)
        {
            cout << "Unknown minute: " << minute << endl;
        }

        // Second

        second = stoi(date_elements[5]);

        if(second < 0 || second > 59)
        {
            cout << "Unknown second: " << second << endl;
        }

        const time_t time = to_time_t(day, month, year, hour, minute, second);

       (*this)(i,column_index) = to_string(time);
    }
}


template <class T>
void Matrix<T>::timestamp_to_date(const string& column_name)
{
    const size_t column_index = get_column_index(column_name);

    for(size_t i = 0; i < rows_number; i++)
    {
        time_t date = stoi((*this)(i, column_index));

        char date_char[20];
        strftime(date_char, 20, "%d/%m/%Y", localtime(&date));

        const string date_string(date_char);

       (*this)(i, column_index) = date_string;

    }
}


/// Arranges a time series data matrix in a proper format for forecasting.
/// Note that this method sets new numbers of rows and columns in the matrix.
/// @param lags_number Number of lags for the prediction.
/// @param column_index Index of the time column.

template <class T>
void Matrix<T>::convert_time_series(const size_t& time_index, const size_t& lags_number, const size_t& steps_ahead_number)
{
    const Vector<T> time = get_column(time_index);

   (*this) = delete_column(time_index);

    const size_t new_rows_number = rows_number - lags_number - steps_ahead_number + 1;
    const size_t new_columns_number = columns_number *(lags_number + steps_ahead_number);

    const Vector<size_t> indices(lags_number, 1, rows_number-1);

    const Vector<T> new_time = time.get_subvector(indices);

    Matrix<T> new_matrix(new_rows_number, new_columns_number);

    Vector<T> new_row(new_columns_number);

    for(size_t i = 0; i < new_rows_number; i++)
    {
        new_row = get_rows(i + 1, i + lags_number + steps_ahead_number);

        new_matrix.set_row(i, new_row);
    }

    new_matrix = new_matrix.insert_column(0, new_time, "time");

    set(new_matrix);
}


template <class T>
Matrix<T> Matrix<T>::get_time_series(const size_t& lags_number, const size_t& time_variable_index)
{
    const Vector<T> time = get_column(time_variable_index);

    Matrix<T> matrix = (*this);

    matrix = matrix.delete_column(time_variable_index);

    const Vector<size_t> indices(lags_number, 1, rows_number - 1);

    const Vector<T> first_column = time.get_subvector(indices);

    const size_t new_rows_number = rows_number - lags_number;
    const size_t new_columns_number = columns_number *(1 + lags_number);

    Matrix<T> new_matrix(new_rows_number, new_columns_number);

    Vector<T> row(rows_number);

    for(size_t i = 0; i < new_rows_number; i++)
    {
        row = get_row(i);

        for(size_t j = 1; j <= lags_number; j++)
        {
            row = row.assemble(get_row(i+j));
        }

        new_matrix.set_row(i, row);
    }

    new_matrix = new_matrix.insert_column(0, first_column);

    return(new_matrix);
}


// @todo

template <class T>
Matrix<T> Matrix<T>::calculate_lag_plot_matrix(const size_t& maximum_lags_number, const size_t& column_index)
{
    Matrix<T> matrix = (*this);

    matrix = matrix.delete_column(column_index);

    const size_t new_rows_number = rows_number - maximum_lags_number;
    const size_t new_columns_number = (columns_number - 1) *(1 + maximum_lags_number);

    Matrix<T> lag_plot_matrix(new_rows_number, new_columns_number);

    Vector<T> row(rows_number);

    for(size_t i = 0; i < new_rows_number; i++)
    {
        row = matrix.get_row(i);

        for(size_t j = 1; j <= maximum_lags_number; j++)
        {
            row = row.assemble(matrix.get_row(i+j));
        }

        lag_plot_matrix.set_row(i, row);
    }

    return lag_plot_matrix;
}


/// Arranges the matrix in a proper format for association.
/// Note that this method sets new numbers of columns in the matrix.

template <class T>
void Matrix<T>::convert_association()
{
    Matrix<T> copy(*this);

    set(copy.assemble_columns(copy));
}


/// Converts a given column, representing angles in degrees, to two different columns with the sinus and the cosinus of the corresponding angles.
/// Note that this method sets a new number of columns in the matrix.
/// @param column_index Index of column to be converted.

template <class T>
void Matrix<T>::convert_angular_variables_degrees(const size_t& column_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "void convert_angular_variables_degrees(const size_t&) method.\n"
              << "Index of column(" << column_index << ") must be less than number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const double pi = 4.0*atan(1.0);

    Vector<T> sin_angle(rows_number);
    Vector<T> cos_angle(rows_number);

    double angle_rad;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) != -99.9)
        {
            angle_rad = pi*(*this)(i,column_index)/180.0;

            sin_angle[i] = sin(angle_rad);
            cos_angle[i] = cos(angle_rad);
        }
        else
        {
            sin_angle[i] = static_cast<T>(-99.9);
            cos_angle[i] = static_cast<T>(-99.9);
        }
    }

    set_column(column_index, sin_angle, "");
    set(insert_column(column_index+1, cos_angle, ""));
}


/// Converts a given column, representing angles in radians, to two different columns with the sinus and the cosinus of the corresponding angles.
/// Note that this method sets a new number of columns in the matrix.
/// @param column_index Index of column to be converted.

template <class T>
void Matrix<T>::convert_angular_variables_radians(const size_t& column_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "void convert_angular_variables_radians(const size_t&) method.\n"
              << "Index of column(" << column_index << ") must be less than number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Vector<T> sin_angle(rows_number);
    Vector<T> cos_angle(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        sin_angle[i] = sin((*this)(i,column_index));
        cos_angle[i] = cos((*this)(i,column_index));
    }

    set_column(column_index, sin_angle, "");
    set(insert_column(column_index+1, cos_angle, ""));
}


/// Returns a vector containing the parameters of a multiple linear regression analysis
/// @param other Independent vector.

template <class T> Vector<T> Matrix<T>::calculate_multiple_linear_regression_parameters(const Vector<T>& other) const
{
  // Control sentence(if debug)

  #ifdef __OPENNN_DEBUG__

    const size_t data_size = this->get_rows_number();
    const size_t other_size = other.size();

    ostringstream buffer;

    if(other_size != data_size) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "LinearRegressionParameters<T> "
                "calculate_multiple_linear_regression_parameters(const Vector<T>&) const "
                "method.\n"
             << "Independent vector size must be equal to this size.\n";

      throw logic_error(buffer.str());
    }

  #endif

  const Matrix<T> matrix_transposed = this->calculate_transpose();

  const Matrix<T> matrix_product = matrix_transposed.dot((*this));

  const Matrix<T> first_factor = matrix_product.calculate_LU_inverse();

  const Vector<T> second_factor = matrix_transposed.dot(other);

  return first_factor.dot(second_factor);
}


template <class T>
double Matrix<T>::calculate_logistic_function(const Vector<double>& coefficients, const Vector<T>& x) const
{
    const size_t coefficients_size = coefficients.size();

    double exponential = coefficients[0];

    for(size_t i = 1; i < coefficients_size; i++)
    {
        exponential += coefficients[i]*x[i-1];
    }

    return(1.0/(1.0+exp(-exponential)));
}


template <class T>
Vector<double> Matrix<T>::calculate_logistic_error_gradient(const Vector<double>& coefficients, const Vector<T>& other) const
{
    const size_t n = this->get_rows_number();

    const size_t other_size = this->size();

    Vector<double> error_gradient(coefficients.size()+1, 0.0);

    size_t negatives_number = 0;
    size_t positives_number = 0;

    for(size_t i = 0; i < other_size; i++)
    {
        if(other[i] == 1)
        {
            positives_number++;
        }
        else if(other[i] == 0)
        {
            negatives_number++;
        }
    }

    double negatives_weight = 1.0;

    double positives_weight = 1.0;

    if(positives_number == 0)
    {
        positives_weight = 1.0;
        negatives_weight = 1.0;
    }
    else if(negatives_number == 0)
    {
        positives_weight = 1.0;
        negatives_weight = 1.0;

        negatives_number = 1;
    }
    else
    {
        positives_weight = static_cast<double>(negatives_number)/static_cast<double>(positives_number);
    }

#pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        Vector<T> x = this->get_row(i);

        double current_logistic_function = this->calculate_logistic_function(coefficients, x);

        double exponential = coefficients[0];

        for(size_t j = 1; j < coefficients.size(); j++)
        {
            exponential += coefficients[j]*x[j-1];
        }

        const double gradient_multiply = exp(-exponential)*(other[i] - current_logistic_function)*current_logistic_function*current_logistic_function;

        Vector<double> this_error_gradient(coefficients.size()+1, 0.0);

        this_error_gradient[0] += (other[i]*positives_weight + (1-other[i])*negatives_weight)*(other[i] - current_logistic_function)*(other[i] - current_logistic_function)/2;
        this_error_gradient[1] -= (other[i]*positives_weight + (1-other[i])*negatives_weight)*gradient_multiply;

        for(size_t j = 2; j <= coefficients.size(); j++)
        {
            this_error_gradient[j] -= (other[i]*positives_weight + (1-other[i])*negatives_weight)*x[j-2]*gradient_multiply;
        }

#pragma omp critical
        {
            error_gradient += this_error_gradient;
        }
    }

    return error_gradient/static_cast<double>(negatives_weight*negatives_number);
}

template <class T>
KMeansResults<T> Matrix<T>::calculate_k_means(const size_t& k) const
{
    KMeansResults<double> k_means_results;

    Vector< Vector<size_t> > clusters(k);

    Matrix<double> previous_means(k, columns_number);
    Matrix<double> means(k, columns_number);

    const Vector<T> minimums = calculate_columns_minimums();
    const Vector<T> maximums = calculate_columns_maximums();

    size_t iterations = 0;

    bool end = false;

    // Calculate initial means

    Vector<size_t> selected_rows(k);

    const size_t initial_center = calculate_random_uniform<size_t>(0, rows_number);

    previous_means.set_row(0, get_row(initial_center));
    selected_rows[0] = initial_center;

    for(size_t i = 1; i < k; i++)
    {
        Vector<double> minimum_distances(rows_number, 0.0);

#pragma omp parallel for

        for(int j = 0; j < rows_number; j++)
        {
            Vector<double> distances(i, 0.0);

            const Vector<T> row_data = get_row(j);

            for(size_t l = 0; l < i; l++)
            {
                distances[l] = row_data.calculate_distance(previous_means.get_row(l));
            }

            const double minimum_distance = distances.calculate_minimum();

            minimum_distances[static_cast<size_t>(j)] = minimum_distance;
        }

        size_t sample_index = minimum_distances.calculate_sample_index_proportional_probability();

        int random_failures = 0;

        while(selected_rows.contains(sample_index))
        {
            sample_index = minimum_distances.calculate_sample_index_proportional_probability();

            random_failures++;

            if(random_failures > 5)
            {
                Vector<double> new_row(columns_number);

                new_row.randomize_uniform(minimums, maximums);

                previous_means.set_row(i, new_row);

                break;
            }
        }

        if(random_failures <= 5)
        {
            previous_means.set_row(i, get_row(sample_index));
        }
    }

    // Main loop

    while(!end)
    {
        clusters.clear();
        clusters.set(k);

#pragma omp parallel for

        for(int i = 0; i < rows_number; i++)
        {
            Vector<double> distances(k, 0.0);

            const Vector<T> current_row = get_row(i);

            for(size_t j = 0; j < k; j++)
            {
                distances[j] = current_row.calculate_distance(previous_means.get_row(j));
            }

            const size_t minimum_distance_index = distances.calculate_minimal_index();

#pragma omp critical
            clusters[minimum_distance_index].push_back(static_cast<size_t>(i));
        }

        for(size_t i = 0; i < k; i++)
        {
            means.set_row(i,calculate_rows_means(clusters[i]));
        }

        if(previous_means == means)
        {
            end = true;
        }
        else if(iterations > 100)
        {
            end = true;
        }

        previous_means = means;
        iterations++;
    }

//    k_means_results.means = means;
    k_means_results.clusters = clusters;

    return(k_means_results);
}


template <class T>
Matrix<T> Matrix<T>::calculate_lower_bounded(const T & lower_bound) const
{
    const size_t this_size = this->size();

    Matrix<T> bounded_matrix(*this);

    for(size_t i = 0; i < this_size; i++)
    {
      if((*this)[i] < lower_bound)
      {
        bounded_matrix[i] = lower_bound;
      }
    }

    return(bounded_matrix);
}


template <class T>
Matrix<T> Matrix<T>::calculate_upper_bounded(const T & upper_bound) const
{
    const size_t this_size = this->size();

    Matrix<T> bounded_matrix(*this);

    for(size_t i = 0; i < this_size; i++)
    {
      if((*this)[i] > upper_bound)
      {
        bounded_matrix[i] = upper_bound;
      }
    }

    return(bounded_matrix);
}


template <class T>
Matrix<T> Matrix<T>::calculate_lower_upper_bounded(const Vector<T>& lower_bounds, const Vector<T>& upper_bounds) const
{
    Matrix<T> bounded_matrix(*this);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(bounded_matrix(i,j) < lower_bounds[j]) bounded_matrix(i,j) = lower_bounds[j];
            else if(bounded_matrix(i,j) > upper_bounds[j]) bounded_matrix(i,j) = upper_bounds[j];
        }
    }

    return bounded_matrix;
}


/// Prints to the screen in the matrix object.

template <class T>
void Matrix<T>::print() const
{
   cout << *this << endl;
}


/// Loads the numbers of rows and columns and the values of the matrix from a data file.
/// @param file_name File name.

template <class T>
void Matrix<T>::load(const string& file_name)
{
   ifstream file(file_name.c_str());

   if(!file.is_open())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "void load(const string&) method.\n"
             << "Cannot open matrix data file: " << file_name << "\n";

      throw logic_error(buffer.str());
   }

   if(file.peek() == ifstream::traits_type::eof())
   {
      //ostringstream buffer;

      //buffer << "OpenNN Exception: Matrix template.\n"
      //       << "void load(const string&) method.\n"
      //       << "Data file " << file_name << " is empty.\n";

      //throw logic_error(buffer.str());

      this->set();

      return;
   }

   //file.is

   // Set matrix sizes

   string line;

   getline(file, line);

   if(line.empty())
   {
      set();
   }
   else
   {
       istringstream buffer(line);

       istream_iterator<string> it(buffer);
       istream_iterator<string> end;

       const vector<string> results(it, end);

      const size_t new_columns_number =static_cast<size_t>(results.size());

      size_t new_rows_number = 1;

      while(file.good())
      {
         getline(file, line);

         if(!line.empty())
         {
            new_rows_number++;
         }
      }

      set(new_rows_number, new_columns_number);

      // Clear file

      file.clear();
      file.seekg(0, ios::beg);

      for(size_t i = 0; i < rows_number; i++)
      {
          for(size_t j = 0; j < columns_number; j++)
          {
              file >>(*this)(i,j);
          }
      }
   }

   // Close file

   file.close();
}


template <class T>
void Matrix<T>::load_csv(const string& file_name, const char& delim, const bool& has_header)
{
    ifstream file(file_name.c_str());

    if(!file.is_open())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "void load_csv(const string&,const char&) method.\n"
              << "Cannot open matrix data file: " << file_name << "\n";

       throw logic_error(buffer.str());
    }

    if(file.peek() == ifstream::traits_type::eof())
    {
       set();

       return;
    }

    // Set matrix sizes

    string line;

    getline(file, line);

    if(line.empty())
    {
       set();
    }
    else
    {
       istringstream buffer(line);

       string token;

       vector<string> results;

       while(getline(buffer, token, delim))
       {
           results.push_back(token);
       }

       const size_t new_columns_number = static_cast<size_t>(results.size());

       size_t new_rows_number = 1;

       while(file.good())
       {
          getline(file, line);

          if(!line.empty())
          {
             new_rows_number++;
          }
       }

       if(has_header)
       {
           new_rows_number--;
           header.set(results);
       }

       set(new_rows_number, new_columns_number);

       if(!has_header)
       {
           header.set();
       }

       // Clear file

       file.clear();
       file.seekg(0, ios::beg);

       if(has_header)
       {
           getline(file, line);

           istringstream header_buffer(line);

           for(size_t j = 0; j < columns_number; j++)
           {
               string token;

               getline(header_buffer, token, delim);

               header[j] = token;
           }
       }

       for(size_t i = 0; i < rows_number; i++)
       {
           getline(file, line);

           //std::replace(line.begin(), line.end(), delim, ' ');

           istringstream buffer(line);

           for(size_t j = 0; j < columns_number; j++)
           {
               string token;

               getline(buffer, token, delim);

               (*this)(i,j) = token;

               //buffer >> (*this)(i,j);
           }
       }
    }

    // Close file

    file.close();
}


/// Loads the numbers of rows and columns and the values of the matrix from a binary file.
/// @param file_name Name of binary file.

template <class T>
void Matrix<T>::load_binary(const string& file_name)
{
    ifstream file;

    file.open(file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix template.\n"
               << "void load_binary(const string&) method.\n"
               << "Cannot open binary file: " << file_name << "\n";

        throw logic_error(buffer.str());
    }

    streamsize size = sizeof(size_t);

    size_t columns_number;
    size_t rows_number;

    file.read(reinterpret_cast<char*>(&columns_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(double);

    double value;

    this->set(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        for(size_t j = 0; j < rows_number; j++)
        {
            file.read(reinterpret_cast<char*>(&value), size);

           (*this)(j,i) = value;
        }
    }

    file.close();
}


template <class T>
void Matrix<T>::load_product_strings(const string& file_name, const char& separator)
{
    Vector<string> products;

    ifstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix template.\n"
               << "void load_csv(const string&,const char&) method.\n"
               << "Cannot open matrix data file: " << file_name << "\n";

        throw logic_error(buffer.str());
    }

    if(file.peek() == ifstream::traits_type::eof())
    {
        this->set();

//        return products;
    }

    //file.is

    // Set matrix sizes

    string line;

    getline(file, line);

    if(line.empty())
    {
        set();
    }
    else
    {
        string token;

        istringstream buffer(line);

        while(getline(buffer, token, separator))
        {
            products.push_back(token);
        }

        size_t new_rows_number = 1;

        while(file.good())
        {
            getline(file, line);

            istringstream buffer(line);

            while(getline(buffer, token, separator))
            {
                products.push_back(token);
            }

            if(!line.empty())
            {
                new_rows_number++;
            }
        }

        products = products.get_unique_elements();

        const size_t new_columns_number = static_cast<size_t>(products.size());

        set(new_rows_number, new_columns_number, 0);

        set_header(products);

        // Clear file

        file.clear();
        file.seekg(0, ios::beg);

        for(size_t i = 0; i < rows_number; i++)
        {
            getline(file, line);

            istringstream buffer(line);

            while(getline(buffer, token, separator))
            {
                const size_t current_column = products.calculate_equal_to_indices(token)[0];

               (*this)(i,current_column) = 1;
            }
        }
    }

    // Close file

    file.close();

}


/// Saves the values of the matrix to a data file separated by spaces.
/// @param file_name File name.

template <class T>
void Matrix<T>::save(const string& file_name) const
{
   ofstream file(file_name.c_str());

   if(!file.is_open())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << endl
             << "void save(const string) method." << endl
             << "Cannot open matrix data file." << endl;

      throw logic_error(buffer.str());
   }

   // Write file

   file.precision(20);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         file <<(*this)(i,j) << " ";
      }

      file << endl;
   }

   // Close file

   file.close();
}


/// Saves the values of the matrix to a binary file.
/// @param file_name File name.

template <class T>
void Matrix<T>::save_binary(const string& file_name) const
{
   ofstream file(file_name.c_str(), ios::binary);

   if(!file.is_open())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << endl
             << "void save(const string) method." << endl
             << "Cannot open matrix binary file." << endl;

      throw logic_error(buffer.str());
   }

   // Write data

   streamsize size = sizeof(size_t);

   size_t m = columns_number;
   size_t n = rows_number;

   file.write(reinterpret_cast<char*>(&m), size);
   file.write(reinterpret_cast<char*>(&n), size);

   size = sizeof(double);

   double value;

   for(int i = 0; i < this->size(); i++)
   {
       value = (*this)[i];

       file.write(reinterpret_cast<char*>(&value), size);
   }

   // Close file

   file.close();
}


/// Saves the values of the matrix to a data file separated by commas.
/// @param file_name File name.
/// @param column_names Names of the columns.

template <class T>
void Matrix<T>::save_csv(const string& file_name, const char& separator,  const Vector<string>& row_names, const string& nameID) const
{
   ofstream file(file_name.c_str());

   if(!file.is_open())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << endl
             << "void save_csv(const string&, const char&, const Vector<string>&, const Vector<string>&) method." << endl
             << "Cannot open matrix data file: " << file_name << endl;

      throw logic_error(buffer.str());
   }

   if(row_names.size() != 0 && row_names.size() != rows_number)
   {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template." << endl
              << "void save_csv(const string&, const char&, const Vector<string>&, const Vector<string>&) method." << endl
              << "Row names must have size 0 or " << rows_number << "." << endl;

       throw logic_error(buffer.str());
   }

   // Write file

   if(header != "")
   {
       if(!row_names.empty())
       {
           file << nameID << separator;
       }

       for(size_t j = 0; j < columns_number; j++)
       {
           file << header[j];

           if(j != columns_number-1)
           {
               file << separator;
           }
       }

       file << endl;
   }

   file.precision(20);

   for(size_t i = 0; i < rows_number; i++)
   {
       if(!row_names.empty())
       {
           file << row_names[i] << separator;
       }

       for(size_t j = 0; j < columns_number; j++)
       {
           file <<(*this)(i,j);

           if(j != columns_number-1)
           {
               file << separator;
           }
       }

       file << endl;
   }

   // Close file

   file.close();
}


/// Saves the values of the matrix to a data file in JSON format.
/// @param file_name File name.
/// @param column_names Names of the columns.

template <class T>
void Matrix<T>::save_json(const string& file_name, const Vector<string>& column_names) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template." << endl
              << "void save_json(const string&, const Vector<string>&) method." << endl
              << "Cannot open matrix data file." << endl;

       throw logic_error(buffer.str());
    }

    if(column_names.size() != 0 && column_names.size() != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix template." << endl
               << "void save_json(const string&, const Vector<string>&) method." << endl
               << "Column names must have size 0 or " << columns_number << "." << endl;

        throw logic_error(buffer.str());
    }

    // Write file

    Vector<string> header;

    if(column_names.empty())
    {
        header.set(columns_number);

        for(size_t i = 0; i < columns_number; i++)
        {
            header[i] = "variable_" + to_string(i);
        }
    }
    else
    {
        header = column_names;
    }

    file.precision(20);

    file << "{ \"rows_number\": " << rows_number
         << ", \"columns_number\": " << columns_number << ", ";

    for(size_t i = 0; i < columns_number; i++)
    {
        file << "\"" << header[i] << "\": [";

        for(size_t j = 0; j < rows_number; j++)
        {
            file <<(*this)(j,i);

            if(j != rows_number-1)
            {
                file << ", ";
            }
        }

        file << "]";

        if(i != columns_number-1)
        {
            file << ", ";
        }
    }

    file << "}";

    // Close file

    file.close();
}


/// This method takes a string representation of a matrix and sets this matrix
/// from that data.
/// @param str String to be parsed.

template <class T>
void Matrix<T>::parse(const string& str)
{
   if(str.empty())
   {
       set();
   }
   else
   {
        // Set matrix sizes

        istringstream str_buffer(str);

        string line;

        getline(str_buffer, line);

        istringstream line_buffer(line);

        istream_iterator<string> it(line_buffer);
        istream_iterator<string> end;

        const vector<string> results(it, end);

        const size_t new_columns_number = static_cast<size_t>(results.size());

        size_t new_rows_number = 1;

        while(str_buffer.good())
        {
            getline(str_buffer, line);

            if(!line.empty())
            {
                new_rows_number++;
            }
        }

        set(new_rows_number, new_columns_number);

      // Clear file

      str_buffer.clear();
      str_buffer.seekg(0, ios::beg);

      for(size_t i = 0; i < rows_number; i++)
      {
         for(size_t j = 0; j < columns_number; j++)
         {
            str_buffer >>(*this)(i,j);
         }
      }
   }
}


/// Returns a string representation of this matrix.
/// The elements are separated by spaces.
/// The rows are separated by the character "\n".

template <class T>
string Matrix<T>::matrix_to_string(const char& separator) const
{
   ostringstream buffer;

   if(rows_number > 0 && columns_number > 0)
   {
       buffer << get_header().vector_to_string(separator);

       for(size_t i = 0; i < rows_number; i++)
       {
           buffer << "\n"
                  << get_row(i).vector_to_string(separator);
       }
   }

   return(buffer.str());
}


template <class T>
Matrix<size_t> Matrix<T>::to_size_t_matrix() const
{
   Matrix<size_t> size_t_matrix(rows_number, columns_number);

   const size_t this_size = this->size();

   for(size_t i = 0; i < this_size; i++)
   {
       size_t_matrix[i] = static_cast<size_t>((*this)[i]);
   }

   return(size_t_matrix);
}


template <class T>
Matrix<double> Matrix<T>::to_double_matrix() const
{
   Matrix<double> new_matrix(rows_number, columns_number);

   const size_t this_size = this->size();

   for(size_t i = 0; i < this_size; i++)
   {
       new_matrix[i] = static_cast<double>((*this)[i]);
   }

   return(new_matrix);
}


template <class T>
Matrix<double> Matrix<T>::bool_to_double(const double& exception_value) const
{
   Matrix<double> new_matrix(rows_number, columns_number);

   const size_t this_size = this->size();

   for(size_t i = 0; i < this_size; i++)
   {
       try
       {
           new_matrix[i] = (*this)[i] ? 1.0 : 0.0;
       }
       catch(const logic_error&)
       {
          new_matrix[i] = exception_value;
       }
   }

   new_matrix.set_header(header);

   return(new_matrix);
}


template <class T>
Matrix<double> Matrix<T>::string_to_double(const double& exception_value) const
{
   Matrix<double> new_matrix(rows_number, columns_number);

   const size_t this_size = this->size();

   for(size_t i = 0; i < this_size; i++)
   {
       try
       {
           new_matrix[i] = stod((*this)[i]);
       }
       catch(const logic_error&)
       {
          new_matrix[i] = exception_value;
       }
   }

   new_matrix.set_header(header);

   return(new_matrix);
}


template <class T>
Matrix<size_t> Matrix<T>::string_to_size_t(const size_t& exception_value) const
{
   Matrix<size_t> new_matrix(rows_number, columns_number);

   const size_t this_size = this->size();

   for(size_t i = 0; i < this_size; i++)
   {
       try
       {
           new_matrix[i] = static_cast<size_t>(stoi((*this)[i]));
       }
       catch(const logic_error&)
       {
          new_matrix[i] = exception_value;
       }
   }

   return(new_matrix);
}


/// Returns a new matrix in which each entry has been converted to a string.

template <class T>
Matrix<string> Matrix<T>::to_string_matrix(const size_t& precision) const
{
   Matrix<string> string_matrix(rows_number, columns_number);

   ostringstream buffer;

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         buffer.str("");
         buffer << setprecision(precision) <<(*this)(i,j);

         string_matrix(i,j) = buffer.str();
      }
   }

   if(header != "") string_matrix.set_header(header);

   return(string_matrix);
}


template <class T>
SparseMatrix<T> Matrix<T>::to_sparse_matrix() const
{
    SparseMatrix<T> sparse_matrix(rows_number,columns_number);

    const size_t nonzero_elements_number = this->count_not_equal_to(T());

    Vector<size_t> rows_indices(nonzero_elements_number);
    Vector<size_t> columns_indices(nonzero_elements_number);
    Vector<T> matrix_values(nonzero_elements_number);

    size_t index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) != T())
            {
                rows_indices[index] = i;
                columns_indices[index] = j;
                matrix_values[index] = (*this)(i,j);

                index++;
            }
        }
    }

    sparse_matrix.set_values(rows_indices, columns_indices, matrix_values);

    return sparse_matrix;
}


/// Returns a vector representation of this matrix.
/// The size of the new vector is equal to the number of elements of this matrix.
/// The entries of the new vector are the entries of this matrix ordered by rows.

template <class T>
vector<T> Matrix<T>::to_std_vector() const
{
    const vector<T> std_vector((*this).begin(),(*this).end());

    return(std_vector);
}


/// Returns a vector representation of this matrix.
/// The size of the new vector is equal to the number of elements of this matrix.
/// The entries of the new vector are the entries of this matrix ordered by rows.

template <class T>
Vector<T> Matrix<T>::to_vector() const
{
    const Vector<T> vector((*this).begin(),(*this).end());

    return(vector);
}


template <class T>
Vector<T> Matrix<T>::rows_to_vector(const char& separator) const
{
    Vector<T> vector(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        vector[i] = (*this).get_row(i).vector_to_string(separator);
    }

   return(vector);
}


/// Returns a vector of vectors representation of this matrix.
/// The number of subvectors is equal to the number of columns of this matrix.
/// The size of each subvector is equal to the number of rows of this matrix.

template <class T>
Vector< Vector<T> > Matrix<T>::to_vector_of_vectors() const
{
    Vector< Vector<T> > vector_of_vectors(columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        vector_of_vectors[i] = this->get_column(i);
    }

   return(vector_of_vectors);
}


/// Prints to the sceen a preview of the matrix,
/// i.e., the first, second and last rows

template <class T>
void Matrix<T>::print_preview() const
{
   cout << "Rows number: " << rows_number << endl
        << "Columns number: " << columns_number << endl;

   cout << "Header:\n" << header << endl;

   if(rows_number > 0)
   {
      const Vector<T> first_row = get_row(0);

      cout << "Row 0:\n" << first_row << endl;
   }

   if(rows_number > 1)
   {
      const Vector<T> second_row = get_row(1);

      cout << "Row 1:\n" << second_row << endl;
   }

   if(rows_number > 3)
   {
      const Vector<T> row = get_row(rows_number-2);

      cout << "Row " << rows_number-1 << ":\n" << row << endl;
   }

   if(rows_number > 2)
   {
      const Vector<T> last_row = get_row(rows_number-1);

      cout << "Row " << rows_number << ":\n" << last_row << endl;
   }
}


template<class T>
Matrix<T> Matrix<T>::impute_missing_values_time_series_value(const size_t& time_index, const double& period, const T& value) const
{
    const Vector<T> time = get_column(time_index);

    const int first_time = static_cast<int>(get_first(time_index));
    const int last_time = static_cast<int>(get_last(time_index));

    const int elapsed_time = last_time - first_time;

    if(elapsed_time%static_cast<int>(period) != 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "Matrix<T> impute_missing_values_time_series_previous(const size_t&, const double&) const.\n"
               << "Wrong time column.\n";

        throw logic_error(buffer.str());
    }

    const size_t new_rows_number = (last_time-first_time)/period;

    if(new_rows_number == rows_number) return Matrix<T>(*this);

    Matrix<T> new_data(new_rows_number, columns_number, value);

    for(size_t i = 0; i < new_rows_number; i++)
    {
        const T new_time = first_time + i*period;

        try
        {
            const size_t row_index = time.get_first_index(new_time);

            const Vector<T> new_row = get_row(row_index);

            new_data.set_row(i, new_row);

        }
        catch (const exception&)
        {
            // do nothing
        }

        new_data(i, time_index) = new_time;
    }

    return new_data;
}


template<class T>
Matrix<T> Matrix<T>::impute_missing_values_time_series_previous(const size_t& time_index, const double& period) const
{
    const T value = 999999999999;

    Matrix<T> new_data = impute_missing_values_time_series_value(time_index, period, value);

    for(size_t i = 1; i < rows_number; i++)
    {
        if(new_data.get_row(i).contains(value))
        {
            const T new_time = new_data(i, time_index);

            const Vector<T> previous_row = get_row(i-1);

            new_data.set_row(i, previous_row);

            new_data(i, time_index) = new_time;
        }
    }

    return new_data;
}


template <class T>
time_t Matrix<T>::to_time_t(const size_t& day, const size_t& month, const size_t& year)
{
    struct tm time = {};
    time.tm_mday = static_cast<int>(day);
    time.tm_mon = static_cast<int>(month) - 1;
    time.tm_year = static_cast<int>(year) - 1900;

    time.tm_hour = 1;
    time.tm_min = 0;
    time.tm_sec = 0;

    return mktime(&time);
}


template <class T>
time_t Matrix<T>::to_time_t(const size_t& day, const size_t& month, const size_t& year, const size_t& hour, const size_t& minute, const size_t& second)
{
    struct tm time = {};
    time.tm_mday = static_cast<int>(day);
    time.tm_mon = static_cast<int>(month) - 1;
    time.tm_year = static_cast<int>(year) - 1900;

    time.tm_hour = static_cast<int>(hour);
    time.tm_min = static_cast<int>(minute);
    time.tm_sec = static_cast<int>(second);

    return mktime(&time);
}


/// This method re-writes the input operator >> for the Matrix template.
/// @param is Input stream.
/// @param m Input matrix.

template<class T>
istream& operator >>(istream& is, Matrix<T>& m)
{
   const size_t rows_number = m.get_rows_number();
   const size_t columns_number = m.get_columns_number();

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         is >> m(i,j);
      }
   }

   return(is);
}


/// This method re-writes the output operator << for the Matrix template.
/// @param os Output stream.
/// @param m Output matrix.

template<class T>
ostream& operator <<(ostream& os, const Matrix<T>& m)
{
   const size_t rows_number = m.get_rows_number();
   const size_t columns_number = m.get_columns_number();

   if(m.get_header() != "") cout << m.get_header() << endl;

   if(rows_number > 0 && columns_number > 0)
   {
       os << m.get_row(0);

       for(size_t i = 1; i < rows_number; i++)
       {
           os << "\n"
              << m.get_row(i);
       }
   }

   return(os);
}


/// This method re-writes the output operator << for matrices of vectors.
/// @param os Output stream.
/// @param m Output matrix of vectors.

template<class T>
ostream& operator << (ostream& os, const Matrix< Vector<T> >& m)
{
   const size_t rows_number = m.get_rows_number();
   const size_t columns_number = m.get_columns_number();

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         os << "subvector_" << i << "_" << j << "\n"
            << m(i,j) << endl;
      }
   }

   return(os);
}


// Output operator

/// This method re-writes the output operator << for matrices of matrices.
/// @param os Output stream.
/// @param m Output matrix of matrices.

template<class T>
ostream& operator << (ostream& os, const Matrix< Matrix<T> >& m)
{
   const size_t rows_number = m.get_rows_number();
   const size_t columns_number = m.get_columns_number();

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         os << "submatrix_" << i << "_" << j << "\n"
            << m(i,j) << endl;
      }
   }

   return(os);
}


template<class T>
Matrix<T> sine(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        y[i] = sin(x[i]);
    }

    return y;
}


template<class T>
Matrix<T> cosine(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        y[i] = cos(x[i]);
    }

    return y;
}


template<class T>
Matrix<T> hyperbolic_tangent(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        y[i] = tanh(x[i]);
    }

    return y;
}


template<class T>
Matrix<T> hyperbolic_tangent_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }

    return y;
}


template<class T>
Matrix<T> linear(const Matrix<T>& x)
{
    return x;
}


template<class T>
Matrix<T> linear_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number(), 1);

    return y;
}


template<class T>
Matrix<T> linear_second_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number(), 0);

    return y;
}


template<class T>
Matrix<T> hyperbolic_tangent_second_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = -2*hyperbolic_tangent*(1 - hyperbolic_tangent*hyperbolic_tangent);
    }

    return y;
}


template<class T>
Matrix<T> logistic(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        y[i] = 1.0/(1.0 + exp(-x[i]));
    }

    return y;
}


template<class T>
Matrix<T> logistic_derivatives(const Matrix<T>& x)
{
     Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }

    return y;
}


template<class T>
Matrix<T> logistic_second_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

   for(size_t i = 0; i < x.size(); i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = (exponential*exponential - exponential)/((1.0 + exponential)*(1.0 + exponential)*(1.0 + exponential));
    }

    return y;
}
/// @todo

template<class T>
Matrix<T> threshold(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

     for(size_t i = 0; i < x.size(); i++)
     {
         if(x[i] < 0)
         {
             y[i] = 0.0;
         }
         else
         {
             y[i] = 1.0;
         }
     }

     return y;
}


template<class T>
Matrix<T> threshold_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

     for(size_t i = 0; i < x.size(); i++)
     {
         if(x[i] == 0)
         {
             ostringstream buffer;

             buffer << "OpenNN Exception: Matrix Template.\n"
                    << "Matrix<T> threshold_derivatives(const Matrix<T>&).\n"
                    << "Derivate does not exist for x equal to 0.\n";

             throw logic_error(buffer.str());
         }
         else
         {
             y[i] = 0.0;
         }
     }

     return y;
}


template<class T>
Matrix<T> threshold_second_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

     for(size_t i = 0; i < x.size(); i++)
     {
         if(x[i] == 0)
         {
             ostringstream buffer;

             buffer << "OpenNN Exception: Matrix Template.\n"
                    << "Matrix<T> threshold_derivatives(const Matrix<T>&).\n"
                    << "Derivate does not exist for x equal to 0.\n";

             throw logic_error(buffer.str());
         }
         else
         {
             y[i] = 0.0;
         }
     }

     return y;
}


/// @todo

template<class T>
Matrix<T> symmetric_threshold(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

     for(size_t i = 0; i < x.size(); i++)
    {
        if(x[i] < 0)
        {
            y[i] = -1.0;
        }
        else
        {
            y[i] = 1.0;
        }
    }

    return y;
}


template<class T>
Matrix<T> symmetric_threshold_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        if(x[i] == 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: Matrix Template.\n"
                   << "Matrix<T> threshold_derivatives(const Matrix<T>&).\n"
                   << "Derivate does not exist for x equal to 0.\n";

            throw logic_error(buffer.str());
        }
        else
        {
            y[i] = 0.0;
        }
    }

    return y;
}


template<class T>
Matrix<T> symmetric_threshold_second_derivatives(const Matrix<T>& x)
{
    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        if(x[i] == 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: Matrix Template.\n"
                   << "Matrix<T> threshold_derivatives(const Matrix<T>&).\n"
                   << "Derivate does not exist for x equal to 0.\n";

            throw logic_error(buffer.str());
        }
        else
        {
            y[i] = 0.0;
        }
    }

    return y;
}


template<class T>
Matrix<T> rectified_linear(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = 0.0 : y[i] = x[i];
    }

    return y;
}


template<class T>
Matrix<T> rectified_linear_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> derivatives(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? derivatives[i] = 0.0 : derivatives[i] = 1.0;
    }

    return derivatives;
}


template<class T>
Matrix<T> rectified_linear_second_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> second_derivatives(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? second_derivatives[i] = 0.0 : second_derivatives[i] = 0.0;
    }


    return second_derivatives;
}


template<class T>
Matrix<T> scaled_exponential_linear(const Matrix<T>& x)
{
    const size_t n = x.size();

    double lambda =1.0507;
    double alpha =1.67326;


    Matrix<T> y(x.get_rows_number(), x.get_columns_number());


    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
    }

    return y;
}


template<class T>
Matrix<T> scaled_exponential_linear_derivate(const Matrix<T>& x)
{
    const size_t n = x.size();

    double lambda =1.0507;
    double alpha =1.67326;


    Matrix<T> derivatives(x.get_rows_number(), x.get_columns_number());


    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? derivatives[i] = lambda * alpha * exp(x[i]) : derivatives[i] = lambda;
    }

    return derivatives;
}

template<class T>
Matrix<T> scaled_exponential_linear_second_derivate(const Matrix<T>& x)
{
    const size_t n = x.size();

    double lambda =1.0507;
    double alpha =1.67326;


    Matrix<T> second_derivate(x.get_rows_number(), x.get_columns_number());


    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? second_derivate[i] = lambda * alpha * exp(x[i]) : second_derivate[i] = 0.0;
    }

    return second_derivate;
}

template<class T>
Matrix<T> soft_plus(const Matrix<T>& x)
{
     const size_t n = x.size();

      Matrix<T> y(x.get_rows_number(), x.get_columns_number());

     for(size_t i = 0; i < n; i++)
     {
         y[i] = log(1 + exp(x[i]));
     }

     return y;
}

template<class T>
Matrix<T> soft_plus_derivatives(const Matrix<T>& x)
{
     const size_t n = x.size();

     Matrix<T> derivatives(x.get_rows_number(), x.get_columns_number());

     for(size_t i = 0; i < n; i++)
     {
         derivatives[i] = 1/(1 + exp(-x[i]));
     }

     return derivatives;
}

template<class T>
Matrix<T> soft_plus_second_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> second_derivatives(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
       second_derivatives[n] = exp(-x[i]) / pow((1 + exp(-x[i])), 2);
    }

    return second_derivatives;
}


template<class T>
Matrix<T> soft_sign(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
    }

    return y;
}


template<class T>
Matrix<T> soft_sign_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> derivatives(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? derivatives[i] = 1 / pow((1 - x[i]), 2) : derivatives[i] = 1 / pow((1 + x[i]), 2);

    }

    return derivatives;
}


template<class T>
Matrix<T> soft_sign_second_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> second_derivatives(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? second_derivatives[i] = -(2 * x[i]) / pow((1 - x[i]), 3) : second_derivatives[i] = -(2 * x[i]) / pow((1 + x[i]), 3);
    }

    return second_derivatives;
}


template<class T>
Matrix<T> hard_sigmoid(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
        if(x[i] < -2.5)
        {
           y[i] = 0;
        }
        else if(x[i] > 2.5)
        {
            y[i] = 1;
        }
        else
        {
            y[i] = 0.2 * x[i] + 0.5;
        }
    }

    return y;
}


template<class T>
Matrix<T> hard_sigmoid_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

   Matrix<T> derivatives(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < n; i++)
    {
        x[i] < -2.5 || x[i] > 2.5 ? derivatives[i] = 0.0 : derivatives[i] = 0.2;
    }

    return derivatives;
}


template<class T>
Matrix<T> hard_sigmoid_second_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> second_derivatives(x.get_rows_number(), x.get_columns_number(), 0.0);

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? second_derivatives[i] = 0.0 : second_derivatives[i] = 0.0;
    }

    return second_derivatives;
}


template<class T>
Matrix<T> exponential_linear(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> y(x.get_rows_number(), x.get_columns_number(), 0.0);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
    }

    return y;
}


template<class T>
Matrix<T> exponential_linear_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> derivatives(x.get_rows_number(), x.get_columns_number(), 0.0);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? derivatives[i] = alpha * exp(x[i]) : derivatives[i] = 1.0;
    }

    return derivatives;
}


template<class T>
Matrix<T> exponential_linear_second_derivatives(const Matrix<T>& x)
{
    const size_t n = x.size();

    Matrix<T> second_derivatives(x.get_rows_number(), x.get_columns_number(), 0.0);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? second_derivatives[i] = alpha * exp(x[i]) : second_derivatives[i] = 0.0;
    }

    return second_derivatives;
}

}

// end namespace

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
