/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V E C T O R   C O N T A I N E R                                                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __VECTOR_H__
#define __VECTOR_H__

// System includes

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <istream>
#include <map>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <climits>
#include <ctime>
#include <time.h>

#ifdef __OPENNN_MPI__
#include <mpi.h>
#endif

// Eigen includes

#include "../eigen/Eigen"

using namespace std;

namespace OpenNN {

// Forward declarations

template <class T> class Matrix;

template <class T> T calculate_random_uniform(const T & = -1, const T & = 1);

template <class T> T calculate_random_normal(const T & = 0.0, const T & = 1.0);

template <class T> string write_elapsed_time(const T&);

template <class T> string write_date_from_time_t(const T&);

template <class T> vector<string> split_string(const T&, const char&);

template <class T> void replace_substring(T& source, const T& find, const T& replace);

template <class T> struct Histogram;
template <class T> struct Statistics;
template <class T> struct LinearRegressionParameters;
template <class T> struct LogisticRegressionParameters;
template <class T> struct KMeansResults;


/// This template represents an array of any kind of numbers or objects.
/// It inherits from the vector of the standard library, and implements
/// additional utilities.

template <typename T> class Vector : public vector<T> {
public:

  // CONSTRUCTORS

  // Default constructor.

  explicit Vector();

  // General constructor.

  explicit Vector(const size_t &);

  // Constant reference initialization constructor.

  explicit Vector(const size_t &, const T &);

  // File constructor.

  explicit Vector(const string &);

  // Sequential constructor.

  explicit Vector(const T &, const double &, const T &);

  // Input iterators constructor

  template <class InputIterator> explicit Vector(InputIterator, InputIterator);

  // Copy constructor.

  Vector(const vector<T> &);

  Vector(const Vector<T> &);

  // Initializer list

  Vector(const initializer_list<T>&);

  Vector(const Vector< Vector<T> >&);


  // DESTRUCTOR

  virtual ~Vector();

  // OPERATORS

  bool operator== (const T &) const;

  bool operator!= (const T &) const;

  bool operator>(const T &) const;

  bool operator<(const T &) const;

  bool operator>= (const T &) const;

  bool operator<= (const T &) const;

  // METHODS

  // Get methods

  // Set methods

  void set();

  void set(const size_t &);

  void set(const size_t &, const T &);

  void set(const string &);

  void set(const T &, const double &, const T &);

  void set(const Vector &);

#ifdef __OPENNN_MPI__
  void set_MPI(const MPI_Datatype);
#endif

  T get_first() const;
  T get_last() const;
  T get_before_last() const;

  // Initialization methods

  void initialize(const T &);
  void initialize_first(const size_t&, const T &);

  void initialize_sequential();

  void randomize_uniform(const double & = -1.0, const double & = 1.0);
  void randomize_uniform(const Vector<double>&, const Vector<double>&);

  void randomize_normal(const double & = 0.0, const double & = 1.0);
  void randomize_normal(const Vector<double> &, const Vector<double> &);

  void randomize_binary(const double & = 0.5, const double & = 0.5);

  void map(Vector<T>&, const T&, const T&);
  void map(Vector<T>&, Vector<T>&, const T&, const T&, const T&);

  void trim();
  Vector<T> trimmed() const;

  // Fill method

  Vector<T> fill_from(const size_t&, const Vector<T>&) const;

  // Checking methods

  bool contains(const T &) const;

  bool contains_greater_than(const T &) const;

  bool contains(const Vector<T> &) const;

  bool has_same_elements(const Vector<T>&) const;

  bool is_in(const T &, const T &) const;

  bool is_constant(const double & = 0.0) const;
  bool is_constant_string() const;

  bool is_crescent() const;

  bool is_decrescent() const;

  bool is_binary() const;
  bool is_binary_0_1() const;
  bool is_binary(const Vector<size_t>&) const;

  bool is_integer() const;
  bool is_integer(const Vector<size_t>&) const;

  bool is_discrete(const size_t&) const;
  bool is_discrete(const Vector<size_t>&, const size_t&) const;


  bool is_positive() const;
  bool is_negative() const;

  bool check_period(const double& period) const;

  bool perform_Lilliefors_normality_test(const double&) const;
  Vector<bool> perform_Lilliefors_normality_test(const Vector<double>&) const;

  double calculate_normal_distribution_distance() const;
  double calculate_half_normal_distribution_distance() const;
  double calculate_uniform_distribution_distance() const;

  Vector<bool> perform_normality_analysis() const;

  double calculate_normality_parameter() const;

  Vector<T> calculate_variation_percentage() const;

  size_t perform_distribution_distance_analysis() const;
  size_t perform_distribution_distance_analysis_missing_values(const Vector<size_t>&) const;

  int get_lower_index(const size_t&, const T&) const;

  int get_upper_index(const size_t&, const T&) const;

  Vector<T> get_reverse() const;

  Vector<T> impute_time_series_missing_values_mean(const T&) const;

  // String methods

  void replace_substring(const string&, const string&);

  // Count methods

  size_t count_equal_to(const T&) const;
  double count_equal_to(const T&, const Vector<double>&) const;
  size_t count_equal_to(const Vector<T>&) const;

  size_t count_not_equal_to(const T&) const;
  size_t count_not_equal_to(const Vector<T>&) const;

  size_t count_positive() const;
  size_t count_negative() const;

  size_t count_integers(const size_t&) const;
  size_t count_integers_missing_values(const Vector<size_t>&, const size_t&) const;

  Vector<size_t> get_indices_equal_to(const T &) const;
  Vector<size_t> get_indices_less_than(const T &) const;
  Vector<size_t> get_indices_greater_than(const T &) const;

  size_t count_greater_than(const T &) const;

  size_t count_less_than(const T &) const;

  size_t count_greater_equal_to(const T &) const;

  size_t count_less_equal_to(const T &) const;

  size_t count_between(const T &, const T &) const;

  Matrix<T> count_daily_series_occurrences() const;
  Matrix<T> count_weekly_series_occurrences() const;
  Matrix<T> count_monthly_series_occurrences() const;
  Matrix<T> count_yearly_series_occurrences() const;

  Matrix<T> count_monthly_series_occurrences(const size_t&, const size_t&, const size_t&, const size_t&) const;

  Matrix<T> count_monthly_occurrences() const;

  size_t count_date_occurrences(const size_t&) const;
  size_t count_month_occurrences(const size_t&) const;
  size_t count_date_occurrences(const size_t&, const size_t&) const;

  size_t count_contains(const string&) const;

  Vector<T> merge(const Vector<T>&, const char&) const;

  Vector<T> filter_equal_to(const T&) const;
  Vector<T> filter_not_equal_to(const T&) const;

  Vector<T> filter_equal_to(const Vector<T>&) const;
  Vector<T> filter_not_equal_to(const Vector<T>&) const;

  Vector<T> filter_numbers() const;
  Vector<T> filter_not_numbers() const;

  Vector<T> get_positive_elements() const;

  Vector<size_t> calculate_between_indices(const T&, const T&) const;

  Vector<size_t> calculate_equal_to_indices(const T &) const;
  Vector<size_t> calculate_equal_to_indices(const Vector<T>&) const;

  Vector<size_t> calculate_not_equal_to_indices(const T &) const;
  Vector<size_t> calculate_not_equal_to_indices(const Vector<T> &) const;

  Vector<T> filter_minimum_maximum(const T&, const T&) const;

  Vector<size_t> calculate_contains_indices(const string&) const;

  Vector<size_t> calculate_less_than_indices(const T &) const;

  Vector<size_t> calculate_greater_than_indices(const T &) const;

  Vector<size_t> calculate_less_equal_to_indices(const T &) const;

  Vector<size_t> calculate_greater_equal_to_indices(const T &) const;

  Vector<size_t> calculate_total_frequencies(const Vector< Histogram<T> > &) const;

  Vector<size_t> calculate_total_frequencies_missing_values(const Vector<size_t> &, const Vector< Histogram<T> > &) const;

  Vector<double> perform_Box_Cox_transformation(const double& = 1) const;

  Vector<double> calculate_percentage(const size_t&) const;

  double calculate_error(const Vector<T>&) const;

  // Statistics methods

  T calculate_minimum() const;

  T calculate_maximum() const;

  Vector<T> calculate_minimum_maximum() const;

  T calculate_minimum_missing_values(const Vector<size_t> &) const;

  T calculate_maximum_missing_values(const Vector<size_t> &) const;

  Vector<T>
  calculate_minimum_maximum_missing_values(const Vector<size_t> &) const;

  Vector<T> calculate_explained_variance() const;

  // Histogram methods

  Histogram<T> calculate_histogram(const size_t & = 10) const;
  Histogram<T> calculate_histogram_centered(const double& = 0.0, const size_t & = 10) const;
  Histogram<T> calculate_histogram_binary() const;

  Histogram<T> calculate_histogram_integers(const size_t & = 10) const;

  Histogram<T> calculate_histogram_doubles() const;

  Histogram<T> calculate_histogram_missing_values(const Vector<size_t> &, const size_t & = 10) const;

  Histogram<T> calculate_histogram_binary_missing_values(const Vector<size_t> &) const;

  Histogram<T> calculate_histogram_integers_missing_values(const Vector<size_t> &, const size_t & = 10) const;

  Vector<T> calculate_moving_average(const T&) const;

  Vector<T> calculate_moving_average_cyclic(const T&) const;

  Vector<double> calculate_simple_moving_average(const size_t&) const;

  Vector<double> calculate_exponential_moving_average(const size_t&) const;

  double calculate_last_exponential_moving_average(const size_t&) const;

  Vector<double> calculate_exponential_moving_average_with_initial_average(const size_t&) const;

  Vector<size_t> calculate_bars_chart() const;

  size_t get_first_index(const T&) const;

  size_t calculate_minimal_index() const;

  size_t calculate_maximal_index() const;

  Vector<size_t> calculate_minimal_indices(const size_t &) const;
  Vector<size_t> calculate_k_minimal_indices(const size_t &) const;

  Vector<size_t> calculate_maximal_indices(const size_t &) const;

  Vector<size_t> calculate_minimal_maximal_index() const;

  Vector<T> calculate_pow(const T &) const;

  Vector<T> calculate_competitive() const;

  Vector<T> calculate_softmax() const;

  Matrix<T> calculate_softmax_Jacobian() const;

  Vector<bool> calculate_binary() const;

  Vector<T> calculate_square_root_elements() const;

  Vector<T> calculate_cumulative() const;

  size_t calculate_cumulative_index(const T &) const;

  size_t calculate_closest_index(const T &) const;

  T calculate_sum() const;

  Vector<T> calculate_sum_gradient() const;

  Matrix<T> calculate_sum_Hessian() const;

  T calculate_partial_sum(const Vector<size_t> &) const;

  T calculate_sum_missing_values(const Vector<size_t> &) const;

  T calculate_product() const;

  double calculate_mean() const;

  double calculate_mean(const size_t&, const size_t&) const;

  double calculate_linear_trend() const;

  double calculate_linear_trend(const size_t&, const size_t&) const;

  double calculate_percentage_of_variation() const;

  Vector<double> calculate_percentage_of_variation(const size_t&) const;

  double calculate_last_percentage_of_variation(const size_t&) const;

  T calculate_mode() const;

  T calculate_mode_missing_values(const Vector<size_t>&) const;

  double calculate_variance() const;

  double calculate_covariance(const Vector<double>&) const;

  double calculate_standard_deviation() const;

  Vector<double> calculate_standard_deviation(const size_t&) const;

  double calculate_asymmetry() const;

  double calculate_kurtosis() const;

  double calculate_median() const;

  Vector<double> calculate_quartiles() const;

  Vector<double> calculate_quartiles_missing_values(const Vector<size_t> &) const;

  Vector<double> calculate_percentiles() const;

  Vector<double> calculate_mean_standard_deviation() const;

  double calculate_mean_missing_values(const Vector<size_t> &) const;

  double calculate_variance_missing_values(const Vector<size_t> &) const;

  double calculate_weighted_mean(const Vector<double> &) const;

  double calculate_standard_deviation_missing_values(const Vector<size_t> &) const;

  double calculate_asymmetry_missing_values(const Vector<size_t> &) const;

  double calculate_kurtosis_missing_values(const Vector<size_t> &) const;

  Statistics<T> calculate_statistics() const;

  Statistics<T> calculate_statistics_missing_values(const Vector<size_t> &) const;

  Vector<double> calculate_shape_parameters() const;

  Vector<double> calculate_shape_parameters_missing_values(const Vector<size_t> &) const;

  Vector<double> calculate_box_plot() const;

  Vector<double> calculate_box_plot_missing_values(const Vector<size_t> &) const;

  size_t calculate_sample_index_proportional_probability() const;

  // Norm methods

  double calculate_L1_norm() const;

  Vector<T> calculate_sign() const;

  Vector<T> calculate_L1_norm_gradient() const;

  Matrix<T> calculate_L1_norm_Hessian() const;

  double calculate_L2_norm() const;

  Vector<T> calculate_L2_norm_gradient() const;

  Matrix<T> calculate_L2_norm_Hessian() const;

  double calculate_Lp_norm(const double &) const;

  Vector<double> calculate_Lp_norm_gradient(const double &) const;

  Vector<T> calculate_normalized() const;

  //double calculate_distance(const Vector<double> &) const;

  double calculate_euclidean_distance(const Vector<T> &) const;
  double calculate_euclidean_weighted_distance(const Vector<T>&, const Vector<double>&) const;
  Vector<double> calculate_euclidean_weighted_distance_vector(const Vector<T>&, const Vector<double>&) const;

  double calculate_manhattan_distance(const Vector<T> &) const;
  double calculate_manhattan_weighted_distance(const Vector<T>&, const Vector<double>&) const;
  Vector<double> calculate_manhattan_weighted_distance_vector(const Vector<T>&, const Vector<double>&) const;

  double calculate_sum_squared_error(const Vector<double> &) const;
  double calculate_sum_squared_error(const Matrix<T> &, const size_t &,
                                     const Vector<size_t> &) const;

  double calculate_Minkowski_error(const Vector<double> &,
                                   const double &) const;

  LinearRegressionParameters<T> calculate_linear_regression_parameters(const Vector<T> &) const;

  Vector<T> calculate_absolute_value() const;

  void apply_absolute_value();

  // Bounding methods

  Vector<T> calculate_lower_bounded(const T &) const;

  Vector<T> calculate_lower_bounded(const Vector<T> &) const;

  Vector<T> calculate_upper_bounded(const T &) const;

  Vector<T> calculate_upper_bounded(const Vector<T> &) const;

  Vector<T> calculate_lower_upper_bounded(const T &, const T &) const;

  Vector<T> calculate_lower_upper_bounded(const Vector<T> &,
                                          const Vector<T> &) const;

  void apply_lower_bound(const T &);

  void apply_lower_bound(const Vector<T> &);

  void apply_upper_bound(const T &);

  void apply_upper_bound(const Vector<T> &);

  void apply_lower_upper_bounds(const T &, const T &);

  void apply_lower_upper_bounds(const Vector<T> &, const Vector<T> &);

  // Rank methods

  Vector<size_t> sort_ascending_indices() const;
  Vector<T> sort_ascending_values() const;

  Vector<size_t> calculate_lower_indices(const size_t&) const;
  Vector<T> calculate_lower_values(const size_t&) const;

  Vector<size_t> sort_descending_indices() const;
  Vector<T> sort_descending_values() const;

  Vector<size_t> calculate_less_rank() const;

  Vector<double> calculate_less_rank_with_ties() const;

  Vector<size_t> calculate_greater_rank() const;

  Vector<size_t> calculate_greater_indices() const;

  Vector<T> sort_rank(const Vector<size_t>&) const;

  // Mathematical operators

  inline Vector<T> operator= (const initializer_list<T> &) const;

  inline Vector<T> operator+ (const T &) const;

  inline Vector<T> operator+ (const Vector<T> &) const;

  inline Vector<T> operator-(const T &) const;

  inline Vector<T> operator-(const Vector<T> &) const;

  inline Vector<T> operator*(const T &) const;

  inline Vector<T> operator*(const Vector<T> &) const;

  inline Matrix<T> operator*(const Matrix<T> &) const;

  inline double dot(const Vector<double> &) const;

  Vector<double> dot(const Matrix<T> &) const;

  Matrix<T> direct(const Vector<T> &) const;

  Vector<T> operator/(const T &) const;

  Vector<T> operator/(const Vector<T> &) const;

  void operator+= (const T &);

  void operator+= (const Vector<T> &);

  void operator-= (const T &);

  void operator-= (const Vector<T> &);

  void operator*= (const T &);

  void operator*= (const Vector<T> &);

  void operator/= (const T &);

  void operator/= (const Vector<T> &);

  // Filtering methods

  Vector<T> filter_positive() const;
  Vector<T> filter_negative() const;

  size_t count_dates(const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;
  Vector<size_t> filter_dates(const size_t&, const size_t&, const size_t&, const size_t&, const size_t&, const size_t&) const;

  Vector<size_t> calculate_Tukey_outliers(const double& = 1.5) const;
  Vector<size_t> calculate_Tukey_outliers_iterative(const double& = 1.5) const;

  Vector<size_t> calculate_histogram_outliers(const size_t&, const size_t&) const;
  Vector<size_t> calculate_histogram_outliers_iterative(const size_t&, const size_t&) const;

  // Scaling methods

  void scale_minimum_maximum(const T &, const T &);

  void scale_minimum_maximum(const Statistics<T> &);

  Statistics<T> scale_minimum_maximum();

  void scale_mean_standard_deviation(const T &, const T &);

  void scale_mean_standard_deviation(const Statistics<T> &);

  Statistics<T> scale_mean_standard_deviation();

  void scale_standard_deviation(const T &);

  void scale_standard_deviation(const Statistics<T> &);

  Statistics<T> scale_standard_deviation();

  void scale_standard_deviation(const Vector<T> &);

  Vector<T> calculate_scaled_minimum_maximum() const;
  Vector<T> calculate_scaled_minimum_maximum_0_1() const;

  Vector<T> calculate_scaled_minimum_maximum(const Vector<T> &,
                                             const Vector<T> &) const;

  Vector<T> calculate_scaled_mean_standard_deviation() const;

  Vector<T> calculate_scaled_mean_standard_deviation(const Vector<T> &,
                                                     const Vector<T> &) const;

  Vector<T> calculate_scaled_standard_deviation(const Vector<T> &) const;

  // Unscaling methods

  Vector<T> calculate_unscaled_minimum_maximum(const Vector<T> &, const Vector<T> &) const;

  Vector<T> calculate_unscaled_mean_standard_deviation(const Vector<T> &, const Vector<T> &) const;

  void unscale_minimum_maximum(const Vector<T> &, const Vector<T> &);

  void unscale_mean_standard_deviation(const Vector<T> &, const Vector<T> &);

  Vector<T> calculate_reverse_scaling(void) const;

  Vector<T> calculate_scaling_between(const T &, const T &, const T &, const T &) const;

  // Arranging methods

  Matrix<T> to_diagonal_matrix() const;

  Vector<T> get_subvector(const size_t&, const size_t&) const;

  Vector<T> get_subvector(const Vector<size_t> &) const;

  Vector<T> get_subvector(const Vector<bool> &) const;

  Vector<T> get_subvector_random(const size_t&) const;

  Vector<T> get_first(const size_t &) const;

  Vector<T> get_last(const size_t &) const;

  Vector<T> delete_first(const size_t &) const;

  Vector<T> delete_last(const size_t &) const;

  Vector<T> get_integer_elements(const size_t&) const;
  Vector<T> get_integer_elements_missing_values(const Vector<size_t>&, const size_t&) const;

  Matrix<T> get_power_matrix(const size_t&) const;


  // File operations

  void load(const string &);

  void save(const string &, const char& = ' ') const;

  void tuck_in(const size_t &, const Vector<T> &);

  Vector<T> insert_element(const size_t &, const T &) const;
  Vector<T> replace_element(const size_t &, const Vector<T> &) const;

  Vector<T> replace_value(const T&, const T&) const;
  Vector<T> replace_value_if_contains(const T&, const T&) const;

  Vector<string> split_element(const size_t &, const char&) const;

  Vector<T> delete_index(const size_t &) const;

  Vector<T> delete_indices(const Vector<size_t> &) const;

  Vector<T> delete_value(const T &) const;
  Vector<T> delete_values(const Vector<T> &) const;

  Vector<T> assemble(const Vector<T> &) const;
  static Vector<T> assemble(const Vector< Vector<T> > &);

  Vector<T> get_difference(const Vector<T> &) const;
  Vector<T> get_union(const Vector<T> &) const;
  Vector<T> get_intersection(const Vector<T> &) const;

  Vector<T> get_unique_items(const char& separator = ' ') const;
  Vector<T> get_unique_elements() const;
  Vector<T> get_unique_elements_unsorted() const;
  Vector<size_t> get_unique_elements_first_indices() const;
  Vector< Vector<size_t> > get_unique_elements_indices() const;
  Vector<size_t> count_unique() const;

  void print_unique() const;

  Vector<T> calculate_top(const size_t&) const;

  Matrix<T> calculate_top_matrix(const size_t&) const;
  Matrix<T> calculate_top_matrix_over(const size_t&, const size_t&) const;

  void print_top(const size_t&) const;

  vector<T> to_std_vector() const;

  Vector<double> to_double_vector() const;
  Vector<int> to_int_vector() const;
  Vector<size_t> to_size_t_vector() const;
  Vector<time_t> to_time_t_vector() const;
  Vector<bool> to_bool_vector() const;

  Vector<string> to_string_vector() const;

  Vector<double> string_to_double(const double& exception_value = 999) const;
  Vector<int> string_to_int(const int& exception_value = 999) const;
  Vector<size_t> string_to_size_t(const size_t& exception_value = 999) const;
  Vector<time_t> string_to_time_t(const time_t& exception_value = 999) const;

  // Date-time

  Vector<time_t> www_mmm_ddd_yyyy_hh_mm_ss_to_timestamp() const;
  Vector<time_t> yyyy_mm_dd_hh_mm_ss_to_timestamp(const char& = '-') const;
  Vector<time_t> yyyy_mm_to_timestamp(const char& = '/') const;
  Vector<time_t> dd_mm_yyyy_to_timestamp(const char& = '/') const;
  Vector<time_t> yyyy_mm_dd_to_timestamp(const char& = '/') const;
  Matrix<T> dd_mm_yyyy_to_dd_yyyy(const char& = '/') const;
  Matrix<T> yyyy_mm_dd_to_dd_yyyy(const char& = '/') const;
  Matrix<T> mm_yyyy_to_mm_yyyy(const char& = '/') const;

  Vector<T> yyyy_mm_dd_to_weekday(const char& = '/') const;
  Vector<T> yyyy_mm_dd_to_yearday(const char& = '/') const;

  Vector<struct tm> timestamp_to_time_structure() const;
  Vector<size_t> timestamp_to_yearday() const;

  Vector< Vector<T> >split(const size_t&) const;


  Matrix<T> to_row_matrix() const;

  Matrix<T> to_column_matrix() const;

  void parse(const string &);

  string to_text(const char& = ',') const;
  string to_text(const string& = ",") const;

  string vector_to_string(const char&, const char&) const;
  string vector_to_string(const char&) const;
  string vector_to_string() const;

  string stack_vector_to_string() const;

  Vector<string> write_string_vector(const size_t & = 5) const;

  Matrix<T> to_matrix(const size_t &, const size_t &) const;

  double calculate_logistic_function(const Vector<double>&, const Vector<T>&) const;
  Vector<double> calculate_logistic_error_gradient(const Vector<double>&, const Vector<T>&) const;
};


// CONSTRUCTORS

/// Default constructor. It creates a vector of size zero.

template <class T> Vector<T>::Vector() : vector<T>() {}

/// General constructor. It creates a vector of size n, containing n copies of
/// the default value for Type.
/// @param new_size Size of vector.

template <class T>
Vector<T>::Vector(const size_t &new_size)
    : vector<T>(new_size) {}

/// Constant reference initialization constructor.
/// It creates a vector of size n, containing n copies of the type value of
/// Type.
/// @param new_size Size of Vector.
/// @param value Initialization value of Type.

template <class T>
Vector<T>::Vector(const size_t &new_size, const T &value)
    : vector<T>(new_size, value) {}

/// File constructor. It creates a vector object by loading its members from a
/// data file.
/// @param file_name Name of vector data file.

template <class T>
Vector<T>::Vector(const string &file_name)
    : vector<T>() {
  load(file_name);
}

/// Sequential constructor.

template <class T>
Vector<T>::Vector(const T &first, const double &step, const T &last)
    : vector<T>() {
  set(first, step, last);
}

/// Input iterators constructor

template <class T>
template <class InputIterator>
Vector<T>::Vector(InputIterator first, InputIterator last)
    : vector<T>(first, last) {}

/// Copy constructor. It creates a copy of an existing Vector.
/// @param other_vector Vector to be copied.

template <class T>
Vector<T>::Vector(const Vector<T> &other_vector) : vector<T>(other_vector) {}


template <class T>
Vector<T>::Vector(const vector<T> &other_vector) : vector<T>(other_vector) {}


template <class T>
Vector<T>::Vector(const initializer_list<T> &list) : vector<T>(list) {}


template <class T>
Vector<T>::Vector(const Vector< Vector<T> >& vectors)
{
    const size_t vectors_size = vectors.size();

    size_t new_size = 0;

    for(size_t i = 0; i < vectors_size; i++)
    {
        new_size += vectors[i].size();
    }

    set(new_size);

    size_t index = 0;

    for(size_t i = 0; i < vectors_size; i++)
    {
        for(size_t j = 0; j < vectors[i].size(); j++)
        {
            (*this)[index] = vectors[i][j];
            index++;
        }
    }
}


/// Destructor.

template <class T> Vector<T>::~Vector() {
    vector<T>().swap(*this);
}

/// Equal to operator between this vector and a Type value.
/// It produces true if all the elements of this vector are equal to the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator== (const T &value) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] != value) {
      return(false);
    }
  }

  return(true);
}


/// Not equivalent relational operator between this vector and a Type value.
/// It produces true if some element of this vector is not equal to the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator!= (const T &value) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] != value) {
      return(true);
    }
  }

  return(false);
}


/// Greater than relational operator between this vector and a Type value.
/// It produces true if all the elements of this vector are greater than the
/// Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator>(const T &value) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] <= value) {
      return(false);
    }
  }

  return(true);
}


/// Less than relational operator between this vector and a Type value.
/// It produces true if all the elements of this vector are less than the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator<(const T &value) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] >= value) {
      return(false);
    }
  }

  return(true);
}


/// Greater than or equal to than relational operator between this vector and a
/// Type value.
/// It produces true if all the elements of this vector are greater than or
/// equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator>= (const T &value) const
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < value) {
      return(false);
    }
  }

  return(true);
}


/// Less than or equal to than relational operator between this vector and a
/// Type value.
/// It produces true if all the elements of this vector are less than or equal
/// to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator<= (const T &value) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > value) {
      return(false);
    }
  }

  return(true);
}

// METHODS

/// Sets the size of a vector to zero.

template <class T> void Vector<T>::set() { this->resize(0); }


/// Sets a new size to the vector. It does not initialize the data.
/// @param new_size Size for the vector.

template <class T> void Vector<T>::set(const size_t &new_size) {
  this->resize(new_size);
}


/// Sets a new size to the vector and initializes all its elements with a given
/// value.
/// @param new_size Size for the vector.
/// @param new_value Value for all the elements.

template <class T>
void Vector<T>::set(const size_t &new_size, const T &new_value) {
  this->resize(new_size);

  initialize(new_value);
}


/// Sets all the members of a vector object by loading them from a data file.
/// The format is specified in the OpenNN manual.
/// @param file_name Name of vector data file.

template <class T> void Vector<T>::set(const string &file_name) {
  load(file_name);
}


/// Makes this vector to have elements starting from a given value, continuing
/// with a step value and finishing with a given value.
/// Depending on the starting, step and finishing values, this method can produce
/// a variety of sizes and data.
/// @param first Starting value.
/// @param step Step value.
/// @param last Finishing value.

template <class T>
void Vector<T>::set(const T &first, const double &step, const T &last) {
  if(first > last && step > 0) {
    this->resize(0);
  } else if(first < last && step < 0) {
    this->resize(0);
  } else {
    const size_t new_size = 1 + static_cast<size_t>((last - first) / step + 0.5);

    this->resize(new_size);

    for(size_t i = 0; i < new_size; i++) {
     (*this)[i] = first + static_cast<T>(i * step);
    }
  }
}


/// Sets the members of this object with the values of another vector.
/// @param other_vector Object to set this vector.

template <class T> void Vector<T>::set(const Vector &other_vector) {
  *this = other_vector;
}

#ifdef __OPENNN_MPI__
// void set_MPI(const MPI_Datatype) method

/// Send the vector to the other MPI processors.
/// @param mpi_datatype MPI type of this vector.

template <class T> void Vector<T>::set_MPI(const MPI_Datatype mpi_datatype) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int vector_size;

    if(rank == 0)
    {
        vector_size = static_cast<int>(this)->size();
    }

    if(rank > 0)
    {
        MPI_Recv(&vector_size, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        set(vector_size);

        MPI_Recv(data(), vector_size, mpi_datatype, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if(rank < size - 1)
    {
        MPI_Send(&vector_size, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);

        MPI_Send(data(), vector_size, mpi_datatype, rank + 1, 2, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

}
#endif


template <class T>
T Vector<T>::get_first() const
{
    return(*this)[0];
}


template <class T>
T Vector<T>::get_last() const
{
    const size_t this_size = this->size();

    return(*this)[this_size-1];
}


template <class T>
T Vector<T>::get_before_last() const
{
    const size_t this_size = this->size();

    return(*this)[this_size-2];
}


template <class T>
Vector<T> Vector<T>::delete_first(const size_t & elements_number) const
{
    const size_t new_size = this->size() - elements_number;

    return get_last(new_size);
}


template <class T>
Vector<T> Vector<T>::delete_last(const size_t & elements_number) const
{
    const size_t new_size = this->size() - elements_number;

    return get_first(new_size);
}


/// Initializes all the elements of the vector with a given value.
/// @param value Type value.

template <class T>
void Vector<T>::initialize(const T &value)
{
  fill((*this).begin(),(*this).end(), value);
}


template <class T>
void Vector<T>::initialize_first(const size_t& first, const T &value)
{
    for(size_t i = 0; i < first; i++) (*this)[i] = value;
}


/// Initializes all the elements of the vector in a sequential order.

template <class T> void Vector<T>::initialize_sequential() {
  for(size_t i = 0; i < this->size(); i++) {
   (*this)[i] = static_cast<T>(i);
  }
}


/// Assigns a random value comprised between a minimum value and a maximum value
/// to each element in
/// the vector.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

template <class T>
void Vector<T>::randomize_uniform(const double &minimum,
                                  const double &maximum) {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(minimum > maximum) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const double&, const double&) method.\n"
           << "Minimum value must be less or equal than maximum value.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = static_cast<T>(calculate_random_uniform(minimum, maximum));
  }
}


/// Assigns a random value comprised between given minimum and a maximum values
/// to every element in the
/// vector.
/// @param minimums Minimum initialization values.
/// @param maximums Maximum initialization values.

template <class T>
void Vector<T>::randomize_uniform(const Vector<double> &minimums,
                                  const Vector<double> &maximums) {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t minimums_size = minimums.size();
  const size_t maximums_size = maximums.size();

  if(minimums_size != this_size || maximums_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Minimum and maximum sizes must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

  if(minimums > maximums) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Minimum values must be less or equal than their corresponding "
              "maximum values.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = calculate_random_uniform(minimums[i], maximums[i]);
  }
}


/// Assigns random values to each element in the vector.
/// These are taken from a normal distribution with single mean and standard
/// deviation values for all the elements.
/// @param mean Mean value of uniform distribution.
/// @param standard_deviation Standard deviation value of uniform distribution.

template <class T>
void Vector<T>::randomize_normal(const double &mean,
                                 const double &standard_deviation) {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(standard_deviation < 0.0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_normal(const double&, const double&) method.\n"
           << "Standard deviation must be equal or greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = calculate_random_normal(mean, standard_deviation);
  }
}


/// Assigns random values to each element in the vector.
/// These are taken from normal distributions with given means and standard
/// deviations for each element.
/// @param mean Mean values of normal distributions.
/// @param standard_deviation Standard deviation values of normal distributions.

template <class T>
void Vector<T>::randomize_normal(const Vector<double> &mean,
                                 const Vector<double> &standard_deviation) {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t mean_size = mean.size();
  const size_t standard_deviation_size = standard_deviation.size();

  if(mean_size != this_size || standard_deviation_size != this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "void randomize_normal(const Vector<double>&, const "
           "Vector<double>&) method.\n"
        << "Mean and standard deviation sizes must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

  if(standard_deviation < 0.0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_normal(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Standard deviations must be equal or greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = calculate_random_normal(mean[i], standard_deviation[i]);
  }
}


template <class T>
void Vector<T>::randomize_binary(const double& negatives_ratio, const double& positives_ratio)
{
    const size_t this_size = this->size();

    if(this_size == 0)
    {
        return;
    }

    const double total_ratio = negatives_ratio + positives_ratio;

    // Get number of instances for training, selection and testing

    const size_t positives_number = static_cast<size_t>(positives_ratio*this_size/total_ratio);
    const size_t negatives_number = this_size - positives_number;

    Vector<size_t> indices(0, 1, this_size-1);
    random_shuffle(indices.begin(), indices.end());

    size_t i = 0;
    size_t index;

    // Positives

    size_t count_positives = 0;

    while(count_positives != positives_number)
    {
       index = indices[i];

      (*this)[index] = 1;
       count_positives++;

       i++;
    }

    // Positives

    size_t count_negatives = 0;

    while(count_negatives != negatives_number)
    {
       index = indices[i];

      (*this)[index] = 0;
       count_negatives++;

       i++;
    }
}


template <class T>
void Vector<T>::map(Vector<T>& other_vector, const T& this_value, const T& other_value)
{
    const size_t this_size = this->size();

    size_t index = this_size;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] == this_value)
        {
            index = i;
            break;
        }
    }

    if(index != this_size)
    {
        other_vector[index] = other_value;
    }
}


template <class T>
void Vector<T>::map(Vector<T>& other_vector_1, Vector<T>& other_vector_2, const T& this_value, const T& other_value_1, const T& other_value_2)
{
    const size_t this_size = this->size();

    size_t index = this_size;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] == this_value)
        {
            index = i;
            break;
        }
    }

    if(index != this_size)
    {
        other_vector_1[index] = other_value_1;
        other_vector_2[index] = other_value_2;
    }
}


/// Removes whitespaces from the start and the end of each element in this vector of strings.

template <class T>
void Vector<T>::trim()
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        //prefixing spaces

       (*this)[i] = (*this)[i].erase(0,(*this)[i].find_first_not_of(' '));

        //surfixing spaces

       (*this)[i] = (*this)[i].erase((*this)[i].find_last_not_of(' ') + 1);
    }
}


/// Returns a vector of strings that has whitespaces removed from the start and the end of each element.

template <class T>
Vector<T> Vector<T>::trimmed() const
{
    Vector<T> new_vector(*this);
    new_vector.trim();

    return(new_vector);
}


template <class T>
Vector<T> Vector<T>::fill_from(const size_t& from_index, const Vector<T>& fill_with) const
{
    Vector<T> new_vector(*this);

    size_t counter = 0;

    for (size_t i = from_index; i < from_index+fill_with.size(); i++)
    {
        new_vector[i] = fill_with[counter];

        counter++;
    }

    return(new_vector);
}

/// Returns true if the vector contains a certain value, and false otherwise.

template <class T>
bool Vector<T>::contains(const T &value) const {

    Vector<T> copy(*this);

    typename vector<T>::iterator it = find(copy.begin(), copy.end(), value);

    return(it != copy.end());
}


template <class T>
bool Vector<T>::contains_greater_than(const T &value) const {

    for(size_t i = 0; i < this->size(); i++)
    {
        if((*this)[i] > value) return true;
    }

    return false;
}


/// Returns true if the vector contains a certain value from a given set, and
/// false otherwise.

template <class T> bool Vector<T>::contains(const Vector<T> &values) const {
  if(values.empty()) {
    return(false);
  }

  Vector<T> copy(*this);

  const size_t values_size = values.size();

  for(size_t j = 0; j < values_size; j++) {
      typename vector<T>::iterator it = find(copy.begin(), copy.end(), values[j]);

      if(it != copy.end())
      {
          return(true);
      }
  }

  return(false);
}


template <class T>
bool Vector<T>::has_same_elements(const Vector<T>& other_vector) const
{
    const size_t this_size = this->size();

    if(this_size != other_vector.size())
    {
        return false;
    }

    for(size_t i = 0; i < this_size; i++)
    {
        if(!other_vector.contains((*this)[i]))
        {
            return false;
        }
    }

    return true;
}


/// Returns true if the value of all the elements fall in some given range,
/// and false otherwise.
/// @param minimum Minimum value of the range.
/// @param maximum Maximum value of the range.

template <class T>
bool Vector<T>::is_in(const T &minimum, const T &maximum) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < minimum ||(*this)[i] > maximum) {
      return(false);
    }
  }

  return(true);
}


/// Returns true if all the elements have the same value within a defined
/// tolerance ,
/// and false otherwise.
/// @param tolerance Tolerance value, so that if abs(max-min) <= tol, then the
/// vector is considered constant.

template <class T> bool Vector<T>::is_constant(const double &tolerance) const {
  const size_t this_size = this->size();

  if(this_size == 0) {
    return(false);
  }

  const T minimum = calculate_minimum();
  const T maximum = calculate_maximum();

  if(fabs(maximum - minimum) <= tolerance) {
    return(true);
  } else {
    return(false);
  }
}


/// Returns true if all the elements in this vector of strings are equal, and false otherwise.

template <class T>
bool Vector<T>::is_constant_string() const {

  const size_t this_size = this->size();

  if(this_size == 0) {
    return(false);
  }

  for(size_t i = 1; i < this_size; i++)
  {
      if((*this)[i] != (*this)[0])
      {
          return(false);
      }
  }

  return(true);
}


/// Returns true if all the elements in the vector have values which increase
/// with the index, and false otherwise.

template <class T> bool Vector<T>::is_crescent() const
{
  for(size_t i = 0; i < this->size() - 1; i++)
  {
    if((*this)[i] >= (*this)[i + 1]) return(false);
  }

  return(true);
}


/// Returns true if all the elements in the vector have values which decrease
/// with the index, and false otherwise.

template <class T> bool Vector<T>::is_decrescent() const
{
  for(size_t i = 0; i < this->size() - 1; i++)
  {
    if((*this)[i] <= (*this)[i + 1]) return(false);
  }

  return(true);
}


/// Returns true if all the elements of this vector are equal or greater than zero, and false otherwise.

template <class T>
bool Vector<T>::is_positive() const
{
  for(size_t i = 0; i < this->size(); i++)
  {
    if((*this)[i] < 0.0)
    {
      return(false);
    }
  }

  return(true);
}


template <class T>
bool Vector<T>::is_negative() const
{
  for(size_t i = 0; i < this->size(); i++)
  {
    if((*this)[i] > 0.0)
    {
      return(false);
    }
  }

  return(true);
}


template <class T>
bool Vector<T>::check_period(const double& period) const
{
    for(size_t i = 1; i < this->size(); i++)
    {
        if((*this)[i] != (*this)[i-1] + period)
        {
            cout << "i: " << i << endl;
            cout << (*this)[i] << endl;
            cout << (*this)[i-1] << endl;
            cout << "Period: " << (*this)[i] - (*this)[i-1] << endl;

            return false;
        }
    }

    return true;
}


/// Returns true if all the elements in the vector have binary values, and false otherwise.

template <class T> bool Vector<T>::is_binary() const
{
    const size_t this_size = this->size();

    Vector<T> values(1,(*this)[0]);

    for(size_t i = 1; i < this_size; i++)
    {
        const bool contains_value = values.contains((*this)[i]);

        if(!contains_value && values.size() == 1)
        {
            values.push_back((*this)[i]);
        }
        else if(!contains_value)
        {
            return false;
        }
    }

    return true;
}


/// Returns true if all the elements in the vector have binary values, and false otherwise.
/// @param missing_indices Indices of the instances with missing values.

template <class T> bool Vector<T>::is_binary(const Vector<size_t>& missing_indices) const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(!missing_indices.contains(i))
        {
            if((*this)[i] != 0 &&(*this)[i] != 1)
            {
                return false;
            }
        }
    }

    return true;
}


template <class T> bool Vector<T>::is_binary_0_1() const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] != 0 &&(*this)[i] != 1)
        {
            return false;
        }
    }

    return true;
}


/// Returns true if all the elements in the vector are integers, and false otherwise.

template <class T> bool Vector<T>::is_integer() const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(floor((*this)[i]) != (*this)[i])
        {
            return false;
        }
    }

    return true;
}


/// Returns true if all the elements in the vector are integers, and false otherwise.
/// @param missing_indices Indices of the instances with missing values.

template <class T> bool Vector<T>::is_integer(const Vector<size_t>& missing_indices) const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(!missing_indices.contains(i))
        {
            if(floor((*this)[i]) != (*this)[i])
            {
                return false;
            }
        }
    }

    return true;
}


template <class T>
bool Vector<T>::is_discrete(const size_t& maximum) const
{
    Vector<T> values;

    for(size_t i = 0; i < this->size(); i++)
    {
        if(!values.contains((*this)[i]))
        {
            values.push_back((*this)[i]);

            if(values.size() > maximum) return false;
        }



    }


    return true;
}

template <class T>
bool Vector<T>::is_discrete(const Vector<size_t>&, const size_t&) const
{
    return true;
}


/// Returns true if the elements in the vector have a normal distribution with a given critical value.
/// @param critical_value Critical value to be used in the test.

template <class T> bool Vector<T>::perform_Lilliefors_normality_test(const double& critical_value) const
{
#ifndef __Cpp11__
    const size_t n = this->size();

    const double mean = this->calculate_mean();
    const double standard_deviation = this->calculate_standard_deviation();

    Vector<T> sorted_vector(*this);

    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    double Fx;
    double Snx;

    double D = -1;

    for(size_t i = 0; i < n; i++)
    {
        Fx = 0.5 * erfc((mean -(*this)[i])/(standard_deviation*sqrt(2)));

        if((*this)[i] < sorted_vector[0])
        {
            Snx = 0.0;
        }
        else if((*this)[i] >= sorted_vector[n-1])
        {
            Snx = 1.0;
        }
        else
        {
            for(size_t j = 0; j < n-1; j++)
            {
                if((*this)[i] >= sorted_vector[j] &&(*this)[i] < sorted_vector[j+1])
                {
                    Snx = static_cast<double>(j+1)/static_cast<double>(n);
                }
            }
        }

        if(D < abs(Fx - Snx))
        {
            D = abs(Fx - Snx);
        }
    }

    if(D < critical_value)
    {
        return true;
    }
    else
    {
        return false;
    }

#else
    return false;
#endif
}


/// Returns true if the elements in the vector have a normal distribution with a given set of critical values.
/// @param critical_values Critical values to be used in the test.

template <class T> Vector<bool> Vector<T>::perform_Lilliefors_normality_test(const Vector<double>& critical_values) const
{
#ifndef __Cpp11__
    const size_t n = this->size();

    const double mean = this->calculate_mean();
    const double standard_deviation = this->calculate_standard_deviation();

    Vector<T> sorted_vector(*this);

    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    double Fx;
    double Snx;

    double D = -1;

    for(size_t i = 0; i < n; i++)
    {
        Fx = 0.5 * erfc((mean -(*this)[i])/(standard_deviation*sqrt(2)));

        if((*this)[i] < sorted_vector[0])
        {
            Snx = 0.0;
        }
        else if((*this)[i] >= sorted_vector[n-1])
        {
            Snx = 1.0;
        }
        else
        {
            for(size_t j = 0; j < n-1; j++)
            {
                if((*this)[i] >= sorted_vector[j] &&(*this)[i] < sorted_vector[j+1])
                {
                    Snx = static_cast<double>(j+1) / static_cast<double>(n);
                }
            }
        }

        if(D < abs(Fx - Snx))
        {
            D = abs(Fx - Snx);
        }
    }

    Vector<bool> normality_test_results(critical_values.size());

    for(size_t i = 0; i < critical_values.size(); i++)
    {
        if(D < critical_values[i])
        {
            normality_test_results[i] = true;
        }
        else
        {
            normality_test_results[i] = false;
        }
    }

    return normality_test_results;

#else
    return normality_test_results;
#endif
}


/// Calculates the distance between the empirical distribution of the vector and the
/// normal distribution.

template <class T> double Vector<T>::calculate_normal_distribution_distance() const
{
    double normal_distribution_distance = 0.0;

    const size_t n = this->size();

    const double mean = this->calculate_mean();
    const double standard_deviation = this->calculate_standard_deviation();

    double normal_distribution; // Normal distribution
    double empirical_distribution; // Empirical distribution

    Vector<T> sorted_vector(*this);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2.0)));
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        normal_distribution_distance += abs(normal_distribution - empirical_distribution);
    }

    return normal_distribution_distance;
}


/// Calculates the distance between the empirical distribution of the vector and the
/// half normal distribution.

template <class T> double Vector<T>::calculate_half_normal_distribution_distance() const
{
    double half_normal_distribution_distance = 0.0;

    const size_t n = this->size();

    const double standard_deviation = this->calculate_standard_deviation();

    double half_normal_distribution; // Half normal distribution
    double empirical_distribution; // Empirical distribution

    Vector<T> sorted_vector(*this);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2)));
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        half_normal_distribution_distance += abs(half_normal_distribution - empirical_distribution);
    }

    return half_normal_distribution_distance;
}


/// Calculates the distance between the empirical distribution of the vector and the
/// uniform distribution.

template <class T> double Vector<T>::calculate_uniform_distribution_distance() const
{
    double uniform_distribution_distance = 0.0;

    const size_t n = this->size();

    double uniform_distribution; // Uniform distribution
    double empirical_distribution; // Empirical distribution

    Vector<T> sorted_vector(*this);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        uniform_distribution_distance += abs(uniform_distribution - empirical_distribution);
    }

    return uniform_distribution_distance;
}


/// Performs the Lilliefors normality tests varying the confindence level from 0.05 to 0.5.
/// It returns a vector containing the results of the tests.

template <class T> Vector<bool> Vector<T>::perform_normality_analysis() const
{
    const size_t size = this->size();

    double significance_level = 0.05;

    double A_significance_level;
    double B_significance_level;
    Vector<double> critical_values(9);

    for(size_t i = 0; i < 9; i++)
    {
        A_significance_level = 6.32207539843126
                               - 17.1398870006148*(1 - significance_level)
                               + 38.42812675101057*pow((1 - significance_level),2)
                               - 45.93241384693391*pow((1 - significance_level),3)
                               + 7.88697700041829*pow((1 - significance_level),4)
                               + 29.79317711037858*pow((1 - significance_level),5)
                               - 18.48090137098585*pow((1 - significance_level),6);

        B_significance_level = 12.940399038404
                               - 53.458334259532*(1 - significance_level)
                               + 186.923866119699*pow((1 - significance_level),2)
                               - 410.582178349305*pow((1 - significance_level),3)
                               + 517.377862566267*pow((1 - significance_level),4)
                               - 343.581476222384*pow((1 - significance_level),5)
                               + 92.123451358715*pow((1 - significance_level),6);

        critical_values[i] = sqrt(1/(A_significance_level*size+B_significance_level));

        significance_level += 0.05;
    }

    return this->Lilliefors_normality_test(critical_values);
}


/// @todo

template <class T>
double Vector<T>::calculate_normality_parameter() const
{
    const double maximum = this->calculate_maximum();
    const double minimum = this->calculate_minimum();

    const size_t n = this->size();

    const double mean = this->calculate_mean();
    const double standard_deviation = this->calculate_standard_deviation();

    double normal_distribution;
    double empirical_distribution;
    double previous_normal_distribution;
    double previous_empirical_distribution;

    Vector<T> sorted_vector(*this);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    double empirical_area = 0.0;
    double normal_area = 0.0;

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2.0)));
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        if(i == 0)
        {
            previous_normal_distribution = normal_distribution;
            previous_empirical_distribution = empirical_distribution;
        }
        else
        {
            normal_area += 0.5*(sorted_vector[i]-sorted_vector[i-1])*(normal_distribution+previous_normal_distribution);
            empirical_area += 0.5*(sorted_vector[i]-sorted_vector[i-1])*(empirical_distribution+previous_empirical_distribution);

            previous_normal_distribution = normal_distribution;
            previous_empirical_distribution = empirical_distribution;
        }
    }

    const double uniform_area = (maximum-minimum)/2.0;

    return uniform_area;
}


template <class T>
Vector<T> Vector<T>::calculate_variation_percentage() const
{
    const size_t this_size = this->size();

    Vector<T> new_vector(this_size, 0);

    for(size_t i = 1; i < this_size; i++)
    {
        if((*this)[i-1] != 0)
        {
            new_vector[i] = ((*this)[i] - (*this)[i-1])*100/(*this)[i-1];
        }
    }

    return new_vector;
}


/// Calculates the distance between the empirical distribution of the vector and
/// the normal, half-normal and uniform cumulative distribution. It returns 0, 1
/// or 2 if the closest distribution is the normal, half-normal or the uniform,
/// respectively.

template <class T> size_t Vector<T>::perform_distribution_distance_analysis() const
{
    Vector<double> distances(2, 0.0);

    const size_t n = this->size();

    Vector<T> sorted_vector(*this);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    const Statistics<T> statistics = this->calculate_statistics();

    const double mean = statistics.mean;
    const double standard_deviation = statistics.standard_deviation;
    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

#pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(n); i++)
    {
        const double normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2)));
//        const double half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2)));
        const double uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);

        double empirical_distribution;

        size_t counter = 0;

        if((*this)[i] < sorted_vector[0])
        {
            empirical_distribution = 0.0;
        }
        else if((*this)[i] >= sorted_vector[n-1])
        {
            empirical_distribution = 1.0;
        }
        else
        {
            counter = static_cast<size_t>(i + 1);

            for(int j = i+1; j < n; j++)
            {
                if(sorted_vector[j] <= sorted_vector[i])
                {
                    counter++;
                }
                else
                {
                    break;
                }
            }

            empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        }

        #pragma omp critical
        {
            distances[0] += abs(normal_distribution - empirical_distribution);
//            distances[1] += abs(half_normal_distribution - empirical_distribution);
            distances[1] += abs(uniform_distribution - empirical_distribution);
        }
    }

    return distances.calculate_minimal_index();
}


/// Calculates the distance between the empirical distribution of the vector and
/// the normal, half-normal and uniform cumulative distribution. It returns 0, 1
/// or 2 if the closest distribution is the normal, half-normal or the uniform,
/// respectively.

template <class T> size_t Vector<T>::perform_distribution_distance_analysis_missing_values(const Vector<size_t>& missing_indices) const
{
    Vector<double> distances(3, 0.0);

    double normal_distribution; // Normal distribution
    double half_normal_distribution; // Half-normal distribution
    double uniform_distribution; // Uniform distribution
    double empirical_distribution; // Empirical distribution

    Vector<size_t> used_indices(1,1,this->size());
    used_indices = used_indices.get_difference(missing_indices);

    const Vector<T> used_values = this->get_subvector(used_indices);
    const size_t n = used_values.size();

    Vector<T> sorted_vector(used_values);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    Statistics<T> statistics = used_values.calculate_statistics();

    const double mean = statistics.mean;
    const double standard_deviation = statistics.standard_deviation;
    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

    if(fabs(minimum - maximum) < numeric_limits<double>::epsilon() || standard_deviation < 1.0e-09)
    {
        return 2;
    }

    size_t counter = 0;

#pragma omp parallel for private(empirical_distribution, normal_distribution, half_normal_distribution, uniform_distribution, counter)

    for(int i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2)));
        half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2)));
        uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        #pragma omp critical
        {
            distances[0] += abs(normal_distribution - empirical_distribution);
            distances[1] += abs(half_normal_distribution - empirical_distribution);
            distances[2] += abs(uniform_distribution - empirical_distribution);
        }
    }

    return distances.calculate_minimal_index();
}


template <class T>
int Vector<T>::get_lower_index(const size_t& index, const T& value) const
{
    if(index != 0)
    {
        for(int i = static_cast<int>(index)-1; i > -1; i--)
        {
            if((*this)[i] != value)
            {
                return static_cast<int>(i);
            }
        }
    }

    return(-1);
}


template <class T>
int Vector<T>::get_upper_index(const size_t& index, const T& value) const
{
    const size_t this_size = this->size();

    if(index != this_size-1)
    {
        for(int i = static_cast<int>(index)+1; i < static_cast<int>(this_size); i++)
        {
            if((*this)[i] != value)
            {
                return static_cast<int>(i);
            }
        }
    }

    return(-1);
}


template <class T>
Vector<T> Vector<T>::get_reverse() const
{
    const size_t this_size = this->size();

    Vector<T> reverse(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        reverse[i] = (*this)[this_size - 1 - i];
    }

    return reverse;
}


template <class T>
Vector<T> Vector<T>::impute_time_series_missing_values_mean(const T& value) const
{
    const size_t this_size = this->size();

    Vector<T> new_vector(*this);

    int lower_index;
    int upper_index;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] == value)
        {
            lower_index = get_lower_index(i, value);

            upper_index = get_upper_index(i, value);

            if(lower_index != -1 && upper_index != -1)
            {
                new_vector[i] = (new_vector[upper_index] + new_vector[lower_index])/2.0;
            }
            else if(lower_index != -1 && upper_index == -1)
            {
                new_vector[i] = new_vector[lower_index];
            }
            else if(lower_index == -1 && upper_index != -1)
            {
                new_vector[i] = new_vector[upper_index];
            }
            else
            {
                cout << "Error: impute_time_series_missing_values_mean" << endl;
            }
        }
    }

    return(new_vector);
}


/// Replaces a substring by another one in each element of this vector.
/// @param find_what String to be replaced.
/// @param replace_with String to be put instead.

template <class T>
void Vector<T>::replace_substring(const string& find_what, const string& replace_with)
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


/// Returns the number of times that a certain value is contained in the vector.
/// @param value Value to be counted.

template <class T>
size_t Vector<T>::count_equal_to(const T &value) const
{
  return count(this->begin(), this->end(), value);
}


template <class T>
double Vector<T>::count_equal_to(const T &value, const Vector<double>& weights) const
{
    double count = 0.0;

    for (size_t i = 0; i < this->size(); i++)
    {
        if ((*this)[i] == value)
        {
            count += weights[i];
        }
    }

    return count;
}


/// Returns the number of times that certain values are contained in the vector.
/// @param values Vector of values to be counted.

template <class T>
size_t Vector<T>::count_equal_to(const Vector<T> &values) const
{
    Vector<T> this_copy(*this);
    pair<typename vector<T>::iterator,typename vector<T>::iterator> bounds;

    sort(this_copy.begin(), this_copy.end());

    const size_t values_size = values.size();

    size_t count = 0;

#pragma omp parallel for private(bounds) reduction(+ : count)

    for(int i = 0; i < static_cast<int>(values_size); i++)
    {
        bounds = equal_range(this_copy.begin(), this_copy.end(), values[i]);

        count += ((bounds.second - this_copy.begin()) - (bounds.first - this_copy.begin()));
    }

    return(count);
}


/// Returns the number of elemements that are not equal to a certain value.
/// @param value Value.

template <class T>
size_t Vector<T>::count_not_equal_to(const T &value) const
{
    const size_t this_size = this->size();

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] != value)
        {
            count++;
        }
    }

  return(count);
}


/// Returns the number of elemements that are not equal to certain values.
/// @param values Vector of values.

template <class T> size_t Vector<T>::count_not_equal_to(const Vector<T> &values) const
{
    const size_t this_size = this->size();

    const size_t equal_to_count = count_equal_to(values);

    return(this_size - equal_to_count);
}


/// Returns the number of elements that are equal or greater than zero.

template <class T>
size_t Vector<T>::count_positive() const
{
    const size_t this_size = this->size();

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] > 0)
        {
            count++;
        }
    }

  return(count);
}


template <class T>
size_t Vector<T>::count_negative() const
{
    const size_t this_size = this->size();

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] < 0)
        {
            count++;
        }
    }

  return(count);
}


//template <class T>
//Vector<double> Vector<T>::get_binary_vector(const Vector<T>& unique_items) const
//{
//    const size_t unique_items_number = unique_items.size();

//    Vector<double> binary_vector(unique_items_number);

//    for(size_t i = 0; i < unique_items_number; i++)
//    {
//        if(this->contains(unique_items[i]))
//        {
//            binary_vector[i] = 1.0;
//        }
//        else
//        {
//            binary_vector[i] = 0.0;
//        }
//    }

//    return(binary_vector);
//}


//template <class T>
//Matrix<T> Vector<T>::get_binary_matrix(const char& separator) const
//{

//    const size_t this_size = this->size();

//    const Vector<T> unique_mixes = get_unique_elements();

//    Vector< Vector<T> > items(unique_mixes.size());

//    Vector<T> unique_items;

//    for(int i = 0; i < unique_mixes.size(); i++)
//    {
//        items[i] = unique_mixes.split_element(i, separator);

//        unique_items = unique_items.assemble(items[i]).get_unique_elements();
//    }

//    const size_t unique_items_number = unique_items.size();

//    Matrix<T> binary_matrix(this_size, unique_items_number, 0.0);

//    Vector<string> elements;

//    Vector<double> binary_items(unique_items_number);

//    for(size_t i = 0; i < this_size; i++)
//    {
//        elements = split_element(i, separator);

//        binary_items = elements.get_binary_vector(unique_items);

//        binary_matrix.set_row(i, binary_items.to_string_vector());
//    }

//    binary_matrix.set_header(unique_items);

//    return(binary_matrix);
//}


///// Returns a binary matrix indicating the elements of the columns.

//template <class T>
//Matrix<T> Vector<T>::get_unique_binary_matrix(const char& separator, const Vector<T>& unique_items) const
//{
//    const size_t this_size = this->size();

//    const size_t unique_items_number = unique_items.size();

//    Matrix<T> binary_matrix(this_size, unique_items_number,0.0);

//    binary_matrix.set_header(unique_items.to_string_vector());

//    Vector<string> elements;

//    Vector<double> binary_items(unique_items_number);

//    for(size_t i = 0; i < this_size; i++)
//    {
//        elements = split_element(i, separator);

//        binary_items = elements.get_binary_vector(unique_items);

//        binary_matrix.set_row(i, binary_items.to_string_vector());
//    }

//    return(binary_matrix);
//}


template <class T>
Vector<T> Vector<T>::filter_equal_to(const T& value) const
{
    const size_t this_size = this->size();

    const size_t new_size = count_equal_to(value);

    Vector<T> new_vector(new_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] == value)
        {
            new_vector[count] = (*this)[i];
            index++;
        }
    }

    return(new_vector);
}


/// Returns the elements that are different to a given value.
/// @param value Comparison value.

template <class T>
Vector<T> Vector<T>::filter_not_equal_to(const T& value) const
{
    const size_t this_size = this->size();

    const size_t new_size = count_not_equal_to(value);

    Vector<T> new_vector(new_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] != value)
        {
            new_vector[index] = (*this)[i];
            index++;
        }
    }

    return(new_vector);
}


template <class T>
Vector<T> Vector<T>::filter_equal_to(const Vector<T>& values) const
{
    const Vector<size_t> indices = calculate_equal_to_indices(values);

    return get_subvector(indices);
}


template <class T>
Vector<T> Vector<T>::filter_not_equal_to(const Vector<T>& values) const
{
    const Vector<size_t> indices = calculate_not_equal_to_indices(values);

    return get_subvector(indices);
}


/// Returns a vector containing the elements of this vector which are equal or greater than zero.

template <class T>
Vector<T> Vector<T>::get_positive_elements() const
{
    const size_t this_size = this->size();
    const size_t new_size = count_positive();

    Vector<T> new_vector(new_size);

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] >= 0)
        {
            new_vector[count] = (*this)[i];
            count++;
        }
    }

    return(new_vector);
}


/// Returns the number of different integers in the vector or 0 if the number of different
/// integers in the vector is greater than a given numbe of if there are numbers in the
/// vector which are not integers.
/// @param maximum_integers Maximum number of different integers to count.

template <class T> size_t Vector<T>::count_integers(const size_t& maximum_integers) const
{
    if(!this->is_integer())
    {
        return 0;
    }

    const size_t this_size = this->size();

    Vector<T> integers;
    size_t integers_count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!integers.contains((*this)[i]))
        {
            integers.push_back((*this)[i]);
            integers_count++;
        }

        if(integers_count > maximum_integers)
        {
            return 0;
        }
    }

    return integers_count;
}


/// Returns the number of different integers in the vector or 0 if the number of different
/// integers in the vector is greater than a given numbe of if there are numbers in the
/// vector which are not integers.
/// @param missing_indices Indices of the instances with missing values.
/// @param maximum_integers Maximum number of different integers to count.

template <class T> size_t Vector<T>::count_integers_missing_values(const Vector<size_t>& missing_indices, const size_t& maximum_integers) const
{
    if(!this->is_integer(missing_indices))
    {
        return 0;
    }

    const size_t this_size = this->size();

    Vector<T> integers;
    size_t integers_count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!missing_indices.contains(i))
        {
            if(!integers.contains((*this)[i]))
            {
                integers.push_back((*this)[i]);
                integers_count++;
            }

            if(integers_count > maximum_integers)
            {
                return 0;
            }
        }
    }

    return integers_count;
}


template <class T>
Vector<size_t> Vector<T>::calculate_between_indices(const T& minimum, const T& maximum) const
{
    const size_t this_size = this->size();

    const size_t new_size = count_between(minimum, maximum);

    Vector<size_t> indices(new_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] >= minimum &&(*this)[i] <= maximum)
        {
            indices[index] = i;

            index++;
        }
    }

    return indices;
}


/// Returns the vector indices at which the vector elements take some given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::calculate_equal_to_indices(const T &value) const
{
  const size_t this_size = this->size();

  const size_t occurrences_number = count_equal_to(value);

  if(occurrences_number == 0)
  {
      Vector<size_t> occurrence_indices;

      return(occurrence_indices);
  }

  Vector<size_t> occurrence_indices(occurrences_number);

  size_t index = 0;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] == value) {
      occurrence_indices[index] = i;
      index++;
    }
  }

  return(occurrence_indices);
}


/// Returns the indices of the elements that are equal to given values.
/// @param values Vector of values.

template <class T>
Vector<size_t> Vector<T>::calculate_equal_to_indices(const Vector<T>&values) const
{
    const size_t this_size = this->size();

    const size_t occurrences_number = count_equal_to(values);

    if(occurrences_number == 0)
    {
        Vector<size_t> occurrence_indices;

        return(occurrence_indices);
    }

    Vector<size_t> occurrence_indices(occurrences_number);

    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(values.contains((*this)[i]))
        {
            occurrence_indices[index] = i;
            index++;
        }
    }

    return(occurrence_indices);
}


/// Returns the indices of the elements that are not equal to a given value.
/// @param value Element value.

template <class T>
Vector<size_t> Vector<T>::calculate_not_equal_to_indices(const T &value) const {

    const size_t this_size = this->size();

  const size_t not_equal_to_count = count_not_equal_to(value);

  Vector<size_t> not_equal_to_indices(not_equal_to_count);

  size_t index = 0;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] != value) {
      not_equal_to_indices[index] = i;
      index++;
    }
  }

  return(not_equal_to_indices);
}


/// Returns the indices of the elements that are not equal to given values.
/// @param values Vector of values.

template <class T>
Vector<size_t> Vector<T>::calculate_not_equal_to_indices(const Vector<T> &values) const
{
    const size_t this_size = this->size();

    const size_t occurrences_number = count_not_equal_to(values);

    if(occurrences_number == 0) return Vector<size_t>();

    Vector<size_t> occurrence_indices(occurrences_number);

    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!values.contains((*this)[i]))
        {
            occurrence_indices[index] = i;
            index++;
        }
    }

    return(occurrence_indices);
}


template <class T>
Vector<size_t> Vector<T>::get_indices_equal_to(const T &value) const {

    const size_t this_size = this->size();

  const size_t equal_to_count = count_equal_to(value);

  Vector<size_t> equal_to_indices(equal_to_count);

  size_t index = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] == value)
    {
      equal_to_indices[index] = i;
      index++;
    }
  }

  return(equal_to_indices);
}


/// Returns the indices of the elements which are less than a given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::get_indices_less_than(const T &value) const {

    const size_t this_size = this->size();

  const size_t less_than_count = count_less_than(value);

  Vector<size_t> less_than_indices(less_than_count);

  size_t index = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] < value)
    {
      less_than_indices[index] = i;
      index++;
    }
  }

  return(less_than_indices);
}


/// Returns the indices of the elements which are less than a given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::get_indices_greater_than(const T &value) const
{
  const size_t this_size = this->size();

  const size_t count = count_greater_than(value);

  Vector<size_t> indices(count);

  size_t index = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] > value)
    {
      indices[index] = i;
      index++;
    }
  }

  return(indices);
}


/// Returns the number of elements which are greater than some given value.
/// @param value Value.

template <class T> size_t Vector<T>::count_greater_than(const T &value) const
{
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > value) {
      count++;
    }
  }

  return(count);
}


/// Returns the number of elements which are less than some given value.
/// @param value Value.

template <class T> size_t Vector<T>::count_less_than(const T &value) const {
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < value) {
      count++;
    }
  }

  return(count);
}


/// Returns the number of elements which are greater than or equal to some given value.
/// @param value Value.

template <class T> size_t Vector<T>::count_greater_equal_to(const T &value) const {
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] >= value) {
      count++;
    }
  }

  return(count);
}


/// Returns the number of elements which are less or equal than some given value.
/// @param value Value.

template <class T>
size_t Vector<T>::count_less_equal_to(const T &value) const {

    const size_t count = count_if(this->begin(), this->end(), [value](T elem){ return elem <= value; });

    return(count);
}


/// Returns the number of elements which are equal or greater than a minimum value
/// and equal or less than a maximum value.
/// @param minimum Minimum value.
/// @param maximum Maximum value.

template <class T>
size_t Vector<T>::count_between(const T &minimum, const T &maximum) const
{
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] >= minimum &&(*this)[i] <= maximum) count++;
  }

  return(count);
}


/// Returns the number of elements in this timestamp vector which correspond to a given year.
/// @param year Year.

template <class T>
size_t Vector<T>::count_date_occurrences(const size_t& year) const
{
    const size_t this_size = this->size();

    time_t time;

    struct tm* date_info;

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        time = (*this)[i];

        date_info = gmtime(&time);

        if(date_info->tm_year+1900 == year)
        {
            count++;
        }
    }

    return(count);
}


/// Returns a matrix with the date occurrences in this vector of timestamps.
/// Data goes from the first to the last date in this vector.
/// The first column is the day of the month.
/// The second column is the month.
/// The third column is the year.
/// The fourth column is the number of occurrences for that date.

template <class T>
Matrix<T> Vector<T>::count_daily_series_occurrences() const
{
    const time_t start_date = calculate_minimum();
    const time_t end_date = calculate_maximum();

    const size_t day_seconds = 60*60*24;

    const size_t days_number = static_cast<size_t>(difftime(end_date, start_date))/day_seconds;

    Matrix<T> count(days_number, 4, 0);

    time_t date;
    struct tm* date_info;

    size_t day;
    size_t month;
    size_t year;

    for(size_t i = 0; i < days_number; i++)
    {
        date = start_date + i*day_seconds;

        date_info = gmtime(&date);

        day = static_cast<size_t>(date_info->tm_mday);
        month = static_cast<size_t>(date_info->tm_mon+1);
        year = static_cast<size_t>(date_info->tm_year+1900);

        count(i, 0) = day;
        count(i, 1) = month;
        count(i, 2) = year;
    }

    const size_t this_size = this->size();

    size_t row_index;

    for(size_t i = 0; i < this_size; i++)
    {
        date = (*this)[i];

        row_index = static_cast<size_t>(difftime(date, start_date))/day_seconds;

        count(row_index, 3)++;
    }

    return(count);
}


/// Returns a matrix with the weekly occurrences in this vector of timestamps.
/// Data goes from the first to the last date in this vector.
/// The first column is the week of the year.
/// The second column is the year.
/// The third column is the number of occurrences for that week.

template <class T>
Matrix<T> Vector<T>::count_weekly_series_occurrences() const
{
    const time_t start_date = calculate_minimum();
    const time_t end_date = calculate_maximum();

    const size_t week_seconds = 60*60*24*7;

    const size_t weeks_number = static_cast<size_t>(difftime(end_date, start_date))/week_seconds;

    Matrix<T> count(weeks_number, 3, 0);

    time_t date;
    struct tm* date_info;

    size_t year;
    size_t week;

    char buffer[64];

    for(size_t i = 0; i < weeks_number; i++)
    {
        date = start_date + i*week_seconds;

        date_info = gmtime(&date);

        if(strftime(buffer, sizeof buffer, "%W", date_info) != 0)
        {
            week = static_cast<size_t>(atoi(buffer));
        }
        else
        {
            cout << "Unkown week number" << endl;
        }

        year = static_cast<size_t>(date_info->tm_year+1900);

        count(i, 0) = week;
        count(i, 1) = year;
    }

    const size_t this_size = this->size();

    size_t row_index;

    for(size_t i = 0; i < this_size; i++)
    {
        date = (*this)[i];

        row_index = static_cast<size_t>(difftime(date, start_date))/week_seconds;

        count(row_index, 2)++;
    }

    return(count);
}


/// Returns a matrix with the month occurrences in this vector of timestamps.
/// Data goes from the first to the last date in this vector.
/// The first column is the month.
/// The second column is the year.
/// The third column is the number of occurrences for that month.

template <class T>
Matrix<T> Vector<T>::count_monthly_series_occurrences() const
{            
    const time_t start_date = calculate_minimum();
    const time_t end_date = calculate_maximum();

    time_t time;
    struct tm* date_info;

    date_info = gmtime(&start_date);

    const int start_month = date_info->tm_mon+1;
    const int start_year = date_info->tm_year+1900;

    date_info = gmtime(&end_date);

    const int end_month = date_info->tm_mon+1;
    const int end_year = date_info->tm_year+1900;

    const size_t months_number = static_cast<size_t>((end_year-start_year)*12 + end_month - start_month + 1);

    Matrix<T> count(months_number, 4, 0);

    size_t month;
    size_t year;
    size_t division;

    for(size_t i = 0; i < months_number; i++)
    {
        // Month

        month = static_cast<size_t>(start_month) + i;

        division = (month-1)/12;

        if(month > 12)
        {
            month = month -(12 * division);
        }

        count(i, 0) = month;

        // Year

        year = static_cast<size_t>(start_year) + (i + static_cast<size_t>(start_month) - 1)/12;

        count(i, 1) = year;
    }

    const size_t this_size = this->size();

    size_t row_index;

    for(size_t i = 0; i < this_size; i++)
    {
        time = (*this)[i];

        date_info = gmtime(&time);

        month = static_cast<size_t>(date_info->tm_mon+1);
        year = static_cast<size_t>(date_info->tm_year+1900);

        row_index = (year-static_cast<size_t>(start_year))*12 + month - static_cast<size_t>(start_month);

        count(row_index, 2)++;
    }

    for(size_t i = 0; i < months_number; i++)
    {
        count(i, 3) = count(i, 2)*100.0/this_size;
    }

    return(count);
}


/// Returns a matrix with the yearly occurrences in this vector of timestamps.
/// Data goes from the first to the last date in this vector.
/// The first column is the year.
/// The fourth column is the number of occurrences for that year.

template <class T>
Matrix<T> Vector<T>::count_yearly_series_occurrences() const
{
    const time_t start_date = calculate_minimum();
    const time_t end_date = calculate_maximum();

    struct tm* date_info;

    date_info = gmtime(&start_date);

    const int start_year = date_info->tm_year+1900;

    date_info = gmtime(&end_date);

    const int end_year = date_info->tm_year+1900;

    const size_t years_number = static_cast<size_t>(end_year-start_year+1);

    Matrix<T> yearly_orders(years_number, 2);

    for(size_t i = 0; i < years_number; i++)
    {
        const size_t year = static_cast<size_t>(start_year) + i;

        const size_t orders_number = count_date_occurrences(year);

        yearly_orders(i, 0) = year;
        yearly_orders(i, 1) = orders_number;
    }

    return(yearly_orders);
}


/// Returns a matrix with the monthly occurrences in this vector of timestamps, between a stard date and an end date.
/// The first column is the month.
/// The second column is the year.
/// The third column is the number of occurrences for that month.
/// @param start_month Start month.
/// @param start_year Start year.
/// @param end_month End month.
/// @param end_year End year.

template <class T>
Matrix<T> Vector<T>::count_monthly_series_occurrences(const size_t& start_month, const size_t& start_year,
                                                      const size_t& end_month, const size_t& end_year) const
{
    time_t time;

    struct tm* date_info;

    const size_t months_number = (end_year-start_year)*12 + end_month - start_month + 1;

    Matrix<T> count(months_number, 3, 0);

    size_t month;
    size_t year;
    size_t division;

    for(size_t i = 0; i < months_number; i++)
    {
        // Month

        month = start_month + i;

        division = (month-1)/12;

        if(month > 12)
        {
            month = month -(12 * division);
        }

        count(i, 0) = month;

        // Year

        year = start_year + (i+start_month-1)/12;

        count(i, 1) = year;
    }

    const size_t this_size = this->size();

    size_t row_index;

    for(size_t i = 0; i < this_size; i++)
    {
        time = (*this)[i];

        date_info = gmtime(&time);

        month = static_cast<size_t>(date_info->tm_mon+1);
        year = static_cast<size_t>(date_info->tm_year+1900);

        row_index = (year-start_year)*12 + month - start_month;

        count(row_index, 2)++;
    }

    return(count);
}


/// Performs the monthly analysis per year with the input vector.
/// It returns a matrix containing 12 rows and the number of columns equal to years number.

template <class T>
Matrix<T> Vector<T>::count_monthly_occurrences() const
{
    const time_t start_date = calculate_minimum();
    const time_t end_date = calculate_maximum();

    struct tm* date_info;

    date_info = gmtime(&start_date);

    const int start_year = date_info->tm_year+1900;

    date_info = gmtime(&end_date);

    const int end_year = date_info->tm_year+1900;

    const size_t months_number = 12;

    const size_t years_number = static_cast<size_t>(end_year - start_year + 1);

    Matrix<T> count(months_number, years_number+1, 0);

    for(size_t i = 0; i < months_number; i++)
    {
        size_t month = i + 1;

        count(i, 0) = static_cast<double>(month);

        for(size_t j = 0; j < years_number; j++)
        {
            size_t year = static_cast<size_t>(start_year) + j;

            const size_t orders_number = count_date_occurrences(month, year);

            count(i,j+1) = static_cast<double>(orders_number);
        }
    }

    return(count);
}


/// Returns the number of elements in this timestamp vector which correspond to a given month.
/// @param month Month.

template <class T>
size_t Vector<T>::count_month_occurrences(const size_t& month) const
{
    const size_t this_size = this->size();

    time_t time;

    struct tm* date_info;

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        time = (*this)[i];

        date_info = gmtime(&time);

        if(date_info->tm_mon+1 == month)
        {
            count++;
        }
    }

    return(count);
}


/// Returns the number of elements in this timestamp vector which correspond to a given date.
/// @param month Month.
/// @param year Year.

template <class T>
size_t Vector<T>::count_date_occurrences(const size_t& month, const size_t& year) const
{
    const size_t this_size = this->size();

    time_t time;

    struct tm* date_info;

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        time = (*this)[i];

        date_info = gmtime(&time);

        if(date_info->tm_mon+1 == month && date_info->tm_year+1900 == year)
        {
            count++;
        }
    }

    return(count);
}


/// Returns the number of elements in this string vector which contains a given substring.
/// @param find_what Substring to be found.

template <class T>
size_t Vector<T>::count_contains(const string& find_what) const
{
    const size_t this_size = this->size();

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i].find(find_what) != string::npos)
        {
             count++;
        }
    }

    return(count);
}


/// Appends a different string onto the end of each string in this vector.
/// @param other_vector Vector of strings to be appended.
/// @param separator Delimiter char between both strings.

template <class T>
Vector<T> Vector<T>::merge(const Vector<T>& other_vector, const char& separator) const
{
    const size_t this_size = this->size();

    Vector<T> merged(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        merged[i] = (*this)[i] + separator + other_vector[i];
    }

    return(merged);
}


/// Returns the elements that are equal or greater than a minimum value
/// and less or equal to a maximum value.
/// @param minimum Minimum value.
/// @param maximum Maximum value.

template <class T>
Vector<T> Vector<T>::filter_minimum_maximum(const T &minimum, const T &maximum) const
{
    const size_t this_size = this->size();
    const size_t new_size = count_between(minimum, maximum);

    Vector<T> new_vector(new_size);

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] >= minimum && (*this)[i] <= maximum) {
      new_vector[count] = (*this)[i];
      count++;
    }
  }

  return(new_vector);
}


/// Returns the vector indices of the elemnts that contains the given value.
/// @param find_what String to find in the vector.

template <class T>
Vector<size_t> Vector<T>::calculate_contains_indices(const string& find_what) const
{
    const size_t this_size = this->size();

    Vector<size_t> indices;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i].find(find_what) != string::npos)
        {
             indices.push_back(i);
        }
    }

    return(indices);
}


/// Returns the vector indices at which the vector elements are less than some
/// given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::calculate_less_than_indices(const T &value) const {
  const size_t this_size = this->size();

  Vector<size_t> less_than_indices;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < value) {
      less_than_indices.push_back(i);
    }
  }

  return(less_than_indices);
}


/// Returns the vector indices at which the vector elements are greater than
/// some given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::calculate_greater_than_indices(const T &value) const {

  const size_t this_size = this->size();

  Vector<size_t> greater_than_indices;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > value) {
      greater_than_indices.push_back(i);
    }
  }

  return(greater_than_indices);
}


/// Returns the indices of the elements which are less or equal to a given value.
/// @param Value Comparison value.

template <class T>
Vector<size_t> Vector<T>::calculate_less_equal_to_indices(const T &value) const
{
  const size_t this_size = this->size();

  Vector<size_t> less_than_indices;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] <= value) {
      less_than_indices.push_back(i);
    }
  }

  return(less_than_indices);
}


template <class T>
Vector<size_t> Vector<T>::calculate_greater_equal_to_indices(const T &value) const {

  const size_t this_size = this->size();

  Vector<size_t> greater_than_indices;

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] >= value) {
      greater_than_indices.push_back(i);
    }
  }

  return(greater_than_indices);
}


/// Returns a vector containing the sum of the frequencies of the bins to which
/// this vector belongs.
/// @param histograms Used histograms.

template <class T>
Vector<size_t> Vector<T>::calculate_total_frequencies(
    const Vector< Histogram<T> > &histograms) const {
  const size_t histograms_number = histograms.size();

  Vector<size_t> total_frequencies(histograms_number);

  for(size_t i = 0; i < histograms_number; i++) {
    total_frequencies[i] = histograms[i].calculate_frequency((*this)[i]);
  }

  return(total_frequencies);
}


/// Returns a vector containing the sum of the frequencies of the bins to which
/// this vector
/// blongs.
/// @param instance_missing_values Missing values
/// @param histograms Used histograms

template <class T>
Vector<size_t> Vector<T>::calculate_total_frequencies_missing_values(
    const Vector<size_t>& instance_missing_values,
    const Vector< Histogram<T> >& histograms) const {
  const size_t histograms_number = histograms.size();

  Vector<size_t> total_frequencies;

  for(size_t i = 0; i < histograms_number; i++) {
    if(!(instance_missing_values.contains(i))) {
      total_frequencies[i] = histograms[i].calculate_frequency((*this)[i]);
    } else {
      total_frequencies[i] = 0;
    }
  }

  return(total_frequencies);
}


/// Returns vector with the Box-Cox transformation.
/// @param lambda Exponent of the Box-Cox transformation.

template <class T> Vector<double> Vector<T>::perform_Box_Cox_transformation(const double& lambda) const
{
    const size_t size = this->size();

    Vector<double> vector_tranformation(size);

    for(size_t i = 0; i < size; i++)
    {
        if(fabs(lambda - 0) < numeric_limits<double>::epsilon())
        {
            vector_tranformation[i] = log(static_cast<double>((*this)[i]));
        }
        else
        {
            vector_tranformation[i] = (pow(static_cast<double>((*this)[i]), lambda) - 1)/lambda;
        }
    }

    return vector_tranformation;
}


/// Returns a vector containing the relative frequencies of the elements.
/// @param total_sum Sum of the elements of the vector

template <class T> Vector<double> Vector<T>::calculate_percentage(const size_t& total_sum) const
{
    const size_t this_size = this->size();

    Vector<double> percentage_vector(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        percentage_vector[i] = static_cast<double>((*this)[i])*100.0/static_cast<double>(total_sum);
    }

    return percentage_vector;
}


template <class T>
double Vector<T>::calculate_error(const Vector<T>& other_vector) const
{
    const size_t this_size = this->size();

    Vector<double> error(this_size,0);

    for(size_t i = 0; i < this_size; i++)
    {
        if(other_vector[i] != 0)
        {
            error[i] = static_cast<double>(abs((*this)[i] - other_vector[i]));
        }
    }

    error = error.filter_not_equal_to(0);

    double error_mean = error.calculate_mean();

    return error_mean;
}


/// Returns the smallest element in the vector.

template <class T> T Vector<T>::calculate_minimum() const {

  Vector<T> copy(*this);

  typename vector<T>::iterator result = min_element(copy.begin(), copy.end());

  return(*result);
}


/// Returns the largest element in the vector.

template <class T> T Vector<T>::calculate_maximum() const {

    Vector<T> copy(*this);

    typename vector<T>::iterator result = max_element(copy.begin(), copy.end());

    return(*result);
}


/// Returns a vector containing the smallest and the largest elements in the
/// vector.

template <class T> Vector<T> Vector<T>::calculate_minimum_maximum() const {

    Vector<T> copy(*this);

    typename vector<T>::iterator minimum = min_element(copy.begin(), copy.end());
    typename vector<T>::iterator maximum = max_element(copy.begin(), copy.end());

    return {*minimum, *maximum};
}


/// Returns the smallest element in the vector.

template <class T>
T Vector<T>::calculate_minimum_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

  T minimum = numeric_limits<T>::max();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < minimum &&
        !missing_indices.contains(i))
    {
      minimum = (*this)[i];
    }
  }

  return(minimum);
}


/// Returns the largest element in the vector.

template <class T>
T Vector<T>::calculate_maximum_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

  T maximum;

  if(numeric_limits<T>::is_signed) {
    maximum = -numeric_limits<T>::max();
  } else {
    maximum = 0;
  }

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > maximum && !missing_indices.contains(i)) {
      maximum = (*this)[i];
    }
  }

  return(maximum);
}


/// Returns a vector containing the smallest and the largest elements in the
/// vector.

template <class T>
Vector<T> Vector<T>::calculate_minimum_maximum_missing_values(
    const Vector<size_t> &missing_indices) const {
  size_t this_size = this->size();

  T minimum = numeric_limits<T>::max();

  T maximum;

  if(numeric_limits<T>::is_signed) {
    maximum = -numeric_limits<T>::max();
  } else {
    maximum = 0;
  }

  for(size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      if((*this)[i] < minimum) {
        minimum = (*this)[i];
      }

      if((*this)[i] > maximum) {
        maximum = (*this)[i];
      }
    }
  }

  return {minimum, maximum};

}


/// Calculates the explained variance for a given vector(principal components analysis).
/// This method returns a vector whose size is the same as the size of the given vector.

template<class T>
Vector<T> Vector<T>::calculate_explained_variance() const
{
    const size_t this_size = this->size();

    #ifdef __OPENNN_DEBUG__

      if(this_size == 0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance() const method.\n"
               << "Size of the vector must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    const double this_sum = this->calculate_absolute_value().calculate_sum();

    #ifdef __OPENNN_DEBUG__

      if(this_sum == 0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance() const method.\n"
               << "Sum of the members of the vector must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    #ifdef __OPENNN_DEBUG__

      if(this_sum < 0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance() const method.\n"
               << "Sum of the members of the vector cannot be negative.\n";

        throw logic_error(buffer.str());
      }

    #endif


    Vector<double> explained_variance(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        explained_variance[i] = ((*this)[i]/this_sum)*100.0;

        if(explained_variance[i] - 0.0 < 1.0e-16)
        {
            explained_variance[i] = 0.0;
        }
    }

    #ifdef __OPENNN_DEBUG__

      if(explained_variance.calculate_sum() != 1.0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance() const method.\n"
               << "Sum of explained variance must be 1.\n";

        throw logic_error(buffer.str());
      }

    #endif

    return explained_variance;
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

template <class T>
Histogram<T> Vector<T>::calculate_histogram(const size_t &bins_number) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Histogram<T> calculate_histogram(const size_t&) const method.\n"
           << "Number of bins is less than one.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> minimums(bins_number);
  Vector<T> maximums(bins_number);

  Vector<T> centers(bins_number);
  Vector<size_t> frequencies(bins_number, 0);

  const Vector<T> minimum_maximum = calculate_minimum_maximum();

  const T minimum = minimum_maximum[0];
  const T maximum = minimum_maximum[1];

  const double length = (maximum - minimum) /static_cast<double>(bins_number);

  minimums[0] = minimum;
  maximums[0] = minimum + length;
  centers[0] = (maximums[0] + minimums[0]) / 2.0;

  // Calculate bins center

  for(size_t i = 1; i < bins_number; i++)
  {
    minimums[i] = minimums[i - 1] + length;
    maximums[i] = maximums[i - 1] + length;

    centers[i] = (maximums[i] + minimums[i]) / 2.0;
  }

  // Calculate bins frequency

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    for(size_t j = 0; j < bins_number - 1; j++) {
      if((*this)[i] >= minimums[j] &&(*this)[i] < maximums[j]) {
        frequencies[j]++;
      }
    }

    if((*this)[i] >= minimums[bins_number - 1]) {
      frequencies[bins_number - 1]++;
    }
  }

  Histogram<T> histogram(bins_number);
  histogram.centers = centers;
  histogram.minimums = minimums;
  histogram.maximums = maximums;
  histogram.frequencies = frequencies;

  return(histogram);
}


template<class T>
Histogram<T> Vector<T>::calculate_histogram_centered(const double& center, const size_t & bins_number) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

      if(bins_number < 1) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Histogram<T> calculate_histogram(const size_t&) const method.\n"
               << "Number of bins is less than one.\n";

        throw logic_error(buffer.str());
      }

    #endif

      size_t bin_center;

      if(bins_number%2 == 0)
      {
          bin_center = static_cast<size_t>(static_cast<double>(bins_number)/2.0);
      }
      else
      {
          bin_center = static_cast<size_t>(static_cast<double>(bins_number)/2.0+1.0/2.0);
      }

      Vector<T> minimums(bins_number);
      Vector<T> maximums(bins_number);

      Vector<T> centers(bins_number);
      Vector<size_t> frequencies(bins_number, 0);

      const Vector<T> minimum_maximum = calculate_minimum_maximum();

      const T minimum = minimum_maximum[0];
      const T maximum = minimum_maximum[1];

      const double length = (maximum - minimum) /static_cast<double>(bins_number);

//      minimums[0] = minimum;
//      maximums[0] = minimum + length;
//      centers[0] = (maximums[0] + minimums[0]) / 2.0;

      minimums[bin_center-1] = center - length;
      maximums[bin_center-1] = center + length;
      centers[bin_center-1] = center;

      // Calculate bins center

      for(int i = bin_center; i < bins_number; i++) // Upper centers
      {
        minimums[i] = minimums[i - 1] + length;
        maximums[i] = maximums[i - 1] + length;

        centers[i] = (maximums[i] + minimums[i]) / 2.0;
      }

      for(int i = bin_center-2; i >= 0; i--) // Lower centers
      {
        minimums[i] = minimums[i + 1] - length;
        maximums[i] = maximums[i + 1] - length;

        centers[i] = (maximums[i] + minimums[i]) / 2.0;
      }

      // Calculate bins frequency

      const size_t this_size = this->size();

      for(size_t i = 0; i < this_size; i++) {
        for(size_t j = 0; j < bins_number - 1; j++) {
          if((*this)[i] >= minimums[j] &&(*this)[i] < maximums[j]) {
            frequencies[j]++;
          }
        }

        if((*this)[i] >= minimums[bins_number - 1]) {
          frequencies[bins_number - 1]++;
        }
      }

      Histogram<T> histogram(bins_number);
      histogram.centers = centers;
      histogram.minimums = minimums;
      histogram.maximums = maximums;
      histogram.frequencies = frequencies;

      return(histogram);
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

template <class T>
Histogram<T> Vector<T>::calculate_histogram_binary() const {
// Control sentence(if debug)

  Vector<T> minimums(2);
  Vector<T> maximums(2);

  Vector<T> centers(2);
  Vector<size_t> frequencies(2, 0);

  minimums[0] = 0.0;
  maximums[0] = 0.0;
  centers[0] = 0.0;

  minimums[1] = 1.0;
  maximums[1] = 1.0;
  centers[1] = 1.0;

  // Calculate bins frequency

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    for(size_t j = 0; j < 2; j++) {
      if((*this)[i] == minimums[j]) {
        frequencies[j]++;
      }
    }
  }

  Histogram<T> histogram(2);
  histogram.centers = centers;
  histogram.minimums = minimums;
  histogram.maximums = maximums;
  histogram.frequencies = frequencies;

  return(histogram);
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

template <class T>
Histogram<T> Vector<T>::calculate_histogram_integers(const size_t& bins_number) const {
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

      if(bins_number < 1) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Histogram<T> calculate_histogram(const size_t&) const method.\n"
               << "Number of bins is less than one.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Vector<T> centers = this->get_integer_elements(bins_number);
    const size_t centers_number = centers.size();

    sort(centers.begin(), centers.end(), less<T>());

    Vector<T> minimums(centers_number);
    Vector<T> maximums(centers_number);
    Vector<size_t> frequencies(centers_number);

    for(size_t i = 0; i < centers_number; i++)
    {
        minimums[i] = centers[i];
        maximums[i] = centers[i];
        frequencies[i] = this->count_equal_to(centers[i]);
    }

    Histogram<T> histogram(centers_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}


template <class T>
Histogram<T> Vector<T>::calculate_histogram_doubles() const {
    // Control sentence(if debug)


    Vector<T> centers = this->get_unique_elements();
    const size_t centers_number = centers.size();

    sort(centers.begin(), centers.end(), less<T>());

    Vector<T> minimums(centers_number);
    Vector<T> maximums(centers_number);
    Vector<size_t> frequencies(centers_number);

    for(size_t i = 0; i < centers_number; i++)
    {
        minimums[i] = centers[i]*1.01;
        maximums[i] = centers[i]*1.01;

        frequencies[i] = this->count_equal_to(centers[i]);
    }

    Histogram<T> histogram(centers_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

template <class T>
Histogram<T> Vector<T>::calculate_histogram_missing_values(
    const Vector<size_t> &missing_indices, const size_t &bins_number) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Histogram<T> calculate_histogram_missing_values(const "
              "Vector<size_t>&, const size_t&) const method.\n"
           << "Number of bins is less than one.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> minimums(bins_number);
  Vector<T> maximums(bins_number);

  Vector<T> centers(bins_number);
  Vector<size_t> frequencies(bins_number, 0);

  const Vector<T> minimum_maximum =
      calculate_minimum_maximum_missing_values(missing_indices);

  const T minimum = minimum_maximum[0];
  const T maximum = minimum_maximum[1];

  const double length = (maximum - minimum) /static_cast<double>(bins_number);

  minimums[0] = minimum;
  maximums[0] = minimum + length;
  centers[0] = (maximums[0] + minimums[0]) / 2.0;

  // Calculate bins center

  for(size_t i = 1; i < bins_number; i++) {
    minimums[i] = minimums[i - 1] + length;
    maximums[i] = maximums[i - 1] + length;

    centers[i] = (maximums[i] + minimums[i]) / 2.0;
  }

  // Calculate bins frequency

  const size_t this_size = this->size();

  for(int i = 0; i < static_cast<int>(this_size); i++) {
    if(!missing_indices.contains(static_cast<size_t>(i))) {
      for(int j = 0; j < static_cast<int>(bins_number) - 1; j++) {
        if((*this)[i] >= minimums[j] &&(*this)[i] < maximums[j]) {
          frequencies[static_cast<size_t>(j)]++;
        }
      }

      if((*this)[i] >= minimums[bins_number - 1]) {
        frequencies[bins_number - 1]++;
      }
    }
  }

  Histogram<T> histogram(bins_number);
  histogram.centers = centers;
  histogram.minimums = minimums;
  histogram.maximums = maximums;
  histogram.frequencies = frequencies;

  return(histogram);
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param missing_indices Indices of the instances with missing values.

template <class T>
Histogram<T> Vector<T>::calculate_histogram_binary_missing_values(const Vector<size_t>& missing_indices) const {
// Control sentence(if debug)

  Vector<T> minimums(2);
  Vector<T> maximums(2);

  Vector<T> centers(2);
  Vector<size_t> frequencies(2, 0);

  minimums[0] = 0.0;
  maximums[0] = 0.0;
  centers[0] = 0.0;

  minimums[1] = 1.0;
  maximums[1] = 1.0;
  centers[1] = 1.0;

  // Calculate bins frequency

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
    for(size_t j = 0; j < 2; j++) {
      if((*this)[i] == minimums[j]) {
        frequencies[j]++;
      }
    }
    }
  }

  Histogram<T> histogram(2);
  histogram.centers = centers;
  histogram.minimums = minimums;
  histogram.maximums = maximums;
  histogram.frequencies = frequencies;

  return(histogram);
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param missing_indices Indices of the instances with missing values.
/// @param bins_number Number of bins of the histogram.

template <class T>
Histogram<T> Vector<T>::calculate_histogram_integers_missing_values(const Vector<size_t>& missing_indices, const size_t& bins_number) const {
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

      if(bins_number < 1) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Histogram<T> calculate_histogram(const size_t&) const method.\n"
               << "Number of bins is less than one.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Vector<T> centers = this->get_integer_elements_missing_values(missing_indices, bins_number);
    const size_t centers_number = centers.size();

    sort(centers.begin(), centers.end(), less<T>());

    Vector<T> minimums(centers_number);
    Vector<T> maximums(centers_number);
    Vector<size_t> frequencies(centers_number);

    for(size_t i = 0; i < centers_number; i++)
    {
        minimums[i] = centers[i];
        maximums[i] = centers[i];
        frequencies[i] = this->count_equal_to(centers[i]);
    }

    Histogram<T> histogram(centers_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}


/// Finds the first element in the vector with a given value, and returns its index.
/// @param value Value to be found.

template <class T>
size_t Vector<T>::get_first_index(const T& value) const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] == value)
        {
            return(i);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t get_first_index(const T&) const.\n"
           << "Value not found in vector.\n";

    throw logic_error(buffer.str());
}


/// Returns the index of the smallest element in the vector.

template <class T> size_t Vector<T>::calculate_minimal_index() const
{
    Vector<T> copy(*this);

    typename vector<T>::iterator result = min_element(copy.begin(), copy.end());

    return(distance(copy.begin(), result));
}


/// Returns the index of the largest element in the vector.

template <class T> size_t Vector<T>::calculate_maximal_index() const {
    Vector<T> copy(*this);

    typename vector<T>::iterator result = max_element(copy.begin(), copy.end());

    return( distance(copy.begin(), result));
}


///// Returns the indices of the smallest elements in the vector.
///// @param number Number of minimal indices to be computed.

template <class T>
Vector<size_t>
Vector<T>::calculate_minimal_indices(const size_t &number) const {
  const size_t this_size = this->size();

  const Vector<size_t> rank = calculate_less_rank();

  Vector<size_t> minimal_indices(number);

//  #pragma omp parallel for

  for(int i = 0; i < static_cast<int>(this_size); i++)
  {
    for(size_t j = 0; j < number; j++)
    {
      if(rank[static_cast<size_t>(i)] == j)
      {
        minimal_indices[j] = static_cast<size_t>(i);
      }
    }
  }

  return(minimal_indices);
}


///// Returns the indices of the smallest elements in the vector.
///// Used for big vectors.
///// @param number Number of minimal indices to be computed.
template <class T>
Vector<size_t>
Vector<T>::calculate_k_minimal_indices(const size_t &number) const{
    const size_t this_size = this->size();

    Vector<size_t> minimal_indices(number);

    Vector<double> minimal_values(number);

    #pragma omp parallel for

    for(int i = 0; i < this_size; i++)
    {
        double current_value = this->data()[i];
        size_t max_value_index = minimal_values.calculate_maximal_index();

        if(i<number)
        {
            minimal_indices[static_cast<size_t>(i)] = static_cast<size_t>(i);
            minimal_values[static_cast<size_t>(i)] = current_value;
        }
        else
        {
            for(int j=0; j<number; j++)
            {
                if(current_value < minimal_values[static_cast<size_t>(j)])
                {
                    minimal_indices[max_value_index] = static_cast<size_t>(i);
                    minimal_values[max_value_index] = current_value;
                    break;
                }
            }
        }
    }

    const Vector<size_t> sorted_indices = minimal_values.calculate_minimal_indices(number);
    Vector<size_t> sorted_minimal_indices(number);

    for(int i = 0; i <number; i++)
    {
        sorted_minimal_indices[static_cast<size_t>(i)] = minimal_indices[sorted_indices[static_cast<size_t>(i)]];
    }

    return(sorted_minimal_indices);
}


/// Returns the indices of the largest elements in the vector.
/// @param number Number of maximal indices to be computed.

template <class T>
Vector<size_t>
Vector<T>::calculate_maximal_indices(const size_t &number) const {
  const size_t this_size = this->size();

  const Vector<size_t> rank = calculate_greater_rank();

  Vector<size_t> maximal_indices(number);

  for(size_t i = 0; i < this_size; i++) {
    for(size_t j = 0; j < number; j++) {
      if(rank[i] == j) {
        maximal_indices[j] = i;
      }
    }
  }

  return(maximal_indices);
}


/// Returns a vector with the indices of the smallest and the largest elements
/// in the vector.

template <class T>
Vector<size_t> Vector<T>::calculate_minimal_maximal_index() const {
  const size_t this_size = this->size();

  T minimum = (*this)[0];
  T maximum = (*this)[0];

  size_t minimal_index = 0;
  size_t maximal_index = 0;

  for(size_t i = 1; i < this_size; i++) {
    if((*this)[i] < minimum) {
      minimum = (*this)[i];
      minimal_index = i;
    }
    if((*this)[i] > maximum) {
      maximum = (*this)[i];
      maximal_index = i;
    }
  }

  Vector<size_t> minimal_maximal_index(2);
  minimal_maximal_index[0] = minimal_index;
  minimal_maximal_index[1] = maximal_index;

  return(minimal_maximal_index);
}


/// Returns a vector with the elements of this vector raised to a power
/// exponent.
/// @param exponent Pow exponent.

template <class T> Vector<T> Vector<T>::calculate_pow(const T &exponent) const {
  const size_t this_size = this->size();

  Vector<T> power(this_size);

  for(size_t i = 0; i < this_size; i++) {
    power[i] = pow((*this)[i], exponent);
  }

  return(power);
}


/// Returns the competitive vector of this vector,
/// whose elements are one the bigest element of this vector, and zero for the
/// other elements.

template <class T>
Vector<T> Vector<T>::calculate_competitive() const {
  const size_t this_size = this->size();

  Vector<T> competitive(this_size, 0);

  const size_t maximal_index = calculate_maximal_index();

  competitive[maximal_index] = 1;

  return(competitive);
}


/// Returns the softmax vector of this vector,
/// whose elements sum one, and can be interpreted as probabilities.

template <class T> Vector<T> Vector<T>::calculate_softmax() const {
  const size_t this_size = this->size();

  Vector<T> softmax(this_size);

  T sum = 0;

  for(size_t i = 0; i < this_size; i++) {
    sum += exp((*this)[i]);
  }

  for(size_t i = 0; i < this_size; i++) {
    softmax[i] = exp((*this)[i]) / sum;
  }

  return(softmax);
}


/// Returns the softmax Jacobian of this vector.

template <class T> Matrix<T> Vector<T>::calculate_softmax_Jacobian() const {
  const size_t this_size = this->size();

  Matrix<T> softmax_Jacobian(this_size, this_size);

  for(size_t i = 0; i < this_size; i++) {
    for(size_t j = 0; j < this_size; j++) {
      if(i == j) {
        softmax_Jacobian(i, i) = (*this)[i] *(1.0 -(*this)[i]);
      } else {
        softmax_Jacobian(i, i) = (*this)[i] *(*this)[j];
      }
    }
  }

  return(softmax_Jacobian);
}


/// This method converts the values of the vector to be binary.
/// The threshold value used is 0.5.

template <class T> Vector<bool> Vector<T>::calculate_binary() const {
  const size_t this_size = this->size();

  Vector<bool> binary(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < 0.5) {
      binary[i] = false;
    } else {
      binary[i] = true;
    }
  }

  return(binary);
}


/// This method calculates the square root of each element in the vector.

template <class T> Vector<T> Vector<T>::calculate_square_root_elements() const {
  const size_t this_size = this->size();

  Vector<T> square(this_size);

  for(size_t i = 0; i < this_size; i++) {

      square[i]=sqrt((*this)[i]);

  }

  return(square);
}


/// Return the cumulative vector of this vector,
/// where each element is summed up with all the previous ones.

template <class T> Vector<T> Vector<T>::calculate_cumulative() const {
  const size_t this_size = this->size();

  Vector<T> cumulative(this_size);

  if(this_size > 0) {
    cumulative[0] = (*this)[0];

    for(size_t i = 1; i < this_size; i++) {
      cumulative[i] = cumulative[i - 1] + (*this)[i];
    }
  }

  return(cumulative);
}


/// This method applies only to cumulative vectors.
/// It returns the index of the first element which is greater than a given
/// value.
/// @param value Value.

template <class T>
size_t Vector<T>::calculate_cumulative_index(const T &value) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t calculate_cumulative_index(const T&) const.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

  T cumulative_value = (*this)[this_size - 1];

  if(value > cumulative_value) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t calculate_cumulative_index(const T&) const.\n"
           << "Value(" << value << ") must be less than cumulative value("
           << cumulative_value << ").\n";

    throw logic_error(buffer.str());
  }

  for(size_t i = 1; i < this_size; i++) {
    if((*this)[i] <(*this)[i - 1]) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "int calculate_cumulative_index(const T&) const.\n"
             << "Vector elements must be crescent.\n";

      throw logic_error(buffer.str());
    }
  }

#endif

  if(value <= (*this)[0]) {
    return(0);
  }

  for(size_t i = 1; i < this_size; i++) {
    if(value >(*this)[i - 1] && value <= (*this)[i]) {
      return(i);
    }
  }

  return(this_size - 1);
}


/// Returns the index of the closest element in the vector to a given value.

template <class T>
size_t Vector<T>::calculate_closest_index(const T &value) const {
  const Vector<T> difference = (*this - value).calculate_absolute_value();

  const size_t closest_index = difference.calculate_minimal_index();

  return(closest_index);
}


/// Returns the sum of the elements in the vector.

template <class T> T Vector<T>::calculate_sum() const {
  const size_t this_size = this->size();

  T sum = 0;

  for(size_t i = 0; i < this_size; i++) {
    sum += (*this)[i];
  }

  return(sum);
}


/// Returns the sum of the elements with the given indices.
/// @param indices Indices of the elementes to sum.

template <class T>
T Vector<T>::calculate_partial_sum(const Vector<size_t> &indices) const {
  const size_t this_size = this->size();

  T sum = 0;

  for(size_t i = 0; i < this_size; i++) {
    if(indices.contains(i)) {
      sum += (*this)[i];
    }
  }

  return(sum);
}


/// Returns the sum of the elements in the vector.

template <class T>
T Vector<T>::calculate_sum_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

  T sum = 0;

  for(size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      sum += (*this)[i];
    }
  }

  return(sum);
}


/// Returns the product of the elements in the vector.

template <class T> T Vector<T>::calculate_product() const {
  const size_t this_size = this->size();

  T product = 1;

  for(size_t i = 0; i < this_size; i++) {
    product *= (*this)[i];
  }

  return(product);
}


template <class T>
Vector<T> Vector<T>::calculate_moving_average_cyclic(const T& parameter) const
{
    const size_t this_size = this->size();

    Vector<T> moving_average(this_size);

    moving_average[0] = ((*this)[0]+ (parameter*((*this)[1]+ (*this)[this_size-1])))/(1+ (2*parameter));

    moving_average[this_size-1] = ((*this)[this_size-1]+ (parameter*((*this)[this_size-2]+ (*this)[0])))/(1+ (2*parameter));

    for(size_t i = 1; i < this->size()-1; i++)
    {
        moving_average[i] = ((*this)[i]+ (parameter*((*this)[i-1]+ (*this)[i+1])))/(1.0+2.0*parameter);
    }

    return(moving_average);
}


template <class T>
Vector<T> Vector<T>::calculate_moving_average(const T& parameter) const
{
    const size_t this_size = this->size();

    Vector<T> moving_average(this_size);

    moving_average[0] = ((*this)[0]+ (parameter*(*this)[1]))/(1.0+parameter);

    moving_average[this_size-1] = ((*this)[this_size-1]+ (parameter*(*this)[this_size-2]))/(1.0+parameter);

    for(size_t i = 1; i < this_size-1; i++)
    {
        moving_average[i] = ((*this)[i]+ (parameter*((*this)[i-1]+ (*this)[i+1])))/(1.0+2.0*parameter);
    }

    return(moving_average);
}


template <class T>
Vector<double> Vector<T>::calculate_simple_moving_average(const size_t& period) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

      if(period < 1)
      {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "double calculate_simple_moving_average(const size_t&) const method.\n"
               << "Period must be equal or greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    const size_t this_size = this->size();

    Vector<double> simple_moving_average(this_size, 0.0);

    #pragma omp parallel for

    for(int i = 0; i < this_size; i++)
    {
        const size_t begin = i < period ? 0 : static_cast<size_t>(i) - period + 1;
        const size_t end = static_cast<size_t>(i);

        simple_moving_average[i] = calculate_mean(begin, end);
    }

    return simple_moving_average;
}


template <class T>
Vector<double> Vector<T>::calculate_exponential_moving_average(const size_t& period) const
{
    const size_t size = this->size();

    Vector<double> exponential_moving_average(size);

    exponential_moving_average[0] = (*this)[0];

    const double multiplier = 2.0 / double(period + 1.0);

    for(size_t i = 1; i < size; i++)
    {
        exponential_moving_average[i] = (*this)[i] * multiplier + exponential_moving_average[i-1] *(1.0 - multiplier);
    }

    return exponential_moving_average;
}


template <class T>
double Vector<T>::calculate_last_exponential_moving_average(const size_t& period) const
{
    const Vector<double> exponential_moving_average = calculate_exponential_moving_average(period);

    return exponential_moving_average.get_last();
}


template <class T>
Vector<double> Vector<T>::calculate_exponential_moving_average_with_initial_average(const size_t& period) const
{
    const size_t size = this->size();

    Vector<double> exponential_moving_average(size);

    double initial_average = 0.0;

    for(size_t i = 0; i < period; i++)
    {
        initial_average += (*this)[i];
    }

    initial_average /= period;

    exponential_moving_average[0] = initial_average;

    const double multiplier = 2 / double(period + 1.0);

    for(size_t i = 1; i < size; i++)
    {
        exponential_moving_average[i] = (*this)[i] * multiplier + exponential_moving_average[i-1] *(1 - multiplier);
    }

    return(exponential_moving_average);
}


/// Returns the mean of the elements in the vector.

template <class T> double Vector<T>::calculate_mean() const
{
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mean() const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const T sum = calculate_sum();

  const double mean = sum /static_cast<double>(this_size);

  return(mean);
}


/// Returns the mean of the subvector defined by a start and end elements.
/// @param begin Start element.
/// @param end End element.

template <class T>
double Vector<T>::calculate_mean(const size_t& begin, const size_t& end) const
{
   // Control sentence(if debug)

  #ifdef __OPENNN_DEBUG__

    if(begin > end) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_mean(const size_t&, const size_t&) const method.\n"
             << "Begin must be less or equal than end.\n";

      throw logic_error(buffer.str());
    }

  #endif

  if(end == begin) return(*this)[begin];

  double sum = 0.0;

  for(size_t i = begin; i <= end; i++)
  {
      sum += (*this)[i];
  }

  return(sum /static_cast<double>(end-begin+1));
}


/// Returns the linear slope of this vector.

template <class T>
double Vector<T>::calculate_linear_trend() const
{
    const size_t this_size = this->size();

    const Vector<double> independent_variable(0.0,1.0,static_cast<double>(this_size-1));

    const LinearRegressionParameters<T> linear_regression_parameters = calculate_linear_regression_parameters(independent_variable);

    return(linear_regression_parameters.slope);
}


/// Returns the linear slope of a subvector defined by two elements.
/// @param start First element.
/// @param end Last element.

template <class T>
double Vector<T>::calculate_linear_trend(const size_t& start, const size_t& end) const
{
    const Vector<size_t> indices(start, 1, end);

    const Vector<double> dependent_variable = get_subvector(indices);

    return(dependent_variable.calculate_linear_trend());
}


template <class T>
double Vector<T>::calculate_percentage_of_variation() const
{
    const double percentage_of_variation = ((*this).get_last()-(*this).get_first())*100.0/(*this).get_first();

    return percentage_of_variation;
}


// @todo

template <class T>
Vector<double> Vector<T>::calculate_percentage_of_variation(const size_t& period) const
{
    const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  if(this_size < period) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<double> calculate_percentage_of_variation(const size_t&) const method.\n"
           << "Size must be greater than period.\n";

    throw logic_error(buffer.str());
  }

#endif

    Vector<double> percentage_of_variation(this_size, 0.0);

    if((*this) == 0.0)
    {
        return percentage_of_variation;
    }

    for(int i = 0; i < this_size; i++)
    {
        const size_t begin = i < period ? 0 : static_cast<size_t>(i) - period + 1;
        const size_t end = static_cast<size_t>(i);

        if((*this)[begin] != 0.0)
        {
            percentage_of_variation[i] = ((*this)[end] -(*this)[begin]) * 100.0 /(*this)[begin];
        }
        else
        {
            percentage_of_variation[static_cast<size_t>(i)] = percentage_of_variation[static_cast<size_t>(i-1)];
        }
    }

    return percentage_of_variation;
}


template <class T>
double Vector<T>::calculate_last_percentage_of_variation(const size_t& period) const
{
    const Vector<double> percentage_of_variation = calculate_percentage_of_variation(period);

    return percentage_of_variation.get_last();
}


/// Returns the mode of the vector, i.e., the element with most occurrences.

template <class T> T Vector<T>::calculate_mode() const
{
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__
    const size_t this_size = this->size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mode() const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const Vector<T> unique = get_unique_elements();

  const size_t maximal_index = count_unique().calculate_maximal_index();

  return(unique[maximal_index]);
}


/// Returns the mode of the vector, when it has missing values.
/// @param missing_indices Indices of the missing values in the vector.

template <class T> T Vector<T>::calculate_mode_missing_values(const Vector<size_t>& missing_indices) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mode_missing_values(const Vector<size_t>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t missing_indices_size = missing_indices.size();

  Vector<T> new_vector(this_size - missing_indices_size);

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++)
  {
      if(!missing_indices.contains(i))
      {
          new_vector[count] = (*this)[i];
          count++;
      }
  }

  return(new_vector.calculate_mode());
}


/// Returns the variance of the elements in the vector.

template <class T> double Vector<T>::calculate_variance() const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_variance() const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1)
  {
    return(0.0);
  }

  double sum = 0.0;
  double squared_sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    sum += (*this)[i];
    squared_sum += (*this)[i] *(*this)[i];
  }

  const double numerator = squared_sum -(sum * sum) /static_cast<double>(this_size);
  const double denominator = this_size - 1.0;

  if(denominator == 0.0)
  {
      return 0.0;
  }
  else
  {
      return(numerator / denominator);
  }
}


/// Returns the covariance of this vector and other vector

template<class T>
double Vector<T>::calculate_covariance(const Vector<double>& other_vector) const
{
   const size_t this_size = this->size();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

     if(this_size == 0) {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector Template.\n"
              << "double calculate_covariance(const Vector<double>&) const method.\n"
              << "Size must be greater than zero.\n";

       throw logic_error(buffer.str());
     }

   #endif

   // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

         if(this_size != other_vector.size()) {
             ostringstream buffer;

             buffer << "OpenNN Exception: Vector Template.\n"
                    << "double calculate_covariance(const Vector<double>&) const method.\n"
                    << "Size of this vectro must be equal to size of other vector.\n";

             throw logic_error(buffer.str());
         }

    #endif

     if(this_size == 1)
     {
         return 0.0;
     }

     const double this_mean = this->calculate_mean();
     const double other_mean = other_vector.calculate_mean();

     double numerator = 0.0;
     double denominator = static_cast<double>(this_size-1);

     for(size_t i = 0; i < this_size; i++)
     {
         numerator += ((*this)[i]-this_mean)*(other_vector[i]-other_mean);
     }

     return(numerator/denominator);
}


/// Returns the standard deviation of the elements in the vector.

template <class T>
double Vector<T>::calculate_standard_deviation() const
{
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_standard_deviation() const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  return(sqrt(calculate_variance()));
}


template <class T>
Vector<double> Vector<T>::calculate_standard_deviation(const size_t& period) const
{
  const size_t this_size = this->size();

  Vector<double> standard_deviation(this_size, 0.0);

  double mean = 0.0;
  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
      const size_t begin = i < period ? 0 : i - period + 1;
      const size_t end = i;

      mean = calculate_mean(begin,end);

      for(size_t j = begin; j < end+1; j++)
      {
          sum += ((*this)[j] - mean) *((*this)[j] - mean);
      }

      standard_deviation[i] = sqrt(sum / double(period));

      mean = 0.0;
      sum = 0.0;
  }

  standard_deviation[0] = standard_deviation[1];

  return(standard_deviation);
}


/// Returns the asymmetry of the elements in the vector

template <class T> double Vector<T>::calculate_asymmetry() const
{
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_asymmetry() const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1)
  {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    sum += ((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean);
  }

  const double numerator = sum /static_cast<double>(this_size);
  const double denominator = standard_deviation * standard_deviation * standard_deviation;

  return(numerator / denominator);
}


/// Returns the kurtosis value of the elements in the vector.

template <class T> double Vector<T>::calculate_kurtosis() const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_kurtosis() const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    sum += ((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean);
  }

  const double numerator = sum/static_cast<double>(this_size);
  const double denominator = standard_deviation*standard_deviation*standard_deviation*standard_deviation;

  return((numerator/denominator)-3.0);
}


/// Returns the mean and the standard deviation of the elements in the vector.

template <class T>
Vector<double> Vector<T>::calculate_mean_standard_deviation() const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mean_standard_deviation().\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const double mean = calculate_mean();
  const double standard_deviation = calculate_standard_deviation();

  return {mean, standard_deviation};
}


/// Returns the median of the elements in the vector

template <class T> double Vector<T>::calculate_median() const {
  const size_t this_size = this->size();

  Vector<T> sorted_vector(*this);

  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  size_t median_index;

  if(this_size % 2 == 0) {
    median_index = static_cast<size_t>(this_size / 2);

    return((sorted_vector[median_index-1] + sorted_vector[median_index]) / 2.0);
  } else {
    median_index = static_cast<size_t>(this_size / 2);

    return(sorted_vector[median_index]);
  }
}


/// Returns the quarters of the elements in the vector.

template <class T> Vector<double> Vector<T>::calculate_quartiles() const {
  const size_t this_size = this->size();
  Vector<T> sorted_vector(*this);

  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  Vector<double> quartiles(3);

  if(this_size == 1)
  {
      quartiles[0] = sorted_vector[0];
      quartiles[1] = sorted_vector[0];
      quartiles[2] = sorted_vector[0];
  }
  else if(this_size == 2)
  {
      quartiles[0] = (sorted_vector[0]+sorted_vector[1])/4;
      quartiles[1] = (sorted_vector[0]+sorted_vector[1])/2;
      quartiles[2] = (sorted_vector[0]+sorted_vector[1])*3/4;
  }
  else if(this_size == 3)
  {
      quartiles[0] = (sorted_vector[0]+sorted_vector[1])/2;
      quartiles[1] = sorted_vector[1];
      quartiles[2] = (sorted_vector[2]+sorted_vector[1])/2;
  }
  else if(this_size % 2 == 0)
  {
    quartiles[0] = sorted_vector.get_first(this_size/2).calculate_median();
    quartiles[1] = sorted_vector.calculate_median();
    quartiles[2] = sorted_vector.get_last(this_size/2).calculate_median();
  }
  else
  {
    quartiles[0] = sorted_vector[int(this_size/4)];
    quartiles[1] = sorted_vector[int(this_size/2)];
    quartiles[2] = sorted_vector[int(this_size*3/4)];
  }

  return(quartiles);
}


template <class T>
Vector<double> Vector<T>::calculate_percentiles() const
{
  const size_t this_size = this->size();

  const Vector<size_t> sorted_vector = this->sort_ascending();

  Vector<double> percentiles(10);

  if(this_size % 2 == 0)
  {
    percentiles[0] = (sorted_vector[this_size / 10] + sorted_vector[this_size / 10 + 1]) / 2;
    percentiles[1] = (sorted_vector[this_size * 2 / 10] + sorted_vector[this_size * 2 / 10 + 1]) / 2;
    percentiles[2] = (sorted_vector[this_size * 3 / 10] + sorted_vector[this_size * 3 / 10 + 1]) / 2;
    percentiles[3] = (sorted_vector[this_size * 4 / 10] + sorted_vector[this_size * 4 / 10 + 1]) / 2;
    percentiles[4] = (sorted_vector[this_size * 5 / 10] + sorted_vector[this_size * 5 / 10 + 1]) / 2;
    percentiles[5] = (sorted_vector[this_size * 6 / 10] + sorted_vector[this_size * 6 / 10 + 1]) / 2;
    percentiles[6] = (sorted_vector[this_size * 7 / 10] + sorted_vector[this_size * 7 / 10 + 1]) / 2;
    percentiles[7] = (sorted_vector[this_size * 8 / 10] + sorted_vector[this_size * 8 / 10 + 1]) / 2;
    percentiles[8] = (sorted_vector[this_size * 9 / 10] + sorted_vector[this_size * 9 / 10 + 1]) / 2;
    percentiles[9] = calculate_maximum();
  }
  else
  {
    percentiles[0] = sorted_vector[this_size / 10];
    percentiles[1] = sorted_vector[this_size * 2 / 10];
    percentiles[2] = sorted_vector[this_size * 3 / 10];
    percentiles[3] = sorted_vector[this_size * 4 / 10];
    percentiles[4] = sorted_vector[this_size * 5 / 10];
    percentiles[5] = sorted_vector[this_size * 6 / 10];
    percentiles[6] = sorted_vector[this_size * 7 / 10];
    percentiles[7] = sorted_vector[this_size * 8 / 10];
    percentiles[8] = sorted_vector[this_size * 9 / 10];
    percentiles[9] = calculate_maximum();
  }

  return(percentiles);
}


/// Returns the quarters of the elements in the vector when there are missing values.
/// @param missing_indices Vector with the indices of the missing values.

template <class T>
Vector<double> Vector<T>::calculate_quartiles_missing_values(const Vector<size_t> & missing_indices) const
{
    const size_t this_size = this->size();
    const size_t missing_indices_number = missing_indices.size();

    const Vector<T> values_to_remove = this->get_subvector(missing_indices);

    Vector<T> sorted_vector(*this);

    sorted_vector = sorted_vector.difference(values_to_remove);

    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    const size_t actual_size = this_size - missing_indices_number;

    Vector<double> quartiles(3);

    if(actual_size % 2 == 0)
    {
      quartiles[0] = (sorted_vector[actual_size / 4] + sorted_vector[actual_size / 4 + 1]) / 2.0;
      quartiles[1] = (sorted_vector[actual_size * 2 / 4] + sorted_vector[actual_size * 2 / 4 + 1]) / 2.0;
      quartiles[2] = (sorted_vector[actual_size * 3 / 4] + sorted_vector[actual_size * 3 / 4 + 1]) / 2.0;
    }
    else
    {
      quartiles[0] = sorted_vector[actual_size / 4];
      quartiles[1] = sorted_vector[actual_size * 2 / 4];
      quartiles[2] = sorted_vector[actual_size * 3 / 4];
    }

    return(quartiles);
}


/// Returns the mean of the elements in the vector.

template <class T>
double Vector<T>::calculate_mean_missing_values(const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mean_missing_values(const Vector<size_t>&) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  T sum = 0;

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      sum += (*this)[i];
      count++;
    }
  }

  const double mean = sum /static_cast<double>(count);

  return(mean);
}


/// Returns the variance of the elements in the vector.

template <class T>
double Vector<T>::calculate_variance_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_variance_missing_values(const Vector<size_t>&) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum = 0.0;
  double squared_sum = 0.0;

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      sum += (*this)[i];
      squared_sum += (*this)[i] *(*this)[i];

      count++;
    }
  }

  if(count <= 1) {
    return(0.0);
  }

  const double numerator = squared_sum -(sum * sum) /static_cast<double>(count);
  const double denominator = this_size - 1.0;

  return(numerator / denominator);
}


/// Returns the weighted mean of the vector.
/// @param weights Weights of the elements of the vector in the mean.

template <class T>
double Vector<T>::calculate_weighted_mean(const Vector<double> & weights) const
{
    const size_t this_size = this->size();

  // Control sentence(if debug)

  #ifdef __OPENNN_DEBUG__

    if(this_size == 0) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_weighted_mean(const Vector<double>&) const method.\n"
             << "Size must be greater than zero.\n";

      throw logic_error(buffer.str());
    }

    const size_t weights_size = weights.size();

    if(this_size != weights_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_weighted_mean(const Vector<double>&) "
                "const method.\n"
             << "Size of weights must be equal to vector size.\n";

      throw logic_error(buffer.str());
    }
  #endif

    double weights_sum = 0;

    T sum = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        sum += weights[i]*(*this)[i];
        weights_sum += weights[i];
    }

    const double mean = sum / weights_sum;

    return(mean);
}


/// Returns the standard deviation of the elements in the vector.

template <class T>
double Vector<T>::calculate_standard_deviation_missing_values(
    const Vector<size_t> &missing_indices) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_standard_deviation_missing_values(const "
              "Vector<size_t>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  return(sqrt(calculate_variance_missing_values(missing_indices)));
}


/// Returns the asymmetry of the elements in the vector.

template <class T>
double Vector<T>::calculate_asymmetry_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_asymmetry_missing_values(const "
              "Vector<size_t>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    if(!missing_indices.contains(i))
    {
      sum += ((*this)[i] - mean) *((*this)[i] - mean) *((*this)[i] - mean);
    }
  }

  const double numerator = sum /static_cast<double>(this_size);
  const double denominator =
      standard_deviation * standard_deviation * standard_deviation;

  return(numerator / denominator);
}


/// Returns the kurtosis of the elements in the vector.

template <class T>
double Vector<T>::calculate_kurtosis_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_kurtosis_missing_values(const Vector<size_t>&) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1)
  {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    if(!missing_indices.contains(i))
    {
      sum += ((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean);
    }
  }

  const double numerator = sum /static_cast<double>(this_size);
  const double denominator = standard_deviation*standard_deviation*standard_deviation*standard_deviation;

  return((numerator/denominator)-3.0);
}


/// Returns the minimum, maximum, mean and standard deviation of the elements in
/// the vector.

template <class T> Statistics<T> Vector<T>::calculate_statistics() const {
// Control sentence(if debug)

    const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_statistics().\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Statistics<T> statistics;

  T minimum = numeric_limits<T>::max();
  T maximum;

  double sum = 0;
  double squared_sum = 0;
  size_t count = 0;

  if(numeric_limits<T>::is_signed)
  {
    maximum = -1*numeric_limits<T>::max();
  }
  else
  {
    maximum = 0;
  }

  for(size_t i = 0; i < this_size; i++)
  {
      if((*this)[i] < minimum)
      {
          minimum = (*this)[i];
      }

      if((*this)[i] > maximum)
      {
          maximum = (*this)[i];
      }

      sum += (*this)[i];
      squared_sum += (*this)[i] *(*this)[i];

      count++;
  }

  const double mean = sum/static_cast<double>(count);

  double standard_deviation;

  if(count <= 1)
  {
    standard_deviation = 0.0;
  }
  else
  {
      const double numerator = squared_sum -(sum * sum) / count;
      const double denominator = this_size - 1.0;

      standard_deviation = numerator / denominator;
  }

  standard_deviation = sqrt(standard_deviation);

  statistics.minimum = minimum;
  statistics.maximum = maximum;
  statistics.mean = mean;
  statistics.standard_deviation = standard_deviation;

  return(statistics);
}


/// Returns the minimum, maximum, mean and standard deviation of the elements in the vector.

template <class T>
Statistics<T> Vector<T>::calculate_statistics_missing_values(
    const Vector<size_t> &missing_indices) const
{
// Control sentence(if debug)

    const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_statistics_missing_values(const "
              "Vector<size_t>&).\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Statistics<T> statistics;

  T minimum = numeric_limits<T>::max();
  T maximum;

  double sum = 0;
  double squared_sum = 0;
  size_t count = 0;

  if(numeric_limits<T>::is_signed) {
    maximum = -numeric_limits<T>::max();
  } else {
    maximum = 0;
  }

  for(size_t i = 0; i < this_size; i++) {
      if(!missing_indices.contains(i))
      {
          if((*this)[i] < minimum)
          {
            minimum = (*this)[i];
          }

          if((*this)[i] > maximum) {
            maximum = (*this)[i];
          }

          sum += (*this)[i];
          squared_sum += (*this)[i] *(*this)[i];

          count++;
      }
  }

  const double mean = sum/static_cast<double>(count);

  double standard_deviation;

  if(count <= 1)
  {
    standard_deviation = 0.0;
  }
  else
  {
      const double numerator = squared_sum -(sum * sum) / count;
      const double denominator = this_size - 1.0;

      standard_deviation = numerator / denominator;
  }

  standard_deviation = sqrt(standard_deviation);

  statistics.minimum = minimum;
  statistics.maximum = maximum;
  statistics.mean = mean;
  statistics.standard_deviation = standard_deviation;

  return(statistics);
}


/// Returns a vector with the asymmetry and the kurtosis values of the elements
/// in the vector.

template <class T>
Vector<double> Vector<T>::calculate_shape_parameters() const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<double> calculate_shape_parameters().\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<double> shape_parameters(2);

  shape_parameters[0] = calculate_asymmetry();
  shape_parameters[1] = calculate_kurtosis();

  return(shape_parameters);
}


/// Returns a vector with the asymmetry and the kurtosis values of the elements
/// in the vector.

template <class T>
Vector<double> Vector<T>::calculate_shape_parameters_missing_values(
    const Vector<size_t> &missing_values) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_shape_parameters_missing_values(const "
              "Vector<size_t>&).\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<double> shape_parameters(2);

  shape_parameters[0] = calculate_asymmetry_missing_values(missing_values);
  shape_parameters[1] = calculate_kurtosis_missing_values(missing_values);

  return(shape_parameters);
}


/// Returns the box and whispers for a vector.

template <class T>
Vector<double> Vector<T>::calculate_box_plot() const {

  Vector<double> box_plots(5, 0.0);

  if(this->empty()) return box_plots;

  const Vector<double> quartiles = calculate_quartiles();

  box_plots[0] = calculate_minimum();
  box_plots[1] = quartiles[0];
  box_plots[2] = quartiles[1];
  box_plots[3] = quartiles[2];
  box_plots[4] = calculate_maximum();

  return(box_plots);
}


/// Returns the box and whispers for a vector when there are missing values.
/// @param missing_indices Vector with the indices of the missing values.

template <class T>
Vector<double> Vector<T>::calculate_box_plot_missing_values(const Vector<size_t> & missing_indices) const
{
    Vector<double> box_plots(5);

    Vector<double> quartiles = calculate_quartiles_missing_values(missing_indices);

    box_plots[0] = calculate_minimum_missing_values(missing_indices);
    box_plots[1] = quartiles[0];
    box_plots[2] = quartiles[1];
    box_plots[3] = quartiles[2];
    box_plots[4] = calculate_maximum_missing_values(missing_indices);

    return(box_plots);
}


template <class T>
size_t Vector<T>::calculate_sample_index_proportional_probability() const
{
    const size_t this_size = this->size();

    Vector<double> cumulative = this->calculate_cumulative();

    const double sum = calculate_sum();

    const double random = calculate_random_uniform(0.,sum);

    size_t selected_index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(i == 0 && random < cumulative[0])
        {
            selected_index = i;
            break;
        }
        else if(random < cumulative[i] && random >= cumulative[i-1])
        {
            selected_index = i;
            break;
        }
    }

    return selected_index;
}



/// Returns the vector norm.

template <class T>
double Vector<T>::calculate_L1_norm() const {
  // Control sentence(if debug)

  return calculate_absolute_value().calculate_sum();
}


/// Returns the gradient of the vector norm.

template <class T> Vector<T> Vector<T>::calculate_sign() const {
  const size_t this_size = this->size();

  // Control sentence(if debug)

  Vector<T> sign_vector(this_size);

  for (size_t i = 0; i < this_size; i++)
  {
    if ((*this)[i] < 0)
    {
        sign_vector[i] = -1.0;
    }
    else if((*this)[i] > 0)
    {
        sign_vector[i] = 1.0;
    }
    else
    {
        throw logic_error("Error: Parameter " + to_string(i) + " is equal to zero: Non-derivative function");
    }
  }

  return(sign_vector);
}


/// Returns the gradient of the vector norm.

template <class T> Vector<T> Vector<T>::calculate_L1_norm_gradient() const {

  // Control sentence(if debug)

  return(calculate_sign());
}


/// Returns the Hessian of the vector norm.

template <class T> Matrix<T> Vector<T>::calculate_L1_norm_Hessian() const {
  const size_t this_size = this->size();

  // Control sentence(if debug)

  Matrix<T> Hessian(this_size, this_size, 0);

  return(Hessian);
}


/// Returns the vector norm.

template <class T>
double Vector<T>::calculate_L2_norm() const {
  const size_t this_size = this->size();

  // Control sentence(if debug)

  double norm = 0.0;

  for(size_t i = 0; i < this_size; i++) {
    norm += (*this)[i] *(*this)[i];
  }

    return sqrt(norm);
}


/// Returns the gradient of the vector norm.

template <class T> Vector<T> Vector<T>::calculate_L2_norm_gradient() const {
  const size_t this_size = this->size();

  // Control sentence(if debug)

  Vector<T> gradient(this_size);

  const double norm = calculate_L2_norm();

  if(norm == 0.0) {
    gradient.initialize(0.0);
  } else {
    gradient = (*this) / norm;
  }

  return(gradient);
}


/// Returns the Hessian of the vector norm.

template <class T> Matrix<T> Vector<T>::calculate_L2_norm_Hessian() const {
  const size_t this_size = this->size();

  // Control sentence(if debug)

  Matrix<T> Hessian(this_size, this_size);

  const double norm = calculate_L2_norm();

  if(norm == 0.0) {
    Hessian.initialize(0.0);
  } else {
    Hessian = (*this).direct(*this)/(norm * norm * norm);
  }

  return(Hessian);
}


/// Returns the vector p-norm.

template <class T> double Vector<T>::calculate_Lp_norm(const double &p) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_p_norm(const double&) const method.\n"
           << "p value must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  // Control sentence(if debug)

  double norm = 0.0;

  for(size_t i = 0; i < this_size; i++) {
    norm += pow(fabs((*this)[i]), p);
  }

  norm = pow(norm, 1.0 / p);

  return(norm);
}


/// Returns the gradient of the vector norm.

template <class T>
Vector<double> Vector<T>::calculate_Lp_norm_gradient(const double &p) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<double> calculate_p_norm_gradient(const double&) const "
              "method.\n"
           << "p value must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  Vector<double> gradient(this_size);

  const double p_norm = calculate_Lp_norm(p);

  if(p_norm == 0.0) {
    gradient.initialize(0.0);
  } else {
    for(size_t i = 0; i < this_size; i++) {
      gradient[i] =
         (*this)[i] * pow(fabs((*this)[i]), p - 2.0) / pow(p_norm, p - 1.0);
    }
  }

  return(gradient);
}


/// Returns this vector divided by its norm.

template <class T> Vector<T> Vector<T>::calculate_normalized() const {
  const size_t this_size = this->size();

  Vector<T> normalized(this_size);

  const double norm = calculate_L2_norm();

  if(norm == 0.0) {
    normalized.initialize(0.0);
  } else {
    normalized = (*this) / norm;
  }

  return(normalized);
}


/// Returns the distance between the elements of this vector and the elements of
/// another vector.
/// @param other_vector Other vector.

template <class T>
double Vector<T>::calculate_euclidean_distance(const Vector<T> &other_vector) const {

    const size_t this_size = this->size();
#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_euclidean_distance(const Vector<T>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < this_size; i++)
    {
        error = (*this)[i] - other_vector[i];

        distance += error * error;
    }

    return(sqrt(distance));
}


template <class T>
double Vector<T>::calculate_euclidean_weighted_distance(const Vector<T>& other_vector, const Vector<double>& weights) const {

    const size_t this_size = this->size();
#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_euclidean_weighted_distance(const Vector<T>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < this_size; i++) {
        error = (*this)[i] - other_vector[i];

        distance += error * error * weights[i];
    }

    return(sqrt(distance));
}


template <class T>
Vector<double> Vector<T>::calculate_euclidean_weighted_distance_vector(const Vector<T>& other_vector, const Vector<double>& weights) const {

    const size_t this_size = this->size();
#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_euclidean_weighted_distance(const Vector<T>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Vector<double> distance(this_size,0.0);
    double error;

    for(size_t i = 0; i < this_size; i++) {
        error = (*this)[i] - other_vector[i];

        distance[i] = error * error * weights[i];
    }

    return(distance);
}

template <class T>
double Vector<T>::calculate_manhattan_distance(const Vector<T> &other_vector) const {

    const size_t this_size = this->size();
#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_manhattan_distance(const Vector<T>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < this_size; i++) {
        error = abs((*this)[i] - other_vector[i]);

        distance += error;
    }

    return(distance);
}


template <class T>
double Vector<T>::calculate_manhattan_weighted_distance(const Vector<T>& other_vector, const Vector<double>& weights) const {

    const size_t this_size = this->size();
#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_manhattan_weighted_distance(const Vector<T>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < this_size; i++) {
        error = abs((*this)[i] - other_vector[i]);

//        cout << (*this)[i] << "/" << other_vector[i] << "/" << weights[i] << "/" << error << endl;

        distance += error * weights[i];

//        cout << distance << endl;
    }

//    system("pause");

    return(distance);
}


template <class T>
Vector<double> Vector<T>::calculate_manhattan_weighted_distance_vector(const Vector<T>& other_vector, const Vector<double>& weights) const {

    const size_t this_size = this->size();
#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_manhattan_weighted_distance(const Vector<T>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Vector<double> distance(this_size,0.0);
    double error;

    for(size_t i = 0; i < this_size; i++) {
        error = abs((*this)[i] - other_vector[i]);

//        if (i==0) cout << error << endl;

        distance[i] = error * weights[i];
    }

    return(distance);
}


/// Returns the sum squared error between the elements of this vector and the
/// elements of another vector.
/// @param other_vector Other vector.

template <class T>
double Vector<T>::calculate_sum_squared_error(const Vector<double> &other_vector) const
{
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_sum_squared_error(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum_squared_error = 0.0;
  double error;

  for(size_t i = 0; i < this_size; i++)
  {
    error = (*this)[i] - other_vector[i];

    sum_squared_error += error * error;
  }

  return(sum_squared_error);
}


/// Returns the sum squared error between the elements of this vector and the
/// elements of a row of a matrix.
/// @param matrix Matrix to compute the error.
/// @param row_index Index of the row of the matrix.
/// @param column_indices Indices of the columns of the matrix to evaluate.

template <class T>
double Vector<T>::calculate_sum_squared_error(
    const Matrix<T> &matrix, const size_t &row_index,
    const Vector<size_t> &column_indices) const {

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

   const size_t this_size = this->size();
   const size_t other_size = column_indices.size();

   if(other_size != this_size)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_sum_squared_error(const Matrix<T>&, const size_t&, const Vector<size_t>&) const method.\n"
             << "Size must be equal to this size.\n";

      throw logic_error(buffer.str());
   }

#endif

  double sum_squared_error = 0.0;
  double error;

  const size_t size = column_indices.size();

  for(size_t i = 0; i < size; i++) {
    error = (*this)[i] - matrix(row_index, column_indices[i]);

    sum_squared_error += error * error;
  }

  return(sum_squared_error);
}


/// Returns the Minkowski squared error between the elements of this vector and
/// the elements of another vector.
/// @param other_vector Other vector.
/// @param Minkowski_parameter Minkowski exponent.

template <class T>
double
Vector<T>::calculate_Minkowski_error(const Vector<double> &other_vector,
                                     const double &Minkowski_parameter) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(this_size == 0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_Minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_Minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "Other size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

  // Control sentence

  if(Minkowski_parameter < 1.0 || Minkowski_parameter > 2.0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_Minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "The Minkowski parameter must be comprised between 1 and 2\n";

    throw logic_error(buffer.str());
  }

#endif

  double Minkowski_error = 0.0;

  for(size_t i = 0; i < this_size; i++) {
    Minkowski_error +=
        pow(fabs((*this)[i] - other_vector[i]), Minkowski_parameter);
  }

  Minkowski_error = pow(Minkowski_error, 1.0 / Minkowski_parameter);

  return(Minkowski_error);
}


/// Calculates the linear regression parameters(intercept, slope and
/// correlation) between another vector and this vector.
/// It returns a linear regression parameters structure.
/// @param other Other vector for the linear regression analysis.

template <class T>
LinearRegressionParameters<T> Vector<T>::calculate_linear_regression_parameters(
    const Vector<T> &x) const {
  const size_t n = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t x_size = x.size();

  ostringstream buffer;

  if(x_size != n) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "LinearRegressionParameters<T> "
              "calculate_linear_regression_parameters(const Vector<T>&) const "
              "method.\n"
           << "Other size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

  T s_x = 0;
  T s_y = 0;

  T s_xx = 0;
  T s_yy = 0;

  T s_xy = 0;

  for(size_t i = 0; i < n; i++) {
    s_x += x[i];
    s_y += (*this)[i];

    s_xx += x[i] * x[i];
    s_yy += (*this)[i] *(*this)[i];

    s_xy += x[i] *(*this)[i];
  }

  LinearRegressionParameters<T> linear_regression_parameters;

  if(s_x == 0 && s_y == 0 && s_xx == 0 && s_yy == 0 && s_xy == 0) {
    linear_regression_parameters.intercept = 0.0;

    linear_regression_parameters.slope = 0.0;

    linear_regression_parameters.correlation = 1.0;
  } else {
    linear_regression_parameters.intercept =
       (s_y * s_xx - s_x * s_xy) /(n * s_xx - s_x * s_x);

    linear_regression_parameters.slope =
       (n * s_xy - s_x * s_y) /(n * s_xx - s_x * s_x);

    if(sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y)) < 1.0e-12)
    {
        linear_regression_parameters.correlation = 1.0;
    }
    else
    {
        linear_regression_parameters.correlation =
           (n * s_xy - s_x * s_y) /
            sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y));
    }
  }

  return(linear_regression_parameters);
}


/// Returns a vector with the absolute values of the current vector.

template <class T> Vector<T> Vector<T>::calculate_absolute_value() const {
  const size_t this_size = this->size();

  Vector<T> absolute_value(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > 0) {
      absolute_value[i] = (*this)[i];
    } else {
      absolute_value[i] = -(*this)[i];
    }
  }

  return(absolute_value);
}


/// Sets the elements of the vector to their absolute values.

template <class T> void Vector<T>::apply_absolute_value() {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < 0) {
     (*this)[i] = -(*this)[i];
    }
  }
}


/// Returns a vector with the bounded elements from below of the current vector.
/// @param lower_bound Lower bound values.

template <class T>
Vector<T> Vector<T>::calculate_lower_bounded(const T &lower_bound) const {
  const size_t this_size = this->size();

  Vector<T> bounded_vector(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
      bounded_vector[i] = lower_bound;
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return(bounded_vector);
}


/// Returns a vector with the bounded elements from above of the current vector.
/// @param lower_bound Lower bound values.

template <class T>
Vector<T>
Vector<T>::calculate_lower_bounded(const Vector<T> &lower_bound) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t lower_bound_size = lower_bound.size();

  if(lower_bound_size != this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "Vector<T> calculate_lower_bounded(const Vector<T>&) const method.\n"
        << "Lower bound size must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> bounded_vector(this_size);

  // Apply lower bound

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
      bounded_vector[i] = lower_bound[i];
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return(bounded_vector);
}


/// This method bounds the elements of the vector if they fall above an upper
/// bound value.
/// @param upper_bound Upper bound value.

template <class T>
Vector<T> Vector<T>::calculate_upper_bounded(const T &upper_bound) const {
  const size_t this_size = this->size();

  Vector<T> bounded_vector(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound) {
      bounded_vector[i] = upper_bound;
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return(bounded_vector);
}


/// This method bounds the elements of the vector if they fall above their
/// corresponding upper bound values.
/// @param upper_bound Upper bound values.

template <class T>
Vector<T>
Vector<T>::calculate_upper_bounded(const Vector<T> &upper_bound) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t upper_bound_size = upper_bound.size();

  if(upper_bound_size != this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "Vector<T> calculate_upper_bounded(const Vector<T>&) const method.\n"
        << "Upper bound size must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> bounded_vector(this_size);

  // Apply upper bound

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound[i]) {
      bounded_vector[i] = upper_bound[i];
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return(bounded_vector);
}


/// This method bounds the elements of the vector if they fall above or below
/// their lower or upper
/// bound values, respectively.
/// @param lower_bound Lower bound value.
/// @param upper_bound Upper bound value.

template <class T>
Vector<T> Vector<T>::calculate_lower_upper_bounded(const T &lower_bound, const T &upper_bound) const
{
  const size_t this_size = this->size();

  Vector<T> bounded_vector(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
      bounded_vector[i] = lower_bound;
    } else if((*this)[i] > upper_bound) {
      bounded_vector[i] = upper_bound;
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return(bounded_vector);
}


/// This method bounds the elements of the vector if they fall above or below
/// their corresponding lower or upper
/// bound values, respectively.
/// @param lower_bound Lower bound values.
/// @param upper_bound Upper bound values.

template <class T>
Vector<T>
Vector<T>::calculate_lower_upper_bounded(const Vector<T> &lower_bound,
                                         const Vector<T> &upper_bound) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t lower_bound_size = lower_bound.size();
  const size_t upper_bound_size = upper_bound.size();

  if(lower_bound_size != this_size || upper_bound_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> calculate_lower_upper_bounded(const Vector<T>&, const "
              "Vector<T>&) const method.\n"
           << "Lower and upper bound sizes must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> bounded_vector(this_size);

  // Apply lower and upper bounds

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
      bounded_vector[i] = lower_bound[i];
    } else if((*this)[i] > upper_bound[i]) {
      bounded_vector[i] = upper_bound[i];
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return(bounded_vector);
}


/// Sets the elements of the vector to a given value if they fall below that
/// value.
/// @param lower_bound Lower bound value.

template <class T> void Vector<T>::apply_lower_bound(const T &lower_bound) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
     (*this)[i] = lower_bound;
    }
  }
}


/// Sets the elements of the vector to given values if they fall below that
/// values.
/// @param lower_bound Lower bound values.

template <class T>
void Vector<T>::apply_lower_bound(const Vector<T> &lower_bound) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
     (*this)[i] = lower_bound[i];
    }
  }
}


/// Sets the elements of the vector to a given value if they fall above that
/// value.
/// @param upper_bound Upper bound value.

template <class T> void Vector<T>::apply_upper_bound(const T &upper_bound) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound) {
     (*this)[i] = upper_bound;
    }
  }
}


/// Sets the elements of the vector to given values if they fall above that
/// values.
/// @param upper_bound Upper bound values.

template <class T>
void Vector<T>::apply_upper_bound(const Vector<T> &upper_bound) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound[i]) {
     (*this)[i] = upper_bound[i];
    }
  }
}


/// Sets the elements of the vector to a given lower bound value if they fall
/// below that value,
/// or to a given upper bound value if they fall above that value.
/// @param lower_bound Lower bound value.
/// @param upper_bound Upper bound value.

template <class T>
void Vector<T>::apply_lower_upper_bounds(const T &lower_bound,
                                         const T &upper_bound) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
     (*this)[i] = lower_bound;
    } else if((*this)[i] > upper_bound) {
     (*this)[i] = upper_bound;
    }
  }
}


/// Sets the elements of the vector to given lower bound values if they fall
/// below that values,
/// or to given upper bound values if they fall above that values.
/// @param lower_bound Lower bound values.
/// @param upper_bound Upper bound values.

template <class T>
void Vector<T>::apply_lower_upper_bounds(const Vector<T> &lower_bound,
                                         const Vector<T> &upper_bound) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
     (*this)[i] = lower_bound[i];
    } else if((*this)[i] > upper_bound[i]) {
     (*this)[i] = upper_bound[i];
    }
  }
}


/// Returns the vector of the indices of the vector sorted by less ranks.

template <class T>
Vector<size_t> Vector<T>::sort_ascending_indices() const
{
    Vector<size_t> indices(this->size());

#ifdef __Cpp11__

    const Vector<size_t> less_rank = this->calculate_less_rank();

    for(size_t i = 0; i < this->size(); i++)
    {
        indices[less_rank[i]] = i;
    }

#else

    indices.initialize_sequential();
    sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {return(*this)[i1] <(*this)[i2];});

#endif

    return(indices);
}


template <class T>
Vector<T> Vector<T>::sort_ascending_values() const
{
    Vector<T> sorted(*this);

    sort(sorted.begin(), sorted.end());

    return sorted;
}


template <class T>
Vector<size_t> Vector<T>::calculate_lower_indices(const size_t& indices_number) const
{
    return sort_ascending_indices().get_subvector(0,indices_number-1);
}


template <class T>
Vector<T> Vector<T>::calculate_lower_values(const size_t& indices_number) const
{
    return sort_ascending_values().get_subvector(0,indices_number-1);
}


/// Returns the vector of the indices of the vector sorted by greater ranks.

template <class T>
Vector<size_t> Vector<T>::sort_descending_indices() const
{
    Vector<size_t> indices(this->size());

#ifdef __Cpp11__

    const Vector<size_t> greater_rank = this->calculate_greater_rank();

    for(size_t i = 0; i < this->size(); i++)
    {
        indices[greater_rank[i]] = i;
    }

#else

    indices.initialize_sequential();
    sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {return(*this)[i1] >(*this)[i2];});

#endif

    return(indices);
}


template <class T>
Vector<T> Vector<T>::sort_descending_values() const
{
    Vector<T> sorted(*this);

    sort(sorted.begin(), sorted.end());

    return sorted.get_reverse();
}


/// Returns a vector with the rank of the elements of this vector.
/// The smallest element will have rank 0, and the greatest element will have
/// size-1.
/// That is, small values correspond with small ranks.

template <class T> Vector<size_t> Vector<T>::calculate_less_rank() const
{
  const size_t this_size = this->size();

  Vector<size_t> rank(this_size);

  Vector<T> sorted_vector(*this);

  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  Vector<size_t> previous_rank;
  previous_rank.set(this_size, -1);

//  #pragma omp parallel for schedule(dynamic)

  for(int i = 0; i < this_size; i++)
  {
    for(int j = 0; j < this_size; j++)
    {
      if(previous_rank.contains(static_cast<size_t>(j)))
      {
        continue;
      }

      if((*this)[i] == sorted_vector[j])
      {
        rank[static_cast<size_t>(i)] = static_cast<size_t>(j);

        previous_rank[static_cast<size_t>(i)] = static_cast<size_t>(j);

        break;
      }
    }
  }

  return(rank);
}


/// Returns a vector with the rank of the elements of this vector.
/// The smallest element will have rank 0, and the greatest element will have
/// size-1.
/// That is, small values correspond with small ranks.
/// Ties are assigned the mean of the ranks.

template <class T> Vector<double> Vector<T>::calculate_less_rank_with_ties() const
{
    Vector<double> indices_this = this->calculate_less_rank().to_double_vector();

    const Vector<double> this_unique = this->get_unique_elements();
    const Vector<size_t> this_unique_frecuency = this->count_unique();

    const size_t n = this->size();

//#pragma omp parallel for

    for(int  i = 0; i < static_cast<int>(this_unique.size()); i++)
    {
        if(this_unique_frecuency[static_cast<size_t>(i)] > 1)
        {
             const double unique = this_unique[static_cast<size_t>(i)];

             Vector<double> indices(this_unique_frecuency[static_cast<size_t>(i)]);

             for(size_t j = 0; j < n; j++)
             {
                 if((*this)[j] == unique)
                 {
                     indices.push_back(indices_this[j]);
                 }
             }

             const double mean_indice = indices.calculate_mean();

             for(size_t j = 0; j < n; j++)
             {
                 if((*this)[j] == unique)
                 {
                     indices_this[j] = mean_indice;
                 }
             }
        }
    }

    return indices_this;
}


/// Returns a vector with the rank of the elements of this vector.
/// The smallest element will have rank size-1, and the greatest element will
/// have 0.
/// That is, small values correspond to big ranks.

template <class T>
Vector<size_t> Vector<T>::calculate_greater_rank() const
{
    const size_t this_size = this->size();

    Vector<size_t> rank(this_size);

    Vector<T> sorted_vector(*this);

    sort(sorted_vector.begin(), sorted_vector.end(), greater<T>());

    Vector<size_t> previous_rank;
    previous_rank.set(this_size, -1);

    for(size_t i = 0; i < this_size; i++)
    {
        for(size_t j = 0; j < this_size; j++)
        {
            if(previous_rank.contains(j))
            {
                continue;
            }
            if((*this)[i] == sorted_vector[j])
            {
                rank[i] = j;

                previous_rank[i] = j;

                break;
            }
        }
    }

    return(rank);
}


// @todo

template <class T>
Vector<size_t> Vector<T>::calculate_greater_indices() const
{
    const size_t this_size = this->size();

    Vector<size_t> y(this_size);
    size_t n(0);
    generate(y.begin(), y.end(), [&]{ return n++; });

    sort(y.begin(), y.end(), [&](size_t i1, size_t i2) { return(*this)[i1] > (*this)[i2]; } );

    return(y);
}


/// Returns a vector sorted according to a given rank.
/// @param rank Given rank.

template <class T>
Vector<T> Vector<T>::sort_rank(const Vector<size_t>& rank) const
{
    const size_t this_size = this->size();

    #ifdef __OPENNN_DEBUG__
    const size_t rank_size = rank.size();

      if(this_size != rank_size) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> sort_rank(const Vector<size_t>&) const.\n"
               << "Sizes of vectors are " << this_size << " and " << rank_size
               << " and they must be the same.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Vector<T> sorted_vector(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        sorted_vector[i] = (*this)[rank[i]];
    }

    return sorted_vector;
}


template <class T>
inline Vector<T> Vector<T>::operator= (const initializer_list<T>& list) const {

  return Vector<T>(list);
}


/// Sum vector+scalar arithmetic operator.
/// @param scalar Scalar value to be added to this vector.

template <class T>
inline Vector<T> Vector<T>::operator+ (const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> sum(this_size);

  transform(this->begin(), this->end(), sum.begin(),
                 bind2nd(plus<T>(), scalar));

  return(sum);
}


/// Sum vector+vector arithmetic operator.
/// @param other_vector Vector to be added to this vector.

template <class T>
inline Vector<T> Vector<T>::operator+ (const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator + (const Vector<T>) const.\n"
           << "Sizes of vectors are " << this_size << " and " << other_size
           << " and they must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> sum(this_size);

  transform(this->begin(), this->end(), other_vector.begin(), sum.begin(),
                 plus<T>());

  return(sum);
}


/// Difference vector-scalar arithmetic operator.
/// @param scalar Scalar value to be subtracted to this vector.

template <class T>
inline Vector<T> Vector<T>::operator-(const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> difference(this_size);

  transform(this->begin(), this->end(), difference.begin(),
                 bind2nd(minus<T>(), scalar));

  return(difference);
}


/// Difference vector-vector arithmetic operator.
/// @param other_vector vector to be subtracted to this vector.

template <class T>
inline Vector<T> Vector<T>::operator-(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator -(const Vector<T>&) const.\n"
           << "Sizes of vectors are " << this_size << " and " << other_size
           << " and they must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> difference(this_size);

  transform(this->begin(), this->end(), other_vector.begin(),
                 difference.begin(), minus<T>());

  return(difference);
}


/// Product vector*scalar arithmetic operator.
/// @param scalar Scalar value to be multiplied to this vector.

template <class T> Vector<T> Vector<T>::operator*(const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> product(this_size);

  transform(this->begin(), this->end(), product.begin(),
                 bind2nd(multiplies<T>(), scalar));

  return(product);
}


/// Element by element product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

template <class T>
inline Vector<T> Vector<T>::operator*(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator *(const Vector<T>&) const.\n"
           << "Size of other vector(" << other_size
           << ") must be equal to size of this vector(" << this_size << ").\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> product(this_size);

  transform(this->begin(), this->end(), other_vector.begin(),
                 product.begin(), multiplies<T>());

  return(product);
}


/// Element by row product vector*matrix arithmetic operator.
/// @param matrix matrix to be multiplied to this vector.

template <class T>
Matrix<T> Vector<T>::operator*(const Matrix<T> &matrix) const {
  const size_t rows_number = matrix.get_rows_number();
  const size_t columns_number = matrix.get_columns_number();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(rows_number != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator *(const Matrix<T>&) const.\n"
           << "Number of matrix rows(" << rows_number
           << ") must be equal to vector size(" << this_size << ").\n";

    throw logic_error(buffer.str());
  }

#endif

  Matrix<T> product(rows_number, columns_number);

  for(size_t i = 0; i < rows_number; i++) {
    for(size_t j = 0; j < columns_number; j++) {
      product(i, j) = (*this)[i] * matrix(i, j);
    }
  }

  return(product);
}


/// Returns the dot product of this vector with a matrix.
/// The number of rows of the matrix must be equal to the size of the vector.
/// @param matrix matrix to be multiplied to this vector.

template <class T>
Vector<double> Vector<T>::dot(const Matrix<T> &matrix) const {
  const size_t rows_number = matrix.get_rows_number();
  const size_t columns_number = matrix.get_columns_number();
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(rows_number != this_size)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> dot(const Matrix<T>&) const method.\n"
           << "Matrix number of rows (" << rows_number << ") must be equal to vector size (" << this_size << ").\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<double> product(columns_number, 0.0);

   for(size_t j = 0; j < columns_number; j++)
   {
      for(size_t i = 0; i < rows_number; i++)
      {
         product[j] += (*this)[i]*matrix(i,j);
      }
   }

//  const Eigen::Map<Eigen::VectorXd> vector_eigen((double *)this->data(), this_size);

//  const Eigen::Map<Eigen::MatrixXd> matrix_eigen((double *)matrix.data(), rows_number, columns_number);

//  Eigen::Map<Eigen::VectorXd> product_eigen(product.data(), columns_number);

//  product_eigen = vector_eigen.transpose() * matrix_eigen;

  return(product);
}


/// Dot product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

template <class T>
inline double Vector<T>::dot(const Vector<double> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Type dot(const Vector<T>&) const method.\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  //   const Eigen::Map<Eigen::VectorXd>
  //   this_vector_eigen((double*)this->data(), this_size);
  //   const Eigen::Map<Eigen::VectorXd>
  //   other_vector_eigen((double*)other_vector.data(), this_size);

  //   return(this_vector_eigen.dot(other_vector_eigen));

  double dot_product = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    dot_product += (*this)[i] * other_vector[i];
  }

  return(dot_product);
}


/// Outer product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

template <class T>
Matrix<T> Vector<T>::direct(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Matrix<T> direct(const Vector<T>&) const method.\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  Matrix<T> direct(this_size, this_size);

#pragma omp parallel for if(this_size > 1000)

  for(int i = 0; i < static_cast<int>(this_size); i++)
  {
    for(size_t j = 0; j < this_size; j++)
    {
      direct(i, j) = (*this)[i] * other_vector[j];
    }
  }

  return(direct);
}


/// Cocient vector/scalar arithmetic operator.
/// @param scalar Scalar value to be divided to this vector.

template <class T> Vector<T> Vector<T>::operator/(const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> cocient(this_size);

  transform(this->begin(), this->end(), cocient.begin(),
                 bind2nd(divides<T>(), scalar));

  return(cocient);
}


/// Cocient vector/vector arithmetic operator.
/// @param other_vector vector to be divided to this vector.

template <class T>
Vector<T> Vector<T>::operator/(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator /(const Vector<T>&) const.\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> cocient(this_size);

  transform(this->begin(), this->end(), other_vector.begin(),
                 cocient.begin(), divides<T>());

  return(cocient);
}


/// Scalar sum and assignment operator.
/// @param value Scalar value to be added to this vector.

template <class T> void Vector<T>::operator+= (const T &value) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] + value;
  }
}


/// Vector sum and assignment operator.
/// @param other_vector Vector to be added to this vector.

template <class T> void Vector<T>::operator+= (const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator += (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] + other_vector[i];
  }
}


/// Scalar rest and assignment operator.
/// @param value Scalar value to be subtracted to this vector.

template <class T> void Vector<T>::operator-= (const T &value) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] - value;
  }
}


/// Vector rest and assignment operator.
/// @param other_vector Vector to be subtracted to this vector.

template <class T> void Vector<T>::operator-= (const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator -= (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] - other_vector[i];
  }
}


/// Scalar product and assignment operator.
/// @param value Scalar value to be multiplied to this vector.

template <class T> void Vector<T>::operator*= (const T &value) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] * value;
  }
}


/// Vector product and assignment operator.
/// @param other_vector Vector to be multiplied to this vector.

template <class T> void Vector<T>::operator*= (const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator *= (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] * other_vector[i];
  }
}


/// Scalar division and assignment operator.
/// @param value Scalar value to be divided to this vector.

template <class T> void Vector<T>::operator/= (const T &value) {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] / value;
  }
}


/// Vector division and assignment operator.
/// @param other_vector Vector to be divided to this vector.

template <class T> void Vector<T>::operator/= (const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator /= (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] / other_vector[i];
  }
}


/// Sets all the negative elements in the vector to zero.

template <class T>
Vector<T> Vector<T>::filter_positive() const
{
  Vector<T> new_vector(*this);

  for(size_t i = 0; i < this->size(); i++) {
    if(new_vector[i] < 0) {
      new_vector[i] = 0;
    }
  }

  return new_vector;
}


/// Sets all the positive elements in the vector to zero.

template <class T> Vector<T> Vector<T>::filter_negative() const {
    Vector<T> new_vector(*this);
  for(size_t i = 0; i < this->size(); i++) {
    if(new_vector[i] > 0) {
      new_vector[i] = 0;
    }
  }
}


template <class T>
size_t Vector<T>::count_dates(const size_t& start_day, const size_t& start_month, const size_t& start_year,
                              const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    struct tm start;

    start.tm_mday = static_cast<int>(start_day);
    start.tm_mon = static_cast<int>(start_month) - 1;
    start.tm_year = static_cast<int>(start_year) - 1900;

    const time_t start_date = mktime(&start) + 3600*24;

    struct tm end;

    end.tm_mday = static_cast<int>(end_day);
    end.tm_mon = static_cast<int>(end_month) - 1;
    end.tm_year = static_cast<int>(end_year) - 1900;

    const time_t end_date = mktime(&end) + 3600*24;

    size_t count = 0;

    for(size_t i = 0; i < this->size(); i++)
    {
        if((*this)[i] >= start_date &&(*this)[i] <= end_date)
        {
            count++;
        }
    }

    return(count);
}


/// Returns the indices of the timestamp elements which fall between two given dates.
/// @param start_month Start month.
/// @param start_year Start year.
/// @param end_month End month.
/// @param end_year End year.

template <class T>
Vector<size_t> Vector<T>::filter_dates(const size_t& start_day, const size_t& start_month, const size_t& start_year,
                                       const size_t& end_day, const size_t& end_month, const size_t& end_year) const
{
    const size_t new_size = count_dates(start_day, start_month, start_year, end_day, end_month, end_year);

    Vector<size_t> indices(new_size);

    struct tm start;
    start.tm_mday = static_cast<int>(start_day);
    start.tm_mon = static_cast<int>(start_month) - 1;
    start.tm_year = static_cast<int>(start_year) - 1900;

    const time_t start_date = mktime(&start) + 3600*24;

    struct tm end;
    end.tm_mday = static_cast<int>(end_day);
    end.tm_mon = static_cast<int>(end_month) - 1;
    end.tm_year = static_cast<int>(end_year) - 1900;

    const time_t end_date = mktime(&end) + 3600*24;

    size_t index = 0;

    for(size_t i = 0; i < this->size(); i++)
    {
        if((*this)[i] >= start_date && (*this)[i] <= end_date)
        {
            indices[index] = i;
            index++;
        }
    }

    return(indices);
}


/// Calculates the outliers in the vector using the Tukey's algorithm,
/// and returns the indices of that elements.
/// @param cleaning_parameter Cleaning parameter in the Tukey's method. The default value is 1.5.

template <class T>
Vector<size_t> Vector<T>::calculate_Tukey_outliers(const double& cleaning_parameter) const
{
    Vector<size_t> outliers_indices;

    if(this->is_binary())
    {
        return outliers_indices;
    }

    const size_t this_size = this->size();

    Vector<double> box_plot;

    double interquartile_range;

    box_plot = calculate_box_plot();

    if(fabs(box_plot[3] - box_plot[1]) < numeric_limits<double>::epsilon())
    {
        return outliers_indices;
    }
    else
    {
        interquartile_range = abs((box_plot[3] - box_plot[1]));
    }

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] <(box_plot[1] - cleaning_parameter*interquartile_range) ||
          (*this)[i] >(box_plot[3] + cleaning_parameter*interquartile_range))
        {
            outliers_indices.push_back(i);
        }
    }

    return(outliers_indices);
}


template <class T>
Vector<size_t> Vector<T>::calculate_Tukey_outliers_iterative(const double& cleaning_parameter) const
{
    const Vector<T> id(1,1,this->size());

    Matrix<T> data(this->size(),2);

    data.set_column(0,id);
    data.set_column(1,(*this));

    Vector<T> id_to_remove;

    for(;;)
    {
        const Vector<T> iteration_id = data.get_column(0);

        const Vector<size_t> iteration_indices = data.get_column(1).calculate_Tukey_outliers(cleaning_parameter);

        if(iteration_indices.empty()) break;

        data = data.delete_rows(iteration_indices);

        id_to_remove = id_to_remove.assemble(iteration_id.get_subvector(iteration_indices));
    }

    const Vector<size_t> tukey_indices = id.calculate_equal_to_indices(id_to_remove).to_size_t_vector();

    return tukey_indices;
}


template <class T>
Vector<size_t> Vector<T>::calculate_histogram_outliers(const size_t& bins_number, const size_t& minimum_frequency) const
{
    Vector<size_t> indices;

    if(this->is_binary()) return indices;

    const Histogram<T> histogram = calculate_histogram(bins_number);

    for(size_t i = 0; i < bins_number; i++)
    {
        if(histogram.frequencies[i] <= minimum_frequency)
        {
            const Vector<size_t> bin_indices = calculate_between_indices(histogram.minimums[i], histogram.maximums[i]);

            indices = indices.assemble(bin_indices);
        }
    }

    return(indices);
}


template <class T>
Vector<size_t> Vector<T>::calculate_histogram_outliers_iterative(const size_t& bins_number, const size_t& minimum_frequency) const
{
    const Vector<T> id(1,1,this->size());

    Matrix<T> data(this->size(),2);

    data.set_column(0,id);
    data.set_column(1,(*this));

    Vector<T> id_to_remove;

    for(;;)
    {
        const Vector<T> iteration_id = data.get_column(0);

        const Vector<size_t> iteration_indices = data.get_column(1).calculate_histogram_outliers(bins_number, minimum_frequency);

        if(iteration_indices.empty()) break;

        data = data.delete_rows(iteration_indices);

        id_to_remove = id_to_remove.assemble(iteration_id.get_subvector(iteration_indices));

    }

    const Vector<size_t> histogram_indices = id.calculate_equal_to_indices(id_to_remove).to_size_t_vector();

    return histogram_indices;
}

/*
template <class T>
Vector<size_t> Vector<T>::calculate_histogram_outliers(const size_t& bins_number, const double& minimum_percentage) const
{
    Vector<size_t> indices;

    if(this->is_binary()) return indices;

    const Histogram<T> histogram = calculate_histogram(bins_number);

    const size_t total = this->size();

    Vector<double> percentages(bins_number);

    for(size_t i  = 0; i < bins_number; i++)
    {
        percentages[i] = static_cast<double>(histogram.frequencies[i] * 100.0) /static_cast<double>(total);
    }

    for(size_t i = 0; i < bins_number; i++)
    {
        if(percentages[i] <= minimum_percentage)
        {
            const Vector<size_t> bin_indices = calculate_between_indices(histogram.minimums[i], histogram.maximums[i]);

            indices = indices.assemble(bin_indices);
        }
    }

    return(indices);
}


template <class T>
Vector<size_t> Vector<T>::calculate_histogram_outliers_iterative(const size_t& bins_number, const double& minimum_percentage) const
{
    const Vector<T> id(1,1,this->size());

    Matrix<T> data(this->size(),2);

    cout << "1" << endl;

    data.set_column(0,id);
    data.set_column(1,(*this));

    Vector<T> id_to_remove;

    cout << "2" << endl;

    for(;;)
    {
        const Vector<T> iteration_id = data.get_column(0);

        const Vector<size_t> iteration_indices = data.get_column(1).calculate_histogram_outliers(bins_number, minimum_percentage);

        if(iteration_indices.empty()) break;

        cout << data.get_rows_number() << "   " << iteration_indices.size() << endl;

        data = data.delete_rows(iteration_indices);

        id_to_remove = id_to_remove.assemble(iteration_id.get_subvector(iteration_indices));
    }

    cout << "3" << endl;

    const Vector<size_t> histogram_indices = id.calculate_equal_to_indices(id_to_remove).to_size_t_vector();

    return histogram_indices;
}
*/

template <class T>
Vector<T> Vector<T>::calculate_scaled_minimum_maximum() const
{
    const double minimum = calculate_minimum();
    const double maximum = calculate_maximum();

    const size_t this_size = this->size();

    Vector<T> new_vector(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
      new_vector[i] = 2.0 *((*this)[i] - minimum) /(maximum - minimum) - 1.0;
    }

    return new_vector;
}


template <class T>
Vector<T> Vector<T>::calculate_scaled_minimum_maximum_0_1() const
{
    const double minimum = calculate_minimum();
    const double maximum = calculate_maximum();

    if(maximum-minimum < numeric_limits<double>::min())
    {
      return (*this);
    }

    const size_t this_size = this->size();

    Vector<T> normalized(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
      normalized[i] = ((*this)[i] - minimum)/(maximum - minimum);
    }

    return(normalized);
}


/// Normalizes the elements of this vector using the minimum and maximum method.
/// @param minimum Minimum value for the scaling.
/// @param maximum Maximum value for the scaling.

template <class T>
void Vector<T>::scale_minimum_maximum(const T &minimum, const T &maximum) {
  if(maximum - minimum < numeric_limits<double>::min()) {
    return;
  }

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = 2.0 *((*this)[i] - minimum) /(maximum - minimum) - 1.0;
  }
}


/// Normalizes the elements of this vector using the minimum and maximum method.
/// @param statistics Statistics structure, which contains the minimum and
/// maximum values for the scaling.

template <class T>
void Vector<T>::scale_minimum_maximum(const Statistics<T> &statistics) {
  scale_minimum_maximum(statistics.minimum, statistics.maximum);
}


/// Normalizes the elements of the vector with the minimum and maximum method.
/// The minimum and maximum values used are those calculated from the vector.
/// It also returns the statistics from the vector.

template <class T> Statistics<T> Vector<T>::scale_minimum_maximum() {
  const Statistics<T> statistics = calculate_statistics();

  scale_minimum_maximum(statistics);

  return(statistics);
}


/// Normalizes the elements of this vector using the mean and standard deviation
/// method.
/// @param mean Mean value for the scaling.
/// @param standard_deviation Standard deviation value for the scaling.

template <class T>
void Vector<T>::scale_mean_standard_deviation(const T &mean,
                                              const T &standard_deviation) {
  if(standard_deviation < numeric_limits<double>::min()) {
    return;
  }

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = ((*this)[i] - mean) / standard_deviation;
  }
}


/// Normalizes the elements of this vector using the mean and standard deviation
/// method.
/// @param statistics Statistics structure,
/// which contains the mean and standard deviation values for the scaling.

template <class T>
void Vector<T>::scale_mean_standard_deviation(const Statistics<T> &statistics) {
  scale_mean_standard_deviation(statistics.mean, statistics.standard_deviation);
}


/// Normalizes the elements of the vector with the mean and standard deviation
/// method.
/// The values used are those calculated from the vector.
/// It also returns the statistics from the vector.

template <class T>
Statistics<T> Vector<T>::scale_mean_standard_deviation() {
  const Statistics<T> statistics = calculate_statistics();

  scale_mean_standard_deviation(statistics);

  return(statistics);
}


/// Normalizes the elements of this vector using standard deviationmethod.
/// @param standard_deviation Standard deviation value for the scaling.

template <class T>
void Vector<T>::scale_standard_deviation(const T &standard_deviation) {
  if(standard_deviation < numeric_limits<double>::min()) {
    return;
  }

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = ((*this)[i]) / standard_deviation;
  }
}


/// Normalizes the elements of this vector using standard deviation method.
/// @param statistics Statistics structure,
/// which contains standard deviation value for the scaling.

template <class T>

void Vector<T>::scale_standard_deviation(const Statistics<T> &statistics) {
  scale_standard_deviation(statistics.standard_deviation);
}


/// Normalizes the elements of the vector with the standard deviation method.
/// The values used are those calculated from the vector.
/// It also returns the statistics from the vector.

template <class T>
Statistics<T> Vector<T>::scale_standard_deviation() {
  const Statistics<T> statistics = calculate_statistics();

  scale_standard_deviation(statistics);

  return(statistics);
}


/// Scales the vector elements with given standard deviation values.
/// It updates the data in the vector.
/// The size of the standard deviation vector must be equal to the
/// size of the vector.
/// @param standard_deviation Standard deviation values.

template <class T>
void
Vector<T>::scale_standard_deviation(const Vector<T> &standard_deviation) {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void scale_standard_deviation(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  // Rescale data

#pragma omp parallel for

  for(int i = 0; i < this_size; i++) {
    if(standard_deviation[i] < numeric_limits<double>::min()) {
//      cout << "OpenNN Warning: Vector class.\n"
//                << "void scale_mean_standard_deviation(const Vector<T>&, const "
//                   "Vector<T>&) method.\n"
//                << "Standard deviation of variable " << i << " is zero.\n"
//                << "Those elements won't be scaled.\n";

      // Do nothing
    } else {
     (*this)[i] = ((*this)[i]) / standard_deviation[i];
    }
  }
}


/// Returns a vector with the scaled elements of this vector acording to the
/// minimum and maximum method.
/// The size of the minimum and maximum vectors must be equal to the size of the
/// vector.
/// @param minimum Minimum values.
/// @param maximum Maximum values.

template <class T>
Vector<T> Vector<T>::calculate_scaled_minimum_maximum(const Vector<T> &minimum,
                                            const Vector<T> &maximum) const {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t minimum_size = minimum.size();

  if(minimum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_scaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_scaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> scaled_minimum_maximum(this_size);

  // Rescale data

  for(size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Vector class.\n"
                << "Vector<T> calculate_scaled_minimum_maximum(const "
                   "Vector<T>&, const Vector<T>&) const method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be scaled.\n";

      scaled_minimum_maximum[i] = (*this)[i];
    } else {
      scaled_minimum_maximum[i] =
          2.0 *((*this)[i] - minimum[i]) /(maximum[i] - minimum[i]) - 1.0;
    }
  }

  return(scaled_minimum_maximum);
}


template <class T>
Vector<T> Vector<T>::calculate_scaled_mean_standard_deviation() const
{
    const double mean = calculate_mean();
    const double standard_deviation = calculate_standard_deviation();

    if(standard_deviation < numeric_limits<double>::min())
    {
      return (*this);
    }

    const size_t this_size = this->size();

    Vector<T> normalized(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        normalized[i] = ((*this)[i] - mean)/standard_deviation;
    }

    return(normalized);
}


/// Returns a vector with the scaled elements of this vector acording to the
/// mean and standard deviation method.
/// The size of the mean and standard deviation vectors must be equal to the
/// size of the vector.
/// @param mean Mean values.
/// @param standard_deviation Standard deviation values.

template <class T>
Vector<T> Vector<T>::calculate_scaled_mean_standard_deviation(
    const Vector<T> &mean, const Vector<T> &standard_deviation) const {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  const size_t mean_size = mean.size();

  if(mean_size != this_size) {
    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_scaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    buffer << "OpenNN Exception: Vector template.\n"
           << "Vector<T> calculate_scaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> scaled_mean_standard_deviation(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Vector template.\n"
                << "Vector<T> calculate_scaled_mean_standard_deviation(const "
                   "Vector<T>&, const Vector<T>&) const method.\n"
                << "Standard deviation of variable " << i << " is zero.\n"
                << "Those elements won't be scaled.\n";

      scaled_mean_standard_deviation = (*this)[i];
    } else {
      scaled_mean_standard_deviation[i] =
         (*this)[i] * standard_deviation[i] + mean[i];
    }
  }

  return(scaled_mean_standard_deviation);
}


/// Returns a vector with the scaled elements of this vector acording to the
/// standard deviation method.
/// The size of the standard deviation vector must be equal to the
/// size of the vector.
/// @param standard_deviation Standard deviation values.

template <class T>
Vector<T> Vector<T>::calculate_scaled_standard_deviation(const Vector<T> &standard_deviation) const {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {

      ostringstream buffer;

    buffer << "OpenNN Exception: Vector template.\n"
           << "Vector<T> calculate_scaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> scaled_standard_deviation(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Vector template.\n"
                << "Vector<T> calculate_scaled_mean_standard_deviation(const "
                   "Vector<T>&, const Vector<T>&) const method.\n"
                << "Standard deviation of variable " << i << " is zero.\n"
                << "Those elements won't be scaled.\n";

      scaled_standard_deviation = (*this)[i];
    } else {
      scaled_standard_deviation[i] =
         (*this)[i] * standard_deviation[i];
    }
  }

  return(scaled_standard_deviation);
}


/// Returns a vector with the unscaled elements of this vector acording to the
/// minimum and maximum method.
/// The size of the minimum and maximum vectors must be equal to the size of the
/// vector.
/// @param minimum Minimum values.
/// @param maximum Maximum values.

template <class T>
Vector<T>
Vector<T>::calculate_unscaled_minimum_maximum(const Vector<T> &minimum,
                                              const Vector<T> &maximum) const {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t minimum_size = minimum.size();

  if(minimum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_unscaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_unscaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> unscaled_minimum_maximum(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Vector template.\n"
                << "Vector<T> calculate_unscaled_minimum_maximum(const "
                   "Vector<T>&, const Vector<T>&) const method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be unscaled.\n";

      unscaled_minimum_maximum[i] = (*this)[i];
    } else {
      unscaled_minimum_maximum[i] =
          0.5 *((*this)[i] + 1.0) *(maximum[i] - minimum[i]) + minimum[i];
    }
  }

  return(unscaled_minimum_maximum);
}


/// Returns a vector with the unscaled elements of this vector acording to the
/// mean and standard deviation method.
/// The size of the mean and standard deviation vectors must be equal to the
/// size of the vector.
/// @param mean Mean values.
/// @param standard_deviation Standard deviation values.

template <class T>
Vector<T> Vector<T>::calculate_unscaled_mean_standard_deviation(
    const Vector<T> &mean, const Vector<T> &standard_deviation) const {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t mean_size = mean.size();

  if(mean_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_unscaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template.\n"
           << "Vector<T> calculate_unscaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> unscaled_mean_standard_deviation(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Vector template.\n"
                << "Vector<T> calculate_unscaled_mean_standard_deviation(const "
                   "Vector<T>&, const Vector<T>&) const method.\n"
                << "Standard deviation of variable " << i << " is zero.\n"
                << "Those elements won't be scaled.\n";

      unscaled_mean_standard_deviation[i] = (*this)[i];
    } else {
      unscaled_mean_standard_deviation[i] =
         (*this)[i] * standard_deviation[i] + mean[i];
    }
  }

  return(unscaled_mean_standard_deviation);
}


/// Unscales the vector elements with given minimum and maximum values.
/// It updates the vector elements.
/// The size of the minimum and maximum vectors must be equal to the size of the
/// vector.
/// @param minimum Minimum values.
/// @param maximum Maximum deviation values.

template <class T>
void Vector<T>::unscale_minimum_maximum(const Vector<T> &minimum,
                                        const Vector<T> &maximum) {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t minimum_size = minimum.size();

  if(minimum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void unscale_minimum_maximum(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void unscale_minimum_maximum(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Vector template.\n"
                << "void unscale_minimum_maximum(const Vector<T>&, const "
                   "Vector<T>&) method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be unscaled.\n";

      // Do nothing
    } else {
     (*this)[i] =
          0.5 *((*this)[i] + 1.0) *(maximum[i] - minimum[i]) + minimum[i];
    }
  }
}


/// Unscales the vector elements with given mean and standard deviation values.
/// It updates the vector elements.
/// The size of the mean and standard deviation vectors must be equal to the
/// size of the vector.
/// @param mean Mean values.
/// @param standard_deviation Standard deviation values.

template <class T>
void Vector<T>::unscale_mean_standard_deviation(
    const Vector<T> &mean, const Vector<T> &standard_deviation) {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t mean_size = mean.size();

  if(mean_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void unscale_mean_standard_deviation(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template.\n"
           << "void unscale_mean_standard_deviation(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Vector template.\n"
                << "void unscale_mean_standard_deviation(const Vector<T>&, "
                   "const Vector<T>&) method.\n"
                << "Standard deviation of variable " << i << " is zero.\n"
                << "Those elements won't be scaled.\n";

      // Do nothing
    } else {
     (*this)[i] = (*this)[i] * standard_deviation[i] + mean[i];
    }
  }
}


template <class T>
Vector<T> Vector<T>::calculate_reverse_scaling(void) const
{
    const size_t this_size = this->size();

    Vector<T> reverse_scaling_vector(this_size);

    reverse_scaling_vector[0] = 1;

    for(size_t i = 1; i < this_size; i++)
    {
        reverse_scaling_vector[i] = ((*this)[this_size-1]-(*this)[i])/((*this)[this_size-1]-(*this)[0]);
    }

    return reverse_scaling_vector;
}


template <class T>
Vector<T> Vector<T>::calculate_scaling_between(const T& min_old, const T& max_old, const T& min_new, const T& max_new) const
{
    const size_t this_size = this->size();

    Vector<T> scaled_vector(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        scaled_vector[i] = min_new + (max_new-min_new) * ((*this)[i]-min_old) / (max_old-min_old);
    }

    return scaled_vector;
}


/// Returns a squared matrix in which the entries outside the main diagonal are
/// all zero.
/// The elements in the diagonal are the elements in this vector.

template <class T> Matrix<T> Vector<T>::to_diagonal_matrix() const
{
  const size_t this_size = this->size();

  Matrix<T> matrix(this_size, this_size, 0.0);

  matrix.set_diagonal(*this);

  return(matrix);
}


template <class T>
Vector<T> Vector<T>::get_subvector(const size_t& first_index, const size_t& last_index) const
{
#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(last_index >= this_size || first_index >= this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> get_subvector(const size_t&, const size_t&) const method.\n"
             << "Index is equal or greater than this size.\n";

      throw logic_error(buffer.str());
  }

#endif

  Vector<T> subvector(last_index-first_index + 1);

  for(size_t i = first_index; i < last_index+1; i++) {
    subvector[i-first_index] = (*this)[i];
  }

  return(subvector);
}


/// Returns another vector whose elements are given by some elements of this
/// vector.
/// @param indices Indices of this vector whose elements are required.

template <class T>
Vector<T> Vector<T>::get_subvector(const Vector<size_t> &indices) const {
  const size_t new_size = indices.size();

  if(new_size == 0) return Vector<T>();


// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  for(size_t i = 0; i < new_size; i++) {
    if(indices[i] > this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> get_subvector(const Vector<T>&) const method.\n"
             << "Index is equal or greater than this size.\n";

      throw logic_error(buffer.str());
    }
  }

#endif

  Vector<T> subvector(new_size);

  for(size_t i = 0; i < new_size; i++) {
    subvector[i] = (*this)[indices[i]];
  }

  return(subvector);
}


template <class T>
Vector<T> Vector<T>::get_subvector(const Vector<bool>& selection) const
{
    const Vector<size_t> indices = selection.calculate_equal_to_indices(true);

    return(get_subvector(indices));
}


template <class T>
Vector<T> Vector<T>::get_subvector_random(const size_t& new_size) const
{
    if(new_size == this->size()) return Vector<T>(*this);

    Vector<T> new_vector(*this);

    random_shuffle(new_vector.begin(), new_vector.end());

    return new_vector.get_first(new_size);
}


/// Returns a vector with the first n elements of this vector.
/// @param elements_number Size of the new vector.

template <class T>
Vector<T>
Vector<T>::get_first(const size_t &elements_number) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(elements_number > this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> get_first(const size_t&) const method.\n"
           << "Number of elements must be equal or greater than this size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> subvector(elements_number);

  for(size_t i = 0; i < elements_number; i++)
  {
    subvector[i] = (*this)[i];
  }

  return(subvector);
}


/// Returns a vector with the last n elements of this vector.
/// @param elements_number Size of the new vector.

template <class T>
Vector<T>
Vector<T>::get_last(const size_t &elements_number) const
{
  const size_t this_size = this->size();

// Control sentence(if debug)
#ifdef __OPENNN_DEBUG__

  if(elements_number > this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> get_last(const size_t&) const method.\n"
           << "Number of elements must be equal or greater than this size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> subvector(elements_number);

  for(size_t i = 0; i < elements_number; i++) {
    subvector[i] = (*this)[i + this_size - elements_number];
  }

  return(subvector);
}


/// Returns a vector with the integers of the vector.
/// @param maximum_integers Maximum number of integers to get.

template <class T> Vector<T> Vector<T>::get_integer_elements(const size_t& maximum_integers) const {

    const size_t this_size = this->size();

    const size_t integers_number = this->count_integers(maximum_integers);

    Vector<T> integers(integers_number);
    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!integers.contains((*this)[i]))
        {
            integers[index] = (*this)[i];
            index++;

            if(index > integers_number)
            {
                break;
            }
        }
    }

    return integers;
}


/// Returns a vector with the integers of the vector.
/// @param missing_indices Indices of the instances with missing values.
/// @param maximum_integers Maximum number of integers to get.

template <class T> Vector<T> Vector<T>::get_integer_elements_missing_values(const Vector<size_t>& missing_indices, const size_t& maximum_integers) const {

    const size_t this_size = this->size();

    const size_t integers_number = this->count_integers_missing_values(missing_indices, maximum_integers);

    Vector<T> integers(integers_number);
    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!missing_indices.contains(i))
        {
            if(!integers.contains((*this)[i]))
            {
                integers[index] = (*this)[i];
                index++;

                if(index > integers_number)
                {
                    break;
                }
            }
        }
    }

    return integers;
}


/// Returns a Matrix in which column i is(this)^i
/// @param order Maximum order.

template <class T>
Matrix<T> Vector<T>::get_power_matrix(const size_t& order) const
{
    Matrix<T> power_matrix(this->size(),order);
    for(size_t i=1; i <=order; i++)
    {
        for(size_t j=0; j<this->size(); j++)
        {
            power_matrix(j,i-1) = pow((*this)[j],i);
        }
    }

    return power_matrix;
}


/// Loads the members of a vector from an data file.
/// Please be careful with the file format, which is specified in the OpenNN
/// manual.
/// @param file_name Name of vector file.

template <class T> void Vector<T>::load(const string &file_name) {
  ifstream file(file_name.c_str());

  stringstream buffer;

  string line;

  while(file.good()) {
    getline(file, line);

    buffer << line;
  }

  istream_iterator<string> it(buffer);
  istream_iterator<string> end;

  const vector<string> results(it, end);

  const size_t new_size = static_cast<size_t>(results.size());

  this->resize(new_size);

  file.clear();
  file.seekg(0, ios::beg);

  // Read data

  for(size_t i = 0; i < new_size; i++) {
    file >>(*this)[i];
  }

  file.close();
}


/// Saves to a data file the elements of the vector.
/// The file format is as follows:
/// element_0 element_1 ... element_N-1
/// @param file_name Name of vector data file.

template <class T> void Vector<T>::save(const string &file_name, const char& separator) const
{
  ofstream file(file_name.c_str());

  if(!file.is_open()) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector template.\n"
           << "void save(const string&) const method.\n"
           << "Cannot open vector data file.\n";

    throw logic_error(buffer.str());
  }

  // Write file

  const size_t this_size = this->size();

  if(this_size > 0) {
    file <<(*this)[0];

    for(size_t i = 1; i < this_size; i++) {
      file << separator <<(*this)[i];
    }

    file << endl;
  }

  // Close file

  file.close();
}


/// Insert another vector starting from a given position.
/// @param position Insertion position.
/// @param other_vector Vector to be inserted.

template <class T>
void Vector<T>::tuck_in(const size_t &position, const Vector<T> &other_vector) {
  const size_t other_size = other_vector.size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(position + other_size > this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void insert(const size_t&, const Vector<T>&) const method.\n"
           << "Cannot tuck in vector.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < other_size; i++) {
   (*this)[position + i] = other_vector[i];
  }
}


/// Returns a new vector with a new element inserted.
/// @param index Position of the new element.
/// @param value Value of the new element.

template <class T>
Vector<T> Vector<T>::insert_element(const size_t &index, const T &value) const
{

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t this_size = this->size();

  if(index > this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "void insert_element(const size_t& index, const T& value) method.\n"
        << "Index is greater than vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> other_vector(*this);

  const auto it = other_vector.begin();

  other_vector.insert(it+index, value);

  return(other_vector);
}


/// Returns a new vector where a given element of this vector is replaced by another vector.
/// @param index Index of the element to be replaced.
/// @param other_vector Replacement vector.

template <class T>
Vector<T> Vector<T>::replace_element(const size_t &index, const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(index > this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "void insert_element(const size_t& index, const T& value) method.\n"
        << "Index is greater than vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t other_size = other_vector.size();
  const size_t new_size = this_size - 1 + other_size;

  Vector<T> new_vector(new_size);

  for(size_t i = 0; i < index; i++)
  {
      new_vector[i] = (*this)[i];
  }

  for(size_t i = index; i < index+other_size; i++)
  {
      new_vector[i] = other_vector[i-index];
  }

  for(size_t i = index+other_size; i < new_size; i++)
  {
      new_vector[i] = (*this)[i+1-other_size];
  }

  return(new_vector);
}


/// Returns a new vector where a given value has been replaced by another one.
/// @param find Value to be replaced.
/// @param replace Replacement value.

template <class T>
Vector<T> Vector<T>::replace_value(const T& find_value, const T& replace_value) const {

  Vector<T> new_vector(*this);

  replace(new_vector.begin(), new_vector.end(), find_value, replace_value);

  return(new_vector);
}


/// Returns a new vector where a given value has been replaced by another one.
/// @param find Value to be found.
/// @param replace Replacement value.

template <class T>
Vector<T> Vector<T>::replace_value_if_contains(const T& find, const T& replace) const {

  Vector<T> new_vector(*this);

  for(size_t i = 0; i < new_vector.size(); i++)
  {
      if((*this)[i].find(find) != string::npos)
      {
           new_vector[i] = replace;
      }
  }

  return(new_vector);
}


/// Splits a string slement into substrings wherever delimiter occurs, and returns the vector of those strings.
/// If sep does not match anywhere in the string, this method returns a single-element vector containing this string.
/// @param index Index of elements.
/// @param delimiter Separator char.

template <class T>
Vector<string> Vector<T>::split_element(const size_t &index, const char &delimiter) const {

   const Vector<string> split(split_string((*this)[index], delimiter));

   return(split);
}


/// Returns a new vector which is a copy of this vector but with a given element
/// removed.
/// Therefore, the size of the new vector is the size of this vector minus one.
/// @param index Index of element to be removed.

template <class T>
Vector<T> Vector<T>::delete_index(const size_t &index) const {
  const size_t this_size = this->size();

// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  if(index >= this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> remove_element(const size_t&) const method.\n"
           << "Index is equal or greater than vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<T> other_vector(this_size - 1);

  for(size_t i = 0; i < this_size; i++) {
    if(i < index)
    {
      other_vector[i] = (*this)[i];
    }
    else if(i > index) {
      other_vector[i - 1] = (*this)[i];
    }
  }

  return(other_vector);
}


/// Returns a new vector which is a copy of this vector but with a given elements
/// removed.
/// Therefore, the size of the new vector is the size of this vector minus the indices vector size.
/// @param indices Vector with the indices of the elements to be removed.

template <class T>
Vector<T> Vector<T>::delete_indices(const Vector<size_t> &indices) const
{
    const size_t this_size = this->size();
    const size_t indices_size = indices.size();

  // Control sentence(if debug)

  #ifdef __OPENNN_DEBUG__

    if(indices.calculate_maximum() >= this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> remove_elements(const Vector<size_t>&) const method.\n"
             << "Maximum index is equal or greater than vector size.\n";

      throw logic_error(buffer.str());
    }

    if(indices_size >= this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> remove_elements(const Vector<size_t>&) const method.\n"
             << "Number of indices to remove is equal or greater than vector size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    Vector<T> other_vector(this_size - indices_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!indices.contains(i))
        {
            other_vector[index] = (*this)[i];

            index++;
        }
    }

    return(other_vector);
}


/// Construct a copy of this vector but without a certain value.
/// Note that the new vector might have a different size than this vector.
/// @param value Value of elements to be removed.

template <class T>
Vector<T> Vector<T>::delete_value(const T &value) const
{
  const size_t this_size = this->size();

  const size_t value_count = count_equal_to(value);

  if(value_count == 0) return Vector<T>(*this);

    const size_t other_size = this_size - value_count;

    Vector<T> other_vector(other_size);

    size_t other_index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
      if((*this)[i] != value)
      {
        other_vector[other_index] = (*this)[i];

        other_index++;
      }
    }

    return other_vector;
}


template <class T>
Vector<T> Vector<T>::delete_values(const Vector<T> &values) const
{
    Vector<T> new_vector(*this);

    for(size_t i = 0; i < values.size(); i++)
    {
        new_vector = new_vector.delete_value(values[i]);
    }

    return(new_vector);
}


/// Assemble two vectors.
/// @param other_vector Vector to be get_assemblyd to this vector.

template <class T>
Vector<T> Vector<T>::assemble(const Vector<T> &other_vector) const  {
  const size_t this_size = this->size();
  const size_t other_size = other_vector.size();

  if(this_size == 0 && other_size == 0) {
    Vector<T> assembly;

    return(assembly);
  } else if(this_size == 0) {
    return(other_vector);
  } else if(other_size == 0) {
    return(*this);
  } else {
    Vector<T> assembly(this_size + other_size);

    for(size_t i = 0; i < this_size; i++) {
      assembly[i] = (*this)[i];
    }

    for(size_t i = 0; i < other_size; i++) {
      assembly[this_size + i] = other_vector[i];
    }

    return(assembly);
  }
}


/// Assemble a vector of vectors to this vector.
/// @param vectors Vector of vectors to be get_assembled to this vector.

template <class T>
Vector<T> Vector<T>::assemble(const Vector< Vector<T> >& vectors)
{
    const size_t vectors_size = vectors.size();

    size_t new_size = 0;

    for(size_t i = 0; i < vectors_size; i++)
    {
        new_size += vectors[i].size();
    }

    Vector<T> new_vector(new_size);

    size_t index = 0;

    for(size_t i = 0; i < vectors_size; i++)
    {
        for(size_t j = 0; j < vectors[i].size(); j++)
        {
            new_vector[index] = vectors[i][j];
            index++;
        }
    }

    return(new_vector);
}


/// Returns a vector which is the difference of this vector and another vector.
/// For instance, if this vector is(1,2,3) and the other vector is(1,4,3,3),
/// the difference is(2), the element in the first vector which is not present in the second.
/// @param other_vector Other vector.

template <class T>
Vector<T> Vector<T>::get_difference(const Vector<T> &other_vector) const
{
    if(this->empty())
    {
        return other_vector;
    }

    if(other_vector.empty())
    {
        Vector<T> copy(*this);
        return copy;
    }

    const size_t this_size = this->size();

    Vector<T> difference(this_size);
    typename vector<T>::iterator iterator;

    Vector<T> copy_this(*this);
    Vector<T> copy_other_vector(other_vector);

    sort(copy_this.begin(), copy_this.end());
    sort(copy_other_vector.begin(), copy_other_vector.end());

    iterator = set_difference(copy_this.begin(),copy_this.end(), copy_other_vector.begin(), copy_other_vector.end(), difference.begin());

    difference.resize(iterator - difference.begin());

    return(difference);
}


/// Returns a vector which is the union of this vector and another vector.
/// For instance, if this vector is(1,2,3) and the other vector is(2,3,4),
/// the union is(2,3), the elements that are present in the two vectors.
/// @param other_vector Other vector.

template <class T>
Vector<T> Vector<T>::get_union(const Vector<T>& other_vector) const
{
    Vector<T> this_copy(*this);
    sort(this_copy.begin(), this_copy.end());

    Vector<T> other_copy(other_vector);
    sort(other_copy.begin(), other_copy.end());

    Vector<T> union_vector;

    set_union(this_copy.begin(), this_copy.end(), other_copy.begin(), other_copy.end(), back_inserter(union_vector));

    return union_vector;
}


/// Returns a vector with the intersection of this vector and another vector.
/// For instance, if this vector is(a, b, c) and the other vector is(b, c, d, e),
/// the new vector will be(b, c).
/// @param other_vector Other vector.

template <class T>
Vector<T> Vector<T>::get_intersection(const Vector<T>& other_vector) const
{
    Vector<T> this_copy(*this);
    sort(this_copy.begin(), this_copy.end());

    Vector<T> other_copy(other_vector);
    sort(other_copy.begin(), other_copy.end());

    Vector<T> intersection;

    set_intersection(this_copy.begin(), this_copy.end(), other_copy.begin(), other_copy.end(), back_inserter(intersection));

    return intersection;
}


/// Returns a vector with the unique items of the vector.
/// For instance, if the vector is("a,b", "b,c", "c,d"), the new vector will be(a, b, c, d).

template <class T>
Vector<T> Vector<T>::get_unique_items(const char& separator) const
{
    const Vector<T> unique_mixes = get_unique_elements();

    Vector< Vector<T> > items(unique_mixes.size());

    Vector<T> unique_items;

    for(int i = 0; i < unique_mixes.size(); i++)
    {
        items[i] = unique_mixes.split_element(i, separator);

        unique_items = unique_items.assemble(items[i]).get_unique_elements();
    }

    return unique_items;
}


/// Returns a vector with the unique values of the vector.
/// For instance, if the vector is(a, b, a), the new vector will be(a, b).

template <class T>
Vector<T> Vector<T>::get_unique_elements() const
{
    Vector<T> copy_vector(*this);

    sort(copy_vector.begin(), copy_vector.end());

    const auto last = unique(copy_vector.begin(), copy_vector.end());

    copy_vector.erase(last, copy_vector.end());

    return(copy_vector);
}


template <class T>
Vector<T> Vector<T>::get_unique_elements_unsorted() const
{
    Vector<T> copy_vector(*this);

    const auto last = unique(copy_vector.begin(), copy_vector.end());

    copy_vector.erase(last, copy_vector.end());

    return(copy_vector);
}


/// Returns a vector with the indices of the unique elements, in the order given by the get_unique_elements method.
/// For instance, if the input vector is(a, b, a), the output vector is(0, 1).

template <class T>
Vector<size_t> Vector<T>::get_unique_elements_first_indices() const
{
    const Vector<T> unique_values = get_unique_elements();

    const size_t unique_size = unique_values.size();

    Vector<size_t> unique_indices(unique_size);

    for(size_t i = 0; i < unique_size; i++)
    {
        unique_indices[i] = get_first_index(unique_values[i]);
    }

    return(unique_indices);
}

template <class T>
Vector< Vector<size_t> > Vector<T>::get_unique_elements_indices() const
{
    const Vector<T> unique_elements = this->get_unique_elements();

    const int unique_elements_size = static_cast<int>(unique_elements.size());

    Vector< Vector<size_t> > unique_leads_indices(static_cast<size_t>(unique_elements_size));

    #pragma omp parallel for

    for (int i = 0; i < unique_elements_size; i++)
    {
        unique_leads_indices[static_cast<size_t>(i)] = this->calculate_equal_to_indices(unique_elements[static_cast<size_t>(i)]);
    }

    return unique_leads_indices;
}


/// Returns a vector with the numbers of the unique elements, in the order given by the get_unique_elements method.
/// For instance, if the input vector is(a, b, a), the output vector is(2, 1).

template <class T>
Vector<size_t> Vector<T>::count_unique() const
{
    const Vector<T> unique = get_unique_elements();

    const size_t unique_size = unique.size();

    Vector<size_t> unique_count(unique_size);

#pragma omp parallel for

    for(int i = 0; i < static_cast<int>(unique_size); i++)
    {
        unique_count[i] = count_equal_to(unique[i]);
    }

    return(unique_count);
}


/// Prints to the screen the unique elements of the vector, the number of that elements and the corresponding percentage.
/// It sorts the elements from greater to smaller.
/// For instance, for the vector(a, b, a), it will print
/// a: 2(66.6%), b: 1(33.3%).

template <class T>
void Vector<T>::print_unique() const
{
    const size_t this_size = this->size();

    const Vector<T> unique = get_unique_elements();

    const Vector<size_t> count = count_unique();

    const Vector<double> percentage = count_unique().to_double_vector()*(100.0/static_cast<double>(this_size));

    const size_t unique_size = unique.size();

    Matrix<T> unique_matrix(unique_size, 3);
    unique_matrix.set_column(0, unique.to_string_vector());
    unique_matrix.set_column(1, count.to_string_vector());
    unique_matrix.set_column(2, percentage.to_string_vector());

    unique_matrix = unique_matrix.sort_descending_strings(1);

    cout << "Total: " << this_size << endl;

    for(size_t i = 0; i < unique_size; i++)
    {
        cout << unique_matrix(i,0) << ": " << unique_matrix(i,1) << "(" <<  unique_matrix(i,2) << "%)" << endl;
    }
}


template <class T>
Vector<T> Vector<T>::calculate_top(const size_t& rank) const
{
    const Vector<T> unique = get_unique_elements();

    const Vector<size_t> count = count_unique();

    const size_t unique_size = unique.size();

    Matrix<T> unique_matrix(unique_size, 2);
    unique_matrix.set_column(0, unique.to_string_vector());
    unique_matrix.set_column(1, count.to_string_vector());

    unique_matrix = unique_matrix.sort_descending_strings(1);

    const size_t end = unique_size < rank ? unique_size : rank;

    const Vector<T> top = unique_matrix.get_column(0).get_first(end);

   return(top);
}


template <class T>
Matrix<T> Vector<T>::calculate_top_matrix(const size_t& rank) const
{
    const Vector<T> unique = get_unique_elements();

    const Vector<size_t> count = count_unique();

    const size_t this_size = this->size();

    const Vector<double> percentage = count_unique().to_double_vector()*(100.0/static_cast<double>(this_size));

    const size_t unique_size = unique.size();

    Matrix<T> unique_matrix(unique_size, 3);
    unique_matrix.set_column(0, unique.to_string_vector());
    unique_matrix.set_column(1, count.to_string_vector());
    unique_matrix.set_column(2, percentage.to_string_vector());

    unique_matrix = unique_matrix.sort_descending_strings(1);

    const Vector<size_t> indices(0,1,rank-1);

    if(rank < unique_size)
    {
        unique_matrix = unique_matrix.get_submatrix_rows(indices);

        return(unique_matrix);
    }
    else
    {
        return(unique_matrix);
    }
}


template <class T>
Matrix<T> Vector<T>::calculate_top_matrix_over(const size_t& rank, const size_t& new_total) const
{
    const Vector<T> unique = get_unique_elements();

    const Vector<size_t> count = count_unique();

    const Vector<double> percentage = count_unique().to_double_vector()*(100.0/static_cast<double>(new_total));

    const size_t unique_size = unique.size();

    Matrix<T> unique_matrix(unique_size, 3);
    unique_matrix.set_column(0, unique.to_string_vector());
    unique_matrix.set_column(1, count.to_string_vector());
    unique_matrix.set_column(2, percentage.to_string_vector());

    unique_matrix = unique_matrix.sort_descending_strings(1);

    const Vector<size_t> indices(0,1,rank-1);

    if(rank < unique_size)
    {
        unique_matrix = unique_matrix.get_submatrix_rows(indices);

        return(unique_matrix);
    }
    else
    {
        return(unique_matrix);
    }
}


/// Prints to the screen the unique elements of the vector, the number of that elements and the corresponding percentage.
/// It sorts the elements from greater to smaller and only prints the top ones.
/// @param rank Number of top elements that are printed.

template <class T>
void Vector<T>::print_top(const size_t& rank) const
{
    const Vector<T> unique = get_unique_elements();

    const Vector<size_t> count = count_unique();

    const size_t this_size = this->size();

    const Vector<double> percentage = count_unique().to_double_vector()*(100.0/static_cast<double>(this_size));

    const size_t unique_size = unique.size();

    if(unique_size == 0) return;

    Matrix<T> unique_matrix(unique_size, 3);
    unique_matrix.set_column(0, unique.to_string_vector());
    unique_matrix.set_column(1, count.to_string_vector());
    unique_matrix.set_column(2, percentage.to_string_vector());

    unique_matrix = unique_matrix.sort_descending_strings(1);

    const size_t end = unique_size < rank ? unique_size : rank;

    for(size_t i = 0; i < end; i++)
    {
        cout << i+1 << ". " << unique_matrix(i,0) << ": " << unique_matrix(i,1) << "(" <<  unique_matrix(i,2) << "%)" << endl;
    }
}


/// Returns a std vector with the size and elements of this OpenNN vector.

template <class T> vector<T> Vector<T>::to_std_vector() const {
  const size_t this_size = this->size();

  vector<T> std_vector(this_size);

  for(size_t i = 0; i < this_size; i++) {
    std_vector[i] = (*this)[i];
  }

  return(std_vector);
}


/// Returns a new vector with the elements of this vector casted to double.

template <class T> Vector<double> Vector<T>::to_double_vector() const
{
  const size_t this_size = this->size();

  Vector<double> double_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      double_vector[i] = static_cast<double>((*this)[i]);
  }

  return(double_vector);
}


/// Returns a new vector with the elements of this vector casted to int.

template <class T>
Vector<int> Vector<T>::to_int_vector() const {
  const size_t this_size = this->size();

  Vector<int> int_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      int_vector[i] = static_cast<int>((*this)[i]);
   }

  return(int_vector);
}


/// Returns a new vector with the elements of this vector casted to size_t.

template <class T>
Vector<size_t> Vector<T>::to_size_t_vector() const {
  const size_t this_size = this->size();

  Vector<size_t> size_t_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      size_t_vector[i] = static_cast<size_t>((*this)[i]);
   }

  return(size_t_vector);
}


/// Returns a new vector with the elements of this vector casted to time_t.

template <class T>
Vector<time_t> Vector<T>::to_time_t_vector() const {
  const size_t this_size = this->size();

  Vector<time_t> size_t_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      size_t_vector[i] = static_cast<time_t>((*this)[i]);
   }

  return(size_t_vector);
}


template <class T>
Vector<bool> Vector<T>::to_bool_vector() const
{
  const Vector<T> unique = get_unique_elements();

  if(unique.size() != 2)
  {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector Template.\n"
              << "Vector<bool> to_bool_vector() const.\n"
              << "Number of unique items(" << get_unique_elements().size() << ") must be 2.\n";

       throw logic_error(buffer.str());
  }

  const size_t this_size = this->size();

  Vector<bool> new_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      if((*this)[i] == unique[0])
      {
          new_vector[i] = true;
      }
      else
      {
          new_vector[i] = false;
      }
  }

  return(new_vector);
}


/// Returns a new vector with the elements of this vector converted to string.

template <class T> Vector<string> Vector<T>::to_string_vector() const {
  const size_t this_size = this->size();

  Vector<string> string_vector(this_size);
  ostringstream buffer;

    for(size_t i = 0; i < this_size; i++)
    {
      buffer.str("");
      buffer <<(*this)[i];

      string_vector[i] = buffer.str();
   }

  return(string_vector);
}


/// Returns a new vector with the elements of this vector casted to double.

template <class T> Vector<double> Vector<T>::string_to_double(const double& exception_value) const
{
  const size_t this_size = this->size();

  Vector<double> double_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      try
      {
          double_vector[i] = stod((*this)[i]);
      }
      catch(const logic_error&)
      {
         double_vector[i] = exception_value;
      }
   }

  return(double_vector);
}


/// Returns a new vector with the elements of this string vector converted to double.

template <class T> Vector<int> Vector<T>::string_to_int(const int& exception_value) const {
  const size_t this_size = this->size();

  Vector<int> int_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      try
      {
          int_vector[i] = stoi((*this)[i]);
      }
      catch(const logic_error&)
      {
         int_vector[i] = exception_value;
      }
   }

  return(int_vector);
}


/// Returns a new vector with the elements of this string vector converted to size_t.

template <class T> Vector<size_t> Vector<T>::string_to_size_t(const size_t& exception_value) const {
  const size_t this_size = this->size();

  Vector<size_t> size_t_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      try
      {
          size_t_vector[i] = static_cast<size_t>(stoi((*this)[i]));
      }
      catch(const logic_error&)
      {
         size_t_vector[i] = exception_value;
      }
   }

  return(size_t_vector);
}


/// Returns a new vector with the elements of this string vector converted to time_t.

template <class T> Vector<time_t> Vector<T>::string_to_time_t(const time_t& exception_value) const {
  const size_t this_size = this->size();

  Vector<time_t> time_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      try
      {
          time_vector[i] = static_cast<time_t>(stoi((*this)[i]));
      }
      catch(const logic_error&)
      {
         time_vector[i] = exception_value;
      }
   }

  return(time_vector);
}


/// Takes a string vector representing a date in the format Mon Jan 30 2017 12:52:24 and returns a timestamp vector.

template <class T>
Vector<time_t> Vector<T>::www_mmm_ddd_yyyy_hh_mm_ss_to_timestamp() const
{
  const size_t this_size = this->size();

  Vector<time_t> time_vector(this_size);

  //Mon Jan 30 2017 12:52:24

  vector<string> date_elements;
  vector<string> time_elements;

  int year;
  int month;
  int month_day;
  int hours;
  int minutes;
  int seconds;

  for(size_t i = 0; i < this_size; i++)
  {
      date_elements = split_string((*this)[i], ' ');

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

      struct tm  timeinfo;
      timeinfo.tm_year = year - 1900;
      timeinfo.tm_mon = month - 1;
      timeinfo.tm_mday = month_day;

      timeinfo.tm_hour = hours;
      timeinfo.tm_min = minutes;
      timeinfo.tm_sec = seconds;

      time_vector[i] = mktime(&timeinfo);
   }

  return(time_vector);
}


template <class T>
Vector<time_t> Vector<T>::yyyy_mm_to_timestamp(const char& delimiter) const
{
  const size_t this_size = this->size();

  Vector<time_t> time(this_size);

  vector<string> date_elements;

  int mm;
  int yyyy;

  for(size_t i = 0; i < this_size; i++)
  {
      date_elements = split_string((*this)[i], delimiter);

      if(date_elements.size() == 0)
      {
          time[i] = 0;
      }
      else if(date_elements.size() != 2)
      {
          ostringstream buffer;

          buffer << "OpenNN Exception: Vector Template.\n"
                 << "Vector<time_t> mm_yyyy_to_time() const method.\n"
                 << "Element " << i << " has a wrong format: \"" <<(*this)[i] << "\"" << endl;

          throw logic_error(buffer.str());
      }
      else
      {
          // Month

          mm = stoi(date_elements[1]);

          // Year

          yyyy = stoi(date_elements[0]);

          struct tm time_info;

          time_info.tm_year = yyyy - 1900;
          time_info.tm_mon = mm - 1;
          time_info.tm_mday = 1;

          time[i] = mktime(&time_info) + 3600*24;

          if(time[i] == static_cast<time_t>(-1))
          {
              ostringstream buffer;

              buffer << "OpenNN Exception: Vector Template.\n"
                     << "Vector<time_t> dd_mm_yyyy_to_time() const method.\n"
                     << "Element " << i << " can not be converted to time_t: \"" <<(*this)[i] << "\"" << endl;

              throw logic_error(buffer.str());
          }
      }
   }

  return(time);
}


template <class T>
Vector<time_t> Vector<T>::yyyy_mm_dd_hh_mm_ss_to_timestamp(const char& delimiter) const
{
  const size_t this_size = this->size();

  Vector<time_t> time(this_size);

  vector<string> date_elements;

  int yyyy;
  int MM;
  int dd;
  int hh;
  int mm;
  int ss;

  for(size_t i = 0; i < this_size; i++)
  {
      date_elements = split_string((*this)[i], delimiter);

      if(date_elements.size() == 0)
      {
          time[i] = 0;
      }
      else if(date_elements.size() != 6)
      {
          ostringstream buffer;

          buffer << "OpenNN Exception: Vector Template.\n"
                 << "Vector<time_t> yyyy_mm_dd_hh_mm_ss_to_timestamp() const method.\n"
                 << "Element " << i << " has a wrong format: \"" <<(*this)[i] << "\"" << endl;

          throw logic_error(buffer.str());
      }
      else
      {
          yyyy = stoi(date_elements[0]);
          MM = stoi(date_elements[1]);
          dd = stoi(date_elements[2]);
          hh = stoi(date_elements[3]);
          mm = stoi(date_elements[4]);
          ss = stoi(date_elements[5]);

          struct tm time_info;

          time_info.tm_year = yyyy - 1900;
          time_info.tm_mon = MM - 1;
          time_info.tm_mday = dd;
          time_info.tm_hour = hh;
          time_info.tm_min = mm;
          time_info.tm_sec = ss;

          time[i] = mktime(&time_info) + 3600*24;

          if(time[i] == static_cast<time_t>(-1))
          {
              ostringstream buffer;

              buffer << "OpenNN Exception: Vector Template.\n"
                     << "Vector<time_t> dd_mm_yyyy_to_time() const method.\n"
                     << "Element " << i << " can not be converted to time_t: \"" <<(*this)[i] << "\"" << endl;

              throw logic_error(buffer.str());
          }
      }
   }

  return(time);
}



/// Takes a string vector representing a date with day, month and year and returns a timestamp vector.
/// For instance, 21/12/2017 or 21-12-2017 to 1513814400.
/// @param delimiter Char between year, month and day(/, -, etc).

template <class T>
Vector<time_t> Vector<T>::dd_mm_yyyy_to_timestamp(const char& delimiter) const
{
  const size_t this_size = this->size();

  Vector<time_t> time(this_size);

  vector<string> date_elements;

  int dd;
  int mm;
  int yyyy;

  for(size_t i = 0; i < this_size; i++)
  {
      date_elements = split_string((*this)[i], delimiter);

      if(date_elements.size() == 0)
      {
          time[i] = 0;
      }
      else if(date_elements.size() != 3)
      {
          ostringstream buffer;

          buffer << "OpenNN Exception: Vector Template.\n"
                 << "Vector<time_t> dd_mm_yyyy_to_time() const method.\n"
                 << "Element " << i << " has a wrong format: \"" <<(*this)[i] << "\"" << endl;

          throw logic_error(buffer.str());
      }
      else
      {
          // Month day

          dd = stoi(date_elements[0]);

          // Month

          mm = stoi(date_elements[1]);

          // Year

          yyyy = stoi(date_elements[2]);

          struct tm time_info;

          time_info.tm_year = yyyy - 1900;
          time_info.tm_mon = mm - 1;
          time_info.tm_mday = dd;

          time[i] = mktime(&time_info) + 3600*24;

          if(time[i] == static_cast<time_t>(-1))
          {
              ostringstream buffer;

              buffer << "OpenNN Exception: Vector Template.\n"
                     << "Vector<time_t> dd_mm_yyyy_to_time() const method.\n"
                     << "Element " << i << " can not be converted to time_t: \"" <<(*this)[i] << "\"" << endl;

              throw logic_error(buffer.str());
          }
      }
   }

  return(time);
}


/// Takes a string vector representing a date with year, month and day and returns a timestamp vector.
/// For instance, 2017/12/21 or 2017-12-21 to 1513814400.
/// @param delimiter Char between year, month and day(/, -, etc).

template <class T>
Vector<time_t> Vector<T>::yyyy_mm_dd_to_timestamp(const char& delimiter) const
{
  const size_t this_size = this->size();

  Vector<time_t> time(this_size);

   #pragma omp parallel for

  for(int i = 0; i < this_size; i++)
  {
      const vector<string> date_elements = split_string((*this)[i], delimiter);

      if(date_elements.size() != 3)
      {
          ostringstream buffer;

          buffer << "OpenNN Exception: Vector Template.\n"
                 << "Vector<time_t> yyyy_mm_dd_to_time(cont char&) const method.\n"
                 << "Date elements of row " << i << " must be 3: \"" <<(*this)[i] << "\"" << endl;

          throw logic_error(buffer.str());
      }

      // Year

      const int yyyy = stoi(date_elements[0]);

      // Month

      const int mm = stoi(date_elements[1]);

      // Month day

      const int dd = stoi(date_elements[2]);

      struct tm time_info;

      time_info.tm_year = yyyy - 1900;
      time_info.tm_mon = mm - 1;
      time_info.tm_mday = dd;

      time[static_cast<size_t>(i)] = mktime(&time_info);
   }

  return time;
}


/// Takes a string vector representing a date with day, month and year
/// and returns a string vector representing a date with day of the year and year.
/// For instance, 02/10/2017 to 41/2017.
/// @param delimiter Char between day, month and year(/, -, etc).

template <class T>
Matrix<T> Vector<T>::dd_mm_yyyy_to_dd_yyyy(const char& delimiter) const
{
    const size_t this_size = this->size();

    const Vector<time_t> time = this->dd_mm_yyyy_to_time(delimiter);

    Matrix<T> output(this_size,2);

    struct tm* date_info;

    for(size_t i = 0; i < this_size; i++)
    {
        date_info = gmtime(&time[i]);

        output(i, 0) = to_string(date_info->tm_yday + 1);
        output(i, 1) = to_string(date_info->tm_year + 1900);
    }

    return(output);
}


/// Takes a string vector representing a date with year, month and day
/// and returns a string vector representing a date with day of the year and year.
/// For instance, 02/10/2017 to 41/2017.
/// @param delimiter Char between year, month and day(/, -, etc).

template <class T>
Matrix<T> Vector<T>::yyyy_mm_dd_to_dd_yyyy(const char& delimiter) const
{
    const size_t this_size = this->size();

    const Vector<time_t> time = this->yyyy_mm_dd_to_time(delimiter);

    Matrix<T> output(this_size,2);

    #pragma omp parallel for

    for(int i = 0; i < this_size; i++)
    {
        struct tm* date_info = gmtime(&time[static_cast<size_t>(i)]);

        output(i, 0) = to_string(date_info->tm_yday + 1);
        output(i, 1) = to_string(date_info->tm_year + 1900);
    }

    return(output);
}


template <class T>
Matrix<T> Vector<T>::mm_yyyy_to_mm_yyyy(const char& delimiter) const
{
    const size_t this_size = this->size();

    const Vector<time_t> time = this->mm_yyyy_to_time(delimiter);

    Matrix<T> output(this_size,2);

    #pragma omp parallel for

    for(size_t i = 0; i < this_size; i++)
    {
        struct tm* date_info = gmtime(&time[i]);

        output(i, 0) = to_string(date_info->tm_yday + 1);
        output(i, 1) = to_string(date_info->tm_year + 1900);
    }

    return(output);
}


/// Takes a string vector representing a date with day, month and year
/// and returns a string vector with the corresponding weekday.
/// For instance, 2017/12/21 to 5.
/// @param delimiter Char between year, month and day(/, -, etc).

template <class T>
Vector<T> Vector<T>::yyyy_mm_dd_to_weekday(const char& delimiter) const
{
    const size_t this_size = this->size();

    const Vector<time_t> time = yyyy_mm_dd_to_time(delimiter);

    Vector<T> output(this_size);

    #pragma omp parallel for

    for(int i = 0; i < this_size; i++)
    {
        struct tm* date_info = gmtime(&time[static_cast<size_t>(i)]);

        output[i] = to_string(date_info->tm_wday + 1);
    }

    return(output);
}


/// Takes a string vector representing a date with year, month and day
/// and returns a string vector with the corresponding day of the year.
/// For instance, 2017/02/10 to 41.
/// @param delimiter Char between year, month and day(/, -, etc).

template <class T>
Vector<T> Vector<T>::yyyy_mm_dd_to_yearday(const char& delimiter) const
{
    const size_t this_size = this->size();

    const Vector<time_t> time = this->yyyy_mm_dd_to_time(delimiter);

    Vector<T> output(this_size);

    #pragma omp parallel for

    for(int i = 0; i < this_size; i++)
    {
        struct tm* date_info = gmtime(&time[static_cast<size_t>(i)]);

        output[i] = to_string(date_info->tm_yday + 1);
    }

    return(output);
}


template <class T>
Vector<struct tm> Vector<T>::timestamp_to_time_structure() const
{
    const size_t this_size = this->size();

    Vector<struct tm> new_vector(this_size);

    time_t timestamp;
    struct tm time_stucture;

    for(size_t i = 0; i < this_size; i++)
    {
        timestamp = (*this)[i];

        time_stucture = *gmtime(&timestamp);

        new_vector[i] = time_stucture;
    }

    return(new_vector);
}


template <class T>
Vector<size_t> Vector<T>::timestamp_to_yearday() const
{
    const size_t this_size = this->size();

    Vector<size_t> yearday(this_size);

    time_t timestamp;
    struct tm time_stucture;

    for(size_t i = 0; i < this_size; i++)
    {
        timestamp = (*this)[i];

        time_stucture = *gmtime(&timestamp);

        yearday[i] = time_stucture.tm_yday;
    }

    return yearday;
}


template <class T>
Vector< Vector<T> > Vector<T>::split(const size_t& n) const
{
    // determine number of sub-vectors of size n

    const size_t batches_number = (this->size() - 1) / n + 1;

    // create array of vectors to store the sub-vectors

    Vector< Vector<T> > batches(batches_number);

    // each iteration of this loop process next set of n elements
    // and store it in a vector at k'th index in vec

    for (size_t k = 0; k < batches_number; ++k)
    {
        // get range for next set of n elements

        auto start_itr = std::next(this->cbegin(), k*n);
        auto end_itr = std::next(this->cbegin(), k*n + n);

        // allocate memory for the sub-vector
        batches[k].resize(n);

        // code to handle the last sub-vector as it might
        // contain less elements

        if (k*n + n > this->size())
        {
            end_itr = this->cend();
            batches[k].resize(this->size() - k*n);
        }

        // copy elements from the input range to the sub-vector
        std::copy(start_itr, end_itr, batches[k].begin());
    }

    return batches;
}


/// Returns a row matrix with number of rows equal to one
/// and number of columns equal to the size of this vector.

template <class T>
Matrix<T> Vector<T>::to_row_matrix() const {
  const size_t this_size = this->size();

  Matrix<T> matrix(1, this_size);

  for(size_t i = 0; i < this_size; i++) {
    matrix(0, i) = (*this)[i];
  }

  return(matrix);
}


/// Returns a column matrix with number of rows equal to the size of this vector
/// and number of columns equal to one.

template <class T> Matrix<T> Vector<T>::to_column_matrix() const {
  const size_t this_size = this->size();

  Matrix<T> matrix(this_size, 1);

  for(size_t i = 0; i < this_size; i++) {
    matrix(i, 0) = (*this)[i];
  }

  return(matrix);
}


/// This method takes a string representation of a vector and sets this vector
/// to have size equal to the number of words and values equal to that words.
/// @param str String to be parsed.

template <class T> void Vector<T>::parse(const string &str) {
  if(str.empty()) {
    set();
  } else {

    istringstream buffer(str);

    istream_iterator<string> first(buffer);
    istream_iterator<string> last;

    Vector<string> str_vector(first, last);

    const size_t new_size = str_vector.size();

    if(new_size > 0) {
      this->resize(new_size);

      buffer.clear();
      buffer.seekg(0, ios::beg);

      for(size_t i = 0; i < new_size; i++) {
        buffer >>(*this)[i];
      }
    }

    }
}


/// Returns a string representation of this vector.
/// @param separator Char between the elements(, -, /, etc).
/// @param quotation Quotation char for the elements(", ').

template <class T>
string Vector<T>::vector_to_string(const char& separator, const char& quotation) const {
    ostringstream buffer;

    const size_t this_size = this->size();

    if(this_size > 0) {

        buffer << quotation <<(*this)[0] << quotation;

        for(size_t i = 1; i < this_size; i++) {

            buffer << separator << quotation <<(*this)[i] << quotation;
        }
    }

    return(buffer.str());
}


/// Returns a string representation of this vector.
/// @param separator Char between the elements(, -, /, etc).

template <class T>
string Vector<T>::vector_to_string(const char& separator) const {
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0) {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++) {
      buffer << separator <<(*this)[i];
    }
  }

  return(buffer.str());
}


/// Returns a string representation of this vector.

template <class T>
string Vector<T>::vector_to_string() const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++)
    {
        buffer << ' ' <<(*this)[i];
    }
  }

  return(buffer.str());
}


template <class T>
string Vector<T>::stack_vector_to_string() const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++)
    {
        buffer <<(*this)[i];
    }
  }

  return(buffer.str());
}


/// Returns a string representation of this vector which can be inserted in a text.

template <class T> string Vector<T>::to_text(const char& separator) const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++) {
      buffer << separator <<(*this)[i];
    }
  }

  return(buffer.str());
}


template <class T> string Vector<T>::to_text(const string& separator) const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++) {
      buffer << separator <<(*this)[i];
    }
  }

  return(buffer.str());
}


/// This method retuns a vector of strings with size equal to the size of this
/// vector and elements equal to string representations of the elements of this
/// vector.

template <class T>
Vector<string>
Vector<T>::write_string_vector(const size_t &precision) const {
  const size_t this_size = this->size();

  Vector<string> string_vector(this_size);

  ostringstream buffer;

  for(size_t i = 0; i < this_size; i++) {
    buffer.str("");
    buffer << setprecision(precision) <<(*this)[i];

    string_vector[i] = buffer.str();
  }

  return(string_vector);
}


/// Returns a matrix with given numbers of rows and columns and with the
/// elements of this vector ordered by rows.
/// The number of rows multiplied by the number of columns must be equal to the
/// size of this vector.
/// @param rows_number Number of rows in the new matrix.
/// @param columns_number Number of columns in the new matrix.

template <class T>
Matrix<T> Vector<T>::to_matrix(const size_t &rows_number,
                               const size_t &columns_number) const {
// Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(rows_number * columns_number != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Matrix<T> to_matrix(const size_t&, const size_t&) method.\n"
           << "The number of rows(" << rows_number
           << ") times the number of colums(" << columns_number
           << ") must be equal to the size of the vector(" << this_size
           << ").\n";

    throw logic_error(buffer.str());
  }

#endif

  Matrix<T> matrix(rows_number, columns_number);

  for(size_t i = 0; i < this->size(); i++)
  {
      matrix[i] = (*this)[i];
  }

  return(matrix);
}


template <class T>
double Vector<T>::calculate_logistic_function(const Vector<double>& coefficients, const Vector<T>& x) const
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
Vector<double> Vector<T>::calculate_logistic_error_gradient(const Vector<double>& coefficients, const Vector<T>& other) const
{
    const size_t n = this->size();

    const size_t other_size = this->size();

    Vector<double> error_gradient(3, 0.0);

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
        Vector<double> x(1);

        x[0] = (*this)[i];

        double current_logistic_function = calculate_logistic_function(coefficients, x);

        const double gradient_multiply = exp(-(coefficients[0]+coefficients[1]*x[0]))*(other[i] - current_logistic_function)*current_logistic_function*current_logistic_function;

        Vector<double> this_error_gradient(3, 0.0);

        this_error_gradient[0] += (other[i]*positives_weight + (1-other[i])*negatives_weight)*(other[i] - current_logistic_function)*(other[i] - current_logistic_function)/2;
        this_error_gradient[1] -= (other[i]*positives_weight + (1-other[i])*negatives_weight)*gradient_multiply;
        this_error_gradient[2] -= (other[i]*positives_weight + (1-other[i])*negatives_weight)*x[0]*gradient_multiply;

#pragma omp critical
        {
            error_gradient += this_error_gradient;
        }
    }

    return error_gradient/static_cast<double>(negatives_weight*negatives_number);
}


// Vector input operator

/// This method re-writes the inputs operator >> for the Vector template.
/// @param is Input stream.
/// @param v Input vector.

template <class T> istream &operator>>(istream &is, Vector<T> &v) {
  const size_t size = v.size();

  for(size_t i = 0; i < size; i++) {
    is >> v[i];
  }

  return(is);
}


/// This method re-writes the output operator << for the Vector template.
/// @param os Output stream.
/// @param v Output vector.

template <class T>
ostream &operator<<(ostream &os, const Vector<T> &v) {
  const size_t this_size = v.size();

  if(this_size > 0) {
    os << v[0];

    const char space = ' ';

    for(size_t i = 1; i < this_size; i++) {
      os << space << v[i];
    }
  }

  return(os);
}


/// This method re-writes the output operator << for vectors of vectors.
/// @param os Output stream.
/// @param v Output vector of vectors.

template <class T>
ostream &operator<<(ostream &os, const Vector< Vector<T> > &v)
{
  for(size_t i = 0; i < v.size(); i++)
  {
    os << "subvector_" << i << "\n" << v[i] << endl;
  }

  return(os);
}


/// This method re-writes the output operator << for vectors of matrices.
/// @param os Output stream.
/// @param v Output vector of matrices.

template <class T>
ostream &operator<<(ostream &os, const Vector< Matrix<T> > &v)
{
  for(size_t i = 0; i < v.size(); i++)
  {
    os << "submatrix_" << i << "\n" << v[i] << endl;
  }

  return(os);
}


/// Returns a random number chosen from a uniform distribution.
/// @param minimum Minimum value.
/// @param maximum Maximum value.

template <class T>
T calculate_random_uniform(const T &minimum, const T &maximum)
{
  const T random = static_cast<T>(rand() /(RAND_MAX + 1.0));

  const T random_uniform = minimum + (maximum - minimum) * random;

  return(random_uniform);
}


template <class T>
string number_to_string(const T& value)
{
  ostringstream ss;

  ss << value;

  return(ss.str());
}


/// Returns a random number chosen from a normal distribution.
/// @param mean Mean value of normal distribution.
/// @param standard_deviation Standard deviation value of normal distribution.

template <class T>
T calculate_random_normal(const T &mean, const T &standard_deviation) {
  const double pi = 4.0 * atan(1.0);

  T random_uniform_1;

  do {
    random_uniform_1 = static_cast<T>(rand()) /(RAND_MAX + 1.0);

  } while(random_uniform_1 == 0.0);

  const T random_uniform_2 = static_cast<T>(rand()) /(RAND_MAX + 1.0);

  // Box-Muller transformation

  const T random_normal = mean +
                          sqrt(-2.0 * log(random_uniform_1)) *
                              sin(2.0 * pi * random_uniform_2) *
                              standard_deviation;

  return(random_normal);
}


template <class T>
string write_elapsed_time(const T& elapsed_time)
{
  string elapsed_time_string;

  const size_t hours = static_cast<size_t>(elapsed_time/3600);

  size_t minutes = static_cast<size_t>(elapsed_time) - hours*3600;

  minutes = static_cast<size_t>(minutes/60);

  const size_t seconds = static_cast<size_t>(elapsed_time) - hours*3600 - minutes*60;

  if(hours != 0)
  {
      elapsed_time_string = to_string(hours) + ":";
  }

  if(minutes < 10)
  {
      elapsed_time_string += "0";
  }
  elapsed_time_string += to_string(minutes) + ":";

  if(seconds < 10)
  {
      elapsed_time_string += "0";
  }
  elapsed_time_string += to_string(seconds);

  return elapsed_time_string;
}


template <class T>
string write_date_from_time_t(const T& date)
{
    char date_char[20];
    strftime(date_char, 20, "%d/%m/%Y", localtime(&date));

    const string date_string(date_char);

    return date_string;
}


/// Splits the string into substrings wherever delimiter occurs, and returns the vector of those strings.
/// If sep does not match anywhere in the string, split() returns a single-element list containing this string.
/// @param source String to be splited.
/// @param delimiter Separator between substrings.

template <class T>
vector<string> split_string(const T& source, const char& delimiter)
{
    vector<string> elements;
    string element;

    istringstream is(source);

    while(getline(is, element, delimiter)) {
      elements.push_back(element);
    }

    return(elements);
}


/// Replaces a substring by another one in a given string.
/// @param source String.
/// @param find Substring to be replaced.
/// @param replace Substring to be put.

template <class T>
void replace_substring(T& source, const T& find, const T& replace)
{
    for(string::size_type i = 0;(i = source.find(find, i)) != string::npos;)
    {
        source.replace(i, find.length(), replace);

        i += replace.length();
    }
}


/// This structure contains the simplest statistics for a set, variable, etc.
/// It includes the minimum, maximum, mean and standard deviation variables.

template <class T> struct Statistics {
  // Default constructor.

  Statistics();

  // Values constructor.

  Statistics(const T &, const T &, const T &, const T &);

  /// Destructor.

  virtual ~Statistics();

  // METHODS

  void set_minimum(const double &);

  void set_maximum(const double &);

  void set_mean(const double &);

  void set_standard_deviation(const double &);

  Vector<T> to_vector() const;

  void initialize_random();

  bool has_minimum_minus_one_maximum_one();
  bool has_mean_zero_standard_deviation_one();

  void save(const string &file_name) const;

  /// Name of variable

  string name;

  /// Smallest value of a set, function, etc.

  T minimum = 0;

  /// Biggest value of a set, function, etc.

  T maximum = 0;

  /// Mean value of a set, function, etc.

  T mean = 0;

  /// Standard deviation value of a set, function, etc.

  T standard_deviation = 0;
};


template <class T>
Statistics<T>::Statistics() {
  name = "Statistics";
  minimum = static_cast<T>(-1.0);
  maximum = static_cast<T>(1.0);
  mean = static_cast<T>(0.0);
  standard_deviation = static_cast<T>(1.0);
}


/// Values constructor.

template <class T>
Statistics<T>::Statistics(const T &new_minimum, const T &new_maximum,
                          const T &new_mean, const T &new_standard_deviation) {
  minimum = new_minimum;
  maximum = new_maximum;
  mean = new_mean;
  standard_deviation = new_standard_deviation;
}


/// Destructor.

template <class T> Statistics<T>::~Statistics() {}

/// Sets a new minimum value in the statistics structure.
/// @param new_minimum Minimum value.

template <class T> void Statistics<T>::set_minimum(const double &new_minimum) {
  minimum = new_minimum;
}


/// Sets a new maximum value in the statistics structure.
/// @param new_maximum Maximum value.

template <class T> void Statistics<T>::set_maximum(const double &new_maximum) {
  maximum = new_maximum;
}


/// Sets a new mean value in the statistics structure.
/// @param new_mean Mean value.

template <class T> void Statistics<T>::set_mean(const double &new_mean) {
  mean = new_mean;
}


/// Sets a new standard deviation value in the statistics structure.
/// @param new_standard_deviation Standard deviation value.

template <class T>
void Statistics<T>::set_standard_deviation(const double &new_standard_deviation) {
  standard_deviation = new_standard_deviation;
}


/// Returns all the statistical parameters contained in a single vector.
/// The size of that vector is seven.
/// The elements correspond to the minimum, maximum, mean and standard deviation
/// values respectively.

template <class T> Vector<T> Statistics<T>::to_vector() const {
  Vector<T> statistics_vector(4);
  statistics_vector[0] = minimum;
  statistics_vector[1] = maximum;
  statistics_vector[2] = mean;
  statistics_vector[3] = standard_deviation;

  return(statistics_vector);
}


/// Initializes the statistics structure with a random
/// minimum(between -1 and 1), maximum(between 0 and 1),
/// mean(between -1 and 1), standard deviation(between 0 and 1).

template <class T> void Statistics<T>::initialize_random() {
  minimum = calculate_random_uniform(-1.0, 0.0);
  maximum = calculate_random_uniform(0.0, 1.0);
  mean = calculate_random_uniform(-1.0, 1.0);
  standard_deviation = calculate_random_uniform(0.0, 1.0);
}


/// Returns true if the minimum value is -1 and the maximum value is +1,
/// and false otherwise.

template <class T> bool Statistics<T>::has_minimum_minus_one_maximum_one() {
  if(-1.000001 < minimum && minimum < -0.999999 && 0.999999 < maximum &&
      maximum < 1.000001) {
    return(true);
  } else {
    return(false);
  }
}


/// Returns true if the mean value is 0 and the standard deviation value is 1,
/// and false otherwise.

template <class T>
bool Statistics<T>::has_mean_zero_standard_deviation_one() {
  if(-0.000001 < mean && mean < 0.000001 && 0.999999 < standard_deviation &&
      standard_deviation < 1.000001) {
    return(true);
  } else {
    return(false);
  }
}


/// Saves to a file the minimum, maximum, standard deviation, asymmetry and
/// kurtosis values
/// of the statistics structure.
/// @param file_name Name of statistics data file.

template <class T>
void Statistics<T>::save(const string &file_name) const {
  ofstream file(file_name.c_str());

  if(!file.is_open()) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics template.\n"
           << "void save(const string&) const method.\n"
           << "Cannot open statistics data file.\n";

    throw logic_error(buffer.str());
  }

  // Write file

  file << "Minimum: " << minimum << endl
       << "Maximum: " << maximum << endl
       << "Mean: " << mean << endl
       << "Standard deviation: " << standard_deviation << endl;

  // Close file

  file.close();
}


/// This method re-writes the output operator << for the Statistics template.
/// @param os Output stream.
/// @param v Output vector.

template <class T>
ostream &operator<<(ostream &os, const Statistics<T> &statistics)
{
  //if(!statistics.name.empty())

  os << statistics.name << endl;

  os << "  Minimum: " << statistics.minimum << endl
     << "  Maximum: " << statistics.maximum << endl
     << "  Mean: " << statistics.mean << endl
     << "  Standard deviation: " << statistics.standard_deviation << endl;

  return(os);
}


///
/// This template contains the data needed to represent a histogram.
///

template <class T> struct Histogram {
  /// Default constructor.

  explicit Histogram();

  /// Destructor.

  virtual ~Histogram();

  /// Bins number constructor.

  Histogram(const size_t &);

  /// Values constructor.

  Histogram(const Vector<T> &, const Vector<size_t> &);

  size_t get_bins_number() const;

  size_t count_empty_bins() const;

  size_t calculate_minimum_frequency() const;

  size_t calculate_maximum_frequency() const;

  size_t calculate_most_populated_bin() const;

  Vector<T> calculate_minimal_centers() const;

  Vector<T> calculate_maximal_centers() const;

  size_t calculate_bin(const T &) const;

  size_t calculate_frequency(const T &) const;

  /// Positions of the bins in the histogram.

  Vector<T> centers;

  /// Minimum of the bins in the histogram.

  Vector<T> minimums;

  /// Maximum of the bins in the histogram.

  Vector<T> maximums;

  /// Population of the bins in the histogram.

  Vector<size_t> frequencies;
};


template <class T> Histogram<T>::Histogram() {}


/// Destructor.

template <class T> Histogram<T>::~Histogram() {}


/// Bins number constructor.
/// @param bins_number Number of bins in the histogram.

template <class T> Histogram<T>::Histogram(const size_t &bins_number) {
  centers.resize(bins_number);
  frequencies.resize(bins_number);
}


/// Values constructor.
/// @param new_centers Center values for the bins.
/// @param new_frequencies Number of variates in each bin.

template <class T>
Histogram<T>::Histogram(const Vector<T> &new_centers,
                        const Vector<size_t> &new_frequencies) {
  centers = new_centers;
  frequencies = new_frequencies;
}


/// Returns the number of bins in the histogram.

template <class T> size_t Histogram<T>::get_bins_number() const {
  return(centers.size());
}


/// Returns the number of bins with zero variates.

template <class T> size_t Histogram<T>::count_empty_bins() const {
  return(frequencies.count_equal_to(0));
}


/// Returns the number of variates in the less populated bin.

template <class T>
size_t Histogram<T>::calculate_minimum_frequency() const {
  return(frequencies.calculate_minimum());
}


/// Returns the number of variates in the most populated bin.

template <class T>
size_t Histogram<T>::calculate_maximum_frequency() const {
  return(frequencies.calculate_maximum());
}


/// Retuns the index of the most populated bin.

template <class T>
size_t Histogram<T>::calculate_most_populated_bin() const {
  return(frequencies.calculate_maximal_index());
}


/// Returns a vector with the centers of the less populated bins.

template <class T>
Vector<T> Histogram<T>::calculate_minimal_centers() const {
  const size_t minimum_frequency = calculate_minimum_frequency();

  const Vector<size_t> minimal_indices =
      frequencies.calculate_equal_to_indices(minimum_frequency);

  return(centers.get_subvector(minimal_indices));
}


/// Returns a vector with the centers of the most populated bins.

template <class T>
Vector<T> Histogram<T>::calculate_maximal_centers() const {
  const size_t maximum_frequency = calculate_maximum_frequency();

  const Vector<size_t> maximal_indices =
      frequencies.calculate_equal_to_indices(maximum_frequency);

  return(centers.get_subvector(maximal_indices));
}


/// Returns the number of the bin to which a given value belongs to.
/// @param value Value for which we want to get the bin.

template <class T> size_t Histogram<T>::calculate_bin(const T &value) const
{
  const size_t bins_number = get_bins_number();

  const double minimum_center = centers[0];
  const double maximum_center = centers[bins_number - 1];

  const double length = static_cast<double>(maximum_center - minimum_center)/static_cast<double>(bins_number - 1.0);

  double minimum_value = centers[0] - length / 2;
  double maximum_value = minimum_value + length;

  if(value < maximum_value) {
    return(0);
  }

  for(size_t j = 1; j < bins_number - 1; j++) {
    minimum_value = minimum_value + length;
    maximum_value = maximum_value + length;

    if(value >= minimum_value && value < maximum_value) {
      return(j);
    }
  }

  if(value >= maximum_value) {
    return(bins_number - 1);
  } else {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<size_t> Histogram<T>::calculate_bin(const T&) const.\n"
           << "Unknown return value.\n";

    throw logic_error(buffer.str());
  }
}


/// Returns the frequency of the bin to which a given value bolongs to.
/// @param value Value for which we want to get the frequency.

template <class T>
size_t Histogram<T>::calculate_frequency(const T &value) const {
  const size_t bin_number = calculate_bin(value);

  const size_t frequency = frequencies[bin_number];

  return(frequency);
}


// Histogram output operator

/// This method re-writes the output operator << for the Histogram template.
/// @param os Output stream.
/// @param v Output vector.

template <class T>
ostream &operator<<(ostream &os, const Histogram<T> &histogram) {
  os << "Histogram structure\n"
     << "Centers: " << histogram.centers << endl
     << "Frequencies: " << histogram.frequencies << endl;

  return(os);
}


///
/// This template defines the parameters of a linear regression analysis between
/// two sets x-y.
///

template <class T> struct LinearRegressionParameters {
  /// Y-intercept of the linear regression.

  double intercept;

  /// Slope of the linear regression.

  double slope;

  /// Correlation coefficient(R-value) of the linear regression.

  double correlation;

  void initialize_random();
};


/// Initializes the linear regression parameters structure with a random
/// intercept (), slope (between -1 and 1)
/// and correlation (between -1 and 1).

template <class T> void LinearRegressionParameters<T>::initialize_random()
{
  intercept = rand();
  slope = calculate_random_uniform(-1.0, 1.0);
  correlation = calculate_random_uniform(-1.0, 1.0);
}


template <class T>
ostream &
operator<<(ostream &os,
           const LinearRegressionParameters<T> &linear_regression_parameters) {
  os << "Linear regression parameters:\n"
     << "Intercept: " << linear_regression_parameters.intercept << "\n"
     << "Slope: " << linear_regression_parameters.slope << "\n"
     << "Correlation: " << linear_regression_parameters.correlation
     << endl;

  return(os);
}


/// This template defines the parameters of a logistic regression analysis
/// between two sets x-y.

template <class T> struct LogisticRegressionParameters
{
  /// Independent coefficient of the logistic function.

  double a;

  /// x coefficient of the logistic function.

  double b;

  /// Correlation coefficient of the logistic regression.

  double correlation;
};


template <class T>
ostream &operator << (
    ostream &os,
    const LogisticRegressionParameters<T> &logistic_regression_parameters) {
  os << "Logistic regression parameters:\n"
     << "a: " << logistic_regression_parameters.a << "\n"
     << "b: " << logistic_regression_parameters.b << "\n"
     << "Correlation: " << logistic_regression_parameters.correlation
     << endl;

  return(os);
}


template<class T>
Vector<T> sine(const Vector<T>& x)
{
    size_t n = x.size();
    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = sin(x[i]);
    }

    return y;
}


template<class T>
Vector<T> cosine(const Vector<T>& x)
{
    size_t n = x.size();
    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = cosine(x[i]);
    }

    return y;
}


template <class T>
struct KMeansResults
{
  Vector< Vector<size_t> > clusters;
};


template<class T>
Vector<T> hyperbolic_tangent(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = tanh(x[i]);
    }

    return y;
}


template<class T>
Vector<T> hyperbolic_tangent_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }

    return y;
}


template<class T>
Vector<T> linear(const Vector<T>& x)
{
    return x;
}


template<class T>
Vector<T> linear_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n, 1);

    return y;
}


template<class T>
Vector<T> linear_second_derivatives(const Vector<T>& x)
{
    const Vector<double> y(x.size(),0);

    return y;
}


template<class T>
Vector<T> hyperbolic_tangent_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = -2*hyperbolic_tangent*(1 - hyperbolic_tangent * hyperbolic_tangent);
    }

    return y;
}


template<class T>
Vector<T> logistic(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = 1.0 / (1.0 + exp(-x[i]));
    }

    return y;
}


template<class T>
Vector<T> logistic_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = exponential / ((1.0 + exponential)*(1.0 + exponential));
    }

    return y;
}


template<class T>
Vector<T> logistic_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = (exponential*exponential - exponential) / ((1.0 + exponential)*(1.0 + exponential)*(1.0 + exponential));
    }

    return y;
}


template<class T>
Vector<T> threshold(const Vector<T>& x)
{
   const size_t n = x.size();

   Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = 0.0 : y[i] = 1.0;
    }

    return y;
}


template<class T>
Vector<T> threshold_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

     for(size_t i = 0; i < n; i++)
     {
         if(x[i] < 0 || x[i] > 0)
         {
             y[i] = 0.0;
         }
         else
         {
             ostringstream buffer;

             buffer << "OpenNN Exception: Matrix Template.\n"
                    << "Matrix<T> threshold_derivatives(const Matrix<T>&).\n"
                    << "Derivate does not exist for x equal to 0.\n";

             throw logic_error(buffer.str());
         }
     }

     return y;
}


template<class T>
Vector<T> threshold_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

     for(size_t i = 0; i < n; i++)
     {
         if(x[i] < 0 || x[i] > 0)
         {
             y[i] = 0.0;
         }
         else
         {
             ostringstream buffer;

             buffer << "OpenNN Exception: Matrix Template.\n"
                    << "Matrix<T> threshold_derivatives(const Matrix<T>&).\n"
                    << "Derivate does not exist for x equal to 0.\n";

             throw logic_error(buffer.str());
         }
     }

     return y;
}


template<class T>
Vector<T> symmetric_threshold(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
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
Vector<T> rectified_linear(const Vector<T>& x)
{
    cout << "HERE" << endl;
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = 0.0 : y[i] = x[i];

    }

    return y;
}


template<class T>
Vector<T> rectified_linear_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
        x[i] <= 0.0 ? derivatives[i] = 0.0 : derivatives[i] = 1.0;
    }

    return derivatives;
}


template<class T>
Vector<T> rectified_linear_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    const Vector<T> second_derivatives(n, 0.0);

    return second_derivatives;
}


template<class T>
Vector<T> scaled_exponential_linear(const Vector<T>& x)
{
    const size_t n = x.size();

    const double lambda = 1.0507;
    const double alpha = 1.67326;

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
    }

    return y;
}


template<class T>
Vector<T> scaled_exponential_linear_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    const double lambda =1.0507;
    const double alpha =1.67326;

    Vector<T> derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? derivatives[i] = lambda * alpha * exp(x[i]) : derivatives[i] = lambda;
    }

    return derivatives;
}


template<class T>
Vector<T> scaled_exponential_linear_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    const double lambda = 1.0507;
    const double alpha = 1.67326;

    Vector<T> second_derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? second_derivatives[i] = lambda * alpha * exp(x[i]) : second_derivatives[i] = 0.0;
    }

    return second_derivatives;
}


template<class T>
Vector<T> soft_plus(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = log(1 + exp(x[i]));
    }

    return y;
}


template<class T>
Vector<T> soft_plus_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
        derivatives[i] = 1/(1 + exp(-x[i]));
    }

    return derivatives;
}


template<class T>
Vector<T> soft_plus_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> second_derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
       second_derivatives[n] = exp(-x[i]) / pow((1 + exp(-x[i])), 2);
    }

    return second_derivatives;
}


template<class T>
Vector<T> soft_sign(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
    }

    return y;
}


template<class T>
Vector<T> soft_sign_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? derivatives[i] = 1 / pow((1 - x[i]), 2) : derivatives[i] = 1 / pow((1 + x[i]), 2);

    }

    return derivatives;
}


template<class T>
Vector<T> soft_sign_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> second_derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? second_derivatives[i] = -(2 * x[i]) / pow((1 - x[i]), 3) : second_derivatives[i] = -(2 * x[i]) / pow((1 + x[i]), 3);
    }

    return second_derivatives;
}


template<class T>
Vector<T> hard_sigmoid(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
    {
        if(x[i] < -2.5)
        {
           y[n] = 0;
        }
        else if(x[i] > 2.5)
        {
            y[n] = 1;
        }
        else
        {
            y[n] = 0.2 * x[i] + 0.5;
        }
    }

    return y;
}


template<class T>
Vector<T> hard_sigmoid_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> derivatives(n);

    for(size_t i = 0; i < n; i++)
    {
        x[i] < -2.5 || x[i] > 2.5 ? derivatives[i] = 0.0 : derivatives[i] = 0.2;
    }

    return derivatives;
}


template<class T>
Vector<T> hard_sigmoid_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> second_derivatives(n, 0.0);

    return second_derivatives;
}



template<class T>
Vector<T> exponential_linear(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
    }

    return y;
}


template<class T>
Vector<T> exponential_linear_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> derivatives(n);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? derivatives[i] = alpha * exp(x[i]) : derivatives[i] = 1.0;
    }

    return derivatives;
}


template<class T>
Vector<T> exponential_linear_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> second_derivatives(n);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? second_derivatives[i] = alpha * exp(x[i]) : second_derivatives[i] = 0.0;
    }

    return second_derivatives;
}


template<class T>
Vector<T> symmetric_threshold_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
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
Vector<T> symmetric_threshold_second_derivatives(const Vector<T>& x)
{
    const size_t n = x.size();

    Vector<T> y(n);

    for(size_t i = 0; i < n; i++)
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

} // end namespace OpenNN

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
