/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V E C T O R   C O N T A I N E R                                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                  */
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
#ifdef __OPENNN_MPI__
#include <mpi.h>
#endif

// Eigen includes

#include "../eigen/Eigen"

namespace OpenNN {

// Forward declarations

template <class T> class Matrix;

template <class T> T calculate_random_uniform(const T & = -1, const T & = 1);

template <class T> T calculate_random_normal(const T & = 0.0, const T & = 1.0);

template <class T> struct Histogram;
template <class T> struct Statistics;
template <class T> struct LinearRegressionParameters;
template <class T> struct LogisticRegressionParameters;

/// This template represents an array of any kind of numbers or objects.
/// It inherits from the vector of the standard library, and implements
/// additional utilities.

template <typename T> class Vector : public std::vector<T> {
public:
  // CONSTRUCTORS

  // Default constructor.

  explicit Vector(void);

  // General constructor.

  explicit Vector(const size_t &);

  // Constant reference initialization constructor.

  explicit Vector(const size_t &, const T &);

  // File constructor.

  explicit Vector(const std::string &);

  // Sequential constructor.

  explicit Vector(const T &, const double &, const T &);

  // Input iterators constructor

  template <class InputIterator> explicit Vector(InputIterator, InputIterator);

  // Copy constructor.

  Vector(const Vector<T> &);

  // DESTRUCTOR

  virtual ~Vector(void);

  // OPERATORS

  bool operator==(const T &) const;

  bool operator!=(const T &) const;

  bool operator>(const T &) const;

  bool operator<(const T &) const;

  bool operator>=(const T &) const;

  bool operator<=(const T &) const;

  // METHODS

  // Get methods

  // Set methods

  void set(void);

  void set(const size_t &);

  void set(const size_t &, const T &);

  void set(const std::string &);

  void set(const T &, const double &, const T &);

  void set(const Vector &);
#ifdef __OPENNN_MPI__
  void set_MPI(const MPI_Datatype);
#endif
  // Initialization methods

  void initialize(const T &);

  void initialize_sequential(void);

  void randomize_uniform(const double & = -1.0, const double & = 1.0);
  void randomize_uniform(const Vector<double> &, const Vector<double> &);

  void randomize_normal(const double & = 0.0, const double & = 1.0);
  void randomize_normal(const Vector<double> &, const Vector<double> &);

  // Checking methods

  bool contains(const T &) const;

  bool contains(const Vector<T> &) const;

  bool is_in(const T &, const T &) const;

  bool is_constant(const double & = 0.0) const;

  bool is_crescent(void) const;

  bool is_decrescent(void) const;

  bool is_binary(void) const;

  bool Lillieforts_normality_test(void) const;

  // Other methods

  size_t count_occurrences(const T &) const;

  Vector<size_t> calculate_occurrence_indices(const T &) const;

  size_t count_greater_than(const T &) const;

  size_t count_less_than(const T &) const;

  Vector<size_t> calculate_equal_than_indices(const T &) const;

  Vector<size_t> calculate_less_than_indices(const T &) const;

  Vector<size_t> calculate_greater_than_indices(const T &) const;

  Vector<size_t>
  calculate_total_frequencies(const Vector< Histogram<T> > &) const;
  Vector<size_t> calculate_total_frequencies_missing_values(
      const Vector<size_t> missing_values, const Vector< Histogram<T> > &) const;

  Vector<double> perform_Box_Cox_transformation(const double& lambda = 1) const;

  // Statistics methods

  T calculate_minimum(void) const;

  T calculate_maximum(void) const;

  Vector<T> calculate_minimum_maximum(void) const;

  T calculate_minimum_missing_values(const Vector<size_t> &) const;

  T calculate_maximum_missing_values(const Vector<size_t> &) const;

  Vector<T>
  calculate_minimum_maximum_missing_values(const Vector<size_t> &) const;

  Vector<T> calculate_explained_variance(void) const;

  // Histogram methods

  Histogram<T> calculate_histogram(const size_t & = 10) const;
  Histogram<T> calculate_histogram_binary(void) const;

  Histogram<T> calculate_histogram_missing_values(const Vector<size_t> &,
                                                  const size_t & = 10) const;

  size_t calculate_minimal_index(void) const;

  size_t calculate_maximal_index(void) const;

  Vector<size_t> calculate_minimal_indices(const size_t &) const;

  Vector<size_t> calculate_maximal_indices(const size_t &) const;

  Vector<size_t> calculate_minimal_maximal_index(void) const;

  Vector<T> calculate_pow(const T &) const;

  Vector<T> calculate_competitive(void) const;

  Vector<T> calculate_softmax(void) const;

  Matrix<T> calculate_softmax_Jacobian(void) const;

  Vector<bool> calculate_binary(void) const;

  Vector<T> calculate_cumulative(void) const;

  size_t calculate_cumulative_index(const T &) const;

  size_t calculate_closest_index(const T &) const;

  T calculate_sum(void) const;

  T calculate_partial_sum(const Vector<size_t> &) const;

  T calculate_sum_missing_values(const Vector<size_t> &) const;

  T calculate_product(void) const;

  double calculate_mean(void) const;

  double calculate_variance(void) const;

  double calculate_covariance(const Vector<double>&) const;

  double calculate_standard_deviation(void) const;

  double calculate_asymmetry(void) const;

  double calculate_kurtosis(void) const;

  double calculate_median(void) const;

  Vector<double> calculate_quartiles(void) const;

  Vector<double> calculate_quartiles_missing_values(const Vector<size_t> &) const;

  Vector<double> calculate_mean_standard_deviation(void) const;

  double calculate_mean_missing_values(const Vector<size_t> &) const;

  double calculate_variance_missing_values(const Vector<size_t> &) const;

  double calculate_weighted_mean(const Vector<double> &) const;

  double
  calculate_standard_deviation_missing_values(const Vector<size_t> &) const;

  double calculate_asymmetry_missing_values(const Vector<size_t> &) const;

  double calculate_kurtosis_missing_values(const Vector<size_t> &) const;

  Statistics<T> calculate_statistics(void) const;

  Statistics<T>
  calculate_statistics_missing_values(const Vector<size_t> &) const;

  Vector<double> calculate_shape_parameters(void) const;

  Vector<double>
  calculate_shape_parameters_missing_values(const Vector<size_t> &) const;

  Vector<double> calculate_box_plots(void) const;

  Vector<double> calculate_box_plots_missing_values(const Vector<size_t> &) const;

  // Norm methods

  double calculate_norm(void) const;

  Vector<T> calculate_norm_gradient(void) const;

  Matrix<T> calculate_norm_Hessian(void) const;

  double calculate_p_norm(const double &) const;

  Vector<double> calculate_p_norm_gradient(const double &) const;

  Vector<T> calculate_normalized(void) const;

  //double calculate_distance(const Vector<double> &) const;

  double calculate_distance(const Vector<T> &) const;

  double calculate_sum_squared_error(const Vector<double> &) const;
  double calculate_sum_squared_error(const Matrix<T> &, const size_t &,
                                     const Vector<size_t> &) const;

  double calculate_Minkowski_error(const Vector<double> &,
                                   const double &) const;

  // Correlation methods

  double calculate_linear_correlation(const Vector<T> &) const;

  T calculate_linear_correlation_missing_values(const Vector<T> &,
                                                const Vector<size_t> &) const;

  Vector<double> calculate_autocorrelation(const size_t & = 10) const;

  Vector<double> calculate_cross_correlation(const Vector<double> &,
                                             const size_t & = 10) const;

  LinearRegressionParameters<T>
  calculate_linear_regression_parameters(const Vector<T> &) const;

  Vector<T> calculate_absolute_value(void) const;

  void apply_absolute_value(void);

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

  Vector<size_t> sort_less_indices(void) const;

  Vector<size_t> sort_greater_indices(void) const;


  Vector<size_t> calculate_less_rank(void) const;

  Vector<size_t> calculate_greater_rank(void) const;

  // Mathematical operators

  inline Vector<T> operator+(const T &) const;

  inline Vector<T> operator+(const Vector<T> &) const;

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

  void operator+=(const T &);

  void operator+=(const Vector<T> &);

  void operator-=(const T &);

  void operator-=(const Vector<T> &);

  void operator*=(const T &);

  void operator*=(const Vector<T> &);

  void operator/=(const T &);

  void operator/=(const Vector<T> &);

  // Filtering methods

  void filter_positive(void);
  void filter_negative(void);

  // Scaling methods

  void scale_minimum_maximum(const T &, const T &);

  void scale_minimum_maximum(const Statistics<T> &);

  Statistics<T> scale_minimum_maximum(void);

  void scale_mean_standard_deviation(const T &, const T &);

  void scale_mean_standard_deviation(const Statistics<T> &);

  Statistics<T> scale_mean_standard_deviation(void);

  void scale_minimum_maximum(const Vector<T> &, const Vector<T> &);

  void scale_mean_standard_deviation(const Vector<T> &, const Vector<T> &);

  Vector<T> calculate_scaled_minimum_maximum(const Vector<T> &,
                                             const Vector<T> &) const;

  Vector<T> calculate_scaled_mean_standard_deviation(const Vector<T> &,
                                                     const Vector<T> &) const;

  // Unscaling methods

  Vector<T> calculate_unscaled_minimum_maximum(const Vector<T> &,
                                               const Vector<T> &) const;

  Vector<T> calculate_unscaled_mean_standard_deviation(const Vector<T> &,
                                                       const Vector<T> &) const;

  void unscale_minimum_maximum(const Vector<T> &, const Vector<T> &);

  void unscale_mean_standard_deviation(const Vector<T> &, const Vector<T> &);

  // Arranging methods

  Matrix<T> arrange_diagonal_matrix(void) const;

  Vector<T> arrange_subvector(const Vector<size_t> &) const;

  Vector<T> arrange_subvector_first(const size_t &) const;

  Vector<T> arrange_subvector_last(const size_t &) const;

  // File operations

  void load(const std::string &);

  void save(const std::string &) const;

  void tuck_in(const size_t &, const Vector<T> &);

  Vector<T> take_out(const size_t &, const size_t &) const;

  Vector<T> insert_element(const size_t &, const T &) const;
  Vector<T> remove_element(const size_t &) const;

  Vector<T> remove_value(const T &) const;

  Vector<T> assemble(const Vector<T> &) const;

  std::vector<T> to_std_vector(void) const;

  Matrix<T> to_row_matrix(void) const;

  Matrix<T> to_column_matrix(void) const;

  void parse(const std::string &);

  std::string to_text() const;

  std::string to_string(const std::string & = " ") const;

  Vector<std::string> write_string_vector(const size_t & = 5) const;

  Matrix<T> to_matrix(const size_t &, const size_t &) const;
};

// CONSTRUCTORS

/// Default constructor. It creates a vector of size zero.

template <class T> Vector<T>::Vector(void) : std::vector<T>() {}

/// General constructor. It creates a vector of size n, containing n copies of
/// the default value for Type.
/// @param new_size Size of vector.

template <class T>
Vector<T>::Vector(const size_t &new_size)
    : std::vector<T>(new_size) {}

/// Constant reference initialization constructor.
/// It creates a vector of size n, containing n copies of the type value of
/// Type.
/// @param new_size Size of Vector.
/// @param value Initialization value of Type.

template <class T>
Vector<T>::Vector(const size_t &new_size, const T &value)
    : std::vector<T>(new_size, value) {}

/// File constructor. It creates a vector object by loading its members from a
/// data file.
/// @param file_name Name of vector data file.

template <class T>
Vector<T>::Vector(const std::string &file_name)
    : std::vector<T>() {
  load(file_name);
}

/// Sequential constructor.

template <class T>
Vector<T>::Vector(const T &first, const double &step, const T &last)
    : std::vector<T>() {
  set(first, step, last);
}

/// Input iterators constructor

template <class T>
template <class InputIterator>
Vector<T>::Vector(InputIterator first, InputIterator last)
    : std::vector<T>(first, last) {}

/// Copy constructor. It creates a copy of an existing Vector.
/// @param other_vector Vector to be copied.

template <class T>
Vector<T>::Vector(const Vector<T> &other_vector)
    : std::vector<T>(other_vector) {}

// DESTRUCTOR

/// Destructor.
template <class T> Vector<T>::~Vector(void) {}

// bool  == (const T&) const

/// Equal to operator between this vector and a Type value.
/// It produces true if all the elements of this vector are equal to the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator==(const T &value) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] != value) {
      return (false);
    }
  }

  return (true);
}

// bool operator != (const T&) const

/// Not equivalent relational operator between this vector and a Type value.
/// It produces true if some element of this vector is not equal to the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator!=(const T &value) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] != value) {
      return (true);
    }
  }

  return (false);
}

// bool operator > (const T&) const

/// Greater than relational operator between this vector and a Type value.
/// It produces true if all the elements of this vector are greater than the
/// Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator>(const T &value) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] <= value) {
      return (false);
    }
  }

  return (true);
}

// bool operator < (const T&) const

/// Less than relational operator between this vector and a Type value.
/// It produces true if all the elements of this vector are less than the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator<(const T &value) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] >= value) {
      return (false);
    }
  }

  return (true);
}

// bool operator >= (const T&) const

/// Greater than or equal to than relational operator between this vector and a
/// Type value.
/// It produces true if all the elements of this vector are greater than or
/// equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator>=(const T &value) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < value) {
      return (false);
    }
  }

  return (true);
}

// bool operator <= (const T&) const

/// Less than or equal to than relational operator between this vector and a
/// Type value.
/// It produces true if all the elements of this vector are less than or equal
/// to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator<=(const T &value) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > value) {
      return (false);
    }
  }

  return (true);
}

// METHODS

// void set(void) method

/// Sets the size of a vector to zero.

template <class T> void Vector<T>::set(void) { this->resize(0); }

// void set(const size_t&) method

/// Sets a new size to the vector. It does not initialize the data.
/// @param new_size Size for the vector.

template <class T> void Vector<T>::set(const size_t &new_size) {
  this->resize(new_size);
}

// void set(const size_t&, const T&) method

/// Sets a new size to the vector and initializes all its elements with a given
/// value.
/// @param new_size Size for the vector.
/// @param new_value Value for all the elements.

template <class T>
void Vector<T>::set(const size_t &new_size, const T &new_value) {
  this->resize(new_size);

  initialize(new_value);
}

// void set(const std::string&) method

/// Sets all the members of a vector object by loading them from a data file.
/// The format is specified in the OpenNN manual.
/// @param file_name Name of vector data file.

template <class T> void Vector<T>::set(const std::string &file_name) {
  load(file_name);
}

// void set(const T&, const double&, const T&) method

/// Makes this vector to have elements starting from a given value, continuing
/// with a step value and finishing with a given value.
/// Depending on the starting, step and finishin values, this method can produce
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
    const size_t new_size = 1 + (size_t)((last - first) / step + 0.5);

    this->resize(new_size);

    for (size_t i = 0; i < new_size; i++) {
      (*this)[i] = first + (T)(i * step);
    }
  }
}

// void set(const Vector&) method

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
        vector_size = (int)this->size();
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

// void initialize(const T&) method

/// Initializes all the elements of the vector with a given value.
/// @param value Type value.

template <class T> void Vector<T>::initialize(const T &value) {
  std::fill((*this).begin(), (*this).end(), value);
}

// void initialize_sequential(void) method

/// Initializes all the elements of the vector in a sequential order.

template <class T> void Vector<T>::initialize_sequential(void) {
  for (size_t i = 0; i < this->size(); i++) {
    (*this)[i] = (T)i;
  }
}

// void randomize_uniform(const double&, const double&) method

/// Assigns a random value comprised between a minimum value and a maximum value
/// to each element in
/// the vector.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

template <class T>
void Vector<T>::randomize_uniform(const double &minimum,
                                  const double &maximum) {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(minimum > maximum) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const double&, const double&) method.\n"
           << "Minimum value must be less or equal than maximum value.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (T)calculate_random_uniform(minimum, maximum);
  }
}

// void randomize_uniform(const Vector<double>&, const Vector<double>&) method

/// Assigns a random value comprised between given minimum and a maximum values
/// to every element in the
/// vector.
/// @param minimums Minimum initialization values.
/// @param maximums Maximum initialization values.

template <class T>
void Vector<T>::randomize_uniform(const Vector<double> &minimums,
                                  const Vector<double> &maximums) {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t minimums_size = minimums.size();
  const size_t maximums_size = maximums.size();

  if(minimums_size != this_size || maximums_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Minimum and maximum sizes must be equal to vector size.\n";

    throw std::logic_error(buffer.str());
  }

  if(minimums > maximums) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Minimum values must be less or equal than their corresponding "
              "maximum values.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = calculate_random_uniform(minimums[i], maximums[i]);
  }
}

// void randomize_normal(const double&, const double&) method

/// Assigns random values to each element in the vector.
/// These are taken from a normal distribution with single mean and standard
/// deviation values for all the elements.
/// @param mean Mean value of uniform distribution.
/// @param standard_deviation Standard deviation value of uniform distribution.

template <class T>
void Vector<T>::randomize_normal(const double &mean,
                                 const double &standard_deviation) {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(standard_deviation < 0.0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_normal(const double&, const double&) method.\n"
           << "Standard deviation must be equal or greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = calculate_random_normal(mean, standard_deviation);
  }
}

// void randomize_normal(const Vector<double>, const Vector<double>) method

/// Assigns random values to each element in the vector.
/// These are taken from normal distributions with given means and standard
/// deviations for each element.
/// @param mean Mean values of normal distributions.
/// @param standard_deviation Standard deviation values of normal distributions.

template <class T>
void Vector<T>::randomize_normal(const Vector<double> &mean,
                                 const Vector<double> &standard_deviation) {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t mean_size = mean.size();
  const size_t standard_deviation_size = standard_deviation.size();

  if(mean_size != this_size || standard_deviation_size != this_size) {
    std::ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "void randomize_normal(const Vector<double>&, const "
           "Vector<double>&) method.\n"
        << "Mean and standard deviation sizes must be equal to vector size.\n";

    throw std::logic_error(buffer.str());
  }

  if(standard_deviation < 0.0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_normal(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Standard deviations must be equal or greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = calculate_random_normal(mean[i], standard_deviation[i]);
  }
}

// bool contains(const T&) const method

/// Returns true if the vector contains a certain value, and false otherwise.

template <class T> bool Vector<T>::contains(const T &value) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] == value) {
      return (true);
    }
  }

  return (false);
}
// bool contains(const Vector<T>&) const method

/// Returns true if the vector contains a certain value from a given set, and
/// false otherwise.

template <class T> bool Vector<T>::contains(const Vector<T> &values) const {
  if(values.empty()) {
    return (false);
  }

  const size_t this_size = this->size();

  const size_t values_size = values.size();

  for (size_t i = 0; i < this_size; i++) {
    for (size_t j = 0; j < values_size; j++) {
      if((*this)[i] == values[j]) {
        return (true);
      }
    }
  }

  return (false);
}

// bool is_in(const T&, const T&) const method

/// Returns true if the value of all the elements fall in some given range,
/// and false otherwise.
/// @param minimum Minimum value of the range.
/// @param maximum Maximum value of the range.

template <class T>
bool Vector<T>::is_in(const T &minimum, const T &maximum) const {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < minimum || (*this)[i] > maximum) {
      return (false);
    }
  }

  return (true);
}

// bool is_constant(const double&) const method

/// Returns true if all the elements have the same value within a defined
/// tolerance ,
/// and false otherwise.
/// @param tolerance Tolerance value, so that if abs(max-min) <= tol, then the
/// vector is considered constant.

template <class T> bool Vector<T>::is_constant(const double &tolerance) const {
  const size_t this_size = this->size();

  if(this_size == 0) {
    return (false);
  }

  const T minimum = calculate_minimum();
  const T maximum = calculate_maximum();

  if(fabs(maximum - minimum) <= tolerance) {
    return (true);
  } else {
    return (false);
  }
}

// bool is_crescent(void) const method

/// Returns true if all the elements in the vector have values which increase
/// with the index, and false otherwise.

template <class T> bool Vector<T>::is_crescent(void) const {
  for (size_t i = 0; i < this->size() - 1; i++) {
    if((*this)[i] > (*this)[i + 1]) {
      return (false);
    }
  }

  return (true);
}

// bool is_decrescent(void) const method

/// Returns true if all the elements in the vector have values which decrease
/// with the index, and false otherwise.

template <class T> bool Vector<T>::is_decrescent(void) const {
  for (size_t i = 0; i < this->size() - 1; i++) {
    if((*this)[i] < (*this)[i + 1]) {
      return (false);
    }
  }

  return (true);
}

// bool is_binary(void) const method

/// Returns true if all the elements in the vector have binary values, and false otherwise.

template <class T> bool Vector<T>::is_binary(void) const
{
    const size_t this_size = this->size();

    for (size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] != 0 && (*this)[i] != 1)
        {
            return false;
        }
    }

    return true;
}

// bool Lillieforts_normality_test(void) const method

/// Returns true if the elements in the vector have a normal distribution.

template <class T> bool Vector<T>::Lillieforts_normality_test(void) const
{
#ifndef __Cpp11__
    const size_t n = this->size();

    const double mean = this->calculate_mean();
    const double standard_deviation = this->calculate_standard_deviation();

    const double fn = (0.83 + n)/std::sqrt(n) - 0.01;
    const double Dna = 0.895/fn;

    Vector<T> sorted_vector(*this);

    std::sort(sorted_vector.begin(), sorted_vector.end(), std::less<double>());

    double Fx;
    double Snx;

    double D = -1;

    for(size_t i = 0; i < n; i++)
    {
        Fx = 0.5 * std::erfc((mean - (*this)[i])/(standard_deviation*std::sqrt(2)));

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
                if((*this)[i] >= sorted_vector[j] && (*this)[i] < sorted_vector[j+1])
                {
                    Snx = (double)(j+1)/(double)n;
                }
            }
        }

        if(D < std::abs(Fx - Snx))
        {
            D = std::abs(Fx - Snx);
        }
    }

    if(D < Dna)
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

// size_t count_occurrences(const T&) const method

/// Returns the number of times that a certain value is contained in the vector.

template <class T> size_t Vector<T>::count_occurrences(const T &value) const {
  const size_t this_size = this->size();

  size_t count = 0;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] == value) {
      count++;
    }
  }

  return (count);
}

// Vector<size_t> calculate_occurrence_indices(const T&) const method

/// Returns the vector indices at which the vector elements take some given
/// value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::calculate_occurrence_indices(const T &value) const {
  const size_t this_size = this->size();

  const size_t occurrences_number = count_occurrences(value);

  Vector<size_t> occurrence_indices(occurrences_number);

  size_t index = 0;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] == value) {
      occurrence_indices[index] = i;
      index++;
    }
  }

  return (occurrence_indices);
}

// size_t count_greater_than(const T&) const method

/// Returns the number of elements which are greater than some given value.
/// @param value Value.

template <class T> size_t Vector<T>::count_greater_than(const T &value) const {
  const size_t this_size = this->size();

  size_t count = 0;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > value) {
      count++;
    }
  }

  return (count);
}

// size_t count_less_than(const T&) const method

/// Returns the number of elements which are less than some given value.
/// @param value Value.

template <class T> size_t Vector<T>::count_less_than(const T &value) const {
  const size_t this_size = this->size();

  size_t count = 0;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < value) {
      count++;
    }
  }

  return (count);
}

// Vector<size_t> calculate_equal_than_indices(const T&) const method

/// Returns the vector indices at which the vector elements are equal than some
/// given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::calculate_equal_than_indices(const T &value) const {
  const size_t this_size = this->size();

  Vector<size_t> equal_than_indices;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] == value) {
      equal_than_indices.push_back(i);
    }
  }

  return (equal_than_indices);
}

// Vector<size_t> calculate_less_than_indices(const T&) const method

/// Returns the vector indices at which the vector elements are less than some
/// given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::calculate_less_than_indices(const T &value) const {
  const size_t this_size = this->size();

  Vector<size_t> less_than_indices;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < value) {
      less_than_indices.push_back(i);
    }
  }

  return (less_than_indices);
}

// Vector<size_t> calculate_greater_than_indices(const T&) const method

/// Returns the vector indices at which the vector elements are greater than
/// some given value.
/// @param value Value.

template <class T>
Vector<size_t> Vector<T>::calculate_greater_than_indices(const T &value) const {

  const size_t this_size = this->size();

  Vector<size_t> greater_than_indices;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > value) {
      greater_than_indices.push_back(i);
    }
  }

  return (greater_than_indices);
}

// Vector<size_t> calculate_total_frequencies(const Vector< Histogram<T> >&)
// const

/// Returns a vector containing the sum of the frequencies of the bins to which
/// this vector belongs.
/// @param histograms Used histograms.

template <class T>
Vector<size_t> Vector<T>::calculate_total_frequencies(
    const Vector< Histogram<T> > &histograms) const {
  const size_t histograms_number = histograms.size();

  Vector<size_t> total_frequencies(histograms_number);

  for (size_t i = 0; i < histograms_number; i++) {
    total_frequencies[i] = histograms[i].calculate_frequency((*this)[i]);
  }

  return (total_frequencies);
}

// Vector<size_t> calculate_total_frequencies_missing_values(const
// Vector<size_t> missing_values, const Vector< Histogram<T> >&);

/// Returns a vector containing the sum of the frequencies of the bins to which
/// this vector
/// blongs.
/// @param instance_missing_values Missing values
/// @param histograms Used histograms

template <class T>
Vector<size_t> Vector<T>::calculate_total_frequencies_missing_values(
    const Vector<size_t> instance_missing_values,
    const Vector< Histogram<T> > &histograms) const {
  const size_t histograms_number = histograms.size();

  Vector<size_t> total_frequencies;

  for (size_t i = 0; i < histograms_number; i++) {
    if(!(instance_missing_values.contains(i))) {
      total_frequencies[i] = histograms[i].calculate_frequency((*this)[i]);
    } else {
      total_frequencies[i] = 0;
    }
  }

  return (total_frequencies);
}

// Vector<double> perform_Box_Cox_transformation(const double&) const method

/// Returns vector with the Box-Cox transformation.
/// @param lambda Exponent of the Box-Cox transformation.

template <class T> Vector<double> Vector<T>::perform_Box_Cox_transformation(const double& lambda) const
{
    const size_t size = this->size();

    Vector<double> vector_tranformation(size);

    for(size_t i = 0; i < size; i++)
    {
        if(lambda == 0)
        {
            vector_tranformation[i] = std::log((double)(*this)[i]);
        }
        else
        {
            vector_tranformation[i] = (std::pow((double)(*this)[i], lambda) - 1)/lambda;

        }
    }

    return vector_tranformation;
}

// T calculate_minimum(void) const method

/// Returns the smallest element in the vector.

template <class T> T Vector<T>::calculate_minimum(void) const {
  const size_t this_size = this->size();

  T minimum = std::numeric_limits<T>::max();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < minimum) {
      minimum = (*this)[i];
    }
  }

  return (minimum);
}

// T calculate_maximum(void) const method

/// Returns the largest element in the vector.

template <class T> T Vector<T>::calculate_maximum(void) const {
  const size_t this_size = this->size();

  T maximum = std::numeric_limits<T>::max();

  if(std::numeric_limits<T>::is_signed) {
    maximum *= -1;
  } else {
    maximum = 0;
  }

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > maximum) {
      maximum = (*this)[i];
    }
  }

  return (maximum);
}

// Vector<T> calculate_minimum_maximum(void) const method

/// Returns a vector containing the smallest and the largest elements in the
/// vector.

template <class T> Vector<T> Vector<T>::calculate_minimum_maximum(void) const {
  size_t this_size = this->size();

  T minimum = std::numeric_limits<T>::max();

  T maximum;

  if(std::numeric_limits<T>::is_signed) {
    maximum = -std::numeric_limits<T>::max();
  } else {
    maximum = 0;
  }

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < minimum) {
      minimum = (*this)[i];
    }

    if((*this)[i] > maximum) {
      maximum = (*this)[i];
    }
  }

  Vector<T> minimum_maximum(2);
  minimum_maximum[0] = minimum;
  minimum_maximum[1] = maximum;

  return (minimum_maximum);
}

// T calculate_minimum_missing_values(const Vector<size_t>&) const method

/// Returns the smallest element in the vector.

template <class T>
T Vector<T>::calculate_minimum_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

  T minimum = std::numeric_limits<T>::max();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < minimum &&
        !missing_indices.contains(i))// && (*this)[i] != -123.456)
    {
      minimum = (*this)[i];
    }
  }

  return (minimum);
}

// T calculate_maximum_missing_values(const Vector<size_t>&) const method

/// Returns the largest element in the vector.

template <class T>
T Vector<T>::calculate_maximum_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

  T maximum;

  if(std::numeric_limits<T>::is_signed) {
    maximum = -std::numeric_limits<T>::max();
  } else {
    maximum = 0;
  }

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > maximum && !missing_indices.contains(i)) {
      maximum = (*this)[i];
    }
  }

  return (maximum);
}

// Vector<T> calculate_minimum_maximum_missing_values(const Vector<size_t>&)
// const method

/// Returns a vector containing the smallest and the largest elements in the
/// vector.

template <class T>
Vector<T> Vector<T>::calculate_minimum_maximum_missing_values(
    const Vector<size_t> &missing_indices) const {
  size_t this_size = this->size();

  T minimum = std::numeric_limits<T>::max();

  T maximum;

  if(std::numeric_limits<T>::is_signed) {
    maximum = -std::numeric_limits<T>::max();
  } else {
    maximum = 0;
  }

  for (size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      if((*this)[i] < minimum) {
        minimum = (*this)[i];
      }

      if((*this)[i] > maximum) {
        maximum = (*this)[i];
      }
    }
  }

  Vector<T> minimum_maximum(2);
  minimum_maximum[0] = minimum;
  minimum_maximum[1] = maximum;

  return (minimum_maximum);
}


// Vector<T> calculate_explained_variance(void) const method

/// Calculates the explained variance for a given vector (principal components analysis).
/// This method returns a vector whose size is the same as the size of the given vector.

template<class T>
Vector<T> Vector<T>::calculate_explained_variance(void) const
{
    const size_t this_size = this->size();

    #ifdef __OPENNN_DEBUG__

      if(this_size == 0) {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance(void) const method.\n"
               << "Size of the vector must be greater than zero.\n";

        throw std::logic_error(buffer.str());
      }

    #endif

    const double this_sum = this->calculate_absolute_value().calculate_sum();

    #ifdef __OPENNN_DEBUG__

      if(this_sum == 0) {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance(void) const method.\n"
               << "Sum of the members of the vector must be greater than zero.\n";

        throw std::logic_error(buffer.str());
      }

    #endif

    #ifdef __OPENNN_DEBUG__

      if(this_sum < 0) {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance(void) const method.\n"
               << "Sum of the members of the vector cannot be negative.\n";

        throw std::logic_error(buffer.str());
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
        std::ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> calculate_explained_variance(void) const method.\n"
               << "Sum of explained variance must be 1.\n";

        throw std::logic_error(buffer.str());
      }

    #endif

    return explained_variance;
}


// Histogram<T> calculate_histogram(const size_t&) const method

/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

template <class T>
Histogram<T> Vector<T>::calculate_histogram(const size_t &bins_number) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Histogram<T> calculate_histogram(const size_t&) const method.\n"
           << "Number of bins is less than one.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> minimums(bins_number);
  Vector<T> maximums(bins_number);

  Vector<T> centers(bins_number);
  Vector<size_t> frequencies(bins_number, 0);

  const Vector<T> minimum_maximum = calculate_minimum_maximum();

  const T minimum = minimum_maximum[0];
  const T maximum = minimum_maximum[1];

  const double length = (maximum - minimum) / (double)bins_number;

  minimums[0] = minimum;
  maximums[0] = minimum + length;
  centers[0] = (maximums[0] + minimums[0]) / 2.0;

  // Calculate bins center

  for (size_t i = 1; i < bins_number; i++)
  {
    minimums[i] = minimums[i - 1] + length;
    maximums[i] = maximums[i - 1] + length;

    centers[i] = (maximums[i] + minimums[i]) / 2.0;
  }

  // Calculate bins frequency

  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    for (size_t j = 0; j < bins_number - 1; j++) {
      if((*this)[i] >= minimums[j] && (*this)[i] < maximums[j]) {
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

  return (histogram);
}


// Histogram<T> calculate_histogram_binary(void) const method

/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

template <class T>
Histogram<T> Vector<T>::calculate_histogram_binary(void) const {
// Control sentence (if debug)

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

  for (size_t i = 0; i < this_size; i++) {
    for (size_t j = 0; j < 2; j++) {
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

  return (histogram);
}


// Histogram<T> calculate_histogram_missing_values(const size_t&) const method

/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

template <class T>
Histogram<T> Vector<T>::calculate_histogram_missing_values(
    const Vector<size_t> &missing_indices, const size_t &bins_number) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Histogram<T> calculate_histogram_missing_values(const "
              "Vector<size_t>&, const size_t&) const method.\n"
           << "Number of bins is less than one.\n";

    throw std::logic_error(buffer.str());
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

  const double length = (maximum - minimum) / (double)bins_number;

  minimums[0] = minimum;
  maximums[0] = minimum + length;
  centers[0] = (maximums[0] + minimums[0]) / 2.0;

  // Calculate bins center

  for (size_t i = 1; i < bins_number; i++) {
    minimums[i] = minimums[i - 1] + length;
    maximums[i] = maximums[i - 1] + length;

    centers[i] = (maximums[i] + minimums[i]) / 2.0;
  }

  // Calculate bins frequency

  const size_t this_size = this->size();

  for (int i = 0; i < (int)this_size; i++) {
    if(!missing_indices.contains(i)) {
      for (int j = 0; j < (int)bins_number - 1; j++) {
        if((*this)[i] >= minimums[j] && (*this)[i] < maximums[j]) {
          frequencies[j]++;
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

  return (histogram);
}

// size_t calculate_minimal_index(void) const method

/// Returns the index of the smallest element in the vector.

template <class T> size_t Vector<T>::calculate_minimal_index(void) const {
  const size_t this_size = this->size();

  T minimum = (*this)[0];
  size_t minimal_index = 0;

  for (size_t i = 1; i < this_size; i++) {
    if((*this)[i] < minimum) {
      minimum = (*this)[i];
      minimal_index = i;
    }
  }

  return (minimal_index);
}

// size_t calculate_maximal_index(void) const method

/// Returns the index of the largest element in the vector.

template <class T> size_t Vector<T>::calculate_maximal_index(void) const {
  const size_t this_size = this->size();

  T maximum = (*this)[0];
  size_t maximal_index = 0;

  for (size_t i = 1; i < this_size; i++) {
    if((*this)[i] > maximum) {
      maximum = (*this)[i];
      maximal_index = i;
    }
  }

  return (maximal_index);
}

// Vector<size_t> calculate_minimal_indices(const size_t&) const method

/// Returns the indices of the smallest elements in the vector.
/// @param number Number of minimal indices to be computed.

template <class T>
Vector<size_t>
Vector<T>::calculate_minimal_indices(const size_t &number) const {
  const size_t this_size = this->size();

  const Vector<size_t> rank = calculate_less_rank();

  Vector<size_t> minimal_indices(number);

  for (size_t i = 0; i < this_size; i++) {
    for (size_t j = 0; j < number; j++) {
      if(rank[i] == j) {
        minimal_indices[j] = i;
      }
    }
  }

  return (minimal_indices);
}

// Vector<size_t> calculate_maximal_indices(const size_t&) const method

/// Returns the indices of the largest elements in the vector.
/// @param number Number of maximal indices to be computed.

template <class T>
Vector<size_t>
Vector<T>::calculate_maximal_indices(const size_t &number) const {
  const size_t this_size = this->size();

  const Vector<size_t> rank = calculate_greater_rank();

  Vector<size_t> maximal_indices(number);

  for (size_t i = 0; i < this_size; i++) {
    for (size_t j = 0; j < number; j++) {
      if(rank[i] == j) {
        maximal_indices[j] = i;
      }
    }
  }

  return (maximal_indices);
}

// Vector<size_t> calculate_minimal_maximal_index(void) const method

/// Returns a vector with the indices of the smallest and the largest elements
/// in the vector.

template <class T>
Vector<size_t> Vector<T>::calculate_minimal_maximal_index(void) const {
  const size_t this_size = this->size();

  T minimum = (*this)[0];
  T maximum = (*this)[0];

  size_t minimal_index = 0;
  size_t maximal_index = 0;

  for (size_t i = 1; i < this_size; i++) {
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

  return (minimal_maximal_index);
}

// Vector<double> calculate_pow(const T&) const method

/// Returns a vector with the elements of this vector raised to a power
/// exponent.
/// @param exponent Pow exponent.

template <class T> Vector<T> Vector<T>::calculate_pow(const T &exponent) const {
  const size_t this_size = this->size();

  Vector<T> power(this_size);

  for (size_t i = 0; i < this_size; i++) {
    power[i] = pow((*this)[i], exponent);
  }

  return (power);
}

// Vector<double> calculate_competitive(void) const method

/// Returns the competitive vector of this vector,
/// whose elements are one the bigest element of this vector, and zero for the
/// other elements.

template <class T> Vector<T> Vector<T>::calculate_competitive(void) const {
  const size_t this_size = this->size();

  Vector<T> competitive(this_size, 0);

  const size_t maximal_index = calculate_maximal_index();

  competitive[maximal_index] = 1;

  return (competitive);
}

// Vector<T> calculate_softmax(void) const method

/// Returns the softmax vector of this vector,
/// whose elements sum one, and can be interpreted as probabilities.

template <class T> Vector<T> Vector<T>::calculate_softmax(void) const {
  const size_t this_size = this->size();

  Vector<T> softmax(this_size);

  T sum = 0;

  for (size_t i = 0; i < this_size; i++) {
    sum += exp((*this)[i]);
  }

  for (size_t i = 0; i < this_size; i++) {
    softmax[i] = exp((*this)[i]) / sum;
  }

  return (softmax);
}

// Matrix<T> calculate_softmax_Jacobian(void) const method

/// Returns the softmax Jacobian of this vector.

template <class T> Matrix<T> Vector<T>::calculate_softmax_Jacobian(void) const {
  const size_t this_size = this->size();

  Matrix<T> softmax_Jacobian(this_size, this_size);

  for (size_t i = 0; i < this_size; i++) {
    for (size_t j = 0; j < this_size; j++) {
      if(i == j) {
        softmax_Jacobian(i, i) = (*this)[i] * (1.0 - (*this)[i]);
      } else {
        softmax_Jacobian(i, i) = (*this)[i] * (*this)[j];
      }
    }
  }

  return (softmax_Jacobian);
}

// Vector<bool> calculate_binary(void) const method

/// This method converts the values of the vector to be binary.
/// The threshold value used is 0.5.

template <class T> Vector<bool> Vector<T>::calculate_binary(void) const {
  const size_t this_size = this->size();

  Vector<bool> binary(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < 0.5) {
      binary[i] = false;
    } else {
      binary[i] = true;
    }
  }

  return (binary);
}

// Vector<T> calculate_cumulative(void) const method

/// Return the cumulative vector of this vector,
/// where each element is summed up with all the previous ones.

template <class T> Vector<T> Vector<T>::calculate_cumulative(void) const {
  const size_t this_size = this->size();

  Vector<T> cumulative(this_size);

  if(this_size > 0) {
    cumulative[0] = (*this)[0];

    for (size_t i = 1; i < this_size; i++) {
      cumulative[i] = cumulative[i - 1] + (*this)[i];
    }
  }

  return (cumulative);
}

// size_t calculate_cumulative_index(const T&) const method

/// This method applies only to cumulative vectors.
/// It returns the index of the first element which is greater than a given
/// value.
/// @param value Value.

template <class T>
size_t Vector<T>::calculate_cumulative_index(const T &value) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t calculate_cumulative_index(const T&) const.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

  T cumulative_value = (*this)[this_size - 1];

  if(value > cumulative_value) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t calculate_cumulative_index(const T&) const.\n"
           << "Value (" << value << ") must be less than cumulative value ("
           << cumulative_value << ").\n";

    throw std::logic_error(buffer.str());
  }

  for (size_t i = 1; i < this_size; i++) {
    if((*this)[i] < (*this)[i - 1]) {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "int calculate_cumulative_index(const T&) const.\n"
             << "Vector elements must be crescent.\n";

      throw std::logic_error(buffer.str());
    }
  }

#endif

  if(value <= (*this)[0]) {
    return (0);
  }

  for (size_t i = 1; i < this_size; i++) {
    if(value > (*this)[i - 1] && value <= (*this)[i]) {
      return (i);
    }
  }

  return (this_size - 1);
}

// size_t calculate_closest_index(const T&) const method

/// Returns the index of the closest element in the vector to a given value.

template <class T>
size_t Vector<T>::calculate_closest_index(const T &value) const {
  const Vector<T> difference = (*this - value).calculate_absolute_value();

  const size_t closest_index = difference.calculate_minimal_index();

  return (closest_index);
}

// T calculate_sum(void) const method

/// Returns the sum of the elements in the vector.

template <class T> T Vector<T>::calculate_sum(void) const {
  const size_t this_size = this->size();

  T sum = 0;

  for (size_t i = 0; i < this_size; i++) {
    sum += (*this)[i];
  }

  return (sum);
}

// T calculate_partial_sum(const Vector<size_t>&) const method

/// Returns the sum of the elements with the given indices.
/// @param indices Indices of the elementes to sum.

template <class T>
T Vector<T>::calculate_partial_sum(const Vector<size_t> &indices) const {
  const size_t this_size = this->size();

  T sum = 0;

  for (size_t i = 0; i < this_size; i++) {
    if(indices.contains(i)) {
      sum += (*this)[i];
    }
  }

  return (sum);
}

// T calculate_sum_missing_values(const Vector<size_t>&) const method

/// Returns the sum of the elements in the vector.

template <class T>
T Vector<T>::calculate_sum_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

  T sum = 0;

  for (size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      sum += (*this)[i];
    }
  }

  return (sum);
}

// T calculate_product(void) const method

/// Returns the product of the elements in the vector.

template <class T> T Vector<T>::calculate_product(void) const {
  const size_t this_size = this->size();

  T product = 1;

  for (size_t i = 0; i < this_size; i++) {
    product *= (*this)[i];
  }

  return (product);
}

// double calculate_mean(void) const method

/// Returns the mean of the elements in the vector.

template <class T> double Vector<T>::calculate_mean(void) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mean(void) const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  const T sum = calculate_sum();

  const double mean = sum / (double)this_size;

  return (mean);
}

// double calculate_variance(void) method

/// Returns the variance of the elements in the vector.

template <class T> double Vector<T>::calculate_variance(void) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_variance(void) const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return (0.0);
  }

  double sum = 0.0;
  double squared_sum = 0.0;

  for (size_t i = 0; i < this_size; i++) {
    sum += (*this)[i];
    squared_sum += (*this)[i] * (*this)[i];
  }

  const double numerator = squared_sum - (sum * sum) / this_size;
  const double denominator = this_size - 1.0;

  return (numerator / denominator);
}


// double calculate_covariance(const Vector<double>&) const method

/// Returns the covariance of this vector and other vector

template<class T>
double Vector<T>::calculate_covariance(const Vector<double>& other_vector) const
{
   const size_t this_size = this->size();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

     if(this_size == 0) {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Vector Template.\n"
              << "double calculate_covariance(const Vector<double>&) const method.\n"
              << "Size must be greater than zero.\n";

       throw std::logic_error(buffer.str());
     }

   #endif

   // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

         if(this_size != other_vector.size()) {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: Vector Template.\n"
                    << "double calculate_covariance(const Vector<double>&) const method.\n"
                    << "Size of this vectro must be equal to size of other vector.\n";

             throw std::logic_error(buffer.str());
         }

    #endif

     if(this_size == 1)
     {
         return 0.0;
     }

     const double this_mean = this->calculate_mean();
     const double other_mean = other_vector.calculate_mean();

     double numerator = 0.0;
     double denominator = (double)(this_size-1);

     for(size_t i = 0; i < this_size; i++)
     {
         numerator += ((*this)[i]-this_mean)*(other_vector[i]-other_mean);
     }

     return (numerator/denominator);
}


// double calculate_standard_deviation(void) const method

/// Returns the variance of the elements in the vector.

template <class T> double Vector<T>::calculate_standard_deviation(void) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_standard_deviation(void) const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  return (sqrt(calculate_variance()));
}

// double calculate_asymmetry(void) const method

/// Returns the asymmetry of the elements in the vector

template <class T> double Vector<T>::calculate_asymmetry(void) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_asymmetry(void) const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for (size_t i = 0; i < this_size; i++)
  {
    sum += ((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean);
  }

  const double numerator = sum / this_size;
  const double denominator =
      standard_deviation * standard_deviation * standard_deviation;

  return (numerator / denominator);
}

// double calculate_kurtosis(void) const method

/// Returns the kurtosis value of the elements in the vector.

template <class T> double Vector<T>::calculate_kurtosis(void) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_kurtosis(void) const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for (size_t i = 0; i < this_size; i++)
  {
    sum += ((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean);
  }

  const double numerator = sum/this_size;
  const double denominator = standard_deviation*standard_deviation*standard_deviation*standard_deviation;

  return ((numerator/denominator)-3.0);
}

// Vector<double> calculate_mean_standard_deviation(void) const method

/// Returns the mean and the standard deviation of the elements in the vector.

template <class T>
Vector<double> Vector<T>::calculate_mean_standard_deviation(void) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mean_standard_deviation(void).\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  const double mean = calculate_mean();
  const double standard_deviation = calculate_standard_deviation();

  Vector<double> mean_standard_deviation(2);
  mean_standard_deviation[0] = mean;
  mean_standard_deviation[1] = standard_deviation;

  return (mean_standard_deviation);
}

// double calculate_median(void) const

/// Returns the median of the elements in the vector

template <class T> double Vector<T>::calculate_median(void) const {
  const size_t this_size = this->size();

  Vector<T> sorted_vector(*this);

  std::sort(sorted_vector.begin(), sorted_vector.end(), std::less<double>());

  size_t median_index;

  if(this_size % 2 == 0) {
    median_index = (size_t)(this_size / 2);

    return ((sorted_vector[median_index] + sorted_vector[median_index]) / 2.0);
  } else {
    median_index = (size_t)(this_size / 2);

    return (sorted_vector[median_index + 1]);
  }
}

// Vector<double> calculate_quartiles(void) const

/// Returns the quarters of the elements in the vector.

template <class T> Vector<double> Vector<T>::calculate_quartiles(void) const {
  const size_t this_size = this->size();

  Vector<T> sorted_vector(*this);

  std::sort(sorted_vector.begin(), sorted_vector.end(), std::less<double>());

  Vector<double> quartiles(4);

  if(this_size % 2 == 0) {
    quartiles[0] = (sorted_vector[this_size / 4] + sorted_vector[this_size / 4 + 1]) / 2;
    quartiles[1] = (sorted_vector[this_size * 2 / 4] +
                   sorted_vector[this_size * 2 / 4 + 1]) /
                  2;
    quartiles[2] = (sorted_vector[this_size * 3 / 4] +
                   sorted_vector[this_size * 3 / 4 + 1]) /
                  2;
    quartiles[3] = calculate_maximum();
  } else {
    quartiles[0] = sorted_vector[this_size / 4 /*+ 1*/];
    quartiles[1] = sorted_vector[this_size * 2 / 4 /*+ 1*/];
    quartiles[2] = sorted_vector[this_size * 3 / 4 /*+ 1*/];
    quartiles[3] = calculate_maximum();
  }

  return (quartiles);
}

// Vector<double> calculate_quartiles_missing_values(const Vector<size_t>&) const

/// Returns the quarters of the elements in the vector when there are missing values.
/// @param missing_indices Vector with the indices of the missing values.

template <class T>
Vector<double> Vector<T>::calculate_quartiles_missing_values(const Vector<size_t> & missing_indices) const
{
    const size_t this_size = this->size();
    const size_t missing_indices_number = missing_indices.size();

    Vector<T> sorted_vector(*this);

    for(size_t i = 0; i < missing_indices_number; i++)
    {
        sorted_vector.remove_element(missing_indices[i]);
    }

    std::sort(sorted_vector.begin(), sorted_vector.end(), std::less<double>());

    const size_t actual_size = this_size - missing_indices_number;

    Vector<double> quartiles(4);

    if(actual_size % 2 == 0) {
      quartiles[0] = (sorted_vector[actual_size / 4] + sorted_vector[actual_size / 4 + 1]) / 2;
      quartiles[1] = (sorted_vector[actual_size * 2 / 4] +
                     sorted_vector[actual_size * 2 / 4 + 1]) /
                    2;
      quartiles[2] = (sorted_vector[actual_size * 3 / 4] +
                     sorted_vector[actual_size * 3 / 4 + 1]) /
                    2;
      quartiles[3] = sorted_vector[actual_size - 1];
    } else {
      quartiles[0] = sorted_vector[actual_size / 4 /*+ 1*/];
      quartiles[1] = sorted_vector[actual_size * 2 / 4 /*+ 1*/];
      quartiles[2] = sorted_vector[actual_size * 3 / 4 /*+ 1*/];
      quartiles[3] = sorted_vector[actual_size - 1];
    }

    return (quartiles);
}

// double calculate_mean_missing_values(const Vector<size_t>&) const method

/// Returns the mean of the elements in the vector.

template <class T>
double Vector<T>::calculate_mean_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_mean_missing_values(const Vector<size_t>&) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  T sum = 0;

  size_t count = 0;

  for (size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      sum += (*this)[i];
      count++;
    }
  }

  const double mean = sum / (double)count;

  return (mean);
}

// double calculate_variance_missing_values(const Vector<size_t>&) method

/// Returns the variance of the elements in the vector.

template <class T>
double Vector<T>::calculate_variance_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_variance_missing_values(const Vector<size_t>&) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  double sum = 0.0;
  double squared_sum = 0.0;

  size_t count = 0;

  for (size_t i = 0; i < this_size; i++) {
    if(!missing_indices.contains(i)) {
      sum += (*this)[i];
      squared_sum += (*this)[i] * (*this)[i];

      count++;
    }
  }

  if(count <= 1) {
    return (0.0);
  }

  const double numerator = squared_sum - (sum * sum) / count;
  const double denominator = this_size - 1.0;

  return (numerator / denominator);
}

// double calculate_weighted_mean(const Vector<double>&) const method

/// Returns the weighted mean of the vector.
/// @param weights Weights of the elements of the vector in the mean.

template <class T>
double Vector<T>::calculate_weighted_mean(const Vector<double> & weights) const
{
    const size_t this_size = this->size();

  // Control sentence (if debug)

  #ifdef __OPENNN_DEBUG__

    if(this_size == 0) {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_weighted_mean(const Vector<double>&) const method.\n"
             << "Size must be greater than zero.\n";

      throw std::logic_error(buffer.str());
    }

    const size_t weights_size = weights.size();

    if(this_size != weights_size) {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_weighted_mean(const Vector<double>&) "
                "const method.\n"
             << "Size of weights must be equal to vector size.\n";

      throw std::logic_error(buffer.str());
    }
  #endif

    double weights_sum = 0;

    T sum = 0;

    for (size_t i = 0; i < this_size; i++)
    {
        sum += weights[i]*(*this)[i];
        weights_sum += weights[i];
    }

    const double mean = sum / weights_sum;

    return (mean);
}

// double calculate_standard_deviation_missing_values(const Vector<size_t>&)
// method

/// Returns the standard deviation of the elements in the vector.

template <class T>
double Vector<T>::calculate_standard_deviation_missing_values(
    const Vector<size_t> &missing_indices) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_standard_deviation_missing_values(const "
              "Vector<size_t>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  return (sqrt(calculate_variance_missing_values(missing_indices)));
}


// double calculate_asymmetry_missing_values(Vector<size_t>&) const method

/// Returns the asymmetry of the elements in the vector.

template <class T>
double Vector<T>::calculate_asymmetry_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_asymmetry_missing_values(const "
              "Vector<size_t>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for (size_t i = 0; i < this_size; i++)
  {
    if(!missing_indices.contains(i))
    {
      sum += ((*this)[i] - mean) * ((*this)[i] - mean) * ((*this)[i] - mean);
    }
  }

  const double numerator = sum / this_size;
  const double denominator =
      standard_deviation * standard_deviation * standard_deviation;

  return (numerator / denominator);
}

// double calculate_kurtosis_missing_values(Vector<size_t>&) const method

/// Returns the kurtosis of the elements in the vector.

template <class T>
double Vector<T>::calculate_kurtosis_missing_values(
    const Vector<size_t> &missing_indices) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_kurtosis_missing_values(const Vector<size_t>&) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  if(this_size == 1)
  {
    return 0.0;
  }

  const double standard_deviation = calculate_standard_deviation();

  const double mean = calculate_mean();

  double sum = 0.0;

  for (size_t i = 0; i < this_size; i++)
  {
    if(!missing_indices.contains(i))
    {
      sum += ((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean)*((*this)[i] - mean);
    }
  }

  const double numerator = sum / this_size;
  const double denominator = standard_deviation*standard_deviation*standard_deviation*standard_deviation;

  return ((numerator/denominator)-3.0);
}

// Statistics<T> calculate_statistics(void) const method

/// Returns the minimum, maximum, mean and standard deviation of the elements in
/// the vector.

template <class T> Statistics<T> Vector<T>::calculate_statistics(void) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_statistics(void).\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Statistics<T> statistics;

  statistics.minimum = calculate_minimum();
  statistics.maximum = calculate_maximum();
  statistics.mean = calculate_mean();
  statistics.standard_deviation = calculate_standard_deviation();

  return (statistics);
}

// Statistics<T> calculate_statistics_missing_values(const Vector<size_t>&)
// const method

/// Returns the minimum, maximum, mean and standard deviation of the elements in
/// the vector.

template <class T>
Statistics<T> Vector<T>::calculate_statistics_missing_values(
    const Vector<size_t> &missing_indices) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_statistics_missing_values(const "
              "Vector<size_t>&).\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Statistics<T> statistics;

  statistics.minimum = calculate_minimum_missing_values(missing_indices);
  statistics.maximum = calculate_maximum_missing_values(missing_indices);
  statistics.mean = calculate_mean_missing_values(missing_indices);
  statistics.standard_deviation =
      calculate_standard_deviation_missing_values(missing_indices);

  return (statistics);
}

// Vector<double> calculate_shape_parameters(void) const

/// Returns a vector with the asymmetry and the kurtosis values of the elements
/// in the vector.

template <class T>
Vector<double> Vector<T>::calculate_shape_parameters(void) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<double> calculate_shape_parameters(void).\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<double> shape_parameters(2);

  shape_parameters[0] = calculate_asymmetry();
  shape_parameters[1] = calculate_kurtosis();

  return (shape_parameters);
}

// Vector<double> calculate_shape_parameters_missing_values(const
// Vector<size_t>&) const

/// Returns a vector with the asymmetry and the kurtosis values of the elements
/// in the vector.

template <class T>
Vector<double> Vector<T>::calculate_shape_parameters_missing_values(
    const Vector<size_t> &missing_values) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(this_size == 0) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_shape_parameters_missing_values(const "
              "Vector<size_t>&).\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<double> shape_parameters(2);

  shape_parameters[0] = calculate_asymmetry_missing_values(missing_values);
  shape_parameters[1] = calculate_kurtosis_missing_values(missing_values);

  return (shape_parameters);
}

// Vector<double> calculate_box_plots(void) const

/// Returns the box and whispers for a vector.

template <class T>
Vector<double> Vector<T>::calculate_box_plots(void) const {
  Vector<double> box_plots(5);

  Vector<double> quartiles = calculate_quartiles();

  box_plots[0] = calculate_minimum();
  box_plots[1] = quartiles[0];
  box_plots[2] = quartiles[1];
  box_plots[3] = quartiles[2];
  box_plots[4] = quartiles[3];

  return (box_plots);
}

// Vector<double> calculate_box_plots_missing_values(const Vector<size_t>&) const

/// Returns the box and whispers for a vector when there are missing values.
/// @param missing_indices Vector with the indices of the missing values.

template <class T>
Vector<double> Vector<T>::calculate_box_plots_missing_values(const Vector<size_t> & missing_indices) const
{
    Vector<double> box_plots(5);

    Vector<double> quartiles = calculate_quartiles_missing_values(missing_indices);

    box_plots[0] = calculate_minimum();
    box_plots[1] = quartiles[0];
    box_plots[2] = quartiles[1];
    box_plots[3] = quartiles[2];
    box_plots[4] = quartiles[3];

    return (box_plots);
}

// double calculate_norm(void) const method

/// Returns the vector norm.

template <class T> double Vector<T>::calculate_norm(void) const {
  const size_t this_size = this->size();

  // Control sentence (if debug)

  double norm = 0.0;

  for (size_t i = 0; i < this_size; i++) {
    norm += (*this)[i] * (*this)[i];
  }

  norm = sqrt(norm);

  return (norm);
}

// Vector<T> calculate_norm_gradient(void) const method

/// Returns the gradient of the vector norm.

template <class T> Vector<T> Vector<T>::calculate_norm_gradient(void) const {
  const size_t this_size = this->size();

  // Control sentence (if debug)

  Vector<T> gradient(this_size);

  const double norm = calculate_norm();

  if(norm == 0.0) {
    gradient.initialize(0.0);
  } else {
    gradient = (*this) / norm;
  }

  return (gradient);
}

// Matrix<T> calculate_norm_Hessian(void) const method

/// Returns the Hessian of the vector norm.

template <class T> Matrix<T> Vector<T>::calculate_norm_Hessian(void) const {
  const size_t this_size = this->size();

  // Control sentence (if debug)

  Matrix<T> Hessian(this_size, this_size);

  const double norm = calculate_norm();

  if(norm == 0.0) {
    Hessian.initialize(0.0);
  } else {
    //       Hessian = (*this).direct(*this)/pow(norm, 3);
    Hessian = (*this).direct(*this) / (norm * norm * norm);
  }

  return (Hessian);
}

// double calculate_p_norm(const double&) const method

/// Returns the vector p-norm.

template <class T> double Vector<T>::calculate_p_norm(const double &p) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  std::ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_p_norm(const double&) const method.\n"
           << "p value must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  // Control sentence (if debug)

  double norm = 0.0;

  for (size_t i = 0; i < this_size; i++) {
    norm += pow(fabs((*this)[i]), p);
  }

  norm = pow(norm, 1.0 / p);

  return (norm);
}

// Vector<double> calculate_p_norm_gradient(const double&) const method

/// Returns the gradient of the vector norm.

template <class T>
Vector<double> Vector<T>::calculate_p_norm_gradient(const double &p) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  std::ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<double> calculate_p_norm_gradient(const double&) const "
              "method.\n"
           << "p value must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  // Control sentence (if debug)

  Vector<double> gradient(this_size);

  const double p_norm = calculate_p_norm(p);

  if(p_norm == 0.0) {
    gradient.initialize(0.0);
  } else {
    for (size_t i = 0; i < this_size; i++) {
      gradient[i] =
          (*this)[i] * pow(fabs((*this)[i]), p - 2.0) / pow(p_norm, p - 1.0);
    }

    //       gradient = (*this)*(*this).calculate_pow(p-2.0)/pow(p_norm, p-1.0);
  }

  return (gradient);
}

// double calculate_normalized(void) const method

/// Returns this vector divided by its norm.

template <class T> Vector<T> Vector<T>::calculate_normalized(void) const {
  const size_t this_size = (*this).size();

  Vector<T> normalized(this_size);

  const double norm = calculate_norm();

  if(norm == 0.0) {
    normalized.initialize(0.0);
  } else {
    normalized = (*this) / norm;
  }

  return (normalized);
}
/*
// double calculate_distance(const Vector<double>&) const method

/// Returns the distance between the elements of this vector and the elements of
/// another vector.
/// @param other_vector Other vector.

template <class T>
double Vector<T>::calculate_distance(const Vector<double> &other_vector) const {
  return (sqrt(calculate_sum_squared_error(other_vector)));
}
*/
// double calculate_distance(const Vector<double>&) const method

/// Returns the distance between the elements of this vector and the elements of
/// another vector.
/// @param other_vector Other vector.

template <class T>
double Vector<T>::calculate_distance(const Vector<T> &other_vector) const {

    const size_t this_size = this->size();
#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_distance(const Vector<T>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw std::logic_error(buffer.str());
  }

#endif
    double distance = 0.0;
    double error;

    for (size_t i = 0; i < this_size; i++) {
        error = (*this)[i] - other_vector[i];

        distance += error * error;
    }

    return (sqrt(distance));
}

// double calculate_sum_squared_error(const Vector<double>&) const method

/// Returns the sum squared error between the elements of this vector and the
/// elements of another vector.
/// @param other_vector Other vector.

template <class T>
double Vector<T>::calculate_sum_squared_error(
    const Vector<double> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_sum_squared_error(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  double sum_squared_error = 0.0;
  double error;

  for (size_t i = 0; i < this_size; i++) {
    error = (*this)[i] - other_vector[i];

    sum_squared_error += error * error;
  }

  return (sum_squared_error);
}

// double calculate_sum_squared_error(const Matrix<T>&, const size_t&, const Vector<size_t>&) const method

/// Returns the sum squared error between the elements of this vector and the
/// elements of a row of a matrix.
/// @param matrix Matrix to compute the error .
/// @param row_index Index of the row of the matrix.
/// @param column_indices Indices of the columns of the matrix to evaluate.

template <class T>
double Vector<T>::calculate_sum_squared_error(
    const Matrix<T> &matrix, const size_t &row_index,
    const Vector<size_t> &column_indices) const {

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

   const size_t this_size = this->size();
   const size_t other_size = column_indices.size();

   if(other_size != this_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_sum_squared_error(const Matrix<T>&, const size_t&, const Vector<size_t>&) const method.\n"
             << "Size must be equal to this size.\n";

      throw std::logic_error(buffer.str());
   }

#endif

  double sum_squared_error = 0.0;
  double error;

  const size_t size = column_indices.size();

  for (size_t i = 0; i < size; i++) {
    error = (*this)[i] - matrix(row_index, column_indices[i]);

    sum_squared_error += error * error;
  }

  return (sum_squared_error);
}

// double calculate_Minkowski_error(const Vector<double>&) const method

/// Returns the Minkowski squared error between the elements of this vector and
/// the elements of another vector.
/// @param other_vector Other vector.
/// @param Minkowski_parameter Minkowski exponent.

template <class T>
double
Vector<T>::calculate_Minkowski_error(const Vector<double> &other_vector,
                                     const double &Minkowski_parameter) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  std::ostringstream buffer;

  if(this_size == 0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_Minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "Size must be greater than zero.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_Minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "Other size must be equal to this size.\n";

    throw std::logic_error(buffer.str());
  }

  // Control sentence

  if(Minkowski_parameter < 1.0 || Minkowski_parameter > 2.0) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "double calculate_Minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "The Minkowski parameter must be comprised between 1 and 2\n";

    throw std::logic_error(buffer.str());
  }

#endif

  double Minkowski_error = 0.0;

  for (size_t i = 0; i < this_size; i++) {
    Minkowski_error +=
        pow(fabs((*this)[i] - other_vector[i]), Minkowski_parameter);
  }

  Minkowski_error = pow(Minkowski_error, 1.0 / Minkowski_parameter);

  return (Minkowski_error);
}

// T calculate_linear_correlation(const Vector<T>&) const method

/// Calculates the linear correlation coefficient (R-value) between another
/// vector and this vector.
/// @param other Vector for computing the linear correlation with this vector.

template <class T>
double Vector<T>::calculate_linear_correlation(const Vector<T> &other) const {
  size_t n = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other.size();

  std::ostringstream buffer;

  if(other_size != n) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "T calculate_linear_correlation(const Vector<T>&) const method.\n"
           << "Other size must be equal to this size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  double s_x = 0;
  double s_y = 0;

  double s_xx = 0;
  double s_yy = 0;

  double s_xy = 0;

  for (size_t i = 0; i < n; i++) {
    s_x += other[i];
    s_y += (*this)[i];

    s_xx += other[i] * other[i];
    s_yy += (*this)[i] * (*this)[i];

    s_xy += other[i] * (*this)[i];
  }

#ifdef __OPENNN_MPI__

  int n_send = (int)n;
  int n_mpi;

  double s_x_mpi = s_x;
  double s_y_mpi = s_y;

  double s_xx_mpi = s_xx;
  double s_yy_mpi = s_yy;

  double s_xy_mpi = s_xy;

  MPI_Allreduce(&n_send,&n_mpi,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  MPI_Allreduce(&s_x_mpi,&s_x,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&s_y_mpi,&s_y,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  MPI_Allreduce(&s_xx_mpi,&s_xx,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&s_yy_mpi,&s_yy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  MPI_Allreduce(&s_xy_mpi,&s_xy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  n = n_mpi;

#endif

  double linear_correlation;

  if(s_x == 0 && s_y == 0 && s_xx == 0 && s_yy == 0 && s_xy == 0) {
    linear_correlation = 1;
  } else {
    const double numerator = (n * s_xy - s_x * s_y);

    const double radicand = (n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y);

    if(radicand <= 0.0) {
      return (0);
    }

    const double denominator = sqrt(radicand);

    if(denominator < 1.0e-50) {
      linear_correlation = 0;
    } else {
      linear_correlation = numerator / denominator;
    }
  }

  return (linear_correlation);
}

// T calculate_linear_correlation_missing_values(const Vector<T>&, const
// Vector<size_t>&) const method

/// Calculates the linear correlation coefficient (R-value) between another
/// vector and this vector when there are missing values in the data.
/// @param other Vector for computing the linear correlation with this vector.
/// @param missing_indices Vector with the indices of the missing values.

template <class T>
T Vector<T>::calculate_linear_correlation_missing_values(
    const Vector<T> &other, const Vector<size_t> &missing_indices) const {
  const size_t n = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other.size();

  std::ostringstream buffer;

  if(other_size != n) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "T calculate_linear_correlation(const Vector<T>&) const method.\n"
           << "Other size must be equal to this size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  size_t count = 0;

  T s_x = 0;
  T s_y = 0;

  T s_xx = 0;
  T s_yy = 0;

  T s_xy = 0;

  for (size_t i = 0; i < n; i++) {
    if(!missing_indices.contains(i)) {
      s_x += other[i];
      s_y += (*this)[i];

      s_xx += other[i] * other[i];
      s_yy += (*this)[i] * (*this)[i];

      s_xy += other[i] * (*this)[i];

      count++;
    }
  }

  T linear_correlation;

  if(s_x == 0 && s_y == 0 && s_xx == 0 && s_yy == 0 && s_xy == 0) {
    linear_correlation = 1;
  } else {
    const double numerator = (count * s_xy - s_x * s_y);
    const double denominator =
        sqrt((count * s_xx - s_x * s_x) * (count * s_yy - s_y * s_y));

    if(denominator < 1.0e-50) {
      linear_correlation = 0;
    } else {
      linear_correlation = numerator / denominator;
    }
  }

  return (linear_correlation);
}

// Vector<double> calculate_autocorrelation(const size_t&) const

/// Calculates autocorrelation for a given number of maximum lags.
/// @param lags_number Maximum lags number.

template <class T>
Vector<double>
Vector<T>::calculate_autocorrelation(const size_t &lags_number) const {
  Vector<double> autocorrelation(lags_number);

  const double mean = calculate_mean();

  const size_t this_size = this->size();

  double numerator = 0;
  double denominator = 0;

  for (size_t i = 0; i < lags_number; i++) {
    for (size_t j = 0; j < this_size - i; j++) {
      numerator +=
          (((*this)[j] - mean) * ((*this)[j + i] - mean)) / (this_size - i);
    }
    for (size_t j = 0; j < this_size; j++) {
      denominator += (((*this)[j] - mean) * ((*this)[j] - mean)) / this_size;
    }

    if(denominator == 0.0) {
      autocorrelation[i] = 1.0;
    } else {
      autocorrelation[i] = numerator / denominator;
    }

    numerator = 0;
    denominator = 0;
  }

  return autocorrelation;
}

// Vector<double> calculate_cross_correlation(cosnt Vector<double>&, const
// size_t&) const

/// Calculates the cross-correlation between this vector and another given
/// vector.
/// @param other Other vector.
/// @param maximum_lags_number Maximum lags for which cross-correlation is
/// calculated.

template <class T>
Vector<double> Vector<T>::calculate_cross_correlation(
    const Vector<double> &other, const size_t &maximum_lags_number) const {
  if(other.size() != this->size()) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<double calculate_cross_correlation(const "
              "Vector<double>&) method.\n"
           << "Both vectors must have the same size.\n";

    throw std::logic_error(buffer.str());
  }

  Vector<double> cross_correlation(maximum_lags_number);

  const double this_mean = calculate_mean();
  const double other_mean = other.calculate_mean();

  const size_t this_size = this->size();

  double numerator = 0;

  double this_denominator = 0;
  double other_denominator = 0;
  double denominator = 0;

  for (size_t i = 0; i < maximum_lags_number; i++) {
    numerator = 0;
    this_denominator = 0;
    other_denominator = 0;

    for (size_t j = 0; j < this_size; j++) {
      this_denominator += ((*this)[j] - this_mean) * ((*this)[j] - this_mean);
      other_denominator += (other[j] - other_mean) * (other[j] - other_mean);
    }

    denominator = sqrt(this_denominator * other_denominator);

    for (size_t j = 0; j < this_size - i; j++) {
      numerator += ((*this)[j] - this_mean) * (other[j + i] - other_mean);
    }

    if(denominator == 0.0) {
      cross_correlation[i] = 0.0;
    } else {
      cross_correlation[i] = numerator / denominator;
    }
  }

  return (cross_correlation);
}

// LinearRegressionParameters<T> calculate_linear_regression_parameters(const
// Vector<T>&) const method

/// Calculates the linear regression parameters (intercept, slope and
/// correlation) between another vector and this vector.
/// It returns a linear regression parameters structure.
/// @param other Other vector for the linear regression analysis.

template <class T>
LinearRegressionParameters<T> Vector<T>::calculate_linear_regression_parameters(
    const Vector<T> &other) const {
  const size_t n = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other.size();

  std::ostringstream buffer;

  if(other_size != n) {
    buffer << "OpenNN Exception: Vector Template.\n"
           << "LinearRegressionParameters<T> "
              "calculate_linear_regression_parameters(const Vector<T>&) const "
              "method.\n"
           << "Other size must be equal to this size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  T s_x = 0;
  T s_y = 0;

  T s_xx = 0;
  T s_yy = 0;

  T s_xy = 0;

  for (size_t i = 0; i < n; i++) {
    s_x += other[i];
    s_y += (*this)[i];

    s_xx += other[i] * other[i];
    s_yy += (*this)[i] * (*this)[i];

    s_xy += other[i] * (*this)[i];
  }

  LinearRegressionParameters<T> linear_regression_parameters;

  if(s_x == 0 && s_y == 0 && s_xx == 0 && s_yy == 0 && s_xy == 0) {
    linear_regression_parameters.intercept = 0;

    linear_regression_parameters.slope = 0;

    linear_regression_parameters.correlation = 1;
  } else {
    linear_regression_parameters.intercept =
        (s_y * s_xx - s_x * s_xy) / (n * s_xx - s_x * s_x);

    linear_regression_parameters.slope =
        (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);

    linear_regression_parameters.correlation =
        (n * s_xy - s_x * s_y) /
        sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));
  }

  return (linear_regression_parameters);
}

// void calculate_absolute_value(void) const method

/// Returns a vector with the absolute values of the current vector.

template <class T> Vector<T> Vector<T>::calculate_absolute_value(void) const {
  const size_t this_size = this->size();

  Vector<T> absolute_value(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > 0) {
      absolute_value[i] = (*this)[i];
    } else {
      absolute_value[i] = -(*this)[i];
    }
  }

  return (absolute_value);
}

// void apply_absolute_value(void) method

/// Sets the elements of the vector to their absolute values.

template <class T> void Vector<T>::apply_absolute_value(void) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < 0) {
      (*this)[i] = -(*this)[i];
    }
  }
}

// Vector<T> calculate_lower_bounded(const T&) const method

/// Returns a vector with the bounded elements from below of the current vector.
/// @param lower_bound Lower bound values.

template <class T>
Vector<T> Vector<T>::calculate_lower_bounded(const T &lower_bound) const {
  const size_t this_size = this->size();

  Vector<T> bounded_vector(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
      bounded_vector[i] = lower_bound;
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return (bounded_vector);
}

// Vector<T> calculate_lower_bounded(const Vector<T>&) const method

/// Returns a vector with the bounded elements from above of the current vector.
/// @param lower_bound Lower bound values.

template <class T>
Vector<T>
Vector<T>::calculate_lower_bounded(const Vector<T> &lower_bound) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t lower_bound_size = lower_bound.size();

  if(lower_bound_size != this_size) {
    std::ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "Vector<T> calculate_lower_bounded(const Vector<T>&) const method.\n"
        << "Lower bound size must be equal to vector size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> bounded_vector(this_size);

  // Apply lower bound

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
      bounded_vector[i] = lower_bound[i];
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return (bounded_vector);
}

// Vector<T> calculate_upper_bounded(const T&) const method

/// This method bounds the elements of the vector if they fall above an upper
/// bound value.
/// @param upper_bound Upper bound value.

template <class T>
Vector<T> Vector<T>::calculate_upper_bounded(const T &upper_bound) const {
  const size_t this_size = this->size();

  Vector<T> bounded_vector(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound) {
      bounded_vector[i] = upper_bound;
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return (bounded_vector);
}

// Vector<T> calculate_upper_bounded(const Vector<T>&) const method

/// This method bounds the elements of the vector if they fall above their
/// corresponding upper bound values.
/// @param upper_bound Upper bound values.

template <class T>
Vector<T>
Vector<T>::calculate_upper_bounded(const Vector<T> &upper_bound) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t upper_bound_size = upper_bound.size();

  if(upper_bound_size != this_size) {
    std::ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "Vector<T> calculate_upper_bounded(const Vector<T>&) const method.\n"
        << "Upper bound size must be equal to vector size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> bounded_vector(this_size);

  // Apply upper bound

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound[i]) {
      bounded_vector[i] = upper_bound[i];
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return (bounded_vector);
}

// Vector<T> calculate_lower_upper_bounded(const T&, const T&) const method

/// This method bounds the elements of the vector if they fall above or below
/// their lower or upper
/// bound values, respectively.
/// @param lower_bound Lower bound value.
/// @param upper_bound Upper bound value.

template <class T>
Vector<T> Vector<T>::calculate_lower_upper_bounded(const T &lower_bound,
                                                   const T &upper_bound) const {
  const size_t this_size = this->size();

  Vector<T> bounded_vector(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
      bounded_vector[i] = lower_bound;
    } else if((*this)[i] > upper_bound) {
      bounded_vector[i] = upper_bound;
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return (bounded_vector);
}

// Vector<T> calculate_lower_upper_bounded(const Vector<T>&, const Vector<T>&)
// const method

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

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t lower_bound_size = lower_bound.size();
  const size_t upper_bound_size = upper_bound.size();

  if(lower_bound_size != this_size || upper_bound_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> calculate_lower_upper_bounded(const Vector<T>&, const "
              "Vector<T>&) const method.\n"
           << "Lower and upper bound sizes must be equal to vector size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> bounded_vector(this_size);

  // Apply lower and upper bounds

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
      bounded_vector[i] = lower_bound[i];
    } else if((*this)[i] > upper_bound[i]) {
      bounded_vector[i] = upper_bound[i];
    } else {
      bounded_vector[i] = (*this)[i];
    }
  }

  return (bounded_vector);
}

// void apply_lower_bound(const T&) method

/// Sets the elements of the vector to a given value if they fall below that
/// value.
/// @param lower_bound Lower bound value.

template <class T> void Vector<T>::apply_lower_bound(const T &lower_bound) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
      (*this)[i] = lower_bound;
    }
  }
}

// void apply_lower_bound(const Vector<T>&) method

/// Sets the elements of the vector to given values if they fall below that
/// values.
/// @param lower_bound Lower bound values.

template <class T>
void Vector<T>::apply_lower_bound(const Vector<T> &lower_bound) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
      (*this)[i] = lower_bound[i];
    }
  }
}

// void apply_upper_bound(const T&) method

/// Sets the elements of the vector to a given value if they fall above that
/// value.
/// @param upper_bound Upper bound value.

template <class T> void Vector<T>::apply_upper_bound(const T &upper_bound) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound) {
      (*this)[i] = upper_bound;
    }
  }
}

// void apply_upper_bound(const Vector<T>&) method

/// Sets the elements of the vector to given values if they fall above that
/// values.
/// @param upper_bound Upper bound values.

template <class T>
void Vector<T>::apply_upper_bound(const Vector<T> &upper_bound) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] > upper_bound[i]) {
      (*this)[i] = upper_bound[i];
    }
  }
}

// void apply_lower_upper_bounds(const T&, const T&) method

/// Sets the elements of the vector to a given lower bound value if they fall
/// below that value,
/// or to a given upper bound value if they fall above that value.
/// @param lower_bound Lower bound value.
/// @param upper_bound Upper bound value.

template <class T>
void Vector<T>::apply_lower_upper_bounds(const T &lower_bound,
                                         const T &upper_bound) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound) {
      (*this)[i] = lower_bound;
    } else if((*this)[i] > upper_bound) {
      (*this)[i] = upper_bound;
    }
  }
}

// void apply_lower_upper_bounds(const Vector<T>&, const Vector<T>&) method

/// Sets the elements of the vector to given lower bound values if they fall
/// below that values,
/// or to given upper bound values if they fall above that values.
/// @param lower_bound Lower bound values.
/// @param upper_bound Upper bound values.

template <class T>
void Vector<T>::apply_lower_upper_bounds(const Vector<T> &lower_bound,
                                         const Vector<T> &upper_bound) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] < lower_bound[i]) {
      (*this)[i] = lower_bound[i];
    } else if((*this)[i] > upper_bound[i]) {
      (*this)[i] = upper_bound[i];
    }
  }
}


// Vector<size_t> sort_less_indices(void) const method

/// Returns the vector of the indices of the vector sorted by less ranks.

template <class T>
Vector<size_t> Vector<T>::sort_less_indices(void) const
{
    Vector<size_t> indices(this->size());

#ifdef __Cpp11__

    const Vector<size_t> less_rank = this->calculate_less_rank();

    for (size_t i = 0; i < this->size(); i++)
    {
        indices[less_rank[i]] = i;
    }

#else

    indices.initialize_sequential();
    std::sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {return (*this)[i1] < (*this)[i2];});

#endif

    return(indices);
}


// Vector<size_t> sort_greater_indices(void) const method

/// Returns the vector of the indices of the vector sorted by greater ranks.

template <class T>
Vector<size_t> Vector<T>::sort_greater_indices(void) const
{
    Vector<size_t> indices(this->size());

#ifdef __Cpp11__

    const Vector<size_t> greater_rank = this->calculate_greater_rank();

    for (size_t i = 0; i < this->size(); i++)
    {
        indices[greater_rank[i]] = i;
    }

#else

    indices.initialize_sequential();
    std::sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {return (*this)[i1] > (*this)[i2];});

#endif

    return(indices);
}


// Vector<size_t> calculate_less_rank(void) const method

/// Returns a vector with the rank of the elements of this vector.
/// The smallest element will have rank 0, and the greatest element will have
/// size-1.
/// That is, small values correspond with small ranks.

template <class T> Vector<size_t> Vector<T>::calculate_less_rank(void) const
{
  const size_t this_size = this->size();

  Vector<size_t> rank(this_size);

  Vector<T> sorted_vector(*this);

  std::sort(sorted_vector.begin(), sorted_vector.end(), std::less<double>());

  Vector<size_t> previous_rank;
  previous_rank.set(this_size, -1);

  for (size_t i = 0; i < this_size; i++)
  {
    for (size_t j = 0; j < this_size; j++)
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

  return (rank);
}

// Vector<size_t> calculate_greater_rank(void) const method

/// Returns a vector with the rank of the elements of this vector.
/// The smallest element will have rank size-1, and the greatest element will
/// have 0.
/// That is, small values correspond to big ranks.

template <class T>
Vector<size_t> Vector<T>::calculate_greater_rank(void) const
{
  const size_t this_size = this->size();

  Vector<size_t> rank(this_size);

  Vector<T> sorted_vector(*this);

  std::sort(sorted_vector.begin(), sorted_vector.end(), std::greater<T>());

  Vector<size_t> previous_rank;
  previous_rank.set(this_size, -1);

  for (size_t i = 0; i < this_size; i++)
  {
    for (size_t j = 0; j < this_size; j++)
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

  return (rank);
}


// Vector<T> operator + (const T&) const method

/// Sum vector+scalar arithmetic operator.
/// @param scalar Scalar value to be added to this vector.

template <class T>
inline Vector<T> Vector<T>::operator+(const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> sum(this_size);

  std::transform(this->begin(), this->end(), sum.begin(),
                 std::bind2nd(std::plus<T>(), scalar));

  return (sum);
}

// Vector<T> operator + (const Vector<T>&) const method

/// Sum vector+vector arithmetic operator.
/// @param other_vector Vector to be added to this vector.

template <class T>
inline Vector<T> Vector<T>::operator+(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator + (const Vector<T>) const.\n"
           << "Size of vectors is " << this_size << " and " << other_size
           << " and they must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> sum(this_size);

  std::transform(this->begin(), this->end(), other_vector.begin(), sum.begin(),
                 std::plus<T>());

  return (sum);
}

// Vector<T> operator - (const T&) const method

/// Difference vector-scalar arithmetic operator.
/// @param scalar Scalar value to be subtracted to this vector.

template <class T>
inline Vector<T> Vector<T>::operator-(const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> difference(this_size);

  std::transform(this->begin(), this->end(), difference.begin(),
                 std::bind2nd(std::minus<T>(), scalar));

  return (difference);
}

// Vector<T> operator - (const Vector<T>&) const method

/// Difference vector-vector arithmetic operator.
/// @param other_vector vector to be subtracted to this vector.

template <class T>
inline Vector<T> Vector<T>::operator-(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator - (const Vector<T>&) const.\n"
           << "Size of vectors is " << this_size << " and " << other_size
           << " and they must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> difference(this_size);

  std::transform(this->begin(), this->end(), other_vector.begin(),
                 difference.begin(), std::minus<T>());

  return (difference);
}

// Vector<T> operator * (const T&) const method

/// Product vector*scalar arithmetic operator.
/// @param scalar Scalar value to be multiplied to this vector.

template <class T> Vector<T> Vector<T>::operator*(const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> product(this_size);

  std::transform(this->begin(), this->end(), product.begin(),
                 std::bind2nd(std::multiplies<T>(), scalar));

  return (product);
}

// Type operator * (const Vector<T>&) const method

/// Element by element product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

template <class T>
inline Vector<T> Vector<T>::operator*(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator * (const Vector<T>&) const.\n"
           << "Size of other vector (" << other_size
           << ") must be equal to size of this vector (" << this_size << ").\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> product(this_size);

  std::transform(this->begin(), this->end(), other_vector.begin(),
                 product.begin(), std::multiplies<T>());

  return (product);
}

// Matrix<T> operator * (const Matrix<T>&) const method

/// Element by row product vector*matrix arithmetic operator.
/// @param matrix matrix to be multiplied to this vector.

template <class T>
Matrix<T> Vector<T>::operator*(const Matrix<T> &matrix) const {
  const size_t rows_number = matrix.get_rows_number();
  const size_t columns_number = matrix.get_columns_number();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(rows_number != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator * (const Matrix<T>&) const.\n"
           << "Number of matrix rows (" << rows_number
           << ") must be equal to vector size (" << this_size << ").\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Matrix<T> product(rows_number, columns_number);

  for (size_t i = 0; i < rows_number; i++) {
    for (size_t j = 0; j < columns_number; j++) {
      product(i, j) = (*this)[i] * matrix(i, j);
    }
  }

  return (product);
}

// Vector<T> dot(const Matrix<T>&) const method

/// Returns the dot product of this vector with a matrix.
/// The number of rows of the matrix must be equal to the size of the vector.
/// @param matrix matrix to be multiplied to this vector.

template <class T>
Vector<double> Vector<T>::dot(const Matrix<T> &matrix) const {
  const size_t rows_number = matrix.get_rows_number();
  const size_t columns_number = matrix.get_columns_number();
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(rows_number != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> dot(const Matrix<T>&) const method.\n"
           << "Matrix number of rows must be equal to vector size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<double> product(columns_number);

  //   for(size_t j = 0; j < columns_number; j++)
  //   {
  //      product[j] = 0;

  //      for(size_t i = 0; i < rows_number; i++)
  //      {
  //         product[j] += (*this)[i]*matrix(i,j);
  //      }
  //   }

  const Eigen::Map<Eigen::VectorXd> vector_eigen((double *)this->data(), this_size);

  const Eigen::Map<Eigen::MatrixXd> matrix_eigen((double *)matrix.data(), rows_number, columns_number);

  Eigen::Map<Eigen::VectorXd> product_eigen(product.data(), columns_number);

  product_eigen = vector_eigen.transpose() * matrix_eigen;

  return (product);
}

// Vector<T> dot(const Vector<T>&) const method

/// Dot product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

template <class T>
inline double Vector<T>::dot(const Vector<double> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Type dot(const Vector<T>&) const method.\n"
           << "Both vector sizes must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  //   const Eigen::Map<Eigen::VectorXd>
  //   this_vector_eigen((double*)this->data(), this_size);
  //   const Eigen::Map<Eigen::VectorXd>
  //   other_vector_eigen((double*)other_vector.data(), this_size);

  //   return(this_vector_eigen.dot(other_vector_eigen));

  double dot_product = 0.0;

  for (size_t i = 0; i < this_size; i++) {
    dot_product += (*this)[i] * other_vector[i];
  }

  return (dot_product);
}

// Matrix<T> direct(const Vector<T>&) const method

/// Outer product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

template <class T>
Matrix<T> Vector<T>::direct(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Matrix<T> direct(const Vector<T>&) const method.\n"
           << "Both vector sizes must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Matrix<T> direct(this_size, this_size);

  for (size_t i = 0; i < this_size; i++) {
    for (size_t j = 0; j < this_size; j++) {
      direct(i, j) = (*this)[i] * other_vector[j];
    }
  }

  return (direct);
}

// Vector<T> operator / (const T&) const method

/// Cocient vector/scalar arithmetic operator.
/// @param scalar Scalar value to be divided to this vector.

template <class T> Vector<T> Vector<T>::operator/(const T &scalar) const {
  const size_t this_size = this->size();

  Vector<T> cocient(this_size);

  std::transform(this->begin(), this->end(), cocient.begin(),
                 std::bind2nd(std::divides<T>(), scalar));

  return (cocient);
}

// Vector<T> operator / (const Vector<T>&) const method

/// Cocient vector/vector arithmetic operator.
/// @param other_vector vector to be divided to this vector.

template <class T>
Vector<T> Vector<T>::operator/(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> operator / (const Vector<T>&) const.\n"
           << "Both vector sizes must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> cocient(this_size);

  std::transform(this->begin(), this->end(), other_vector.begin(),
                 cocient.begin(), std::divides<T>());

  return (cocient);
}

// void operator += (const T&)

/// Scalar sum and assignment operator.
/// @param value Scalar value to be added to this vector.

template <class T> void Vector<T>::operator+=(const T &value) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] + value;
  }
}

// void operator += (const Vector<T>&)

/// Vector sum and assignment operator.
/// @param other_vector Vector to be added to this vector.

template <class T> void Vector<T>::operator+=(const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator += (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] + other_vector[i];
  }
}

// void operator -= (const T&)

/// Scalar rest and assignment operator.
/// @param value Scalar value to be subtracted to this vector.

template <class T> void Vector<T>::operator-=(const T &value) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] - value;
  }
}

// void operator -= (const Vector<T>&)

/// Vector rest and assignment operator.
/// @param other_vector Vector to be subtracted to this vector.

template <class T> void Vector<T>::operator-=(const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator -= (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] - other_vector[i];
  }
}

// void operator *= (const T&)

/// Scalar product and assignment operator.
/// @param value Scalar value to be multiplied to this vector.

template <class T> void Vector<T>::operator*=(const T &value) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] * value;
  }
}

// void operator *= (const Vector<T>&)

/// Vector product and assignment operator.
/// @param other_vector Vector to be multiplied to this vector.

template <class T> void Vector<T>::operator*=(const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator *= (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] * other_vector[i];
  }
}

// void operator /= (const T&)

/// Scalar division and assignment operator.
/// @param value Scalar value to be divided to this vector.

template <class T> void Vector<T>::operator/=(const T &value) {
  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] / value;
  }
}

// void operator /= (const Vector<T>&)

/// Vector division and assignment operator.
/// @param other_vector Vector to be divided to this vector.

template <class T> void Vector<T>::operator/=(const Vector<T> &other_vector) {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t other_size = other_vector.size();

  if(other_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void operator /= (const Vector<T>&).\n"
           << "Both vector sizes must be the same.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = (*this)[i] / other_vector[i];
  }
}

// void filter_positive(void) method

/// Sets all the negative elements in the vector to zero.

template <class T> void Vector<T>::filter_positive(void) {
  for (size_t i = 0; i < this->size(); i++) {
    if((*this)[i] < 0) {
      (*this)[i] = 0;
    }
  }
}

// void filter_negative(void) method

/// Sets all the positive elements in the vector to zero.

template <class T> void Vector<T>::filter_negative(void) {
  for (size_t i = 0; i < this->size(); i++) {
    if((*this)[i] > 0) {
      (*this)[i] = 0;
    }
  }
}

// void scale_minimum_maximum(const T&, const T&) method

/// Normalizes the elements of this vector using the minimum and maximum method.
/// @param minimum Minimum value for the scaling.
/// @param maximum Maximum value for the scaling.

template <class T>
void Vector<T>::scale_minimum_maximum(const T &minimum, const T &maximum) {
  if(maximum - minimum < 1.0e-99) {
    return;
  }

  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = 2.0 * ((*this)[i] - minimum) / (maximum - minimum) - 1.0;
  }
}

// void scale_minimum_maximum(const Statistics<T>&) method

/// Normalizes the elements of this vector using the minimum and maximum method.
/// @param statistics Statistics structure, which contains the minimum and
/// maximum values for the scaling.

template <class T>
void Vector<T>::scale_minimum_maximum(const Statistics<T> &statistics) {
  scale_minimum_maximum(statistics.minimum, statistics.maximum);
}

// Statistics<T> scale_minimum_maximum(void) method

/// Normalizes the elements of the vector with the minimum and maximum method.
/// The minimum and maximum values used are those calculated from the vector.
/// It also returns the statistics from the vector.

template <class T> Statistics<T> Vector<T>::scale_minimum_maximum(void) {
  const Statistics<T> statistics = calculate_statistics();

  scale_minimum_maximum(statistics);

  return (statistics);
}

// void scale_mean_standard_deviation(const T&, const T&) method

/// Normalizes the elements of this vector using the mean and standard deviation
/// method.
/// @param mean Mean value for the scaling.
/// @param standard_deviation Standard deviation value for the scaling.

template <class T>
void Vector<T>::scale_mean_standard_deviation(const T &mean,
                                              const T &standard_deviation) {
  if(standard_deviation < 1.0e-99) {
    return;
  }

  const size_t this_size = this->size();

  for (size_t i = 0; i < this_size; i++) {
    (*this)[i] = ((*this)[i] - mean) / standard_deviation;
  }
}

// void scale_mean_standard_deviation(const Statistics<T>&) method

/// Normalizes the elements of this vector using the mean and standard deviation
/// method.
/// @param statistics Statistics structure,
/// which contains the mean and standard deviation values for the scaling.

template <class T>
void Vector<T>::scale_mean_standard_deviation(const Statistics<T> &statistics) {
  scale_mean_standard_deviation(statistics.mean, statistics.standard_deviation);
}

// Statistics<T> scale_mean_standard_deviation(void) method

/// Normalizes the elements of the vector with the mean and standard deviation
/// method.
/// The values used are those calculated from the vector.
/// It also returns the statistics from the vector.

template <class T>
Statistics<T> Vector<T>::scale_mean_standard_deviation(void) {
  const Statistics<T> statistics = calculate_statistics();

  scale_mean_standard_deviation(statistics);

  return (statistics);
}

// void scale_minimum_maximum(const Vector<T>&, const Vector<T>&) method

/// Scales the vectir elements with given minimum and maximum values.
/// It updates the data in the vector.
/// The size of the minimum and maximum vectors must be equal to the size of the
/// vector.
/// @param minimum Minimum values.
/// @param maximum Maximum values.

template <class T>
void Vector<T>::scale_minimum_maximum(const Vector<T> &minimum,
                                      const Vector<T> &maximum) {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t minimum_size = minimum.size();

  if(minimum_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void scale_minimum_maximum(const Vector<T>&, const Vector<T>&) "
              "method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void scale_minimum_maximum(const Vector<T>&, const Vector<T>&) "
              "method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  // Rescale data

  for (size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector class.\n"
                << "void scale_minimum_maximum(const Vector<T>&, const "
                   "Vector<T>&) method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be scaled.\n";

      // Do nothing
    } else {
      (*this)[i] =
          2.0 * ((*this)[i] - minimum[i]) / (maximum[i] - minimum[i]) - 1.0;
    }
  }
}

// void scale_mean_standard_deviation(const Vector<T>&, const Vector<T>&) method

/// Scales the vector elements with given mean and standard deviation values.
/// It updates the data in the vector.
/// The size of the mean and standard deviation vectors must be equal to the
/// size of the vector.
/// @param mean Mean values.
/// @param standard_deviation Standard deviation values.

template <class T>
void
Vector<T>::scale_mean_standard_deviation(const Vector<T> &mean,
                                         const Vector<T> &standard_deviation) {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t mean_size = mean.size();

  if(mean_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void scale_mean_standard_deviation(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void scale_mean_standard_deviation(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  // Rescale data

  for (size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector class.\n"
                << "void scale_mean_standard_deviation(const Vector<T>&, const "
                   "Vector<T>&) method.\n"
                << "Standard deviation of variable " << i << " is zero.\n"
                << "Those elements won't be scaled.\n";

      // Do nothing
    } else {
      (*this)[i] = ((*this)[i] - mean[i]) / standard_deviation[i];
    }
  }
}

// Vector<T> calculate_scaled_minimum_maximum(const Vector<T>&, const
// Vector<T>&) const method

/// Returns a vector with the scaled elements of this vector acording to the
/// minimum and maximum method.
/// The size of the minimum and maximum vectors must be equal to the size of the
/// vector.
/// @param minimum Minimum values.
/// @param maximum Maximum values.

template <class T>
Vector<T>
Vector<T>::calculate_scaled_minimum_maximum(const Vector<T> &minimum,
                                            const Vector<T> &maximum) const {
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  const size_t minimum_size = minimum.size();

  if(minimum_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_scaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_scaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> scaled_minimum_maximum(this_size);

  // Rescale data

  for (size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector class.\n"
                << "Vector<T> calculate_scaled_minimum_maximum(const "
                   "Vector<T>&, const Vector<T>&) const method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be scaled.\n";

      scaled_minimum_maximum[i] = (*this)[i];
    } else {
      scaled_minimum_maximum[i] =
          2.0 * ((*this)[i] - minimum[i]) / (maximum[i] - minimum[i]) - 1.0;
    }
  }

  return (scaled_minimum_maximum);
}

// Vector<T> calculate_scaled_mean_standard_deviation(const Vector<T>&, const
// Vector<T>&) const method

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

  std::ostringstream buffer;

  const size_t mean_size = mean.size();

  if(mean_size != this_size) {
    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_scaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    buffer << "OpenNN Exception: Vector template.\n"
           << "Vector<T> calculate_scaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> scaled_mean_standard_deviation(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector template.\n"
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

  return (scaled_mean_standard_deviation);
}

// Vector<T> calculate_unscaled_minimum_maximum(const Vector<T>&, const
// Vector<T>&) const method

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
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_unscaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_unscaled_minimum_maximum(const Vector<T>&, "
              "const Vector<T>&) const method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> unscaled_minimum_maximum(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector template.\n"
                << "Vector<T> calculate_unscaled_minimum_maximum(const "
                   "Vector<T>&, const Vector<T>&) const method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be unscaled.\n";

      unscaled_minimum_maximum[i] = (*this)[i];
    } else {
      unscaled_minimum_maximum[i] =
          0.5 * ((*this)[i] + 1.0) * (maximum[i] - minimum[i]) + minimum[i];
    }
  }

  return (unscaled_minimum_maximum);
}

// Vector<T> calculate_unscaled_mean_standard_deviation(const Vector<T>&, const
// Vector<T>&) const method

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
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "Vector<T> calculate_unscaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template.\n"
           << "Vector<T> calculate_unscaled_mean_standard_deviation(const "
              "Vector<T>&, const Vector<T>&) const method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> unscaled_mean_standard_deviation(this_size);

  for (size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector template.\n"
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

  return (unscaled_mean_standard_deviation);
}

// void unscale_minimum_maximum(const Vector<T>&, const Vector<T>&) method

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
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void unscale_minimum_maximum(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void unscale_minimum_maximum(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector template.\n"
                << "void unscale_minimum_maximum(const Vector<T>&, const "
                   "Vector<T>&) method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be unscaled.\n";

      // Do nothing
    } else {
      (*this)[i] =
          0.5 * ((*this)[i] + 1.0) * (maximum[i] - minimum[i]) + minimum[i];
    }
  }
}

// void unscale_mean_standard_deviation(const Vector<T>&, const Vector<T>&)
// method

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
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template."
           << "void unscale_mean_standard_deviation(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template.\n"
           << "void unscale_mean_standard_deviation(const Vector<T>&, const "
              "Vector<T>&) method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < 1e-99) {
      std::cout << "OpenNN Warning: Vector template.\n"
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

// Matrix<T> arrange_diagonal_matrix(void) const method

/// Returns a squared matrix in which the entries outside the main diagonal are
/// all zero.
/// The elements in the diagonal are the elements in this vector.
/// @todo

template <class T> Matrix<T> Vector<T>::arrange_diagonal_matrix(void) const {
  const size_t this_size = this->size();

  Matrix<T> matrix = new Matrix<T>(this_size, this_size, 0.0);

  for (size_t i = 0; i < this_size; i++) {
    matrix(i, i) = (*this)[i];
  }

  return (matrix);
}

// Vector<T> arrange_subvector(const Vector<size_t>&) const

/// Returns another vector whose elements are given by some elements of this
/// vector.
/// @param indices Indices of this vector whose elements are required.

template <class T>
Vector<T> Vector<T>::arrange_subvector(const Vector<size_t> &indices) const {
  const size_t new_size = indices.size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  for (size_t i = 0; i < new_size; i++) {
    if(indices[i] > this_size) {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> arrange_subvector(const Vector<T>&) const method.\n"
             << "Index is equal or greater than this size.\n";

      throw std::logic_error(buffer.str());
    }
  }

#endif

  Vector<T> subvector(new_size);

  for (size_t i = 0; i < new_size; i++) {
    subvector[i] = (*this)[indices[i]];
  }

  return (subvector);
}

// Vector<T> arrange_subvector_first(const size_t&) const method

/// Returns a vector with the first n elements of this vector.
/// @param elements_number Size of the new vector.

template <class T>
Vector<T>
Vector<T>::arrange_subvector_first(const size_t &elements_number) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(elements_number > this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> arrange_subvector_first(const size_t&) const method.\n"
           << "Number of elements must be equal or greater than this size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> subvector(elements_number);

  for (size_t i = 0; i < elements_number; i++) {
    subvector[i] = (*this)[i];
  }

  return (subvector);
}

// Vector<T> arrange_subvector_last(const size_t&) const method

/// Returns a vector with the last n elements of this vector.
/// @param elements_number Size of the new vector.

template <class T>
Vector<T>
Vector<T>::arrange_subvector_last(const size_t &elements_number) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(elements_number > this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> arrange_subvector_last(const size_t&) const method.\n"
           << "Number of elements must be equal or greater than this size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> subvector(elements_number);

  for (size_t i = 0; i < elements_number; i++) {
    subvector[i] = (*this)[i + this_size - elements_number];
  }

  return (subvector);
}

// void load(const std::string&) method

/// Loads the members of a vector from an data file.
/// Please be careful with the file format, which is specified in the OpenNN
/// manual.
/// @param file_name Name of vector file.

template <class T> void Vector<T>::load(const std::string &file_name) {
  std::ifstream file(file_name.c_str());

  std::stringstream buffer;

  std::string line;

  while (file.good()) {
    getline(file, line);

    buffer << line;
  }

  std::istream_iterator<std::string> it(buffer);
  std::istream_iterator<std::string> end;

  const std::vector<std::string> results(it, end);

  const size_t new_size = (size_t)results.size();

  this->resize(new_size);

  file.clear();
  file.seekg(0, std::ios::beg);

  // Read data

  for (size_t i = 0; i < new_size; i++) {
    file >> (*this)[i];
  }

  file.close();
}

// void save(const std::string&) const method

/// Saves to a data file the elements of the vector.
/// The file format is as follows:
/// element_0 element_1 ... element_N-1
/// @param file_name Name of vector data file.

template <class T> void Vector<T>::save(const std::string &file_name) const
{
  std::ofstream file(file_name.c_str());

  if(!file.is_open()) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector template.\n"
           << "void save(const std::string&) const method.\n"
           << "Cannot open vector data file.\n";

    throw std::logic_error(buffer.str());
  }

  // Write file

  const size_t this_size = this->size();

  if(this_size > 0) {
    file << (*this)[0];

    const char space = ' ';

    for (size_t i = 1; i < this_size; i++) {
      file << space << (*this)[i];
    }

    file << std::endl;
  }

  // Close file

  file.close();
}

// void tuck_in(const size_t&, const Vector<T>&) const method

/// Insert another vector starting from a given position.
/// @param position Insertion position.
/// @param other_vector Vector to be inserted.

template <class T>
void Vector<T>::tuck_in(const size_t &position, const Vector<T> &other_vector) {
  const size_t other_size = other_vector.size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(position + other_size > this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void tuck_in(const size_t&, const Vector<T>&) const method.\n"
           << "Cannot tuck in vector.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  for (size_t i = 0; i < other_size; i++) {
    (*this)[position + i] = other_vector[i];
  }
}

// Vector<T> take_out(const size_t&, const size_t&) method

/// Extract a vector of a given size from a given position
/// @param position Extraction position.
/// @param other_size Size of vector to be extracted.

template <class T>
Vector<T> Vector<T>::take_out(const size_t &position,
                              const size_t &other_size) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(position + other_size > this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> take_out(const size_t&, const size_t&) method.\n"
           << "Cannot take out vector.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  const Vector<T> other_vector((*this).begin() + position,
                               (*this).begin() + position + other_size);

  //   for(size_t i = 0; i < other_size; i++)
  //   {
  //      other_vector[i] = (*this)[position + i];
  //   }

  return (other_vector);
}

// void insert_element(const size_t& index, const T& value) method

/// Returns a new vector with a new element inserted.
/// @param index Position of the new element.
/// @param value Value of the new element.

template <class T>
Vector<T> Vector<T>::insert_element(const size_t &index, const T &value) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(index > this_size) {
    std::ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "void insert_element(const size_t& index, const T& value) method.\n"
        << "Index is greater than vector size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> other_vector(this_size + 1);

  for (size_t i = 0; i <= this_size; i++) {
    if(i < index) {
      other_vector[i] = (*this)[i];
    }
    if(i == index) {
      other_vector[i] = value;
    } else if(i > index) {
      other_vector[i] = (*this)[i - 1];
    }
  }

  return (other_vector);
}

// Vector<T> remove_element(const size_t) const

/// Returns a new vector which is a copy of this vector but with a given element
/// removed.
/// Therefore, the size of the new vector is the size of this vector minus one.
/// @param index Index of element to be removed.

template <class T>
Vector<T> Vector<T>::remove_element(const size_t &index) const {
  const size_t this_size = this->size();

// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  if(index >= this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> remove_element(const size_t&) const method.\n"
           << "Index is equal or greater than vector size.\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Vector<T> other_vector(this_size - 1);

  for (size_t i = 0; i < this_size; i++) {
    if(i < index) {
      other_vector[i] = (*this)[i];
    } else if(i > index) {
      other_vector[i - 1] = (*this)[i];
    }
  }

  return (other_vector);
}

// Vector<T> remove_value(const T&) const

/// Construct a copy of this vector but without a certain value.
/// Note that the new vector might have a different size than this vector.
/// @param value Value of elements to be removed.

template <class T> Vector<T> Vector<T>::remove_value(const T &value) const {
  const size_t this_size = this->size();

  size_t value_count = 0;

  for (size_t i = 0; i < this_size; i++) {
    if((*this)[i] == value) {
      value_count++;
    }
  }

  if(value_count == 0) {
    return (*this);
  } else {
    const size_t other_size = this_size - value_count;

    Vector<T> other_vector(other_size);

    size_t other_index = 0;

    for (size_t i = 0; i < this_size; i++) {
      if((*this)[i] != value) {
        other_vector[other_index] = (*this)[i];

        other_index++;
      }
    }

    return (other_vector);
  }
}

// Vector<T> assemble(const Vector<T>&) const method

/// Assemble two vectors.
/// @param other_vector Vector to be get_assemblyd to this vector.

template <class T>
Vector<T> Vector<T>::assemble(const Vector<T> &other_vector) const {
  const size_t this_size = this->size();
  const size_t other_size = other_vector.size();

  if(this_size == 0 && other_size == 0) {
    Vector<T> assembly;

    return (assembly);
  } else if(this_size == 0) {
    return (other_vector);
  } else if(other_size == 0) {
    return (*this);
  } else {
    Vector<T> assembly(this_size + other_size);

    for (size_t i = 0; i < this_size; i++) {
      assembly[i] = (*this)[i];
    }

    for (size_t i = 0; i < other_size; i++) {
      assembly[this_size + i] = other_vector[i];
    }

    return (assembly);
  }
}

// std::vector<T> to_std_vector(void) const method

/// Returns a std vector with the size and elements of this OpenNN vector.

template <class T> std::vector<T> Vector<T>::to_std_vector(void) const {
  const size_t this_size = this->size();

  std::vector<T> std_vector(this_size);

  for (size_t i = 0; i < this_size; i++) {
    std_vector[i] = (*this)[i];
  }

  return (std_vector);
}

// Matrix<T> to_row_matrix(void) const method

/// Returns a row matrix with number of rows equal to one
/// and number of columns equal to the size of this vector.

template <class T> Matrix<T> Vector<T>::to_row_matrix(void) const {
  const size_t this_size = this->size();

  Matrix<T> matrix(1, this_size);

  for (size_t i = 0; i < this_size; i++) {
    matrix(0, i) = (*this)[i];
  }

  return (matrix);
}

// Matrix<T> to_column_matrix(void) const method

/// Returns a column matrix with number of rows equal to the size of this vector
/// and number of columns equal to one.

template <class T> Matrix<T> Vector<T>::to_column_matrix(void) const {
  const size_t this_size = this->size();

  Matrix<T> matrix(this_size, 1);

  for (size_t i = 0; i < this_size; i++) {
    matrix(i, 0) = (*this)[i];
  }

  return (matrix);
}

// void parse(const std::string&) method

/// This method takes a string representation of a vector and sets this vector
/// to have size equal to the number of words and values equal to that words.
/// @param str String to be parsed.

template <class T> void Vector<T>::parse(const std::string &str) {
  if(str.empty()) {
    set();
  } else {
    std::istringstream buffer(str);

    std::istream_iterator<std::string> first(buffer);
    std::istream_iterator<std::string> last;

    Vector<std::string> str_vector(first, last);

    const size_t new_size = str_vector.size();

    if(new_size > 0) {
      this->resize(new_size);

      buffer.clear();
      buffer.seekg(0, std::ios::beg);

      for (size_t i = 0; i < new_size; i++) {
        buffer >> (*this)[i];
      }
    }
  }
}

// std::string to_string(const std::string&)

/// Returns a string representation of this vector.

template <class T>
std::string Vector<T>::to_string(const std::string &separator) const {
  std::ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0) {
    buffer << (*this)[0];

    for (size_t i = 1; i < this_size; i++) {
      buffer << separator << (*this)[i];
    }
  }

  return (buffer.str());
}

// std::string to_text()

/// Returns a string representation of this vector which can be inserted in a
/// text.

template <class T> std::string Vector<T>::to_text() const {
  std::ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0) {
    buffer << (*this)[0];

    for (size_t i = 1; i < this_size - 1; i++) {
      buffer << ", " << (*this)[i];
    }

    if(this_size > 1) {
      buffer << " and " << (*this)[this_size - 1];
    }
  }

  return (buffer.str());
}

// Vector<std::string> write_string_vector(const size_t& precision) const

/// This method retuns a vector of strings with size equal to the size of this
/// vector and elements equal to string representations of the elements of this
/// vector.

template <class T>
Vector<std::string>
Vector<T>::write_string_vector(const size_t &precision) const {
  const size_t this_size = this->size();

  Vector<std::string> string_vector(this_size);

  std::ostringstream buffer;

  for (size_t i = 0; i < this_size; i++) {
    buffer.str("");
    buffer << std::setprecision(precision) << (*this)[i];

    string_vector[i] = buffer.str();
  }

  return (string_vector);
}

// Matrix<T> to_matrix(const size_t&, const size_t&) method

/// Returns a matrix with given numbers of rows and columns and with the
/// elements of this vector ordered by rows.
/// The number of rows multiplied by the number of columns must be equal to the
/// size of this vector.
/// @param rows_number Number of rows in the new matrix.
/// @param columns_number Number of columns in the new matrix.

template <class T>
Matrix<T> Vector<T>::to_matrix(const size_t &rows_number,
                               const size_t &columns_number) const {
// Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(rows_number * columns_number != this_size) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Matrix<T> to_matrix(const size_t&, const size_t&) method.\n"
           << "The number of rows (" << rows_number
           << ") times the number of colums (" << columns_number
           << ") must be equal to the size of the vector (" << this_size
           << ").\n";

    throw std::logic_error(buffer.str());
  }

#endif

  Matrix<T> matrix(rows_number, columns_number);

  size_t index = 0;

  for (size_t i = 0; i < rows_number; i++) {
    for (size_t j = 0; j < columns_number; j++) {
      matrix(i, j) = (*this)[index];
      index++;
    }
  }

  return (matrix);
}

// Vector input operator

/// This method re-writes the inputs operator >> for the Vector template.
/// @param is Input stream.
/// @param v Input vector.

template <class T> std::istream &operator>>(std::istream &is, Vector<T> &v) {
  const size_t size = v.size();

  for (size_t i = 0; i < size; i++) {
    is >> v[i];
  }

  return (is);
}

// Vector output operator

/// This method re-writes the output operator << for the Vector template.
/// @param os Output stream.
/// @param v Output vector.

template <class T>
std::ostream &operator<<(std::ostream &os, const Vector<T> &v) {
  const size_t this_size = v.size();

  if(this_size > 0) {
    os << v[0];

    const char space = ' ';

    for (size_t i = 1; i < this_size; i++) {
      os << space << v[i];
    }
  }

  return (os);
}

// Vector of vectors output operator

/// This method re-writes the output operator << for vectors of vectors.
/// @param os Output stream.
/// @param v Output vector of vectors.

template <class T>
std::ostream &operator<<(std::ostream &os, const Vector< Vector<T> > &v) {
  for (size_t i = 0; i < v.size(); i++) {
    os << "subvector_" << i << "\n" << v[i] << std::endl;
  }

  return (os);
}

// Vector of matrices output operator

/// This method re-writes the output operator << for vectors of matrices.
/// @param os Output stream.
/// @param v Output vector of matrices.

template <class T>
std::ostream &operator<<(std::ostream &os, const Vector< Matrix<T> > &v) {
  for (size_t i = 0; i < v.size(); i++) {
    os << "submatrix_" << i << "\n" << v[i];
  }

  return (os);
}

// double calculate_random_uniform(const double&, const double&) method

/// Returns a random number chosen from a uniform distribution.
/// @param minimum Minimum value.
/// @param maximum Maximum value.

template <class T>
T calculate_random_uniform(const T &minimum, const T &maximum) {
  const T random = (T)rand() / (RAND_MAX + 1.0);

  const T random_uniform = minimum + (maximum - minimum) * random;

  return (random_uniform);
}

template <class T>
std::string number_to_string(const T &value) {
  std::ostringstream ss;

  ss << value;

  return (ss.str());
}

// double calculate_random_normal(const double&, const double&) method

/// Returns a random number chosen from a normal distribution.
/// @param mean Mean value of normal distribution.
/// @param standard_deviation Standard deviation value of normal distribution.

template <class T>
T calculate_random_normal(const T &mean, const T &standard_deviation) {
  const double pi = 4.0 * atan(1.0);

  T random_uniform_1;

  do {
    random_uniform_1 = (T)rand() / (RAND_MAX + 1.0);

  } while (random_uniform_1 == 0.0);

  const T random_uniform_2 = (T)rand() / (RAND_MAX + 1.0);

  // Box-Muller transformation

  const T random_normal = mean +
                          sqrt(-2.0 * log(random_uniform_1)) *
                              sin(2.0 * pi * random_uniform_2) *
                              standard_deviation;

  return (random_normal);
}

/// This structure contains the simplest statistics for a set, variable, etc.
/// It includes the minimum, maximum, mean and standard deviation variables.

template <class T> struct Statistics {
  // Default constructor.

  Statistics(void);

  // Values constructor.

  Statistics(const T &, const T &, const T &, const T &);

  /// Destructor.

  virtual ~Statistics(void);

  // METHODS

  void set_minimum(const double &);

  void set_maximum(const double &);

  void set_mean(const double &);

  void set_standard_deviation(const double &);

  Vector<T> to_vector(void) const;

  void initialize_random(void);

  bool has_minimum_minus_one_maximum_one(void);
  bool has_mean_zero_standard_deviation_one(void);

  void save(const std::string &file_name) const;

  /// Smallest value of a set, function, etc.

  T minimum;

  /// Biggest value of a set, function, etc.

  T maximum;

  /// Mean value of a set, function, etc.

  T mean;

  /// Standard deviation value of a set, function, etc.

  T standard_deviation;
};

template <class T> Statistics<T>::Statistics(void) {
  minimum = (T)-1.0;
  maximum = (T)1.0;
  mean = (T)0.0;
  standard_deviation = (T)1.0;
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

template <class T> Statistics<T>::~Statistics(void) {}

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
void
Statistics<T>::set_standard_deviation(const double &new_standard_deviation) {
  standard_deviation = new_standard_deviation;
}

/// Returns all the statistical parameters contained in a single vector.
/// The size of that vector is seven.
/// The elements correspond to the minimum, maximum, mean and standard deviation
/// values respectively.

template <class T> Vector<T> Statistics<T>::to_vector(void) const {
  Vector<T> statistics_vector(4);
  statistics_vector[0] = minimum;
  statistics_vector[1] = maximum;
  statistics_vector[2] = mean;
  statistics_vector[3] = standard_deviation;

  return (statistics_vector);
}

/// Initializes the statistics structure with a random
/// minimum (between -1 and 1), maximum (between 0 and 1),
/// mean (between -1 and 1), standard deviation (between 0 and 1).

template <class T> void Statistics<T>::initialize_random(void) {
  minimum = calculate_random_uniform(-1.0, 0.0);
  maximum = calculate_random_uniform(0.0, 1.0);
  mean = calculate_random_uniform(-1.0, 1.0);
  standard_deviation = calculate_random_uniform(0.0, 1.0);
}

/// Returns true if the minimum value is -1 and the maximum value is +1,
/// and false otherwise.

template <class T> bool Statistics<T>::has_minimum_minus_one_maximum_one(void) {
  if(-1.000001 < minimum && minimum < -0.999999 && 0.999999 < maximum &&
      maximum < 1.000001) {
    return (true);
  } else {
    return (false);
  }
}

/// Returns true if the mean value is 0 and the standard deviation value is 1,
/// and false otherwise.

template <class T>
bool Statistics<T>::has_mean_zero_standard_deviation_one(void) {
  if(-0.000001 < mean && mean < 0.000001 && 0.999999 < standard_deviation &&
      standard_deviation < 1.000001) {
    return (true);
  } else {
    return (false);
  }
}

/// Saves to a file the minimum, maximum, standard deviation, asymmetry and
/// kurtosis values
/// of the statistics structure.
/// @param file_name Name of statistics data file.

template <class T>
void Statistics<T>::save(const std::string &file_name) const {
  std::ofstream file(file_name.c_str());

  if(!file.is_open()) {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Statistics template.\n"
           << "void save(const std::string&) const method.\n"
           << "Cannot open statistics data file.\n";

    throw std::logic_error(buffer.str());
  }

  // Write file

  file << "Minimum: " << minimum << std::endl << "Maximum: " << maximum
       << std::endl << "Mean: " << mean << std::endl
       << "Standard deviation: " << standard_deviation << std::endl;

  // Close file

  file.close();
}

// Statistics output operator

/// This method re-writes the output operator << for the Statistics template.
/// @param os Output stream.
/// @param v Output vector.

template <class T>
std::ostream &operator<<(std::ostream &os, const Statistics<T> &statistics) {
  os << "Statistics structure\n"
     << "   Minimum: " << statistics.minimum << std::endl
     << "   Maximum: " << statistics.maximum << std::endl
     << "   Mean: " << statistics.mean << std::endl
     << "   Standard deviation: " << statistics.standard_deviation << std::endl;

  return (os);
}

///
/// This template contains the data needed to represent a histogram.
///

template <class T> struct Histogram {
  /// Default constructor.

  explicit Histogram(void);

  /// Destructor.

  virtual ~Histogram(void);

  /// Bins number constructor.

  Histogram(const size_t &);

  /// Values constructor.

  Histogram(const Vector<T> &, const Vector<size_t> &);

  size_t get_bins_number(void) const;

  size_t count_empty_bins(void) const;

  size_t calculate_minimum_frequency(void) const;

  size_t calculate_maximum_frequency(void) const;

  size_t calculate_most_populated_bin(void) const;

  Vector<T> calculate_minimal_centers(void) const;

  Vector<T> calculate_maximal_centers(void) const;

  size_t calculate_bin(const T &) const;

  size_t calculate_frequency(const T &) const;

  // Vector<size_t> calculate_total_frequencies(const Vector< Histogram<T> >&)
  // const;

  /// Positions of the bins in the histogram.

  Vector<T> centers;

  /// Minimum of the bins in the histogram.

  Vector<T> minimums;

  /// Maximum of the bins in the histogram.

  Vector<T> maximums;

  /// Population of the bins in the histogram.

  Vector<size_t> frequencies;
};

template <class T> Histogram<T>::Histogram(void) {}

/// Destructor.

template <class T> Histogram<T>::~Histogram(void) {}

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

template <class T> size_t Histogram<T>::get_bins_number(void) const {
  return (centers.size());
}

/// Returns the number of bins with zero variates.

template <class T> size_t Histogram<T>::count_empty_bins(void) const {
  return (frequencies.count_occurrences(0));
}

/// Returns the number of variates in the less populated bin.

template <class T>
size_t Histogram<T>::calculate_minimum_frequency(void) const {
  return (frequencies.calculate_minimum());
}

/// Returns the number of variates in the most populated bin.

template <class T>
size_t Histogram<T>::calculate_maximum_frequency(void) const {
  return (frequencies.calculate_maximum());
}


/// Retuns the index of the most populated bin.

template <class T>
size_t Histogram<T>::calculate_most_populated_bin(void) const {
  return (frequencies.calculate_maximal_index());
}


/// Returns a vector with the centers of the less populated bins.

template <class T>
Vector<T> Histogram<T>::calculate_minimal_centers(void) const {
  const size_t minimum_frequency = calculate_minimum_frequency();

  const Vector<size_t> minimal_indices =
      frequencies.calculate_occurrence_indices(minimum_frequency);

  return (centers.arrange_subvector(minimal_indices));
}

/// Returns a vector with the centers of the most populated bins.

template <class T>
Vector<T> Histogram<T>::calculate_maximal_centers(void) const {
  const size_t maximum_frequency = calculate_maximum_frequency();

  const Vector<size_t> maximal_indices =
      frequencies.calculate_occurrence_indices(maximum_frequency);

  return (centers.arrange_subvector(maximal_indices));
}

// Vector<size_t> Histogram<T>::calculate_bin(const T&) const

/// Returns the number of the bin to which a given value belongs to.
/// @param value Value for which we want to get the bin.

template <class T> size_t Histogram<T>::calculate_bin(const T &value) const {
  const size_t bins_number = get_bins_number();

  const double minimum_center = centers[0];
  const double maximum_center = centers[bins_number - 1];

  const double length =
      (double)(maximum_center - minimum_center) / (double)(bins_number - 1);

  double minimum_value = centers[0] - length / 2;
  double maximum_value = minimum_value + length;

  if(value < maximum_value) {
    return (0);
  }

  for (size_t j = 1; j < bins_number - 1; j++) {
    minimum_value = minimum_value + length;
    maximum_value = maximum_value + length;

    if(value >= minimum_value && value < maximum_value) {
      return (j);
    }
  }

  if(value >= maximum_value) {
    return (bins_number - 1);
  } else {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<size_t> Histogram<T>::calculate_bin(const T&) const.\n"
           << "Unknown return value.\n";

    throw std::logic_error(buffer.str());
  }
}

// size_t Histogram<T>::calculate_frequency(const T&) const

/// Returns the frequency of the bin to which a given value bolongs to.
/// @param value Value for which we want to get the frequency.

template <class T>
size_t Histogram<T>::calculate_frequency(const T &value) const {
  const size_t bin_number = calculate_bin(value);

  const size_t frequency = frequencies[bin_number];

  return (frequency);
}

// Histogram output operator

/// This method re-writes the output operator << for the Histogram template.
/// @param os Output stream.
/// @param v Output vector.

template <class T>
std::ostream &operator<<(std::ostream &os, const Histogram<T> &histogram) {
  os << "Histogram structure\n"
     << "Centers: " << histogram.centers << std::endl
     << "Frequencies: " << histogram.frequencies << std::endl;

  return (os);
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

  /// Correlation coefficient (R-value) of the linear regression.

  double correlation;
};

template <class T>
std::ostream &
operator<<(std::ostream &os,
           const LinearRegressionParameters<T> &linear_regression_parameters) {
  os << "Linear regression parameters:\n"
     << "Intercept: " << linear_regression_parameters.intercept << "\n"
     << "Slope: " << linear_regression_parameters.slope << "\n"
     << "Correlation: " << linear_regression_parameters.correlation
     << std::endl;

  return (os);
}

///
/// This template defines the parameters of a logistic regression analysis
/// between two sets x-y.
///

template <class T> struct LogisticRegressionParameters {
  /// Position of the midpoint of the logistic regression.

  double position;

  /// Slope at the midpoint of the logistic regression.

  double slope;

  /// Correlation coefficient (R-value) of the logistic regression.

  double correlation;
};

template <class T>
std::ostream &operator<<(
    std::ostream &os,
    const LogisticRegressionParameters<T> &logistic_regression_parameters) {
  os << "Logistic regression parameters:\n"
     << "Position: " << logistic_regression_parameters.position << "\n"
     << "Slope: " << logistic_regression_parameters.slope << "\n"
     << "Correlation: " << logistic_regression_parameters.correlation
     << std::endl;

  return (os);
}

} // end namespace OpenNN

#endif

// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
