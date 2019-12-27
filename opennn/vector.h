//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V E C T O R   C O N T A I N E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef VECTOR_H
#define VECTOR_H

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

using namespace std;

namespace OpenNN {

// Forward declarations

template <class T> class Matrix;
template <class T> class Tensor;

template <class T> T calculate_random_uniform(const T&, const T&);
template <class T> T calculate_random_normal(const T&, const T&);


/// This template represents an array of any kind of numbers or objects.

///
/// It inherits from the vector of the standard library, and implements additional utilities.

template <typename T>
class Vector : public vector<T> {

public:

  // Constructors

  explicit Vector();

  explicit Vector(const size_t &);

  explicit Vector(const size_t &, const T&);

  explicit Vector(const string &);

  explicit Vector(const T&, const double &, const T&);

  template <class InputIterator> explicit Vector(const InputIterator&, const InputIterator&);

  Vector(const vector<T>&);

  Vector(const Vector<T>&);

  Vector(const initializer_list<T>&);

  Vector(const Vector<Vector<T>>&);

  // Destructor

  virtual ~Vector();

  // Operators

  inline Vector<T>& operator = (const Vector<T>&);

  bool operator == (const T&) const;

  bool operator != (const T&) const;

  bool operator > (const T&) const;

  bool operator < (const T&) const;

  bool operator >= (const T&) const;

  bool operator <= (const T&) const;

  // Set methods

  void set();

  void set(const size_t &);

  void set(const size_t &, const T&);

  void set(const string &);

  void set(const T&, const double &, const T&);

  void set(const Vector<T> &);

   // Get methods

  const T& get_first() const;
  const T& get_last() const;
  const T& get_before_last() const;

  Vector<T> get_positive_elements() const;
  Vector<T> get_negative_elements() const;

  // Initialization methods

  void initialize(const T&);
  void initialize_first(const size_t&, const T&);

  void initialize_sequential();

  void randomize_uniform(const T&, const T&);
  void randomize_uniform(const Vector<T>&, const Vector<T>&);

  void randomize_normal(const double & = 0.0, const double & = 1.0);
  void randomize_normal(const Vector<double>&, const Vector<double>&);

  void randomize_binary(const double & = 0.5, const double & = 0.5);

  // String methods

  void trim();

  Vector<T> trimmed() const;

  // Fill method

  Vector<T> fill_from(const size_t&, const Vector<T>&) const;

  // Checking methods

  bool contains(const T&) const;
  bool contains(const Vector<T>&) const;

  bool contains_greater_than(const T&) const;

  bool has_same_elements(const Vector<T>&) const;

  bool is_in(const T&, const T&) const;

  bool is_constant(const double & = 0.0) const;
  bool is_constant_string() const;

  bool is_crescent() const;
  bool is_decrescent() const;

  bool is_binary() const;
  bool is_binary_missing_values() const;
  bool is_binary_0_1() const;

  bool is_integer() const;
  bool is_integer_missing_values() const;

  bool is_positive() const;
  bool is_negative() const;

  bool check_period(const double& period) const;

  bool has_nan() const;

  Vector<T> get_reverse() const;

  // Replace methods

  void replace_value(const T&, const T&);

  // Count methods

  size_t count_equal_to(const T&) const;
  size_t count_equal_to(const Vector<T>&) const;

  size_t count_NAN() const;
  size_t count_not_NAN() const;

  size_t count_not_equal_to(const T&) const;
  size_t count_not_equal_to(const Vector<T>&) const;

  size_t count_positive() const;
  size_t count_negative() const;

  size_t count_integers(const size_t&) const;
  size_t count_integers_missing_values(const size_t&) const;

  size_t count_greater_than(const T&) const;
  size_t count_less_than(const T&) const;
  size_t count_greater_equal_to(const T&) const;

  size_t count_less_equal_to(const T&) const;

  size_t count_between(const T&, const T&) const;

  size_t count_contains(const string&) const;

  Vector<size_t> count_unique() const;

  // Indices methods

  size_t get_first_index(const T&) const;

  Vector<size_t> get_indices_equal_to(const T&) const;
  Vector<size_t> get_indices_equal_to(const Vector<T>&) const;

  Vector<size_t> get_indices_not_equal_to(const T&) const;
  Vector<size_t> get_indices_not_equal_to(const Vector<T>&) const;

  Vector<size_t> get_indices_less_than(const T&) const;
  Vector<size_t> get_indices_greater_than(const T&) const;

  Vector<size_t> get_indices_less_equal_to(const T&) const;
  Vector<size_t> get_indices_greater_equal_to(const T&) const;

  Vector<size_t> get_between_indices(const T&, const T&) const;

  Vector<size_t> get_indices_that_contains(const string&) const;

  Vector<T> merge(const string&, const char&) const;

  Vector< Matrix<T> > to_vector_matrix(const size_t&, const size_t&, const size_t&) const;

  // Filtering methods

  Vector<T> filter_equal_to(const T&) const;
  Vector<T> filter_not_equal_to(const T&) const;

  Vector<T> filter_equal_to(const Vector<T>&) const;
  Vector<T> filter_not_equal_to(const Vector<T>&) const;

  Vector<T> filter_positive() const;
  Vector<T> filter_negative() const;

  Vector<T> filter_minimum_maximum(const T&, const T&) const;

  Vector<double> perform_Box_Cox_transformation(const double& = 1) const;

  // Descriptives methods

  Vector<double> calculate_percentage(const size_t&) const;
  Vector<double> calculate_percentage() const;

  size_t calculate_cumulative_index(const T&) const;

  T calculate_sum() const;

  T calculate_partial_sum(const Vector<size_t>&) const;

  T calculate_sum_missing_values() const;

  T calculate_product() const;

  // Rank methods

  Vector<size_t> sort_ascending_indices() const;
  Vector<T> sort_ascending_values() const;

  Vector<size_t> calculate_lower_indices(const size_t&) const;
  Vector<T> calculate_lower_values(const size_t&) const;

  Vector<size_t> sort_descending_indices() const;
  Vector<T> sort_descending_values() const;

  Vector<size_t> calculate_less_rank() const;
  Vector<size_t> calculate_greater_rank() const;

  Vector<T> sort_rank(const Vector<size_t>&) const;

  // Mathematical operators

  inline Vector<T> operator = (const initializer_list<T>&) const;

  inline Vector<T> operator + (const T&) const;

  inline Vector<T> operator + (const Vector<T>&) const;

  inline Vector<T> operator - (const T&) const;

  inline Vector<T> operator - (const Vector<T>&) const;

  inline Vector<T> operator * (const T&) const;

  inline Vector<T> operator * (const Vector<T>&) const;

  inline Matrix<T> operator * (const Matrix<T>&) const;

  Vector<T> operator / (const T&) const;

  Vector<T> operator / (const Vector<T>&) const;

  void operator += (const T&);

  void operator += (const Vector<T>&);

  void operator -= (const T&);

  void operator -= (const Vector<T>&);

  void operator *= (const T&);

  void operator *= (const Vector<T>&);

  void operator /= (const T&);

  void operator /= (const Vector<T>&);


  // Arranging methods

  Matrix<T> to_diagonal_matrix() const;

  Vector<T> get_subvector(const size_t&, const size_t&) const;

  Vector<T> get_subvector(const Vector<size_t>&) const;

  Vector<T> get_subvector(const Vector<bool>&) const;

  Vector<T> get_subvector_random(const size_t&) const;

  Vector<T> get_first(const size_t &) const;

  Vector<T> get_last(const size_t &) const;

  Vector<T> delete_first(const size_t &) const;

  Vector<T> delete_last(const size_t &) const;

  Vector<T> get_integer_elements(const size_t&) const;
  Vector<T> get_integer_elements_missing_values(const size_t&) const;

  // File operations

  void load(const string &);

  void save(const string &, const char& = ' ') const;

  void embed(const size_t &, const Vector<T>&);

  Vector<T> insert_elements(const size_t &, const Vector<T>&);

  Vector<T> insert_element(const size_t &, const T&) const;

  Vector<T> delete_index(const size_t &) const;

  Vector<T> delete_indices(const Vector<size_t>&) const;

  Vector<T> delete_value(const T&) const;
  Vector<T> delete_values(const Vector<T>&) const;

  Vector<T> assemble(const Vector<T>&) const;
  static Vector<T> assemble(const Vector<Vector<T>>&);

  Vector<T> get_difference(const Vector<T>&) const;
  Vector<T> get_union(const Vector<T>&) const;
  Vector<T> get_intersection(const Vector<T>&) const;

  Vector<T> get_unique_elements() const;

  void print_unique_string() const;
  void print_unique_number() const;

  Vector<T> calculate_top_string(const size_t&) const;
  Vector<T> calculate_top_number(const size_t&) const;

  void print_top_string(const size_t&) const;

  vector<T> to_std_vector() const;

  Vector<float> to_float_vector() const;
  Vector<double> to_double_vector() const;
  Vector<int> to_int_vector() const;
  Vector<size_t> to_size_t_vector() const;
  Vector<time_t> to_time_t_vector() const;
  Vector<bool> to_bool_vector() const;

  Matrix<T> to_binary_matrix() const;

  Vector<string> to_string_vector() const;

  Vector<double> string_to_double() const;
  Vector<int> string_to_int() const;
  Vector<size_t> string_to_size_t() const;
  Vector<time_t> string_to_time_t() const;

  Vector<Vector<T>> split(const size_t&) const;

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
  Tensor<T> to_tensor(const Vector<size_t>&) const;


};


/// Default constructor.
/// It creates a vector of size zero.

template <class T> Vector<T>::Vector() : vector<T>() {}


/// Size constructor.
/// It creates a vector of a given number of elements.
/// @param new_size Size of vector.

template <class T>
Vector<T>::Vector(const size_t &new_size)
    : vector<T>(new_size) {}


/// Size and initialization constructor.
/// It creates a vector of size n,
/// and initializes all the elements to a given value.
/// @param new_size Size of Vector.
/// @param value Initialization value of Type.

template <class T>
Vector<T>::Vector(const size_t &new_size, const T&value)
    : vector<T>(new_size, value) {}


/// File constructor.
/// It creates a vector object by loading its members from a data file.
/// @param file_name Name of vector data file.

template <class T>
Vector<T>::Vector(const string &file_name)
    : vector<T>() {
  load(file_name);
}


/// Sequential constructor.
/// It creates a vector of size n=(last-first)/step, and initializes the items sequentially.
/// Depending on the starting, step and finishing values, this method can produce
/// a variety of sizes and data, related by n~(last-first)/step.
/// @param first First value of the vector.
/// @param step Sequence step.
/// @param last Last value of the vector.

template <class T>
Vector<T>::Vector(const T&first, const double &step, const T&last)
    : vector<T>() {
  set(first, step, last);
}


/// Input iterators constructor

template <class T>
template <class InputIterator>
Vector<T>::Vector(const InputIterator& first, const InputIterator& last): vector<T>(first, last) {}


/// Copy constructor.
/// It creates a copy of an existing Vector.
/// @param other_vector Vector to be copied.

template <class T>
Vector<T>::Vector(const Vector<T>&other_vector) : vector<T>(other_vector) {}


/// Copy constructor.
/// It creates a copy of an existing Vector.
/// @param other_vector Vector to be copied.

template <class T>
Vector<T>::Vector(const vector<T>&other_vector) : vector<T>(other_vector) {}


/// List constructor.
/// It creates a vector object by loading its members from a list.
/// @param list Number of vector list.

template <class T>
Vector<T>::Vector(const initializer_list<T>&list) : vector<T>(list) {}

/// Vector of vectors constructor.
/// It creates a Vector object from Vector of Vectors by loading members into
/// a vectors.
/// @param vectors Vector of vectors.

template <class T>
Vector<T>::Vector(const Vector<Vector<T>>& vectors)
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


/// Destroys the vector.

template <class T>
Vector<T>::~Vector()
{
}


template <class T>
Vector<T>& Vector<T>::operator = (const Vector<T>& other_vector)
{
    if(other_vector.size() != this->size())
    {
        this->resize(other_vector.size());
    }

    copy(other_vector.begin(), other_vector.end(), this->begin());

    return *this;
}


/// Equal to operator between this vector and a Type value.
/// It produces true if all the elements of this vector are equal to the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator == (const T&value) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] != value) {
      return false;
    }
  }

  return true;
}


/// Not equivalent relational operator between this vector and a Type value.
/// It produces true if some element of this vector is not equal to the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T> bool Vector<T>::operator!= (const T&value) const {
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] != value) {
      return true;
    }
  }

  return false;
}


/// Greater than relational operator between this vector and a Type value.
/// It produces true if all the elements of this vector are greater than the
/// Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Vector<T>::operator>(const T&value) const
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] <= value) {
      return false;
    }
  }

  return true;
}


/// Less than relational operator between this vector and a Type value.
/// It produces true if all the elements of this vector are less than the Type
/// value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Vector<T>::operator<(const T&value) const
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] >= value) {
      return false;
    }
  }

  return true;
}


/// Greater than or equal to than relational operator between this vector and a
/// Type value.
/// It produces true if all the elements of this vector are greater than or
/// equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Vector<T>::operator>= (const T&value) const
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < value) {
      return false;
    }
  }

  return true;
}


/// Less than or equal to than relational operator between this vector and a
/// Type value.
/// It produces true if all the elements of this vector are less than or equal
/// to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Vector<T>::operator<= (const T&value) const
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] > value) {
      return false;
    }
  }

  return true;
}


/// Sets the size of this vector to zero.

template <class T> void Vector<T>::set() { this->resize(0); }


/// Sets a new size to the vector.
/// It does not initialize the data.
/// @param new_size Size for the vector.

template <class T>
void Vector<T>::set(const size_t &new_size)
{
  this->resize(new_size);
}


/// Sets a new size to the vector and initializes all its elements with a given value.
/// @param new_size Size for the vector.
/// @param new_value Value to initialize.

template <class T>
void Vector<T>::set(const size_t &new_size, const T&new_value)
{
  this->resize(new_size);

  initialize(new_value);
}


/// Sets all the elements of a vector object by loading them from a data file.
/// The format is specified in the OpenNN manual.
/// @param file_name Name of data file.

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
void Vector<T>::set(const T&first, const double &step, const T&last) {
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

template <class T>
void Vector<T>::set(const Vector<T> &other_vector)
{
  if(other_vector.size() != this->size())
  {
      this->resize(other_vector.size());
  }

  copy(other_vector.begin(), other_vector.end(), this->begin());
}


/// Return the first element of the vector.

template <class T>
const T& Vector<T>::get_first() const
{
    return (*this)[0];
}

/// Return the last element of the vector.

template <class T>
const T& Vector<T>::get_last() const
{
    const size_t this_size = this->size();

    return (*this)[this_size-1];
}

/// Return the penultimate element of the vector.

template <class T>
const T& Vector<T>::get_before_last() const
{
    const size_t this_size = this->size();

    return (*this)[this_size-2];
}


/// Erases the first elements of the vector.
/// @param elements_number Number of elements to be deleted.

template <class T>
Vector<T> Vector<T>::delete_first(const size_t & elements_number) const
{
    const size_t new_size = this->size() - elements_number;

    return get_last(new_size);
}


/// Erases the last elements of the vector.
/// @param elements_number Number of elements to be deleted.

template <class T>
Vector<T> Vector<T>::delete_last(const size_t & elements_number) const
{
    const size_t new_size = this->size() - elements_number;

    return get_first(new_size);
}


/// Initializes all the elements of the vector with a given value.
/// @param value Type value.

template <class T>
void Vector<T>::initialize(const T&value)
{
  fill(this->begin(),this->end(), value);
}

///Initializes all the elements of the vector with a given value.
/// @param first Number of elements what are going to be initializated.
/// @param value Value given to replace the original ones.

template <class T>
void Vector<T>::initialize_first(const size_t& first, const T&value)
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
/// to each element in the vector.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

template <class T>
void Vector<T>::randomize_uniform(const T& minimum,
                                  const T& maximum)
{


#ifdef __OPENNN_DEBUG__

  if(minimum > maximum) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const T&, const T&) method.\n"
           << "Minimum value must be less or equal than maximum value.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++)
  {
        (*this)[i] = calculate_random_uniform(minimum, maximum);
  }
}


/// Assigns a random value comprised between given minimum and a maximum values
/// to every element in the vector.
/// @param minimums Minimum initialization values.
/// @param maximums Maximum initialization values.

template <class T>
void Vector<T>::randomize_uniform(const Vector<T>&minimums,
                                  const Vector<T>&maximums)
{

    const size_t this_size = this->size();



#ifdef __OPENNN_DEBUG__

  const size_t minimums_size = minimums.size();
  const size_t maximums_size = maximums.size();

  if(minimums_size != this_size || maximums_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void randomize_uniform(const Vector<T>&, const Vector<T>&) method.\n"
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

  for(size_t i = 0; i < this_size; i++)
  {
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
                                 const double &standard_deviation)
{

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

  for(size_t i = 0; i < this_size; i++)
  {
   (*this)[i] = calculate_random_normal(mean, standard_deviation);
  }
}

/// Assigns random values to each element in the vector.
/// These are taken from normal distributions with given means and standard
/// deviations for each element.
/// @param mean Mean values of normal distributions.
/// @param standard_deviation Standard deviation values of normal distributions.

template <class T>
void Vector<T>::randomize_normal(const Vector<double>&mean,
                                 const Vector<double>&standard_deviation)
{
  const size_t this_size = this->size();

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


/// Assigns random values to each element in the vector.
/// These are taken from normal distributions with given positives and negatives ratio.
/// @param negatives_ratio .
/// @param positives_ratio .

template <class T>
void Vector<T>::randomize_binary(const double& negatives_ratio, const double& positives_ratio)
{
    const size_t this_size = this->size();

    if(this_size == 0)
    {
        return;
    }

    const double total_ratio = abs(negatives_ratio) + positives_ratio;

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

    // Negatives

    size_t count_negatives = 0;

    while(count_negatives != negatives_number)
    {
       index = indices[i];

      (*this)[index] = 0;
       count_negatives++;

       i++;
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

    return new_vector;
}

/// This method obtains values until a given index from this vector,
/// then assembles these values and the given values to fill in.
/// It returns a new vector of dimension n+n'.
/// @param from_index Index of elements to be obtained.
/// @param fill_with Values vector to fill.

template <class T>
Vector<T> Vector<T>::fill_from(const size_t& from_index, const Vector<T>& fill_with) const
{
    Vector<T> new_vector(*this);

    size_t counter = 0;

    for(size_t i = from_index; i < from_index+fill_with.size(); i++)
    {
        new_vector[i] = fill_with[counter];

        counter++;
    }

    return new_vector;
}


/// Returns true if the vector contains a given value, and false otherwise.
/// @param value Value to be compared.

template <class T>
bool Vector<T>::contains(const T&value) const
{
    Vector<T> copy(*this);

    typename vector<T>::iterator it = find(copy.begin(), copy.end(), value);

    return(it != copy.end());
}


/// Returns true if all elements of the vector is greater than given value,
/// and false otherwise.
/// @param value Value to be compared.

template <class T>
bool Vector<T>::contains_greater_than(const T&value) const
{

    for(size_t i = 0; i < this->size(); i++)
    {
        if((*this)[i] > value) return true;
    }

    return false;
}


/// Returns true if the vector contains a certain value from a given set, and
/// false otherwise.
/// @param values Vector values to be compared.

template <class T>
bool Vector<T>::contains(const Vector<T>&values) const
{
  if(values.empty()) {
    return false;
  }

  Vector<T> copy(*this);

  const size_t values_size = values.size();

  for(size_t j = 0; j < values_size; j++)
  {
      typename vector<T>::iterator it = find(copy.begin(), copy.end(), values[j]);

      if(it != copy.end())
      {
          return true;
      }
  }

  return false;
}


/// Returns true if Vector contains the same elements than other vector,
/// and false otherwise.
/// @param other_vector Vector to be compared.

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
bool Vector<T>::is_in(const T&minimum, const T&maximum) const
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
    if((*this)[i] < minimum ||(*this)[i] > maximum) {
      return false;
    }
  }

  return true;
}


/// Returns true if all the elements have the same value within a defined
/// tolerance , and false otherwise.
/// @param tolerance Tolerance value, so that if abs(max-min) <= tol, then the
/// vector is considered constant.

template <class T>
bool Vector<T>::is_constant(const double &tolerance) const
{
  const size_t this_size = this->size();

  if(this_size == 0) {
    return false;
  }

  for(size_t i = 0; i < this_size; i++)
  {
      if(abs((*this)[i] - (*this)[0]) > tolerance) return false;
  }

  return true;
}


/// Returns true if all the elements in this vector of strings are equal, and false otherwise.

template <class T>
bool Vector<T>::is_constant_string() const
{

  const size_t this_size = this->size();

  if(this_size == 0) {
    return false;
  }

  for(size_t i = 1; i < this_size; i++)
  {
      if((*this)[i] != (*this)[0])
      {
          return false;
      }
  }

  return true;
}


/// Returns true if all the elements in the vector have values which increase
/// with the index, and false otherwise.

template <class T>
bool Vector<T>::is_crescent() const
{
    if(this->size() == 1) return true;

  for(size_t i = 0; i < this->size() - 1; i++)
  {
    if((*this)[i] > (*this)[i + 1]) return false;
  }

  return true;
}


/// Returns true if all the elements in the vector have values which decrease
/// with the index, and false otherwise.

template <class T>
bool Vector<T>::is_decrescent() const
{
  if(this->size() == 1) return true;

  for(size_t i = 0; i < this->size() - 1; i++)
  {
    if((*this)[i] < (*this)[i+1]) return false;
  }

  return true;
}


/// Returns true if all the elements of this vector are equal or greater than zero, and false otherwise.

template <class T>
bool Vector<T>::is_positive() const
{
  for(size_t i = 0; i < this->size(); i++)
  {
    if((*this)[i] < 0.0)
    {
      return false;
    }
  }

  return true;
}

/// Returns true if all the elements of this vector are less than zero, and false otherwise.

template <class T>
bool Vector<T>::is_negative() const
{
  for(size_t i = 0; i < this->size(); i++)
  {
    if((*this)[i] > 0.0)
    {
      return false;
    }
  }

  return true;
}


/// Returns true if all the elemnts are separated by the same period, and false otherwise.
/// @param period Value of period by which the elements are separeted.

template <class T>
bool Vector<T>::check_period(const double& period) const
{
    for(size_t i = 1; i < this->size(); i++)
    {
        if((*this)[i] != (*this)[i-1] + period)
        {
            return false;
        }
    }

    return true;
}


/// Returns true if all the elements in the vector have binary values, and false otherwise.

template <class T>
bool Vector<T>::is_binary() const
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

template <class T>
bool Vector<T>::is_binary_missing_values() const
{
    if(this->get_unique_elements().size() != 2)
    {
        return false;
    }
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(!::isnan((*this)[i]))
        {
            if((*this)[i] != 0 &&(*this)[i] != 1)
            {
                return false;
            }
        }
    }

    return true;
}


///Returns true if all elements are zero or one.

template <class T>
bool Vector<T>::is_binary_0_1() const
{
    if(this->get_unique_elements().size() != 2)
    {
        return false;
    }
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

template <class T>
bool Vector<T>::is_integer() const
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

template <class T>
bool Vector<T>::is_integer_missing_values() const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(!::isnan((*this)[i]))
        {
            if(floor((*this)[i]) != (*this)[i])
            {
                return false;
            }
        }
    }

    return true;
}


/// Returns the reverse vector.
/// Note that the new vector has different order.

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


/// Returns the number of elements that are equal to a given value is contained in the vector.
/// @param value Value to be counted.

template <class T>
size_t Vector<T>::count_equal_to(const T& value) const
{
  return count(this->begin(), this->end(), value);
}


/// Returns the number of elements that are equal to a given values is contained in the vector.
/// @param values Vector of values to be counted.

template <class T>
size_t Vector<T>::count_equal_to(const Vector<T>& values) const
{
    size_t count = 0;

    const size_t this_size = this->size();
    const size_t values_size = values.size();

    for(size_t i = 0; i < this_size; i++)
    {
        for(size_t j = 0; j < values_size; j++)
        {
            if((*this)[i] == values[j]) count++;
        }
    }

    return count;
}


/// Returns the number of std NANs that are contained in the vector.

template <class T>
size_t Vector<T>::count_NAN() const
{
    size_t count = 0;

    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(::isnan((*this)[i])) count++;
    }

    return count;
}


/// Returns the number of std not NANs that are contained in the vector.

template <class T>
size_t Vector<T>::count_not_NAN() const
{
    size_t count = 0;

    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(!::isnan((*this)[i])) count++;
    }

    return count;
}


/// Returns the number of elements that are not equal to a given value is contained in the vector.
/// @param value Value to be counted.

template <class T>
size_t Vector<T>::count_not_equal_to(const T&value) const
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

  return count;
}


/// Returns the number of elements that are not equal to a given value is contained in the vector.
/// @param values Values vector to be counted.

template <class T> size_t Vector<T>::count_not_equal_to(const Vector<T>&values) const
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
        if((*this)[i] >= 0)
        {
            count++;
        }
    }

  return count;
}

/// Returns the number of elements that are smaller than zero.

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

  return count;
}


/// Returns a vector with the elements that are equal to a given value is contained in the vector.
/// @param value Comparison value.

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
            new_vector[index] = (*this)[i];
            index++;
        }
    }

    return new_vector;
}

/// Returns a indices vector with contains the elements that are equal to a given values.
/// @param values Values vector with the comparison values.

template <class T>
Vector<T> Vector<T>::filter_equal_to(const Vector<T>& values) const
{
    const Vector<size_t> indices = get_indices_equal_to(values);

    return get_subvector(indices);
}


/// Returns a vector with the elements of this vector that are different from a given value.
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

    return new_vector;
}


/// Returns a vector with the elements of this vector that are different from a given values.
/// @param values Vector with the comparison values.

template <class T>
Vector<T> Vector<T>:: filter_not_equal_to(const Vector<T>& values) const
{
    const Vector<size_t> indices = get_indices_not_equal_to(values);

    return get_subvector(indices);
}


/// Returns a vector that contains the elements of this vector that are equal to or greater than zero.

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

    return new_vector;

}


/// Returns a vector that contains the elements of this vector that are less than zero.

template <class T>
Vector<T> Vector<T>::get_negative_elements() const
{
    const size_t this_size = this->size();
    const size_t new_size = count_negative();

    Vector<T> new_vector(new_size);

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] < 0)
        {
            new_vector[count] = (*this)[i];
            count++;
        }
    }

    return new_vector;
}

/// Returns true if any element of this vector contains at least one NAN, and false otherwise.

template <class T>
bool Vector<T>::has_nan() const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(::isnan((*this)[i])) return true;
    }

    return false;
}


/// Returns the number of different integers in the vector
/// or 0 if the number of different integers in the vector is greater than a given number,
/// also returns 0 if the vector does not contain any integers.
/// @param maximum_integers Maximum number of different integers to count.

template <class T>
size_t Vector<T>::count_integers(const size_t& maximum_integers) const
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


/// Returns the number of different integers in the vector
/// or 0 if the number of different integers in the vector is greater than a given number,
/// also returns 0 if the vector does not contain any integers.
/// @param maximum_integers Maximum number of different integers to count.

template <class T>
size_t Vector<T>::count_integers_missing_values(const size_t& maximum_integers) const
{
    if(!this->is_integer_missing_values())
    {
        return 0;
    }

    const size_t this_size = this->size();

    Vector<T> integers;
    size_t integers_count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!::isnan((*this)[i]))
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


/// Returns a indices vector in which the elements of this vector take some value between the given values.
/// @param minimum Type value that represent the lower bound.
/// @param maximum Type value that represent the upper bound.

template <class T>
Vector<size_t> Vector<T>::get_between_indices(const T& minimum, const T& maximum) const
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


/// Returns the indices vector of the elements that are equal to given values.
/// @param values Vector of values to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_equal_to(const Vector<T>&values) const
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


/// Returns the indices vector of the elements that are not equal to a given value.
/// @param value Value to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_not_equal_to(const T&value) const
{

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


/// Returns the indices vector of the elements that are not equal to given values.
/// @param values Vector of values to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_not_equal_to(const Vector<T>&values) const
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

/// Returns the indices vector of the elements that are equal to a given value.
/// @param value Value to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_equal_to(const T&value) const
{

  const size_t this_size = this->size();

  const size_t equal_to_count = count_equal_to(value);

  if(equal_to_count == 0)
  {
      Vector<size_t> occurrence_indices;

      return(occurrence_indices);
  }

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


/// Returns the indices vector of the elements that are less than a given value.
/// @param value Value to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_less_than(const T&value) const
{

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


/// Returns the indices vector of the elements that are grater than a given value.
/// @param value Value to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_greater_than(const T&value) const
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

  return indices;
}


/// Returns the number of elements that are greater than a given value.
/// @param value Value to be compared.

template <class T>
size_t Vector<T>::count_greater_than(const T&value) const
{
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] > value)
    {
      count++;
    }
  }

  return count;
}


/// Returns the number of elements that are less than a given value.
/// @param value Value to be compared.

template <class T>
size_t Vector<T>::count_less_than(const T&value) const
{
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] < value)
    {
      count++;
    }
  }

  return count;
}


/// Returns the number of elements that are greater than or equal to a given value.
/// @param value Value to be compared.

template <class T>
size_t Vector<T>::count_greater_equal_to(const T&value) const
{
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] >= value)
    {
      count++;
    }
  }

  return count;
}


/// Returns the number of elements that are less than or equal to a given value.
/// @param value Value to be compared.

template <class T>
size_t Vector<T>::count_less_equal_to(const T&value) const
{
    const size_t count = count_if(this->begin(), this->end(), [value](T elem){ return elem <= value; });

    return count;
}


/// Returns the number of elements which are equal or greater than a minimum given value
/// and equal or less than a maximum given value.
/// @param minimum Minimum value.
/// @param maximum Maximum value.

template <class T>
size_t Vector<T>::count_between(const T&minimum, const T&maximum) const
{
  const size_t this_size = this->size();

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] >= minimum &&(*this)[i] <= maximum) count++;
  }

  return count;
}


/// Returns the number of occurrences of a given substring in the vector.
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

    return count;
}


/// Insert a string at the end of each element in this vector.
/// @param substring Substring to be appended.
/// @param separator Delimiter char between both strings.

template <class T>
Vector<T> Vector<T>::merge(const string& substring, const char& separator) const
{
    const size_t this_size = this->size();

    Vector<T> merged(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        merged[i] = (*this)[i] + separator + substring;
    }

    return(merged);
}


/// Returns a vector with the elements that are equal or greater than a given minimum value
/// and less or equal to a given maximum value of this vector.
/// @param minimum Minimum value to be filtered.
/// @param maximum Maximum value to be filtered.

template <class T>
Vector<T> Vector<T>::filter_minimum_maximum(const T&minimum, const T&maximum) const
{
    const size_t this_size = this->size();
    const size_t new_size = count_between(minimum, maximum);

    Vector<T> new_vector(new_size);

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] >= minimum && (*this)[i] <= maximum)
    {
      new_vector[count] = (*this)[i];
      count++;
    }
  }

  return new_vector;
}


/// Returns a vector with the indices of the elements of this vector that contain a given string.
/// @param find_what String to be found in the vector.

template <class T>
Vector<size_t> Vector<T>::get_indices_that_contains(const string& find_what) const
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

    return indices;
}


/// Returns a vector with the indices of the elements of this vector are less or equal to a given value.
/// @param value Value to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_less_equal_to(const T&value) const
{
  const size_t this_size = this->size();

  Vector<size_t> less_than_indices;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] <= value)
    {
      less_than_indices.push_back(i);
    }
  }

  return less_than_indices;
}


/// Returns a vector with the indices of the elements of this vector are greater or equal to a given value.
/// @param value Value to be compared.

template <class T>
Vector<size_t> Vector<T>::get_indices_greater_equal_to(const T&value) const
{

  const size_t this_size = this->size();

  Vector<size_t> greater_than_indices;

  for(size_t i = 0; i < this_size; i++)
  {
    if((*this)[i] >= value)
    {
      greater_than_indices.push_back(i);
    }
  }

  return greater_than_indices;
}


/// Returns vector with the Box-Cox transformation.
/// @param lambda Exponent of the Box-Cox transformation.

template <class T> Vector<double> Vector<T>::perform_Box_Cox_transformation(const double& lambda) const
{
    const size_t size = this->size();

    Vector<double> vector_tranformation(size);

    for(size_t i = 0; i < size; i++)
    {
        if(abs(lambda - 0) < numeric_limits<double>::epsilon())
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
/// @param total_sum Sum of all the elements of the vector

template <class T>
Vector<double> Vector<T>::calculate_percentage(const size_t& total_sum) const
{
    const size_t this_size = this->size();

    Vector<double> percentage_vector(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        percentage_vector[i] = static_cast<double>((*this)[i])*100.0/static_cast<double>(total_sum);
    }

    return percentage_vector;
}


/// Returns a vector containing the relative frequencies of the elements.

template <class T> Vector<double> Vector<T>::calculate_percentage() const
{
    const size_t this_size = this->size();

    const size_t total_sum = this->calculate_sum();

    Vector<double> percentage_vector(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        percentage_vector[i] = static_cast<double>((*this)[i])*100.0/static_cast<double>(total_sum);
    }

    return percentage_vector;
}


/// Finds the first element in the vector with a given value, and returns its index.
/// @param value Value to be found.

template <class T>
size_t Vector<T>::get_first_index(const T& value) const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if((*this)[i] == value) return i;
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t get_first_index(const T&) const.\n"
           << "Value not found in vector.\n";

    throw logic_error(buffer.str());
}


/// This method applies only to cumulative vectors.
/// It returns the index of the first element which is greater than a given value.
/// @param value Value to be found.

template <class T>
size_t Vector<T>::calculate_cumulative_index(const T&value) const
{
  const size_t this_size = this->size();


#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t calculate_cumulative_index(const T&) const.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

  T cumulative_value = (*this)[this_size - 1];

  if(value > cumulative_value)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "size_t calculate_cumulative_index(const T&) const.\n"
           << "Value(" << value << ") must be less than cumulative value("
           << cumulative_value << ").\n";

    throw logic_error(buffer.str());
  }

  for(size_t i = 1; i < this_size; i++)
  {
    if((*this)[i] <(*this)[i - 1])
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "int calculate_cumulative_index(const T&) const.\n"
             << "Vector elements must be crescent.\n";

      throw logic_error(buffer.str());
    }
  }

#endif

  if(value <= (*this)[0])
  {
    return 0;
  }

  for(size_t i = 1; i < this_size; i++)
  {
    if(value >(*this)[i - 1] && value <= (*this)[i])
    {
      return(i);
    }
  }

  return(this_size - 1);
}


/// Returns the sum of the elements of this vector.

template <class T>
T Vector<T>::calculate_sum() const
{
  const size_t this_size = this->size();

  T sum = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    sum += (*this)[i];
  }

  return sum;
}


/// Returns the sum of the elements with the given indices.
/// @param indices Indices of the elementes to sum.

template <class T>
T Vector<T>::calculate_partial_sum(const Vector<size_t>&indices) const
{
  const size_t this_size = this->size();

  T sum = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if(indices.contains(i))
    {
      sum += (*this)[i];
    }
  }

  return sum;
}


/// Returns the sum of the elements of this vector, ignoring the given values.

template <class T>
T Vector<T>::calculate_sum_missing_values() const
{
  const size_t this_size = this->size();

  T sum = 0;

  for(size_t i = 0; i < this_size; i++)
  {
    if(!::isnan((*this)[i]))
    {
      sum += (*this)[i];
    }
  }

  return sum;
}


/// Returns the product of the elements of this vector.

template <class T>
T Vector<T>::calculate_product() const
{
  const size_t this_size = this->size();

  T product = 1;

  for(size_t i = 0; i < this_size; i++)
  {
    product *= (*this)[i];
  }

  return product;
}


/// Returns a vector of the indices of this vector sorted by ascending values.

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
    sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {return (*this)[i1] < (*this)[i2];});

#endif

    return indices;
}


/// Returns a vector of the elements of this vector sorted by ascending values.

template <class T>
Vector<T> Vector<T>::sort_ascending_values() const
{
    Vector<T> sorted(*this);

    sort(sorted.begin(), sorted.end());

    return sorted;
}


/// Returns a vector with the indices of the smallest value elements of this vector.
/// Note that the size of the new vector is given by the number of elements chosen.
/// @param indices_number Number of subvector size with lower indices.

template <class T>
Vector<size_t> Vector<T>::calculate_lower_indices(const size_t& indices_number) const
{
    return sort_ascending_indices().get_subvector(0,indices_number-1);
}


/// Returns a vector with the smallest value elements of this vector.
/// Note that the size of the new vector is given by the number of elements chosen.
/// @param indices_number Number of subvector size with lower values elements.

template <class T>
Vector<T> Vector<T>::calculate_lower_values(const size_t& indices_number) const
{
    return sort_ascending_values().get_subvector(0,indices_number-1);
}


/// Returns a vector of the indices of this vector sorted by descending values.

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

    sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {return (*this)[i1] > (*this)[i2];});

#endif

    return indices;
}


/// Returns a vector of the elements of this vector sorted by descending values.

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

template <class T>
Vector<size_t> Vector<T>::calculate_less_rank() const
{
  const size_t this_size = this->size();

  Vector<size_t> rank(this_size);

  Vector<T> sorted_vector(*this);

  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  Vector<size_t> previous_rank;
  previous_rank.set(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
    for(size_t j = 0; j < this_size; j++)
    {
      if(previous_rank.contains(static_cast<size_t>(j))) continue;

      if((*this)[i] == sorted_vector[j])
      {
        rank[static_cast<size_t>(i)] = j;

        previous_rank[static_cast<size_t>(i)] = j;

        break;
      }
    }
  }

  return rank;
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
    previous_rank.set(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        for(size_t j = 0; j < this_size; j++)
        {
            if(previous_rank.contains(j)) continue;

            if(sorted_vector[i] == (*this)[j])
            {
                rank[i] = j;

                previous_rank[i] = j;

                break;
            }
        }
    }

    return rank;
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


/// Sets all the entries to a given list.
/// @param list List of values.

template <class T>
inline Vector<T> Vector<T>::operator = (const initializer_list<T>& list) const
{
  return Vector<T>(list);
}


/// Sum vector+scalar arithmetic operator.
/// @param scalar Scalar value to be added to this vector.

template <class T>
inline Vector<T> Vector<T>::operator+ (const T&scalar) const
{
  const size_t this_size = this->size();

  Vector<T> sum(this_size);

  transform(this->begin(), this->end(), sum.begin(),
                 bind2nd(plus<T>(), scalar));

  return sum;
}


/// Sum vector+vector arithmetic operator.
/// @param other_vector Vector to be added to this vector.

template <class T>
inline Vector<T> Vector<T>::operator+ (const Vector<T>&other_vector) const
{
  const size_t this_size = this->size();

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

  return sum;
}


/// Difference vector-scalar arithmetic operator.
/// @param scalar Scalar value to be subtracted to this vector.

template <class T>
inline Vector<T> Vector<T>::operator-(const T&scalar) const
{
  const size_t this_size = this->size();

  Vector<T> difference(this_size);

  transform(this->begin(), this->end(), difference.begin(),
                 bind2nd(minus<T>(), scalar));

  return(difference);
}


/// Difference vector-vector arithmetic operator.
/// @param other_vector vector to be subtracted to this vector.

template <class T>
inline Vector<T> Vector<T>::operator-(const Vector<T>&other_vector) const
{
  const size_t this_size = this->size();



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

template <class T> Vector<T> Vector<T>::operator*(const T&scalar) const
{
  const size_t this_size = this->size();

  Vector<T> product(this_size);

  transform(this->begin(), this->end(), product.begin(),
                 bind2nd(multiplies<T>(), scalar));

  return product;
}


/// Element by element product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

template <class T>
inline Vector<T> Vector<T>::operator*(const Vector<T>&other_vector) const
{
  const size_t this_size = this->size();

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

  return product;
}


/// Element by row product vector*matrix arithmetic operator.
/// @param matrix matrix to be multiplied to this vector.

template <class T>
Matrix<T> Vector<T>::operator*(const Matrix<T>& matrix) const
{
  const size_t rows_number = matrix.get_rows_number();
  const size_t columns_number = matrix.get_columns_number();

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(rows_number != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Matrix<T> operator *(const Matrix<T>&) const.\n"
           << "Number of matrix rows(" << rows_number << ") must be equal to vector size(" << this_size << ").\n";

    throw logic_error(buffer.str());
  }

#endif

  Matrix<T> product(rows_number, columns_number);

  for(size_t i = 0; i < rows_number; i++) {
    for(size_t j = 0; j < columns_number; j++) {
      product(i, j) = (*this)[i] * matrix(i, j);
    }
  }

  return product;
}


/// Cocient vector/scalar arithmetic operator.
/// @param scalar Scalar value to be divided to this vector.

template <class T> Vector<T> Vector<T>::operator/(const T&scalar) const
{
  const size_t this_size = this->size();

  Vector<T> cocient(this_size);

  transform(this->begin(), this->end(), cocient.begin(),
                 bind2nd(divides<T>(), scalar));

  return(cocient);
}


/// Cocient vector/vector arithmetic operator.
/// @param other_vector vector to be divided to this vector.

template <class T>
Vector<T> Vector<T>::operator/(const Vector<T>&other_vector) const
{
  const size_t this_size = this->size();



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

template <class T> void Vector<T>::operator+= (const T&value)
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] + value;
  }
}


/// Vector sum and assignment operator.
/// @param other_vector Vector to be added to this vector.

template <class T> void Vector<T>::operator+= (const Vector<T>&other_vector)
{
  const size_t this_size = this->size();



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

template <class T> void Vector<T>::operator-= (const T&value)
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] - value;
  }
}


/// Vector rest and assignment operator.
/// @param other_vector Vector to be subtracted to this vector.

template <class T> void Vector<T>::operator-= (const Vector<T>&other_vector)
{
  const size_t this_size = this->size();



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

template <class T> void Vector<T>::operator*= (const T&value)
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++) {
   (*this)[i] = (*this)[i] * value;
  }
}


/// Vector product and assignment operator.
/// @param other_vector Vector to be multiplied to this vector.

template <class T>
void Vector<T>::operator*= (const Vector<T>&other_vector)
{
  const size_t this_size = this->size();

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

template <class T> void Vector<T>::operator/= (const T&value)
{
  const size_t this_size = this->size();

  for(size_t i = 0; i < this_size; i++)
  {
   (*this)[i] = (*this)[i] / value;
  }
}


/// Vector division and assignment operator.
/// @param other_vector Vector to be divided to this vector.

template <class T> void Vector<T>::operator/= (const Vector<T>&other_vector)
{
  const size_t this_size = this->size();



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

  for(size_t i = 0; i < this_size; i++)
  {
   (*this)[i] = (*this)[i] / other_vector[i];
  }
}


/// Sets all the negative elements in this vector to zero.

template <class T>
Vector<T> Vector<T>::filter_positive() const
{
  Vector<T> new_vector(*this);

  for(size_t i = 0; i < this->size(); i++)
  {
    if(new_vector[i] < 0)
    {
      new_vector[i] = 0;
    }
  }

  return new_vector;
}


/// Sets all the positive elements in this vector to zero.

template <class T>
Vector<T> Vector<T>::filter_negative() const
{
    Vector<T> new_vector(*this);
  for(size_t i = 0; i < this->size(); i++)
  {
    if(new_vector[i] > 0)
    {
      new_vector[i] = 0;
    }
  }
  return  new_vector;
}


/// Returns a squared matrix in which the entries outside the main diagonal are
/// all zero.
/// The elements in the diagonal are the elements in this vector.

template <class T>
Matrix<T> Vector<T>::to_diagonal_matrix() const
{
  const size_t this_size = this->size();

  Matrix<T> matrix(this_size, this_size, 0.0);

  matrix.set_diagonal(*this);

  return matrix;
}


/// Returns another vector whose elements are the elements of this vector between two given indices.
/// Note that the new vector will be smaller than this vector, the size is given by n = last_index - first_index
/// @param first_index Index of lower element from whose the new vector begins.
/// @param last_index Index of upper element from whose the new vector endata_set.

template <class T>
Vector<T> Vector<T>::get_subvector(const size_t& first_index, const size_t& last_index) const
{
#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(first_index >= this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> get_subvector(const size_t&, const size_t&) const method.\n"
             << "First index (" << first_index << ") is equal or greater than size (" << this_size << ").\n";

      throw logic_error(buffer.str());
  }

  if(last_index >= this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> get_subvector(const size_t&, const size_t&) const method.\n"
             << "Last index (" << last_index << ") is equal or greater than size (" << this_size << ").\n";

      throw logic_error(buffer.str());
  }

#endif

  Vector<T> subvector(last_index-first_index + 1);

  for(size_t i = first_index ; i < last_index+1 ; i++)
  {
    subvector[i-first_index] = (*this)[i];
  }

  return subvector;
}


/// Returns another vector whose elements are given by some elements of this vector.
/// Note that the new vector will be smaller than this vector,
/// the size is given by the size of the given index vector.
/// @param indices Indices of this vector whose elements are required.

template <class T>
Vector<T> Vector<T>::get_subvector(const Vector<size_t>&indices) const
{
  const size_t new_size = indices.size();

  if(new_size == 0) return Vector<T>();

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  for(size_t i = 0; i < new_size; i++) {
    if(indices[i] > this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> get_subvector(const Vector<T>&) const method.\n"
             << "Index (" << indices[i] << ") is equal or greater than this size.\n";

      throw logic_error(buffer.str());
    }
  }

#endif

  Vector<T> subvector(new_size);

  for(size_t i = 0; i < new_size; i++)
  {
    subvector[i] = (*this)[indices[i]];
  }

  return subvector;
}


/// Returns a subvector that contains the true boolean.
/// @param selection Boolean vector.

template <class T>
Vector<T> Vector<T>::get_subvector(const Vector<bool>& selection) const
{
    const Vector<size_t> indices = selection.get_indices_equal_to(true);

    return(get_subvector(indices));
}


/// Returns a subvector obtained from the elements of this vector randomly.
/// @param new_size Size of new vector.

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
Vector<T> Vector<T>::get_first(const size_t &elements_number) const
{


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

  return subvector;
}


/// Returns a vector with the last n elements of this vector.
/// @param elements_number Size of the new vector.

template <class T>
Vector<T> Vector<T>::get_last(const size_t &elements_number) const
{
  const size_t this_size = this->size();


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

  for(size_t i = 0; i < elements_number; i++)
  {
    subvector[i] = (*this)[i + this_size - elements_number];
  }

  return subvector;
}


/// Returns a vector with the diferent integers of the vector
/// or return 0 if this number is greater than maximum number of integers.
/// @param maximum_integers Maximum number of integers to get.

template <class T>
Vector<T> Vector<T>::get_integer_elements(const size_t& maximum_integers) const
{

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


/// Returns a vector with the integers of this vector.

template <class T>
Vector<T> Vector<T>::get_integer_elements_missing_values(const size_t& maximum_integers) const
{

    const size_t this_size = this->size();

    const size_t integers_number = this->count_integers_missing_values(maximum_integers);

    Vector<T> integers(integers_number);
    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!::isnan((*this)[i]))
        {
            if(!integers.contains((*this)[i]))
            {
                integers[index] = (*this)[i];
                index++;

                if(index > integers_number) break;
            }
        }
    }

    return integers;
}


/// Loads the members of a vector from an data file.
/// Please be careful with the file format, which is specified in the OpenNN
/// manual.
/// @param file_name Name of vector file.

template <class T> void Vector<T>::load(const string &file_name)
{
  ifstream file(file_name.c_str());

  stringstream buffer;

  string line;

  while(file.good())
  {
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
/// @param separator Char delimiter.

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
      file << separator << (*this)[i];
    }

    file << endl;
  }

  // Close file

  file.close();
}


/// Insert another vector starting from a given position.
/// Note that the size of the new vector may be different.
/// @param position Insertion position.
/// @param other_vector Vector to be inserted.

template <class T>
void Vector<T>::embed(const size_t &position, const Vector<T>&other_vector)
{
  const size_t other_size = other_vector.size();

#ifdef __OPENNN_DEBUG__

  const size_t this_size = this->size();

  if(position + other_size > this_size)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "void tuck_in(const size_t &, const Vector<T>&) method.\n"
           << "Cannot tuck in vector.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < other_size; i++)
  {
   (*this)[position + i] = other_vector[i];
  }
}


template <class T>
Vector<T> Vector<T>::insert_elements(const size_t & position, const Vector<T>& other_vector)
{

    const size_t other_size =other_vector.size();
    const size_t new_size = this->size() + other_size;

    Vector<T> new_vector(new_size);

    for(size_t i = 0 ; i < position ; i++)
    {
        new_vector[i] = (*this)[i];
    }
    for(size_t i = position ; i < position + other_size ; i++)
    {
        new_vector[i] = other_vector[i-position];
    }
    for(size_t i = position + other_size ; i < new_size ; i++)
    {
        new_vector[i] = (*this)[i-other_size];
    }

    return  new_vector;
}

/// Returns a new vector with a new element inserted in a given position.
/// @param index Position of the new element.
/// @param value Value of the new element to be inserted.

template <class T>
Vector<T> Vector<T>::insert_element(const size_t &index, const T&value) const
{
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

  return other_vector;
}


/// Returns a new vector where a given value has been replaced by another value.
/// @param find_value Value to be replaced.
/// @param replace_value Replacement value.

template <class T>
void Vector<T>::replace_value(const T& find_value, const T& replace_value)
{
    for(size_t i = 0; i < this->size(); i++)
    {
        if((*this)[i] == find_value) (*this)[i] = replace_value;
    }
}


/// Returns a new vector which is a copy of this vector but with a given element
/// removed.
/// Therefore, the size of the new vector is the size of this vector minus one.
/// @param index Index of element to be removed.

template <class T>
Vector<T> Vector<T>::delete_index(const size_t &index) const
{
  const size_t this_size = this->size();

#ifdef __OPENNN_DEBUG__

  if(index >= this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> remove_element(const size_t&) const method.\n"
           << "Index("<<index<<") is equal or greater than vector size("<<this_size<<").\n";

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

  return other_vector;
}


/// Returns a new vector which is a copy of this vector but with a given elements
/// removed.
/// Therefore, the size of the new vector is the size of this vector minus the indices vector size.
/// @param indices Vector with the indices of the elements to be removed.

template <class T>
Vector<T> Vector<T>::delete_indices(const Vector<size_t>&indices) const
{
    const size_t this_size = this->size();
    const size_t indices_size = indices.size();

  

  #ifdef __OPENNN_DEBUG__

//    if(maximum(indices) >= this_size) {
//      ostringstream buffer;

//      buffer << "OpenNN Exception: Vector Template.\n"
//             << "Vector<T> remove_elements(const Vector<size_t>&) const method.\n"
//             << "Maximum index is equal or greater than vector size.\n";

//      throw logic_error(buffer.str());
//    }

    if(indices_size > this_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "Vector<T> delete_indices(const Vector<size_t>&) const method.\n"
             << "Number of indices("<< indices_size << ") to remove is greater than vector size(" << this_size << ").\n";

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

    return other_vector;
}


/// Construct a copy of this vector but without a certain value.
/// Note that the new vector might have a different size than this vector.
/// @param value Value of elements to be removed.

template <class T>
Vector<T> Vector<T>::delete_value(const T&value) const
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

/// Construct a copy of this vector but without a given values.
/// Note that the new vector might have a different size than this vector.
/// @param values Vector values to be removed.

template <class T>
Vector<T> Vector<T>::delete_values(const Vector<T>&values) const
{
    Vector<T> new_vector(*this);

    for(size_t i = 0; i < values.size(); i++)
    {
        new_vector = new_vector.delete_value(values[i]);
    }

    return new_vector;
}


/// Assemble two vectors.
/// @param other_vector Vector to be assembled to this vector.

template <class T>
Vector<T> Vector<T>::assemble(const Vector<T>&other_vector) const
{
  const size_t this_size = this->size();
  const size_t other_size = other_vector.size();

  if(this_size == 0 && other_size == 0)
  {
    Vector<T> assembly;

    return assembly;
  }
  else if(this_size == 0)
  {
    return other_vector;
  }
  else if(other_size == 0)
  {
    return *this;
  }
  else
  {
    Vector<T> assembly(this_size + other_size);

    for(size_t i = 0; i < this_size; i++)
    {
      assembly[i] = (*this)[i];
    }

    for(size_t i = 0; i < other_size; i++)
    {
      assembly[this_size + i] = other_vector[i];
    }

    return assembly;
  }
}


/// Assemble a vector of vectors to this vector.
/// @param vectors Vector of vectors to be assembled to this vector.

template <class T>
Vector<T> Vector<T>::assemble(const Vector<Vector<T>>& vectors)
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

    return new_vector;
}


/// Returns a vector which is the difference of this vector and another vector.
/// For instance, if this vector is(1,2,3) and the other vector is(1,4,3,3),
/// the difference is (2), the element in the first vector which is not present in the second.
/// @param other_vector Other vector to be compared.

template <class T>
Vector<T> Vector<T>::get_difference(const Vector<T>&other_vector) const
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
/// the union is(1,2,3,4), the elements that are present in the two vectors.
/// @param other_vector Vector to be joined.

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
/// the new vector will be (b, c).
/// @param other_vector Another vector.

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


/// Returns a vector with the unique values of the vector sorted.
/// For instance, if the vector is (a, b, a), the new vector will be (a, b).

template <class T>
Vector<T> Vector<T>::get_unique_elements() const
{
    Vector<T> copy_vector(*this);

    sort(copy_vector.begin(), copy_vector.end());

    const auto last = unique(copy_vector.begin(), copy_vector.end());

    copy_vector.erase(last, copy_vector.end());

    return copy_vector;
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


/// Prints to the screen the unique elements of the string vector, the number of that elements and the corresponding percentage.
/// It sorts the elements from greater than smaller.
/// For instance, for the vector({"a", "b", "a"}), it will print
/// a: 2(66.6%), 
/// b: 1(33.3%).

template <class T>
void Vector<T>::print_unique_string() const
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
        cout << unique_matrix(i,0) << ": " << unique_matrix(i,1) << " (" <<  unique_matrix(i,2) << "%)" << endl;
    }
}


/// Prints to the screen the unique elements of the number vector, the number of that elements and the corresponding percentage.
/// It sorts the elements from greater than smaller.
/// For instance, for the vector({1, 2, 1}), it will print
/// 2: 1(33.3%),
/// 1: 2(66.6%).

template <class T>
void Vector<T>::print_unique_number() const
{
    const size_t this_size = this->size();

    const Vector<T> unique = get_unique_elements();

    const Vector<size_t> count = count_unique();

    const Vector<double> percentage = count_unique().to_double_vector()*(100.0/static_cast<double>(this_size));

    const size_t unique_size = unique.size();


    Matrix<T> unique_matrix(unique_size, 3);
    unique_matrix.set_column(0, unique);
    unique_matrix.set_column(1, count);
    unique_matrix.set_column(2, percentage);

    unique_matrix = unique_matrix.sort_descending(1);

    cout << "Total: " << this_size << endl;

    for(size_t i = 0; i < unique_size; i++)
    {
        cout << unique_matrix(i,0) << ": " << unique_matrix(i,1) << " (" <<  unique_matrix(i,2) << "%)" << endl;
    }
}


/// Return a vector which contains the elements of a given string vector
/// according to the number of times that they appear in these vector
/// For instance, if the given vector is ({"a", "a", "b", "b", "c", "b"}), it returns ({"b", "a", "c"})
/// @param rank Number of top elements that are printed

template <class T>
Vector<T> Vector<T>::calculate_top_string(const size_t& rank) const
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


/// Return a vector which contains the elements of a given number vector
/// according to the number of times that they appear in these vector
/// For instance, if the given vector is ({"1", "1", "2", "2", "3", "2"}), it returns ({"2", "1", "3"})
/// @param rank Number of top elements that are printed

template <class T>
Vector<T> Vector<T>::calculate_top_number(const size_t& rank) const
{
    const Vector<T> unique = get_unique_elements();

    const Vector<size_t> count = count_unique();

    const Vector<double> count_double = count.to_double_vector();

    const size_t unique_size = unique.size();

    Matrix<T> unique_matrix(unique_size, 2);
    unique_matrix.set_column(0, unique);
    unique_matrix.set_column(1, count_double);

    unique_matrix = unique_matrix.sort_descending(1);

    const size_t end = unique_size < rank ? unique_size : rank;

    const Vector<T> top = unique_matrix.get_column(0).get_first(end);

   return top;
}

/// Prints to the screen the unique elements of this vector, the number of that elements and the corresponding percentage.
/// It sorts the elements from greater than smaller and only prints the top ones.
/// @param rank Number of top elements that are printed.

template <class T>
void Vector<T>::print_top_string(const size_t& rank) const
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
        cout << i+1 << ". " << unique_matrix(i,0) << ": " << unique_matrix(i,1) << " (" <<  unique_matrix(i,2) << "%)" << endl;
    }
}


/// Returns a std vector with the size and elements of this OpenNN vector.

template <class T>
vector<T> Vector<T>::to_std_vector() const
{
  const size_t this_size = this->size();

  vector<T> std_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
    std_vector[i] = (*this)[i];
  }

  return(std_vector);
}


/// Returns a new vector with the elements of this vector casted to float.

template <class T>
Vector<float> Vector<T>::to_float_vector() const
{
  const size_t this_size = this->size();

  Vector<float> float_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      float_vector[i] = static_cast<float>((*this)[i]);
  }

  return(float_vector);
}


/// Returns a new vector with the elements of this vector casted to double.

template <class T>
Vector<double> Vector<T>::to_double_vector() const
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
Vector<int> Vector<T>::to_int_vector() const
{
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
Vector<size_t> Vector<T>::to_size_t_vector() const
{
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
Vector<time_t> Vector<T>::to_time_t_vector() const
{
  const size_t this_size = this->size();

  Vector<time_t> size_t_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      size_t_vector[i] = static_cast<time_t>((*this)[i]);
  }

  return(size_t_vector);
}


/// Returns a boolean vector with the same size, where returns true if that value is unique
/// and false otherwise.

template <class T>
Vector<bool> Vector<T>::to_bool_vector() const
{
  const Vector<T> unique = get_unique_elements();

//  if(unique.size() != 2)
//  {
//       ostringstream buffer;

//       buffer << "OpenNN Exception: Vector Template.\n"
//              << "Vector<bool> to_bool_vector() const.\n"
//              << "Number of unique items(" << get_unique_elements().size() << ") must be 2.\n";

//       throw logic_error(buffer.str());
//  }

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

  return new_vector;
}


/// Returns an array in binary form, where the unique elements of this vector take the value 1
/// and the rest of elements 0.
/// Note that the size of the matrix has rows as elements have the vector,
/// and columns the number of unique elements that the vector has.

template <class T>
Matrix<T> Vector<T>::to_binary_matrix() const
{
  const Vector<T> unique = get_unique_elements();

  const size_t unique_number = unique.size();

  const size_t this_size = this->size();

  Matrix<T> new_matrix(this_size, unique_number, 0.0);

  for(size_t i = 0; i < unique_number; i++)
  {
      for(size_t j = 0; j < this_size; j++)
      {
          if((*this)[j] == unique[j])
          {
              new_matrix(j, i) = 1.0;
          }
      }
  }

  return new_matrix;
}



/// Returns a new vector with the elements of this vector converted to string.

template <class T>
Vector<string> Vector<T>::to_string_vector() const
{
  const size_t this_size = this->size();

  Vector<string> string_vector(this_size);
  ostringstream buffer;

    for(size_t i = 0; i < this_size; i++)
    {
      buffer.str("");
      buffer << (*this)[i];

      string_vector[i] = buffer.str();
   }

  return(string_vector);
}


/// Returns a new vector with the elements of this string vector casted to double.

template <class T>
Vector<double> Vector<T>::string_to_double() const
{
  const size_t this_size = this->size();

  Vector<double> double_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
      try
      {
          stringstream buffer;

          buffer << (*this)[i];

          double_vector[i] = stod(buffer.str());
      }
      catch(const logic_error&)
      {
         double_vector[i] = nan("");
      }
   }

  return double_vector;
}


/// Returns a new vector with the elements of this string vector casted to int.

template <class T>
Vector<int> Vector<T>::string_to_int() const
{
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
         int_vector[i] = -999999999;
      }
   }

  return int_vector;
}


/// Returns a new vector with the elements of this string vector casted to size_t.

template <class T>
Vector<size_t> Vector<T>::string_to_size_t() const
{
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
         size_t_vector[i] = 999999;
      }
   }

  return size_t_vector;
}


/// Returns a new vector with the elements of this string vector casted to time_t.

template <class T>
Vector<time_t> Vector<T>::string_to_time_t() const
{
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
         time_vector[i] = -1;
      }
   }

  return time_vector;
}

/// Returns a vector of vectors whit vectors size n. This method split vector
/// in several vectors.
/// @param n Vectors size.

template <class T>
Vector<Vector<T>> Vector<T>::split(const size_t& n) const
{
    if(this->size() < n)
    {
        return Vector<Vector<T>>({*this});
    }

    // determine number of sub-vectors of size n

    //    const size_t batches_number = (this->size() - 1) / n + 1;
    const size_t batches_number = this->size() / n;

    // create array of vectors to store the sub-vectors

    Vector<Vector<T>> batches(batches_number);

    // each iteration of this loop process next set of n elements
    // and store it in a vector at k'th index in vec

    for(size_t k = 0; k < batches_number; ++k)
    {
        // get range for next set of n elements

        auto start_itr = next(this->cbegin(), k*n);

        auto end_itr = k*n + n > this->size() ? this->cend() : next(this->cbegin(), k*n + n);

        // allocate memory for the sub-vector

        batches[k].resize(n);

        // code to handle the last sub-vector as it might
        // contain less elements

//        if(k*n + n > this->size())
//        {
//            batches[k].resize(this->size() - k*n);
//        }

        // copy elements from the input range to the sub-vector

        copy(start_itr, end_itr, batches[k].begin());
    }

    return batches;
}


/// Returns a row matrix with number of rows equal to one
/// and number of columns equal to the size of this vector.

template <class T>
Matrix<T> Vector<T>::to_row_matrix() const
{

  const size_t this_size = this->size();

  Matrix<T> matrix(1, this_size);

  for(size_t i = 0; i < this_size; i++)
  {
    matrix(0, i) = (*this)[i];
  }

  return matrix;
}


/// Returns a column matrix with number of rows equal to the size of this vector
/// and number of columns equal to one.

template <class T>
Matrix<T> Vector<T>::to_column_matrix() const
{
  const size_t this_size = this->size();

  Matrix<T> matrix(this_size, 1);

  for(size_t i = 0; i < this_size; i++) {
    matrix(i, 0) = (*this)[i];
  }

  return matrix;
}


/// This method takes a string representation of a vector and sets this vector
/// to have size equal to the number of words and values equal to that words.
/// @param str String to be parsed.

template <class T>
void Vector<T>::parse(const string &str)
{
  if(str.empty())
  {
    set();
  }
  else
  {
    istringstream buffer(str);

    istream_iterator<string> first(buffer);
    istream_iterator<string> last;

    Vector<string> str_vector(first, last);

    const size_t new_size = str_vector.size();

    if(new_size > 0)
    {
      this->resize(new_size);

      buffer.clear();
      buffer.seekg(0, ios::beg);

      for(size_t i = 0; i < new_size; i++)
      {
        buffer >>(*this)[i];
      }
    }

   }
}


/// Returns a string representation of this vector.
/// @param separator Char between the elements(, -, /, etc).
/// @param quotation Quotation char for the elements(", ').

template <class T>
string Vector<T>::vector_to_string(const char& separator, const char& quotation) const
{
    ostringstream buffer;

    const size_t this_size = this->size();

    if(this_size > 0)
    {

        buffer << quotation <<(*this)[0] << quotation;

        for(size_t i = 1; i < this_size; i++)
        {

            buffer << separator << quotation << (*this)[i] << quotation;
        }
    }

    return buffer.str();
}


/// Returns a string representation of this vector.
/// @param separator Char between the elements(, -, /, etc).

template <class T>
string Vector<T>::vector_to_string(const char& separator) const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++)
    {
      buffer << separator << (*this)[i];
    }
  }

  return buffer.str();
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
        buffer << ' ' << (*this)[i];
    }
  }

  return buffer.str();
}


/// Returns a new vector with the elements of this stack vector casted to string.

template <class T>
string Vector<T>::stack_vector_to_string() const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1 ; i < this_size ; i++)
    {
        buffer << (*this)[i];
    }
  }

  return buffer.str();
}


/// Returns a string representation of this vector which can be inserted in a text.

template <class T>
string Vector<T>::to_text(const char& separator) const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++)
    {
      buffer << separator << (*this)[i];
    }
  }

  return buffer.str();
}

/// Returns a string representation of this vector which can be inserted in a text.

template <class T>
string Vector<T>::to_text(const string& separator) const
{
  ostringstream buffer;

  const size_t this_size = this->size();

  if(this_size > 0)
  {
    buffer <<(*this)[0];

    for(size_t i = 1; i < this_size; i++)
    {
      buffer << separator << (*this)[i];
    }
  }

  return buffer.str();
}


/// This method retuns a vector of strings with size equal to the size of this
/// vector and elements equal to string representations of the elements of this
/// vector.

template <class T>
Vector<string> Vector<T>::write_string_vector(const size_t &precision) const
{
  const size_t this_size = this->size();

  Vector<string> string_vector(this_size);

  ostringstream buffer;

  for(size_t i = 0; i < this_size; i++)
  {
    buffer.str("");
    buffer << setprecision(precision) << (*this)[i];

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
                               const size_t &columns_number) const
{


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

  return matrix;
}


/// Returns a Tensor representation of this vector.
/// @param dimensions Numbers of tensor dimensions.

template <class T>
Tensor<T> Vector<T>::to_tensor(const Vector<size_t>& dimensions) const
{
//  Tensor<T> tensor({dimensions});
  Tensor<T> tensor(dimensions); // for linux

  for(size_t i = 0; i < this->size(); i++)
  {
      tensor[i] = (*this)[i];
  }

  return tensor;
}


/// This method re-writes the inputs operator >> for the Vector template.
/// @param is Input stream.
/// @param v Input vector.

template <class T>
istream &operator>>(istream &is, Vector<T>&v)
{
  const size_t size = v.size();

  for(size_t i = 0; i < size; i++)
  {
    is >> v[i];
  }

  return is;
}


/// This method re-writes the output operator << for the Vector template.
/// @param os Output stream.
/// @param v Output vector.

template <class T>
ostream &operator<<(ostream &os, const Vector<T>&v)
{
  const size_t this_size = v.size();

  if(this_size > 0)
  {
    os << v[0];

    const char space = ' ';

    for(size_t i = 1; i < this_size; i++)
    {
      os << space << v[i];
    }
  }

  return os;
}


/// This method re-writes the output operator << for vectors of vectors.
/// @param os Output stream.
/// @param v Output vector of vectors.

template <class T>
ostream &operator<<(ostream &os, const Vector<Vector<T>>&v)
{
  for(size_t i = 0; i < v.size(); i++)
  {
    os << "subvector_" << i << "\n" << v[i] << endl;
  }

  return os;
}


/// This method re-writes the output operator << for vectors of matrices.
/// @param os Output stream.
/// @param v Output vector of matrices.

template <class T>
ostream &operator<<(ostream &os, const Vector< Matrix<T> >&v)
{
  for(size_t i = 0; i < v.size(); i++)
  {
    os << "submatrix_" << i << "\n" << v[i] << endl;
  }

  return os;
}


/// Returns a random number chosen from a uniform distribution.
/// @param minimum Minimum value.
/// @param maximum Maximum value.

template <class T>
T calculate_random_uniform(const T&minimum, const T&maximum)
{
  const T random = static_cast<T>(rand() /(RAND_MAX + 1.0));

  const T random_uniform = minimum + (maximum - minimum) * random;

  return(random_uniform);
}


/// Returns a number in string format.

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
T calculate_random_normal(const T& mean, const T& standard_deviation)
{
  const double pi = 4.0 * atan(1.0);

  T random_uniform_1;

  do
  {
    random_uniform_1 = static_cast<T>(rand()) /(RAND_MAX + 1.0);

  }
  while(random_uniform_1 == 0.0);

  const T random_uniform_2 = static_cast<T>(rand()) /(RAND_MAX + 1.0);

  // Box-Muller transformation

  const T random_normal = mean +
                          sqrt(-2.0 * log(random_uniform_1)) *
                              sin(2.0 * pi * random_uniform_2) *
                              standard_deviation;

  return(random_normal);
}


/// Prints on screen the elapsed time.
/// Can be used to know the runtime of a program.
/// @param elapsed_time Elapsed time in secondata_set.

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


/// This method converts a matrix vector into a conventional vector.
/// @param array Matrix vector to be converted.

template <class T>
Vector<T> to_vector(const Vector< Matrix<T> >& array)
{

    Vector<T> new_vector;

    for(size_t i = 0; i < array.size(); i++)
    {
        const Vector<T> other_vector = array[i].to_vector();
        new_vector.insert(new_vector.end(), other_vector.begin(), other_vector.end());
    }

    return new_vector;
}


/// This method converts this vector into an matrix vector.
/// @param rows_number Number of rows in the matrix.
/// @param columns_number Number of columns in the matrix.
/// @param channels_number Number of elements in the matrix vector.

template <class T>
Vector< Matrix<T> > Vector<T>::to_vector_matrix(const size_t& rows_number,
                                                const size_t& columns_number,
                                                const size_t& channels_number) const
{
    Vector< Matrix<T> > new_vector_matrix(channels_number, Matrix<T>(rows_number, columns_number, 0.0));
    const size_t vector_size = this->size();
    Vector<Vector<T>> channels_vector = this->split(vector_size / channels_number);
    for(size_t i = 0; i < channels_number; i++)
    {
        new_vector_matrix[i] = channels_vector[i].to_matrix(rows_number, columns_number);
    }

    return new_vector_matrix;
}

} // end namespace OpenNN

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
