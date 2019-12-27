//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T R I X   C O N T A I N E R                                       
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef MATRIX_H
#define MATRIX_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// OpenNN includes

#include "vector.h"
#include "tensor.h"
#include "opennn_strings.h"

using namespace std;

namespace OpenNN
{

/// This template class defines a matrix for general purpose use.

///
/// This matrix also implements some mathematical methods which can be useful. 

template <class T>
class Matrix : public vector<T>
{

public:

    // CONSTRUCTORS

    explicit Matrix();

    explicit Matrix(const size_t&, const size_t&);

    explicit Matrix(const size_t&, const size_t&, const T&);

    explicit Matrix(const string&, const char&, const bool&);

    Matrix(const Matrix&);

    Matrix(const initializer_list<Vector<T>>&);

    Matrix(const initializer_list<Vector<T>>&, const initializer_list<string>&);

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

    bool operator > (const Matrix<T>&) const;

    bool operator > (const T& value) const;

    bool operator < (const Matrix<T>&) const;

    bool operator < (const T& value) const;

    bool operator >= (const Matrix<T>&) const;

    bool operator >= (const T&) const;

    bool operator <= (const Matrix<T>&) const;

    bool operator <= (const T&) const;

    Matrix<T> operator + (const T&) const;

    Matrix<T> operator + (const Matrix<T>&) const;

    Matrix<T> operator - (const T& scalar) const;

    Matrix<T> operator - (const Matrix<T>&) const;

    Matrix<T> operator * (const T&) const;

    Matrix<T> operator * (const Matrix<T>&) const;

    Matrix<T> operator * (const Tensor<T>&) const;

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

    // Get methods

    const size_t& get_rows_number() const;

    const size_t& get_columns_number() const;

    const Vector<string> get_header() const;

    const string get_header(const size_t&) const;

    size_t get_column_index(const string&) const;

    Vector<size_t> get_columns_indices(const Vector<string>&) const;

    Vector<size_t> get_binary_columns_indices() const;

    Vector<size_t> get_rows_equal_to(const Vector<T>&) const;

    Matrix<T> get_submatrix(const Vector<size_t>&, const Vector<size_t>&) const;

    Tensor<T> get_tensor(const Vector<size_t>&, const Vector<size_t>&, const Vector<size_t>&) const;

    Matrix<T> get_submatrix_rows(const Vector<size_t>&) const;
    Matrix<T> get_submatrix_columns(const Vector<size_t>&) const;

    Vector<T> get_row(const size_t&) const;

    Vector<T> get_rows(const size_t&, const size_t&) const;

    Vector<T> get_row(const size_t&, const Vector<size_t>&) const;

    Vector<T> get_column(const size_t&) const;
    Vector<T> get_column(const string&) const;

    Matrix<T> get_columns(const Vector<string>&) const;

    Vector<T> get_column(const size_t&, const Vector<size_t>&) const;

    Vector<T> get_diagonal() const;

    T get_first(const size_t&) const;
    T get_first(const string&) const;

    T get_last(const size_t&) const;
    T get_last(const string&) const;

    Vector<size_t> get_constant_columns_indices() const;

    Matrix<T> get_first_rows(const size_t&) const;
    Matrix<T> get_first_columns(const size_t&) const;
    Matrix<T> get_last_columns(const size_t&) const;
    Matrix<T> get_last_rows(const size_t&) const;

    // Set methods

    void set();

    void set(const size_t&, const size_t&);

    void set(const size_t&, const size_t&, const T&);

    void set(const Matrix<T>&);

    void set(const string&);

    void set(const initializer_list<Vector<T>>&);

    void set_identity(const size_t&);

    void set_rows_number(const size_t&);

    void set_columns_number(const size_t&);

    void set_header(const Vector<string>&);

    void set_header(const size_t&, const string&);

    void set_row(const size_t&, const Vector<T>&);

    void set_row(const size_t&, const T&);

    void set_submatrix_rows(const size_t&, const Matrix<T>&);

    void set_column(const size_t&, const Vector<T>&, const string& = "");
    void set_column(const string&, const Vector<T>&, const string& = "");

    void set_column(const string&, const T&, const string& = "");
    void set_column(const size_t&, const T&, const string& = "");

    void set_diagonal(const T&);

    void set_diagonal(const Vector<T>&);

    // Check methods

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

    bool is_row_equal_to(const size_t&, const Vector<size_t>&, const T&) const;

    bool has_column_value(const size_t&, const T&) const;

    // Count methods

    size_t count_diagonal_elements() const;

    size_t count_off_diagonal_elements() const;

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

    size_t count_rows_equal_to(const Vector<size_t>&, const T&) const;
    size_t count_columns_equal_to(const Vector<T>&, const T&) const;

    Vector<size_t> count_unique_elements() const;

    Vector<size_t> count_column_occurrences(const T&) const;

    // Not a number

    bool has_nan() const;

    bool has_nan_row(const size_t&) const;
    bool has_nan_column(const size_t&) const;

    size_t count_nan() const;
    size_t count_not_NAN() const;

    size_t count_rows_with_nan() const;
    Vector<size_t> get_nan_indices() const;
    size_t count_columns_with_nan() const;

    Vector<size_t> count_nan_rows() const;
    Vector<size_t> count_nan_columns() const;

    // FILTER

    void filter(const T&, const T&);

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

    Matrix<T> filter_column_minimum_maximum(const size_t&, const T&, const T&) const;
    Matrix<T> filter_column_minimum_maximum(const string&, const T&, const T&) const;

    Matrix<T> filter_extreme_values(const size_t&, const double&, const double&) const;
    Matrix<T> filter_extreme_values(const string&, const double&, const double&) const;

    // INITIALIZATION

    void initialize(const T&);
    void initialize_sequential();

    void randomize_uniform(const double& = -1.0, const double& = 1.0);
    void randomize_uniform(const Vector<double>&, const Vector<double>&);
    void randomize_uniform(const Matrix<double>&, const Matrix<double>&);

    void randomize_normal(const double& = 0.0, const double& = 1.0);
    void randomize_normal(const Vector<double>&, const Vector<double>&);
    void randomize_normal(const Matrix<double>&, const Matrix<double>&);

    void initialize_identity();

    void initialize_diagonal(const T&);

    void initialize_diagonal(const size_t&, const T&);

    void initialize_diagonal(const size_t&, const Vector<T>&);

    void append_header(const string&);

    void embed(const size_t&, const size_t&, const Matrix<T>&);

    void embed(const size_t&, const size_t&, const Vector<T>&);

    void sum_diagonal(const T&);

    void multiply_diagonal(const T&);

    void sum_diagonal(const Vector<T>&);

    Matrix<T> append_row(const Vector<T>&) const;

    Matrix<T> append_column(const Vector<T>&, const string& = "") const;

    Matrix<T> insert_row(const size_t&, const Vector<T>&) const;

    void insert_row_values(const size_t&, const size_t&, const Vector<T>&);

    Matrix<T> insert_column(const size_t&, const Vector<T>&, const string& = "") const;
    Matrix<T> insert_column(const string&, const Vector<T>&, const string& = "") const;

    Matrix<T> insert_matrix(const size_t&, const Matrix<T>&) const;

    Matrix<T> insert_padding(const size_t& , const size_t&) const;

    Matrix<T> add_columns(const size_t&) const;
    Matrix<T> add_columns_first(const size_t&) const;

    void split_column(const string&, const Vector<string>&, const char& = ',', const string& = "NA");
    
    void split_column(const string&, const string&, const string&, const size_t&, const size_t&);

    void swap_columns(const size_t&, const size_t&);
    void swap_columns(const string&, const string&);

    // Merge mthods

    void merge_columns(const string&, const string&, const string&, const char&);
    void merge_columns(const size_t&, const size_t&, const char&);

    Matrix<T> merge_matrices(const Matrix<T>&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> merge_matrices(const Matrix<T>&, const size_t&, const size_t&) const;

    // Join methods

    Matrix<T> right_join(const Matrix<T>&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> right_join(const Matrix<T>&, const size_t&, const size_t&) const;

    Matrix<T> left_join(const Matrix<T>&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> left_join(const Matrix<T>&, const string&, const string&, const string&, const string&, const string& = "", const string& = "") const;
    Matrix<T> left_join(const Matrix<T>&, const size_t&, const size_t&) const;

    // Delete methods

    Matrix<T> delete_row(const size_t&) const;
    Matrix<T> delete_rows(const Vector<size_t>&) const;

    Matrix<T> delete_rows_with_value(const T&) const;
    Matrix<T> delete_columns_with_value(const T&) const;

    Matrix<T> delete_rows_equal_to(const T&) const;

    Matrix<T> delete_first_rows(const size_t&) const;
    Matrix<T> delete_last_rows(const size_t&) const;

    Matrix<T> delete_first_columns(const size_t&) const;
    Matrix<T> delete_last_columns(const size_t&) const;

    Matrix<T> delete_column(const size_t&) const;
    Matrix<T> delete_columns(const Vector<size_t>&) const;

    Matrix<T> delete_column(const string&) const;
    Matrix<T> delete_columns(const Vector<string>&) const;

    Matrix<T> delete_columns_name_contains(const Vector<string>&) const;

    Matrix<T> delete_constant_rows() const;
    Matrix<T> delete_constant_columns() const;

    Matrix<T> delete_binary_columns() const;

    Matrix<T> delete_binary_columns(const double&) const;

    // Assemble

    Matrix<T> assemble_rows(const Matrix<T>&) const;

    Matrix<T> assemble_columns(const Matrix<T>&) const;

    // SORTING

    Matrix<T> sort_ascending(const size_t&) const;
    Matrix<T> sort_descending(const size_t&) const;

    Matrix<T> sort_ascending_strings(const size_t&) const;
    Matrix<T> sort_descending_strings(const size_t&) const;

    Matrix<T> sort_rank_rows(const Vector<size_t>&) const;

    Matrix<T> sort_columns(const Vector<size_t>&) const;
    Matrix<T> sort_columns(const Vector<string>&) const;

    // REPLACE

    void replace(const T&, const T&);

    void replace_header(const string&, const string&);

    void replace_in_row(const size_t&, const T&, const T&);
    void replace_in_column(const size_t&, const T&, const T&);
    void replace_in_column(const string&, const T&, const T&);

    void replace_substring(const string&, const string&);
    void replace_substring(const size_t&, const string&, const string&);
    void replace_substring(const string&, const string&, const string&);

    void replace_contains(const string&, const string&);
    void replace_contains_in_row(const size_t&, const string&, const string&);

    void replace_column_equal_to(const size_t&, const T&, const T&);
    void replace_column_equal_to(const string&, const T&, const T&);
    void replace_column_not_equal_to(const string&, const T&, const T&);
    void replace_column_not_equal_to(const string&, const Vector<T>&, const T&);

    void replace_column_less_than_string(const string&, const double&, const T&);

    void replace_column_contains(const string&, const string&, const string&);
    size_t count_column_contains(const string&, const string&) const;

    // Mathematical methods

    T calculate_sum() const;

    Vector<T> calculate_rows_sum() const;
    Vector<T> calculate_columns_sum() const;
    T calculate_column_sum(const size_t&) const;
    T calculate_row_sum(const size_t&) const;

    void sum_row(const size_t&, const Vector<T>&);
    void sum_column(const size_t&, const Vector<T>&);
    Matrix<T> sum_rows(const Vector<T>&) const;
    Matrix<T> subtract_rows(const Vector<T>&) const;
    Matrix<T> multiply_rows(const Vector<T>&) const;
    Vector<Matrix<T>> multiply_rows(const Matrix<T>&) const;


    double calculate_trace() const;

    Vector<double> calculate_missing_values_percentage() const;

    Matrix<size_t> get_indices_less_than(const T&) const;

    Matrix<size_t> get_indices_greater_than(const T&) const;

    Matrix<T> calculate_reverse_columns() const;

    Matrix<T> calculate_transpose() const;

    bool compare_rows(const size_t&, const Matrix<T>&, const size_t&) const;

    void divide_rows(const Vector<T>&);

    // CONVERSIONS

    string matrix_to_string(const char& = ' ') const;

    Matrix<size_t> to_size_t_matrix() const;
    Matrix<float> to_float_matrix() const;
    Matrix<double> to_double_matrix() const;
    Matrix<string> to_string_matrix(const size_t& = 3) const;

    Matrix<double> to_zeros() const;
    Matrix<double> to_ones() const;

    Matrix<double> bool_to_double() const;

    Matrix<size_t> string_to_size_t() const;
    Matrix<double> string_to_double() const;

    vector<T> to_std_vector() const;

    Vector<T> to_vector() const;

    Vector<Vector<T>> to_vector_of_vectors() const;

    Vector< Matrix<T> > to_vector_matrix(const size_t&, const size_t&, const size_t&) const;

    Matrix<T> to_categorical(const size_t&) const;
    Tensor<T> to_tensor() const;

    // Serialization methods

    void print() const;

    void load_csv(const string&, const char& = ',', const bool& = false, const string& = "NA");
    void load_csv_string(const string&, const char& = ',', const bool& = true);
    void load_binary(const string&);

    void save_binary(const string&) const;

    void save_csv(const string&, const char& = ',',  const Vector<string>& = Vector<string>(), const string& = "Id") const;

    void save_json(const string&, const Vector<string>& = Vector<string>()) const;

    void parse(const string&);

    void print_preview() const;

private:

    /// Number of rows in the matrix.

    size_t rows_number;

    /// Number of columns in the matrix.

    size_t columns_number;

    /// Header with names of columns.

    Vector<string> header;

};


/// Default constructor. It creates a matrix with zero rows and zero columns.

template <class T>
Matrix<T>::Matrix() : vector<T>()
{
    set();
}


/// Constructor.
/// It creates a matrix with n rows and m columns, containing n*m copies of the default value for Type.
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


/// Constructor.
/// It creates a matrix with n rows and m columns, containing n*m copies of the type value of Type.
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
/// @param separator Separator type: ';' , ',', '/'.
/// @param header If matrix contains header write true, and false otherwise.

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
Matrix<T>::Matrix(const initializer_list<Vector<T>>& new_columns) : vector<T>()
{
    set(new_columns);
}


template <class T>
Matrix<T>::Matrix(const initializer_list<Vector<T>>& new_columns, const initializer_list<string>& new_header) : vector<T>()
{
    set(new_columns);

    set_header(new_header);
}


/// Destructor.

template <class T>
Matrix<T>::~Matrix()
{
    rows_number = 0;
    columns_number = 0;
}


/// Assignment operator. It assigns to self a copy of an existing matrix.
/// @param other_matrix Matrix to be assigned.

template <class T>
Matrix<T>& Matrix<T>::operator = (const Matrix<T>& other_matrix)
{
    if(other_matrix.rows_number != rows_number || other_matrix.columns_number != columns_number)
    {
        rows_number = other_matrix.rows_number;
        columns_number = other_matrix.columns_number;

        set(rows_number, columns_number);
    }

    copy(other_matrix.begin(), other_matrix.end(), this->begin());

    copy(other_matrix.header.begin(), other_matrix.header.end(), header.begin());

    return *this;
}


// REFERENCE OPERATORS

/// Reference operator.

/// Returns the element(i,j) of the matrix.
/// @param row Index of row.
/// @param column Index of column.

template <class T>
inline T& Matrix<T>::operator()(const size_t& row, const size_t& column)
{
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

   return (*this)[rows_number*column+row];
}


/// Reference operator.

/// Returns the element(i,j) of the matrix.
/// @param row Index of row.
/// @param column Index of column.

template <class T>
inline const T& Matrix<T>::operator()(const size_t& row, const size_t& column) const
{
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

   return (*this)[rows_number*column+row];
}


template <class T>
inline T& Matrix<T>::operator()(const size_t& row, const string& column_name)
{
   const size_t column = get_column_index(column_name);

   return (*this)(row, column);
}


/// Reference operator.

template <class T>
inline const T& Matrix<T>::operator()(const size_t& row, const string& column_name) const
{
    const size_t column = get_column_index(column_name);

   return (*this)(row, column);
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
       return false;
   }
   else if(other_columns_number != columns_number)
   {
        return false;
   }
   else
   {
       for(size_t i = 0; i < this->size(); i++)
       {
            if((*this)[i] != other_matrix[i])
            {
                return false;
            }
       }
   }

   return true;
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
            return false;
         }
   }

   return true;
}


/// Not equivalent relational operator between this matrix and other matrix.
/// It produces true if the two matrices have any not equal element, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator != (const Matrix<T>& other_matrix) const
{
   

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
            return true;
        }
   }

   return false;
}


/// Not equivalent relational operator between this matrix and a Type value.
/// It produces true if some element of this matrix is not equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator != (const T& value) const
{
   

   for(size_t i = 0; i < this->size(); i++)
   {
     if((*this)[i] != value)
     {
        return true;
     }
   }

   return false;
}


/// Greater than relational operator between this matrix and other vector.
/// It produces true if all the elements of this matrix are greater than the corresponding elements of the other matrix,
/// and false otherwise.
/// @param other_matrix matrix to be compared with.

template <class T>
bool Matrix<T>::operator >(const Matrix<T>& other_matrix) const
{
   

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
            return false;
        }
   }

   return true;
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
            return false;
        }
    }

    return true;
}


/// Less than relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are less than the corresponding elements of the other matrix,
/// and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator <(const Matrix<T>& other_matrix) const
{
   

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
           return false;
         }
   }

   return true;
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
           return false;
         }
   }

   return true;
}


/// Greater than or equal to relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are greater than or equal to the corresponding elements of the
/// other matrix, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator >= (const Matrix<T>& other_matrix) const
{
   

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
            return false;
         }
   }

   return true;
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
            return false;
         }
   }

   return true;
}


/// Less than or equal to relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are less than or equal to the corresponding elements of the
/// other matrix, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator <= (const Matrix<T>& other_matrix) const
{
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
            return false;
         }
   }

   return true;
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
            return false;
        }
   }

   return true;
}


/// Returns the number of rows in the matrix.

template <class T>
const size_t& Matrix<T>::get_rows_number() const
{
   return rows_number;
}


/// Returns the number of columns in the matrix.

template <class T>
const size_t& Matrix<T>::get_columns_number() const
{
   return columns_number;
}


/// Returns a string vector with the header.

template <class T>
const Vector<string> Matrix<T>::get_header() const
{
   return(header);
}


/// Returns a string with header in the column with index given.
/// @param index Column index.s

template <class T>
const string Matrix<T>::get_header(const size_t& index) const
{
   return(header[index]);
}


/// Returns the column index.
/// @param column_given String with the name of the column.

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

    return index;
}


/// Returns the indices of the column headers that are passed to the function
/// @param names Names of the headers columns to be obteined.

template <class T>
Vector<size_t> Matrix<T>::get_columns_indices(const Vector<string>& names) const
{
    const size_t size = names.size();

    if(header == "")
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "Vector<size_t> get_columns_indices(const Vector<string>&) const.\n"
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
               << "Vector<size_t> get_columns_indices(const Vector<string>&) const.\n"
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


///Returns a vector with the index of the binary columns

template <class T>
Vector<size_t> Matrix<T>::get_binary_columns_indices() const
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
    if(other_matrix.rows_number != this->rows_number || other_matrix.columns_number != this->columns_number)
    {
        rows_number = other_matrix.rows_number;
        columns_number = other_matrix.columns_number;

        set(rows_number, columns_number);
    }

    copy(other_matrix.begin(), other_matrix.end(), this->begin());

    copy(other_matrix.header.begin(), other_matrix.header.end(), header.begin());
}


/// Sets the members of this object by loading them from a data file.
/// @param file_name Name of data file.

template <class T>
void Matrix<T>::set(const string& file_name)
{
   load_csv(file_name);
}


/// Set the columns of the matrix

template <class T>
void Matrix<T>::set(const initializer_list<Vector<T>>& columns)
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
              << "void set(const initializer_list<Vector<T>>&) method.\n"
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
                   << "Matrix(const initializer_list<Vector<T>>& list) constructor.\n"
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


/// Sets a new header in the matrix.
/// @param new_header Names of matrix columns.

template <class T>
void Matrix<T>::set_header(const Vector<string>& new_header)
{
   this->header = new_header;
}


/// Sets a new header in the matrix.
/// @param index Column index to replace header.
/// @param index_name Name of matrix column.

template <class T>
void Matrix<T>::set_header(const size_t& index, const string& index_name)
{
    header[index] = index_name;
}


///Append a string in all the elememts of the header

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
void Matrix<T>::embed(const size_t& row_position, const size_t& column_position, const Matrix<T>& other_matrix)
{
   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

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
void Matrix<T>::embed(const size_t& row_position, const size_t& column_position, const Vector<T>& other_vector)
{
   const size_t other_columns_number = other_vector.size();

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

     for(size_t j = 0; j < other_columns_number; j++)
     {
       (*this)(row_position,column_position+j) = other_vector[j];
     }
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

    return count;
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

    return count;
}


/// Returns the number of elements in a given column that are equal to a given value.
/// @param value Value to find.

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

    return count;
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

    return count;
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

    return count;
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

    return count;
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

    return count;
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

    return count;
}


/// Looks for elements that have a given value in one column.
/// It returns the number of elements found.
/// @param column_1_name Name of a column.
/// @param value Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(count_equal_to(column_index, value));
}


/// Looks for elements that have a given values in one column.
/// It returns the number of elements found.
/// @param column_1_name Name of a column.
/// @param values Values to be found in the column above.

template <class T>
size_t Matrix<T>::count_equal_to(const string& column_name, const Vector<T>& values) const
{
    const size_t column_index = get_column_index(column_name);

    return(count_equal_to(column_index, values));
}


/// Looks for elements that have a given value in one column,
/// and another given value in another column.
/// It returns the number of elements found.
/// @param column_1_name Name of a column.
/// @param value_1 Value to be found in the column above.
/// @param column_2_name Name of another column.
/// @param value_2 Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_equal_to(const string& column_1_name, const T& value_1,
                                 const string& column_2_name, const T& value_2) const
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);

    return(count_equal_to(column_1_index, value_1, column_2_index, value_2));
}


/// Looks for elements that have a given value in all rows, row a row.
/// It returns a vector with the number of elements found,
/// each vector element correponds with the number of elements that contains given value in one row.
/// @param value Value to be found in the rows.

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


/// Looks for elements that have a given value in all rows, row a row.
/// It returns a vector with the number of elements found,
/// each vector element correponds with the number of elements that contains given value in one row.
/// @param value Value to be found in the rows.
/// @param weights Weights asigned to each row.

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


/// Returns the number of elements in a given column that are not equal to a given value.
/// @param value Value to find.

template <class T>
size_t Matrix<T>::count_not_equal_to(const T& value) const
{
    const size_t this_size = this->size();

    return(this_size-count_equal_to(value));
}


/// Returns the number of elements in a given column that are equal to a given value.
/// @param column_index Index of column.
/// @param value Value to find.

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

    return count;
}


/// Returns the number of elements in a given column that are not equal to a given set of values.
/// @param column_index Index of column.
/// @param values Vector of values.

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

    return index;
}


/// Looks for elements that not have a given value in one column,
/// and another given value in another column.
/// It returns the number of elements found.
/// @param column_1_index Index of a column.
/// @param value_1 Value to be found in the column above.
/// @param column_2_index Index of another column.
/// @param value_2 Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_not_equal_to(const size_t& column_1_index, const T& value_1,
                                     const size_t& column_2_index, const T& value_2) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i, column_1_index) != value_1 &&(*this)(i, column_2_index) != value_2)
        {
            count++;
        }
    }

    return count;
}


/// Returns the number of elements in a given column that are not equal to a given value.
/// @param column_name Name of column.
/// @param value Value to find.

template <class T>
size_t Matrix<T>::count_not_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(count_not_equal_to(column_index, value));
}


/// Looks for elements that not have a given value in one column,
/// and another given value in another column.
/// It returns the number of elements found.
/// @param column_1_name Name of a column.
/// @param value_1 Value to be found in the column above.
/// @param column_2_name Name of another column.
/// @param value_2 Value to be found in the column above.

template <class T>
size_t Matrix<T>::count_not_equal_to(const string& column_1_name, const T& value_1,
                                     const string& column_2_name, const T& value_2) const
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);

    return(count_not_equal_to(column_1_index, value_1, column_2_index, value_2));
}


/// Looks for elements that have a given value in all rows, row a row.
/// It returns the number of rows that all columns contains a given value.
/// @param value Value to be found in the rows.

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


/// Looks for elements that not have a given value in all rows, row a row.
/// It returns the number of rows that all columns contains a given value.
/// @param value Value to be found in the rows.

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


/// Returns a vector with each elemnt represent a row index of the matrix that
/// all columns have a given value.
/// @param value Value to be found.

template <class T>
Vector<size_t> Matrix<T>::get_rows_equal_to(const Vector<T>& vector) const
{
    Vector<size_t> indices;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(get_row(i) == vector)
        {
            indices.push_back(i);
        }
    }

    return indices;
}


/// Looks for elements that have a given value in all rows, row a row.
/// It returns the number of rows that given columns contains this value.
/// @param columns_indices Vector of column indices to check.
/// @param value Value to be found in the rows.

template <class T>
size_t Matrix<T>::count_rows_equal_to(const Vector<size_t>& columns_indices, const T& value) const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(is_row_equal_to(i, columns_indices, value))
        {
            count++;
        }
    }

    return count;
}


/// Check if a given row the given columns that are equal to a given value.
/// Returns true if given columns contains a given value, and false otherwise.
/// @param row_index Index of a row.
/// @param columns_indices Vector of column indices to check.
/// @param value Value to be found in the row.

template <class T>
bool Matrix<T>::is_row_equal_to(const size_t& row_index, const Vector<size_t>& columns_indices, const T& value) const
{
    const size_t columns_indices_size = columns_indices.size();

    for(size_t i = 0; i < columns_indices_size; i++)
    {
        if((*this)(row_index,columns_indices[i]) != value)
        {
            return false;
        }
    }

    return true;
}


/// Returns a matrix with the values of given rows and columns from this matrix.
/// @param row_indices Indices of matrix rows.
/// @param columns_indices Indices of matrix columns.

template <class T>
Matrix<T> Matrix<T>::get_submatrix(const Vector<size_t>& row_indices, const Vector<size_t>& columns_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t columns_indices_size = columns_indices.size();

   Matrix<T> sub_matrix(row_indices_size, columns_indices_size);

   size_t row_index;
   size_t column_index;

   for(size_t i = 0; i < row_indices_size; i++)
   {
      row_index = row_indices[i];

      for(size_t j = 0; j < columns_indices_size; j++)
      {
         column_index = columns_indices[j];
         sub_matrix(i,j) = (*this)(row_index,column_index);
      }
   }

   return sub_matrix;
}


template <class T>
Tensor<T> Matrix<T>::get_tensor(const Vector<size_t>& rows_indices,
                                const Vector<size_t>& columns_indices,
                                const Vector<size_t>& columns_dimensions) const
{
#ifdef __OPENNN_DEBUG__

    if(columns_indices.size() != columns_dimensions.calculate_product())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Tensor<T> get_tensor(const Vector<size_t>&, const Vector<size_t>&, const Vector<size_t>&) const method.\n"
              << "Size of columns indices(" << columns_indices.size() << ") must be equal to product of columns dimensions(" << columns_dimensions.calculate_product() << ").\n";

       throw logic_error(buffer.str());
    }

#endif

   const size_t rows_number = rows_indices.size();
   const size_t columns_number = columns_indices.size();

   const Vector<size_t> dimensions = Vector<size_t>(1, rows_number).assemble(columns_dimensions);

   Tensor<T> tensor(dimensions);

   size_t row_index;
   size_t column_index;

   size_t tensor_index = 0;

   for(size_t j = 0; j < columns_number; j++)
   {
       column_index = columns_indices[j];

      for(size_t i = 0; i < rows_number; i++)
      {
          row_index = rows_indices[i];

         tensor[tensor_index] = (*this)(row_index, column_index);

         tensor_index++;
      }
   }

   return tensor;
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

   return sub_matrix;
}


/// Returns a submatrix with the values of given columns from this matrix.
/// @param columns_indices Indices of matrix columns.

template <class T>
Matrix<T> Matrix<T>::get_submatrix_columns(const Vector<size_t>& columns_indices) const
{
    const size_t columns_indices_size = columns_indices.size();

    #ifdef __OPENNN_DEBUG__

    for(size_t i = 0; i < columns_indices_size; i++)
    {
        if(columns_indices[i] >= columns_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: Matrix Template.\n"
                  << "Matrix<T> get_submatrix_columns(const Vector<size_t>&) const method.\n"
                  << "Column index (" << i << ") must be less than number of columns(" << columns_number << ").\n";

           throw logic_error(buffer.str());
        }
    }

    #endif

   Matrix<T> sub_matrix(rows_number, columns_indices_size);

   size_t column_index;

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_indices_size; j++)
      {
         column_index = columns_indices[j];

         sub_matrix(i,j) = (*this)(i,column_index);
      }
   }

   if(!header.empty())
   {
       const Vector<string> sub_header = header.get_subvector(columns_indices);

       sub_matrix.set_header(sub_header);
   }

   return sub_matrix;
}


/// Returns the nth row of the matrix.
/// @param index Index of row.

template <class T>
Vector<T> Matrix<T>::get_row(const size_t& index) const
{
   #ifdef __OPENNN_DEBUG__

   if(index >= rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_row(const size_t&) const method.\n"
             << "Row index (" << index << ") must be less than number of rows (" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<T> row(columns_number);

   for(size_t j = 0; j < columns_number; j++)
   {
      row[j] = (*this)(index,j);
   }

   return row;
}


/// Returns the nth rows of the matrix in vector fomat.
/// @param first_index Index of the first row.
/// @param last_index Index of the last row.

template <class T>
Vector<T> Matrix<T>::get_rows(const size_t& first_index, const size_t& last_index) const
{
    #ifdef __OPENNN_DEBUG__

    if(last_index > rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Vector<T> get_rows(const size_t&, const size_t&) const method.\n"
              << "Last index (" << last_index << ") must be less than number of rows(" << rows_number << ").\n";

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


/// Returns the nth row of the matrix, but only the elements specified by given indices.
/// @param row_index Index of row.
/// @param columns_indices Column indices of row.

template <class T>
Vector<T> Matrix<T>::get_row(const size_t& row_index, const Vector<size_t>& columns_indices) const
{
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

   const size_t size = columns_indices.size();

   Vector<T> row(size);

   for(size_t i = 0; i < size; i++)
   {
      row[i] = (*this)(row_index,columns_indices[i]);
   }

   return row;
}


/// Returns the nth-column of the matrix.
/// @param column_index Index of column.

template <class T>
Vector<T> Matrix<T>::get_column(const size_t& column_index) const
{
   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_column(const size_t&) const method.\n"
             << "Column index (" << column_index << ") must be less than number of columns(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<T> column(rows_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      column[i] = (*this)(i,column_index);
   }

   return column;
}


/// Returns the nth-column of the matrix.
/// @param column_name Name of column.

template <class T>
Vector<T> Matrix<T>::get_column(const string& column_name) const
{
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



/// Returns a matrix with the elements of the columns which name has been given.
/// @param column_names Vector of columns names.

template <class T>
Matrix<T> Matrix<T>::get_columns(const Vector<string>& column_names) const
{
    const Vector<size_t> indices = get_columns_indices(column_names);

    return(get_submatrix_columns(indices));
}


/// Returns the nth-column of the matrix, but only those elements specified by given indices.
/// @param column_index Index of column.
/// @param row_indices Row indices of column.

template <class T>
Vector<T> Matrix<T>::get_column(const size_t& column_index, const Vector<size_t>& row_indices) const
{
   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_column(const size_t&, const Vector<size_t>&) const method.\n"
             << "Column index (" << column_index << ") must be less than number of rows(" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t size = row_indices.size();

   Vector<T> column(size);

   for(size_t i = 0; i < size; i++)
   {
      column[i] = (*this)(row_indices[i],column_index);
   }

   return column;
}


/// Returns true if matrix contains nan, and false otherwise.

template <class T>
bool Matrix<T>::has_nan() const
{
    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(::isnan((*this)[i])) return true;
    }

    return false;
}

/// Returns true if the row of the matrix contains nan, and false otherwise
/// @param row_index Row of the matrix

template <class T>
bool Matrix<T>::has_nan_row(const size_t& row_index) const
{
    for(size_t j = 0; j < columns_number; j++)
    {
        if(::isnan((*this)(row_index, j))) return true;
    }

    return false;
}


/// Returns true if the column of the matrix contains nan, and false otherwise
/// @param row_index Column of the matrix

template <class T>
bool Matrix<T>::has_nan_column(const size_t& column_index) const
{
    for(size_t i = 0; i < rows_number; i++)
    {
        if(::isnan((*this)(i, column_index))) return true;
    }

    return false;
}


/// Returns the number of elements that contains nan.

template <class T>
size_t Matrix<T>::count_nan() const
{
    const size_t this_size = this->size();

    size_t count = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(::isnan((*this)[i])) count++;
    }

    return count;
}


/// Returns the number of elements that not contains nan.

template <class T>
size_t Matrix<T>::count_not_NAN() const
{
    size_t count = 0;

    const size_t this_size = this->size();

    for(size_t i = 0; i < this_size; i++)
    {
        if(!::isnan((*this)[i])) count++;
    }

    return count;
}


/// Returns the number of rows which contains at least one nan.

template <class T>
size_t Matrix<T>::count_rows_with_nan() const
{
    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(::isnan((*this)(i,j)))
            {
                count++;

                break;
            }
        }

    }

    return count;
}


/// Returns the number of rows which contains at least one nan.

template <class T>
Vector<size_t> Matrix<T>::get_nan_indices() const
{
    Vector<size_t> nan_indices;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(::isnan((*this)(i,j)))
            {
                nan_indices.push_back(i);

                break;
            }
        }

    }

    return nan_indices;
}


/// Returns the number of columns which contains at least one nan.

template <class T>
size_t Matrix<T>::count_columns_with_nan() const
{
    size_t count = 0;

    for(size_t j = 0; j < columns_number; j++)
    {
        for(size_t i = 0; i < rows_number; i++)
        {
            if(::isnan((*this)(i,j))) count++;
            break;
        }
    }

    return count;
}


/// Return a vector with each element represent the number of nan
/// that contain each row.
/// Note the vector is the same dimenssion that the rows number of the matrix.

template <class T>
Vector<size_t> Matrix<T>::count_nan_rows() const
{
    Vector<size_t> count(rows_number, 0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(::isnan((*this)(i,j))) count[i]++;
        }
    }

    return count;
}


/// Return a vector with each element represent the number of nan
/// that contain each column.
/// Note the vector is the same dimesion that the column number of the matrix.

template <class T>
Vector<size_t> Matrix<T>::count_nan_columns() const
{
    Vector<size_t> count(columns_number, 0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(::isnan((*this)(i,j))) count[j]++;
        }
    }

    return count;
}


///Return the number of elements that are unquique in eacher column of the matrix.

template <class T>
Vector<size_t> Matrix<T>::count_unique_elements() const
{
    Vector<size_t> unique_elements;

    for(size_t i = 0; i < columns_number; i++)
    {
        unique_elements.push_back(this->get_column(i).get_unique_elements().size());
    }

    return unique_elements;
}


/// Returns the diagonal of the matrix.

template <class T>
Vector<T> Matrix<T>::get_diagonal() const
{
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
     this->set_row(row_index+i,submatrix.get_row(i));
   }

   if(header == "" && row_index == 0)
   {
       set_header(submatrix.get_header());
   }
}


/// Sets new values of a single column in the matrix.
/// @param column_index Index of column.
/// @param new_column New values of single column.
/// @param new_name New name of single column.

template <class T>
void Matrix<T>::set_column(const size_t& column_index, const Vector<T>& new_column, const string& new_name)
{
   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_column(const size_t&, const Vector<T>&).\n"
             << "index (" << column_index << ") must be less than number of columns(" << columns_number << ").\n";

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


/// Sets new values of a single column in the matrix.
/// @param column_name Name of column.
/// @param new_column New values of single column.
/// @param new_name New name of single column.

template <class T>
void Matrix<T>::set_column(const string& column_name, const Vector<T>& new_column, const string& new_name)
{
    const size_t column_index = get_column_index(column_name);

    this->set_column(column_index, new_column, new_name);
}


/// Sets new value of a single column in the matrix.
/// @param column_name Name of column.
/// @param value New value of single column.
/// @param new_name New name of single column.

template <class T>
void Matrix<T>::set_column(const string& column_name, const T& value, const string& new_name)
{
    const size_t column_index = get_column_index(column_name);

    set_column(column_index, value, new_name);
}


/// Sets new value of a single column in the matrix.
/// @param column_index Index of column.
/// @param value New value of single column.
/// @param new_name New name of single column.

template <class T>
void Matrix<T>::set_column(const size_t& column_index, const T& value, const string& new_name)
{
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
   this->set(new_size, new_size, 0);
   this->set_diagonal(new_value);
}


/// Sets this matrix to be diagonal.
/// A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero.
/// It also initializes the elements on the main diagonal to given values.
/// @param new_size Number of rows and colums in the matrix.
/// @param new_values Values of the elements in the main diagonal.

template <class T>
void Matrix<T>::initialize_diagonal(const size_t& new_size, const Vector<T>& new_values)
{
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


/// This method multiplies a new value to the diagonal elements in the matrix.
/// The matrix must be square.
/// @param value New multiplying value.

template <class T>
void Matrix<T>::multiply_diagonal(const T& value)
{
   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "multiply_diagonal(const T&).\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,i) = (*this)(i,i)*value;
   }
}


/// This method sums new values to the diagonal in the matrix.
/// The matrix must be square.
/// @param new_summing_values Vector of summing values.

template <class T>
void Matrix<T>::sum_diagonal(const Vector<T>& new_summing_values)
{
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

    if(!header.empty()) copy.set_header(header);

    return copy;
}


/// This method appends a new column to the matrix.
/// The size of the column vector must be equal to the number of rows of the matrix.
/// Note that resizing is necessary here and therefore this method can be very inefficient.
/// @param new_column Column to be appended.
/// @param new_name Name to the new column to be append.

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

   if(!header.empty())
   {
       Vector<string> new_header = get_header();

       new_header.push_back(new_name);

       new_matrix.set_header(new_header);
   }

   return new_matrix;
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

   return new_matrix;
}


///Replace a value chosen for another one given to the function
/// @param row_index Row chosen
/// @param column_position Column chosen
/// @param values New values If the vector has more than one element, the function changes the next elements in the row besides the chosen one

template <class T>
void Matrix<T>::insert_row_values(const size_t& row_index, const size_t& column_position, const Vector<T>& values)
{
    const size_t values_size = values.size();

    for(size_t i = 0; i < values_size; i++)
    {
       (*this)(row_index, column_position + i) = values[i];
    }
}


/// Inserts a new column in a given position.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param position Index of new column.
/// @param new_column Vector with the column contents.
/// @param new_name New name of column.

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

   if(!header.empty())
   {
       Vector<string> new_header = get_header();

       new_matrix.set_header(new_header.insert_element(position, new_name));
   }

   return new_matrix;
}


/// Inserts a new column in a given position.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param column_name Name of before column position.
/// @param new_column Vector with the column contents.
/// @param new_name New name of column.

template <class T>
Matrix<T> Matrix<T>::insert_column(const string& column_name, const Vector<T>& new_column, const string& new_name) const
{
    const size_t column_index = get_column_index(column_name);

    return insert_column(column_index, new_column, new_name);
}


/// Inserts a new matrix in a given position.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param position Position of the new matrix.
/// @param other_matrix Matrix to be inserted.

template <class T>
Matrix<T> Matrix<T>::insert_matrix(const size_t& position, const Matrix<T>& other_matrix) const
{

    #ifdef __OPENNN_DEBUG__

        if(position > columns_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: Matrix Template.\n"
                  << "insert_matrix(const size_t& , const Matrix<T>& ) const.\n"
                  << "Position (" << position << ") must be less or equal than number of columns (" << columns_number << ").\n";

           throw logic_error(buffer.str());
        }

       const size_t size = static_cast<size_t>(other_matrix.get_rows_number());

       if(size != rows_number)
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: Matrix Template.\n"
                 << "insert_matrix(const size_t& , const Matrix<T>& ) const.\n"
                 << "Size(" << size << ") must be equal to number of rows(" << rows_number << ").\n";

          throw logic_error(buffer.str());
       }

       #endif

       const size_t columns_other_matrix = other_matrix.get_columns_number();
       const size_t new_columns_number = columns_number + columns_other_matrix;

       Matrix<T> new_matrix(rows_number, new_columns_number);

       for(size_t i = 0; i < rows_number; i++)
       {
           for(size_t j = 0; j < position; j++)
           {
               new_matrix(i,j) = (*this)(i,j);
           }

           for(size_t j = position ; j < position+columns_other_matrix ; j++)
           {
               new_matrix(i,j) = other_matrix(i,j-position);
           }

           for(size_t j = position+columns_other_matrix; j < new_columns_number; j++)
           {
               new_matrix(i,j) = (*this)(i,j-columns_other_matrix);
           }
       }


       if(!header.empty())
       {
           Vector<string> old_header = get_header();
           Vector<string> new_header = other_matrix.get_header();

           new_matrix.set_header(old_header.insert_elements(position,new_header));
       }

       return new_matrix;
}


/// Add a new columns.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param columns_to_add Number of columns to add.

template <class T>
Matrix<T> Matrix<T>::add_columns(const size_t& columns_to_add) const
{
    Matrix<T> new_matrix(rows_number,columns_number+columns_to_add, T());

    Vector<string> new_header(columns_number+columns_to_add, "");

    for(size_t j = 0; j < columns_number; j++)
    {
        new_header[j] = header[j];
    }

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


/// Add a new columns at the beginning of the matrix.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param columns_to_add Number of columns to add.

template <class T>
Matrix<T> Matrix<T>::add_columns_first(const size_t& columns_to_add) const
{
    Matrix<T> new_matrix(rows_number,columns_number+columns_to_add, T());

    Vector<string> new_header(columns_number+columns_to_add, "");

    for(size_t j = 0; j < columns_number; j++)
    {
        new_header[columns_to_add+j] = header[j];
    }

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
void Matrix<T>::split_column(const string& column_name, const Vector<string>& new_columns_name,
                             const char& delimiter, const string& missing_value_label)
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
void Matrix<T>::split_column(const string& column_name, const string& column_1_name, const string& column_2_name,
                             const size_t& size_1, const size_t& size_2)
{
    const size_t column_index = get_column_index(column_name);

    const Vector<T> column = get_column(column_index);

    Vector<T> column_1(rows_number);
    Vector<T> column_2(rows_number);

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


///This method swap two columns
/// @param column_1_index Index of the first column to swap
/// @param column_2_index Index of the second column to swap

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


/// This method swap a columns.
/// @param column_1_name Name of the first column to swap
/// @param column_2_name Name of the second column to swap

template <class T>
void Matrix<T>::swap_columns(const string& column_1_name, const string& column_2_name)
{
    const size_t column_1_index = get_column_index(column_1_name);
    const size_t column_2_index = get_column_index(column_2_name);

    swap_columns(column_1_index, column_2_index);
}


///This method merges two colums given and separates it by the wished separator
///@param column_1_name Name of the header of the first column to merge. This is the one which is going to merge.
///@param column_2_name Name of the header of the second column to merge. This one is not going to suffer changes.
///@param merge_column_name Header of the new column.
///@param separator Character to separate the merged columns.

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


///This method merges two colums given and separates it by the wished separator
///@param column_1_index Index of the first column to merge. This is the one which is going to merge.
///@param column_2_index Index of the second column to merge. This one is not going to suffer changes.
///@param separator Character to separate the merged columns.

template <class T>
void Matrix<T>::merge_columns(const size_t& column_1_index, const size_t& column_2_index, const char& separator)
{
    const Vector<T> column_1 = get_column(column_1_index);
    const Vector<T> column_2 = get_column(column_2_index);

    Vector<T> merged_column(column_1.size());

    for(size_t i = 0; i < column_1.size(); i++)
    {
        merged_column[i] = column_1[i] + separator + column_2[i];
    }

    set_column(column_1_index, merged_column);

    delete_column(column_2_index);
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

        if(columns_1[columns_1_sorted_indices[columns_1_pointer]] > current_index_value) continue;

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
cout << "other_columns_number: " << other_columns_number << endl;
    const size_t columns_1_index = this->get_column_index(columns_1_name);
cout << "columns_1_index: " << columns_1_index << endl;
    const size_t columns_2_index = other_matrix.get_column_index(columns_2_name);
cout << "columns_2_index: " << columns_2_index << endl;
    const Vector<T> columns_1 = this->get_column(columns_1_index);
cout << "columns_1: " << columns_1 << endl;
    const Vector<T> columns_2 = other_matrix.get_column(columns_2_index);
cout << "columns_2: " << columns_2 << endl;

    const size_t columns_1_size = columns_1.size();
    const size_t columns_2_size = columns_2.size();

    const Vector<T> header_1 = this->get_header();
    const Vector<T> header_2 = other_matrix.get_header();

    Vector<T> merged_header = header_1.delete_index(columns_1_index) + left_header_tag;

    merged_header = merged_header.assemble(header_2.delete_index(columns_2_index) + right_header_tag);

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

        if(columns_2[columns_2_sorted_indices[columns_2_pointer]] > current_index_value) continue;

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

        if(columns_2[columns_2_sorted_indices[columns_2_pointer]] > current_index_value) continue;
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

        if(matrix_2_columns_1[columns_2_sorted_indices[columns_2_pointer]] > current_index_1_value) continue;

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

   if(row_index > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> delete_row(const size_t&) const.\n"
             << "Index of row must be less than number of rows.\n"
             << "row index: " << row_index << "rows_number" << rows_number << "\n";

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

   if(!header.empty())
   {
       new_matrix.set_header(header);
   }

   return new_matrix;
}


/// This method removes the rows with given indices.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param row_to_remove Indices of rows to be removed.

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


/// This method removes the rows that contains a given value.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param value Value to be removed.

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

    return get_submatrix_rows(indices);
}


/// This method removes the columns that contains a given value.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param value Value to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_columns_with_value(const T& value) const
{
    Vector<T> column(rows_number);

    size_t count = 0;

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i);

        if(!column.contains(value))
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

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i);

        if(!column.contains(value))
        {
            indices[index] = i;
            index++;
        }
    }

    return get_submatrix_columns(indices);
}


/// This method removes the rows that contains a given value.
/// Note that resizing is here necessary.
/// @param value Value to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_rows_equal_to(const T& value) const
{
    const Vector<size_t> indices = get_columns_indices(value);

    return delete_rows(indices);
}


/// This method removes the first rows.
/// Note that resizing is here necessary.
/// @param number Number of the first rows to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_first_rows(const size_t& number) const
{
    const Vector<size_t> indices(number, 1, rows_number-1);

    return get_submatrix_rows(indices);
}


/// This method removes the first columns.
/// Note that resizing is here necessary.
/// @param number Number of the first columns to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_first_columns(const size_t& number) const
{
    const Vector<size_t> indices(number, 1, columns_number-1);

    return get_submatrix_columns(indices);
}


/// Returns a submatrix with the first rows.
/// @param number Number of the first rows to get.

template <class T>
Matrix<T> Matrix<T>::get_first_rows(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, number-1);

    return get_submatrix_rows(indices);
}


/// Returns a submatrix with the first columns.
/// @param number Number of the first columns to get.

template <class T>
Matrix<T> Matrix<T>::get_first_columns(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, number-1);

    return get_submatrix_columns(indices);
}


/// Returns the first value in a given column.
/// @param column_index Index of column.

template <class T>
T Matrix<T>::get_first(const size_t& column_index) const
{
    return (*this)(0, column_index);
}


/// Returns the first value in a given column.
/// @param column_name Name of column.

template <class T>
T Matrix<T>::get_first(const string& column_name) const
{
    const size_t column_index = get_column_index(column_name);

    return (*this)(0, column_index);
}


/// Returns the last value in a given column.
/// @param column_index Index of column.

template <class T>
T Matrix<T>::get_last(const size_t& column_index) const
{
    return (*this)(rows_number-1, column_index);

}


/// Returns the last value in a given column.
/// @param column_name Name of column.

template <class T>
T Matrix<T>::get_last(const string& column_name) const
{
    const size_t column_index = get_column_index(column_name);

    return (*this)(rows_number-1, column_index);
}


/// This method remove the last rows of the matrix.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param number Number of rows to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_last_rows(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, rows_number-number-1);

    return get_submatrix_rows(indices);
}


/// This method remove the last columns of the matrix.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param number Number of columns to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_last_columns(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, columns_number-number-1);

    return get_submatrix_columns(indices);
}


/// Returns a submatrix with the last columns of the original matrix.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param number Number of columns to get.

template <class T>
Matrix<T> Matrix<T>::get_last_columns(const size_t& number) const
{
    const size_t columns_number = get_columns_number();

    const Vector<size_t> indices(columns_number-number, 1, columns_number-1);

    return get_submatrix_columns(indices);
}


/// Returns a submatrix with the last rows of the original matrix.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param number Number of rows to get.

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

   if(!header.empty())
   {
       const Vector<string> new_header = get_header().delete_index(column_index);

       new_matrix.set_header(new_header);
   }

   return new_matrix;
}


/// This method removes the column with given index.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param column_name Name of column to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_column(const string& column_name) const
{
    const Vector<size_t> indices = header.get_indices_equal_to(column_name);

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


/// This method removes the column with given indices.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param delete_indices Indices vector that contains the column indices to be removed.

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

        if(!header.empty()) new_data.set_header(header.get_subvector(keep_indices));
    }

    return new_data;
}


/// This method removes the column with given names.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param delete_names Names vector that contains the columns names to be removed.

template <class T>
Matrix<T> Matrix<T>::delete_columns(const Vector<string>& delete_names) const
{
    const Vector<size_t> indices = get_columns_indices(delete_names);

    return delete_columns(indices);
}


/// This method removes the column with given substring.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param substring Substring to be removed.

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


/// Returns a indices vector with the index of columns that are constant.

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


/// This method remove the columns that are constant.
/// Note that resizing is here necessary and this method can be very inefficient.

template <class T>
Matrix<T> Matrix<T>::delete_constant_columns() const
{
    return delete_columns(get_constant_columns_indices());
}


/// This method remove the columns that are binary.
/// Note that resizing is here necessary and this method can be very inefficient.

template <class T>
Matrix<T> Matrix<T>::delete_binary_columns() const
{
    const Vector<size_t> binary_columns = get_binary_columns_indices();

    return delete_columns(binary_columns);
}


/// This method remove the columns that are binary.
/// Note that resizing is here necessary and this method can be very inefficient.

template <class T>
Matrix<T> Matrix<T>::delete_binary_columns(const double& minimum_support) const
{
    const Vector<size_t> binary_columns = get_binary_columns_indices();

    const size_t binary_columns_number = binary_columns.size();

    Vector<size_t> columns_to_remove;

    for(size_t i = 0; i < binary_columns_number; i++)
    {
        const double support = get_column(binary_columns[i]).calculate_sum()/static_cast<double>(rows_number);

        if(support < minimum_support) columns_to_remove.push_back(binary_columns[i]);
    }

    return delete_columns(columns_to_remove);
}


/// This method remove the rows that are constant.
/// Note that resizing is here necessary and this method can be very inefficient.

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

   if(rows_number == 0 && other_rows_number == 0)
   {
       return Matrix<T>();
   }
   else if(rows_number == 0)
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

   if(!header.empty())
   {
       assembly.set_header(header);
   }
   else if(other_matrix.get_header() != "")
   {
       assembly.set_header(other_matrix.get_header());
   }

   return assembly;
}


/// Sorts the rows of the matrix in ascending order attending to the values of the column with given index.
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

    return sorted;
}


/// Sorts the rows of the matrix in ascending order attending to the values in string format of the column with given index.
/// It returns a new sorted matrix, it does not change the original one.
/// @param column_index Index of column to sort.

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

    if(!header.empty()) sorted.set_header(header);

    return sorted;
}


/// Sorts the rows of the matrix in descending order attending to the values in string format of the column with given index.
/// It returns a new sorted matrix, it does not change the original one.
/// @param column_index Index of column to sort.

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

    if(!header.empty()) sorted.set_header(header);

    return sorted;
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
        sorted_matrix.set_row(i,this->get_row(rank[i]));
    }

    if(!header.empty()) sorted_matrix.set_header(header);

    return sorted_matrix;
}


template <class T>
Matrix<T> Matrix<T>::sort_columns(const Vector<size_t>& rank) const
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

    for(size_t i = 0; i < columns_number; i++)
    {
        sorted_matrix.set_column(i,this->get_column(rank[i]),this->get_header()[rank[i]]);
    }

    sorted_matrix.set_header(header);

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

      const size_t count = new_header.count_equal_to(this->get_header());

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

        for(size_t j = 0; j < columns_number; j++)
        {
            const string current_column_name = this->get_header()[j];

            if(current_variable == current_column_name)
            {
                sorted_matrix.set_column(i,this->get_column(j),current_variable);

                break;
            }
        }
    }

    return sorted_matrix;
}


/// Sorts the rows of the matrix in descending order attending to the values of the column with given index.
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

    return sorted;
}

/// Assemble two matrices by columns.
/// @param other_matrix matrix to be get_assembled to this matrix.

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

   if(!header.empty() && other_matrix.get_header() != "") assembly.set_header(header.assemble(other_matrix.get_header()));

   return assembly;
}


/// Initializes all the elements of the matrix with a given value.
/// @param value Type value.

template <class T>
void Matrix<T>::initialize(const T& value)
{
    fill(this->begin(),this->end(), value);
}


/// Initializes all the elements of the matrix in a sequential order (0, 1, 2...).

template <class T>
void Matrix<T>::initialize_sequential()
{
  for(size_t i = 0; i < this->size(); i++) {
   (*this)[i] = static_cast<T>(i);
  }
}


/// Replace a value for another one given to the function
/// @param find_what Element is going to be changed
/// @param replace_with New value

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

/// Replace the chosen header for anohter one given to the function
/// @param find_what Header to be changed
/// @param replace_with New header

template <class T>
void Matrix<T>::replace_header(const string& find_what, const string& replace_with)
{
    for(size_t i = 0; i < columns_number; i++)
        if(header[i] == find_what)
            header[i] = replace_with;
}


/// /// Replace the element of a selected row for another one given to the function
/// @param  row_index Row selected to change one element
/// @param find_what Element to be changed
/// @param replace_with New element

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


/// Replace the element of a selected column for another one given to the function
/// @param  column_index Row selected to change one element
/// @param find_what Element to be changed
/// @param replace_with New element

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


/// Replace the element of a selected column for another one given to the function
/// @param  column_name Header of the row selected to change one element
/// @param find_what Element to be changed
/// @param replace_with New element

template <class T>
void Matrix<T>::replace_in_column(const string& column_name, const T& find_what, const T& replace_with)
{
    const size_t column_index = get_column_index(column_name);

    replace_in_column(column_index, find_what, replace_with);
}

///Replace the chosen substring for another one given to the function
/// @param find_what String to be changed
/// @param replace_with New string

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


///Replace the chosen substring for another one given to the function
/// @param column_index Column selected
/// @param find_what String to be changed
/// @param replace_with New string

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


///Replace the chosen substring for another one given to the function
/// @param column_name Header of the column selected
/// @param find_what String to be changed
/// @param replace_with New string

template <class T>
void Matrix<T>::replace_substring(const string& column_name, const string& find_what, const string& replace_with)
{
    const size_t column_index = get_column_index(column_name);

    replace_substring(column_index, find_what, replace_with);
}


///Replace the columns that contains the given substring for another one given to the function
/// @param find_what String to be changed
/// @param replace_with New string

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


///Replace the elements that contains the given substring in a selected row for another one given to the function
/// @param row_index row selected
/// @param find_what String to be changed
/// @param replace_with New string

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


///Replace the elements of a selected column that are equal to a value given for another value given to the function
/// @param column_name Header of the selected column
/// @param find_value Value to be changed
/// @param replace_value New value

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


///Replace the elements of a selected column that are equal to a value given for another value given to the function
/// @param column_index Index of the selected column
/// @param find_value Value to be changed
/// @param replace_value New value

template <class T>
void Matrix<T>::replace_column_equal_to(const size_t& column_index, const T& find_value, const T& replace_value)
{
    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == find_value)
        {
          (*this)(i,column_index) = replace_value;
        }
    }
}


///Replace the elements of a selected column that are not equal to a value given for another value given to the function
/// @param column_name Header of the selected column
/// @param find_value Value to be changed
/// @param replace_value New value

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


///Replace the elements of a selected column that are equal to a value given for another value given to the function
/// @param column_name Header of the selected column
/// @param find_values Values to be changed
/// @param replace_value New value

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


///Replace the elements of a selected column that are equal to a string given for another string given to the function
/// @param column_name Header of the selected column
/// @param find_what String to be changed
/// @param replace_with New string

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


///Returns the number of columns that have a sting given to the function
/// @param column_name Header of the column where the function is going to compare
/// @param find_what String to find in the matrix

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

    return count;
}


///Return the times that a given value is contain in each column of hte matrix

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


///Return true if the selected column has the given value
/// @param colum_index Column selected
/// @param value Value that wants to know if is in the selected column

template <class T>
bool Matrix<T>::has_column_value(const size_t& column_index, const T& value) const
{
    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) == value)
        {
             return true;
        }
    }

    return false;
}


/// Initializes all the elements in the matrix with random values comprised between a minimum and a maximum
/// values.
/// @param minimum Minimum possible value.
/// @param maximum Maximum possible value.

template <class T>
void Matrix<T>::randomize_uniform(const double& minimum, const double& maximum)
{
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

  this->initialize(0);

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


/// Returns the sum of all the elements in the matrix.

template <class T>
T Matrix<T>::calculate_sum() const
{
   T sum = 0;

   for(size_t i = 0; i < this->size(); i++)
   {
        sum += (*this)[i];
   }

   return sum;
}


/// Returns the sum of all the rows in the matrix.

template <class T>
Vector<T> Matrix<T>::calculate_rows_sum() const
{
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


/// Returns the sum of all the columns in the matrix.

template <class T>
Vector<T> Matrix<T>::calculate_columns_sum() const
{
    

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


/// Returns the sum of a selected column in the matrix.
/// @param calculate_column_sum Column chosen to calculate its sum.

template <class T>
T Matrix<T>::calculate_column_sum(const size_t& column_index) const
{
   T column_sum = 0;

   for(size_t i = 0; i < rows_number; i++)
   {
       column_sum += (*this)(i,column_index);
   }

   return(column_sum);
}


/// Returns the sum of a selected row in the matrix.
/// @param calculate_column_sum Row chosen to calculate its sum.

template <class T>
T Matrix<T>::calculate_row_sum(const size_t& row_index) const
{
   T row_sum = 0;

   for(size_t i = 0; i < columns_number; i++)
   {
       row_sum += (*this)(row_index,i);
   }

   return(row_sum);
}


/// Sums the values of a given row with the values of a given vector.
/// The size of the vector must be equal to the number of columns.

template <class T>
void Matrix<T>::sum_row(const size_t& row_index, const Vector<T>& vector)
{
    

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


/// Sums the values of a given column with the values of a given vector.
/// The size of the vector must be equal to the number of rows.

template <class T>
void Matrix<T>::sum_column(const size_t& column_index, const Vector<T>& vector)
{


    #ifdef __OPENNN_DEBUG__

    if(vector.size() != rows_number)
    {
       ostringstream buffer;

       cout << "OpenNN Exception: Matrix Template.\n"
                 << "void sum_column(const size_t&, const Vector<T>&) method.\n"
                 << "Size of vector must be equal to number of rows.\n";

       throw logic_error(buffer.str());
    }

    #endif

    for(size_t i = 0; i < rows_number; i++)
    {
       (*this)(i,column_index) += vector[i];
    }
}


/// Sums the values of all the rows of this matrix with the values of a given vector.
/// The size of the vector must be equal to the number of columns.

template <class T>
Matrix<T> Matrix<T>::sum_rows(const Vector<T>& vector) const
{
    

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


///Multiply the elements of the rows for a vector given to the function
/// Notice that the vector must have the same size that the rows of the matrix
/// @param vector Vector to multiply the matrix

template <class T>
Matrix<T> Matrix<T>::multiply_rows(const Vector<T>& vector) const
{
    

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

    Matrix<T> new_matrix(rows_number, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
           new_matrix(i,j) = (*this)(i,j) * vector[j];
        }
    }

    return new_matrix;
}


///Returns a vector which contains the rows'  multiplication of two matrix.
///@param matrix Matrix to multiply the other one.

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


///Returns the percentage of missing values in each column.

template <class T>
Vector<double> Matrix<T>::calculate_missing_values_percentage() const
{
    Vector<double> missing_values(columns_number);

    Vector<T> column(rows_number);

    for(size_t i = 0; i < columns_number; i++)
    {
       column = get_column(i);

       missing_values[i] = column.count_NAN()*100.0/static_cast<double>(rows_number-1.0);
    }

    return missing_values;
}


/// Returns the matrix indices at which the elements are less than some given value.
/// @param value Value.

template <class T>
Matrix<size_t> Matrix<T>::get_indices_less_than(const T& value) const
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

   return indices;
}


/// Returns the matrix indices at which the elements are greater than some given value.
/// @param value Value.

template <class T>
Matrix<size_t> Matrix<T>::get_indices_greater_than(const T& value) const
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

   return indices;
}


/// Return a matrix the elements of the matrix changes. The last one will be the first.

template <class T>
Matrix<T> Matrix<T>::calculate_reverse_columns() const
{
    Matrix<T> reverse(rows_number,columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
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


/// Divide the rows of the matrix for a vector given to the function
/// Notice that the rows of the matrix and the vector must have the same size.

template <class T>
void Matrix<T>::divide_rows(const Vector<T>& vector)
{
    #ifdef __OPENNN_DEBUG__

     if(rows_number != vector.size())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "divide_rows(const Vector<T>&) method.\n"
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


///Filter a matrix between two values.
/// @param minimum Minimum value allowed to the elements to not  be filter.
/// @param maximun Maximum value allowed to the elements to not  be filter.

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

   return sum;
}


/// Sum matrix+matrix arithmetic operator.
/// @param other_matrix Matrix to be added to this vector.

template <class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& other_matrix) const
{
   

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

   return sum;
}


/// Difference matrix-scalar arithmetic operator.
/// @param scalar Scalar value to be subtracted to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator -(const T& scalar) const
{
   Matrix<T> difference(rows_number, columns_number);

   transform(this->begin(), this->end(), difference.begin(), bind2nd(minus<T>(), scalar));

   return(difference);
}


/// Difference matrix-matrix arithmetic operator.
/// @param other_matrix Matrix to be subtracted to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator -(const Matrix<T>& other_matrix) const
{
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

   transform(this->begin(), this->end(), other_matrix.begin(), difference.begin(), minus<T>());

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

    return product;
}


/// Product matrix*matrix arithmetic operator.
/// @param other_matrix Matrix to be multiplied to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator *(const Matrix<T>& other_matrix) const
{
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

   return product;
}


template <class T>
Matrix<T> Matrix<T>::operator * (const Tensor<T>& tensor) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = tensor.get_dimension(0);
   const size_t other_columns_number = tensor.get_dimension(1);

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator *(const Tensor<T>&) const method.\n"
             << "Sizes of tensor (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<T> product(rows_number, columns_number);

   for(size_t i = 0; i < this->size(); i++)
   {
         product[i] = (*this)[i]*tensor[i];
   }

   return product;
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

    return results;
}


/// Cocient matrix/vector arithmetic operator.
/// @param vector Vector to be divided to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator /(const Vector<T>& vector) const
{
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


/// Returns true if number of rows and columns is zero.

template <class T>
bool Matrix<T>::empty() const
{
   if(rows_number == 0 && columns_number == 0)
   {
      return true;
   }
   else
   {
      return false;
   }
}


/// Returns true if this matrix is square.
/// A square matrix has the same numbers of rows and columns.

template <class T>
bool Matrix<T>::is_square() const
{
   if(rows_number == columns_number)
   {
      return true;
   }
   else
   {
      return false;
   }
}


/// Returns true if this matrix is symmetric.
/// A symmetric matrix is a squared matrix which is equal to its transpose.

template <class T>
bool Matrix<T>::is_symmetric() const
{
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
       return true;
   }
   else
   {
       return false;
   }
}


/// Returns true if this matrix is antysymmetric.
/// A symmetric matrix is a squared matrix which its opposed is equal to its transpose.

template <class T>
bool Matrix<T>::is_antisymmetric() const
{
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
       return true;
   }
   else
   {
       return false;
   }
}


/// Returns true if this matrix is diagonal.
/// A diagonal matrix is which the entries outside the main diagonal are zero.

template <class T>
bool Matrix<T>::is_diagonal() const
{
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
            return false;
         }
      }
   }

   return true;
}


/// Returns true if this matrix is scalar.
/// A scalar matrix is a diagonal matrix whose diagonal elements all contain the same scalar.

template <class T>
bool Matrix<T>::is_scalar() const
{
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
            return false;
         }
         else if(i == j &&(*this)(i,j) != 1)
         {
            return false;
         }
      }
   }

   return true;
}


/// Returns true if this matrix has binary values.

template <class T>
bool Matrix<T>::is_binary() const
{
   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] != 0 && (*this)[i] != 1)
         {
            return false;
         }
   }

   return true;
}


/// Returns true if a column this matrix has binary values.

template <class T>
bool Matrix<T>::is_column_binary(const size_t& j) const
{
    #ifdef __OPENNN_DEBUG__

    if(j >= columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "bool is_column_binary(const size_t) const method method.\n"
              << "Index of column(" << j << ") must be less than number of columns("<<columns_number<<").\n";

       throw logic_error(buffer.str());
    }

    #endif

   for(size_t i = 0; i < rows_number; i++)
   {
         if((*this)(i,j) != 0 &&(*this)(i,j) != 1 &&(*this)(i,j) != static_cast<double>(NAN))
         {
            return false;
         }
   }

   return true;
}


/// Compare two rows of two different matrix. It doen't have to be the same row.
/// @param  row_index Index of the row of the first row.
/// @param  other_matrix The other matrix to compare.
/// @param  other_row_index Indez of the row of the second row.

template <class T>
bool Matrix<T>::compare_rows(const size_t& row_index, const Matrix<T>& other_matrix, const size_t& other_row_index) const
{
    for(size_t j = 0; j < columns_number; j++)
    {
        if((*this)(row_index, j) != other_matrix(other_row_index, j))
        {
            return false;
        }
    }

    return true;
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
    return false;
  }

  const T initial_value = (*this)(0, column_index);

  for(size_t i = 1; i < rows_number; i++)
  {
      if((*this)(i,column_index) != initial_value)
      {
          return false;
      }
  }

  return true;
}


///Return true if all the elements of the matrix are positive. False otherwhise.

template <class T>
bool Matrix<T>::is_positive() const
{

  for(size_t i = 0; i < this->size(); i++)
  {
      if((*this)[i] < 0)
      {
          return false;
      }
  }

  return true;
}


/// Returns a new matrix where a given column has been filtered.
/// @param column_index Index of column.
/// @param minimum Minimum filtering value.
/// @param maximum Maximum filtering value.

template <class T>
Matrix<T> Matrix<T>::filter_column_minimum_maximum(const size_t& column_index, const T& minimum, const T& maximum) const
{
    const Vector<T> column = get_column(column_index);

    const size_t new_rows_number = column.count_between(minimum, maximum);

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) >= minimum && (*this)(i,column_index) <= maximum)
        {
            row = get_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    return new_matrix;
}


template <class T>
Matrix<T> Matrix<T>::filter_column_minimum_maximum(const string& column_name, const T& minimum, const T& maximum) const
{
    const size_t column_index = get_column_index(column_name);

    return filter_column_minimum_maximum(column_index, minimum, maximum);
}


template <class T>
Matrix<T> Matrix<T>::filter_extreme_values(const size_t& column_index, const double& lower_ratio, const double& upper_ratio) const
{
    const size_t lower_index = rows_number*lower_ratio;
    const size_t upper_index = rows_number*upper_ratio;

    const Vector<T> column = get_column(column_index).sort_ascending_values();

    const T lower_value = column[lower_index];
    const T upper_value = column[upper_index];

//    return filter_minimum_maximum(column_index, lower_value, upper_value);
    return Matrix<double>();
}


template <class T>
Matrix<T> Matrix<T>::filter_extreme_values(const string& column_name, const double& lower_ratio, const double& upper_ratio) const
{
    const size_t column_index = get_column_index(column_name);

    return filter_extreme_values(column_index, lower_ratio, upper_ratio);
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

    return new_matrix;
}


/// Returns a new matrix where certain columns has been filtered.

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

    return new_matrix;
}


/// Returns a new matrix where certain columns has been filtered.

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


/// Returns a new matrix where a given column has been filtered.
/// @param column_index Index of column.
/// @param value Vector to filtering value.

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

    return new_matrix;
}


/// Returns a new matrix where a given column has been filtered.
/// @param column_index Index of column.
/// @param value Filtering value.

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

    return new_matrix;
}


/// Returns a new matrix where a given column has been filtered.

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

    return new_matrix;
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const size_t& column_1_index, const Vector<T>& values_1,
                                            const size_t& column_2_index, const T& value_2,
                                            const size_t& column_3_index, const T& value_3,
                                            const size_t& column_4_index, const T& value_4) const
{
    const size_t new_rows_number = count_equal_to(column_1_index, values_1,
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

    return new_matrix;
}


/// Returns a new matrix where a given column has been filtered.

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


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_equal_to(column_index, value));
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& column_name, const Vector<T>& values) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_equal_to(column_index, values));
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_not_equal_to(const string& column_name, const Vector<T>& values) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_not_equal_to(column_index, values));
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_equal_to(const string& name_1, const Vector<T>& values_1,
                                             const string& name_2, const T& value_2) const
{
    const size_t index_1 = get_column_index(name_1);
    const size_t index_2 = get_column_index(name_2);

    return(filter_column_equal_to(index_1, values_1, index_2, value_2));
}


/// Returns a new matrix where a given column has been filtered.

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

    return new_matrix;
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_not_equal_to(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_not_equal_to(column_index, value));
}


/// Returns a new matrix where a given column has been filtered.

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

    return new_matrix;
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_less_than(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_less_than(column_index, value));
}


/// Returns a new matrix where a given column has been filtered.

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

    return new_matrix;
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_greater_than(const string& column_name, const T& value) const
{
    const size_t column_index = get_column_index(column_name);

    return(filter_column_greater_than(column_index, value));
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_less_than_string(const string& name, const double& value) const
{
    const Vector<size_t> indices = get_column(name).string_to_double().get_indices_less_than(value);

    return delete_rows(indices);
}


/// Returns a new matrix where a given column has been filtered.

template <class T>
Matrix<T> Matrix<T>::filter_column_greater_than_string(const string& name, const double& value) const
{
    const Vector<size_t> indices = get_column(name).string_to_double().get_indices_greater_than(value);

    return delete_rows(indices);
}


/// Prints to the screen in the matrix object.

template <class T>
void Matrix<T>::print() const
{
   cout << *this << endl;
}


///Load a csv file.
/// @param file_name Name of the file.
/// @param delim Delimiter used in the file.
/// @param has_columns_names True if the columns have names. False otherwhise.

template <class T>
void Matrix<T>::load_csv(const string& file_name, const char& delim, const bool& has_columns_names, const string& missing_label)
{
    ifstream file(file_name.c_str());

    if(!file.is_open())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "void load_csv(const string&, const char&) method.\n"
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

       size_t new_columns_number = static_cast<size_t>(results.size());

       size_t new_rows_number = 1;

       while(file.good())
       {
          getline(file, line);

//          trim(line);

//          OpenNN::erase(line, '"');

          if(!line.empty())
          {
             new_rows_number++;
          }
       }

       if(has_columns_names)
       {
           new_rows_number--;
           header.set(results);
       }

       set(new_rows_number, new_columns_number);

       if(!has_columns_names)
       {
           header.set();
       }

       // Clear file

       file.clear();
       file.seekg(0, ios::beg);

       if(has_columns_names)
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

           istringstream buffer(line);

           for(size_t j = 0; j < columns_number; j++)
           {
               string token;

               getline(buffer, token, delim);

               if(is_same<T, double>::value)
               {
                   if(token == missing_label) (*this)(i,j) = NAN;
                   else (*this)(i,j) = stod(token);
               }
               else if(is_same<T, float>::value)
               {
                   if(token == missing_label) (*this)(i,j) = NAN;
                   else (*this)(i,j) = stof(token);
               }
               else if(is_same<T, int>::value)
               {
                   if(token == missing_label) (*this)(i,j) = NAN;
                   else (*this)(i,j) = stoi(token);
               }
               else if(is_same<T, long int>::value)
               {
                   if(token == missing_label) (*this)(i,j) = NAN;
                   else (*this)(i,j) = stol(token);
               }
               else if(is_same<T, unsigned long>::value)
               {
                   if(token == missing_label) (*this)(i,j) = NAN;
                   else (*this)(i,j) = stoul(token);
               }
               else if(is_same<T, size_t>::value)
               {
                   if(token == missing_label) (*this)(i,j) = NAN;                   
                   else (*this)(i,j) = stoul(token);
               }
               else
               {
                   ostringstream buffer;

                   buffer << "OpenNN Exception: Matrix template.\n"
                          << "void load_csv(const string&, const char&) method.\n"
                          << "Template could not be recognized \n";

                   throw logic_error(buffer.str());
               }
           }
       }
    }

    file.close();
}


///Load a csv file.
/// @param file_name Name of the file.
/// @param delim Delimiter used in the file.
/// @param has_columns_names True if the columns have names. False otherwhise.

template <class T>
void Matrix<T>::load_csv_string(const string& file_name, const char& delim, const bool& has_columns_names)
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

       if(has_columns_names)
       {
           new_rows_number--;
           header.set(results);
       }

       set(new_rows_number, new_columns_number);

       if(!has_columns_names)
       {
           header.set();
       }

       // Clear file

       file.clear();
       file.seekg(0, ios::beg);

       if(has_columns_names)
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

           istringstream buffer(line);

           for(size_t j = 0; j < columns_number; j++)
           {
               string token;

               getline(buffer, token, delim);

               (*this)(i,j) = token;

           }
       }
    }

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

    size = sizeof(T);

    double value;

    this->set(rows_number, columns_number);

    for(size_t i = 0; i < this->size(); i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

       (*this)[i] = value;
    }

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

   if(!header.empty() && header != "")
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

   return buffer.str();
}


/// Returns a size_t representation of this matrix.

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


/// Returns a float representation of this matrix.

template <class T>
Matrix<float> Matrix<T>::to_float_matrix() const
{
   Matrix<float> new_matrix(rows_number, columns_number);

   const size_t this_size = this->size();

   for(size_t i = 0; i < this_size; i++)
   {
       new_matrix[i] = static_cast<float>((*this)[i]);
   }

   return new_matrix;
}


/// Returns a double representation of this matrix.

template <class T>
Matrix<double> Matrix<T>::to_double_matrix() const
{
   Matrix<double> new_matrix(rows_number, columns_number);

   const size_t this_size = this->size();

   for(size_t i = 0; i < this_size; i++)
   {
       new_matrix[i] = static_cast<double>((*this)[i]);
   }

   return new_matrix;
}


/// Returns a double representation of this bool matrix.

template <class T>
Matrix<double> Matrix<T>::bool_to_double() const
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
          new_matrix[i] = nan("NA");
       }
   }

   new_matrix.set_header(header);

   return new_matrix;
}


/// Returns a double representation of this string matrix.

template <class T>
Matrix<double> Matrix<T>::string_to_double() const
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
          new_matrix[i] = nan("NA");
       }
   }

   new_matrix.set_header(header);

   return new_matrix;
}


/// Returns a size_t representation of this string matrix.

template <class T>
Matrix<size_t> Matrix<T>::string_to_size_t() const
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
          new_matrix[i] = numeric_limits<size_t>::max();
       }
   }

   return new_matrix;
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

   if(!header.empty()) string_matrix.set_header(header);

   return string_matrix;
}


///Set the elements of the matrix to zero.

template <class T>
Matrix<double> Matrix<T>::to_zeros() const
{
    return Matrix<T>(rows_number, columns_number, 0);
}


///Set the elements of the matrix to zero.

template <class T>
Matrix<double> Matrix<T>::to_ones() const
{
    return Matrix<T>(rows_number, columns_number, 1);
}


/// Returns a vector representation of this matrix.
/// The size of the new vector is equal to the number of elements of this matrix.
/// The entries of the new vector are the entries of this matrix ordered by rows.

template <class T>
vector<T> Matrix<T>::to_std_vector() const
{
    const vector<T> std_vector(this->begin(),this->end());

    return(std_vector);
}


/// Returns a vector representation of this matrix.
/// The size of the new vector is equal to the number of elements of this matrix.
/// The entries of the new vector are the entries of this matrix ordered by rows.

template <class T>
Vector<T> Matrix<T>::to_vector() const
{
    const Vector<T> vector(this->begin(),this->end());

    return vector;
}


/// Returns a vector of vectors representation of this matrix.
/// The number of subvectors is equal to the number of columns of this matrix.
/// The size of each subvector is equal to the number of rows of this matrix.

template <class T>
Vector<Vector<T>> Matrix<T>::to_vector_of_vectors() const
{
    Vector<Vector<T>> vector_of_vectors(columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        vector_of_vectors[i] = this->get_column(i);
    }

   return vector_of_vectors;
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

      cout << "Row " << rows_number-2 << ":\n" << row << endl;
   }

   if(rows_number > 2)
   {
      const Vector<T> last_row = get_row(rows_number-1);

      cout << "Row " << rows_number-1 << ":\n" << last_row << endl;
   }
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

   return is;
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

   return os;
}


/// This method re-writes the output operator << for matrices of vectors.
/// @param os Output stream.
/// @param m Output matrix of vectors.

template<class T>
ostream& operator << (ostream& os, const Matrix<Vector<T>>& m)
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

   return os;
}


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

   return os;
}


template <class T>
Matrix<T> Matrix<T>::insert_padding(const size_t& width, const size_t& height) const
{
    Matrix<T> input(*this);

    if(!width && !height) return input;

    Matrix<T> zero_padding_matrix(input);

    for(size_t i = 0; i < width; i++)
    {
        const size_t rows_number = zero_padding_matrix.get_rows_number();

        const Vector<double> zero_vector_rows(rows_number, 0.0);

        zero_padding_matrix = zero_padding_matrix.append_column(zero_vector_rows);

        zero_padding_matrix = zero_padding_matrix.insert_column(0, zero_vector_rows);
    }

    for(size_t i = 0; i < height; i++)
    {
        const size_t columns_number = zero_padding_matrix.get_columns_number();

        const Vector<double> zero_vector_columns(columns_number, 0.0);

        zero_padding_matrix = zero_padding_matrix.append_row(zero_vector_columns);

        zero_padding_matrix = zero_padding_matrix.insert_row(0, zero_vector_columns);
    }

    return zero_padding_matrix;
}


/// This method converts a variable into categorical.
/// Note that the new matrix has a different size.
/// @param column_index Index of the column to be converted.

template <class T>
Matrix<T> Matrix<T>::to_categorical(const size_t& column_index) const
{
    const size_t rows_number = get_rows_number();

    const Vector<T> column = get_column(column_index);

    const Vector<T> categories = column.get_unique_elements();

    const size_t new_columns_number = categories.size();

    categories.sort_ascending_values();

    Matrix<T> new_columns(rows_number,new_columns_number,0);

    for(size_t i = 0 ; i < new_columns_number ; i++)
    {
        for(size_t j = 0 ; j < rows_number ; j++)
        {
            if(column[j] == categories[i]) new_columns(j,i) = 1;
        }
    }

    if (!header.empty())
    {
        Vector<string> new_header = categories.to_string_vector();

        new_columns.set_header(new_header);
    }

    (this)->delete_column(column_index);

    return (this)->insert_matrix(column_index,new_columns);

}


/// Transform the original matrix into a tensor.
/// Returns a tensor with dimenssion (rows of matrix, columns of matrix).

template <class T>
Tensor<T> Matrix<T>::to_tensor() const
{
    Tensor<T> tensor(rows_number, columns_number);

    for(size_t i = 0; i < this->size(); i++)
    {
        tensor[i] = (*this)[i];
    }

    return tensor;
}

///Transform a matrix into a vector of matrix.
/// @param vector_size Size of the vector.
/// @param new_rows_number Number of the rows of the new matrix.
/// @param new_columns_number Number of the columns of the new matrix.

template <class T>
Vector< Matrix<T> > Matrix<T>::to_vector_matrix(const size_t& vector_size, const size_t& new_rows_number, const size_t& new_columns_number) const
{
    Vector<Matrix<T>> output(vector_size, Matrix<T>(new_rows_number, new_columns_number, 0.0));
    Matrix<double> temp = (*this);

    for(size_t i = 0; i < vector_size; i++)
    {
        output[i] = temp.get_first_rows(new_rows_number);

        temp = temp.delete_first_rows(new_rows_number);
    }

    return output;
}

}

// end namespace

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
