/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T R I X   C O N T A I N E R                                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

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

namespace OpenNN
{

/// This template class defines a matrix for general purpose use.
/// This matrix also implements some mathematical methods which can be useful. 

template <class T>
class Matrix : public std::vector<T>
{

public:

    // CONSTRUCTORS

    explicit Matrix(void);

    explicit Matrix(const size_t&, const size_t&);

    explicit Matrix(const size_t&, const size_t&, const T&);

    explicit Matrix(const std::string&);

    Matrix(const Matrix&);

    // DESTRUCTOR

    virtual ~Matrix(void);

    // ASSIGNMENT OPERATORS

    inline Matrix<T>& operator = (const Matrix<T>&);

    // REFERENCE OPERATORS

    inline T& operator () (const size_t&, const size_t&);

    inline const T& operator () (const size_t&, const size_t&) const;

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

    // METHODS

    // Get methods

    const size_t& get_rows_number(void) const;

    const size_t& get_columns_number(void) const;

    // Set methods


    void set(void);

    void set(const size_t&, const size_t&);

    void set(const size_t&, const size_t&, const T&);


    void set(const Matrix<T>&);

    void set(const std::string&);

    void set_identity(const size_t&);

    void set_rows_number(const size_t&);

    void set_columns_number(const size_t&);

    void tuck_in(const size_t&, const size_t&, const Matrix<T>&);

    size_t count_diagonal_elements(void) const;

    size_t count_off_diagonal_elements(void) const;

    Matrix<T> arrange_submatrix(const Vector<size_t>&, const Vector<size_t>&) const;

    Matrix<T> arrange_submatrix_rows(const Vector<size_t>&) const;

    Matrix<T> arrange_submatrix_columns(const Vector<size_t>&) const;

    Vector<T> arrange_row(const size_t&) const;

    Vector<T> arrange_row(const size_t&, const Vector<size_t>&) const;

    Vector<T> arrange_column(const size_t&) const;

    Vector<T> arrange_column(const size_t&, const Vector<size_t>&) const;

    Vector<T> get_diagonal(void) const;

    void set_row(const size_t&, const Vector<T>&);

    void set_row(const size_t&, const T&);

    void set_column(const size_t&, const Vector<T>&);

    void set_column(const size_t&, const T&);

    void set_diagonal(const T&);

    void set_diagonal(const Vector<T>&);

    void initialize_diagonal(const size_t&, const T&);

    void initialize_diagonal(const size_t&, const Vector<T>&);

    Matrix<T> sum_diagonal(const T&) const;

    Matrix<T> sum_diagonal(const Vector<T>&) const;

    void append_row(const Vector<T>&);

    void append_column(const Vector<T>&) ;

    void insert_row(const size_t&, const Vector<T>&);

    void insert_column(const size_t&, const Vector<T>&);

    void subtract_row(const size_t&);

    void subtract_column(const size_t&);

    Matrix<T> assemble_rows(const Matrix<T>&) const;

    Matrix<T> assemble_columns(const Matrix<T>&) const;

    Matrix<T> sort_less_rows(const size_t&) const;
    Matrix<T> sort_greater_rows(const size_t&) const;

    void initialize(const T&);

    void randomize_uniform(const double& = -1.0, const double& = 1.0);
    void randomize_uniform(const Vector<double>&, const Vector<double>&);
    void randomize_uniform(const Matrix<double>&, const Matrix<double>&);

    void randomize_normal(const double& = 0.0, const double& = 1.0);
    void randomize_normal(const Vector<double>&, const Vector<double>&);
    void randomize_normal(const Matrix<double>&, const Matrix<double>&);

    void initialize_identity(void);

    void initialize_diagonal(const T&);

    // Mathematical methods

    T calculate_sum(void) const;

    Vector<T> calculate_rows_sum(void) const;

    void sum_row(const size_t&, const Vector<T>&);

    double calculate_trace(void) const;

    Vector<double> calculate_mean(void) const;
    double calculate_mean(const size_t&) const;

    Vector<double> calculate_mean(const Vector<size_t>&) const;

    Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector<double> calculate_mean_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Vector<double> > calculate_mean_standard_deviation(void) const;

    Vector< Vector<double> > calculate_mean_standard_deviation(const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const;

    T calculate_minimum(void) const;

    T calculate_maximum(void) const;

    Vector< Vector<T> > calculate_minimum_maximum(void) const;

    Vector< Vector<T> > calculate_minimum_maximum(const Vector<size_t>&) const;

    Vector< Vector<T> > calculate_minimum_maximum(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_statistics(void) const;

    Vector< Statistics<T> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const;

    Vector< Statistics<T> > calculate_statistics(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_rows_statistics(const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_rows_statistics_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Statistics<T> > calculate_columns_statistics(const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_columns_statistics_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >) const;

    Vector< Vector<double> > calculate_shape_parameters(void) const;

    Vector< Vector<double> > calculate_shape_parameters_missing_values(const Vector<Vector<size_t> > &) const;

    Vector< Vector<double> > calculate_shape_parameters(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_rows_shape_parameters(const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_rows_shape_parameters_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Vector<double> > calculate_columns_shape_parameters(const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_columns_shape_parameters_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Matrix<double> calculate_covariance_matrix(void) const;

    Vector< Histogram<T> > calculate_histograms(const size_t& = 10) const;

    Vector< Histogram<T> > calculate_histograms_missing_values(const Vector< Vector<size_t> >&, const size_t& = 10) const;

    Matrix<size_t> calculate_less_than_indices(const T&) const;

    Matrix<size_t> calculate_greater_than_indices(const T&) const;

    void scale_mean_standard_deviation(const Vector< Statistics<T> >&);

    Vector< Statistics<T> > scale_mean_standard_deviation(void);

    void scale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_minimum_maximum(const Vector< Statistics<T> >&);

    Vector< Statistics<T> > scale_minimum_maximum(void);

    void scale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_mean_standard_deviation(const Vector< Statistics<T> >&);

    void unscale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_minimum_maximum(const Vector< Statistics<T> >&);

    void unscale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void unscale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&);

    Vector<size_t> calculate_minimal_indices(void) const;

    Vector<size_t> calculate_maximal_indices(void) const;

    Vector< Vector<size_t> > calculate_minimal_maximal_indices(void) const;

    double calculate_sum_squared_error(const Matrix<double>&) const;

    double calculate_sum_squared_error(const Vector<double>&) const;

    Vector<double> calculate_rows_norm(void) const;

    Matrix<T> calculate_absolute_value(void) const;

    Matrix<T> calculate_transpose(void) const;

    T calculate_determinant(void) const;

    Matrix<T> calculate_cofactor(void) const;

    Matrix<T> calculate_inverse(void) const;

    Matrix<T> calculate_LU_inverse(void) const;

    Vector<T> solve_LDLT(const Vector<double>&) const;

    double calculate_distance(const size_t&, const size_t&) const;

    Matrix<T> operator + (const T&) const;

    Matrix<T> operator + (const Vector<T>&) const;

    Matrix<T> operator + (const Matrix<T>&) const;

    Matrix<T> operator - (const T& scalar) const;

    Matrix<T> operator - (const Vector<T>&) const;

    Matrix<T> operator - (const Matrix<T>&) const;

    Matrix<T> operator * (const T&) const;

    Matrix<T> operator * (const Vector<T>&) const;

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

//    void sum_diagonal(const T&);

    Vector<double> dot(const Vector<double>&) const;

    Matrix<double> dot(const Matrix<double>&) const;

    Matrix<double> calculate_eigenvalues(void) const;
    Matrix<double> calculate_eigenvectors(void) const;

    Matrix<T> direct(const Matrix<T>&) const;

    bool empty(void) const;

    bool is_square(void) const;

    bool is_symmetric(void) const;

    bool is_antisymmetric(void) const;

    bool is_diagonal(void) const;

    bool is_scalar(void) const;

    bool is_identity(void) const;

    bool is_binary(void) const;
    bool is_column_binary(const size_t&) const;

    Matrix<T> filter(const size_t&, const T&, const T&) const;

    void convert_time_series(const size_t&);
    void convert_association(void);

    void convert_angular_variables_degrees(const size_t&);

    void convert_angular_variables_radians(const size_t&);

    // Serialization methods

    void print(void) const;

    void load(const std::string&);
    void load_binary(const std::string&);

    void save(const std::string&) const;

    void save_binary(const std::string&) const;

    void save_csv(const std::string&, const Vector<std::string>& = Vector<std::string>()) const;

    void parse(const std::string&);

    std::string to_string(const std::string& = " ") const;

    Matrix<std::string> write_string_matrix(const size_t& = 3) const;

    std::vector<T> to_std_vector(void) const;

    Vector<T> to_vector(void) const;

    void print_preview(void) const;

private:

    /// Number of rows in the matrix.

    size_t rows_number;

    /// Number of columns in the matrix.

    size_t columns_number;
};


// CONSTRUCTORS

/// Default constructor. It creates a matrix with zero rows and zero columns.

template <class T>
Matrix<T>::Matrix(void) : std::vector<T>()
{
   rows_number = 0;
   columns_number = 0;
}


/// Constructor. It creates a matrix with n rows and m columns, containing n*m copies of the default value for Type.
/// @param new_rows_number Number of rows in matrix.
/// @param new_columns_number Number of columns in matrix.

template <class T>
Matrix<T>::Matrix(const size_t& new_rows_number, const size_t& new_columns_number) : std::vector<T>(new_rows_number*new_columns_number)
{
   if(new_rows_number == 0 && new_columns_number == 0)
   {
      rows_number = 0;
      columns_number = 0;
   }
   else if(new_rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Constructor Matrix(const size_t&, const size_t&).\n"
             << "Number of rows must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else if(new_columns_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Constructor Matrix(const size_t&, const size_t&).\n"
             << "Number of columns must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else
   {
      rows_number = new_rows_number;
      columns_number = new_columns_number;
   }
}


/// Constructor. It creates a matrix with n rows and m columns, containing n*m copies of the type value of Type.
/// @param new_rows_number Number of rows in matrix.
/// @param new_columns_number Number of columns in matrix.
/// @param value Value of Type.

template <class T>
Matrix<T>::Matrix(const size_t& new_rows_number, const size_t& new_columns_number, const T& value) : std::vector<T>(new_rows_number*new_columns_number)
{
   if(new_rows_number == 0 && new_columns_number == 0)
   {
      rows_number = 0;
      columns_number = 0;
   }
   else if(new_rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Constructor Matrix(const size_t&, const size_t&, const T&).\n"
             << "Number of rows must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else if(new_columns_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Constructor Matrix(const size_t&, const size_t&, const T&).\n"
             << "Number of columns must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else
   {
      // Set sizes

      rows_number = new_rows_number;
      columns_number = new_columns_number;

      (*this).initialize(value);
   }
}


/// File constructor. It creates a matrix which members are loaded from a data file.
/// @param file_name Name of matrix data file.

template <class T>
Matrix<T>::Matrix(const std::string& file_name) : std::vector<T>()
{
   rows_number = 0;
   columns_number = 0;

   load(file_name);
}


/// Copy constructor. It creates a copy of an existing matrix.
/// @param other_matrix Matrix to be copied.

template <class T>
Matrix<T>::Matrix(const Matrix& other_matrix) : std::vector<T>(other_matrix.begin(), other_matrix.end())
{
   rows_number = other_matrix.rows_number;
   columns_number = other_matrix.columns_number;
}


// DESTRUCTOR

/// Destructor.

template <class T>
Matrix<T>::~Matrix(void)
{
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

    std::copy(other_matrix.begin(), other_matrix.end(), (*this).begin());

    return(*this);
}


// REFERENCE OPERATORS

/// Reference operator.

/// Returns the element (i,j) of the matrix.
/// @param row Index of row.
/// @param column Index of column.

template <class T>
inline T& Matrix<T>::operator () (const size_t& row, const size_t& column)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(row >= rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator () (const size_t&, const size_t&).\n"
             << "Row index (" << row << ") must be less than number of rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }
   else if(column >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator () (const size_t&, const size_t&).\n"
             << "Column index (" << column << ") must be less than number of columns (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Return matrix element

   return((*this)[rows_number*column+row]);
}


/// Reference operator.

/// Returns the element (i,j) of the matrix.
/// @param row Index of row.
/// @param column Index of column.

template <class T>
inline const T& Matrix<T>::operator () (const size_t& row, const size_t& column) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(row >= rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator () (const size_t&, const size_t&).\n"
             << "Row index (" << row << ") must be less than number of rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }
   else if(column >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "operator () (const size_t&, const size_t&).\n"
             << "Column index (" << column << ") must be less than number of columns (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Return matrix element

   return((*this)[rows_number*column+row]);

}


// bool operator == (const Matrix<T>&) const

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


// bool operator == (const T&)

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


// bool operator != (const Matrix<T>&)

/// Not equivalent relational operator between this matrix and other matrix.
/// It produces true if the two matrices have any not equal element, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator != (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator != (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator != (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
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


// bool operator != (const T&) const

/// Not equivalent relational operator between this matrix and a Type value.
/// It produces true if some element of this matrix is not equal to the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator != (const T& value) const
{
   // Control sentence (if debug)

   for(size_t i = 0; i < this->size(); i++)
   {
     if((*this)[i] != value)
     {
        return(true);
     }
   }

   return(false);
}


// bool operator > (const Matrix<T>&) const

/// Greater than relational operator between this matrix and other vector.
/// It produces true if all the elements of this matrix are greater than the corresponding elements of the other matrix,
/// and false otherwise.
/// @param other_matrix matrix to be compared with.

template <class T>
bool Matrix<T>::operator > (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator > (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator > (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
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


// bool operator > (const T&) const

/// Greater than relational operator between this matrix and a Type value.
/// It produces true if all the elements of this matrix are greater than the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator > (const T& value) const
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


// bool operator < (const Matrix<T>&) const

/// Less than relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are less than the corresponding elements of the other matrix,
/// and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator < (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator < (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator < (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
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


// bool operator < (const T&) const

/// Less than relational operator between this matrix and a Type value.
/// It produces true if all the elements of this matrix are less than the Type value, and false otherwise.
/// @param value Type value to be compared with.

template <class T>
bool Matrix<T>::operator < (const T& value) const
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


// bool operator >= (const Matrix<T>&) const

/// Greater than or equal to relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are greater than or equal to the corresponding elements of the
/// other matrix, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator >= (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
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


// bool operator >= (const T&) const

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


// bool operator <= (const Matrix<T>&) const

/// Less than or equal to relational operator between this matrix and other matrix.
/// It produces true if all the elements of this matrix are less than or equal to the corresponding elements of the
/// other matrix, and false otherwise.
/// @param other_matrix Matrix to be compared with.

template <class T>
bool Matrix<T>::operator <= (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }
   else if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool operator >= (const Matrix<T>&) const.\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
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


// bool operator <= (const T&) const

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

// size_t get_rows_number(void) const method

/// Returns the number of rows in the matrix.

template <class T>
const size_t& Matrix<T>::get_rows_number(void) const
{
   return(rows_number);
}


// size_t get_columns_number(void) const method

/// Returns the number of columns in the matrix.

template <class T>
const size_t& Matrix<T>::get_columns_number(void) const
{
   return(columns_number);
}


// void set(void) method

/// This method set the numbers of rows and columns of the matrix to zero.

template <class T>
void Matrix<T>::set(void)
{
   rows_number = 0;
   columns_number = 0;
   this->clear();
}


// void set(const size_t&, const size_t&) method

/// This method set new numbers of rows and columns in the matrix.
/// @param new_rows_number Number of rows.
/// @param new_columns_number Number of columns.

template <class T>
void Matrix<T>::set(const size_t& new_rows_number, const size_t& new_columns_number)
{
   // Control sentence (if debug)

   if(new_rows_number == rows_number && new_columns_number == columns_number)
   {
      // do nothing
   }
   else if(new_rows_number == 0 && new_columns_number == 0)
   {
      set();
   }
   else if(new_rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void set(const size_t&, const size_t&) method.\n"
             << "Number of rows must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else if(new_columns_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void set(const size_t&, const size_t&) method.\n"
             << "Number of columns must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else
   {
      rows_number = new_rows_number;
      columns_number = new_columns_number;

      this->resize(rows_number*columns_number);
   }
}


// void set(const size_t&, const size_t&, const T&) method

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
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void set(const size_t&, const size_t&, const T&) method.\n"
             << "Number of rows must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else if(new_columns_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
            << "void set(const size_t&, const size_t&, const T&) method.\n"
            << "Number of columns must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }
   else
   {
      set(new_rows_number, new_columns_number);
      initialize(value);
   }
}


// void set(const Matrix&) method

/// Sets all the members of the matrix to those of another matrix.
/// @param other_matrix Setting matrix.

template <class T>
void Matrix<T>::set(const Matrix<T>& other_matrix)
{
    rows_number = other_matrix.rows_number;
    columns_number = other_matrix.columns_number;

    this->resize(rows_number*columns_number);

    for(size_t i = 0; i < (size_t)this->size(); i++)
    {
        (*this)[i] = other_matrix[i];
    }
}


// void set(const std::string&) method

/// Sets the members of this object by loading them from a data file.
/// @param file_name Name of data file.

template <class T>
void Matrix<T>::set(const std::string& file_name)
{
   load(file_name);
}


// void set_identity(const size_t&) method

/// Sets the matrix to be squared, with elements equal one in the diagonal and zero outside the diagonal.
/// @param new_size New number of rows and columns in this matrix.

template <class T>
void Matrix<T>::set_identity(const size_t& new_size)
{
   set(new_size, new_size);
   initialize_identity();
}


// void set_rows_number(const size_t&) method

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


// void set_columns_number(const size_t&) method

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


// void tuck_in(const size_t&, const size_t&, const Matrix<T>&) const method

/// Tuck in another matrix starting from a given position.
/// @param row_position Insertion row position.
/// @param column_position Insertion row position.
/// @param other_matrix Matrix to be inserted.

template <class T>
void Matrix<T>::tuck_in(const size_t& row_position, const size_t& column_position, const Matrix<T>& other_matrix)
{
   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_position + other_rows_number > rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void tuck_in(const size_t&, const size_t&, const Matrix<T>&) const method.\n"
             << "Cannot tuck in matrix.\n";

      throw std::logic_error(buffer.str());
   }

   if(column_position + other_columns_number > columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void tuck_in(const size_t&, const size_t&, const Matrix<T>&) const method.\n"
             << "Cannot tuck in matrix.\n";

      throw std::logic_error(buffer.str());
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


// size_t count_diagonal_elements(void) const method

/// Returns the number of elements in the diagonal which are not zero.
/// This method is only defined for square matrices.

template <class T>
size_t Matrix<T>::count_diagonal_elements(void) const
{
    #ifdef __OPENNN_DEBUG__

    if(!is_square())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t count_diagonal_elements(void) const method.\n"
              << "The matrix is not square.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = get_rows_number();

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


// size_t count_off_diagonal_elements(void) const method

/// Returns the number of elements outside the diagonal which are not zero.
/// This method is only defined for square matrices.

template <class T>
size_t Matrix<T>::count_off_diagonal_elements(void) const
{
    #ifdef __OPENNN_DEBUG__

    if(!is_square())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "size_t count_off_diagonal_elements(void) const method.\n"
              << "The matrix is not square.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = get_rows_number();
    const size_t columns_number = get_columns_number();

    size_t count = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(i != j && (*this)(i,j) != 0)
            {
                count++;
            }
        }
    }

    return(count);
}



// Matrix<T> arrange_submatrix(const Vector<size_t>&, const Vector<size_t>&) const method

/// Returns a matrix with the values of given rows and columns from this matrix.
/// @param row_indices Indices of matrix rows.
/// @param column_indices Indices of matrix columns.

template <class T>
Matrix<T> Matrix<T>::arrange_submatrix(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
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


// Matrix<T> arrange_submatrix_rows(const Vector<size_t>&) const method

/// Returns a submatrix with the values of given rows from this matrix.
/// @param row_indices Indices of matrix rows.

template <class T>
Matrix<T> Matrix<T>::arrange_submatrix_rows(const Vector<size_t>& row_indices) const
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

   return(sub_matrix);
}


// Matrix<T> arrange_submatrix_columns(const Vector<size_t>&) const method

/// Returns a submatrix with the values of given columns from this matrix.
/// @param column_indices Indices of matrix columns.

template <class T>
Matrix<T> Matrix<T>::arrange_submatrix_columns(const Vector<size_t>& column_indices) const
{
   const size_t column_indices_size = column_indices.size();

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

   return(sub_matrix);
}


// Vector<T> arrange_row(const size_t&) const method

/// Returns the row i of the matrix.
/// @param i Index of row.

template <class T>
Vector<T> Matrix<T>::arrange_row(const size_t& i) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(i >= rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> arrange_row(const size_t&) const method.\n"
             << "Row index (" << i << ") must be less than number of rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector<T> row(columns_number);

   for(size_t j = 0; j < columns_number; j++)
   {
      row[j] = (*this)(i,j);
   }

   return(row);
}


// Vector<T> arrange_row(const size_t&, const Vector<size_t>&) const method

/// Returns the row i of the matrix, but only the elements specified by given indices.
/// @param row_index Index of row.
/// @param column_indices Column indices of row.

template <class T>
Vector<T> Matrix<T>::arrange_row(const size_t& row_index, const Vector<size_t>& column_indices) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> arrange_row(const size_t&, const Vector<size_t>&) const method.\n"
             << "Row index (" << row_index << ") must be less than number of rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
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


// Vector<T> arrange_column(const size_t&) const method

/// Returns the column j of the matrix.
/// @param j Index of column.

template <class T>
Vector<T> Matrix<T>::arrange_column(const size_t& j) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(j >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> arrange_column(const size_t&) const method.\n"
             << "Column index (" << j << ") must be less than number of columns (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector<T> column(rows_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      column[i] = (*this)(i,j);
   }

   return(column);
}


// Vector<T> arrange_column(const size_t&) const method

/// Returns the column j of the matrix, but only those elements specified by given indices.
/// @param column_index Index of column.
/// @param row_indices Row indices of column.

template <class T>
Vector<T> Matrix<T>::arrange_column(const size_t& column_index, const Vector<size_t>& row_indices) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> arrange_column(const size_t&) const method.\n"
             << "Column index (" << column_index << ") must be less than number of rows (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
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


// Vector<T> get_diagonal(void) const method

/// Returns the diagonal of the matrix.

template <class T>
Vector<T> Matrix<T>::get_diagonal(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> get_diagonal(void) const method.\n"
             << "Matrix must be squared.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector<T> diagonal(rows_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      diagonal[i] = (*this)(i,i);
   }

   return(diagonal);
}


// void set_row(const size_t&, const Vector<T>&) const method

/// Sets new values of a single row in the matrix.
/// @param row_index Index of row.
/// @param new_row New values of single row.

template <class T>
void Matrix<T>::set_row(const size_t& row_index, const Vector<T>& new_row)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const Vector<T>&) method.\n"
             << "Index must be less than number of rows.\n";

      throw std::logic_error(buffer.str());
   }

   const size_t size = new_row.size();

   if(size != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const Vector<T>&) method.\n"
             << "Size (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set new row

   for(size_t i = 0; i < columns_number; i++)
   {
      (*this)(row_index,i) = new_row[i];
   }
}


// void set_row(const size_t&, const T&) method

/// Sets a new value of a single row in the matrix.
/// @param row_index Index of row.
/// @param value New value of single row.

template <class T>
void Matrix<T>::set_row(const size_t& row_index, const T& value)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_row(const size_t&, const T&) method.\n"
             << "Index must be less than number of rows.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set new row

   for(size_t i = 0; i < columns_number; i++)
   {
      (*this)(row_index,i) = value;
   }
}


// void set_column(const size_t&, const Vector<T>&) method

/// Sets new values of a single column in the matrix.
/// @param column_index Index of column.
/// @param new_column New values of single column.

template <class T>
void Matrix<T>::set_column(const size_t& column_index, const Vector<T>& new_column)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_column(const size_t&, const Vector<T>&).\n"
             << "Index (" << column_index << ") must be less than number of columns (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   const size_t size = new_column.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_column(const size_t&, const Vector<T>&).\n"
             << "Size must be equal to number of rows.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,column_index) = new_column[i];
   }
}


// void set_column(const size_t&, const T&) method

/// Sets a new values of a single column in the matrix.
/// @param column_index Index of column.
/// @param value New value of single column.

template <class T>
void Matrix<T>::set_column(const size_t& column_index, const T& value)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_column(const size_t&, const T&).\n"
             << "Index must be less than number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,column_index) = value;
   }
}


// void set_diagonal(const T&) method


/// Sets a new value for the diagonal elements in the matrix.
/// The matrix must be square.
/// @param new_diagonal New value of diagonal.

template <class T>
void Matrix<T>::set_diagonal(const T& new_diagonal)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_diagonal(const T&).\n"
             << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,i) = new_diagonal;
   }
}


// void set_diagonal(const Vector<T>&) method

/// Sets new values of the diagonal in the matrix.
/// The matrix must be square.
/// @param new_diagonal New values of diagonal.

template <class T>
void Matrix<T>::set_diagonal(const Vector<T>& new_diagonal)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_diagonal(const Vector<T>&) const.\n"
             << "Matrix is not square.\n";

      throw std::logic_error(buffer.str());
   }

   const size_t size = new_diagonal.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "set_diagonal(const Vector<T>&) const.\n"
             << "Size of diagonal (" << size << ") is not equal to size of matrix (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set new column

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,i) = new_diagonal[i];
   }
}


// void initialize_diagonal(const size_t&, const T&) method

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


// void initialize_diagonal(const size_t&, const Vector<T>&) method

/// Sets this matrix to be diagonal.
/// A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero.
/// It also initializes the elements on the main diagonal to given values.
/// @param new_size Number of rows and colums in the matrix.
/// @param new_values Values of the elements in the main diagonal.

template <class T>
void Matrix<T>::initialize_diagonal(const size_t& new_size, const Vector<T>& new_values)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t new_values_size = new_values.size();

   if(new_values_size != new_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "initialize_diagonal(const size_t&, const size_t&) const.\n"
             << "Size of new values is not equal to size of square matrix.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   set(new_size, new_size, 0.0);
   set_diagonal(new_values);
}


// Matrix<T> sum_diagonal(const T&) const method

/// This method sums a new value to the diagonal elements in the matrix.
/// The matrix must be square.
/// @param value New summing value.

template <class T>
Matrix<T> Matrix<T>::sum_diagonal(const T& value) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "sum_diagonal(const T&) const.\n"
             << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> sum(*this);

   for(size_t i = 0; i < rows_number; i++)
   {
      sum(i,i) += value;
   }

   return(sum);
}


// Matrix<T> sum_diagonal(const Vector<T>&) const method

/// This method sums new values to the diagonal in the matrix.
/// The matrix must be square.
/// @param new_summing_values Vector of summing values.

template <class T>
Matrix<T> Matrix<T>::sum_diagonal(const Vector<T>& new_summing_values) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "sum_diagonal(const Vector<T>&) const.\n"
             << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
   }

   const size_t size = new_summing_values.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "sum_diagonal(const Vector<T>&) const.\n"
             << "Size must be equal to number of rows.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> sum(*this);

   for(size_t i = 0; i < rows_number; i++)
   {
      sum(i,i) += new_summing_values[i];
   }

   return(sum);
}


// void append_row(const Vector<T>&) const method

/// This method appends a new row to the matrix.
/// The size of the row vector must be equal to the number of columns of the matrix.
/// Note that resizing is necessary here and therefore this method can be very inefficient.
/// @param new_row Row to be appended.

template <class T>
void Matrix<T>::append_row(const Vector<T>& new_row)
{
    #ifdef __OPENNN_DEBUG__

    const size_t size = new_row.size();

    if(size != columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "append_row(const Vector<T>&) const.\n"
              << "Size (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    Matrix<T> copy(*this);

    set(rows_number+1, columns_number);

    for(size_t i = 0; i < copy.get_rows_number(); i++)
    {
        for(size_t j = 0; j < copy.get_columns_number(); j++)
        {
            (*this)(i,j) = copy(i,j);
        }
    }

    set_row(rows_number-1, new_row);
}


// void append_column(const Vector<T>&) const method

/// This method appends a new column to the matrix.
/// The size of the column vector must be equal to the number of rows of the matrix.
/// Note that resizing is necessary here and therefore this method can be very inefficient.
/// @param new_column Column to be appended.

template <class T>
void Matrix<T>::append_column(const Vector<T>& new_column)
{
   #ifdef __OPENNN_DEBUG__

   const size_t size = new_column.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "append_column(const Vector<T>&) const.\n"
             << "Size (" << size << ") must be equal to number of rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   set(rows_number, columns_number+1);

   set_column(columns_number-1, new_column);
}


// void insert_row(const size_t&, const Vector<T>&) const method

/// Inserts a new row in a given position.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param position Index of new row.
/// @param new_row Vector with the row contents.

template <class T>
void Matrix<T>::insert_row(const size_t& position, const Vector<T>& new_row)
{
   #ifdef __OPENNN_DEBUG__

    if(position > rows_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "insert_row(const size_t&, const Vector<T>&) const.\n"
              << "Position must be less or equal than number of rows.\n";

       throw std::logic_error(buffer.str());
    }

   const size_t size = new_row.size();

   if(size != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "insert_row(const size_t&, const Vector<T>&) const.\n"
             << "Size must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
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

   set(new_matrix);
}


// void insert_column(const size_t&, const Vector<T>&) const method

/// Inserts a new column in a given position.
/// Note that this method resizes the matrix, which can be computationally expensive.
/// @param position Index of new column.
/// @param new_column Vector with the column contents.

template <class T>
void Matrix<T>::insert_column(const size_t& position, const Vector<T>& new_column)
{
   #ifdef __OPENNN_DEBUG__

    if(position > columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "insert_column(const size_t&, const Vector<T>&) const.\n"
              << "Position must be less or equal than number of columns.\n";

       throw std::logic_error(buffer.str());
    }

   const size_t size = (size_t)new_column.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "insert_column(const size_t, const Vector<T>&) const.\n"
             << "Size must be equal to number of rows.\n";

      throw std::logic_error(buffer.str());
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

   set(new_matrix);
}


// void subtract_row(const size_t&) const method

/// This method removes the row with given index.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param row_index Index of row to be removed.

template <class T>
void Matrix<T>::subtract_row(const size_t& row_index)
{
   #ifdef __OPENNN_DEBUG__

   if(row_index >= rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "subtract_row(const size_t&) const.\n"
             << "Index of row must be less than number of rows.\n";

      throw std::logic_error(buffer.str());
   }
   else if(rows_number < 2)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "subtract_row(const size_t&) const.\n"
             << "Number of rows must be equal or greater than two.\n";

      throw std::logic_error(buffer.str());
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

   *this = new_matrix;
}


// void subtract_column(const size_t&) method

/// This method removes the column with given index.
/// Note that resizing is here necessary and this method can be very inefficient.
/// @param column_index Index of column to be removed.

template <class T>
void Matrix<T>::subtract_column(const size_t& column_index)
{
   #ifdef __OPENNN_DEBUG__

   if(column_index >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "subtract_column(const size_t&) const.\n"
             << "Index of column must be less than number of columns.\n";

      throw std::logic_error(buffer.str());
   }
   else if(columns_number < 2)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "subtract_column(const size_t&) const.\n"
             << "Number of columns must be equal or greater than two.\n";

      throw std::logic_error(buffer.str());
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

   *this = new_matrix;
}


// Matrix<T> assemble_rows(const Matrix<T>&) const method

/// Assemble two matrices.
/// @param other_matrix matrix to be get_assembled to this matrix.

template <class T>
Matrix<T> Matrix<T>::assemble_rows(const Matrix<T>& other_matrix) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> assemble_rows(const Matrix<T>&) const method.\n"
             << "Number of columns of other matrix (" << other_columns_number << ") must be equal to number of columns of this matrix (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   const size_t other_rows_number = other_matrix.get_rows_number();

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

   return(assembly);
}


// Matrix<T> sort_less_rows(const size_t&) method

/// Sorts the rows of the matrix in descending order attending to the values of the column with given index.
/// It returns a new sorted matrix, it does not change the original one.
/// @param column_index Index of column to sort.

template <class T>
Matrix<T> Matrix<T>::sort_less_rows(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<T> column = arrange_column(column_index);

    const Vector<size_t> indices = column.sort_less_indices();

    size_t index;

// #pragma parallel for private(index, i, j)

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
bool compare(size_t a, size_t b, const Vector<T>& data)
{
    return data[a]<data[b];
}


// Matrix<T> sort_greater_rows(const size_t&) method

/// Sorts the rows of the matrix in ascending order attending to the values of the column with given index.
/// It returns a new sorted matrix, it does not change the original one.
/// @param column_index Index of column to sort.

template <class T>
Matrix<T> Matrix<T>::sort_greater_rows(const size_t& column_index) const
{
    Matrix<T> sorted(rows_number, columns_number);

    const Vector<T> column = arrange_column(column_index);

    //  std::sort(std::begin(indices), std::end(indices), [&data](size_t i1, size_t i2) {return data[i1] > data[i2];});

    const Vector<size_t> indices = column.sort_greater_indices();

//    const Vector<size_t> sorted_indices = column.calculate_maximal_indices(rows_number);

    //Vector<T> sorted_indices(*column);

    //std::sort(sorted_vector.begin(), sorted_vector.end(), std::greater<double>());

//    std::sort(column.begin(), column.end(), [](data const &a, data const &b) { return a.number < b.number; });


    size_t index;

// #pragma parallel for private(index, i, j)

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


// Matrix<T> assemble_columns(const Matrix<T>&) const method

/// Assemble two matrices.
/// @param other_matrix matrix to be get_assemblyd to this matrix.

template <class T>
Matrix<T> Matrix<T>::assemble_columns(const Matrix<T>& other_matrix) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> assemble_columns(const Matrix<T>&) const method.\n"
             << "Number of rows of other matrix (" << other_rows_number << ") must be equal to number of rows of this matrix (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
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

   return(assembly);
}


// void initialize(const T&) method

/// Initializes all the elements of the matrix with a given value.
/// @param value Type value.

template <class T>
void Matrix<T>::initialize(const T& value)
{
    std::fill((*this).begin(), (*this).end(), value);
}


// void randomize_uniform(const double&, const double&) method

/// Initializes all the elements in the matrix with random values comprised between a minimum and a maximum
/// values.
/// @param minimum Minimum possible value.
/// @param maximum Maximum possible value.

template <class T>
void Matrix<T>::randomize_uniform(const double& minimum, const double& maximum)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(minimum > maximum)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const double&, const double&) const method.\n"
             << "Minimum value must be less or equal than maximum value.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
      (*this)[i] = (T)calculate_random_uniform(minimum, maximum);
   }
}


// void randomize_uniform(const Vector<double>&, const Vector<double>&) const method

/// Initializes all the elements in the matrix with random values comprised between a minimum and a maximum
/// values for each element.
/// @param minimums Minimum possible values.
/// @param maximums Maximum possible values.

template <class T>
void Matrix<T>::randomize_uniform(const Vector<double>& minimums, const Vector<double>& maximums)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(minimums.size() != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of minimums must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   if(maximums.size() != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of maximums must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   if(minimums > maximums)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Minimums must be less or equal than maximums.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector<double> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
        column.randomize_uniform(minimums[i], maximums[i]);

        set_column(i, column);
   }
}


// void randomize_uniform(const Matrix<double>&, const Matrix<double>&) const method

/// Initializes all the elements in the matrix with random values comprised between a minimum and a maximum
/// values for each element.
/// @param minimum Minimum possible values.
/// @param maximum Maximum possible values.

template <class T>
void Matrix<T>::randomize_uniform(const Matrix<double>& minimum, const Matrix<double>& maximum)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(minimum > maximum)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_uniform(const Matrix<double>&, const Matrix<double>&) const method.\n"
             << "Minimum values must be less or equal than their respective maximum values.\n";

      throw std::logic_error(buffer.str());
   }

   #endif


   for(size_t i = 0; i < this->size(); i++)
   {
         (*this)[i] = calculate_random_uniform(minimum[i], maximum[i]);
   }
}


// void randomize_normal(const double&, const double&) method

/// Assigns random values to each element in the matrix, taken from a normal distribution with
/// a given mean and a given standard deviation.
/// @param mean Mean value of uniform distribution.
/// @param standard_deviation Standard deviation value of uniform distribution.

template <class T>
void Matrix<T>::randomize_normal(const double& mean, const double& standard_deviation)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(standard_deviation < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const double&, const double&) method.\n"
             << "Standard deviation must be equal or greater than zero.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
      (*this)[i] = calculate_random_normal(mean, standard_deviation);
   }
}


// void randomize_normal(const Vector<double>&, const Vector<double>&) const method

/// Assigns random values to each element in the matrix, taken from a normal distribution with
/// a given mean and a given standard deviation.
/// @param means Means values of uniform distribution.
/// @param standard_deviations Standard deviations values of uniform distribution.

template <class T>
void Matrix<T>::randomize_normal(const Vector<double>& means, const Vector<double>& standard_deviations)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(means.size() != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of means must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   if(standard_deviations.size() != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of standard deviations must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   if(means < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Means must be less or equal than zero.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector<double> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
        column.randomize_normal(means[i], standard_deviations[i]);

        set_column(i, column);
   }
}


// void randomize_normal(const Matrix<double>&, const Matrix<double>&) const method

/// Assigns random values to each element in the vector, taken from normal distributions with
/// given means and standard deviations for each element.
/// @param mean Mean values of uniform distributions.
/// @param standard_deviation Standard deviation values of uniform distributions.

template <class T>
void Matrix<T>::randomize_normal(const Matrix<double>& mean, const Matrix<double>& standard_deviation)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(standard_deviation < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void randomize_normal(const Matrix<double>&, const Matrix<double>&) const method.\n"
             << "Standard deviations must be equal or greater than zero.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < this->size(); i++)
   {
         (*this)[i] = calculate_random_uniform(mean[i], standard_deviation[i]);
   }
}


// void initialize_identity(void) const method

/// Sets the diagonal elements in the matrix with ones and the rest elements with zeros. The matrix
/// must be square.

template <class T>
void Matrix<T>::initialize_identity(void)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      std::cout << "OpenNN Exception: Matrix Template.\n"
                << "initialize_identity(void) const method.\n"
                << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   (*this).initialize(0);

   for(size_t i = 0; i < rows_number; i++)
   {
      (*this)(i,i) = 1;
   }
}


// void initialize_diagonal(const T&) method

/// Sets the diagonal elements in the matrix with a given value and the rest elements with zeros.
/// The matrix must be square.

template <class T>
void Matrix<T>::initialize_diagonal(const T& value)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      std::cout << "OpenNN Exception: Matrix Template.\n"
                << "initialize_diagonal(const T&) const method.\n"
                << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
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


// T calculate_sum(void) const method

/// Returns the sum of all the elements in the matrix.

template <class T>
T Matrix<T>::calculate_sum(void) const
{
   T sum = 0;

   for(size_t i = 0; i < this->size(); i++)
   {
        sum += (*this)[i];
   }

   return(sum);
}


// Vector<T> calculate_rows_sum(void) const method

/// Returns the sum of all the rows in the matrix.

template <class T>
Vector<T> Matrix<T>::calculate_rows_sum(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(this->empty())
    {
       std::ostringstream buffer;

       std::cout << "OpenNN Exception: Matrix Template.\n"
                 << "Vector<T> calculate_rows_sum(void) const method.\n"
                 << "Matrix is empty.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   Vector<T> rows_sum(columns_number, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
            rows_sum[j] += (*this)(i,j);
       }
   }

   return(rows_sum);
}


// void sum_row(const size_t&, const Vector<T>&) method

/// Sums the values of a given row with the values of a given vector.
/// The size of the vector must be equal to the number of columns.

template <class T>
void Matrix<T>::sum_row(const size_t& row_index, const Vector<T>& vector)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(vector.size() != columns_number)
    {
       std::ostringstream buffer;

       std::cout << "OpenNN Exception: Matrix Template.\n"
                 << "void sum_row(const size_t&, const Vector<T>&) method.\n"
                 << "Size of vector must be equal to number of columns.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    for(size_t j = 0; j < columns_number; j++)
    {
        (*this)(row_index,j) += vector[j];
    }
}


// double calculate_trace(void) const method

/// Returns the trace of the matrix, which is defined to be the sum of the main diagonal elements.
/// The matrix must be square.

template <class T>
double Matrix<T>::calculate_trace(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(!is_square())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_trace(void) const method.\n"
             << "Matrix is not square.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   double trace = 0.0;

   for(size_t i = 0; i < rows_number; i++)
   {
      trace += (*this)(i,i);
   }

   return(trace);
}


// Vector<double> calculate_mean(void) const method

/// Returns a vector with the mean values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

template <class T>
Vector<double> Matrix<T>::calculate_mean(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean(void) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw std::logic_error(buffer.str());
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

      mean[j] /= (double)rows_number;
   }

   return(mean);
}


// double calculate_mean(const size_t&) const method

/// Returns a vector with the mean values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

template <class T>
double Matrix<T>::calculate_mean(const size_t& column_index) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_mean(const size_t&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw std::logic_error(buffer.str());
   }

   if(column_index >= columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double calculate_mean(const size_t&) const method.\n"
             << "Index of column must be less than number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Mean

   double mean = 0.0;

    for(size_t i = 0; i < rows_number; i++)
    {
        mean += (*this)(i,column_index);
    }

   mean /= (double)rows_number;

   return(mean);
}


// Vector<double> calculate_mean(const Vector<size_t>&) const method

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

      mean[j] /= (double)rows_number;
   }

   return(mean);
}


// Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method

/// Returns a vector with the mean values of given columns for given rows.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.

template <class T>
Vector<double> Matrix<T>::calculate_mean(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t column_indices_size = column_indices.size();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices (" << row_indices_size << ") is greater than number of rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw std::logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw std::logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw std::logic_error(buffer.str());
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

      mean[j] /= (double)rows_number;
   }

   return(mean);
}


// Vector<double> calculate_mean_missing_values(const Vector< Vector<size_t> >&) const method

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


// Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method

/// Returns a vector with the mean values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param column_indices Indices of columns.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector<double> Matrix<T>::calculate_mean_missing_values(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices, const Vector< Vector<size_t> >& missing_indices) const
{
   const size_t row_indices_size = row_indices.size();
   const size_t column_indices_size = column_indices.size();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Size of row indices (" << row_indices_size << ") is greater than number of rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, Vector< Vector<size_t> >&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw std::logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw std::logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw std::logic_error(buffer.str());
      }
   }

   #endif

   size_t row_index;
   size_t column_index;

   // Mean

   Vector<double> mean(column_indices_size, 0.0);

   Vector<size_t> count(column_indices_size, 0);

   for(size_t j = 0; j < column_indices_size; j++)
   {
      column_index = column_indices[j];

      for(size_t i = 0; i < row_indices_size; i++)
      {
         row_index = row_indices[i];

         if(!missing_indices[j].contains(row_index))
         {
            mean[j] += (*this)(row_index,column_index);
            count[j]++;
         }
      }

      if(count[j] != 0)
      {
          mean[j] /= (double)count[j];
      }
   }

   return(mean);
}


// Vector<double> calculate_mean_standard_deviation(void) const method

/// Returns a vector of vectors with the mean and standard deviation values of all the matrix columns.
/// The size of the vector is two.
/// The size of each element is equal to the number of columns in the matrix.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_mean_standard_deviation(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation(void) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Mean

   Vector<double> mean(columns_number, 0.0);
   Vector<double> standard_deviation(columns_number, 0.0);

   for(size_t i = 0; i < columns_number; i++)
   {
      mean[i] = arrange_column(i).calculate_mean();
      standard_deviation[i] = arrange_column(i).calculate_standard_deviation();

   }

   // Mean and standard deviation of data

   Vector< Vector<double> > mean_standard_deviation(2);

   mean_standard_deviation[0] = mean;
   mean_standard_deviation[1] = standard_deviation;

   return(mean_standard_deviation);
}


// Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&) const method

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

      column = arrange_column(column_index);

      mean[i] = column.calculate_mean();
      standard_deviation[i] = column.calculate_standard_deviation();
   }

   // Mean and standard deviation

   Vector< Vector<double> > mean_standard_deviation(2);

   mean_standard_deviation[0] = mean;
   mean_standard_deviation[1] = standard_deviation;

   return(mean_standard_deviation);
}


// Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method

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

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Row indices size must be equal or less than rows number.\n";

      throw std::logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw std::logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw std::logic_error(buffer.str());
   }

   // Columns check

   if(column_indices_size > columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw std::logic_error(buffer.str());
   }

   for(size_t i = 0; i < column_indices_size; i++)
   {
      if(column_indices[i] >= columns_number)
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw std::logic_error(buffer.str());
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

      mean[j] /= (double)rows_number;
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

   Vector< Vector<double> > mean_standard_deviation(2);

   mean_standard_deviation[0] = mean;
   mean_standard_deviation[1] = standard_deviation;

   return(mean_standard_deviation);
}


// Type calculate_minimum(void) const method

/// Returns the minimum value from all elements in the matrix.

template <class T>
T Matrix<T>::calculate_minimum(void) const
{
   T minimum = (T)1.0e99;

   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] < minimum)
         {
            minimum = (*this)[i];
         }
   }

   return(minimum);
}


// Type calculate_maximum(void) const method

/// Returns the maximum value from all elements in the matrix.

template <class T>
T Matrix<T>::calculate_maximum(void) const
{
    T maximum = (T)-1.0e99;

    for(size_t i = 0; i < this->size(); i++)
    {
          if((*this)[i] > maximum)
          {
             maximum = (*this)[i];
          }
    }

   return(maximum);
}


// Vector< Vector<T> > calculate_minimum_maximum(void) const method

/// Returns a vector of vectors with the minimum and maximum values of all the matrix columns.
/// The size of the vector is two.
/// The size of each element is equal to the number of columns in the matrix.

template <class T>
Vector< Vector<T> > Matrix<T>::calculate_minimum_maximum(void) const
{
   Vector< Vector<T> > minimum_maximum(2);

   Vector<T> minimum(columns_number, (T)1.0e99);
   Vector<T> maximum(columns_number, (T)-1.0e99);

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

   // Minimum and maximum

   minimum_maximum[0] = minimum;
   minimum_maximum[1] = maximum;

   return(minimum_maximum);
}


// Vector<double> calculate_minimum_maximum(const Vector<size_t>&) const method

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
         std::ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template."
                << "Vector<T> calculate_minimum_maximum(const Vector<size_t>&) const method.\n"
                << "Index of column must be less than number of columns.\n";

         throw std::logic_error(buffer.str());
      }
   }

   #endif

   size_t column_index;

   Vector<T> minimum(column_indices_size,  (T)1.0e99);
   Vector<T> maximum(column_indices_size, (T)-1.0e99);

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

   // Minimum and maximum

   Vector< Vector<T> > minimum_maximum(2);

   minimum_maximum[0] = minimum;
   minimum_maximum[1] = maximum;

   return(minimum_maximum);
}


// Vector<double> calculate_minimum_maximum(const Vector<size_t>&, const Vector<size_t>&) const method

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

   Vector<T> minimum(column_indices_size, (T) 1.0e99);
   Vector<T> maximum(column_indices_size, (T)-1.0e99);

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

   // Minimum and maximum

   Vector< Vector<T> > minimum_maximum(2);

   minimum_maximum[0] = minimum;
   minimum_maximum[1] = maximum;

   return(minimum_maximum);
}


// Vector< Statistics<T> > calculate_statistics(void) const method

/// Returns the basic statistics of the columns.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of columns in this matrix.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_statistics(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Statistics<double> > calculate_statistics(void) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector< Statistics<T> > statistics(columns_number);

   Vector<T> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = arrange_column(i);

      statistics[i] = column.calculate_statistics();
   }

   return(statistics);
}


// Vector< Statistics<T> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const method

/// Returns the basic statistics of the columns when the matrix has missing values.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_statistics_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Statistics<double> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw std::logic_error(buffer.str());
   }

   if(missing_indices.size() != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Statistics<double> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Size of missing indices (" << missing_indices.size() << ") must be equal to to number of columns (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector< Statistics<T> > statistics(columns_number);

   Vector<T> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = arrange_column(i);

      statistics[i] = column.calculate_statistics_missing_values(missing_indices[i]);
   }

   return(statistics);
}


// Vector< Statistics<T> > calculate_statistics(const Vector<size_t>&, const Vector<size_t>&) const method

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

    size_t index;

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = column_indices[i];

        column = arrange_column(index, row_indices);

        statistics[i] = column.calculate_statistics();
    }

    return statistics;
}


// Vector< Statistics<T> > calculate_rows_statistics(const Vector<size_t>&) const method

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
        column = arrange_column(i, row_indices);

        statistics[i] = column.calculate_statistics();
    }

    return statistics;
}


// Vector< Statistics<T> > calculate_rows_statistics_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const method

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
        column = arrange_column(i, row_indices);

        statistics[i] = column.calculate_statistics_missing_values(missing_indices[i]);
    }

    return statistics;
}


// Vector< Statistics<T> > calculate_columns_statistics(const Vector<size_t>&) const method

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

        column = arrange_column(index);

        statistics[i] = column.calculate_statistics();
    }

    return statistics;
}


// Vector< Statistics<T> > calculate_columns_statistics_missing_values(const Vector<size_t>&, const Vector<size_t>&) const method

/// Returns the basic statistics of given columns when the matrix has missing values.
/// The format is a vector of statistics structures.
/// The size of that vector is equal to the number of given columns.
/// @param column_indices Indices of the columns for which the statistics are to be computed.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Statistics<T> > Matrix<T>::calculate_columns_statistics_missing_values(const Vector<size_t>& column_indices, const Vector< Vector<size_t> > missing_indices) const
{
    const size_t column_indices_size = column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

#pragma omp parallel for private(index, column)

    for(int i = 0; i < column_indices_size; i++)
    {
        index = column_indices[i];

        column = arrange_column(index);

        statistics[i] = column.calculate_statistics_missing_values(missing_indices[index]);
    }

    return statistics;
}


// Vector < Vector <double> > calculate_shape_parameters(void) const method

/// Returns the asymmetry and the kurtosis of the columns.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of columns in this matrix.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_shape_parameters(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector< Vector<double> > calculate_shape_parameters(void) const method.\n"
              << "Number of rows must be greater than one.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    Vector< Vector<double> > shape_parameters(columns_number);

    Vector<T> column(rows_number);

    for(size_t i = 0; i < columns_number; i++)
    {
       column = arrange_column(i);

       shape_parameters[i] = column.calculate_shape_parameters();
    }

    return(shape_parameters);
}


// Vector< Vector<double> > calculate_shape_parameters_missing_values(const Vector<size_t>&) const

/// Returns the asymmetry and the kurtosis of the columns when the matrix has missing values.
/// The format is a vector of subvectors.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param missing_indices Vector of vectors with the indices of the missing values.

template <class T>
Vector< Vector<double> > Matrix<T>::calculate_shape_parameters_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Vector<double> > calculate_shape_parameters_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw std::logic_error(buffer.str());
   }

   if(missing_indices.size() != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector< Vector<double> > calculate_shape_parameters_missing_values(const Vector< Vector<size_t> >&) const method.\n"
             << "Size of missing indices (" << missing_indices.size() << ") must be equal to to number of columns (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Vector< Vector<double> > shape_parameters(columns_number);

   Vector<T> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = arrange_column(i);

      shape_parameters[i] = column.calculate_shape_parameters_missing_values(missing_indices[i]);
   }

   return(shape_parameters);
}


// Vector< Vector<double> > calculate_shape_parameters(const Vector<size_t>&, const Vector<size_t>&) const method

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

        column = arrange_column(index, row_indices);

        shape_parameters[i] = column.calculate_shape_parameters();
    }

    return shape_parameters;
}


// Vector< Vector<double> > calculate_rows_statistics(const Vector<size_t>&) const method

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
        column = arrange_column(i, row_indices);

        shape_parameters[i] = column.calculate_shape_parameters();
    }

    return shape_parameters;
}


// Vector< Vector<double> > calculate_rows_shape_parameters_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const method

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
        column = arrange_column(i, row_indices);

        shape_parameters[i] = column.calculate_shape_parameters_missing_values(missing_indices[i]);
    }

    return shape_parameters;
}


// Vector< Vector<double> > calculate_columns_shape_parameters(const Vector<size_t>&) const method

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

        column = arrange_column(index);

        shape_parameters[i] = column.calculate_shape_parameters();
    }

    return shape_parameters;
}


// Vector< Vector<double> > calculate_columns_shape_parameters_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const method

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

        column = arrange_column(index);

        shape_parameters[i] = column.calculate_shape_parameters_missing_values(missing_indices[index]);
    }

    return shape_parameters;
}


// Matrix<double> calculate_covariance_matrix(void) const method

/// Retruns the covariance matrix of this matrix.
/// The number of columns and rows of the matrix is equal to the number of columns of this matrix.

template <class T>
Matrix<double> Matrix<T>::calculate_covariance_matrix(void) const
{
    const size_t size = (*this).get_columns_number();

    #ifdef __OPENNN_DEBUG__

    if(size == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "void calculate_covariance_matrix(void) const method.\n"
              << "Number of columns must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    Matrix<double> covariance_matrix(size, size, 0.0);

    Vector<double> first_column;
    Vector<double> second_column;

    for(size_t i = 0; i < size; i++)
    {
        first_column = (*this).arrange_column(i);

        for(size_t j = i; j < size; j++)
        {
            second_column = (*this).arrange_column(j);

            covariance_matrix(i,j) = first_column.calculate_covariance(second_column);
            covariance_matrix(j,i) = covariance_matrix(i,j);
        }
    }

    return covariance_matrix;
}


// Vector<Histogram<T> > calculate_histograms(const size_t&) const method

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
      column = arrange_column(i);

      if (column.is_binary())
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


// Vector<Histogram<T> > calculate_histograms_missing_values(const Vector<size_t>&, const size_t&) const method

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
      column = arrange_column(i);

      histograms[i] = column.calculate_histogram_missing_values(missing_indices[i], bins_number);
   }

   return(histograms);
}


// Matrix<size_t> calculate_less_than_indices(const T&) const method

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


// Matrix<size_t> calculate_greater_than_indices(const T&) const method

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



// void scale_mean_standard_deviation(const Vector< Statistics<T> >&) method

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
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void scale_mean_standard_deviation(const Vector< Statistics<T> >&) const method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].standard_deviation < 1e-99)
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


// Vector< Statistics<T> > scale_mean_standard_deviation(void) method

/// Scales the data using the mean and standard deviation method and
/// the mean and standard deviation values calculated from the matrix.
/// It also returns the statistics of all the columns.

template <class T>
Vector< Statistics<T> > Matrix<T>::scale_mean_standard_deviation(void)
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_mean_standard_deviation(statistics);

    return(statistics);
}


// void scale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) const

/// Scales given rows from the matrix using the mean and standard deviation method.
/// @param statistics Vector of statistics for all the columns.
/// @param row_indices Indices of rows to be scaled.

template <class T>
void Matrix<T>::scale_rows_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to number of columns.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    size_t row_index;

    // Scale columns

    for(size_t j = 0; j < columns_number; j++)
    {
       if(statistics[j].standard_deviation < 1e-99)
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


// void scale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method

/// Scales given columns of this matrix with the mean and standard deviation method.
/// @param statistics Vector of statistics structure containing the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param columns_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

template <class T>
void Matrix<T>::scale_columns_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& columns_indices)
{
   const size_t columns_indices_size = columns_indices.size();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t statistics_size = statistics.size();

   if(statistics_size != columns_indices_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Vector template.\n"
             << "void scale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
             << "Size of statistics must be equal to size of columns indices.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   size_t column_index;

   // Scale columns

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      if(statistics[j].standard_deviation < 1e-99)
      {
         // Do nothing
      }
      else
      {
         column_index = columns_indices[j];

         for(size_t i = 0; i < rows_number; i++)
         {
            (*this)(i,column_index) = ((*this)(i,column_index) - statistics[j].mean)/statistics[j].standard_deviation;
         }
      }
   }
}


// void scale_minimum_maximum(const Vector< Statistics<T> >&) method

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
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void scale_minimum_maximum(const Vector< Statistics<T> >&) method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].maximum - statistics[j].minimum < 1e-99)
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


// Vector< Statistics<T> > scale_minimum_maximum(void) method

/// Scales the data using the minimum and maximum method and
/// the minimum and maximum values calculated from the matrix.
/// It also returns the statistics of all the columns.

template <class T>
Vector< Statistics<T> > Matrix<T>::scale_minimum_maximum(void)
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_minimum_maximum(statistics);

    return(statistics);
}


// void scale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&)

/// Scales given rows from the matrix using the minimum and maximum method.
/// @param statistics Vector of statistics for all the columns.
/// @param row_indices Indices of rows to be scaled.

template <class T>
void Matrix<T>::scale_rows_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& row_indices)
{
    // Control sentence (if debug)

    const size_t row_indices_size = row_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to number of columns.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    // Rescale targets data

    size_t row_index;

    for(size_t j = 0; j < columns_number; j++)
    {
       if(statistics[j].maximum - statistics[j].minimum < 1e-99)
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


// void scale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method

/// Scales given columns of this matrix with the minimum and maximum method.
/// @param statistics Vector of statistics structure containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param column_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

template <class T>
void Matrix<T>::scale_columns_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    // Control sentence (if debug)

    const size_t column_indices_size = column_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != column_indices_size)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Vector template.\n"
              << "void scale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
              << "Size of statistics must be equal to size of columns indices.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Rescale targets data

    for(size_t j = 0; j < column_indices_size; j++)
    {
       column_index = column_indices[j];

       if(statistics[j].maximum - statistics[j].minimum < 1e-99)
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
             (*this)(i,column_index) = 2.0*((*this)(i,column_index) - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum) - 1.0;
          }
       }
    }
}


// void unscale_mean_standard_deviation(const Vector< Statistics<T> >&) method

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
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void unscale_mean_standard_deviation(const Vector< Statistics<T> >&) const method.\n"
             << "Size of statistics vector must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].standard_deviation < 1e-99)
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


// void unscale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method

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
       if(statistics[j].standard_deviation < 1e-99)
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


// void unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method

/// Unscales given columns of this matrix with the mean and standard deviation method.
/// @param statistics Vector of statistics structure containing the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns in the matrix.
/// @param column_indices Vector of indices with the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be scaled.

template <class T>
void Matrix<T>::unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    #ifdef __OPENNN_DEBUG__

    if(statistics.size() != columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "void unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) const method.\n"
              << "Size of statistics vector must be equal to number of columns.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   size_t column_index;

   // Unscale columns

   for(size_t j = 0;  j < column_indices.size(); j++)
   {
      column_index = column_indices[j];

      if(statistics[column_index].standard_deviation < 1e-99)
      {
         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
            (*this)(i,column_index) = (*this)(i,column_index)*statistics[column_index].standard_deviation + statistics[column_index].mean;
         }
      }
   }
}


// void unscale_minimum_maximum(const Vector< Statistics<T> >&) method

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
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template."
             << "void unscale_minimum_maximum(const Vector< Statistics<T> >&) method.\n"
             << "Size of minimum vector must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(statistics[j].maximum - statistics[j].minimum < 1e-99)
      {
         std::cout << "OpenNN Warning: Matrix template.\n"
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


// void unscale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method

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
       if(statistics[j].maximum - statistics[j].minimum < 1e-99)
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


// void unscale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method

/// Unscales given columns in the matrix with the minimum and maximum method.
/// @param statistics Vector of statistics structures containing the minimum and maximum values for the unscaling.
/// The size of that vector must be equal to the number of columns in the matrix.
/// @param column_indices Vector of indices of the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be unscaled.

template <class T>
void Matrix<T>::unscale_columns_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& column_indices)
{
    #ifdef __OPENNN_DEBUG__

    if(statistics.size() != columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "void unscale_columns_minimum_maximum_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) const method.\n"
              << "Size of statistics vector must be equal to number of columns.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Unscale columns

    for(size_t j = 0; j < column_indices.size(); j++)
    {
        column_index = column_indices[j];

       if(statistics[column_index].maximum - statistics[column_index].minimum < 1e-99)
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
             (*this)(i,column_index) = 0.5*((*this)(i,column_index) + 1.0)*(statistics[column_index].maximum-statistics[column_index].minimum)
             + statistics[column_index].minimum;
          }
       }
    }
}


// Vector<size_t> calculate_minimal_indices(void) const method

/// Returns the row and column indices corresponding to the entry with minimum value.

template <class T>
Vector<size_t> Matrix<T>::calculate_minimal_indices(void) const
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


// Vector<size_t> calculate_maximal_indices(void) const method

/// Returns the row and column indices corresponding to the entry with maximum value.

template <class T>
Vector<size_t> Matrix<T>::calculate_maximal_indices(void) const
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


// Vector< Vector<size_t> > calculate_minimal_maximal_indices(void) const method

/// Returns the row and column indices corresponding to the entries with minimum and maximum values.
/// The format is a vector of two vectors.
/// Each subvector also has two elements.
/// The first vector contains the minimal indices, and the second vector contains the maximal indices.

template <class T>
Vector< Vector<size_t> > Matrix<T>::calculate_minimal_maximal_indices(void) const
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


// double calculate_sum_squared_error(const Matrix<double>&) const method

/// Returns the sum squared error between the elements of this matrix and the elements of another matrix.
/// @param other_matrix Other matrix.

template <class T>
double Matrix<T>::calculate_sum_squared_error(const Matrix<double>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
             << "Other number of rows must be equal to this number of rows.\n";

      throw std::logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
             << "Other number of columns must be equal to this number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif


   double sum_squared_error = 0.0;

   for(size_t i = 0; i < rows_number; i++)
   {
        sum_squared_error += ((*this)[i] - other_matrix[i])*((*this)[i] - other_matrix[i]);
   }

   return(sum_squared_error);
}


// double calculate_sum_squared_error(const Vector<double>&) const method

/// This method retuns the sum squared error between the elements of this matrix and the elements of a vector, by columns.
/// The size of the vector must be equal to the number of columns of this matrix.
/// @param vector Vector to be compared to this matrix.

template <class T>
double Matrix<T>::calculate_sum_squared_error(const Vector<double>& vector) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "double calculate_sum_squared_error(const Vector<double>&) const method.\n"
             << "Size must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
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


// Vector<double> calculate_rows_norm(void) const method

/// Returns a vector with the norm of each row.
/// The size of that vector is the number of rows.

template <class T>
Vector<double> Matrix<T>::calculate_rows_norm(void) const
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


// Matrix<T> calculate_absolute_value(void) const method

/// Returns a matrix with the absolute values of this matrix.

template <class T>
Matrix<T> Matrix<T>::calculate_absolute_value(void) const
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



// Matrix<T> calculate_transpose(void) const method

/// Returns the transpose of the matrix.

template <class T>
Matrix<T> Matrix<T>::calculate_transpose(void) const
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


// Type calculate_determinant(void) const method

/// Returns the determinant of a square matrix.

template <class T>
T Matrix<T>::calculate_determinant(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "calculate_determinant(void) const method.\n"
              << "Matrix is empty.\n";

       throw std::logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_determinant(void) const method.\n"
             << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   T determinant = 0;

   if(rows_number == 1)
   {
      determinant = (*this)(0,0);
   }
   else if(rows_number == 2)
   {
      determinant = (*this)(0,0)*(*this)(1,1) - (*this)(1,0)*(*this)(0,1);
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

         //sign = (size_t)(pow(-1.0, row_index+2.0));

         sign = static_cast<int>( (((row_index + 2) % 2) == 0) ? 1 : -1 );

         determinant += sign*(*this)(0,row_index)*sub_matrix.calculate_determinant();
      }
   }

   return(determinant);
}


// Matrix<T> calculate_cofactor(void) const method

/// Returns the cofactor matrix.

template <class T>
Matrix<T> Matrix<T>::calculate_cofactor(void) const
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
            if(ii == i)
            {
               continue;
            }

            size_t j1 = 0;

            for(size_t jj = 0; jj < rows_number; jj++)
            {
               if(jj == j)
               {
                  continue;
               }

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


// Matrix<T> calculate_inverse(void) const method

/// Returns the inverse of a square matrix.
/// An error message is printed if the matrix is singular.

template <class T>
Matrix<T> Matrix<T>::calculate_inverse(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "calculate_inverse(void) const method.\n"
              << "Matrix is empty.\n";

       throw std::logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_inverse(void) const method.\n"
             << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   const double determinant = calculate_determinant();

   if(determinant == 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_inverse(void) const method.\n"
             << "Matrix is singular.\n";

      throw std::logic_error(buffer.str());
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


// Matrix<T> calculate_LU_inverse(void) const method

/// Returns the inverse of a square matrix using the LU decomposition method.
/// The given matrix must be invertible.

template <class T>
Matrix<T> Matrix<T>::calculate_LU_inverse(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "calculate_LU_inverse(void) const method.\n"
              << "Matrix is empty.\n";

       throw std::logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "calculate_LU_inverse(void) const method.\n"
             << "Matrix must be square.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> inverse(rows_number, columns_number);

   const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
   Eigen::Map<Eigen::MatrixXd> inverse_eigen(inverse.data(), rows_number, columns_number);

   inverse_eigen = this_eigen.inverse();

   return(inverse);
}


// Vector<T> solve_LDLT(const Vector<double>&) const method

/// Solve a sisem of the form Ax = b, using the Cholesky decomposition.
/// A is this matrix and must be positive or negative semidefinite.
/// @param b Independent term of the system.

template <class T>
Vector<T> Matrix<T>::solve_LDLT(const Vector<double>& b) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

    if(empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "solve_LLT(const Vector<double>&) const method.\n"
              << "Matrix is empty.\n";

       throw std::logic_error(buffer.str());
    }

    if(rows_number != columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "solve_LLT(const Vector<double>&) const method.\n"
              << "Matrix must be squared.\n";

       throw std::logic_error(buffer.str());
    }

   #endif

   Vector<T> solution(rows_number);

   const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
   const Eigen::Map<Eigen::VectorXd> b_eigen((double*)b.data(),rows_number);
   Eigen::Map<Eigen::VectorXd> solution_eigen(solution.data(), rows_number);

   solution_eigen = this_eigen.ldlt().solve(b_eigen);

   return(solution);
}


// double calculate_distances(const size_t&, const size_t&) const

/// Calculates the distance between two rows in the matix

template <class T>
double Matrix<T>::calculate_distance(const size_t& first_index, const size_t& second_index) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

     if(empty())
     {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: Matrix Template.\n"
               << "calculate_distance(const size_t&, const size_t&) const method.\n"
               << "Matrix is empty.\n";

        throw std::logic_error(buffer.str());
     }

     #endif

    const Vector<T> first_row = arrange_row(first_index);
    const Vector<T> second_row = arrange_row(second_index);

    return(first_row.calculate_distance(second_row));
}


// Matrix<T> operator + (const T&) const method

/// Sum matrix+scalar arithmetic operator.
/// @param scalar Scalar value to be added to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator + (const T& scalar) const
{
   Matrix<T> sum(rows_number, columns_number);

   std::transform(this->begin(), this->end(), sum.begin(), std::bind2nd(std::plus<T>(), scalar));

   return(sum);
}


// Matrix<T> operator + (const Vector<T>&) const method

/// Sum matrix+vector arithmetic operator.
/// @param vector Vector to be added to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator + (const Vector<T>& vector) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator + (const Vector<T>&) const.\n"
             << "Size of vector must be equal to number of rows.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> sum(rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         sum(i,j) = (*this)(i,j) + vector[i];
      }
   }

   return(sum);
}


// Matrix<T> operator + (const Matrix<T>&) const method

/// Sum matrix+matrix arithmetic operator.
/// @param other_matrix Matrix to be added to this vector.

template <class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator + (const Matrix<T>&) const.\n"
             << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> sum(rows_number, columns_number);

   std::transform(this->begin(), this->end(), other_matrix.begin(), sum.begin(), std::plus<T>());

   return(sum);
}


// Matrix<T> operator - (const T&) const method

/// Difference matrix-scalar arithmetic operator.
/// @param scalar Scalar value to be subtracted to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator - (const T& scalar) const
{
   Matrix<T> difference(rows_number, columns_number);

   std::transform( this->begin(), this->end(), difference.begin(), std::bind2nd(std::minus<T>(), scalar));

   return(difference);
}


// Matrix<T> operator - (const Vector<T>&) const method

/// Sum matrix-vector arithmetic operator.
/// @param vector Vector to be subtracted to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator - (const Vector<T>& vector) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator - (const Vector<T>&) const.\n"
             << "Size of vector must be equal to number of rows.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> difference(rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         difference(i,j) = (*this)(i,j) - vector[i];
      }
   }

   return(difference);
}


// Matrix<T> operator - (const Matrix<T>&) const method

/// Difference matrix-matrix arithmetic operator.
/// @param other_matrix Matrix to be subtracted to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator - (const Matrix<T>&) const method.\n"
             << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix ("<< rows_number << "," << columns_number <<").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> difference(rows_number, columns_number);

   std::transform( this->begin(), this->end(), other_matrix.begin(), difference.begin(), std::minus<T>());

   return(difference);
}


// Matrix<T> operator * (const T&) const method

/// Product matrix*scalar arithmetic operator.
/// @param scalar Scalar value to be multiplied to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator * (const T& scalar) const
{
    Matrix<T> product(rows_number, columns_number);

    for(size_t i = 0; i < this->size(); i++)
    {
        product[i] = (*this)[i]*scalar;
    }

    return(product);
}


// Matrix<T> operator * (const Vector<T>&) const  method

/// Row by element matrix*row arithmetic operator.
/// @param vector vector to be multiplied to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator * (const Vector<T>& vector) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator * (const Vector<T>&) const method.\n"
             << "Vector size (" << size << ")  must be equal to number of matrix rows (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> product(rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         product(i,j) = (*this)(i,j)*vector[i];
      }
   }

   return(product);
}


// Matrix<T> operator * (const Matrix<T>&) const  method

/// Product matrix*matrix arithmetic operator.
/// @param other_matrix Matrix to be multiplied to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator * (const Matrix<T>&) const method.\n"
             << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> product(rows_number, columns_number);

   for(size_t i = 0; i < this->size(); i++)
   {
         product[i] = (*this)[i]*other_matrix[i];
   }

   return(product);
}


// Matrix<T> operator / (const T&) const method

/// Cocient Matrix/scalar arithmetic operator.
/// @param scalar Value of scalar.

template <class T>
Matrix<T> Matrix<T>::operator / (const T& scalar) const
{
    Matrix<T> results(rows_number, columns_number);

    for(size_t i = 0; i < results.size(); i++)
    {
        results[i] = (*this)[i]/scalar;
    }

    return(results);
}


// Matrix<T> operator / (const Vector<T>&) const method

/// Cocient matrix/vector arithmetic operator.
/// @param vector Vector to be divided to this matrix.

template <class T>
Matrix<T> Matrix<T>::operator / (const Vector<T>& vector) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator / (const Vector<T>&) const.\n"
             << "Size of vector must be equal to number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> cocient(rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         cocient(i,j) = (*this)(i,j)/vector[j];
      }
   }

   return(cocient);
}


// Matrix<T> operator / (const Matrix<T>&) const  method

/// Cocient matrix/matrix arithmetic operator.
/// @param other_matrix Matrix to be divided to this vector.

template <class T>
Matrix<T> Matrix<T>::operator / (const Matrix<T>& other_matrix) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_rows_number != rows_number || other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> operator / (const Matrix<T>&) const method.\n"
             << "Both matrix sizes must be the same.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> cocient(rows_number, columns_number);

   for(size_t i = 0; i < rows_number; i++)
   {
         cocient[i] = (*this)[i]/other_matrix[i];
   }

   return(cocient);
}


// void operator += (const T&)

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


// void operator += (const Matrix<T>&)

/// Matrix sum and assignment operator.
/// @param other_matrix Matrix to be added to this matrix.

template <class T>
void Matrix<T>::operator += (const Matrix<T>& other_matrix)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator += (const Matrix<T>&).\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator += (const Matrix<T>&).\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         (*this)(i,j) += other_matrix(i,j);
      }
   }
}


// void operator -= (const T&)

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


// void operator -= (const Matrix<T>&)

/// Matrix rest and assignment operator.
/// @param other_matrix Matrix to be subtracted to this matrix.

template <class T>
void Matrix<T>::operator -= (const Matrix<T>& other_matrix)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator -= (const Matrix<T>&).\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator -= (const Matrix<T>&).\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         (*this)(i,j) -= other_matrix(i,j);
      }
   }
}


// void operator *= (const T&)

/// Scalar product and assignment operator.
/// @param value Scalar value to be multiplied to this matrix.

template <class T>
void Matrix<T>::operator *= (const T& value)
{
   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         (*this)(i,j) *= value;
      }
   }
}


// void operator *= (const Matrix<T>&)

/// Matrix product and assignment operator.
/// @param other_matrix Matrix to be multiplied to this matrix.

template <class T>
void Matrix<T>::operator *= (const Matrix<T>& other_matrix)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   const size_t rows_number = get_rows_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator *= (const Matrix<T>&).\n"
             << "The number of rows in the other matrix (" << other_rows_number << ")"
             << " is not equal to the number of rows in this matrix (" << rows_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         (*this)(i,j) *= other_matrix(i,j);
      }
   }
}


// void operator /= (const T&)

/// Scalar division and assignment operator.
/// @param value Scalar value to be divided to this matrix.

template <class T>
void Matrix<T>::operator /= (const T& value)
{
   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         (*this)(i,j) /= value;
      }
   }
}


// void operator /= (const Matrix<T>&)

/// Matrix division and assignment operator.
/// @param other_matrix Matrix to be divided to this matrix.

template <class T>
void Matrix<T>::operator /= (const Matrix<T>& other_matrix)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t other_rows_number = other_matrix.get_rows_number();

   if(other_rows_number != rows_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator /= (const Matrix<T>&).\n"
             << "Both numbers of rows must be the same.\n";

      throw std::logic_error(buffer.str());
   }

   const size_t other_columns_number = other_matrix.get_columns_number();

   if(other_columns_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "void operator /= (const Matrix<T>&).\n"
             << "Both numbers of columns must be the same.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         (*this)(i,j) /= other_matrix(i,j);
      }
   }
}


// void sum_diagonal(const T&) method
/*
template <class T>
void Matrix<T>::sum_diagonal(const T& value)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!is_square())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "void sum_diagonal(const T&) method.\n"
              << "Matrix must be squared.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    for(size_t i = 0; i < rows_number; i++)
    {
        (*this)(i,i) += value;
    }
}
*/

// Vector<double> dot(const Vector<double>&) const method

/// Returns the dot product of this matrix with a vector.
/// The size of the vector must be equal to the number of columns of the matrix.
/// @param vector Vector to be multiplied to this matrix.

template <class T>
Vector<double> Matrix<T>::dot(const Vector<double>& vector) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = vector.size();

   if(size != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Vector<T> dot(const Vector<T>&) const method.\n"
             << "Vector size must be equal to matrix number of columns.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Calculate matrix-vector poduct

   Vector<double> product(rows_number);

//   for(size_t i = 0; i < rows_number; i++)
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


// Matrix<double> dot(const Matrix<double>&) const method

/// Returns the dot product of this matrix with another matrix.
/// @param other_matrix Matrix to be multiplied to this matrix.

template <class T>
Matrix<double> Matrix<T>::dot(const Matrix<double>& other_matrix) const
{
   const size_t other_columns_number = other_matrix.get_columns_number();
   const size_t other_rows_number = other_matrix.get_rows_number();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(other_rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> dot(const Matrix<T>&) const method.\n"
             << "The number of rows of the other matrix (" << other_rows_number << ") must be equal to the number of columns of this matrix (" << columns_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   Matrix<T> product(rows_number, other_columns_number);

//   for(size_t i = 0; i < rows_number; i++) {
//     for(size_t j = 0; j < other_columns_number; j++) {
//       for(size_t k = 0; k < columns_number; k++) {
//         product(i,j) += (*this)(i,k)*other_matrix(k,j);
//       }
//     }
//   }

   const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
   const Eigen::Map<Eigen::MatrixXd> other_eigen((double*)other_matrix.data(), other_rows_number, other_columns_number);
   Eigen::Map<Eigen::MatrixXd> product_eigen(product.data(), rows_number, other_columns_number);

   product_eigen = this_eigen*other_eigen;

   return(product);
}


// Matrix<double> calculate_eigenvalues(void) const method

/// Calculates the eigen values of this matrix, which must be squared.
/// Returns a matrix with only one column and rows the same as this matrix with the eigenvalues.

template<class T>
Matrix<double> Matrix<T>::calculate_eigenvalues(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values(void) const method.\n"
              << "Number of columns must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_rows_number() == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values(void) const method.\n"
              << "Number of rows must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() != (*this).get_rows_number())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values(void) const method.\n"
              << "The matrix must be squared.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    Matrix<T> eigenvalues(rows_number, 1);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> matrix_eigen(this_eigen, Eigen::EigenvaluesOnly);
    Eigen::Map<Eigen::MatrixXd> eigenvalues_eigen(eigenvalues.data(), rows_number, 1);

    eigenvalues_eigen = matrix_eigen.eigenvalues();

    return(eigenvalues);
}


// Matrix<double> calculate_eigenvectors(void) const method

/// Calculates the eigenvectors of this matrix, which must be squared.
/// Returns a matrix whose columns are the eigenvectors.

template<class T>
Matrix<double> Matrix<T>::calculate_eigenvectors(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values(void) const method.\n"
              << "Number of columns must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_rows_number() == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values(void) const method.\n"
              << "Number of rows must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() != (*this).get_rows_number())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "Matrix<T> calculate_eigen_values(void) const method.\n"
              << "The matrix must be squared.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    Matrix<T> eigenvectors(rows_number, rows_number);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> matrix_eigen(this_eigen, Eigen::ComputeEigenvectors);
    Eigen::Map<Eigen::MatrixXd> eigenvectors_eigen(eigenvectors.data(), rows_number, rows_number);

    eigenvectors_eigen = matrix_eigen.eigenvectors();

    return(eigenvectors);
}


// Matrix<T> direct(const Matrix<T>&) const method

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


// bool empty(void) const method

/// Returns true if number of rows and columns is zero.

template <class T>
bool Matrix<T>::empty(void) const
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


// bool is_square(void) const method

/// Returns true if this matrix is square.
/// A square matrix has the same numbers of rows and columns.

template <class T>
bool Matrix<T>::is_square(void) const
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


// bool is_symmetric(void) const method

/// Returns true if this matrix is symmetric.
/// A symmetric matrix is a squared matrix which is equal to its transpose.

template <class T>
bool Matrix<T>::is_symmetric(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_symmetric(void) const method.\n"
             << "Matrix must be squared.\n";

      throw std::logic_error(buffer.str());
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


// bool is_antisymmetric(void) const method

/// Returns true if this matrix is antysymmetric.
/// A symmetric matrix is a squared matrix which its opposed is equal to its transpose.

template <class T>
bool Matrix<T>::is_antisymmetric(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_antisymmetric(void) const method.\n"
             << "Matrix must be squared.\n";

      throw std::logic_error(buffer.str());
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


// bool is_diagonal(void) const method

/// Returns true if this matrix is diagonal.
/// A diagonal matrix is which the entries outside the main diagonal are zero.

template <class T>
bool Matrix<T>::is_diagonal(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_diagonal(void) const method.\n"
             << "Matrix must be squared.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(i != j && (*this)(i,j) != 0)
         {
            return(false);
         }
      }
   }

   return(true);
}


// bool is_scalar(void) const method

/// Returns true if this matrix is scalar.
/// A scalar matrix is a diagonal matrix whose diagonal elements all contain the same scalar.

template <class T>
bool Matrix<T>::is_scalar(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_scalar(void) const method.\n"
             << "Matrix must be squared.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // @todo

   return(false);
}


/// Returns true if this matrix is the identity.
/// The identity matrix or unit matrix is a square matrix with ones on the main diagonal and zeros elsewhere.

template <class T>
bool Matrix<T>::is_identity(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(rows_number != columns_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "bool is_unity(void) const method.\n"
             << "Matrix must be squared.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(i != j && (*this)(i,j) != 0)
         {
            return(false);
         }
         else if(i == j && (*this)(i,j) != 1)
         {
            return(false);
         }
      }
   }

   return(true);
}


// bool is_binary(void) const method

/// Returns true if this matrix has binary values.

template <class T>
bool Matrix<T>::is_binary(void) const
{
   for(size_t i = 0; i < this->size(); i++)
   {
         if((*this)[i] != 0 && (*this)[i] != 1)
         {
            return(false);
         }
   }

   return(true);
}


// bool is_column_binary(const size_t) const method

/// Returns true if a column this matrix has binary values.

template <class T>
bool Matrix<T>::is_column_binary(const size_t& j) const
{

    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t columns_number = get_columns_number();

    if(j >= columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "bool is_column_binary(const size_t) const method method.\n"
              << "Index of column (" << j << ") must be less than number of columns.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = get_rows_number();

   for(size_t i = 0; i < rows_number; i++)
   {
         if((*this)(i,j) != 0 && (*this)(i,j) != 1)
         {
            return(false);
         }
   }

   return(true);
}


// Matrix<T> filter(const size_t&, const T&, const T&) const method

/// Returns a new matrix where a given column has been filtered.
/// @param column_index Index of column.
/// @param minimum Minimum filtering value.
/// @param maximum Maximum filtering value.

template <class T>
Matrix<T> Matrix<T>::filter(const size_t& column_index, const T& minimum, const T& maximum) const
{
    const Vector<T> column = arrange_column(column_index);

    const size_t new_rows_number = rows_number
    - column.count_less_than(minimum) - column.count_greater_than(maximum);

    Matrix<T> new_matrix(new_rows_number, columns_number);

    size_t row_index = 0;

    Vector<T> row(columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        if((*this)(i,column_index) >= minimum && (*this)(i,column_index) <= maximum)
        {
            row = arrange_row(i);

            new_matrix.set_row(row_index, row);

            row_index++;
        }
    }

    return(new_matrix);
}

// void convert_time_series(const size_t&) method

/// Arranges a time series data matrix in a proper format for forecasting.
/// Note that this method sets new numbers of rows and columns in the matrix.
/// @param lags_number Number of lags for the prediction.
/// @todo

template <class T>
void Matrix<T>::convert_time_series(const size_t& lags_number)
{
    const size_t new_rows_number = rows_number - lags_number;
    const size_t new_columns_number = columns_number*(1 + lags_number);

    Matrix<T> new_matrix(new_rows_number, new_columns_number);

    Vector<T> row(rows_number);

    for(size_t i = 0; i < new_rows_number; i++)
    {
        row = arrange_row(i);

        for(size_t j = 1; j <= lags_number; j++)
        {
            row = row.assemble(arrange_row(i+j));
        }

        new_matrix.set_row(i, row);
    }

    set(new_matrix);
}



// void convert_association(void) method

/// Arranges the matrix in a proper format for association.
/// Note that this method sets new numbers of columns in the matrix.

template <class T>
void Matrix<T>::convert_association(void)
{
    Matrix<T> copy(*this);

    set(copy.assemble_columns(copy));
}


// void convert_angular_variables_degrees(const size_t&) method

/// Converts a given column, representing angles in degrees, to two different columns with the sinus and the cosinus of the corresponding angles.
/// Note that this method sets a new number of columns in the matrix.
/// @param column_index Index of column to be converted.

template <class T>
void Matrix<T>::convert_angular_variables_degrees(const size_t& column_index)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "void convert_angular_variables_degrees(const size_t&) method.\n"
              << "Index of column (" << column_index << ") must be less than number of columns.\n";

       throw std::logic_error(buffer.str());
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
            sin_angle[i] = (T)-99.9;
            cos_angle[i] = (T)-99.9;
        }
    }

    set_column(column_index, sin_angle);
    insert_column(column_index+1, cos_angle);
}


// void convert_angular_variables_radians(const size_t&) method

/// Converts a given column, representing angles in radians, to two different columns with the sinus and the cosinus of the corresponding angles.
/// Note that this method sets a new number of columns in the matrix.
/// @param column_index Index of column to be converted.

template <class T>
void Matrix<T>::convert_angular_variables_radians(const size_t& column_index)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix Template.\n"
              << "void convert_angular_variables_radians(const size_t&) method.\n"
              << "Index of column (" << column_index << ") must be less than number of columns.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    Vector<T> sin_angle(rows_number);
    Vector<T> cos_angle(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        sin_angle[i] = sin((*this)(i,column_index));
        cos_angle[i] = cos((*this)(i,column_index));
    }

    set_column(column_index, sin_angle);
    insert_column(column_index+1, cos_angle);
}


// void print(void) const method

/// Prints to the screen in the matrix object.

template <class T>
void Matrix<T>::print(void) const
{
   std::cout << *this;
}


// void load(const std::string&) method

/// Loads the numbers of rows and columns and the values of the matrix from a data file.
/// @param file_name File name.

template <class T>
void Matrix<T>::load(const std::string& file_name)
{
   std::ifstream file(file_name.c_str());

   if(!file.is_open())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot open matrix data file: " << file_name << "\n";

      throw std::logic_error(buffer.str());
   }

   if(file.peek() == std::ifstream::traits_type::eof())
   {
      //std::ostringstream buffer;

      //buffer << "OpenNN Exception: Matrix template.\n"
      //       << "void load(const std::string&) method.\n"
      //       << "Data file " << file_name << " is empty.\n";

      //throw std::logic_error(buffer.str());

      this->set();

      return;
   }

   //file.is

   // Set matrix sizes

   std::string line;

   std::getline(file, line);

   if(line.empty())
   {
      set();
   }
   else
   {
      std::istringstream buffer(line);

      std::istream_iterator<std::string> it(buffer);
      std::istream_iterator<std::string> end;

      const std::vector<std::string> results(it, end);

      const size_t new_columns_number = (size_t)results.size();

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
      file.seekg(0, std::ios::beg);

      for(size_t i = 0; i < rows_number; i++)
      {
          for(size_t j = 0; j < columns_number; j++)
          {
                file >> (*this)(i,j);
          }
      }
   }

   // Close file

   file.close();
}


// void load_binary(const std::string&) method

/// Loads the numbers of rows and columns and the values of the matrix from a binary file.
/// @param file_name Name of binary file.

template <class T>
void Matrix<T>::load_binary(const std::string& file_name)
{
    std::ifstream file;

    file.open(file_name.c_str(), std::ios::binary);

    if(!file.is_open())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: Matrix template.\n"
               << "void load_binary(const std::string&) method.\n"
               << "Cannot open binary file: " << file_name << "\n";

        throw std::logic_error(buffer.str());
    }

    std::streamsize size = sizeof(size_t);

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


// void save(const std::string&) const method

/// Saves the values of the matrix to a data file separated by spaces.
/// @param file_name File name.

template <class T>
void Matrix<T>::save(const std::string& file_name) const
{
   std::ofstream file(file_name.c_str());

   if(!file.is_open())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << std::endl
             << "void save(const std::string) method." << std::endl
             << "Cannot open matrix data file." << std::endl;

      throw std::logic_error(buffer.str());
   }

   // Write file

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         file << (*this)(i,j) << " ";
      }

      file << std::endl;
   }

   // Close file

   file.close();
}


// void save_binary(const std::string&) const method

/// Saves the values of the matrix to a binary file.
/// @param file_name File name.

template <class T>
void Matrix<T>::save_binary(const std::string& file_name) const
{
   std::ofstream file(file_name.c_str(), std::ios::binary);

   if(!file.is_open())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << std::endl
             << "void save(const std::string) method." << std::endl
             << "Cannot open matrix binary file." << std::endl;

      throw std::logic_error(buffer.str());
   }

   // Write data

   std::streamsize size = sizeof(size_t);

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



// void save_csv(const std::string&, const Vector<std::string>&) const method

/// Saves the values of the matrix to a data file separated by commas.
/// @param file_name File name.
/// @param column_names Names of the columns.

template <class T>
void Matrix<T>::save_csv(const std::string& file_name, const Vector<std::string>& column_names) const
{
   std::ofstream file(file_name.c_str());

   if(!file.is_open())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << std::endl
             << "void save_csv(const std::string, const Vector<std::string>&) method." << std::endl
             << "Cannot open matrix data file." << std::endl;

      throw std::logic_error(buffer.str());
   }

   if(column_names.size() != 0 && column_names.size() != columns_number)
   {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template." << std::endl
              << "void save_csv(const std::string, const Vector<std::string>&) method." << std::endl
              << "Column names must have size 0 or " << columns_number << "." << std::endl;

       throw std::logic_error(buffer.str());
   }

   // Write file

   if(column_names.size() == 0)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
           file << "c" << j+1;

           if(j != columns_number-1)
           {
               file << ",";
           }
       }
   }
   else
   {
       for(size_t j = 0; j < columns_number; j++)
       {
           file << column_names[j];

           if(j != columns_number-1)
           {
               file << ",";
           }
       }
   }

   file << std::endl;

   file.precision(20);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         file << (*this)(i,j);

         if(j != columns_number-1)
         {
             file << ",";
         }
      }

      file << std::endl;
   }

   // Close file

   file.close();
}


// void parse(const std::string&) method

/// This method takes a string representation of a matrix and sets this matrix
/// from that data.
/// @param str String to be parsed.

template <class T>
void Matrix<T>::parse(const std::string& str)
{
   if(str.empty())
   {
       set();
   }
   else
   {
        // Set matrix sizes

        std::istringstream str_buffer(str);

        std::string line;

        std::getline(str_buffer, line);

        std::istringstream line_buffer(line);

        std::istream_iterator<std::string> it(line_buffer);
        std::istream_iterator<std::string> end;

        const std::vector<std::string> results(it, end);

        const size_t new_columns_number = (size_t)results.size();

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
      str_buffer.seekg(0, std::ios::beg);

      for(size_t i = 0; i < rows_number; i++)
      {
         for(size_t j = 0; j < columns_number; j++)
         {
            str_buffer >> (*this)(i,j);
         }
      }
   }
}


// std::string to_string(const std::string&) const method

/// Returns a string representation of this matrix.
/// The elements are separated by spaces.
/// The rows are separated by the character "\n".

template <class T>
std::string Matrix<T>::to_string(const std::string& separator) const
{
   std::ostringstream buffer;

   if(rows_number > 0 && columns_number > 0)
   {
       buffer << arrange_row(0).to_string(separator);

       for(size_t i = 1; i < rows_number; i++)
       {
           buffer << "\n"
                  << arrange_row(i).to_string(separator);
       }
   }

   return(buffer.str());
}


// Matrix<std::string> write_string_matrix(const size_t&) const

/// Returns a new matrix in which each entry has been converted to a string.

template <class T>
Matrix<std::string> Matrix<T>::write_string_matrix(const size_t& precision) const
{
   Matrix<std::string> string_matrix(rows_number, columns_number);

   std::ostringstream buffer;

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         buffer.str("");
         buffer << std::setprecision(precision) << (*this)(i,j);

         string_matrix(i,j) = buffer.str();
      }
   }

   return(string_matrix);
}


// vector<T> to_std_vector(void) const

/// Returns a std::vector representation of this matrix.
/// The size of the new vector is equal to the number of elements of this matrix.
/// The entries of the new vector are the entries of this matrix ordered by rows.

template <class T>
std::vector<T> Matrix<T>::to_std_vector(void) const
{
    const std::vector<T> std_vector((*this).begin(), (*this).end());

    return(std_vector);
}


// Vector<T> to_vector(void) const

/// Returns a vector representation of this matrix.
/// The size of the new vector is equal to the number of elements of this matrix.
/// The entries of the new vector are the entries of this matrix ordered by rows.

template <class T>
Vector<T> Matrix<T>::to_vector(void) const
{
    Vector<T> vector(rows_number*columns_number);

    for(size_t i = 0; i < rows_number*columns_number; i++)
    {
        vector[i] = (*this)[i];
    }

   return(vector);
}


// void print_preview(void) const method

/// Prints to the sceen a preview of the matrix,
/// i.e., the first, second and last rows

template <class T>
void Matrix<T>::print_preview(void) const
{
    std::cout << "Rows number: " << rows_number << std::endl
              << "Columns number: " << columns_number << std::endl;

       if(rows_number > 0)
       {
          const Vector<T> first_row = arrange_row(0);

          std::cout << "Row 0:\n" << first_row << std::endl;
       }

       if(rows_number > 1)
       {
          const Vector<T> second_row = arrange_row(1);

          std::cout << "Row 1:\n" << second_row << std::endl;
       }

       if(rows_number > 2)
       {
          const Vector<T> last_row = arrange_row(rows_number-1);

          std::cout << "Row " << rows_number << ":\n" << last_row << std::endl;
       }
}


/// This method re-writes the input operator >> for the Matrix template.
/// @param is Input stream.
/// @param m Input matrix.

template<class T>
std::istream& operator >> (std::istream& is, Matrix<T>& m)
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


// Output operator

/// This method re-writes the output operator << for the Matrix template.
/// @param os Output stream.
/// @param m Output matrix.

template<class T>
std::ostream& operator << (std::ostream& os, const Matrix<T>& m)
{
   const size_t rows_number = m.get_rows_number();
   const size_t columns_number = m.get_columns_number();

   if(rows_number > 0 && columns_number > 0)
   {
       os << m.arrange_row(0);

       for(size_t i = 1; i < rows_number; i++)
       {
           os << "\n"
              << m.arrange_row(i);
       }
   }

   return(os);
}


// Output operator

/// This method re-writes the output operator << for matrices of vectors.
/// @param os Output stream.
/// @param m Output matrix of vectors.

template<class T>
std::ostream& operator << (std::ostream& os, const Matrix< Vector<T> >& m)
{
   const size_t rows_number = m.get_rows_number();
   const size_t columns_number = m.get_columns_number();

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         os << "subvector_" << i << "_" << j << "\n"
            << m(i,j) << std::endl;
      }
   }

   return(os);
}


// Output operator

/// This method re-writes the output operator << for matrices of matrices.
/// @param os Output stream.
/// @param m Output matrix of matrices.

template<class T>
std::ostream& operator << (std::ostream& os, const Matrix< Matrix<T> >& m)
{
   const size_t rows_number = m.get_rows_number();
   const size_t columns_number = m.get_columns_number();

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         os << "submatrix_" << i << "_" << j << "\n"
            << m(i,j);
      }
   }

   return(os);
}
} // end namespace

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

