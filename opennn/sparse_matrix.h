/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S P A R S E   M A T R I X   C O N T A I N E R                                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __SPARSEMATRIX_H__
#define __SPARSEMATRIX_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// OpenNN includes

#include "matrix.h"

namespace OpenNN
{

/// This template class defines a sparse matrix for general purpose use.
/// This sparse matrix also implements some mathematical methods which can be useful.

template <class T>
class SparseMatrix
{

public:

    // CONSTRUCTORS

    explicit SparseMatrix();

    explicit SparseMatrix(const size_t&, const size_t&);

    explicit SparseMatrix(const string&);

    SparseMatrix(const SparseMatrix&);

    // DESTRUCTOR

    virtual ~SparseMatrix();

    // ASSIGNMENT OPERATORS

    inline SparseMatrix<T>& operator = (const SparseMatrix<T>&);

    // REFERENCE OPERATORS

    inline const T operator()(const size_t&, const size_t&) const;

    bool operator == (const SparseMatrix<T>&) const;

    bool operator == (const Matrix<T>&) const;

    bool operator == (const T&) const;

    bool operator != (const SparseMatrix<T>&) const;

    bool operator != (const Matrix<T>&) const;

    bool operator >(const SparseMatrix<T>&) const;

    bool operator >(const Matrix<T>&) const;

    bool operator >(const T&) const;

    bool operator <(const SparseMatrix<T>&) const;

    bool operator <(const Matrix<T>&) const;

    bool operator <(const T&) const;

    bool operator >= (const SparseMatrix<T>&) const;

    bool operator >= (const Matrix<T>&) const;

    bool operator >= (const T&) const;

    bool operator <= (const SparseMatrix<T>&) const;

    bool operator <= (const Matrix<T>&) const;

    bool operator <= (const T&) const;

    // METHODS

    // Get methods

    const size_t& get_rows_number() const;

    const size_t& get_columns_number() const;

    const Vector<size_t>& get_rows_indices() const;

    const Vector<size_t>& get_columns_indices() const;

    const Vector<T>& get_matrix_values() const;

    // Set methods

    void set();

    void set(const size_t&, const size_t&);

    void set(const SparseMatrix<T>&);

    void set(const string&);

    void set_identity(const size_t&);

    void set_rows_number(const size_t&);

    void set_columns_number(const size_t&);

    void set_element(const size_t&, const size_t&, const T&);

    void set_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<T>&);

    // Count methods

    size_t count_diagonal_elements() const;

    size_t count_off_diagonal_elements() const;

    size_t count_equal_to(const T&) const;
    size_t count_equal_to(const size_t&, const T&) const;

    size_t count_not_equal_to(const T&) const;
    size_t count_not_equal_to(const size_t&, const T&) const;

    size_t count_rows_equal_to(const Vector<size_t>&, const T&) const;
    bool is_row_equal_to(const size_t&, const Vector<size_t>&, const T&) const;

    Vector<size_t> get_row_indices_equal_to(const Vector<size_t>&, const T&) const;

    SparseMatrix<T> get_sub_sparse_matrix(const Vector<size_t>&, const Vector<size_t>&) const;

    SparseMatrix<T> get_sub_sparse_matrix_rows(const Vector<size_t>&) const;

    SparseMatrix<T> get_sub_sparse_matrix_columns(const Vector<size_t>&) const;

    Vector<T> get_row(const size_t&) const;

    Vector<T> get_rows(const size_t&, const size_t&) const;

    Vector<T> get_row(const size_t&, const Vector<size_t>&) const;

    Vector<T> get_column(const size_t&) const;

    size_t count_unique() const;

    Vector<T> get_diagonal() const;

    void set_row(const size_t&, const Vector<T>&);

    void set_row(const size_t&, const T&);

    void set_column(const size_t&, const Vector<T>&);

    void set_column(const size_t&, const T&);

    void set_diagonal(const T&);

    void set_diagonal(const Vector<T>&);

    void initialize_diagonal(const T&);

    void initialize_diagonal(const size_t&, const T&);

    void initialize_diagonal(const size_t&, const Vector<T>&);

    void initialize_identity();

    SparseMatrix<T> sum_diagonal(const T&) const;

    SparseMatrix<T> sum_diagonal(const Vector<T>&) const;

    void append_row(const Vector<T>&);

    void append_column(const Vector<T>&);

    SparseMatrix<T> insert_row(const size_t&, const Vector<T>&) const;

    SparseMatrix<T> insert_column(const size_t&, const Vector<T>&);

    SparseMatrix<T> merge_matrices(const SparseMatrix<T>&, const size_t&, const size_t&) const;
    Matrix<T> merge_matrices(const Matrix<T>&, const size_t&, const size_t&) const;

    SparseMatrix<T> delete_row(const size_t&) const;
    SparseMatrix<T> delete_rows(const Vector<size_t>&) const;

    SparseMatrix<T> delete_rows_with_value(const T&) const;

    SparseMatrix<T> delete_first_rows(const size_t&) const;
    SparseMatrix<T> get_first_rows(const size_t&) const;

    SparseMatrix<T> delete_last_rows(const size_t&) const;
    SparseMatrix<T> get_last_rows(const size_t&) const;

    SparseMatrix<T> delete_column(const size_t&) const;
    SparseMatrix<T> delete_columns(const Vector<size_t>&) const;

    SparseMatrix<T> remove_constant_rows() const;
    SparseMatrix<T> remove_constant_columns() const;

    SparseMatrix<T> assemble_rows(const SparseMatrix<T>&) const;
    Matrix<T> assemble_rows(const Matrix<T>&) const;

    SparseMatrix<T> assemble_columns(const SparseMatrix<T>&) const;
    Matrix<T> assemble_columns(const Matrix<T>&) const;

    SparseMatrix<T> sort_ascending(const size_t&) const;
    SparseMatrix<T> sort_descending(const size_t&) const;

    void replace(const T&, const T&);
    void replace_in_row(const size_t&, const T&, const T&);
    void replace_in_column(const size_t&, const T&, const T&);

    bool has_column_value(const size_t&, const T&) const;

    // Mathematical methods

    T calculate_sum() const;

    Vector<int> calculate_rows_sum_int() const;
    Vector<T> calculate_rows_sum() const;
    Vector<T> calculate_columns_sum() const;

    Vector<size_t> calculate_most_frequent_columns_indices(const size_t& = 10);

    void sum_row(const size_t&, const Vector<T>&);

    double calculate_trace() const;

    Vector<double> calculate_mean() const;
    double calculate_mean(const size_t&) const;

    Vector<double> calculate_mean(const Vector<size_t>&) const;

    Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector<double> calculate_mean_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Vector<double> > calculate_mean_standard_deviation() const;

    Vector< Vector<double> > calculate_mean_standard_deviation(const Vector<size_t>&) const;

    Vector< Vector<double> > calculate_mean_standard_deviation(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector<double> calculate_median() const;
    double calculate_median(const size_t&) const;

    Vector<double> calculate_median(const Vector<size_t>&) const;

    Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector<double> calculate_median_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    T calculate_minimum() const;

    T calculate_maximum() const;

    T calculate_column_minimum(const size_t&) const;

    T calculate_column_maximum(const size_t&) const;

    Vector<T> calculate_means_binary() const;
    Vector<T> calculate_means_binary_column() const;
    Vector<T> calculate_means_binary_columns() const;

    Vector<T> calculate_means_binary_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<T> calculate_means_binary_column_missing_values(const Vector< Vector<size_t> >&) const;
    Vector<T> calculate_means_binary_columns_missing_values(const Vector< Vector<size_t> >&) const;

    Vector< Vector<T> > calculate_minimum_maximum() const;

    Vector< Vector<T> > calculate_minimum_maximum(const Vector<size_t>&) const;

    Vector< Vector<T> > calculate_minimum_maximum(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_statistics() const;

    Vector< Statistics<T> > calculate_statistics(const Vector<size_t>&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_statistics(const Vector< Vector<size_t> >&, const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const;

    Vector< Statistics<T> > calculate_columns_statistics_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >) const;

    Vector< Statistics<T> > calculate_rows_statistics(const Vector<size_t>&) const;

    Vector< Statistics<T> > calculate_rows_statistics_missing_values(const Vector<size_t>&, const Vector< Vector<size_t> >&) const;

    Vector< Statistics<T> > calculate_columns_statistics(const Vector<size_t>&) const;

    Vector<T> calculate_rows_means(const Vector<size_t>& = Vector<size_t>()) const;

    Vector<T> calculate_columns_minimums(const Vector<size_t>& = Vector<size_t>()) const;

    Vector<T> calculate_columns_maximums(const Vector<size_t>& = Vector<size_t>()) const;

    Vector< Vector<double> > calculate_box_plots(const Vector< Vector<size_t> >&, const Vector<size_t>&) const;

    SparseMatrix<double> calculate_covariance_sparse_matrix() const;

    Vector< Histogram<T> > calculate_histograms(const size_t& = 10) const;

    Vector< Histogram<T> > calculate_histograms_missing_values(const Vector< Vector<size_t> >&, const size_t& = 10) const;

    void scale_mean_standard_deviation(const Vector< Statistics<T> >&);

    Vector< Statistics<T> > scale_mean_standard_deviation();

    void scale_rows_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&);

    void scale_minimum_maximum(const Vector< Statistics<T> >&);

    Vector< Statistics<T> > scale_minimum_maximum();

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

    Vector<size_t> calculate_maximal_indices() const;

    Vector< Vector<size_t> > calculate_minimal_maximal_indices() const;

    double calculate_sum_squared_error(const SparseMatrix<double>&) const;
    double calculate_sum_squared_error(const Matrix<T>&) const;

    double calculate_minkowski_error(const SparseMatrix<double>&, const double&) const;
    double calculate_minkowski_error(const Matrix<T>&, const double&) const;

    double calculate_sum_squared_error(const Vector<double>&) const;

    Vector<double> calculate_rows_norm() const;

    SparseMatrix<T> calculate_absolute_value() const;

    SparseMatrix<T> calculate_transpose() const;

    T calculate_determinant() const;

    SparseMatrix<T> calculate_cofactor() const;

    SparseMatrix<T> calculate_inverse() const;

    SparseMatrix<T> calculate_LU_inverse() const;

    Vector<T> solve_LDLT(const Vector<double>&) const;

    double calculate_distance(const size_t&, const size_t&) const;

    SparseMatrix<T> operator + (const T&) const;

    SparseMatrix<T> operator + (const Vector<T>&) const;

    SparseMatrix<T> operator + (const SparseMatrix<T>&) const;
    SparseMatrix<T> operator + (const Matrix<T>&) const;

    SparseMatrix<T> operator -(const T&) const;

    SparseMatrix<T> operator -(const Vector<T>&) const;

    SparseMatrix<T> operator -(const SparseMatrix<T>&) const;
    SparseMatrix<T> operator -(const Matrix<T>&) const;

    SparseMatrix<T> operator *(const T&) const;

    SparseMatrix<T> operator *(const Vector<T>&) const;

    SparseMatrix<T> operator *(const SparseMatrix<T>&) const;
    SparseMatrix<T> operator *(const Matrix<T>&) const;

    SparseMatrix<T> operator /(const T&) const;

    SparseMatrix<T> operator /(const Vector<T>&) const;

    SparseMatrix<T> operator /(const SparseMatrix<T>&) const;
    SparseMatrix<T> operator /(const Matrix<T>&) const;

    void operator += (const T&);

    void operator += (const Vector<T>&);

    void operator += (const SparseMatrix<T>&);
    void operator += (const Matrix<T>&);

    void operator -= (const T&);

    void operator -= (const Vector<T>&);

    void operator -= (const SparseMatrix<T>&);
    void operator -= (const Matrix<T>&);

    void operator *= (const T&);

    void operator *= (const Vector<T>&);

    void operator *= (const SparseMatrix<T>&);
    void operator *= (const Matrix<T>&);

    void operator /= (const T&);

    void operator /= (const Vector<T>&);

    void operator /= (const SparseMatrix<T>&);
    void operator /= (const Matrix<T>&);

    Vector<double> dot(const Vector<double>&) const;

    SparseMatrix<double> dot(const SparseMatrix<double>&) const;
    Matrix<T> dot(const Matrix<T>&) const;

    Matrix<T> calculate_eigenvalues() const;
    Matrix<T> calculate_eigenvectors() const;

    SparseMatrix<T> direct(const SparseMatrix<T>&) const;
    SparseMatrix<T> direct(const Matrix<T>&) const;

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

    bool is_dense(const double& = 1) const;

    void convert_association();

    KMeansResults<T> calculate_k_means(const size_t&) const;

    // Correlation methods

    Vector<T> calculate_multiple_linear_regression_parameters(const Vector<T>&) const;
    double calculate_multiple_linear_correlation(const Vector<T>&) const;

    // Serialization methods

    void print() const;

    void load(const string&);
    Vector<string> load_product_strings(const string&, const char& = ',');
    void load_binary(const string&);

    void save(const string&) const;

    void save_binary(const string&) const;

    void save_csv(const string&, const char& = ',',  const Vector<string>& = Vector<string>(),  const Vector<string>& = Vector<string>(), const string& = "Id") const;

    void parse(const string&);

    string SparseMatrix_to_string(const char& = ' ') const;

    SparseMatrix<size_t> to_size_t_SparseMatrix() const;
    SparseMatrix<double> to_double_SparseMatrix() const;
    SparseMatrix<string> to_string_SparseMatrix(const size_t& = 3) const;

    Matrix<T> to_matrix() const;
    Vector< Vector<T> > to_vector_of_vectors() const;

    Vector< Vector<size_t> > to_CSR(Vector<T>&) const;
    void from_CSR(const Vector<size_t>&, const Vector<size_t>&, const Vector<T>&);

    void print_preview() const;

private:

    /// Number of rows in the sparse matrix.

    size_t rows_number;

    /// Number of columns in the sparse matrix.

    size_t columns_number;

    /// Indices of the rows with values different than 0 in the sparse matrix.

    Vector<size_t> rows_indices;

    /// Indices of the columns with values different than 0 in the sparse matrix.

    Vector<size_t> columns_indices;

    /// Values different than 0 in the sparse matrix.

    Vector<T> matrix_values;
};

// CONSTRUCTORS

/// Default constructor. It creates a sparse matrix with zero rows and zero columns.

template <class T>
SparseMatrix<T>::SparseMatrix()
{
    rows_number = 0;
    columns_number = 0;
}


/// Constructor. It creates a SparseMatrix with n rows and m columns, containing n*m copies of the default value for Type.
/// @param new_rows_number Number of rows in SparseMatrix.
/// @param new_columns_number Number of columns in SparseMatrix.

template <class T>
SparseMatrix<T>::SparseMatrix(const size_t& new_rows_number, const size_t& new_columns_number)
{
    if(new_rows_number == 0 && new_columns_number == 0)
    {
        rows_number = 0;
        columns_number = 0;
    }
    else if(new_rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Constructor SparseMatrix(const size_t&, const size_t&).\n"
               << "Number of rows must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
    else if(new_columns_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Constructor SparseMatrix(const size_t&, const size_t&).\n"
               << "Number of columns must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
    else
    {
        rows_number = new_rows_number;
        columns_number = new_columns_number;
    }
}

/// File constructor. It creates a SparseMatrix which members are loaded from a data file.
/// @param file_name Name of SparseMatrix data file.

template <class T>
SparseMatrix<T>::SparseMatrix(const string& file_name)
{
    rows_number = 0;
    columns_number = 0;

    load(file_name);
}


/// Copy constructor. It creates a copy of an existing SparseMatrix.
/// @param other_sparse_matrix SparseMatrix to be copied.

template <class T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix& other_sparse_matrix)
{
    rows_number = other_sparse_matrix.rows_number;
    columns_number = other_sparse_matrix.columns_number;

    rows_indices = other_sparse_matrix.rows_indices;
    columns_indices = other_sparse_matrix.columns_indices;
    matrix_values = other_sparse_matrix.matrix_values;
}


// DESTRUCTOR

/// Destructor.

template <class T>
SparseMatrix<T>::~SparseMatrix()
{
    rows_number = 0;
    columns_number = 0;

    Vector<size_t>().swap(rows_indices);
    Vector<size_t>().swap(columns_indices);
    Vector<T>().swap(matrix_values);
}


// ASSIGNMENT OPERATORS

/// Assignment operator. It assigns to self a copy of an existing SparseMatrix.
/// @param other_sparse_matrix SparseMatrix to be assigned.

template <class T>
SparseMatrix<T>& SparseMatrix<T>::operator = (const SparseMatrix<T>& other_sparse_matrix)
{
    if(other_sparse_matrix.rows_number != rows_number || other_sparse_matrix.columns_number != columns_number)
    {
        rows_number = other_sparse_matrix.rows_number;
        columns_number = other_sparse_matrix.columns_number;
    }

    Vector<size_t>().swap(rows_indices);
    Vector<size_t>().swap(columns_indices);
    Vector<T>().swap(matrix_values);

    rows_indices = other_sparse_matrix.rows_indices;
    columns_indices = other_sparse_matrix.columns_indices;
    matrix_values = other_sparse_matrix.matrix_values;

    return(*this);
}


// REFERENCE OPERATORS

/// Reference operator.

/// Returns the element(i,j) of the SparseMatrix.
/// @param row Index of row.
/// @param column Index of column.

template <class T>
inline const T SparseMatrix<T>::operator()(const size_t& row, const size_t& column) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(row >= rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "operator()(const size_t&, const size_t&).\n"
               << "Row index (" << row << ") must be less than number of rows (" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }
    else if(column >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "operator()(const size_t&, const size_t&).\n"
               << "Column index (" << column << ") must be less than number of columns (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    T value = T();

    if(rows_indices.contains(row) && columns_indices.contains(column))
    {
        const size_t rows_indices_size = rows_indices.size();

        for(size_t i = 0; i < rows_indices_size; i++)
        {
            if(rows_indices[i] == row && columns_indices[i] == column)
            {
                value = matrix_values[i];

                break;
            }
        }
    }

    return(value);
}

// bool operator == (const SparseMatrix<T>&) const

/// Equivalent relational operator between this other_sparse_matrix and other other_sparse_matrix.
/// It produces true if all the elements of the two matrices are equal, and false otherwise.
/// @param other_sparse_matrix Sparse matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator == (const SparseMatrix<T>& other_sparse_matrix) const
{
    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    const size_t other_nonzero_columns_number = other_sparse_matrix.get_columns_indices().size();
    const size_t other_nonzero_rows_number = other_sparse_matrix.get_rows_indices().size();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        return(false);
    }
    else if(other_nonzero_columns_number != columns_indices.size() || other_nonzero_rows_number != rows_indices.size())
    {
        return false;
    }
    else if(other_nonzero_columns_number != 0 && other_nonzero_rows_number != 0)
    {
        for(size_t i = 0; i < other_nonzero_rows_number; i++)
        {
            const size_t current_row_index = other_sparse_matrix.rows_indices[i];
            const size_t current_column_index = other_sparse_matrix.columns_indices[i];

            const Vector<size_t> this_equal_rows_indices = rows_indices.calculate_equal_to_indices(current_row_index);
            const Vector<size_t> this_equal_columns_indices = columns_indices.calculate_equal_to_indices(current_column_index);

            const Vector<size_t> intersection = this_equal_rows_indices.get_intersection(this_equal_columns_indices);

            if(intersection.size() != 1 || matrix_values[intersection[0]] != other_sparse_matrix.matrix_values[i])
            {
                return false;
            }
        }
    }

    return(true);
}

// bool operator == (const T&)

/// Equivalent relational operator between this sparse matrix and a dense matrix.
/// @param other_matrix Dense matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator == (const Matrix<T>& other_matrix) const
{
    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        return false;
    }

    for(size_t i = 0; i < other_rows_number; i++)
    {
        if(!rows_indices.contains(i) && other_matrix.get_row(i).count_equal_to(T()) == other_columns_number)
        {
            continue;
        }

        for(size_t j = 0; j < other_columns_number; j++)
        {
            if(other_matrix(i,j) != (*this)(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}

template <class T>
bool SparseMatrix<T>::operator == (const T& value) const
{
    if(value == T())
    {
        return(matrix_values == value);
    }
    else if(is_dense())
    {
        return(matrix_values == value);
    }
    else
    {
        return false;
    }
}

// bool operator != (const SparseMatrix<T>&)

/// Not equivalent relational operator between this sparse matrix and other sparse matrix.
/// It produces true if the two matrices have any not equal element, and false otherwise.
/// @param other_sparse_matrix Sparse matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator != (const SparseMatrix<T>& other_sparse_matrix) const
{
    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    const size_t other_nonzero_columns_number = other_sparse_matrix.get_columns_indices().size();
    const size_t other_nonzero_rows_number = other_sparse_matrix.get_rows_indices().size();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        return(true);
    }
    else if(other_nonzero_columns_number != columns_indices.size() || other_nonzero_rows_number != rows_indices.size())
    {
        return true;
    }
    else if(other_nonzero_columns_number != 0 && other_nonzero_rows_number != 0)
    {
        for(size_t i = 0; i < other_nonzero_rows_number; i++)
        {
            const size_t current_row_index = other_sparse_matrix.rows_indices[i];
            const size_t current_column_index = other_sparse_matrix.columns_indices[i];

            const Vector<size_t> this_equal_rows_indices = rows_indices.calculate_equal_to_indices(current_row_index);
            const Vector<size_t> this_equal_columns_indices = columns_indices.calculate_equal_to_indices(current_column_index);

            const Vector<size_t> intersection = this_equal_rows_indices.get_intersection(this_equal_columns_indices);

            if(intersection.size() != 1 || matrix_values[intersection[0]] != other_sparse_matrix.matrix_values[i])
            {
                return true;
            }
        }
    }

    return(false);
}


// bool operator != (const Matrix<T>&) const

/// Not equivalent relational operator between this sparse matrix and a dense matrix.
/// @param other_matrix Dense matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator != (const Matrix<T>& other_matrix) const
{
    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        return false;
    }

    for(size_t i = 0; i < other_rows_number; i++)
    {
        if(!rows_indices.contains(i) && other_matrix.get_row(i).count_equal_to(T()) == other_columns_number)
        {
            continue;
        }

        for(size_t j = 0; j < other_columns_number; j++)
        {
            if(other_matrix(i,j) != (*this)(i,j))
            {
                return true;
            }
        }
    }

    return(false);
}

// bool operator >(const SparseMatrix<T>&) const

/// Greater than relational operator between this sparse matrix and other matrix.
/// It produces true if all the elements of this sparse matrix are greater than the corresponding elements of the other sparse matrix,
/// and false otherwise.
/// @param other_sparse_matrix Sparse matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator >(const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >(const SparseMatrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >(const SparseMatrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) <= other_sparse_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}


// bool operator >(const Matrix<T>&) const

/// Greater than relational operator between this sparse matrix and a dense matrix.
/// @param other_matrix Dense matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator >(const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >(const Matrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >(const Matrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) <= other_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}

template <class T>
bool SparseMatrix<T>::operator >(const T& value) const
{
    if(value >= T() && is_dense())
    {
        return(matrix_values > value);
    }
    if(value < T())
    {
        return(matrix_values > value);
    }
    else
    {
        return false;
    }
}

// bool operator <(const SparseMatrix<T>&) const

/// Less than relational operator between this sparse matrix and other sparse matrix.
/// It produces true if all the elements of this sparse matrix are less than the corresponding elements of the other sparse matrix,
/// and false otherwise.
/// @param other_sparse_matrix Sparse matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator <(const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <(const SparseMatrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <(const SparseMatrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) >= other_sparse_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}


// bool operator <(const Matrix<T>&) const

/// Less than relational operator between this sparse matrix and a dense matrix.
/// @param other_matrix Dense matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator <(const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <(const Matrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <(const Matrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) >= other_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}

template <class T>
bool SparseMatrix<T>::operator <(const T& value) const
{
    if(value <= T() && is_dense())
    {
        return(matrix_values < value);
    }
    if(value > T())
    {
        return(matrix_values < value);
    }
    else
    {
        return false;
    }
}

// bool operator >= (const SparseMatrix<T>&) const

/// Greater than or equal to relational operator between this sparse matrix and other sparse matrix.
/// It produces true if all the elements of this sparse matrix are greater than or equal to the corresponding elements of the
/// other sparse matrix, and false otherwise.
/// @param other_sparse_matrix Sparse matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator >= (const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >= (const SparseMatrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >= (const SparseMatrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) < other_sparse_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}


// bool operator >= (const Matrix<T>&) const

/// Greater than or equal to than relational operator between this sparse matrix and a dense matrix.
/// @param other_matrix Dense matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator >= (const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >= (const Matrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator >= (const Matrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) < other_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}

template <class T>
bool SparseMatrix<T>::operator >= (const T& value) const
{
    if(value > T() && is_dense())
    {
        return(matrix_values >= value);
    }
    else if(value <= T())
    {
        return(matrix_values >= value);
    }
    else
    {
        return false;
    }
}

// bool operator <= (const SparseMatrix<T>&) const

/// Less than or equal to relational operator between this sparse matrix and other sparse matrix.
/// It produces true if all the elements of this sparse matrix are less than or equal to the corresponding elements of the
/// other sparse matrix, and false otherwise.
/// @param other_sparse_matrix Sparse matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator <= (const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <= (const SparseMatrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <= (const SparseMatrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) > other_sparse_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}


// bool operator <= (const Matrix<T>&) const

/// Less than or equal to than relational operator between this sparse matrix and a dense matrix.
/// @param other_matrix Dense matrix to be compared with.

template <class T>
bool SparseMatrix<T>::operator <= (const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <= (const Matrix<T>&) const.\n"
               << "Both numbers of rows must be the same.\n";

        throw logic_error(buffer.str());
    }
    else if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool operator <= (const Matrix<T>&) const.\n"
               << "Both numbers of columns must be the same.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if((*this)(i,j) > other_matrix(i,j))
            {
                return false;
            }
        }
    }

    return(true);
}

template <class T>
bool SparseMatrix<T>::operator <= (const T& value) const
{
    if(value >= T())
    {
        return(matrix_values <= value);
    }
    else if(value < T() && is_dense())
    {
        return(matrix_values <= value);
    }
    else
    {
        return false;
    }
}

// METHODS

// Get methods

// size_t get_rows_number() const method

/// Returns the number of rows in the sparse matrix.

template <class T>
const size_t& SparseMatrix<T>::get_rows_number() const
{
    return(rows_number);
}

// size_t get_columns_number() const method

/// Returns the number of columns in the sparse matrix.

template <class T>
const size_t& SparseMatrix<T>::get_columns_number() const
{
    return(columns_number);
}

// Vector<size_t> get_rows_indices() const method

/// Returns the indices of rows in the sparse matrix.

template <class T>
const Vector<size_t>& SparseMatrix<T>::get_rows_indices() const
{
    return(rows_indices);
}

// Vector<size_t> get_columns_indices() const method

/// Returns the indices of columns in the sparse matrix.

template <class T>
const Vector<size_t>& SparseMatrix<T>::get_columns_indices() const
{
    return(columns_indices);
}

// size_t get_columns_indices() const method

/// Returns the values in the sparse matrix.

template <class T>
const Vector<T>& SparseMatrix<T>::get_matrix_values() const
{
    return(matrix_values);
}

// Set methods

// void set() method

/// This method set the numbers of rows and columns of the sparse matrix to zero.

template <class T>
void SparseMatrix<T>::set()
{
    rows_number = 0;
    columns_number = 0;
    Vector<size_t>().swap(rows_indices);
    Vector<size_t>().swap(columns_indices);
    Vector<T>().swap(matrix_values);
}

// void set(const size_t&, const size_t&) method

/// This method set new numbers of rows and columns in the sparse matrix.
/// @param new_rows_number Number of rows.
/// @param new_columns_number Number of columns.

template <class T>
void SparseMatrix<T>::set(const size_t& new_rows_number, const size_t& new_columns_number)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(new_rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void set(const size_t&, const size_t&) method.\n"
               << "Number of rows must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
    else if(new_columns_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void set(const size_t&, const size_t&) method.\n"
               << "Number of columns must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
#endif

    Vector<size_t>().swap(rows_indices);
    Vector<size_t>().swap(columns_indices);
    Vector<T>().swap(matrix_values);

    rows_number = new_rows_number;
    columns_number = new_columns_number;
}


// void set(const SparseMatrix<T>&) method

/// Sets all the members of the sparse matrix to those of another sparse matrix.
/// @param other_sparse_matrix Setting sparse matrix.

template <class T>
void SparseMatrix<T>::set(const SparseMatrix<T>& other_sparse_matrix)
{
    if(other_sparse_matrix.rows_number != rows_number || other_sparse_matrix.columns_number != columns_number)
    {
        rows_number = other_sparse_matrix.rows_number;
        columns_number = other_sparse_matrix.columns_number;
    }

    Vector<size_t>().swap(rows_indices);
    Vector<size_t>().swap(columns_indices);
    Vector<T>().swap(matrix_values);

    rows_indices = other_sparse_matrix.rows_indices;
    columns_indices = other_sparse_matrix.columns_indices;
    matrix_values = other_sparse_matrix.matrix_values;
}


// void set(const string&) method

/// Sets the members of this object by loading them from a data file.
/// @param file_name Name of data file.

template <class T>
void SparseMatrix<T>::set(const string& file_name)
{
    set();
    load(file_name);
}


// void set_identity(const size_t&) method

/// Sets the sparse matrix to be squared, with elements equal one in the diagonal and zero outside the diagonal.
/// @param new_size New number of rows and columns in this sparse matrix.

template <class T>
void SparseMatrix<T>::set_identity(const size_t& new_size)
{
    set(new_size, new_size);
    initialize_identity();
}


// void set_rows_number(const size_t&) method

/// Sets a new number of rows in the sparse matrix.
/// @param new_rows_number Number of sparse matrix rows.

template <class T>
void SparseMatrix<T>::set_rows_number(const size_t& new_rows_number)
{
    const size_t nonzero_elements_number = matrix_values.size();

    Vector<bool> indices_to_remove(nonzero_elements_number, false);

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        if(rows_indices[i] >= new_rows_number)
        {
            indices_to_remove[i] = true;
        }
    }

    Vector<size_t> greater_than_indices = indices_to_remove.calculate_equal_to_indices(true);

    if(greater_than_indices.size() > 0)
    {
        rows_indices = rows_indices.delete_indices(greater_than_indices);
        columns_indices = columns_indices.delete_indices(greater_than_indices);
        matrix_values = matrix_values.delete_indices(greater_than_indices);
    }

    rows_number = new_rows_number;
}


// void set_columns_number(const size_t&) method

/// Sets a new number of columns in the sparse matrix.
/// @param new_columns_number Number of sparse matrix columns.

template <class T>
void SparseMatrix<T>::set_columns_number(const size_t& new_columns_number)
{
    Vector<size_t> greater_than_indices = columns_indices.calculate_greater_equal_to_indices(new_columns_number);

    if(greater_than_indices.size() > 0)
    {
        rows_indices = rows_indices.delete_indices(greater_than_indices);
        columns_indices = columns_indices.delete_indices(greater_than_indices);
        matrix_values = matrix_values.delete_indices(greater_than_indices);
    }

    columns_number = new_columns_number;
}

template <class T>
void SparseMatrix<T>::set_element(const size_t& row_index, const size_t& column_index, const T& value)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(row_index >= rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_element(const size_t&, const size_t&, const T&).\n"
               << "Row index (" << row_index << ") must be less than number of rows (" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }
    else if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_element(const size_t&, const size_t&, const T&).\n"
               << "Column index (" << column_index << ") must be less than number of columns (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    if(value == T())
    {
        return;
    }

    Vector<size_t> equal_row_indices = rows_indices.calculate_equal_to_indices(row_index);
    Vector<size_t> equal_columns_indices = columns_indices.calculate_equal_to_indices(column_index);

    Vector<size_t> intersection = equal_row_indices.get_intersection(equal_columns_indices);

    if(intersection.size() == 0 && value != T())
    {
        rows_indices.push_back(row_index);
        columns_indices.push_back(column_index);

        matrix_values.push_back(value);
    }
    else if(intersection.size() != 0 && value != T())
    {
        matrix_values[intersection[0]] = value;
    }
    else if(intersection.size() != 0 && value == T())
    {
        rows_indices = rows_indices.delete_index(intersection[0]);
        columns_indices = columns_indices.delete_index(intersection[0]);
        matrix_values = matrix_values.delete_index(intersection[0]);
    }
}

template <class T>
void SparseMatrix<T>::set_values(const Vector<size_t>& new_rows_indices, const Vector<size_t>& new_columns_indices, const Vector<T>& new_matrix_values)
{
#ifdef __OPENNN_DEBUG__

    if(new_rows_indices.size() != new_columns_indices.size() || new_columns_indices.size() != new_matrix_values.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void set_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<T>&) const method.\n"
               << "Size of the three vector must be the same.\n";

        throw logic_error(buffer.str());
    }

    const size_t maximum_new_rows_indices = new_rows_indices.calculate_maximum();
    if(maximum_new_rows_indices > rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void set_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<T>&) const method.\n"
               << "Maximum of new row indices(" << maximum_new_rows_indices << ") must be less than the number of rows(" << rows_number <<  ").\n";

        throw logic_error(buffer.str());
    }

    const size_t maximum_new_columns_indices = new_columns_indices.calculate_maximum();
    if(maximum_new_columns_indices > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void set_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<T>&) const method.\n"
               << "Maximum of new columns indices(" << maximum_new_columns_indices << ") must be less than the number of columns(" << columns_number <<  ").\n";

        throw logic_error(buffer.str());
    }
#endif

    Vector<size_t>().swap(rows_indices);
    Vector<size_t>().swap(columns_indices);
    Vector<T>().swap(matrix_values);

    rows_indices = new_rows_indices;
    columns_indices = new_columns_indices;
    matrix_values = new_matrix_values;
}

// Count methods

// size_t count_diagonal_elements() const method

/// Returns the number of elements in the diagonal which are not zero.
/// This method is only defined for square matrices.

template <class T>
size_t SparseMatrix<T>::count_diagonal_elements() const
{
    const size_t elements_number = matrix_values.size();

    size_t count = 0;

    for(size_t i = 0; i < elements_number; i++)
    {
        if(rows_indices[i] == columns_indices[i])
        {
            count++;
        }
    }

    return count;
}

// size_t count_off_diagonal_elements() const method

/// Returns the number of elements outside the diagonal which are not zero.
/// This method is only defined for square matrices.

template <class T>
size_t SparseMatrix<T>::count_off_diagonal_elements() const
{
    const size_t elements_number = matrix_values.size();

    size_t count = 0;

    for(size_t i = 0; i < elements_number; i++)
    {
        if(rows_indices[i] != columns_indices[i])
        {
            count++;
        }
    }

    return count;
}

// size_t count_equal_to(const T&) const method

/// Returns the number of elements in the sparse matrix that are equal to a given value.
/// @param value Value to find.

template <class T>
size_t SparseMatrix<T>::count_equal_to(const T& value) const
{
    size_t count = 0;

    if(value == T())
    {
        count = rows_number*columns_number - matrix_values.size();
    }
    else
    {
        count = matrix_values.count_equal_to(value);
    }

    return count;
}

// size_t count_equal_to(const size_t&, const T&) const method

/// Returns the number of elements in a given column that are equal to a given value.
/// @param column_index Index of column.
/// @param value Value to find.

template <class T>
size_t SparseMatrix<T>::count_equal_to(const size_t& column_index, const T& value) const
{
#ifdef __OPENNN_DEBUG__

    if(column_index > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "size_t count_equal_to(const size_t&, const T&) const method.\n"
               << "Column index(" << column_index << ") must be less than number of columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }
#endif

    size_t count = 0;

    const Vector<size_t> column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

    if(value == T())
    {
        count = rows_number - column_nonzero_indices.size();
    }
    else
    {
        count = matrix_values.get_subvector(column_nonzero_indices).count_equal_to(value);
    }

    return(count);
}

// size_t count_not_equal_to(const T&) const method

/// Returns the number of elements in the sparse matrix that are not equal to a given value.
/// @param value Value to find.

template <class T>
size_t SparseMatrix<T>::count_not_equal_to(const T& value) const
{
    size_t count = 0;

    if(value == T())
    {
        count = matrix_values.size();
    }
    else
    {
        count = rows_number*columns_number - matrix_values.count_equal_to(value);
    }

    return count;
}

// size_t count_not_equal_to(const size_t&, const T&) const method

/// Returns the number of elements in a given column that are not equal to a given value.
/// @param column_index Index of column.
/// @param value Value to find.

template <class T>
size_t SparseMatrix<T>::count_not_equal_to(const size_t& column_index, const T& value) const
{
#ifdef __OPENNN_DEBUG__

    if(column_index > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "size_t count_not_equal_to(const size_t&, const T&) const method.\n"
               << "Column index(" << column_index << ") must be less than number of columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }
#endif

    size_t count = 0;

    const Vector<size_t> column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

    if(value == T())
    {
        count = column_nonzero_indices.size();
    }
    else
    {
        count = rows_number - matrix_values.get_subvector(column_nonzero_indices).count_equal_to(value);
    }

    return(count);
}

template <class T>
size_t SparseMatrix<T>::count_rows_equal_to(const Vector<size_t>& column_indices, const T& value) const
{
    const size_t column_size = column_indices.size();

    Vector< Vector<T> > columns(column_size);

    for(size_t i = 0; i < column_size; i++)
    {
        columns[i] = get_column(column_indices[i]);
    }

    Vector<bool> found(rows_number, true);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < column_size; j++)
        {
            if(columns[j][i] != value)
            {
                found[i] = false;

                break;
            }
        }
    }

    return(found.count_equal_to(true));
}


template <class T>
bool SparseMatrix<T>::is_row_equal_to(const size_t& row_index, const Vector<size_t>& column_indices, const T& value) const
{
    const size_t column_indices_size = column_indices.size();

    const Vector<T> current_row = get_row(row_index);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if(current_row[column_indices[i]] != value)
        {
            return(false);
        }
    }

    return(true);
}

template <class T>
Vector<size_t> SparseMatrix<T>::get_row_indices_equal_to(const Vector<size_t>& column_indices, const T& value) const
{
    const size_t column_size = column_indices.size();

    Vector< Vector<T> > columns(column_size);

    for(size_t i = 0; i < column_size; i++)
    {
        columns[i] = get_column(column_indices[i]);
    }

    Vector<bool> found(rows_number, true);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < column_size; j++)
        {
            if(columns[j][i] != value)
            {
                found[i] = false;

                break;
            }
        }
    }

    return(found.calculate_equal_to_indices(true));
}


/// Returns a sparse matrix with the values of given rows and columns from this sparse matrix.
/// @param row_indices Indices of sparse matrix rows.
/// @param column_indices Indices of sparse matrix columns.

template <class T>
SparseMatrix<T> SparseMatrix<T>::get_sub_sparse_matrix(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
{
    const size_t row_indices_size = row_indices.size();
    const size_t column_indices_size = column_indices.size();

    SparseMatrix<T> sub_sparse_matrix(row_indices_size, column_indices_size);

    size_t row_index;
    size_t column_index;

    for(size_t i = 0; i < row_indices_size; i++)
    {
        row_index = row_indices[i];

        for(size_t j = 0; j < column_indices_size; j++)
        {
            column_index = column_indices[j];

            const T current_value = (*this)(row_index,column_index);

            if(current_value != T())
            {
                sub_sparse_matrix.set_element(i,j,current_value);
            }
        }
    }

    return(sub_sparse_matrix);
}

/// Returns a sub sparse matrix with the values of given rows from this sparse matrix.
/// @param row_indices Indices of sparse matrix rows.

template <class T>
SparseMatrix<T> SparseMatrix<T>::get_sub_sparse_matrix_rows(const Vector<size_t>& row_indices) const
{
    const size_t row_indices_size = row_indices.size();

    SparseMatrix<T> sub_sparse_matrix(row_indices_size, columns_number);

    size_t row_index;

    for(size_t i = 0; i < row_indices_size; i++)
    {
        row_index = row_indices[i];

        for(size_t j = 0; j < columns_number; j++)
        {
            const T current_value = (*this)(row_index,j);

            if(current_value != T())
            {
                sub_sparse_matrix.set_element(i,j,current_value);
            }
        }
    }

    return(sub_sparse_matrix);
}


/// Returns a sub sparse matrix with the values of given columns from this sparse matrix.
/// @param column_indices Indices of sparse matrix columns.

template <class T>
SparseMatrix<T> SparseMatrix<T>::get_sub_sparse_matrix_columns(const Vector<size_t>& column_indices) const
{
    const size_t column_indices_size = column_indices.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    for(size_t i  = 0; i < column_indices_size; i++)
    {
        if(column_indices[i] >= columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix Template.\n"
                   << "SparseMatrix<T> get_sub_sparse_matrix_columns(const Vector<size_t>&) const method.\n"
                   << "Column index(" << i << ") must be less than number of columns(" << columns_number << ").\n";

            throw logic_error(buffer.str());
        }
    }

#endif

    SparseMatrix<T> sub_sparse_matrix(rows_number, column_indices_size);

    size_t column_index;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < column_indices_size; j++)
        {
            column_index = column_indices[j];

            const T current_value = (*this)(i,column_index);

            if(current_value != T())
            {
                sub_sparse_matrix.set_element(i,j,current_value);
            }
        }
    }

    return(sub_sparse_matrix);
}

// Vector<T> get_row(const size_t&) const method

/// Returns the row i of the sparse matrix.
/// @param i Index of row.

template <class T>
Vector<T> SparseMatrix<T>::get_row(const size_t& i) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(i >= rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Vector<T> get_row(const size_t&) const method.\n"
               << "Row index (" << i << ") must be less than number of rows (" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<T> row(columns_number, T());

    const size_t rows_indices_size = rows_indices.size();

    for(size_t j = 0; j < rows_indices_size; j++)
    {
        if(rows_indices[j] == i)
        {
            const size_t current_column_index = columns_indices[j];

            row[current_column_index] = matrix_values[j];
        }
    }

    return(row);
}

template <class T>
Vector<T> SparseMatrix<T>::get_rows(const size_t& first_index, const size_t& last_index) const
{
#ifdef __OPENNN_DEBUG__

    if(last_index > rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Vector<T> get_rows(const size_t&, const size_t&) const method.\n"
               << "Last index(" << last_index << ") must be less than number of rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<T> new_row;

    for(size_t i = first_index-1; i < last_index; i++)
    {
        new_row = new_row.assemble(get_row(i));
    }

    return new_row;
}


// Vector<T> get_row(const size_t&, const Vector<size_t>&) const method

/// Returns the row i of the sparse matrix, but only the elements specified by given indices.
/// @param row_index Index of row.
/// @param column_indices Column indices of row.

template <class T>
Vector<T> SparseMatrix<T>::get_row(const size_t& row_index, const Vector<size_t>& column_indices) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(row_index >= rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Vector<T> get_row(const size_t&, const Vector<size_t>&) const method.\n"
               << "Row index (" << row_index << ") must be less than number of rows (" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t size = column_indices.size();

    Vector<T> row(size, T());

    for(size_t i = 0; i < size; i++)
    {
        row[i] = (*this)(row_index,column_indices[i]);
    }

    return(row);
}

// Vector<T> get_column(const size_t&) const method

/// Returns the column j of the sparse matrix.
/// @param j Index of column.

template <class T>
Vector<T> SparseMatrix<T>::get_column(const size_t& j) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(j >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Vector<T> get_column(const size_t&) const method.\n"
               << "Column index(" << j << ") must be less than number of columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<T> column(rows_number, T());

    if(!columns_indices.contains(j))
    {
        return column;
    }

    const size_t columns_indices_size = columns_indices.size();

    for(size_t i = 0; i < columns_indices_size; i++)
    {
        if(columns_indices[i] == j)
        {
            const size_t current_row_index = rows_indices[i];
            column[current_row_index] = matrix_values[i];
        }
    }

    return(column);
}

template <class T>
size_t SparseMatrix<T>::count_unique() const
{
    size_t count = 0;

    if(matrix_values.size() < rows_number*columns_number &&
            !matrix_values.contains(T()))
    {
        count++;
    }

    count += matrix_values.count_unique();

    return count;
}

template <class T>
Vector<T> SparseMatrix<T>::get_diagonal() const
{
    Vector<T> diagonal(rows_number, T());

    const size_t values_number = rows_indices.size();

    for(size_t i = 0; i < values_number; i++)
    {
        if(rows_indices[i] == columns_indices[i])
        {
            diagonal[rows_indices[i]] = matrix_values[i];
        }
    }

    return diagonal;
}

template <class T>
void SparseMatrix<T>::set_row(const size_t& row_index, const Vector<T>& new_row)
{
    const size_t size = new_row.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(row_index >= rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_row(const size_t&, const Vector<T>&) method.\n"
               << "Index must be less than number of rows.\n";

        throw logic_error(buffer.str());
    }

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_row(const size_t&, const Vector<T>&) method.\n"
               << "Size(" << size << ") must be equal to number of columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    if(rows_indices.size() != 0)
    {
        const Vector<size_t> row_occurrences_indices = rows_indices.calculate_equal_to_indices(row_index);

        rows_indices = rows_indices.delete_indices(row_occurrences_indices);
        columns_indices = columns_indices.delete_indices(row_occurrences_indices);
        matrix_values = matrix_values.delete_indices(row_occurrences_indices);
    }

    for(size_t i = 0; i < size; i++)
    {
        if(new_row[i] != T())
        {
            rows_indices.push_back(row_index);
            columns_indices.push_back(i);
            matrix_values.push_back(new_row[i]);
        }
    }
}

template <class T>
void SparseMatrix<T>::set_row(const size_t& row_index, const T& value)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(row_index >= rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_row(const size_t&, const T&) method.\n"
               << "Index must be less than number of rows.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(rows_indices.size() != 0)
    {
        const Vector<size_t> row_occurrences_indices = rows_indices.calculate_equal_to_indices(row_index);

        rows_indices = rows_indices.delete_indices(row_occurrences_indices);
        columns_indices = columns_indices.delete_indices(row_occurrences_indices);
        matrix_values = matrix_values.delete_indices(row_occurrences_indices);
    }

    if(value == T())
    {
        return;
    }

    for(size_t i = 0; i < columns_number; i++)
    {
        rows_indices.push_back(row_index);
        columns_indices.push_back(i);
        matrix_values.push_back(value);
    }
}

template <class T>
void SparseMatrix<T>::set_column(const size_t& column_index, const Vector<T>& new_column)
{
    const size_t size = new_column.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_column(const size_t&, const Vector<T>&).\n"
               << "Index(" << column_index << ") must be less than number of columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_column(const size_t&, const Vector<T>&).\n"
               << "Size must be equal to number of rows.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(columns_indices.size() != 0)
    {
        const Vector<size_t> column_occurrences_indices = columns_indices.calculate_equal_to_indices(column_index);

        rows_indices = rows_indices.delete_indices(column_occurrences_indices);
        columns_indices = columns_indices.delete_indices(column_occurrences_indices);
        matrix_values = matrix_values.delete_indices(column_occurrences_indices);
    }

    for(size_t i = 0; i < size; i++)
    {
        if(new_column[i] != T())
        {
            rows_indices.push_back(i);
            columns_indices.push_back(column_index);
            matrix_values.push_back(new_column[i]);
        }
    }
}

template <class T>
void SparseMatrix<T>::set_column(const size_t& column_index, const T& value)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_column(const size_t&, const T&).\n"
               << "Index must be less than number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(columns_indices.size() != 0)
    {
        const Vector<size_t> column_occurrences_indices = columns_indices.calculate_equal_to_indices(column_index);

        rows_indices = rows_indices.delete_indices(column_occurrences_indices);
        columns_indices = columns_indices.delete_indices(column_occurrences_indices);
        matrix_values = matrix_values.delete_indices(column_occurrences_indices);
    }

    if(value == T())
    {
        return;
    }

    for(size_t i = 0; i < rows_number; i++)
    {
        rows_indices.push_back(i);
        columns_indices.push_back(column_index);
        matrix_values.push_back(value);
    }
}

template <class T>
void SparseMatrix<T>::set_diagonal(const T& new_diagonal)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_diagonal(const T&).\n"
               << "SparseMatrix must be square.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t previous_nonzero_values_number = matrix_values.size();

    if(previous_nonzero_values_number != 0)
    {
        Vector<size_t> diagonal_indices;

        for(size_t i = 0; i < previous_nonzero_values_number; i++)
        {
            if(rows_indices[i] == columns_indices[i])
            {
                diagonal_indices.push_back(i);
            }
        }

        rows_indices = rows_indices.delete_indices(diagonal_indices);
        columns_indices = columns_indices.delete_indices(diagonal_indices);
        matrix_values = matrix_values.delete_indices(diagonal_indices);
    }

    if(new_diagonal == T())
    {
        return;
    }

    for(size_t i = 0; i < rows_number; i++)
    {
        rows_indices.push_back(i);
        columns_indices.push_back(i);
        matrix_values.push_back(new_diagonal);
    }
}

template <class T>
void SparseMatrix<T>::set_diagonal(const Vector<T>& new_diagonal)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_diagonal(const Vector<T>&) const.\n"
               << "SparseMatrix is not square.\n";

        throw logic_error(buffer.str());
    }

    const size_t size = new_diagonal.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "set_diagonal(const Vector<T>&) const.\n"
               << "Size of diagonal(" << size << ") is not equal to size of Sparsematrix (" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t previous_nonzero_values_number = matrix_values.size();

    if(previous_nonzero_values_number != 0)
    {
        Vector<size_t> diagonal_indices;

        for(size_t i = 0; i < previous_nonzero_values_number; i++)
        {
            if(rows_indices[i] == columns_indices[i])
            {
                diagonal_indices.push_back(i);
            }
        }

        rows_indices = rows_indices.delete_indices(diagonal_indices);
        columns_indices = columns_indices.delete_indices(diagonal_indices);
        matrix_values = matrix_values.delete_indices(diagonal_indices);
    }

    for(size_t i = 0; i < rows_number; i++)
    {
        if(new_diagonal[i] != T())
        {
            rows_indices.push_back(i);
            columns_indices.push_back(i);
            matrix_values.push_back(new_diagonal[i]);
        }
    }
}

template <class T>
void SparseMatrix<T>::initialize_diagonal(const T& value)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number || rows_number == 0)
    {
        ostringstream buffer;

        cout << "OpenNN Exception: SparseMatrix Template.\n"
             << "initialize_diagonal(const T&) const method.\n"
             << "SparseMatrix must be square.\n";

        throw logic_error(buffer.str());
    }

#endif
    initialize_diagonal(rows_number,value);
}

template <class T>
void SparseMatrix<T>::initialize_diagonal(const size_t& new_size, const T& new_value)
{
    set(new_size,new_size);
    set_diagonal(new_value);
}

template <class T>
void SparseMatrix<T>::initialize_diagonal(const size_t& new_size, const Vector<T>& new_values)
{
    set(new_size,new_size);
    set_diagonal(new_values);
}

template <class T>
void SparseMatrix<T>::initialize_identity()
{
    initialize_diagonal(rows_number,1);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::sum_diagonal(const T& value) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "sum_diagonal(const T&) const.\n"
               << "SparseMatrix must be square.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> sum(*this);

    Vector<T> new_diagonal(rows_number,T());

    const size_t nonzero_elements_number = matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        if(rows_indices[i] == columns_indices[i])
        {
            new_diagonal[rows_indices[i]] = matrix_values[i] + value;
        }
    }

    for(size_t i = 0; i < rows_number; i++)
    {
        if(new_diagonal[i] == T())
        {
            new_diagonal[i] = value;
        }
    }

    sum.set_diagonal(new_diagonal);

    return(sum);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::sum_diagonal(const Vector<T>& values) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "sum_diagonal(const Vector<T>&) const.\n"
               << "SparseMatrix must be square.\n";

        throw logic_error(buffer.str());
    }

    const size_t size = values.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "sum_diagonal(const Vector<T>&) const.\n"
               << "Size must be equal to number of rows.\n";

        throw logic_error(buffer.str());
    }

#endif
    SparseMatrix<T> sum(*this);

    Vector<T> new_diagonal(rows_number,T());

    const size_t nonzero_elements_number = matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        if(rows_indices[i] == columns_indices[i])
        {
            new_diagonal[rows_indices[i]] = matrix_values[i] + values[rows_indices[i]];
        }
    }

    for(size_t i = 0; i < rows_number; i++)
    {
        if(new_diagonal[i] == T())
        {
            new_diagonal[i] = values[rows_indices[i]];
        }
    }

    sum.set_diagonal(new_diagonal);

    return(sum);
}

template <class T>
void SparseMatrix<T>::append_row(const Vector<T>& new_row)
{
    const size_t size = new_row.size();

#ifdef __OPENNN_DEBUG__

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "append_row(const Vector<T>&) const.\n"
               << "Size(" << size << ") must be equal to number of columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    rows_number += 1;

    for(size_t i = 0; i < size; i++)
    {
        if(new_row[i] != T())
        {
            rows_indices.push_back(rows_number-1);
            columns_indices.push_back(i);
            matrix_values.push_back(new_row[i]);
        }
    }
}

template <class T>
void SparseMatrix<T>::append_column(const Vector<T>& new_column)
{
    const size_t size = new_column.size();

#ifdef __OPENNN_DEBUG__

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "append_column(const Vector<T>&) const.\n"
               << "Size(" << size << ") must be equal to number of rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    columns_number += 1;

    for(size_t i = 0; i < size; i++)
    {
        if(new_column[i] != T())
        {
            rows_indices.push_back(i);
            columns_indices.push_back(columns_number-1);
            matrix_values.push_back(new_column[i]);
        }
    }
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::insert_row(const size_t& position, const Vector<T>& new_row) const
{
    const size_t size = new_row.size();

#ifdef __OPENNN_DEBUG__

    if(position > rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "insert_row(const size_t&, const Vector<T>&) const.\n"
               << "Position must be less or equal than number of rows.\n";

        throw logic_error(buffer.str());
    }

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "insert_row(const size_t&, const Vector<T>&) const.\n"
               << "Size must be equal to number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(*this);

    new_sparse_matrix.set_rows_number(rows_number+1);

    Vector<size_t> new_rows_indices = new_sparse_matrix.get_rows_indices();
    Vector<size_t> new_columns_indices = new_sparse_matrix.get_columns_indices();
    Vector<T> new_matrix_values = new_sparse_matrix.get_matrix_values();

    const Vector<size_t> greater_rows_than_position_indices = new_rows_indices.calculate_greater_equal_to_indices(position);

    const size_t greater_rows_than_position_number = greater_rows_than_position_indices.size();

    for(size_t i = 0; i < greater_rows_than_position_number; i++)
    {
        new_rows_indices[greater_rows_than_position_indices[i]] += 1;
    }

    for(size_t i = 0; i < size; i ++)
    {
        if(new_row[i] != T())
        {
            new_rows_indices.push_back(position);
            new_columns_indices.push_back(i);
            new_matrix_values.push_back(new_row[i]);
        }
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::insert_column(const size_t& position, const Vector<T>& new_column)
{
    const size_t size = new_column.size();

#ifdef __OPENNN_DEBUG__

    if(position > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "insert_column(const size_t&, const Vector<T>&) const.\n"
               << "Position must be less or equal than number of columns.\n";

        throw logic_error(buffer.str());
    }

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "insert_column(const size_t, const Vector<T>&) const.\n"
               << "Size(" << size << ") must be equal to number of rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(*this);

    new_sparse_matrix.set_columns_number(columns_number+1);

    Vector<size_t> new_rows_indices = new_sparse_matrix.get_rows_indices();
    Vector<size_t> new_columns_indices = new_sparse_matrix.get_columns_indices();
    Vector<T> new_matrix_values = new_sparse_matrix.get_matrix_values();

    const Vector<size_t> greater_columns_than_position_indices = new_columns_indices.calculate_greater_equal_to_indices(position);

    const size_t greater_columns_than_position_number = greater_columns_than_position_indices.size();

    for(size_t i = 0; i < greater_columns_than_position_number; i++)
    {
        new_columns_indices[greater_columns_than_position_indices[i]] += 1;
    }

    for(size_t i = 0; i < size; i ++)
    {
        if(new_column[i] != T())
        {
            new_rows_indices.push_back(i);
            new_columns_indices.push_back(position);
            new_matrix_values.push_back(new_column[i]);
        }
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::merge_matrices(const SparseMatrix<T>& other_sparse_matrix, const size_t& columns_1_index, const size_t& columns_2_index) const
{
    const Vector<T> columns_1 = this->get_column(columns_1_index);
    const Vector<T> columns_2 = other_sparse_matrix.get_column(columns_2_index);

    const size_t columns_1_size = columns_1.size();

    size_t merged_rows_number = 0;

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const T current_index_value = columns_1[i];

        const size_t columns_2_equal_number = columns_2.count_equal_to(current_index_value);

        merged_rows_number += columns_2_equal_number;
    }

    const size_t merged_columns_number = columns_number + other_sparse_matrix.get_columns_number() - 1;

    SparseMatrix<T> merged_sparse_matrix(merged_rows_number, merged_columns_number);

    size_t current_row_index = 0;

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const T current_index_value = columns_1[i];

        const Vector<size_t> columns_2_equal_indices = columns_2.calculate_equal_to_indices(current_index_value);

        const size_t columns_2_equal_indices_size = columns_2_equal_indices.size();

        for(size_t j = 0; j < columns_2_equal_indices_size; j++)
        {
            Vector<T> current_row = this->get_row(i);

            current_row = current_row.assemble(other_sparse_matrix.get_row(columns_2_equal_indices[j]).remove_element(columns_2_index));

            merged_sparse_matrix.set_row(current_row_index, current_row);

            current_row_index++;
        }
    }

    return merged_sparse_matrix;
}

template <class T>
Matrix<T> SparseMatrix<T>::merge_matrices(const Matrix<T>& other_matrix, const size_t& columns_1_index, const size_t& columns_2_index) const
{
    const Vector<T> columns_1 = this->get_column(columns_1_index);
    const Vector<T> columns_2 = other_matrix.get_column(columns_2_index);

    const size_t columns_1_size = columns_1.size();

    size_t merged_rows_number = 0;

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const T current_index_value = columns_1[i];

        const size_t columns_2_equal_number = columns_2.count_equal_to(current_index_value);

        merged_rows_number += columns_2_equal_number;
    }

    const size_t merged_columns_number = columns_number + other_matrix.get_columns_number() - 1;

    Matrix<T> merged_matrix(merged_rows_number, merged_columns_number);

    size_t current_row_index = 0;

    for(size_t i = 0; i < columns_1_size; i++)
    {
        const T current_index_value = columns_1[i];

        const Vector<size_t> columns_2_equal_indices = columns_2.calculate_equal_to_indices(current_index_value);

        const size_t columns_2_equal_indices_size = columns_2_equal_indices.size();

        for(size_t j = 0; j < columns_2_equal_indices_size; j++)
        {
            Vector<T> current_row = this->get_row(i);

            current_row = current_row.assemble(other_matrix.get_row(columns_2_equal_indices[j]).remove_element(columns_2_index));

            merged_matrix.set_row(current_row_index, current_row);

            current_row_index++;
        }
    }

    return merged_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::delete_row(const size_t& row_index) const
{
#ifdef __OPENNN_DEBUG__

    if(row_index >= rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> delete_row(const size_t&) const.\n"
               << "Index of row must be less than number of rows.\n";

        throw logic_error(buffer.str());
    }
    else if(rows_number < 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> delete_row(const size_t&) const.\n"
               << "Number of rows must be equal or greater than two.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(rows_number-1,columns_number);

    Vector<size_t> new_rows_indices = rows_indices;
    Vector<size_t> new_columns_indices = columns_indices;
    Vector<T> new_matrix_values = matrix_values;

    const size_t nonzero_elements_number = new_matrix_values.size();

    Vector<size_t> indices_to_remove;

    for(size_t j = 0; j < nonzero_elements_number; j++)
    {
        if(new_rows_indices[j] == row_index)
        {
            indices_to_remove.push_back(j);
        }
        else if(new_rows_indices[j] > row_index)
        {
            new_rows_indices[j]--;
        }
    }

    new_rows_indices = new_rows_indices.delete_indices(indices_to_remove);
    new_columns_indices = new_columns_indices.delete_indices(indices_to_remove);
    new_matrix_values = new_matrix_values.delete_indices(indices_to_remove);

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::delete_rows(const Vector<size_t>& rows_to_remove) const
{
    const size_t rows_to_remove_size = rows_to_remove.size();

    SparseMatrix<T> new_sparse_matrix(rows_number-rows_to_remove_size, columns_number);

    Vector<size_t> indices_to_keep = rows_indices.calculate_not_equal_to_indices(rows_to_remove);

    const size_t indices_to_keep_size = indices_to_keep.size();

    Vector<size_t> new_rows_indices(indices_to_keep_size);
    Vector<size_t> new_columns_indices(indices_to_keep_size);
    Vector<T> new_matrix_values(indices_to_keep_size);

    for(size_t i = 0; i < indices_to_keep_size; i++)
    {
        const size_t current_index = indices_to_keep[i];

        new_rows_indices[i] = rows_indices[current_index];
        new_columns_indices[i] = columns_indices[current_index];
        new_matrix_values[i] = matrix_values[current_index];
    }

    const size_t nonzero_elements_number = new_matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        new_rows_indices[i] -= rows_to_remove.count_less_equal_to(new_rows_indices[i]);
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::delete_rows_with_value(const T& value) const
{
    Vector<size_t> valid_indices;

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i);

        if(!current_row.contains(value))
        {
            valid_indices.push_back(i);
        }
    }

    return get_sub_sparse_matrix_rows(valid_indices);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::delete_first_rows(const size_t& number) const
{
    const Vector<size_t> indices(number, 1, rows_number-1);

    return get_sub_sparse_matrix_rows(indices);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::get_first_rows(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, number-1);

    return get_sub_sparse_matrix_rows(indices);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::delete_last_rows(const size_t& number) const
{
    const Vector<size_t> indices(0, 1, rows_number-number-1);

    return get_sub_sparse_matrix_rows(indices);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::get_last_rows(const size_t& number) const
{
    const Vector<size_t> indices(rows_number-number, 1, rows_number-1);

    return get_sub_sparse_matrix_rows(indices);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::delete_column(const size_t& column_index) const
{
#ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "delete_column(const size_t&) const.\n"
               << "Index of column must be less than number of columns.\n";

        throw logic_error(buffer.str());
    }
    else if(columns_number < 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "delete_column(const size_t&) const.\n"
               << "Number of columns must be equal or greater than two.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(*this);

    new_sparse_matrix.set_columns_number(columns_number-1);

    Vector<size_t> new_rows_indices = new_sparse_matrix.get_rows_indices();
    Vector<size_t> new_columns_indices = new_sparse_matrix.get_columns_indices();
    Vector<T> new_matrix_values = new_sparse_matrix.get_matrix_values();

    const Vector<size_t> columns_equal_than_position_indices = new_columns_indices.calculate_equal_to_indices(column_index);

    new_rows_indices = new_rows_indices.delete_indices(columns_equal_than_position_indices);
    new_columns_indices = new_columns_indices.delete_indices(columns_equal_than_position_indices);
    new_matrix_values = new_matrix_values.delete_indices(columns_equal_than_position_indices);

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::delete_columns(const Vector<size_t>& columns_to_remove) const
{
    const size_t rows_to_remove_size = columns_to_remove.size();

    SparseMatrix<T> new_sparse_matrix(rows_number-rows_to_remove_size, columns_number);

    Vector<size_t> indices_to_keep = rows_indices.calculate_not_equal_to_indices(columns_to_remove);

    const size_t indices_to_keep_size = indices_to_keep.size();

    Vector<size_t> new_rows_indices(indices_to_keep_size);
    Vector<size_t> new_columns_indices(indices_to_keep_size);
    Vector<T> new_matrix_values(indices_to_keep_size);

    for(size_t i = 0; i < indices_to_keep_size; i++)
    {
        const size_t current_index = indices_to_keep[i];

        new_rows_indices[i] = rows_indices[current_index];
        new_columns_indices[i] = columns_indices[current_index];
        new_matrix_values[i] = matrix_values[current_index];
    }

    const size_t nonzero_elements_number = new_matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        new_columns_indices[i] -= columns_to_remove.count_less_equal_to(new_columns_indices[i]);
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::remove_constant_rows() const
{
    Vector<size_t> rows_to_remove;

    for(size_t i = 0; i < rows_number; i++)
    {
        const size_t occurrences_number = rows_indices.count_equal_to(i);

        if(occurrences_number == 0)
        {
            rows_to_remove.push_back(i);

            continue;
        }

        const Vector<T> row = get_row(i);

        if(row.is_constant())
        {
            rows_to_remove.push_back(i);
        }
    }

    return delete_rows(rows_to_remove);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::remove_constant_columns() const
{
    Vector<size_t> columns_to_remove;

    for(size_t i = 0; i < columns_number; i++)
    {
        const size_t occurrences_number = columns_indices.count_equal_to(i);

        if(occurrences_number == 0)
        {
            columns_to_remove.push_back(i);

            continue;
        }

        const Vector<T> column = get_column(i);

        if(column.is_constant())
        {
            columns_to_remove.push_back(i);
        }
    }

    return delete_columns(columns_to_remove);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::assemble_rows(const SparseMatrix<T>& other_sparse_matrix) const
{
#ifdef __OPENNN_DEBUG__

    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> assemble_rows(const SparseMatrix<T>&) const method.\n"
               << "Number of columns of other Sparsematrix (" << other_columns_number << ") must be equal to number of columns of this Sparsematrix (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(*this);

    new_sparse_matrix.set_rows_number(rows_number + other_sparse_matrix.get_rows_number());

    Vector<size_t> new_rows_indices = new_sparse_matrix.get_rows_indices();
    Vector<size_t> new_columns_indices = new_sparse_matrix.get_columns_indices();
    Vector<T> new_matrix_values = new_sparse_matrix.get_matrix_values();

    Vector<size_t> other_rows_indices = other_sparse_matrix.get_rows_indices();
    Vector<size_t> other_columns_indices = other_sparse_matrix.get_columns_indices();
    Vector<size_t> other_matrix_values= other_sparse_matrix.get_matrix_values();

    other_rows_indices = other_rows_indices + rows_number;

    new_rows_indices = new_rows_indices.assemble(other_rows_indices);
    new_columns_indices = new_columns_indices.assemble(other_columns_indices);
    new_matrix_values = new_matrix_values.assemble(other_matrix_values);

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
Matrix<T> SparseMatrix<T>::assemble_rows(const Matrix<T>& other_matrix) const
{
    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

#ifdef __OPENNN_DEBUG__

    if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Matrix<T> assemble_rows(const Matrix<T>&) const method.\n"
               << "Number of columns of other matrix (" << other_columns_number << ") must be equal to number of columns of this Sparsematrix (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(*this);

    new_sparse_matrix.set_rows_number(rows_number + other_matrix.get_rows_number());

    Vector<size_t> new_rows_indices = new_sparse_matrix.get_rows_indices();
    Vector<size_t> new_columns_indices = new_sparse_matrix.get_columns_indices();
    Vector<T> new_matrix_values = new_sparse_matrix.get_matrix_values();

    for(size_t i = 0; i < other_rows_number; i++)
    {
        for(size_t j = 0; j < other_columns_number; j++)
        {
            if(other_matrix(i,j) != T())
            {
                new_rows_indices.push_back(rows_number + i);
                new_columns_indices.push_back(j);
                new_matrix_values.push_back(other_matrix(i,j));
            }
        }
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::assemble_columns(const SparseMatrix<T>& other_sparse_matrix) const
{
#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> assemble_columns(const SparseMatrix<T>&) const method.\n"
               << "Number of rows of other Sparsematrix (" << other_rows_number << ") must be equal to number of rows of this Sparsematrix (" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(*this);

    new_sparse_matrix.set_rows_number(columns_number + other_sparse_matrix.get_columns_number());

    Vector<size_t> new_rows_indices = new_sparse_matrix.get_rows_indices();
    Vector<size_t> new_columns_indices = new_sparse_matrix.get_columns_indices();
    Vector<T> new_matrix_values = new_sparse_matrix.get_matrix_values();

    Vector<size_t> other_rows_indices = other_sparse_matrix.get_rows_indices();
    Vector<size_t> other_columns_indices = other_sparse_matrix.get_columns_indices();
    Vector<size_t> other_matrix_values= other_sparse_matrix.get_matrix_values();

    other_columns_indices = other_columns_indices + columns_number;

    new_rows_indices = new_rows_indices.assemble(other_rows_indices);
    new_columns_indices = new_columns_indices.assemble(other_columns_indices);
    new_matrix_values = new_matrix_values.assemble(other_matrix_values);

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
Matrix<T> SparseMatrix<T>::assemble_columns(const Matrix<T>& other_matrix) const
{
    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

#ifdef __OPENNN_DEBUG__

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Matrix<T> assemble_columns(const Matrix<T>&) const method.\n"
               << "Number of rows of other matrix (" << other_rows_number << ") must be equal to number of rows of this Sparsematrix (" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> new_sparse_matrix(*this);

    new_sparse_matrix.set_rows_number(rows_number + other_matrix.get_rows_number());

    Vector<size_t> new_rows_indices = new_sparse_matrix.get_rows_indices();
    Vector<size_t> new_columns_indices = new_sparse_matrix.get_columns_indices();
    Vector<T> new_matrix_values = new_sparse_matrix.get_matrix_values();

    for(size_t i = 0; i < other_rows_number; i++)
    {
        for(size_t j = 0; j < other_columns_number; j++)
        {
            if(other_matrix(i,j) != T())
            {
                new_rows_indices.push_back(i);
                new_columns_indices.push_back(columns_number + j);
                new_matrix_values.push_back(other_matrix(i,j));
            }
        }
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::sort_ascending(const size_t& column_index) const
{
    Vector<T> current_column = get_column(column_index);

    Vector<size_t> sorted_indices = current_column.sort_ascending_indices();

    SparseMatrix<T> new_sparse_matrix(rows_number, columns_number);

    Vector<size_t> new_rows_indices = this->get_rows_indices();
    Vector<size_t> new_columns_indices = this->get_columns_indices();
    Vector<T> new_matrix_values = this->get_matrix_values();

    const size_t nonzero_elements_number = new_matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        const size_t sorted_index = sorted_indices.calculate_equal_to_indices(new_rows_indices[i])[0];

        new_rows_indices[i] = sorted_index;
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::sort_descending(const size_t& column_index) const
{
    Vector<T> current_column = get_column(column_index);

    Vector<size_t> sorted_indices = current_column.sort_descending_indices();

    SparseMatrix<T> new_sparse_matrix(rows_number, columns_number);

    Vector<size_t> new_rows_indices = this->get_rows_indices();
    Vector<size_t> new_columns_indices = this->get_columns_indices();
    Vector<T> new_matrix_values = this->get_matrix_values();

    const size_t nonzero_elements_number = new_matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        const size_t sorted_index = sorted_indices.calculate_equal_to_indices(new_rows_indices[i])[0];

        new_rows_indices[i] = sorted_index;
    }

    new_sparse_matrix.set_values(new_rows_indices, new_columns_indices, new_matrix_values);

    return new_sparse_matrix;
}

template <class T>
void SparseMatrix<T>::replace(const T& find_what, const T& replace_with)
{
    if(find_what == replace_with)
    {
        return;
    }
    else if(find_what != T() && replace_with != T())
    {
        matrix_values = matrix_values.replace_value(find_what, replace_with);
    }
    else if(find_what == T())
    {
        matrix_values = matrix_values.replace_value(find_what, replace_with);

        const Vector<size_t> sequential_indices(0,1,columns_indices);

        for(size_t i = 0; i < rows_number; i++)
        {
            const Vector<size_t> current_row_nonzero_indices = rows_indices.calculate_equal_to_indices(i);

            const Vector<size_t> current_zero_columns = sequential_indices.get_difference(columns_indices.get_subvector(current_row_nonzero_indices));

            const size_t current_zero_columns_size = current_zero_columns.size();

            for(size_t j = 0; j < current_zero_columns_size; j++)
            {
                rows_indices.push_back(i);
                columns_indices.push_back(current_zero_columns[j]);
                matrix_values.push_back(replace_with);
            }
        }
    }
    else if(replace_with == T())
    {
        const Vector<size_t> equal_than_find_indices = matrix_values.calculate_equal_to_indices(find_what);

        rows_indices = rows_indices.delete_indices(equal_than_find_indices);
        columns_indices = columns_indices.delete_indices(equal_than_find_indices);
        matrix_values = matrix_values.delete_indices(equal_than_find_indices);
    }
}

template <class T>
void SparseMatrix<T>::replace_in_row(const size_t& row_index, const T& find_what, const T& replace_with)
{
    if(find_what == replace_with)
    {
        return;
    }
    else if(find_what != T() && replace_with != T())
    {
        const Vector<size_t> current_row_nonzero_indices = rows_indices.calculate_equal_to_indices(row_index);

        const size_t current_row_nonzero_number = current_row_nonzero_indices.size();

        for(size_t i = 0; i < current_row_nonzero_number; i++)
        {
            if(matrix_values[current_row_nonzero_indices[i]] == find_what)
            {
                matrix_values[current_row_nonzero_indices[i]] == replace_with;
            }
        }
    }
    else if(find_what == T())
    {
        const Vector<size_t> current_row_nonzero_indices = rows_indices.calculate_equal_to_indices(row_index);

        const size_t current_row_nonzero_number = current_row_nonzero_indices.size();

        for(size_t i = 0; i < current_row_nonzero_number; i++)
        {
            if(matrix_values[current_row_nonzero_indices[i]] == find_what)
            {
                matrix_values[current_row_nonzero_indices[i]] == replace_with;
            }
        }

        const Vector<size_t> sequential_indices(0,1,columns_indices);

        const Vector<size_t> current_zero_columns = sequential_indices.get_difference(columns_indices.get_subvector(current_row_nonzero_indices));

        const size_t current_zero_columns_size = current_zero_columns.size();

        for(size_t j = 0; j < current_zero_columns_size; j++)
        {
            rows_indices.push_back(row_index);
            columns_indices.push_back(current_zero_columns[j]);
            matrix_values.push_back(replace_with);
        }
    }
    else if(replace_with == T())
    {
        const Vector<size_t> current_row_nonzero_indices = rows_indices.calculate_equal_to_indices(row_index);

        const Vector<size_t> equal_than_find_indices = matrix_values.get_subvector(current_row_nonzero_indices).calculate_equal_to_indices(find_what);

        rows_indices = rows_indices.delete_indices(equal_than_find_indices);
        columns_indices = columns_indices.delete_indices(equal_than_find_indices);
        matrix_values = matrix_values.delete_indices(equal_than_find_indices);
    }
}

template <class T>
void SparseMatrix<T>::replace_in_column(const size_t& column_index, const T& find_what, const T& replace_with)
{
    if(find_what == replace_with)
    {
        return;
    }
    else if(find_what != T() && replace_with != T())
    {
        const Vector<size_t> current_column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

        const size_t current_column_nonzero_number = current_column_nonzero_indices.size();

        for(size_t i = 0; i < current_column_nonzero_number; i++)
        {
            if(matrix_values[current_column_nonzero_indices[i]] == find_what)
            {
                matrix_values[current_column_nonzero_indices[i]] == replace_with;
            }
        }
    }
    else if(find_what == T())
    {
        const Vector<size_t> current_column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

        const size_t current_column_nonzero_number = current_column_nonzero_indices.size();

        for(size_t i = 0; i < current_column_nonzero_number; i++)
        {
            if(matrix_values[current_column_nonzero_indices[i]] == find_what)
            {
                matrix_values[current_column_nonzero_indices[i]] == replace_with;
            }
        }

        const Vector<size_t> sequential_indices(0,1,rows_indices);

        const Vector<size_t> current_zero_rows = sequential_indices.get_difference(rows_indices.get_subvector(current_column_nonzero_indices));

        const size_t current_zero_rows_size = current_zero_rows.size();

        for(size_t j = 0; j < current_zero_rows_size; j++)
        {
            rows_indices.push_back(current_zero_rows[j]);
            columns_indices.push_back(column_index);
            matrix_values.push_back(replace_with);
        }
    }
    else if(replace_with == T())
    {
        const Vector<size_t> current_column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

        const Vector<size_t> equal_than_find_indices = matrix_values.get_subvector(current_column_nonzero_indices).calculate_equal_to_indices(find_what);

        rows_indices = rows_indices.delete_indices(equal_than_find_indices);
        columns_indices = columns_indices.delete_indices(equal_than_find_indices);
        matrix_values = matrix_values.delete_indices(equal_than_find_indices);
    }
}

template <class T>
bool SparseMatrix<T>::has_column_value(const size_t& column_index, const T& value) const
{
    return get_column(column_index).contains(value);
}

// Mathematical methods

template <class T>
T SparseMatrix<T>::calculate_sum() const
{
    return matrix_values.calculate_sum();
}

template <class T>
Vector<int> SparseMatrix<T>::calculate_rows_sum_int() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        cout << "OpenNN Exception: SparseMatrix Template.\n"
             << "Vector<int> calculate_rows_sum_int() const method.\n"
             << "SparseMatrix is empty.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<T> rows_sum(rows_number, T());

    const size_t rows_indices_size = rows_indices.size();

    for(size_t i = 0; i < rows_indices_size; i++)
    {
        const size_t current_row_index = rows_indices[i];

        rows_sum[current_row_index] += (int)matrix_values[i];
    }

    return rows_sum;
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_rows_sum() const
{
#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        cout << "OpenNN Exception: SparseMatrix Template.\n"
             << "Vector<T> calculate_rows_sum() const method.\n"
             << "SparseMatrix is empty.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<T> rows_sum(rows_number, T());

    const size_t rows_indices_size = rows_indices.size();

    for(size_t i = 0; i < rows_indices_size; i++)
    {
        const size_t current_row_index = rows_indices[i];

        rows_sum[current_row_index] += matrix_values[i];
    }

    return rows_sum;
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_columns_sum() const
{
#ifdef __OPENNN_DEBUG__

    if(columns_number == 0)
    {
        ostringstream buffer;

        cout << "OpenNN Exception: SparseMatrix Template.\n"
             << "Vector<T> calculate_columns_sum() const method.\n"
             << "SparseMatrix is empty.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<T> columns_sum(columns_number, T());

    const size_t columns_indices_size = columns_indices.size();

    for(size_t i = 0; i < columns_indices_size; i++)
    {
        const size_t current_column_index = columns_indices[i];

        columns_sum[current_column_index] += matrix_values[i];
    }

    return columns_sum;
}

template <class T>
Vector<size_t> SparseMatrix<T>::calculate_most_frequent_columns_indices(const size_t& top_number)
{
    const Vector<T> columns_sum = calculate_columns_sum();

    return columns_sum.calculate_maximal_indices(min(columns_number,top_number));
}

template <class T>
void SparseMatrix<T>::sum_row(const size_t& row_index, const Vector<T>& vector)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(vector.size() != columns_number)
    {
        ostringstream buffer;

        cout << "OpenNN Exception: SparseMatrix Template.\n"
             << "void sum_row(const size_t&, const Vector<T>&) method.\n"
             << "Size of vector must be equal to number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Vector<T> current_row = get_row(row_index) + vector;

    set_row(row_index, current_row);
}

template <class T>
double SparseMatrix<T>::calculate_trace() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(!is_square())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "double calculate_trace() const method.\n"
               << "SparseMatrix is not square.\n";

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

template <class T>
Vector<double> SparseMatrix<T>::calculate_mean() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_mean() const method.\n"
               << "Number of rows must be greater than one.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Mean

    Vector<double> mean(columns_number, 0.0);

    mean = calculate_columns_sum();

    mean /= (double)rows_number;

    return(mean);
}

template <class T>
double SparseMatrix<T>::calculate_mean(const size_t& column_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "double calculate_mean(const size_t&) const method.\n"
               << "Number of rows must be greater than one.\n";

        throw logic_error(buffer.str());
    }

    if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "double calculate_mean(const size_t&) const method.\n"
               << "Index of column must be less than number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Mean

    const Vector<size_t> current_column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

    const size_t current_column_nonzero_number = current_column_nonzero_indices.size();

    double mean = 0.0;

    for(size_t i = 0; i < current_column_nonzero_number; i++)
    {
        mean += matrix_values[current_column_nonzero_indices[i]];
    }

    mean /= (double)rows_number;

    return(mean);
}

template <class T>
Vector<double> SparseMatrix<T>::calculate_mean(const Vector<size_t>& means_column_indices) const
{
    const size_t means_column_size = means_column_indices.size();

    Vector<double> mean(means_column_size, 0.0);

    for(size_t i = 0; i < means_column_size; i++)
    {
        mean = calculate_mean(means_column_indices[i]);
    }

    return mean;
}

template <class T>
Vector<double> SparseMatrix<T>::calculate_mean(const Vector<size_t>& means_row_indices, const Vector<size_t>& means_column_indices) const
{
    const size_t row_indices_size = means_row_indices.size();
    const size_t column_indices_size = means_column_indices.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    // Rows check

    if(row_indices_size > rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
               << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < row_indices_size; i++)
    {
        if(means_row_indices[i] >= rows_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
                   << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                   << "Row index " << i << " must be less than rows number.\n";

            throw logic_error(buffer.str());
        }
    }

    if(row_indices_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
               << "Size of row indices must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    // Columns check

    if(column_indices_size > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_mean(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
               << "Column indices size must be equal or less than columns number.\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if(means_column_indices[i] >= columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
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
        column_index = means_column_indices[j];

        const Vector<size_t> current_column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

        const size_t current_column_nonzero_number = current_column_nonzero_indices.size();

        for(size_t i = 0; i < current_column_nonzero_number; i++)
        {
            row_index = rows_indices[current_column_nonzero_indices[i]];

            if(means_row_indices.contains(row_index))
            {
                mean[j] += matrix_values[current_column_nonzero_indices[i]];
            }
        }

        mean[j] /= (double)row_indices_size;
    }

    return(mean);
}

template <class T>
Vector<double> SparseMatrix<T>::calculate_mean_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<size_t> row_indices(0, 1, rows_number-1);
    Vector<size_t> column_indices(0, 1, columns_number-1);

    return(calculate_mean_missing_values(row_indices, column_indices, missing_indices));
}

template <class T>
Vector<double> SparseMatrix<T>::calculate_mean_missing_values(const Vector<size_t>& means_row_indices, const Vector<size_t>& means_column_indices,
                                                              const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t column_indices_size = means_column_indices.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t row_indices_size = means_row_indices.size();

    // Rows check

    if(row_indices_size > rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
               << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < row_indices_size; i++)
    {
        if(means_row_indices[i] >= rows_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
                   << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, Vector< Vector<size_t> >&) const method.\n"
                   << "Row index " << i << " must be less than rows number.\n";

            throw logic_error(buffer.str());
        }
    }

    if(row_indices_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
               << "Size of row indices must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    // Columns check

    if(column_indices_size > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
               << "Column indices size must be equal or less than columns number.\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if(means_column_indices[i] >= columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
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

    for(int i = 0; i < (int)column_indices_size; i++)
    {
        used_rows[i] = means_row_indices.get_difference(missing_indices[i]);
    }

#pragma omp parallel for schedule(dynamic)

    for(int j = 0; j < (int)column_indices_size; j++)
    {
        const size_t column_index = means_column_indices[j];

        const size_t current_rows_number = used_rows[j].size();

        for(size_t i = 0; i < current_rows_number; i++)
        {
            const size_t row_index = used_rows[j][i];

            mean[j] += (*this)(row_index,column_index);
        }

        if(current_rows_number != 0)
        {
            mean[j] /= (double)current_rows_number;
        }
    }

    return(mean);
}

template <class T>
Vector< Vector<double> > SparseMatrix<T>::calculate_mean_standard_deviation() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
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
        const Vector<T> current_column = get_column(i);
        mean[i] = current_column.calculate_mean();
        standard_deviation[i] = current_column.calculate_standard_deviation();

    }

    return {mean, standard_deviation};
}

template <class T>
Vector< Vector<double> > SparseMatrix<T>::calculate_mean_standard_deviation(const Vector<size_t>& column_indices) const
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

template <class T>
Vector< Vector<double> > SparseMatrix<T>::calculate_mean_standard_deviation(const Vector<size_t>& row_indices, const Vector<size_t>& column_indices) const
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

        column = column.get_subvector(row_indices);

        mean[i] = column.calculate_mean();
        standard_deviation[i] = column.calculate_standard_deviation();
    }

    return {mean, standard_deviation};
}

template <class T>
Vector<double> SparseMatrix<T>::calculate_median() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(columns_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_median() const method.\n"
               << "Number of columns must be greater than one.\n";

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

template <class T>
double SparseMatrix<T>::calculate_median(const size_t& column_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "double calculate_median(const size_t&) const method.\n"
               << "Number of rows must be greater than one.\n";

        throw logic_error(buffer.str());
    }

    if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
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

template <class T>
Vector<double> SparseMatrix<T>::calculate_median(const Vector<size_t>& median_column_indices) const
{
    const size_t column_indices_size = median_column_indices.size();

    size_t column_index;

    // median

    Vector<double> median(column_indices_size, 0.0);

    for(size_t j = 0; j < column_indices_size; j++)
    {
        column_index = median_column_indices[j];

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

template <class T>
Vector<double> SparseMatrix<T>::calculate_median(const Vector<size_t>& median_row_indices, const Vector<size_t>& median_column_indices) const
{
    const size_t row_indices_size = median_row_indices.size();
    const size_t column_indices_size = median_column_indices.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    // Rows check

    if(row_indices_size > rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
               << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < row_indices_size; i++)
    {
        if(median_row_indices[i] >= rows_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
                   << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                   << "Row index " << i << " must be less than rows number.\n";

            throw logic_error(buffer.str());
        }
    }

    if(row_indices_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
               << "Size of row indices must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    // Columns check

    if(column_indices_size > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
               << "Column indices size must be equal or less than columns number.\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if(median_column_indices[i] >= columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
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
        column_index = median_column_indices[j];

        Vector<T> sorted_column(this->get_column(column_index));

        sorted_column = sorted_column.get_subvector(median_row_indices);

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

template <class T>
Vector<double> SparseMatrix<T>::calculate_median_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<size_t> row_indices(0, 1, rows_number-1);
    Vector<size_t> column_indices(0, 1, columns_number-1);

    return(calculate_median_missing_values(row_indices, column_indices, missing_indices));
}

template <class T>
Vector<double> SparseMatrix<T>::calculate_median_missing_values(const Vector<size_t>& median_row_indices,
                                                                const Vector<size_t>& median_column_indices,
                                                                const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t column_indices_size = median_column_indices.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t row_indices_size = median_row_indices.size();

    // Rows check

    if(row_indices_size > rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
               << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < row_indices_size; i++)
    {
        if(median_row_indices[i] >= rows_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
                   << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, Vector< Vector<size_t> >&) const method.\n"
                   << "Row index " << i << " must be less than rows number.\n";

            throw logic_error(buffer.str());
        }
    }

    if(row_indices_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
               << "Size of row indices must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    // Columns check

    if(column_indices_size > columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector<double> calculate_median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector< Vector<size_t> >&) const method.\n"
               << "Column indices size must be equal or less than columns number.\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if(median_column_indices[i] >= columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template.\n"
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

    for(int i = 0; i < (int)column_indices_size; i++)
    {
        used_rows[i] = median_row_indices.get_difference(missing_indices[i]);
    }

#pragma omp parallel for schedule(dynamic)

    for(int j = 0; j < (int)column_indices_size; j++)
    {
        const size_t column_index = median_column_indices[j];

        const size_t current_rows_number = used_rows[j].size();

        Vector<T> sorted_column(this->get_column(column_index));

        sorted_column = sorted_column.get_subvector(used_rows[j]);

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

template <class T>
T SparseMatrix<T>::calculate_minimum() const
{
    return min(T(), matrix_values.calculate_minimum());
}

template <class T>
T SparseMatrix<T>::calculate_maximum() const
{
    return min(T(), matrix_values.calculate_maximum());
}

template <class T>
T SparseMatrix<T>::calculate_column_minimum(const size_t& column_index) const
{
    return get_column(column_index).calculate_minimum();
}

template <class T>
T SparseMatrix<T>::calculate_column_maximum(const size_t& column_index) const
{
    return get_column(column_index).calculate_maximum();
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_means_binary() const
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

template <class T>
Vector<T> SparseMatrix<T>::calculate_means_binary_column() const
{
    Vector<T> means(2,0.0);

    size_t count = 0;

    Vector<T> first_column = get_column(0);
    Vector<T> second_column = get_column(1);

    for(size_t i = 0; i < rows_number; i++)
    {
        if(first_column[i] == 0.0)
        {
            means[0] += second_column[i];
            count++;
        }
        else if(first_column[i] == 1.0)
        {
            means[1] += second_column[i];
            count++;
        }
    }

    if(count != 0)
    {
        means[0] = (T)means[0]/(T)count;
        means[1] = (T)means[1]/(T)count;
    }
    else
    {
        means[0] = 0.0;
        means[1] = 0.0;
    }

    return means;
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_means_binary_columns() const
{
    Vector<T> means(columns_number-1);

    T sum = 0.0;
    size_t count = 0;

    const Vector<T> last_column = get_column(columns_number-1);

    for(size_t i = 0; i < columns_number-1; i++)
    {
        sum = 0.0;
        count = 0;

        const Vector<T> current_column = get_column(i);

        for(size_t j = 0; j < rows_number; j++)
        {
            if(current_column[j] == 1.0)
            {
                sum += last_column[j];
                count++;
            }
        }

        if(count != 0)
        {
            means[i] = (T)sum/(T)count;
        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_means_binary_missing_values(const Vector< Vector<size_t> >& missing_indices) const
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

template <class T>
Vector<T> SparseMatrix<T>::calculate_means_binary_column_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<T> means(2,0.0);

    size_t count = 0;

    Vector<T> first_column = get_column(0);
    Vector<T> second_column = get_column(1);

    for(size_t i = 0; i < rows_number; i++)
    {
        if(!missing_indices[0].contains(i))
        {
            if(first_column[i] == 0.0)
            {
                means[0] += second_column[i];
                count++;
            }
            else if(first_column[i] == 1.0)
            {
                means[1] += second_column[i];
                count++;
            }
        }
    }

    if(count != 0)
    {
        means[0] = (T)means[0]/(T)count;
        means[1] = (T)means[1]/(T)count;
    }
    else
    {
        means[0] = 0.0;
        means[1] = 0.0;
    }

    return means;
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_means_binary_columns_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    Vector<T> means(columns_number-1);

    T sum = 0.0;
    size_t count = 0;

    const Vector<T> last_column = get_column(columns_number-1);

    for(size_t i = 0; i < columns_number-1; i++)
    {
        sum = 0.0;
        count = 0;

        const Vector<T> current_column = get_column(i);

        for(size_t j = 0; j < rows_number; j++)
        {
            if(current_column[j] == 1.0 && !missing_indices[i].contains(j))
            {
                sum += last_column[j];
                count++;
            }
        }

        if(count != 0)
        {
            means[i] = (T)sum/(T)count;
        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;
}

template <class T>
Vector< Vector<T> > SparseMatrix<T>::calculate_minimum_maximum() const
{
    Vector<T> minimum(columns_number,(T)numeric_limits<double>::max());
    Vector<T> maximum(columns_number,(T)-numeric_limits<double>::max());

    for(size_t j = 0; j < columns_number; j++)
    {
        const Vector<size_t> current_column_nonzero_indices = columns_indices.calculate_equal_to_indices(j);

        const Vector<T> current_column_nonzero_values = matrix_values.get_subvector(current_column_nonzero_indices);

        Vector<T> current_minimum_maximum = current_column_nonzero_values.calculate_minimum_maximum();

        minimum[j] = min(T(),current_minimum_maximum[0]);
        maximum[j] = max(T(),current_minimum_maximum[1]);
    }

    return {minimum, maximum};
}


template <class T>
Vector< Vector<T> > SparseMatrix<T>::calculate_minimum_maximum(const Vector<size_t>& calculate_column_indices) const
{
    const size_t column_indices_size = calculate_column_indices.size();

#ifdef __OPENNN_DEBUG__

    for(size_t i = 0; i < column_indices_size; i++)
    {
        if(calculate_column_indices[i] >= columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: SparseMatrix template."
                   << "Vector<T> calculate_minimum_maximum(const Vector<size_t>&) const method.\n"
                   << "Index of column must be less than number of columns.\n";

            throw logic_error(buffer.str());
        }
    }

#endif

    Vector< Vector<T> > minimum_maximum(2);

    Vector<T> minimum(column_indices_size,(T)numeric_limits<double>::max());
    Vector<T> maximum(column_indices_size,(T)-numeric_limits<double>::max());

    for(size_t j = 0; j < column_indices_size; j++)
    {
        const size_t column_index = calculate_column_indices[j];

        const Vector<size_t> current_column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

        const Vector<T> current_column_nonzero_values = matrix_values.get_subvector(current_column_nonzero_indices);

        Vector<T> current_minimum_maximum = current_column_nonzero_values.calculate_minimum_maximum();

        minimum[j] = min(T(),current_minimum_maximum[0]);
        maximum[j] = max(T(),current_minimum_maximum[1]);
    }

    return {minimum, maximum};
}


template <class T>
Vector< Vector<T> > SparseMatrix<T>::calculate_minimum_maximum(const Vector<size_t>& calculate_row_indices, const Vector<size_t>& calculate_column_indices) const
{
    const size_t row_indices_size = calculate_row_indices.size();
    const size_t column_indices_size = calculate_column_indices.size();

    Vector<T> minimum(column_indices_size,(T) numeric_limits<double>::max());
    Vector<T> maximum(column_indices_size,(T)-numeric_limits<double>::max());

    size_t row_index;
    size_t column_index;

    for(size_t j = 0; j < column_indices_size; j++)
    {
        column_index = calculate_column_indices[j];

        const Vector<T> current_column = get_column(j);

        for(size_t i = 0; i < row_indices_size; i++)
        {
            row_index = calculate_row_indices[i];

            if(current_column[row_index] < minimum[j])
            {
                minimum[j] = current_column[row_index];
            }

            if(current_column[row_index] > maximum[j])
            {
                maximum[j] = current_column[row_index];
            }
        }
    }

    return {minimum, maximum};
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_statistics() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector< Statistics<T> > calculate_statistics() const method.\n"
               << "Number of rows must be greater than one.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector< Statistics<T> > statistics(columns_number);

    Vector<T> column(rows_number);

#pragma omp parallel for private(column)

    for(int i = 0; i < (int)columns_number; i++)
    {
        column = get_column(i);

        statistics[i] = column.calculate_statistics();
    }

    return(statistics);
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_statistics(const Vector<size_t>& calculate_row_indices, const Vector<size_t>& calculate_column_indices) const
{
    const size_t column_indices_size = calculate_column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    Vector<size_t> unused_rows(0,1,rows_number);

    unused_rows = unused_rows.get_difference(calculate_row_indices);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        const Vector<T> column = get_column(calculate_column_indices[i]);

        statistics[i] = column.calculate_statistics_missing_values(unused_rows);
    }

    return statistics;
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_statistics(const Vector< Vector<size_t> >& calculate_row_indices, const Vector<size_t>& calculate_column_indices) const
{
    const size_t column_indices_size = calculate_column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    Vector<size_t> sequential_row_indices(0,1,rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        const Vector<T> column = get_column(calculate_column_indices[i]);

        const Vector<size_t> current_unused_rows = sequential_row_indices.get_difference(calculate_row_indices[i]);

        statistics[i] = column.calculate_statistics_missing_values(current_unused_rows);
    }

    return statistics;
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_statistics_missing_values(const Vector< Vector<size_t> >& missing_indices) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "Vector< Statistics<double> > calculate_statistics_missing_values(const Vector< Vector<size_t> >&) const method.\n"
               << "Number of rows must be greater than one.\n";

        throw logic_error(buffer.str());
    }

    if(missing_indices.size() != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
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

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_columns_statistics_missing_values(const Vector<size_t>& calculate_column_indices,
                                                                                     const Vector< Vector<size_t> > missing_indices) const
{
    const size_t column_indices_size = calculate_column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

#pragma omp parallel for private(index, column) schedule(dynamic)

    for(int i = 0; i < (int)column_indices_size; i++)
    {
        index = calculate_column_indices[i];

        column = get_column(index);

        statistics[i] = column.calculate_statistics_missing_values(missing_indices[index]);
    }

    return statistics;
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_rows_statistics(const Vector<size_t>& calculate_row_indices) const
{
    const size_t row_indices_size = calculate_row_indices.size();

    Vector< Statistics<T> > statistics(row_indices_size);

    Vector<size_t> unused_rows(0,1,rows_number);

    unused_rows = unused_rows.get_difference(calculate_row_indices);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> column = get_column(i);

        statistics[i] = column.calculate_statistics_missing_values(unused_rows);
    }

    return statistics;
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_rows_statistics_missing_values(const Vector<size_t>& calculate_row_indices,
                                                                                  const Vector< Vector<size_t> >& missing_indices) const
{
    const size_t row_indices_size = calculate_row_indices.size();

    Vector< Statistics<T> > statistics(columns_number);

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i);

        column = column.get_subvector(calculate_row_indices);

        statistics[i] = column.calculate_statistics_missing_values(missing_indices[i]);
    }

    return statistics;
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::calculate_columns_statistics(const Vector<size_t>& calculate_column_indices) const
{
    const size_t column_indices_size = calculate_column_indices.size();

    Vector< Statistics<T> > statistics(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = calculate_column_indices[i];

        column = get_column(index);

        statistics[i] = column.calculate_statistics();
    }

    return statistics;
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_rows_means(const Vector<size_t>& calculate_row_indices) const
{
    Vector<size_t> used_row_indices;

    if(calculate_row_indices.empty())
    {
        used_row_indices.set(this->get_rows_number());
        used_row_indices.initialize_sequential();
    }
    else
    {
        used_row_indices = calculate_row_indices;
    }

    const size_t row_indices_size = used_row_indices.size();

    Vector<T> means(columns_number);

    Vector<T> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = get_column(i);

        column = column.get_subvector(used_row_indices);

        means[i] = column.calculate_mean();
    }

    return means;
}

template <class T>
Vector<T> SparseMatrix<T>::calculate_columns_minimums(const Vector<size_t>& calculate_column_indices) const
{
    Vector<size_t> used_column_indices;

    if(calculate_column_indices.empty())
    {
        used_column_indices.set(columns_number);
        used_column_indices.initialize_sequential();
    }
    else
    {
        used_column_indices = calculate_column_indices;
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
Vector<T> SparseMatrix<T>::calculate_columns_maximums(const Vector<size_t>& calculate_column_indices) const
{
    Vector<size_t> used_column_indices;

    if(calculate_column_indices.empty())
    {
        used_column_indices.set(columns_number);
        used_column_indices.initialize_sequential();
    }
    else
    {
        used_column_indices = calculate_column_indices;
    }

    const size_t column_indices_size = used_column_indices.size();

    Vector<T> minimums(column_indices_size);

    size_t index;
    Vector<T> column(rows_number);

    for(size_t i = 0; i < column_indices_size; i++)
    {
        index = used_column_indices[i];

        column = get_column(index);

        minimums[i] = column.calculate_maximum();
    }

    return minimums;
}

template <class T>
Vector< Vector<double> > SparseMatrix<T>::calculate_box_plots(const Vector<Vector<size_t> >& calculate_rows_indices,
                                                              const Vector<size_t>& calculate_columns_indices) const
{
    const size_t calculate_columns_number = calculate_columns_indices.size();

#ifdef __OPENNN_DEBUG__

    if(calculate_columns_number == calculate_rows_indices.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
               << "Vector< Vector<double> > calculate_box_plots(const Vector<Vector<size_t> >&, const Vector<size_t>&) const method.\n"
               << "Size of row indices must be equal to the number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector< Vector<double> > box_plots(calculate_columns_number);

    for(size_t i = 0; i < calculate_columns_number; i++)
    {
        const Vector<size_t> current_column = get_column(calculate_columns_indices[i]).get_subvector(calculate_rows_indices[i]);

        box_plots[i] = current_column.calculate_box_plot();
    }

    return box_plots;
}

template <class T>
SparseMatrix<double> SparseMatrix<T>::calculate_covariance_sparse_matrix() const
{
    const size_t size = columns_number;

#ifdef __OPENNN_DEBUG__

    if(size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
               << "SparseMatrix<double> calculate_covariance_sparse_matrix() const method.\n"
               << "Number of columns must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<double> covariance_sparse_matrix(size, size);

    Vector<T> first_column;
    Vector<T> second_column;

    for(size_t i = 0; i < size; i++)
    {
        first_column = get_column(i);

        if(first_column == T())
        {
            continue;
        }

        for(size_t j = i; j < size; j++)
        {
            second_column = get_column(j);

            if(second_column == T())
            {
                continue;
            }

            const double covariance = first_column.calculate_covariance(second_column);

            if(covariance != 0.0)
            {
                covariance_sparse_matrix.set_element(i,j,covariance);
                covariance_sparse_matrix.set_element(j,i,covariance);
            }
        }
    }

    return covariance_sparse_matrix;
}

template <class T>
Vector< Histogram<T> > SparseMatrix<T>::calculate_histograms(const size_t& bins_number) const
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

template <class T>
Vector< Histogram<T> > SparseMatrix<T>::calculate_histograms_missing_values(const Vector< Vector<size_t> >& missing_indices, const size_t& bins_number) const
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

template <class T>
void SparseMatrix<T>::scale_mean_standard_deviation(const Vector< Statistics<T> >& statistics)
{
#ifdef __OPENNN_DEBUG__

    const size_t size = statistics.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
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
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < rows_number; i++)
            {
                current_column[i] = (current_column[i] - statistics[j].mean)/statistics[j].standard_deviation;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::scale_mean_standard_deviation()
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_mean_standard_deviation(statistics);

    return(statistics);
}

template <class T>
void SparseMatrix<T>::scale_rows_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_row_indices)
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

    const size_t scale_row_number = scale_row_indices.size();

    // Scale columns

    for(size_t j = 0; j < columns_number; j++)
    {
        if(statistics[j].standard_deviation < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < scale_row_number; i++)
            {
                row_index = scale_row_indices[i];

                current_column[row_index] = (current_column[row_index] - statistics[j].mean)/statistics[j].standard_deviation;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::scale_columns_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_columns_indices)
{
    const size_t columns_indices_size = scale_columns_indices.size();

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
            column_index = scale_columns_indices[j];

            Vector<double> current_column = get_column(column_index);

#pragma omp parallel for

            for(int i = 0; i < static_cast<int>(rows_number); i++)
            {
                current_column[i] = (current_column[i] - statistics[j].mean)/statistics[j].standard_deviation;
            }

            set_column(column_index, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::scale_minimum_maximum(const Vector< Statistics<T> >& statistics)
{
#ifdef __OPENNN_DEBUG__

    const size_t size = statistics.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
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
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < rows_number; i++)
            {
                current_column[i] = 2.0*(current_column[i] - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)-1.0;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::scale_minimum_maximum()
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_minimum_maximum(statistics);

    return(statistics);
}

template <class T>
void SparseMatrix<T>::scale_rows_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_row_indices)
{
    // Control sentence(if debug)

    const size_t row_indices_size = scale_row_indices.size();

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
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < row_indices_size; i++)
            {
                row_index = scale_row_indices[i];

                current_column[row_index] = 2.0*(current_column[row_index] - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum) - 1.0;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::scale_columns_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_columns_indices)
{
    const size_t columns_indices_size = scale_columns_indices.size();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_indices_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector template.\n"
               << "void scale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
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
            column_index = scale_columns_indices[j];

            Vector<double> current_column = get_column(column_index);

#pragma omp parallel for

            for(int i = 0; i < static_cast<int>(rows_number); i++)
            {
                current_column[i] = (current_column[i] - statistics[j].mean)/statistics[j].standard_deviation;
            }

            set_column(column_index, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::scale_logarithmic(const Vector< Statistics<T> >& statistics)
{
#ifdef __OPENNN_DEBUG__

    const size_t size = statistics.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
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
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < rows_number; i++)
            {
                current_column[i] = log(1.0+ (2.0*(current_column[i] - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)));
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
Vector< Statistics<T> > SparseMatrix<T>::scale_logarithmic()
{
    const Vector< Statistics<T> > statistics = calculate_statistics();

    scale_logarithmic(statistics);

    return(statistics);
}

template <class T>
void SparseMatrix<T>::scale_rows_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_row_indices)
{
    // Control sentence(if debug)

    const size_t row_indices_size = scale_row_indices.size();

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
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < row_indices_size; i++)
            {
                row_index = scale_row_indices[i];

                current_column[row_index] = log(1.0+ (2.0*(current_column[row_index] - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)));
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::scale_columns_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_column_indices)
{
    // Control sentence(if debug)

    const size_t column_indices_size = scale_column_indices.size();

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
        column_index = scale_column_indices[j];

        if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(column_index);

#pragma omp parallel for

            for(int i = 0; i < static_cast<int>(rows_number); i++)
            {
                current_column[i] = log(1.0+ (2.0*(current_column[i] - statistics[j].minimum)/(statistics[j].maximum-statistics[j].minimum)));
            }

            set_column(column_index, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_mean_standard_deviation(const Vector< Statistics<T> >& statistics)
{
#ifdef __OPENNN_DEBUG__

    const size_t size = statistics.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
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
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < rows_number; i++)
            {
                current_column[i] = current_column[i]*statistics[j].standard_deviation + statistics[j].mean;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_rows_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_row_indices)
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

    const size_t scale_row_number = scale_row_indices.size();

    // Scale columns

    for(size_t j = 0; j < columns_number; j++)
    {
        if(statistics[j].standard_deviation < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < scale_row_number; i++)
            {
                row_index = scale_row_indices[i];

                current_column[row_index] = current_column[row_index]*statistics[j].standard_deviation + statistics[j].mean;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_column_indices)
{
    // Control sentence(if debug)

    const size_t column_indices_size = scale_column_indices.size();

#ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != column_indices_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector template.\n"
               << "void unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
               << "Size of statistics must be equal to size of columns indices.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t column_index;

    // Rescale data

    for(size_t j = 0; j < column_indices_size; j++)
    {
        column_index = scale_column_indices[j];

        if(statistics[j].standard_deviation < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(column_index);

#pragma omp parallel for

            for(int i = 0; i < static_cast<int>(rows_number); i++)
            {
                current_column[i] = current_column[i]*statistics[j].standard_deviation + statistics[j].mean;
            }

            set_column(column_index, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_minimum_maximum(const Vector< Statistics<T> >& statistics)
{
#ifdef __OPENNN_DEBUG__

    const size_t size = statistics.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
               << "void unscale_minimum_maximum(const Vector< Statistics<T> >&) const method.\n"
               << "Size of statistics vector must be equal to number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t j = 0; j < columns_number; j++)
    {
        if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < rows_number; i++)
            {
                current_column[i] = 0.5*(current_column[i] + 1.0)*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_rows_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_row_indices)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector template.\n"
               << "void unscale_rows_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
               << "Size of statistics must be equal to number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t row_index;

    const size_t scale_row_number = scale_row_indices.size();

    // Scale columns

    for(size_t j = 0; j < columns_number; j++)
    {
        if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < scale_row_number; i++)
            {
                row_index = scale_row_indices[i];

                current_column[row_index] = 0.5*(current_column[row_index] + 1.0)*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_columns_minimum_maximum(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_column_indices)
{
    // Control sentence(if debug)

    const size_t column_indices_size = scale_column_indices.size();

#ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != column_indices_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector template.\n"
               << "void unscale_columns_minimum_maximum(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
               << "Size of statistics must be equal to size of columns indices.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t column_index;

    // Rescale data

    for(size_t j = 0; j < column_indices_size; j++)
    {
        column_index = scale_column_indices[j];

        if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(column_index);

#pragma omp parallel for

            for(int i = 0; i < static_cast<int>(rows_number); i++)
            {
                current_column[i] = 0.5*(current_column[i] + 1.0)*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
            }

            set_column(column_index, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_logarithmic(const Vector< Statistics<T> >& statistics)
{
#ifdef __OPENNN_DEBUG__

    const size_t size = statistics.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template."
               << "void unscale_logarithmic(const Vector< Statistics<T> >&) const method.\n"
               << "Size of statistics vector must be equal to number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t j = 0; j < columns_number; j++)
    {
        if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < rows_number; i++)
            {
                current_column[i] = 0.5*(exp(current_column[i]))*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_rows_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_row_indices)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector template.\n"
               << "void unscale_rows_logarithmic(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
               << "Size of statistics must be equal to number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t row_index;

    const size_t scale_row_number = scale_row_indices.size();

    // Scale columns

    for(size_t j = 0; j < columns_number; j++)
    {
        if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(j);

            for(size_t i = 0; i < scale_row_number; i++)
            {
                row_index = scale_row_indices[i];

                current_column[row_index] = 0.5*(exp(current_column[row_index]))*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
            }

            set_column(j, current_column);
        }
    }
}

template <class T>
void SparseMatrix<T>::unscale_columns_logarithmic(const Vector< Statistics<T> >& statistics, const Vector<size_t>& scale_column_indices)
{
    // Control sentence(if debug)

    const size_t column_indices_size = scale_column_indices.size();

#ifdef __OPENNN_DEBUG__

    const size_t statistics_size = statistics.size();

    if(statistics_size != column_indices_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector template.\n"
               << "void unscale_columns_mean_standard_deviation(const Vector< Statistics<T> >&, const Vector<size_t>&) method.\n"
               << "Size of statistics must be equal to size of columns indices.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t column_index;

    // Rescale data

    for(size_t j = 0; j < column_indices_size; j++)
    {
        column_index = scale_column_indices[j];

        if(statistics[j].maximum - statistics[j].minimum < numeric_limits<double>::min())
        {
            // Do nothing
        }
        else
        {
            Vector<double> current_column = get_column(column_index);

#pragma omp parallel for

            for(int i = 0; i < static_cast<int>(rows_number); i++)
            {
                current_column[i] = 0.5*(exp(current_column[i]))*(statistics[j].maximum-statistics[j].minimum) + statistics[j].minimum;
            }

            set_column(column_index, current_column);
        }
    }
}

template <class T>
Vector<size_t> SparseMatrix<T>::calculate_minimal_indices() const
{
    T minimum = calculate_minimum();

    Vector<size_t> minimal_indices(2,0);

    if(minimum != T())
    {
        const size_t minimum_index = matrix_values.calculate_equal_to_indices(minimum)[0];

        minimal_indices[0] = rows_indices[minimum_index];
        minimal_indices[1] = columns_indices[minimum_index];
    }
    else
    {
        for(size_t i = 0; i < rows_number; i++)
        {
            const Vector<size_t> current_row = get_row(i);

            if(current_row.contains(minimum))
            {
                minimal_indices[0] = i;
                minimal_indices[1] = current_row.calculate_equal_to_indices(minimum);

                break;
            }
        }
    }

    return minimal_indices;
}

template <class T>
Vector<size_t> SparseMatrix<T>::calculate_maximal_indices() const
{
    T maximum = calculate_maximum();

    Vector<size_t> maximal_indices(2,0);

    if(maximum != T())
    {
        const size_t maximum_index = matrix_values.calculate_equal_to_indices(maximum)[0];

        maximal_indices[0] = rows_indices[maximum_index];
        maximal_indices[1] = columns_indices[maximum_index];
    }
    else
    {
        for(size_t i = 0; i < rows_number; i++)
        {
            const Vector<size_t> current_row = get_row(i);

            if(current_row.contains(maximum))
            {
                maximal_indices[0] = i;
                maximal_indices[1] = current_row.calculate_equal_to_indices(maximum);

                break;
            }
        }
    }

    return maximal_indices;
}

template <class T>
Vector< Vector<size_t> > SparseMatrix<T>::calculate_minimal_maximal_indices() const
{
    const Vector<size_t> minimal_indices = calculate_minimal_indices();
    const Vector<size_t> maximal_indices = calculate_maximal_indices();

    Vector< Vector<size_t> > minimal_maximal_indices(2);
    minimal_maximal_indices[0] = minimal_indices;
    minimal_maximal_indices[1] = maximal_indices;

    return(minimal_maximal_indices);
}


template <class T>
double SparseMatrix<T>::calculate_sum_squared_error(const SparseMatrix<double>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_sum_squared_error(const SparseMatrix<double>&) const method.\n"
               << "Other number of rows must be equal to this number of rows.\n";

        throw logic_error(buffer.str());
    }

    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_sum_squared_error(const SparseMatrix<double>&) const method.\n"
               << "Other number of columns must be equal to this number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    double sum_squared_error = 0.0;

    //#pragma omp parallel for reduction(+:sum_squared_error)

    for(int i = 0; i < columns_number; i++)
    {
        const Vector<T> this_current_column = this->get_column(i);
        const Vector<T> other_current_column = other_sparse_matrix.get_column(i);

        sum_squared_error += this_current_column.calculate_sum_squared_error(other_current_column);
    }

    return(sum_squared_error);
}

template <class T>
double SparseMatrix<T>::calculate_sum_squared_error(const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
               << "Other number of rows must be equal to this number of rows.\n";

        throw logic_error(buffer.str());
    }

    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_sum_squared_error(const Matrix<double>&) const method.\n"
               << "Other number of columns must be equal to this number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    double sum_squared_error = 0.0;

    //#pragma omp parallel for reduction(+:sum_squared_error)

    for(int i = 0; i < columns_number; i++)
    {
        const Vector<T> this_current_column = this->get_column(i);
        const Vector<T> other_current_column = other_matrix.get_column(i);

        sum_squared_error += this_current_column.calculate_sum_squared_error(other_current_column);
    }

    return(sum_squared_error);
}

template <class T>
double SparseMatrix<T>::calculate_minkowski_error(const SparseMatrix<double>& other_sparse_matrix, const double& minkowski_parameter) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_minkowski_error(const SparseMatrix<double>&, const double&) const method.\n"
               << "Other number of rows must be equal to this number of rows.\n";

        throw logic_error(buffer.str());
    }

    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_minkowski_error(const SparseMatrix<double>&, const double&) const method.\n"
               << "Other number of columns must be equal to this number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    double minkowski_error = 0.0;

    //#pragma omp parallel for reduction(+:minkowski_error)

    for(int i = 0; i < columns_number; i++)
    {
        const Vector<T> this_current_column = this->get_column(i);
        const Vector<T> other_current_column = other_sparse_matrix.get_column(i);

        minkowski_error += this_current_column.calculate_Minkowski_error(other_current_column, minkowski_parameter);
    }

    return(minkowski_error);
}

template <class T>
double SparseMatrix<T>::calculate_minkowski_error(const Matrix<T>& other_matrix, const double& minkowski_parameter) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_minkowski_error(const Matrix<double>&, const double&) const method.\n"
               << "Other number of rows must be equal to this number of rows.\n";

        throw logic_error(buffer.str());
    }

    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_minkowski_error(const Matrix<double>&, const double&) const method.\n"
               << "Other number of columns must be equal to this number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    double minkowski_error = 0.0;

    //#pragma omp parallel for reduction(+:minkowski_error)

    for(int i = 0; i < columns_number; i++)
    {
        const Vector<T> this_current_column = this->get_column(i);
        const Vector<T> other_current_column = other_matrix.get_column(i);

        minkowski_error += this_current_column.calculate_Minkowski_error(other_current_column, minkowski_parameter);
    }

    return(minkowski_error);
}

template <class T>
double SparseMatrix<T>::calculate_sum_squared_error(const Vector<double>& vector) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "double calculate_sum_squared_error(const Vector<double>&) const method.\n"
               << "Size must be equal to number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    double sum_squared_error = 0.0;

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i);

        sum_squared_error += current_row.calculate_sum_squared_error(vector);
    }

    return(sum_squared_error);
}

template <class T>
Vector<double> SparseMatrix<T>::calculate_rows_norm() const
{
    Vector<double> rows_norm(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i);

        rows_norm[i] = current_row*current_row;

        rows_norm[i] = sqrt(rows_norm[i]);
    }

    return(rows_norm);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::calculate_absolute_value() const
{
    SparseMatrix<T> absolute_value(rows_number,columns_number);

    const Vector<T> absolute_matrix_values = matrix_values.calculate_absolute_value();

    absolute_value.set_values(rows_indices, columns_indices, absolute_matrix_values);

    return absolute_value;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::calculate_transpose() const
{
    SparseMatrix<T> transpose(columns_number,rows_number);

    transpose.set_values(columns_indices, rows_indices, matrix_values);

    return transpose;
}

template <class T>
T SparseMatrix<T>::calculate_determinant() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_determinant() const method.\n"
               << "Sparse matrix is empty.\n";

        throw logic_error(buffer.str());
    }

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_determinant() const method.\n"
               << "Sparse matrix must be square.\n";

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
        const Vector<T> first_row = get_row(0);
        const Vector<T> second_row = get_row(1);

        determinant = first_row[0]*second_row[1] - second_row[0]*first_row[1];
    }
    else
    {
        int sign;

        const Vector<T> first_row = get_row(0);

        for(size_t column_index = 0; column_index < columns_number; column_index++)
        {
            // Calculate sub data

            SparseMatrix<T> sub_sparse_matrix(rows_number-1, columns_number-1);

            Vector<size_t> sub_rows_indices = rows_indices - 1;
            Vector<size_t> sub_columns_indices = columns_indices;
            Vector<T> sub_matrix_values = matrix_values;

            const Vector<size_t> current_column_indices = columns_indices.calculate_equal_to_indices(column_index);

            sub_rows_indices = sub_rows_indices.delete_indices(current_column_indices);
            sub_columns_indices = sub_columns_indices.delete_indices(current_column_indices);
            sub_matrix_values = sub_matrix_values.delete_indices(current_column_indices);

            for(size_t i = 0; i < sub_columns_indices.size(); i++)
            {
                if(sub_columns_indices[i] > column_index)
                {
                    sub_columns_indices[i]--;
                }
            }

            sub_sparse_matrix.set_values(sub_rows_indices, sub_columns_indices, sub_matrix_values);
            //sign = (size_t)(pow(-1.0, row_index+2.0));

            sign = static_cast<int>((((column_index + 2) % 2) == 0) ? 1 : -1 );

            determinant += sign*first_row[column_index]*sub_sparse_matrix.calculate_determinant();
        }
    }

    return(determinant);
}


template <class T>
SparseMatrix<T> SparseMatrix<T>::calculate_cofactor() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_cofactor() const method.\n"
               << "Sparse matrix is empty.\n";

        throw logic_error(buffer.str());
    }

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_cofactor() const method.\n"
               << "Sparse matrix must be square.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> cofactor(rows_number, columns_number);

    SparseMatrix<T> c;

    const Vector<size_t> sequential_row_indices(0,1,rows_number-1);
    const Vector<size_t> sequential_column_indices(0,1,columns_number-1);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            const Vector<size_t> this_row_indices = sequential_row_indices.delete_index(i);
            const Vector<size_t> this_column_indices = sequential_column_indices.delete_index(j);

            c = this->get_sub_sparse_matrix(this_row_indices, this_column_indices);

            const double determinant = c.calculate_determinant();

            const T value = static_cast<T>((((i + j) % 2) == 0) ? 1 : -1)*determinant;

            cofactor.set_element(i,j,value);
            //cofactor(i,j) = pow(-1.0, i+j+2.0)*determinant;
        }
    }

    return(cofactor);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::calculate_inverse() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_inverse() const method.\n"
               << "Sparse matrix is empty.\n";

        throw logic_error(buffer.str());
    }

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_inverse() const method.\n"
               << "Sparse matrix must be square.\n";

        throw logic_error(buffer.str());
    }

#endif

    const double determinant = calculate_determinant();

    if(determinant == 0.0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_inverse() const method.\n"
               << "Sparse matrix is singular.\n";

        throw logic_error(buffer.str());
    }

    if(rows_number == 1)
    {
        SparseMatrix<T> inverse(1, 1);

        inverse.set_element(0,0, 1.0/determinant);

        return(inverse);
    }

    // Calculate cofactor SparseMatrix

    const SparseMatrix<T> cofactor = calculate_cofactor();

    // Adjoint SparseMatrix is the transpose of cofactor SparseMatrix

    const SparseMatrix<T> adjoint = cofactor.calculate_transpose();

    // Inverse SparseMatrix is adjoint SparseMatrix divided by SparseMatrix determinant

    const SparseMatrix<T> inverse = adjoint/determinant;

    return(inverse);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::calculate_LU_inverse() const /// @todo
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_LU_inverse() const method.\n"
               << "Sparse matrix is empty.\n";

        throw logic_error(buffer.str());
    }

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_LU_inverse() const method.\n"
               << "Sparse matrix must be square.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> inverse(rows_number, columns_number);

    //    Eigen::Map<Eigen::SparseMatrix<double> > sm1(rows,cols,nnz,outerIndexPtr,innerIndices,values);
    //    const Eigen::Map<Eigen::SparseMatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    //    Eigen::Map<Eigen::SparseMatrixXd> inverse_eigen(inverse.data(), rows_number, columns_number);

    //    inverse_eigen = this_eigen.inverse();

    return(inverse);
}


/// @todo

template <class T>
Vector<T> SparseMatrix<T>::solve_LDLT(const Vector<double>&) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "solve_LLT(const Vector<double>&) const method.\n"
               << "Sparse matrix is empty.\n";

        throw logic_error(buffer.str());
    }

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "solve_LLT(const Vector<double>&) const method.\n"
               << "Sparse matrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    //Vector<T> solution(rows_number);
    //const Eigen::Map<Eigen::SparseMatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    //const Eigen::Map<Eigen::VectorXd> b_eigen((double*)b.data(),rows_number);
    //Eigen::Map<Eigen::VectorXd> solution_eigen(solution.data(), rows_number);

    //    solution_eigen = this_eigen.ldlt().solve(b_eigen);

    return Vector<T>();
}

template <class T>
double SparseMatrix<T>::calculate_distance(const size_t& first_index, const size_t& second_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "calculate_distance(const size_t&, const size_t&) const method.\n"
               << "SparseMatrix is empty.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Vector<T> first_row = get_row(first_index);
    const Vector<T> second_row = get_row(second_index);

    return(first_row.calculate_distance(second_row));
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator + (const T& scalar) const
{
    SparseMatrix<T> sum(rows_number, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i) + scalar;

        sum.set_row(i, current_row);
    }

    return(sum);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator + (const Vector<T>& vector) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator + (const Vector<T>&) const.\n"
               << "Size of vector must be equal to number of rows.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> sum(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) + vector;

        sum.set_column(i, current_column);
    }

    return(sum);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator + (const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator + (const SparseMatrix<T>&) const.\n"
               << "Sizes of other sparse matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this sparse matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> sum(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) + other_sparse_matrix.get_column(i);

        sum.set_column(i, current_column);
    }

    return(sum);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator + (const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator + (const Matrix<T>&) const.\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> sum(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) + other_matrix.get_column(i);

        sum.set_column(i, current_column);
    }

    return(sum);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator -(const T& scalar) const
{
    SparseMatrix<T> difference(rows_number, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i) - scalar;

        difference.set_row(i, current_row);
    }

    return(difference);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator -(const Vector<T>& vector) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator -(const Vector<T>&) const.\n"
               << "Size of vector must be equal to number of rows.\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> difference(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) - vector;

        difference.set_column(i, current_column);
    }

    return(difference);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator -(const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator -(const SparseMatrix<T>&) const.\n"
               << "Sizes of other sparse matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this sparse matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> difference(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) - other_sparse_matrix.get_column(i);

        difference.set_column(i, current_column);
    }

    return(difference);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator -(const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator -(const Matrix<T>&) const.\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> difference(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) - other_matrix.get_column(i);

        difference.set_column(i, current_column);
    }

    return(difference);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator *(const T& scalar) const
{
    if(scalar == T())
    {
        SparseMatrix<T> product_sparse_matrix(rows_number,columns_number);

        return(product_sparse_matrix);
    }

    Vector<T> product(matrix_values);

    transform(matrix_values.begin(), matrix_values.end(), product.begin(),
              bind2nd(multiplies<T>(), scalar));

    SparseMatrix<T> product_sparse_matrix(rows_number,columns_number);

    product_sparse_matrix.set_values(rows_indices, columns_indices, product);

    return(product_sparse_matrix);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator *(const Vector<T>& vector) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator *(const Vector<T>&) const method.\n"
               << "Vector size(" << size << ")  must be equal to number of SparseMatrix rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> product(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) * vector;

        product.set_column(i, current_column);
    }

    return(product);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator *(const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator *(const SparseMatrix<T>&) const method.\n"
               << "Sizes of other sparse matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this sparse matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> product(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) * other_sparse_matrix.get_column(i);

        product.set_column(i, current_column);
    }

    return(product);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator *(const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator *(const Matrix<T>&) const method.\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif
    SparseMatrix<T> product(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) * other_matrix.get_column(i);

        product.set_column(i, current_column);
    }

    return(product);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator /(const T& scalar) const
{
    Vector<T> cocient(matrix_values);

    transform(matrix_values.begin(), matrix_values.end(), cocient.begin(),
              bind2nd(divides<T>(), scalar));

    SparseMatrix<T> cocient_sparse_matrix(rows_number,columns_number);

    cocient_sparse_matrix.set_values(rows_indices, columns_indices, cocient);

    return(cocient_sparse_matrix);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator /(const Vector<T>& vector) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator /(const Vector<T>&) const method.\n"
               << "Vector size(" << size << ")  must be equal to number of SparseMatrix columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> cocient(rows_number, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i) / vector;

        cocient.set_row(i, current_row);
    }

    return(cocient);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator /(const SparseMatrix<T>& other_sparse_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator /(const SparseMatrix<T>&) const method.\n"
               << "Sizes of other Sparsematrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this Sparsematrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> cocient(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) / other_sparse_matrix.get_column(i);

        cocient.set_column(i, current_column);
    }

    return(cocient);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator /(const Matrix<T>& other_matrix) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> operator /(const Matrix<T>&) const method.\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> cocient(rows_number, columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) / other_matrix.get_column(i);

        cocient.set_column(i, current_column);
    }

    return(cocient);
}

template <class T>
void SparseMatrix<T>::operator += (const T& scalar)
{
    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) + scalar;

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator += (const Vector<T>& vector)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator += (const Vector<T>&).\n"
               << "Size of vector must be equal to number of rows.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) + vector;

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator += (const SparseMatrix<T>& other_sparse_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator += (const SparseMatrix<T>&).\n"
               << "Sizes of other sparse matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this sparse matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) + other_sparse_matrix.get_column(i);

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator += (const Matrix<T>& other_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator += (const Matrix<T>&).\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) + other_matrix.get_column(i);

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator -= (const T& scalar)
{
    for(size_t i = 0; rows_number; i++)
    {
        const Vector<T> current_row = get_row(i) - scalar;

        set_row(i, current_row);
    }
}

template <class T>
void SparseMatrix<T>::operator -= (const Vector<T>& vector)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator -= (const Vector<T>&).\n"
               << "Size of vector must be equal to number of rows.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) - vector;

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator -= (const SparseMatrix<T>& other_sparse_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator -= (const SparseMatrix<T>&).\n"
               << "Sizes of other sparse matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this sparse matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) - other_sparse_matrix.get_column(i);

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator -= (const Matrix<T>& other_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator -= (const Matrix<T>&).\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be the same than sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) - other_matrix.get_column(i);

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator *= (const T& scalar)
{
    if(scalar == T())
    {
        set(rows_number,columns_number);
        return;
    }

    Vector<T> product(matrix_values);

    transform(matrix_values.begin(), matrix_values.end(), product.begin(),
              bind2nd(multiplies<T>(), scalar));

    set_values(rows_indices, columns_indices, product);
}

template <class T>
void SparseMatrix<T>::operator *= (const Vector<T>& vector)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator *= (const Vector<T>&) method.\n"
               << "Vector size(" << size << ")  must be equal to number of SparseMatrix rows(" << rows_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) * vector;

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator *= (const SparseMatrix<T>& other_sparse_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator *= (const SparseMatrix<T>&) method.\n"
               << "Sizes of other sparse matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this sparse matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) * other_sparse_matrix.get_column(i);

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator *= (const Matrix<T>& other_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator *= (const Matrix<T>&) method.\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) * other_matrix.get_column(i);

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator /= (const T& scalar)
{
    Vector<T> cocient(matrix_values);

    transform(matrix_values.begin(), matrix_values.end(), cocient.begin(),
              bind2nd(divides<T>(), scalar));

    set_values(rows_indices, columns_indices, cocient);
}

template <class T>
void SparseMatrix<T>::operator /= (const Vector<T>& vector)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator /= (const Vector<T>&) method.\n"
               << "Vector size(" << size << ")  must be equal to number of SparseMatrix columns(" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i) / vector;

        set_row(i, current_row);
    }
}

template <class T>
void SparseMatrix<T>::operator /= (const SparseMatrix<T>& other_sparse_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator /= (const SparseMatrix<T>&) method.\n"
               << "Sizes of other Sparsematrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this Sparsematrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) / other_sparse_matrix.get_column(i);

        set_column(i, current_column);
    }
}

template <class T>
void SparseMatrix<T>::operator /= (const Matrix<T>& other_matrix)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    if(other_rows_number != rows_number || other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "void operator /= (const Matrix<T>&) method.\n"
               << "Sizes of other matrix (" << other_rows_number << "," << other_columns_number << ") must be equal to sizes of this matrix (" << rows_number << "," << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i) / other_matrix.get_column(i);

        set_column(i, current_column);
    }
}

/// @todo

template <class T>
Vector<double> SparseMatrix<T>::dot(const Vector<double>&) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = vector.size();

    if(size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "Vector<T> dot(const Vector<T>&) const method.\n"
               << "Vector size must be equal to sparse matrix number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Calculate SparseMatrix-vector poduct

    Vector<double> product(rows_number);

    //    const Eigen::Map<Eigen::SparseMatrixXd> SparseMatrix_eigen((double*)this->data(), rows_number, columns_number);
    //    const Eigen::Map<Eigen::VectorXd> vector_eigen((double*)vector.data(), columns_number);
    //    Eigen::Map<Eigen::VectorXd> product_eigen(product.data(), rows_number);

    //    product_eigen = SparseMatrix_eigen*vector_eigen;

    return(product);
}

template <class T>
SparseMatrix<double> SparseMatrix<T>::dot(const SparseMatrix<double>& other_sparse_matrix) const /// @todo
{
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_sparse_matrix.get_rows_number();

    if(other_rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> dot(const SparseMatrix<T>&) const method.\n"
               << "The number of rows of the other sparse matrix (" << other_rows_number << ") must be equal to the number of columns of this sparse matrix (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> product(rows_number, other_columns_number);

    //    const Eigen::Map<Eigen::SparseMatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    //    const Eigen::Map<Eigen::SparseMatrixXd> other_eigen((double*)other_sparse_matrix.data(), other_rows_number, other_columns_number);
    //    Eigen::Map<Eigen::SparseMatrixXd> product_eigen(product.data(), rows_number, other_columns_number);

    //    product_eigen = this_eigen*other_eigen;

    return(product);
}

template <class T>
Matrix<T> SparseMatrix<T>::dot(const Matrix<T>& other_matrix) const /// @todo
{
    const size_t other_columns_number = other_matrix.get_columns_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = other_matrix.get_rows_number();

    if(other_rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> dot(const SparseMatrix<T>&) const method.\n"
               << "The number of rows of the other sparse matrix (" << other_rows_number << ") must be equal to the number of columns of this sparse matrix (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    SparseMatrix<T> product(rows_number, other_columns_number);

    //    const Eigen::Map<Eigen::SparseMatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    //    const Eigen::Map<Eigen::SparseMatrixXd> other_eigen((double*)other_sparse_matrix.data(), other_rows_number, other_columns_number);
    //    Eigen::Map<Eigen::SparseMatrixXd> product_eigen(product.data(), rows_number, other_columns_number);

    //    product_eigen = this_eigen*other_eigen;

    return(product);
}

template <class T>
Matrix<T> SparseMatrix<T>::calculate_eigenvalues() const /// @todo
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_eigen_values() const method.\n"
               << "Number of columns must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

#ifdef __OPENNN_DEBUG__

    if((*this).get_rows_number() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_eigen_values() const method.\n"
               << "Number of rows must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

#ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() != (*this).get_rows_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_eigen_values() const method.\n"
               << "The SparseMatrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    Matrix<T> eigenvalues(rows_number, 1);

    //    const Eigen::Map<Eigen::SparseMatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    //    const Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrixXd> SparseMatrix_eigen(this_eigen, Eigen::EigenvaluesOnly);
    //    Eigen::Map<Eigen::SparseMatrixXd> eigenvalues_eigen(eigenvalues.data(), rows_number, 1);

    //    eigenvalues_eigen = SparseMatrix_eigen.eigenvalues();

    return(eigenvalues);
}

template <class T>
Matrix<T> SparseMatrix<T>::calculate_eigenvectors() const /// @todo
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_eigen_values() const method.\n"
               << "Number of columns must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

#ifdef __OPENNN_DEBUG__

    if((*this).get_rows_number() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_eigen_values() const method.\n"
               << "Number of rows must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

#ifdef __OPENNN_DEBUG__

    if((*this).get_columns_number() != (*this).get_rows_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "SparseMatrix<T> calculate_eigen_values() const method.\n"
               << "The sparse matrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    Matrix<T> eigenvectors(rows_number, rows_number);

    //    const Eigen::Map<Eigen::SparseMatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
    //    const Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrixXd> SparseMatrix_eigen(this_eigen, Eigen::ComputeEigenvectors);
    //    Eigen::Map<Eigen::SparseMatrixXd> eigenvectors_eigen(eigenvectors.data(), rows_number, rows_number);

    //    eigenvectors_eigen = SparseMatrix_eigen.eigenvectors();

    return(eigenvectors);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::direct(const SparseMatrix<T>& other_sparse_matrix) const
{
    const size_t other_rows_number = other_sparse_matrix.get_rows_number();
    const size_t other_columns_number = other_sparse_matrix.get_columns_number();

    const Vector<size_t> other_rows_indices = other_sparse_matrix.get_rows_indices();
    const Vector<size_t> other_columns_indices = other_sparse_matrix.get_columns_indices();
    const Vector<T> other_matrix_values = other_sparse_matrix.get_matrix_values();

    const size_t other_nonzero_elements_number = other_matrix_values.size();

    SparseMatrix<T> direct(rows_number*other_rows_number, columns_number*other_columns_number);

    size_t alpha;
    size_t beta;

    const size_t this_nonzero_elements_number = matrix_values.size();

    for(size_t i = 0; i < this_nonzero_elements_number; i++)
    {
        const size_t this_current_row = rows_indices[i];

        const size_t this_current_column = columns_indices[i];

        for(size_t j = 0; j < other_nonzero_elements_number; j++)
        {
            const size_t other_current_row = other_rows_indices[j];

            const size_t other_current_column = other_columns_indices[j];

            alpha = other_rows_number*this_current_row+other_current_row;
            beta = other_columns_number*this_current_column+other_current_column;

            direct.set_element(alpha,beta,matrix_values[i]*other_matrix_values[j]);
        }
    }

    return(direct);
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::direct(const Matrix<T>& other_matrix) const
{
    const size_t other_rows_number = other_matrix.get_rows_number();
    const size_t other_columns_number = other_matrix.get_columns_number();

    SparseMatrix<T> direct(rows_number*other_rows_number, columns_number*other_columns_number);

    size_t alpha;
    size_t beta;

    const size_t nonzero_elements_number = matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        const size_t this_current_row = rows_indices[i];

        const size_t this_current_column = columns_indices[i];

        for(size_t j = 0; j < other_rows_number; j++)
        {
            for(size_t k = 0; k < other_columns_number; k++)
            {
                alpha = other_rows_number*this_current_row+j;
                beta = other_columns_number*this_current_column+k;

                direct.set_element(alpha,beta,matrix_values[i]*other_matrix(j,k));
            }
        }
    }

    return(direct);
}

template <class T>
bool SparseMatrix<T>::empty() const
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

template <class T>
bool SparseMatrix<T>::is_square() const
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

template <class T>
bool SparseMatrix<T>::is_symmetric() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_symmetric() const method.\n"
               << "Sparse matrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    const SparseMatrix<T> transpose = calculate_transpose();

    if((*this) == transpose)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}

template <class T>
bool SparseMatrix<T>::is_antisymmetric() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_antisymmetric() const method.\n"
               << "Sparse matrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    const SparseMatrix<T> transpose = calculate_transpose();

    if((*this) == transpose*(-1))
    {
        return(true);
    }
    else
    {
        return(false);
    }
}

template <class T>
bool SparseMatrix<T>::is_diagonal() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_diagonal() const method.\n"
               << "Sparse matrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t nonzero_elements = matrix_values.size();

    for(size_t i = 0; i < nonzero_elements; i++)
    {
        if(rows_indices[i] != columns_indices[i])
        {
            return false;
        }
    }

    return true;
}

template <class T>
bool SparseMatrix<T>::is_scalar() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_scalar() const method.\n"
               << "Sparse matrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    return get_diagonal().is_constant();
}

template <class T>
bool SparseMatrix<T>::is_identity() const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(rows_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_unity() const method.\n"
               << "Sparse matrix must be squared.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t nonzero_elements = matrix_values.size();

    for(size_t i = 0; i < nonzero_elements; i++)
    {
        if(rows_indices[i] != columns_indices[i] || matrix_values[i] != 1)
        {
            return false;
        }
    }

    return true;
}

template <class T>
bool SparseMatrix<T>::is_binary() const
{
    return(matrix_values == 1);
}

template <class T>
bool SparseMatrix<T>::is_column_binary(const size_t& column_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_column_binary(const size_t&) const method method.\n"
               << "Index of column(" << column_index << ") must be less than number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Vector<size_t> column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

    if(column_nonzero_indices.size() == 0)
    {
        return true;
    }

    return(matrix_values.get_subvector(column_nonzero_indices) == 1);
}

template <class T>
bool SparseMatrix<T>::is_column_constant(const size_t& column_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(column_index >= columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_column_constant(const size_t&) const method method.\n"
               << "Index of column(" << column_index << ") must be less than number of columns.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Vector<size_t> column_nonzero_indices = columns_indices.calculate_equal_to_indices(column_index);

    if(column_nonzero_indices.size() == 0)
    {
        return true;
    }

    return(matrix_values.get_subvector(column_nonzero_indices).is_constant());
}

template <class T>
bool SparseMatrix<T>::is_dense(const double& density_percentage) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    if(density_percentage > 1 || density_percentage <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix Template.\n"
               << "bool is_dense(const double&) const method method.\n"
               << "Density percentage must be between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(matrix_values.size() >= rows_number*columns_number*density_percentage)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <class T>
void SparseMatrix<T>::convert_association()
{
    SparseMatrix<T> copy(*this);

    set(copy.assemble_columns(copy));
}

template <class T>
KMeansResults<T> SparseMatrix<T>::calculate_k_means(const size_t&) const
{
    KMeansResults<double> k_means_results;
/*
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

    previous_means.set_row(0, this->get_row(initial_center));
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

            minimum_distances[j] = minimum_distance;
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
            clusters[minimum_distance_index].push_back(i);
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
*/
    return(k_means_results);
}

// Correlation methods

template <class T>
Vector<T> SparseMatrix<T>::calculate_multiple_linear_regression_parameters(const Vector<T>& other) const
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
double SparseMatrix<T>::calculate_multiple_linear_correlation(const Vector<T>& other) const
{
    if(columns_number == 1) // Simple linear correlation
    {
        return this->get_column(0).calculate_linear_correlation(other);
    }

    const Vector<double> multiple_linear_regression_parameters = calculate_multiple_linear_regression_parameters(other);

    const Vector<double> other_approximation = (*this).dot(multiple_linear_regression_parameters);

    return other.calculate_linear_correlation(other_approximation);
}

// Serialization methods

// void print() const method

/// Prints to the screen in the SparseMatrix object.

template <class T>
void SparseMatrix<T>::print() const
{
    cout << *this << endl;
}

template <class T>
void SparseMatrix<T>::load(const string& file_name)
{
    ifstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "void load(const string&) method.\n"
               << "Cannot open sparse matrix data file: " << file_name << "\n";

        throw logic_error(buffer.str());
    }

    if(file.peek() == ifstream::traits_type::eof())
    {
        this->set();

        return;
    }

    //file.is

    // Set SparseMatrix sizes

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
        file.seekg(0, ios::beg);

        for(size_t i = 0; i < rows_number; i++)
        {
            Vector<T> current_row(columns_number);

            for(size_t j = 0; j < columns_number; j++)
            {
                file >> current_row[j];
            }

            set_row(i, current_row);
        }
    }

    // Close file

    file.close();
}

template <class T>
Vector<string> SparseMatrix<T>::load_product_strings(const string& file_name, const char& separator)
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

        return products;
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

        const size_t number_of_nonzeros = products.size();

        products = products.get_unique_elements();

        const size_t new_columns_number = (size_t)products.size();

        set(new_rows_number, new_columns_number);

        rows_indices.set(number_of_nonzeros);
        columns_indices.set(number_of_nonzeros);
        matrix_values.set(number_of_nonzeros);

        // Clear file

        file.clear();
        file.seekg(0, ios::beg);

        size_t index = 0;

        for(size_t i = 0; i < rows_number; i++)
        {
            getline(file, line);

            istringstream buffer(line);

            while(getline(buffer, token, separator))
            {
                const size_t current_column = products.calculate_equal_to_indices(token)[0];

                rows_indices[index] = i;
                columns_indices[index] = current_column;
                matrix_values[index] = 1;

                index++;
            }
        }
    }

    // Close file

    file.close();

    return products;
}

template <class T>
void SparseMatrix<T>::load_binary(const string& file_name)
{
    ifstream file;

    file.open(file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template.\n"
               << "void load_binary(const string&) method.\n"
               << "Cannot open binary file: " << file_name << "\n";

        throw logic_error(buffer.str());
    }

    streamsize size = sizeof(size_t);

    size_t new_columns_number;
    size_t new_rows_number;

    file.read(reinterpret_cast<char*>(&new_columns_number), size);
    file.read(reinterpret_cast<char*>(&new_rows_number), size);

    size = sizeof(double);

    double value;

    this->set(new_rows_number, new_columns_number);

    for(size_t i = 0; i < new_columns_number; i++)
    {
        Vector<T> current_column(i);

        for(size_t j = 0; j < new_rows_number; j++)
        {
            file.read(reinterpret_cast<char*>(&value), size);

            current_column[j] = value;
        }

        set_column(i, current_column);
    }

    file.close();
}

template <class T>
void SparseMatrix<T>::save(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template." << endl
               << "void save(const string) method." << endl
               << "Cannot open sparse matrix data file." << endl;

        throw logic_error(buffer.str());
    }

    // Write file

    file.precision(20);

    for(size_t i = 0; i < rows_number; i++)
    {
        const Vector<T> current_row = get_row(i);

        for(size_t j = 0; j < columns_number; j++)
        {
            file << current_row[j] << " ";
        }

        file << endl;
    }

    // Close file

    file.close();
}

template <class T>
void SparseMatrix<T>::save_binary(const string& file_name) const
{
    ofstream file(file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template." << endl
               << "void save(const string) method." << endl
               << "Cannot open SparseMatrix binary file." << endl;

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

    for(size_t i = 0; i < columns_number; i++)
    {
        const Vector<T> current_column = get_column(i);

        for(size_t j = 0; j < rows_number; j++)
        {
            value = current_column[j];

            file.write(reinterpret_cast<char*>(&value), size);
        }
    }

    // Close file

    file.close();
}

template <class T>
void SparseMatrix<T>::save_csv(const string& file_name, const char& separator, const Vector<string>& column_names,  const Vector<string>& row_names, const string& nameID) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template." << endl
               << "void save_csv(const string&, const char&, const Vector<string>&, const Vector<string>&) method." << endl
               << "Cannot open sparse matrix data file: " << file_name << endl;

        throw logic_error(buffer.str());
    }

    if(column_names.size() != 0 && column_names.size() != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template." << endl
               << "void save_csv(const string&, const char&, const Vector<string>&, const Vector<string>&) method." << endl
               << "Column names must have size 0 or " << columns_number << "." << endl;

        throw logic_error(buffer.str());
    }

    if(row_names.size() != 0 && row_names.size() != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SparseMatrix template." << endl
               << "void save_csv(const string&, const char&, const Vector<string>&, const Vector<string>&) method." << endl
               << "Row names must have size 0 or " << rows_number << "." << endl;

        throw logic_error(buffer.str());
    }

    // Write file

    if(!column_names.empty())
    {
        if(!row_names.empty())
        {
            file << nameID << separator;
        }

        for(size_t j = 0; j < columns_number; j++)
        {
            file << column_names[j];

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

        const Vector<T> current_row = get_row(i);

        for(size_t j = 0; j < columns_number; j++)
        {
            file << current_row[j];

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

template <class T>
void SparseMatrix<T>::parse(const string& str)
{
    if(str.empty())
    {
        set();
    }
    else
    {
        // Set SparseMatrix sizes

        istringstream str_buffer(str);

        string line;

        getline(str_buffer, line);

        istringstream line_buffer(line);

        istream_iterator<string> it(line_buffer);
        istream_iterator<string> end;

        const vector<string> results(it, end);

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
        str_buffer.seekg(0, ios::beg);

        for(size_t i = 0; i < rows_number; i++)
        {
            Vector<T> current_row(columns_number);

            for(size_t j = 0; j < columns_number; j++)
            {
                str_buffer >> current_row[j];
            }

            set_row(i, current_row);
        }
    }
}

template <class T>
string SparseMatrix<T>::SparseMatrix_to_string(const char& separator) const
{
    ostringstream buffer;

    if(rows_number > 0 && columns_number > 0)
    {
        buffer << get_row(0).vector_to_string(separator);

        for(size_t i = 1; i < rows_number; i++)
        {
            buffer << "\n"
                   << get_row(i).vector_to_string(separator);
        }
    }

    return(buffer.str());
}

template <class T>
SparseMatrix<size_t> SparseMatrix<T>::to_size_t_SparseMatrix() const
{
    SparseMatrix<size_t> size_t_sparse_matrix(rows_number, columns_number);

    const size_t nonzero_elements_number = matrix_values.size();

    Vector<T> new_matrix_values(nonzero_elements_number);

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        new_matrix_values[i] = (size_t)matrix_values[i];
    }

    size_t_sparse_matrix.set_values(rows_indices, columns_indices, new_matrix_values);

    return(size_t_sparse_matrix);
}

template <class T>
SparseMatrix<double> SparseMatrix<T>::to_double_SparseMatrix() const
{
    SparseMatrix<double> double_sparse_matrix(rows_number, columns_number);

    const size_t nonzero_elements_number = matrix_values.size();

    Vector<double> new_matrix_values(nonzero_elements_number);

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        new_matrix_values[i] = (double)matrix_values[i];
    }

    double_sparse_matrix.set_values(rows_indices, columns_indices, new_matrix_values);

    return(double_sparse_matrix);
}

template <class T>
SparseMatrix<string> SparseMatrix<T>::to_string_SparseMatrix(const size_t& precision) const
{
    SparseMatrix<string> string_sparse_matrix(rows_number, columns_number);

    const size_t nonzero_elements_number = matrix_values.size();

    Vector<string> new_matrix_values(nonzero_elements_number);

    ostringstream buffer;

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        buffer.str("");
        buffer << setprecision(precision) << matrix_values[i];

        new_matrix_values[i] = buffer.str();
    }

    string_sparse_matrix.set_values(rows_indices, columns_indices, new_matrix_values);

    return(string_sparse_matrix);
}

template <class T>
Matrix<T> SparseMatrix<T>::to_matrix() const
{
    Matrix<T> dense_matrix(rows_number,columns_number, T());

    const size_t nonzero_elements_number = matrix_values.size();

    for(size_t i = 0; i < nonzero_elements_number; i++)
    {
        const size_t row_index = rows_indices[i];
        const size_t column_index = columns_indices[i];

        dense_matrix(row_index, column_index) = matrix_values[i];
    }

    return dense_matrix;
}

template <class T>
Vector< Vector<T> > SparseMatrix<T>::to_vector_of_vectors() const
{
    Vector< Vector<T> > vector_of_vectors(columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        vector_of_vectors[i] = get_column(i);
    }

    return vector_of_vectors;
}

template <class T>
Vector< Vector<size_t> > SparseMatrix<T>::to_CSR(Vector<T>& new_matrix_values) const
{
    const size_t nonzero_elements_number = matrix_values.size();

    Vector< Vector<size_t> > CSR_indices(2);

    CSR_indices[0].set(nonzero_elements_number);
    CSR_indices[1].set(columns_number+1, 0);

    new_matrix_values.set(nonzero_elements_number);

    const Vector<size_t> unique_columns_sorted = columns_indices.get_unique_elements();
    const size_t unique_columns_number = unique_columns_sorted.size();

    size_t index = 0;

    for(size_t i = 0; i < unique_columns_number; i++)
    {
        const size_t current_column_index = unique_columns_sorted[i];

        const Vector<size_t> current_columns_indices = columns_indices.calculate_equal_to_indices(current_column_index);

        const Vector<size_t> current_rows_indices = rows_indices.get_subvector(current_columns_indices).sort_ascending_indices();
        const size_t current_rows_number = current_rows_indices.size();

        for(size_t j = 0; j < current_rows_number; j++)
        {
            const size_t current_row_index = rows_indices[current_columns_indices[current_rows_indices[j]]];

            for(size_t k = current_column_index+1; k <= columns_number; k++)
            {
                CSR_indices[1][k]++;
            }

            CSR_indices[0][index] = current_row_index;
            new_matrix_values[index] = matrix_values[current_columns_indices[current_rows_indices[j]]];

            index++;
        }
    }

    return CSR_indices;
}

template <class T>
void SparseMatrix<T>::from_CSR(const Vector<size_t>& csr_rows_indices, const Vector<size_t>& csr_columns_indices, const Vector<T>& csr_matrix_values)
{
    const size_t maximum_csr_rows_indices = csr_rows_indices.calculate_maximum();
    const size_t csr_columns_number = csr_columns_indices.size() - 1;

    if(rows_number == 0 || columns_number == 0)
    {
        set(maximum_csr_rows_indices,csr_columns_number);
    }
    if(rows_number < maximum_csr_rows_indices)
    {
        set(maximum_csr_rows_indices,columns_number);
    }
    if(columns_number < csr_columns_number)
    {
        set(rows_number,csr_columns_number);
    }

    rows_indices = csr_rows_indices;
    matrix_values = csr_matrix_values;

    columns_indices.set(matrix_values.size());

    size_t index = 0;

    for(size_t i = 1; i < csr_columns_indices.size(); i++)
    {
        const size_t nonzero_column_elements = csr_columns_indices[i] - csr_columns_indices[i-1];

        for(size_t j = 0; j < nonzero_column_elements; j++)
        {
            columns_indices[index] = i-1;
            index++;
        }
    }
}

template <class T>
void SparseMatrix<T>::print_preview() const
{
    cout << "Rows number: " << rows_number << endl
         << "Columns number: " << columns_number << endl;

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

// Output operator

/// This method re-writes the output operator << for the Matrix template.
/// @param os Output stream.
/// @param m Output matrix.

template<class T>
ostream& operator <<(ostream& os, const SparseMatrix<T>& m)
{
    const size_t rows_number = m.get_rows_number();
    const size_t columns_number = m.get_columns_number();

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

}
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

