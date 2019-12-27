//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   C O N T A I N E R                                       
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef TENSOR_H
#define TENSOR_H

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

using namespace std;

namespace OpenNN
{

/// This template class defines a tensor for general purpose use.

///
/// This tensor also implements some mathematical methods which can be useful.

template <typename T>
class Tensor : public vector<T>
{

public:

    // Constructors

    explicit Tensor();

    explicit Tensor(const Vector<size_t>&);

    explicit Tensor(const Vector<size_t>&, const T&);

    explicit Tensor(const size_t&);

    explicit Tensor(const size_t&, const size_t&);

    explicit Tensor(const size_t&, const size_t&, const size_t&);

    explicit Tensor(const size_t&, const size_t&, const size_t&, const size_t&);

    explicit Tensor(const Matrix<T>&);

    // Destructor

    virtual ~Tensor();

    // Operators

    inline Tensor<T>& operator = (const Tensor<T>&);

    inline bool operator == (const T&);
    inline bool operator == (const Tensor<T>&);

    inline bool operator >= (const T&);
    inline bool operator <= (const T&);

    inline T& operator()(const size_t&);    
    inline T& operator()(const size_t&, const size_t&);    
    inline T& operator()(const size_t&, const size_t&, const size_t&);
    inline T& operator()(const size_t&, const size_t&, const size_t&, const size_t&);
	
    inline const T& operator()(const size_t&) const;
    inline const T& operator()(const size_t&, const size_t&) const;
    inline const T& operator()(const size_t&, const size_t&, const size_t&) const;
    inline const T& operator()(const size_t&, const size_t&, const size_t&, const size_t&) const;

    Tensor<T> operator + (const T&) const;
    Tensor<T> operator - (const T&) const;
    Tensor<T> operator * (const T&) const;
    Tensor<T> operator / (const T&) const;

    Tensor<T> operator + (const Tensor<T>&) const;
    Tensor<T> operator - (const Tensor<T>&) const;
    Tensor<T> operator * (const Tensor<T>&) const;
    Tensor<T> operator / (const Tensor<T>&) const;

    void operator *= (const Tensor<T>&);

    // Get methods

    size_t get_dimensions_number() const;
    size_t get_dimension(const size_t&) const;

    size_t get_element(const size_t&, const size_t&) const;

    void add_matrix(const Matrix<T>&);

    Vector<T> get_row(const size_t&) const;
    Vector<T> get_column(const size_t&) const;
    Vector<string> get_header() const;

    Tensor<T> get_tensor(const size_t&) const;
    Matrix<T> get_matrix(const size_t&) const;
    Matrix<T> get_matrix(const size_t&, const size_t&) const;

    void embed(const size_t&, const size_t&, const Matrix<T>&);

    void embed(const size_t&, const size_t&, const Tensor<T>&);

    Vector<T> to_vector() const;
    Matrix<T> to_matrix() const;

    Tensor<T> to_2d_tensor() const;

    // Get methods

    Vector<size_t> get_dimensions() const;

    // Set methods

    void set();
    void set(const size_t&);
    void set(const size_t&, const size_t&);
    void set(const Vector<size_t>&);
    void set(const Vector<size_t>&, const T&);
    void set(const Tensor<T>&);
    void set_row(const size_t&, const Vector<T>&);
    void set_matrix(const size_t&, const size_t&, const Matrix<T>&);
    void set_matrix(const size_t&, const Matrix<T>&);
    void set_tensor(const size_t&, const Tensor<T>&);

    void embed(const size_t&, const Vector<T>&);

    void initialize(const T&);
    void initialize_sequential();

    void randomize_uniform(const T&, const T&);
    void randomize_normal(const double & = 0.0, const double & = 1.0);

    T calculate_sum() const;

    Tensor<T> divide(const Vector<T>&, const size_t&) const;

private:

    Vector<size_t> dimensions;

};


/// Default constructor.
/// It creates an empty tensor.

template <class T>
Tensor<T>::Tensor() : vector<T>()
{
    set();
}


/// Dimensions constructor.
/// It creates a tensor with n dimensions.
/// Note that this method does not initialize the tensor.
/// @param new_dimensions Dimension of the tensor.

template <class T>
Tensor<T>::Tensor(const Vector<size_t>& new_dimensions) : vector<T> (new_dimensions.calculate_product())
{
    dimensions = new_dimensions;
}


/// Constructor and initializer.
/// It creates a tensor with given dimensions and initializes it with a given value.
/// @param new_dimensions Dimension of the tensor.
/// @param value Value to initializes.

template <class T>
Tensor<T>::Tensor(const Vector<size_t>& new_dimensions, const T& value) : vector<T> (new_dimensions.calculate_product(), value)
{
    dimensions = new_dimensions;
}


/// Constructor of a first order tensor.
/// Note that this method does not initialize the tensor.
/// @param dimension_1 Number of items.

template <class T>
Tensor<T>::Tensor(const size_t& dimension_1) : vector<T>(dimension_1)
{
    dimensions = Vector<size_t>({dimension_1});
}


/// Constructor of a second order tensor.
/// Note that this method not initialize the tensor.
/// @param dimension_1 Number of items in the first dimension.
/// @param dimension_2 Number of items in the second dimension.

template <class T>
Tensor<T>::Tensor(const size_t& dimension_1, const size_t& dimension_2) : vector<T>(dimension_1*dimension_2)
{
    dimensions = Vector<size_t>({dimension_1, dimension_2});
}


/// Constructor of a third order tensor.
/// Note that this method not initialize the tensor.
/// @param dimension_1 Number of items in the first dimension.
/// @param dimension_2 Number of items in the second dimension.
/// @param dimension_3 Number of items in the third dimension.

template <class T>
Tensor<T>::Tensor(const size_t& dimension_1, const size_t& dimension_2, const size_t& dimension_3) : vector<T>(dimension_1*dimension_2*dimension_3)
{
    dimensions = Vector<size_t>({dimension_1, dimension_2, dimension_3});
}


/// Constructor of a fourth order tensor.
/// Note that this method not initialize the tensor.
/// @param dimension_1 Number of items in the first dimension.
/// @param dimension_2 Number of items in the second dimension.
/// @param dimension_3 Number of items in the third dimension.
/// @param dimension_4 Number of items in the fourth dimension.

template <class T>
Tensor<T>::Tensor(const size_t& dimension_1, const size_t& dimension_2, const size_t& dimension_3, const size_t& dimension_4) : vector<T>(dimension_1*dimension_2*dimension_3*dimension_4)
{
    dimensions = Vector<size_t>({dimension_1, dimension_2, dimension_3, dimension_4});
}


/// Copy constructor.
/// It creates a copy of an existing Matrix.
/// @param matrix Matrix to be copied.

template <class T>
Tensor<T>::Tensor(const Matrix<T>& matrix) : vector<T> (matrix)
{
    dimensions = Vector<size_t>({ matrix.get_rows_number(), matrix.get_columns_number() });

    for(size_t i = 0; i < matrix.size(); i++)
    {
        (*this)[i] = matrix[i];
    }
}


/// Destructor.

template <class T>
Tensor<T>::~Tensor()
{
}


/// This method re-writes the output operator << for the Tensor Template.
/// @param os Output stream.
/// @param tensor Output matrix.

template<class T>
ostream& operator << (ostream& os, const Tensor<T>& tensor)
{
    const size_t dimensions_number = tensor.get_dimensions_number();

    os << "Dimensions number: " << dimensions_number << endl;
    os << "Dimensions: " << tensor.get_dimensions() << endl;
    os << "Values: " << endl;

    if(dimensions_number == 1)
    {
        const size_t size = tensor.get_dimension(0);

        for(size_t i = 0; i < size; i ++)
        {
            os << tensor[i] << " ";
        }

        os << "\n";
    }
    else if(dimensions_number == 2)
    {
        const size_t rows_number = tensor.get_dimension(0);
        const size_t columns_number = tensor.get_dimension(1);

        for(size_t i = 0; i < rows_number; i ++)
        {
            for(size_t j = 0; j < columns_number; j++)
            {
                os << tensor(i,j) << " ";
            }

            os << endl;
        }
    }
    else if(dimensions_number == 3)
    {
        const size_t rank = tensor.get_dimension(2);
        const size_t rows_number = tensor.get_dimension(0);
        const size_t columns_number = tensor.get_dimension(1);

        for(size_t k = 0; k < rank; k ++)
        {
            os << "submatrix_" << k << "\n";

            for(size_t i = 0; i < rows_number; i ++)
            {
                for(size_t j = 0; j < columns_number; j++)
                {
                    os << tensor(i,j,k) << "\t";
                }

                os << "\n";
            }
        }
    }
    else if(dimensions_number > 3)
    {
        const size_t rank_1 = tensor.get_dimension(3);
        const size_t rank_2 = tensor.get_dimension(2);
        const size_t rows_number = tensor.get_dimension(0);
        const size_t columns_number = tensor.get_dimension(1);

        for(size_t l = 0; l < rank_1; l++)
        {
            for(size_t k = 0; k < rank_2; k ++)
            {
                os << "submatrix_" << l << "_" << k << "\n";

                for(size_t i = 0; i < rows_number; i ++)
                {
                    for(size_t j = 0; j < columns_number; j++)
                    {
                        os << tensor(i,j,k,l) << "\t";
                    }

                    os << "\n";
                }
            }
        }
    }

   return os;
}


/// Sets all the entries to a given list.
/// @param other Tensor to be sets.

template <class T>
Tensor<T>& Tensor<T>::operator = (const Tensor<T>& other_tensor)
{
    if(other_tensor.dimensions != this->dimensions)
    {
        set(other_tensor.dimensions);
    }

    copy(other_tensor.begin(), other_tensor.end(), this->begin());

    return *this;
}


///Return true if the tensor is equal to an other tensor given to the function
/// @param other The other tensor to compare.

template <class T>
bool Tensor<T>::operator == (const Tensor<T>& other)
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
        if((*this)[i] != other[i]) return false;

    return true;
}


///Return true if all the elements of the tensor are equal to a value given to the matrix.
///@param value Value to compare.

template <class T>
bool Tensor<T>::operator == (const T& value)
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
        if((*this)[i] != value) return false;

    return true;
}


///Return true if all the elements of the tensor are equal or greater than an other value given to the matrix.
///@param value Value to compare.

template <class T>
bool Tensor<T>::operator >= (const T& value)
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
        if((*this)[i] < value) return false;

    return true;
}


///Return true if all the elements of the tensor are equal or less than an other value given to the matrix.
///@param value Value to compare.

template <class T>
bool Tensor<T>::operator <= (const T& value)
{
    const size_t size = this->size();

    for(size_t i = 0; i < size; i++)
        if((*this)[i] > value) return false;

    return true;
}


template<class T>
inline const T& Tensor<T>::operator()(const size_t& row, const size_t& column) const
{
#ifdef __OPENNN_DEBUG__

 const size_t dimensions_number = get_dimensions_number();

 if(dimensions_number != 2)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "const T& Tensor<T>::operator()(const size_t&, const size_t&) const.\n"
           << "Number of dimensions (" << dimensions_number << ") must be 2.\n";

    throw logic_error(buffer.str());
 }

 if(row >= dimensions[0])
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "const T& Tensor<T>::operator()(const size_t&, const size_t&) const.\n"
           << "Row index (" << row << ") must be less than rows number (" << dimensions[0] << ").\n";

    throw logic_error(buffer.str());
 }

 if(column >= dimensions[1])
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "const T& Tensor<T>::operator()(const size_t&, const size_t&) const.\n"
           << "Column index (" << column << ") must be less than columns number (" << dimensions[1] << ").\n";

    throw logic_error(buffer.str());
 }

#endif

    return (*this)[dimensions[0]*column + row];
}


template <class T>
inline const T& Tensor<T>::operator()(const size_t& index_0 , const size_t& index_1, const size_t& index_2) const
{
#ifdef __OPENNN_DEBUG__

 const size_t dimensions_number = get_dimensions_number();

 if(dimensions_number != 3)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "const T& Tensor<T>::operator()(const size_t&, const size_t&, const size_t&) const.\n"
           << "Number of dimensions (" << dimensions_number << ") must be 3.\n";

    throw logic_error(buffer.str());
 }

#endif

    return (*this)[index_2 * dimensions[0] * dimensions[1] + index_1 * dimensions[0] + index_0];
}


template <class T>
inline const T& Tensor<T>::operator()(const size_t& index_0 , const size_t& index_1, const size_t& index_2, const size_t& index_3) const
{
#ifdef __OPENNN_DEBUG__

 const size_t dimensions_number = get_dimensions_number();

 if(dimensions_number != 4)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "const T& Tensor<T>::operator()(const size_t&, const size_t&, const size_t&, const size_t&) const.\n"
           << "Number of dimensions (" << dimensions_number << ") must be 4.\n";

    throw logic_error(buffer.str());
 }

#endif

    return (*this)[index_3 * dimensions[0] * dimensions[1] * dimensions[2] + index_2 *dimensions[0] * dimensions[1] + index_1 * dimensions[0] + index_0];
}


template <class T>
T& Tensor<T>::operator()(const size_t& row, const size_t& column)
{
#ifdef __OPENNN_DEBUG__

 const size_t dimensions_number = get_dimensions_number();

 if(dimensions_number != 2)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "T& Tensor<T>::operator()(const size_t&, const size_t&) const.\n"
           << "Number of dimensions (" << dimensions_number << ") must be 2.\n";

    throw logic_error(buffer.str());
 }

 if(row >= dimensions[0])
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "T& Tensor<T>::operator()(const size_t&, const size_t&) const.\n"
           << "Row index (" << row << ") must be less than rows number (" << dimensions[0] << ").\n";

    throw logic_error(buffer.str());
 }

 if(column >= dimensions[1])
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "T& Tensor<T>::operator()(const size_t&, const size_t&) const.\n"
           << "Column index (" << column << ") must be less than columns number (" << dimensions[1] << ").\n";

    throw logic_error(buffer.str());
 }

#endif

    return (*this)[dimensions[0]*column+row];
}


template <class T>
T& Tensor<T>::operator()(const size_t& index_0 , const size_t& index_1, const size_t& index_2)
{
#ifdef __OPENNN_DEBUG__

 const size_t dimensions_number = get_dimensions_number();

 if(dimensions_number != 3)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "const T& Tensor<T>::operator()(const size_t&, const size_t&, const size_t&) const.\n"
           << "Number of dimensions (" << dimensions_number << ") must be 3.\n";

    throw logic_error(buffer.str());
 }

#endif

    return (*this)[index_2 * (dimensions[0] * dimensions[1]) + index_1 * dimensions[0] + index_0];
}


template <class T>
T& Tensor<T>::operator()(const size_t& index_0 , const size_t& index_1, const size_t& index_2, const size_t& index_3)
{
#ifdef __OPENNN_DEBUG__

 const size_t dimensions_number = get_dimensions_number();

 if(dimensions_number != 4)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor template.\n"
           << "const T& Tensor<T>::operator()(const size_t&, const size_t&, const size_t&, const size_t&) const.\n"
           << "Number of dimensions (" << dimensions_number << ") must be 4.\n";

    throw logic_error(buffer.str());
 }

#endif

    return (*this)[index_3 * dimensions[0] * dimensions[1] * dimensions[2] + index_2 *dimensions[0] * dimensions[1] + index_1 * dimensions[0] + index_0];
}


/// Sum tensor+scalar arithmetic operator.
/// @param value Scalar value to be added to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator + (const T& value) const
{
    Tensor<T> output(dimensions);

    for(size_t i = 0; i < this->size(); i++)
    {
        output[i] = (*this)[i] + value;
    }

    return output;
}


/// Sum tensor+tensor arithmetic operator.
/// @param vector Tensor to be added to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator + (const Tensor<T>& other) const
{
    Tensor<T> output(other.get_dimensions());

    for(size_t i = 0; i < other.size(); i++)
    {
        output[i] = (*this)[i] + other[i];
    }

    return output;
}


/// Difference tensor-scalar arithmetic operator.
/// @param value Scalar value to be subtracted to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator - (const T& value) const
{
    Tensor<T> output(dimensions);

    for(size_t i = 0; i < this->size(); i++)
    {
        output[i] = (*this)[i] - value;
    }

    return output;
}


/// Difference tensor-tensor arithmetic operator.
/// @param other Tensor to be subtracted to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator - (const Tensor<T>& other) const
{
    Tensor<T> output(other.get_dimensions());

    for(size_t i = 0; i < other.size(); i++)
    {
        output[i] = (*this)[i]-other[i];
    }

    return output;
}


/// Product tensor*scalar arithmetic operator.
/// @param value Scalar value to be multiplied to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator * (const T& value) const
{
    Tensor<T> output(dimensions);

    for(size_t i = 0; i < this->size(); i++)
    {
        output[i] = (*this)[i] * value;
    }

    return output;
}


/// Element by element product tensor*tensor arithmetic operator.
/// @param other Tensor to be multiplied to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator * (const Tensor<T>& other) const
{
    Tensor<T> output(other.get_dimensions());

    for(size_t i = 0; i < other.size(); i++)
    {
        output[i] = (*this)[i]*other[i];
    }

    return output;
}


/// Quotient tensor/scalar arithmetic operator.
/// @param value Scalar value to be divided to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator / (const T& value) const
{
    Tensor<T> output(dimensions);

    for(size_t i = 0; i < this->size(); i++)
    {
        output[i] = (*this)[i] / value;
    }

    return output;
}


/// Quotient tensor/tensor arithmetic operator.
/// @param other Tensor to be divided to this tensor.

template <class T>
Tensor<T> Tensor<T>::operator / (const Tensor<T>& other) const
{
    Tensor<T> output(other.get_dimensions());

    for(size_t i = 0; i < other.size(); i++)
    {
        output[i] = (*this)[i]/other[i];
    }

    return output;
}


/// Tensor product and assignment operator.
/// @param other Tensor to be multiplied to this tensor.

template <class T>
void Tensor<T>::operator *= (const Tensor<T>& other)
{
    for(size_t i = 0; i < this->size(); i++)
    {
        (*this)[i] *= other[i];
    }
}


/// Returns the total number of dimensions of the tensor.

template<class T>
Vector<size_t> Tensor<T>::get_dimensions() const
{
    return dimensions;
}


/// Returns the number of itemns in the dimesion with given index.
/// This may be : dimesion_0 , dimension_1, dimension_2 ...
/// @param index_dimesion Selected dimension..

template <class T>
size_t Tensor<T>::get_dimension(const size_t& index_dimension) const
{
    return dimensions[index_dimension];
}


/// Returns the number of dimensions of the tensor.

template <class T>
size_t Tensor<T>::get_dimensions_number() const
{
    return dimensions.size();
}


/// Returns the element of this tensor with the given indices i,j.
/// Note that this method is only valid for tensors of order two.
/// @param row Row index.
/// @param column Column index.

template <class T>
size_t Tensor<T>::get_element(const size_t& row, const size_t& column) const
{
    return (*this)[dimensions[0]*column+row];
}


/// Returns a vector with the row of the given index.
/// Note that this method is only valid for tensors of order two.
/// @param i Row index.

template <class T>
Vector<T> Tensor<T>::get_row(const size_t& i) const
{
    Vector<T> row(dimensions[1]);

    for(size_t j = 0; j < dimensions[1]; j++)
    {
       row[j] = (*this)(i,j);
    }

    return row;
}


/// Returns a vector with the column of the given index.
/// Note that this method is only valid for tensors of order two.
/// @param j Column index.

template <class T>
Vector<T> Tensor<T>::get_column(const size_t& j) const
{
   Vector<T> column(dimensions[0]);

   for(size_t i = 0; i < dimensions[0]; i++)
   {
      column[i] = (*this)(i,j);
   }

   return column;
}


/// Returns a string vector with the header.

template <class T>
Vector<string> Tensor<T>::get_header() const
{
    Vector<string> header(dimensions[1]);

    for(size_t j = 0; j < dimensions[1]; j++)
    {
       header[j] = (*this)(0,j);
    }

    return header;
}


/// Returns a tensor with the elements corresponding to the given index.

template <class T>
Tensor<T> Tensor<T>::get_tensor(const size_t& index_0) const
{
    Tensor<T> tensor(Vector<size_t>({dimensions[1], dimensions[2], dimensions[3]}));

    for(size_t i = 0; i < dimensions[1]; i++)
        for(size_t j = 0; j < dimensions[2]; j++)
            for(size_t k = 0; k < dimensions[3]; k++)
                tensor(i,j,k) = (*this)(index_0,i,j,k);

    return tensor;
}


/// Returns a matrix with the column and rows of the given indices.
/// @param matrix_index Matrix index.

template <class T>
Matrix<T> Tensor<T>::get_matrix(const size_t& matrix_index) const
{
    const size_t dimensions_number = get_dimensions_number();

    if(dimensions_number == 2)
    {
        const size_t rows_number = dimensions[0];
        const size_t columns_number = dimensions[1];
        const size_t elements_number = rows_number * columns_number;

        Matrix<T> matrix(rows_number, columns_number);

        for(size_t i = 0; i < elements_number; i++)
        {
            matrix[i] = (*this)[i];
        }

        return matrix;
    }
    else if(dimensions_number > 2)
    {

        if(matrix_index > dimensions[2])
        {
//            throw exception("Matrix out of bounds");
        }

        const size_t rows_number = dimensions[0];
        const size_t columns_number = dimensions[1];

        const size_t elements_number = rows_number * columns_number;

        Matrix<T> matrix(rows_number, columns_number);

        for(size_t i = 0; i < elements_number; i ++)
            matrix[i] = (*this)[matrix_index * elements_number + i];

        return matrix;
    }
    else
    {
        return Matrix<T>();
    }
}


/// Returns a matrix with the elements of this vector between some given indices.
/// @param index_0 Matrix index.
/// @param index_1

template <class T>
Matrix<T> Tensor<T>::get_matrix(const size_t& index_0, const size_t& index_1) const
{
    const size_t dimension_2 = dimensions[2];
    const size_t dimension_3 = dimensions[3];

    Matrix<T> matrix(dimension_2, dimension_3);

    for(size_t i = 0; i < dimension_2; i++)
    {
        for(size_t j = 0; j < dimension_3; j++)
        {
            matrix(i,j) = (*this)(index_0, index_1, i, j);
        }
    }

    return matrix;
}


template <class T>
void Tensor<T>::add_matrix(const Matrix<T>& a_matrix)
{
    const size_t order = 2;

    if(order < 2)
    {
//        throw exception("OpenNN Exception: Tensor<T> template\n\
//        Cannot add a new matrix to this tensor.");

    }
    else if(order == 2)
    {
        this->insert(this->end(), a_matrix.begin(), a_matrix.end());

        dimensions = Vector<size_t>({ dimensions[0], dimensions[1], 2} );

    }
    else if(order == 3)
    {
        // set_dimensions({ dimensions[0], dimensions[1], dimensions[2] });

        this->insert(this->end(), a_matrix.begin(), a_matrix.end());

        dimensions[2] += 1;

        cout << *this << endl;
    }
    else
    {
        //
    }
}


/// Sets the size of this tensor to zero.

template <class T>
void Tensor<T>::set()
{
    this->resize(0);
}


/// Sets a new dimensions to the tensor.
/// It does not initialize the data.
/// @param size Size for the tensor.

template <class T>
void Tensor<T>::set(const size_t& size)
{
    dimensions.set({size});

    this->resize(size);
}


template <class T>
void Tensor<T>::set(const size_t&, const size_t&)
{
}


/// Sets a new dimension to the tensor.
/// It does not initialize the data.
/// @param new_dimensions Dimensions for the tensor.

template <class T>
void Tensor<T>::set(const Vector<size_t>& new_dimensions)
{
    dimensions = new_dimensions;

    this->resize(new_dimensions.calculate_product());
}


/// Sets a new dimension to the vector and initializes all its elements with a given value.
/// @param new_dimensions Dimensions for the tensor.
/// @param new_value Value to initialize.

template <class T>
void Tensor<T>::set(const Vector<size_t>& new_dimensions, const T& value)
{
    dimensions = new_dimensions;

    this->resize(new_dimensions.calculate_product());

    this->initialize(value);
}


template <class T>
void Tensor<T>::set(const Tensor<T>& other_tensor)
{
    if(other_tensor.dimensions != this->dimensions)
    {
        set(other_tensor.dimensions);
    }

    copy(other_tensor.begin(), other_tensor.end(), this->begin());
}


/// Sets new values of a single row in the matrix.
/// @param row_index Index of row.
/// @param new_row New values of single row.
// @tochek
template <class T>
void Tensor<T>::set_row(const size_t& row_index, const Vector<T>& new_row)
{
    const size_t size = new_row.size();

    for(size_t i = 0; i < size; i++)
    {
        (*this)(row_index, i) = new_row[i];
    }
}


/// Embed another matrix starting from a given position.
/// Note that this method is only valid for tensors of order two.
/// @param row_position Insertion row position.
/// @param column_position Insertion row position.
/// @param other_matrix Matrix to be inserted.

template <class T>
void Tensor<T>::embed(const size_t& row_position, const size_t& column_position, const Matrix<T>& other_matrix)
{
   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

//   if(row_position + other_rows_number > this.rows_number)
   if(row_position + other_rows_number > this->get_dimensions()[0])
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Tensor Template.\n"
             << "void insert(const size_t&, const size_t&, const Matrix<T>&) const method.\n"
             << "Cannot tuck in matrix.\n";

      throw logic_error(buffer.str());
   }

//   if(column_position + other_columns_number > this.columns_number)
   if(row_position + other_rows_number > this->get_dimensions()[0])
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Tensor Template.\n"
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


/// Embed another matrix starting from a given position.
/// Note that this method is only valid for tensors of order two.
/// @param row_position Insertion row position.
/// @param column_position Insertion row position.
/// @param other_tensor Tensor to be inserted.

template <class T>
void Tensor<T>::embed(const size_t& row_position, const size_t& column_position, const Tensor<T>& other_tensor)
{
   const size_t other_rows_number = other_tensor.get_rows_number();
   const size_t other_columns_number = other_tensor.get_columns_number();

   #ifdef __OPENNN_DEBUG__

//   if(row_position + other_rows_number > this.rows_number)
   if(row_position + other_rows_number > this->get_dimensions()[1])
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Tensor Template.\n"
             << "void insert(const size_t&, const size_t&, const Matrix<T>&) const method.\n"
             << "Cannot tuck in matrix.\n";

      throw logic_error(buffer.str());
   }

//   if(column_position + other_columns_number > this.columns_number)
   if(row_position + other_rows_number > this->get_dimensions()[1])
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Tensor Template.\n"
             << "void insert(const size_t&, const size_t&, const Matrix<T>&) const method.\n"
             << "Cannot tuck in matrix.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < other_rows_number; i++)
   {
      for(size_t j = 0; j < other_columns_number; j++)
      {
        (*this)(row_position+i,column_position+j) = other_tensor(i,j);
      }
   }
}


/// This method transforms the tensor into a vector.

template <class T>
Vector<T> Tensor<T>::to_vector() const
{
    return Vector<T>(this->begin(),this->end());
}


/// This method transforms the tensor into a matrix.

template <class T>
Matrix<T> Tensor<T>::to_matrix() const
{

#ifdef __OPENNN_DEBUG__

const size_t dimensions_number = get_dimensions_number();

if(dimensions_number != 2)
{
   ostringstream buffer;

   buffer << "OpenNN Exception: Tensor Template.\n"
          << "Matrix<T> to_matrix() const method.\n"
          << "Dimension of tensor ("<< dimensions_number << ") must be 2.\n";

   throw logic_error(buffer.str());
}

#endif

    Matrix<T> matrix(dimensions[0], dimensions[1]);

    for(size_t i = 0; i < this->size(); i++)
        matrix[i] = (*this)[i];

    return matrix;
}


/// Returns the tensor reshaped as a 2-dimensional tensor.

template <class T>
Tensor<T> Tensor<T>::to_2d_tensor() const
{
    Tensor<T> tensor;

    const size_t size = this->size();

    if(get_dimensions_number() == 1)
    {
        tensor.set(Vector<size_t>({1, size}));
        for(size_t s = 0; s < size; s++)
        {
            tensor(0, s) = (*this)[s];
        }
    }
    else if(get_dimensions_number() == 2)
    {
        tensor = (*this);
    }
    else if(get_dimensions_number() == 3)
    {
        tensor.set(Vector<size_t>({1, size}));
        for(size_t s = 0; s < size; s++)
        {
            tensor(0, s) = (*this)(s%get_dimension(0), s/get_dimension(0), s/(get_dimension(0)*get_dimension(1)));
        }
    }
    else if(get_dimensions_number() == 4)
    {
        tensor.set(Vector<size_t>({get_dimension(0), get_dimension(1) * get_dimension(2)* get_dimension(3)}));
        for(size_t s = 0; s < get_dimension(1) * get_dimension(2) * get_dimension(3); s++)
        {
            for(size_t n = 0; n < get_dimension(0); n++)
            {
                tensor(n, s) = (*this)(n, s%get_dimension(1), (s/get_dimension(1))%get_dimension(2), s/(get_dimension(1) * get_dimension(2)));
            }
        }
    }
    else if (get_dimensions_number() > 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Tensor template.\n"
               << "void Tensor<T>::set_new_dimensions(const Vector<size_t>&) method.\n"
               << "The number of dimensions of tensor (" << get_dimensions_number() << ") must not be greater than four.\n";

        throw logic_error(buffer.str());
    }

    return tensor;
}

/// This method sets new matrix into this tensor at the given positions.
/// @param index_0 Index of first position.
/// @param index_1 Index of Second position.
/// @param matrix Matrix to be inserted in this tensor.

// Number of dimension of tensor must be 3?
template <class T>
void Tensor<T>::set_matrix(const size_t& index_0, const size_t& index_1, const Matrix<T>& matrix)
{
#ifdef __OPENNN_DEBUG__

    const size_t dimensions_number = get_dimensions_number();

if(dimensions_number != 4)
{
   ostringstream buffer;

   buffer << "OpenNN Exception: Tensor Template.\n"
          << "Matrix<T> set_matrix(const size_t&, const size_t&, const Matrix<double>&) const method.\n"
          << "Number of dimension of tensor must be 4.\n";

   throw logic_error(buffer.str());
}

#endif

    const size_t rows_number = dimensions[2];
    const size_t columns_number = dimensions[3];

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            (*this)(index_0, index_1, i, j) = matrix(i,j);
        }
    }
}


/// This method sets new matrix into this tensor at the given positions.
/// @param index_0 Index of first position.
/// @param matrix Matrix to be inserted in this tensor.

template <class T>
void Tensor<T>::set_matrix(const size_t& index_0, const Matrix<T>& matrix)
{

    const size_t rows_number = dimensions[0];
    const size_t columns_number = dimensions[1];

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            (*this)(i, j, index_0)  = matrix(i,j);
        }
    }
}


/// This method sets a new tensor into this tensor at the given position.
/// @param index_0 Position the next tensor is to be inserted into.
/// @param tensor Tensor to be inserted into this tensor.

template <class T>
void Tensor<T>::set_tensor(const size_t& index_0, const Tensor<T>& tensor)
{
    for(size_t i = 0; i < dimensions[1]; i++)
        for(size_t j = 0; j < dimensions[2]; j++)
            for(size_t k = 0; k < dimensions[3]; k++)
                (*this)(index_0,i,j,k) = tensor(i,j,k);
}


/// This method sets a vector of given values into the tensor (by rows) starting
/// from a given index.
/// @param index Starting index
/// @param vector Vector of values

template <class T>
void Tensor<T>::embed(const size_t& index, const Vector<T>& vector)
{
    const size_t dimension_0 = dimensions[0];

    const size_t vector_size = vector.size();

    for(size_t i = 0; i < vector_size; i++)
    {
        (*this)[index+i*dimension_0] = vector[i];
    }
}


/// Initializes all the elements of the tensor with a given value.
/// @param value Type value.

template <class T>
void Tensor<T>::initialize(const T&value)
{
  fill(this->begin(),this->end(), value);
}


/// Initializes all the elements of the tensor in a sequential order (0, 1, 2...).

template <class T>
void Tensor<T>::initialize_sequential() {
  for(size_t i = 0; i < this->size(); i++) {
   (*this)[i] = static_cast<T>(i);
  }
}


/// Assigns a random value comprised between a minimum value and a maximum value
/// to each element in the tensor.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

template <class T>
void Tensor<T>::randomize_uniform(const T& minimum, const T& maximum)
{
#ifdef __OPENNN_DEBUG__

  if(minimum > maximum) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor Template.\n"
           << "void randomize_uniform(const double&, const double&) method.\n"
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


/// Assigns random values to each element in the tensor.
/// These are taken from a normal distribution with single mean and standard
/// deviation values for all the elements.
/// @param mean Mean value of uniform distribution.
/// @param standard_deviation Standard deviation value of uniform distribution.

template <class T>
void Tensor<T>::randomize_normal(const double &mean,
                                 const double &standard_deviation)
{
#ifdef __OPENNN_DEBUG__

  if(standard_deviation < 0.0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Tensor Template.\n"
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


/// Returns the sum of all the elements in the tensor.

template <class T>
T Tensor<T>::calculate_sum() const
{
   T sum = 0;

   for(size_t i = 0; i < this->size(); i++)
   {
        sum += (*this)[i];
   }

   return sum;
}


template <class T>
Tensor<T> Tensor<T>::divide(const Vector<T>&, const size_t&) const
{
    return Tensor<double>();
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

