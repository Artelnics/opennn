//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E T R I C S   F U N C T I O N S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "metrics.h"

namespace OpenNN
{

double dot(const Vector<double>& a, const Vector<double>& b)
{
    const size_t a_size = a.size();

  #ifdef __OPENNN_DEBUG__

    const size_t b_size = b.size();

    if(a_size != b_size)
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "double dot(const Vector<double>&, const Vector<double>&) method.\n"
             << "Both vector sizes must be the same.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double dot_product = 0.0;

    for(size_t i = 0; i < a_size; i++)
    {
      dot_product += a[i] * b[i];
    }

    return dot_product;
}


Vector<double> dot(const Vector<double>& vector, const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

  #ifdef __OPENNN_DEBUG__
    const size_t vector_size = vector.size();

    if(rows_number != vector_size)
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "Vector<double> dot(const Vector<double>&, const Matrix<double>&) method.\n"
             << "Matrix number of rows (" << rows_number << ") must be equal to vector size (" << vector_size << ").\n";

      throw logic_error(buffer.str());
    }

  #endif

    Vector<double> product(columns_number, 0.0);

     for(size_t j = 0; j < columns_number; j++)
     {
        for(size_t i = 0; i < rows_number; i++)
        {
           product[j] += vector[i]*matrix(i,j);
       }
     }

    return product;
}


Vector<double> dot(const Matrix<double>& matrix, const Vector<double>& vector)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<double> product(rows_number);

    const Eigen::Map<Eigen::MatrixXd> matrix_eigen((double*)matrix.data(), static_cast<int>(rows_number), static_cast<int>(columns_number));
    const Eigen::Map<Eigen::VectorXd> vector_eigen((double*)vector.data(), static_cast<int>(columns_number));
    Eigen::Map<Eigen::VectorXd> product_eigen(product.data(), static_cast<int>(rows_number));

    product_eigen = matrix_eigen*vector_eigen;

    return product;
}


Matrix<double> dot(const Matrix<double>& matrix_1, const Matrix<double>& matrix_2)
{
    const size_t rows_number = matrix_1.get_rows_number();
    const size_t columns_number = matrix_1.get_columns_number();

    const size_t other_rows_number = matrix_2.get_rows_number();
    const size_t other_columns_number = matrix_2.get_columns_number();

    Matrix<double> product(rows_number, other_columns_number);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)matrix_1.data(), static_cast<int>(rows_number), static_cast<int>(columns_number));
    const Eigen::Map<Eigen::MatrixXd> other_eigen((double*)matrix_2.data(), static_cast<int>(other_rows_number), static_cast<int>(other_columns_number));
    Eigen::Map<Eigen::MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number), static_cast<int>(other_columns_number));

    product_eigen = this_eigen*other_eigen;

    return product;
}


Tensor<double> dot(const Tensor<double>& tensor, const Matrix<double>& matrix)
{
    const size_t rows_number = tensor.get_dimension(0);
    const size_t columns_number = tensor.get_dimension(1);

    const size_t other_rows_number = matrix.get_rows_number();
    const size_t other_columns_number = matrix.get_columns_number();

    Tensor<double> product(rows_number, other_columns_number);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)tensor.data(), static_cast<int>(rows_number), static_cast<int>(columns_number));
    const Eigen::Map<Eigen::MatrixXd> other_eigen((double*)matrix.data(), static_cast<int>(other_rows_number), static_cast<int>(other_columns_number));
    Eigen::Map<Eigen::MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number), static_cast<int>(other_columns_number));

    product_eigen = this_eigen*other_eigen;

    return product;
}


Tensor<double> dot_2d_2d(const Tensor<double>& tensor_1, const Tensor<double>& tensor_2)
{
#ifdef __OPENNN_DEBUG__

    const size_t dimensions_number_1 = tensor_1.get_dimensions_number();
    const size_t dimensions_number_2 = tensor_2.get_dimensions_number();

  if(dimensions_number_1 != 2)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<double> dot(const Tensor<double>&, const Tensor<double>&) method.\n"
           << "Dimensions number of tensor 1 (" << dimensions_number_1 << ") must be 2.\n";

    throw logic_error(buffer.str());
  }

  if(dimensions_number_2 != 2)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<double> dot(const Tensor<double>&, const Tensor<double>&) method.\n"
           << "Dimensions number of tensor 2 (" << dimensions_number_2 << ") must be 2.\n";

    throw logic_error(buffer.str());
  }

#endif

    const size_t rows_number = tensor_1.get_dimension(0);
    const size_t columns_number = tensor_1.get_dimension(1);

    const size_t other_rows_number = tensor_2.get_dimension(0);
    const size_t other_columns_number = tensor_2.get_dimension(1);

    Tensor<double> product(rows_number, other_columns_number);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)tensor_1.data(), static_cast<int>(rows_number), static_cast<int>(columns_number));
    const Eigen::Map<Eigen::MatrixXd> other_eigen((double*)tensor_2.data(), static_cast<int>(other_rows_number), static_cast<int>(other_columns_number));
    Eigen::Map<Eigen::MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number), static_cast<int>(other_columns_number));

    product_eigen = this_eigen*other_eigen;

    return product;
}


Tensor<double> dot_2d_3d(const Tensor<double>& tensor_1, const Tensor<double>& tensor_2)
{
#ifdef __OPENNN_DEBUG__

    const size_t dimensions_number_1 = tensor_1.get_dimensions_number();
    const size_t dimensions_number_2 = tensor_2.get_dimensions_number();

  if(dimensions_number_1 != 2)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<double> dot(const Tensor<double>&, const Tensor<double>&) method.\n"
           << "Dimensions number of tensor 1 (" << dimensions_number_1 << ") must be 2.\n";

    throw logic_error(buffer.str());
  }

  if(dimensions_number_2 != 3)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<double> dot(const Tensor<double>&, const Tensor<double>&) method.\n"
           << "Dimensions number of tensor 2 (" << dimensions_number_2 << ") must be 3.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t n = tensor_2.get_dimensions()[2];

  const Vector<size_t> dimensions_1 = tensor_1.get_dimensions();
  const Vector<size_t> dimensions_2 = tensor_2.get_dimensions();

  const Matrix<double> tensor_1_matrix = tensor_1.get_matrix(0);

  Tensor<double> product(dimensions_1[0], dimensions_2[1]);

  for(size_t i = 0; i < n; i ++)
  {
      const Matrix<double> i_matrix = tensor_2.get_matrix(i);

      const Matrix<double> i_row = tensor_1_matrix.get_submatrix_rows({i} );

      Matrix<double> dot_product = dot(i_row, i_matrix);

      for(size_t k = 0; k < dimensions_2[1]; k++)
      {
          product(i,k) = dot_product(0,k);
      }
  }

  return product;
}


Matrix<double> dot(const Matrix<double>& matrix, const Tensor<double>& tensor)
{
    const size_t order = tensor.get_dimensions_number();

    if(order == 2)
    {
        return dot(matrix, tensor.get_matrix(0));
    }
    else if(order > 2)
    {
        const size_t n = tensor.get_dimensions()[2];

        Matrix<double> outputs(n, matrix.get_columns_number());

        for(size_t i = 0; i < n; i ++)
        {
            const Matrix<double> i_matrix = tensor.get_matrix(i);

            const Matrix<double> i_row = matrix.get_submatrix_rows({i} );

            Matrix<double> dot_product = dot(i_row, i_matrix);

            outputs.set_row(i, dot_product.to_vector() );
        }

        return outputs;
    }

    return Matrix<double>();
}


/// Returns the vector norm.

double l1_norm(const Vector<double>& vector)
{
  return absolute_value(vector).calculate_sum();
}


/// Returns the vector norm.

double l2_norm(const Vector<double>& vector)
{
  const size_t x_size = vector.size();



  double norm = 0.0;

  for(size_t i = 0; i < x_size; i++) {
    norm += vector[i] *vector[i];
  }

    return sqrt(norm);
}


/// Returns the gradient of the vector norm.

Vector<double> l2_norm_gradient(const Vector<double>& vector)
{
  const size_t x_size = vector.size();

  Vector<double> gradient(x_size);

  const double norm = l2_norm(vector);

  if(norm == 0.0) {
    gradient.initialize(0.0);
  } else {
    gradient = vector/ norm;
  }

  return gradient;
}


/// Returns the hessian of the vector norm.

Matrix<double> l2_norm_hessian(const Vector<double>& vector)
{
  const size_t x_size = vector.size();

  Matrix<double> hessian(x_size, x_size);

  const double norm = l2_norm(vector);

  if(norm == 0.0) {
    hessian.initialize(0.0);
  } else {
    hessian = direct(vector, vector)/(norm * norm * norm);
  }

  return(hessian);
}


/// Returns the vector p-norm.

double lp_norm(const Vector<double>& vector, const double &p)
{
#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double calculate_p_norm(const double&) method.\n"
           << "p value must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t x_size = vector.size();

  double norm = 0.0;

  for(size_t i = 0; i < x_size; i++) {
    norm += pow(abs(vector[i]), p);
  }

  norm = pow(norm, 1.0 / p);

  return norm;
}


/// Returns the gradient of the vector norm.

Vector<double> lp_norm_gradient(const Vector<double>& vector, const double &p)
{
#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Vector<double> calculate_p_norm_gradient(const double&) const "
              "method.\n"
           << "p value must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const size_t x_size = vector.size();

  Vector<double> gradient(x_size);

  const double p_norm = lp_norm(vector, p);

  if(p_norm == 0.0)
  {
    gradient.initialize(0.0);
  }
  else
  {
    for(size_t i = 0; i < x_size; i++)
    {
      gradient[i] =
         vector[i] * pow(abs(vector[i]), p - 2.0) / pow(p_norm, p - 1.0);
    }
  }

  return gradient;
}


/// Outer product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

Matrix<double> direct(const Vector<double>& x, const Vector<double>& y)
{
  const size_t x_size = x.size();

#ifdef __OPENNN_DEBUG__

  const size_t y_size = y.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Matrix<double> direct(const Vector<double>&) method.\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  Matrix<double> direct(x_size, x_size);

   #pragma omp parallel for if(x_size > 1000)

  for(int i = 0; i < static_cast<int>(x_size); i++)
  {
    for(size_t j = 0; j < x_size; j++)
    {
      direct(static_cast<size_t>(i), j) = x[static_cast<size_t>(i)] * y[j];
    }
  }

  return direct;
}


/// Returns the determinant of a square matrix.

double determinant(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

    if(matrix.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "determinant() method.\n"
              << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "determinant() method.\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   double determinant = 0;

   if(rows_number == 1)
   {
      determinant = matrix(0,0);
   }
   else if(rows_number == 2)
   {
      determinant = matrix(0,0)*matrix(1,1) - matrix(1,0)*matrix(0,1);
   }
   else
   {
      int sign;

      for(size_t row_index = 0; row_index < rows_number; row_index++)
      {
         // Calculate sub data

         Matrix<double> sub_matrix(rows_number-1, columns_number-1);

         for(size_t i = 1; i < rows_number; i++)
         {
            size_t j2 = 0;

            for(size_t j = 0; j < columns_number; j++)
            {
               if(j == row_index) continue;

               sub_matrix(i-1,j2) = matrix(i,j);

               j2++;
            }
         }

         sign = static_cast<int>(((row_index + 2) % 2 == 0) ? 1 : -1 );

         determinant += sign*matrix(0,row_index)*OpenNN::determinant(sub_matrix);
      }
   }

   return determinant;
}


/// Returns the cofactor matrix.

Matrix<double> cofactor(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   Matrix<double> cofactor(rows_number, columns_number);

   Matrix<double> c(rows_number-1, columns_number-1);

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

               c(i1,j1) = matrix(ii,jj);
               j1++;
            }
            i1++;
         }

         const double determinant = OpenNN::determinant(c);

         cofactor(i,j) = static_cast<double>((i + j) % 2 == 0 ? 1 : -1)*determinant;
      }
   }

   return cofactor;
}


/// Returns the inverse of a square matrix.
/// An error message is printed if the matrix is singular.

Matrix<double> inverse(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

    if(matrix.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "inverse() method.\n"
              << "Matrix is empty.\n";

       throw logic_error(buffer.str());
    }

   if(rows_number != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "inverse() method.\n"
             << "Matrix must be square.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const double determinant = OpenNN::determinant(matrix);

   if(determinant == 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "inverse() method.\n"
             << "Matrix is singular.\n";

      throw logic_error(buffer.str());
   }

   if(rows_number == 1)
   {
        Matrix<double> inverse(1, 1, 1.0/determinant);

        return inverse;
   }

   // Calculate cofactor matrix

   const Matrix<double> cofactor = OpenNN::cofactor(matrix);

   // Adjoint matrix is the transpose of cofactor matrix

   const Matrix<double> adjoint = cofactor.calculate_transpose();

   // Inverse matrix is adjoint matrix divided by matrix determinant

   const Matrix<double> inverse = adjoint/determinant;

   return inverse;
}


/// Calculates the eigen values of this matrix, which must be squared.
/// Returns a matrix with only one column and rows the same as this matrix with the eigenvalues.

Matrix<double> eigenvalues(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();

    #ifdef __OPENNN_DEBUG__

    if(matrix.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Matrix<double> calculate_eigen_values() method.\n"
              << "Number of columns must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if(matrix.get_rows_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Matrix<double> calculate_eigen_values() method.\n"
              << "Number of rows must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if(matrix.get_columns_number() != matrix.get_rows_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Matrix<double> calculate_eigen_values() method.\n"
              << "The matrix must be squared.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<double> eigenvalues(rows_number, 1);

//    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
//    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> matrix_eigen(this_eigen, Eigen::EigenvaluesOnly);
//    Eigen::Map<Eigen::MatrixXd> eigenvalues_eigen(eigenvalues.data(), rows_number, 1);

//    eigenvalues_eigen = matrix_eigen.eigenvalues();

    return(eigenvalues);
}


/// Calculates the eigenvectors of this matrix, which must be squared.
/// Returns a matrix whose columns are the eigenvectors.

Matrix<double> eigenvectors(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    #ifdef __OPENNN_DEBUG__

    if(columns_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Matrix<double> eigenvectors(const Matrix<double>&) method.\n"
              << "Number of columns must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(rows_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Matrix<double> eigenvectors(const Matrix<double>&) method.\n"
              << "Number of rows must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(columns_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Matrix<double> eigenvectors(const Matrix<double>&) method.\n"
              << "The matrix must be squared.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<double> eigenvectors(rows_number, rows_number);

    const Eigen::Map<Eigen::MatrixXd> this_eigen((double*)matrix.data(), rows_number, columns_number);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> matrix_eigen(this_eigen, Eigen::ComputeEigenvectors);
    Eigen::Map<Eigen::MatrixXd> eigenvectors_eigen(eigenvectors.data(), rows_number, rows_number);

    eigenvectors_eigen = matrix_eigen.eigenvectors();

    return eigenvectors;
}


/// Calculates the direct product of this matrix with another matrix.
/// This product is also known as the Kronecker product.
/// @param other_matrix Second product term.

Matrix<double> direct(const Matrix<double>& matrix, const Matrix<double>& other_matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   const size_t other_rows_number = other_matrix.get_rows_number();
   const size_t other_columns_number = other_matrix.get_columns_number();

   Matrix<double> direct(rows_number*other_rows_number, columns_number*other_columns_number);

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

                   direct(alpha,beta) = matrix(i,j)*other_matrix(k,l);
               }
           }
       }
   }

   return direct;
}


/// Returns the matrix p-norm by rows.

Vector<double> lp_norm(const Matrix<double>& matrix, const double& p)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Vector<double> calculate_lp_norm(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Vector<double> norm(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            norm[i] += pow(abs(matrix(i,j)), p);
        }

        norm[i] = pow(norm[i], 1.0 / p);
    }

    return norm;
}


/// Returns the matrix p-norm by rows.
/// The tensor must be a matrix

Vector<double> lp_norm(const Tensor<double>& matrix, const double& p)
{
    if(matrix.get_dimensions_number() > 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double lp_norm(const Tensor<double>&, const double&) method.\n"
               << "The number os dimensions of the tensor should be 2.\n";

        throw logic_error(buffer.str());
    }

    const size_t rows_number = matrix.get_dimension(0);
    const size_t columns_number = matrix.get_dimension(1);

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0)
      {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Vector<double> calculate_lp_norm(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Vector<double> norm(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            norm[i] += pow(abs(matrix(i,j)), p);
        }

        norm[i] = pow(norm[i], 1.0 / p);
    }

    return norm;
}


/// Returns the gradient of the matrix norm.
/// The tensor must be a matrix.

Tensor<double> lp_norm_gradient(const Tensor<double>& matrix, const double& p)
{
    if(matrix.get_dimensions_number() > 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double lp_norm(const Tensor<double>&, const double&) method.\n"
               << "The number of dimensions of the tensor should be 2.\n";

        throw logic_error(buffer.str());
    }

    const size_t rows_number = matrix.get_dimension(0);
    const size_t columns_number = matrix.get_dimension(1);

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Matrix<double> calculate_p_norm_gradient(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

      Tensor<double> gradient(rows_number, columns_number);

      const Vector<double> p_norm = lp_norm(matrix, p);

      if(p_norm == 0.0)
      {
        gradient.initialize(0.0);
      }
      else
      {
        for(size_t i = 0; i < rows_number; i++)
        {
            for(size_t j = 0; j < columns_number; j++)
            {
                gradient(i,j) = matrix(i,j) * pow(abs(matrix(i,j)), p - 2.0) / pow(p_norm[i], p - 1.0);
            }
        }
      }

      return gradient;
}


/// Calculates matrix_1·matrix_2 + vector
/// The size of the output is rows_1*columns_2.

Tensor<double> linear_combinations(const Tensor<double>& matrix_1, const Matrix<double>& matrix_2, const Vector<double>& vector)
{
    const size_t rows_number_1 = matrix_1.get_dimension(0);
    const size_t columns_number_1 = matrix_1.get_dimension(1);

    const size_t rows_number_2 = matrix_2.get_rows_number();
    const size_t columns_number_2 =matrix_2.get_columns_number();

//    const size_t size = vector.size();

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   if(false)
   {
     buffer << "OpenNN Exception: Metrics functions.\n"
            << "Matrix<double> calculate_p_norm_gradient(const double&) const method.\n"
            << "p value must be greater than zero.\n";

     throw logic_error(buffer.str());
   }

   if(rows_number_2 != columns_number_1)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "Tensor<double> linear_combinations(const Tensor<double>&, const Matrix<double>&, const Vector<double>&) method.\n"
             << "The number of rows of matrix 2 (" << rows_number_2 << ") must be equal to the number of columns of matrix 1 (" << columns_number_1 << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t new_rows_number = rows_number_1;
   const size_t new_columns_number = columns_number_2;

   Tensor<double> new_matrix(new_rows_number, new_columns_number);

   double sum;

   for(size_t i = 0; i < rows_number_1; i++)
   {
     for(size_t j = 0; j < columns_number_2; j++)
     {
        sum = 0.0;

       for(size_t k = 0; k < columns_number_1; k++)
       {
            sum += matrix_1(i,k)*matrix_2(k,j);            
       }

       sum += vector[j];

       new_matrix(i,j) = sum;
     }
   }

   return new_matrix;
}


/// Returns the distance between the elements of this vector and the elements of
/// another vector.
/// @param other_vector Other vector.

double euclidean_distance(const Vector<double>& vector, const Vector<double>& other_vector)
{
    const size_t x_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const size_t y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double euclidean_distance(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < x_size; i++)
    {
        error = vector[i] - other_vector[i];

        distance += error * error;
    }

    return sqrt(distance);
}


double euclidean_weighted_distance(const Vector<double>& vector, const Vector<double>& other_vector, const Vector<double>& weights)
{

    const size_t x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const size_t y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double euclidean_weighted_distance(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < x_size; i++) {
        error = vector[i] - other_vector[i];

        distance += error * error * weights[i];
    }

    return(sqrt(distance));
}


Vector<double> euclidean_weighted_distance_vector(const Vector<double>& vector, const Vector<double>& other_vector, const Vector<double>& weights)
{

    const size_t x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const size_t y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double euclidean_weighted_distance(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Vector<double> distance(x_size,0.0);
    double error;

    for(size_t i = 0; i < x_size; i++) {
        error = vector[i] - other_vector[i];

        distance[i] = error * error * weights[i];
    }

    return(distance);
}


double manhattan_distance(const Vector<double>& vector, const Vector<double>&other_vector)
{

    const size_t x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const size_t y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double manhattan_distance(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < x_size; i++) {
        error = abs(vector[i] - other_vector[i]);

        distance += error;
    }

    return(distance);
}


double manhattan_weighted_distance(const Vector<double>& vector, const Vector<double>& other_vector, const Vector<double>& weights)
{

    const size_t x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const size_t y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double manhattan_weighted_distance(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(size_t i = 0; i < x_size; i++)
    {
        error = abs(vector[i] - other_vector[i]);

        distance += error * weights[i];
    }

    return(distance);
}


Vector<double> manhattan_weighted_distance_vector(const Vector<double>& vector, const Vector<double>& other_vector, const Vector<double>& weights)
{

    const size_t x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const size_t y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double manhattan_weighted_distance(const Vector<double>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Vector<double> distance(x_size,0.0);
    double error;

    for(size_t i = 0; i < x_size; i++) {
        error = abs(vector[i] - other_vector[i]);

//        if(i==0) cout << error << endl;

        distance[i] = error * weights[i];
    }

    return(distance);
}


/// Returns the sum squared error between the elements of this vector and the
/// elements of another vector.
/// @param other_vector Other vector.

double sum_squared_error(const Vector<double>& x, const Vector<double>& y)
{
  const size_t x_size = x.size();

#ifdef __OPENNN_DEBUG__

  const size_t y_size = y.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double sum_squared_error(const Vector<double>&) const method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum_squared_error = 0.0;
  double error;

  for(size_t i = 0; i < x_size; i++)
  {
    error = x[i] - y[i];

    sum_squared_error += error * error;
  }

  return sum_squared_error;
}


/// Returns the Minkowski squared error between the elements of this vector and
/// the elements of another vector.
/// @param vector This vector.
/// @param other_vector Other vector.
/// @param minkowski_parameter Minkowski exponent.

double minkowski_error(const Vector<double>& vector,
                       const Vector<double>& other_vector,
                       const double& minkowski_parameter)
{
  const size_t x_size = vector.size();

#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(x_size == 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

  const size_t y_size = other_vector.size();

  if(y_size != x_size) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "Other size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

  // Control sentence

  if(minkowski_parameter < 1.0 || minkowski_parameter > 2.0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double minkowski_error(const Vector<double>&) const "
              "method.\n"
           << "The Minkowski parameter must be comprised between 1 and 2\n";

    throw logic_error(buffer.str());
  }

#endif

  double minkowski_error = 0.0;

  for(size_t i = 0; i < x_size; i++)
  {
    minkowski_error +=
        pow(abs(vector[i] - other_vector[i]), minkowski_parameter);
  }

  minkowski_error = pow(minkowski_error, 1.0 / minkowski_parameter);

  return(minkowski_error);
}


double sum_squared_error(const Tensor<double>& x, const Tensor<double>& y)
{
    const size_t size = x.size();

    double sum_squared_error = 0.0;

    double error;

    for(size_t i = 0; i < size; i++)
    {
        error = x[i]-y[i];

        sum_squared_error += error*error;
    }

    return sum_squared_error;
}


Vector<double> euclidean_distance(const Matrix<double>& matrix, const Vector<double>& instance)
{
    const size_t rows_number = matrix.get_rows_number();

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_distance(const Vector<double>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = euclidean_distance(matrix.get_row(i), instance);
    }

    return distances;
}


Vector<double> euclidean_distance(const Matrix<double>& matrix, const Matrix<double>& other_matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<double> distances(rows_number, 0.0);
    double error;

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            error = matrix(i,j) - other_matrix(i,j);

            distances[i] += error * error;
        }

        distances[i] = sqrt(distances[i]);
    }

    return distances;
}


Vector<double> euclidean_weighted_distance(const Matrix<double>& matrix, const Vector<double>& instance, const Vector<double>& weights)
{
    const size_t rows_number = matrix.get_rows_number();

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_weighted_distance(const Vector<double>&, const Vector<double>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = euclidean_weighted_distance(matrix.get_row(i), instance, weights);
    }

    return distances;
}


Matrix<double> euclidean_weighted_distance_matrix(const Matrix<double>& matrix, const Vector<double>& instance, const Vector<double>& weights)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_weighted_distance(const Vector<double>&, const Vector<double>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Matrix<double> distances(rows_number,columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances.set_row(i, euclidean_weighted_distance_vector(matrix.get_row(i), instance,weights));
    }

    return distances;
}


/// Calculates the distance between two rows in the matrix

double manhattan_distance(const Matrix<double>& matrix, const size_t& first_index, const size_t& second_index)
{
    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_distance(const size_t&, const size_t&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Vector<double> first_row = matrix.get_row(first_index);
    const Vector<double> second_row = matrix.get_row(second_index);

    return manhattan_distance(first_row, second_row);
}


Vector<double> manhattan_distance(const Matrix<double>& matrix, const Vector<double>& instance)
{
    const size_t rows_number = matrix.get_rows_number();

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_distance(const size_t&, const size_t&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = manhattan_distance(matrix.get_row(i), instance);
    }

    return distances;
}


Vector<double> manhattan_weighted_distance(const Matrix<double>& matrix, const Vector<double>& instance, const Vector<double>& weights)
{
    const size_t rows_number = matrix.get_rows_number();

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_weighted_distance(const Vector<double>&, const Vector<double>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Vector<double> distances(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances[i] = manhattan_weighted_distance(matrix.get_row(i), instance, weights);
    }

    return distances;
}


Matrix<double> manhattan_weighted_distance_matrix(const Matrix<double>& matrix, const Vector<double>& instance, const Vector<double>& weights)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_weighted_distance(const Vector<double>&, const Vector<double>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Matrix<double> distances(rows_number,columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        distances.set_row(i,manhattan_weighted_distance_vector(matrix.get_row(i),instance,weights));
    }

    return distances;
}


Vector<double> error_rows(const Tensor<double>& matrix, const Tensor<double>& other_matrix)
{
    const size_t rows_number = matrix.get_dimension(0);
    const size_t columns_number = matrix.get_dimension(1);

    Vector<double> error_rows(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            error_rows[i] += (matrix(i,j) - other_matrix(i,j))*(matrix(i,j) - other_matrix(i,j));
        }

        error_rows[i] = sqrt(error_rows[i]);
    }

    return error_rows;
}


Vector<double> weighted_error_rows(const Tensor<double>& matrix, const Tensor<double>& other_matrix, const double& weight1, const double& weight2)
{
    const size_t rows_number = matrix.get_dimension(0);
    const size_t columns_number = matrix.get_dimension(1);

    Vector<double> weighted_error_rows(rows_number, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j =0; j < columns_number; j++)
        {
            if(other_matrix(i,j) == 1.0)
            {
                weighted_error_rows[i] += weight1*(matrix(i,j) - other_matrix(i,j))*(matrix(i,j) - other_matrix(i,j));
            }
            else
            {
                weighted_error_rows[i] += weight2*(matrix(i,j) - other_matrix(i,j))*(matrix(i,j) - other_matrix(i,j));
            }
        }

        weighted_error_rows[i] = sqrt(weighted_error_rows[i]);
    }

    return weighted_error_rows;
}


double cross_entropy_error(const Tensor<double>& x, const Tensor<double>& y)
{
    const size_t x_rows_number = x.get_dimension(0);
    const size_t x_columns_number = x.get_dimension(1);

    #ifdef __OPENNN_DEBUG__

    const size_t y_rows_number = y.get_dimension(0);

    if(y_rows_number != x_rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double cross_entropy_error(const Tensor<double>&, const Tensor<double>&) method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const size_t y_columns_number = y.get_dimension(1);

    if(y_columns_number != x_columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double cross_entropy_error(const Tensor<double>&, const Tensor<double>&) method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    double cross_entropy_error = 0.0;

    for(size_t i = 0; i < x_rows_number; i++)
    {
        for(size_t j = 0; j < x_columns_number; j++)
        {
            const double y_value = y(static_cast<unsigned>(i), static_cast<unsigned>(j));
            const double x_value = x(i,j);

            if(y_value == 0.0 && x_value == 0.0)
            {
                cross_entropy_error -= 0.0;
            }
            else if(y_value == 1.0 && x_value == 1.0)
            {
                cross_entropy_error -= 0.0;
            }
            else if(x_value == 0.0)
            {
                cross_entropy_error -= (1.0 - y_value)*log(1.0-x_value) + y_value*log(0.000000001);
            }
            else if(x_value == 1.0)
            {
                cross_entropy_error -= (1.0 - y_value)*log(1.0-x_value) + y_value*log(0.999999999);
            }
            else
            {
                cross_entropy_error -= (1.0 - y_value)*log(1.0-x_value) + y_value*log(x_value);
            }

        }
    }

    return cross_entropy_error;
}


/// Returns the minkowski error between the elements of this tensor and the elements of another tensor.
/// @param x Tensor.
/// @param y Other tensor.
/// @param minkowski_parameter Minkowski exponent value.

double minkowski_error(const Tensor<double>& x, const Tensor<double>& y, const double& minkowski_parameter)
{
    const size_t rows_number = x.get_dimension(0);
    const size_t columns_number = x.get_dimension(1);

#ifdef __OPENNN_DEBUG__

    const size_t other_rows_number = y.get_dimension(0);

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Tensor<double>&, const Tensor<double>&, const double&) method.\n"
              << "Other number of rows " << other_rows_number << " must be equal to this number of rows " << rows_number << ".\n";

       throw logic_error(buffer.str());
    }

    const size_t other_columns_number = y.get_dimension(1);

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Tensor<double>&, const Tensor<double>&, const double&) method.\n"
              << "Other number of columns (" << other_columns_number << ") must be equal to this number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    double minkowski_error = 0.0;
    double row_minkowski_error = 0.0;

    for(size_t i = 0; i < rows_number; i++)
    {
        row_minkowski_error = 0.0;

        for(size_t j = 0; j < columns_number; j++)
        {
            row_minkowski_error += pow(abs(x(i,j) - y(i,j)), minkowski_parameter);
        }

        minkowski_error += pow(row_minkowski_error, 1.0 / minkowski_parameter);
    }

    return minkowski_error;
}


double weighted_sum_squared_error(const Tensor<double>& x, const Tensor<double>& y, const double& positives_weight, const double& negatives_weight)
{
#ifdef __OPENNN_DEBUG__

    const size_t rows_number = x.get_dimension(0);
    const size_t columns_number = x.get_dimension(1);

    const size_t other_rows_number = y.get_dimension(0);

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Matrix<double>&, const double&) method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const size_t other_columns_number = y.get_dimension(1);

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Matrix<double>&, const double&) method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    double weighted_sum_squared_error = 0.0;

    double error = 0.0;

    for(size_t i = 0; i < x.size(); i++)
    {
        error = x[i] - y[i];

        if(y[i] == 1.0)
        {
            weighted_sum_squared_error += positives_weight*error*error;
        }
        else if(y[i] == 0.0)
        {
            weighted_sum_squared_error += negatives_weight*error*error;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: Metrics functions.\n"
                   << "double calculate_error() method.\n"
                   << "Other matrix is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }
    }

    return weighted_sum_squared_error;
}


/// Returns the gradient of the vector norm.

Vector<double> l1_norm_gradient(const Vector<double>& vector)
{
  return sign(vector);
}


/// Returns the hessian of the vector norm.

Matrix<double> l1_norm_hessian(const Vector<double>& vector)
{
  const size_t x_size = vector.size();

  return Matrix<double>(x_size, x_size, 0);
}

}
