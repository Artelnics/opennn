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

double sum_squared_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const auto error = y - x;

    const Eigen::array<Eigen::IndexPair<int>, 2> product_dimensions = { Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1) };

    const Tensor<type, 0> sse = error.contract(error, product_dimensions);

    return sse(0);
}


double l2_norm(const ThreadPoolDevice& threadPoolDevice, const Tensor<type, 1>& x)
{
   Tensor<type, 0> y;

   y.device(threadPoolDevice) = x.square().sum(Eigen::array<int, 1>({0}));

   return y(0);

 /*
  const int x_size = vector.size();

  //vector.sum()

  double norm = 0.0;

  for(int i = 0; i < x_size; i++) {
    norm += vector[i] *vector[i];
  }

    return sqrt(norm);
*/
}



/*
double dot(const Tensor<type, 1>& a, const Tensor<type, 1>& b)
{
    const int a_size = a.size();

  #ifdef __OPENNN_DEBUG__

    const int b_size = b.size();

    if(a_size != b_size)
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "double dot(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
             << "Both vector sizes must be the same.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double dot_product = 0.0;

    for(int i = 0; i < a_size; i++)
    {
      dot_product += a[i] * b[i];
    }

    return dot_product;
}


Tensor<type, 1> dot(const Tensor<type, 1>& vector, const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

  #ifdef __OPENNN_DEBUG__
    const int vector_size = vector.size();

    if(rows_number != vector_size)
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "Tensor<type, 1> dot(const Tensor<type, 1>&, const Tensor<type, 2>&) method.\n"
             << "Matrix number of rows (" << rows_number << ") must be equal to vector size (" << vector_size << ").\n";

      throw logic_error(buffer.str());
    }

  #endif

    Tensor<type, 1> product(columns_number, 0.0);

     for(int j = 0; j < columns_number; j++)
     {
        for(int i = 0; i < rows_number; i++)
        {
           product[j] += vector[i]*matrix(i,j);
       }
     }

    return product;
}


Tensor<type, 1> dot(const Tensor<type, 2>& matrix, const Tensor<type, 1>& vector)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 1> product(rows_number);

    const Map<MatrixXd> matrix_eigen((double*)matrix.data(), static_cast<int>(rows_number), static_cast<int>(columns_number));
    const Map<VectorXd> vector_eigen((double*)vector.data(), static_cast<int>(columns_number));
    Map<VectorXd> product_eigen(product.data(), static_cast<int>(rows_number));

    product_eigen = matrix_eigen*vector_eigen;

    return product;
}


Tensor<type, 2> dot(const Tensor<type, 2>& matrix_1, const Tensor<type, 2>& matrix_2)
{
    const int rows_number_1 = matrix_1.dimension(0);
    const int columns_number_1 = matrix_1.dimension(1);

    const int rows_number_2 = matrix_2.dimension(0);
    const int columns_number_2 = matrix_2.dimension(1);

    Tensor<type, 2> product(rows_number_1, columns_number_2);

    const Map<MatrixXd> eigen_1((double*)matrix_1.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_1));
    const Map<MatrixXd> eigen_2((double*)matrix_2.data(), static_cast<int>(rows_number_2), static_cast<int>(columns_number_2));
    Map<MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_2));

    product_eigen = eigen_1*eigen_2;

    return product;
}


void dot(const Tensor<type, 2>& matrix_1, const MatrixXd& matrix_2, Tensor<type, 2>& result)
{
    const int rows_number_1 = matrix_1.dimension(0);
    const int columns_number_1 = matrix_1.dimension(1);

    const int rows_number_2 = matrix_2.rows();
    const int columns_number_2 = matrix_2.cols();

    const Map<MatrixXd> eigen_1((double*)matrix_1.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_1));
//    const Map<MatrixXd> eigen_2((double*)matrix_2.data(), static_cast<int>(rows_number_2), static_cast<int>(columns_number_2));
    Map<MatrixXd> product_eigen(result.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_2));

    product_eigen = eigen_1*matrix_2;
}


Tensor<type, 2> dot(const Tensor<type, 2>& matrix_1, const MatrixXd& matrix_2)
{

    const int rows_number_1 = matrix_1.dimension(0);
    const int columns_number_1 = matrix_1.dimension(1);

    const int rows_number_2 = matrix_2.rows();
    const int columns_number_2 = matrix_2.cols();

    Tensor<type, 2> product(rows_number_1, columns_number_2);

    const Map<MatrixXd> eigen_1((double*)matrix_1.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_1));
//    const Map<MatrixXd> eigen_2((double*)matrix_2.data(), static_cast<int>(rows_number_2), static_cast<int>(columns_number_2));
    Map<MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_2));

    product_eigen = eigen_1*matrix_2;

    return product;

//    MatrixXd eigen_1 = tensor_to_eigen(matrix_1);
//    MatrixXd eigen_2 = matrix_to_eigen(matrix_2);

//    return eigen_to_tensor(eigen_1*eigen_2);

}


void dot(const Tensor<type, 2>& matrix_1, const MatrixXd& matrix_2, Tensor<type, 2>& product)
{
    const int rows_number_1 = matrix_1.dimension(0);
    const int columns_number_1 = matrix_1.dimension(1);

    const int rows_number_2 = matrix_2.rows();
    const int columns_number_2 = matrix_2.cols();

    const Map<MatrixXd> eigen_1((double*)matrix_1.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_1));
//    const Map<MatrixXd> eigen_2((double*)matrix_2.data(), static_cast<int>(rows_number_2), static_cast<int>(columns_number_2));
    Map<MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_2));

    product_eigen = eigen_1*matrix_2;
}


void dot(const Tensor<type, 2>& matrix_1, const Tensor<type, 2>& matrix_2, Tensor<type, 2>& product)
{
    const int rows_number_1 = matrix_1.dimension(0);
    const int columns_number_1 = matrix_1.dimension(1);

    const int rows_number_2 = matrix_2.dimension(0);
    const int columns_number_2 = matrix_2.dimension(1);

    const Map<MatrixXd> eigen_1((double*)matrix_1.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_1));
    const Map<MatrixXd> eigen_2((double*)matrix_1.data(), static_cast<int>(rows_number_2), static_cast<int>(columns_number_2));
    Map<MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number_1), static_cast<int>(columns_number_2));

    product_eigen = eigen_1*eigen_2;
}


Tensor<type, 2> dot_2d_2d(const Tensor<type, 2>& tensor_1, const Tensor<type, 2>& tensor_2)
{
#ifdef __OPENNN_DEBUG__

    const int dimensions_number_1 = tensor_1.rank();
    const int dimensions_number_2 = tensor_2.rank();

  if(dimensions_number_1 != 2)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<type, 2> dot(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
           << "Dimensions number of tensor 1 (" << dimensions_number_1 << ") must be 2.\n";

    throw logic_error(buffer.str());
  }

  if(dimensions_number_2 != 2)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<type, 2> dot(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
           << "Dimensions number of tensor 2 (" << dimensions_number_2 << ") must be 2.\n";

    throw logic_error(buffer.str());
  }

#endif

    const int rows_number = tensor_1.dimension(0);
    const int columns_number = tensor_1.dimension(1);

    const int other_rows_number = tensor_2.dimension(0);
    const int other_columns_number = tensor_2.dimension(1);

    Tensor<type, 2> product(rows_number, other_columns_number);

    const Map<MatrixXd> eigen_1((double*)tensor_1.data(), static_cast<int>(rows_number), static_cast<int>(columns_number));
    const Map<MatrixXd> eigen_2((double*)tensor_2.data(), static_cast<int>(other_rows_number), static_cast<int>(other_columns_number));
    Map<MatrixXd> product_eigen(product.data(), static_cast<int>(rows_number), static_cast<int>(other_columns_number));

    product_eigen = eigen_1*eigen_2;

    return product;
}


Tensor<type, 2> dot_2d_3d(const Tensor<type, 2>& tensor_1, const Tensor<type, 2>& tensor_2)
{
#ifdef __OPENNN_DEBUG__

    const int dimensions_number_1 = tensor_1.rank();
    const int dimensions_number_2 = tensor_2.rank();

  if(dimensions_number_1 != 2)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<type, 2> dot(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
           << "Dimensions number of tensor 1 (" << dimensions_number_1 << ") must be 2.\n";

    throw logic_error(buffer.str());
  }

  if(dimensions_number_2 != 3)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<type, 2> dot(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
           << "Dimensions number of tensor 2 (" << dimensions_number_2 << ") must be 3.\n";

    throw logic_error(buffer.str());
  }

#endif

  const int n = tensor_2.dimensions()[2];

  const auto& dimensions_1 = tensor_1.dimensions();
  const auto& dimensions_2 = tensor_2.dimensions();

  const Tensor<type, 2> tensor_1_matrix = tensor_1.get_matrix(0);

  Tensor<type, 2> product(dimensions_1[0], dimensions_2[1]);

  for(int i = 0; i < n; i ++)
  {
      const Tensor<type, 2> i_matrix = tensor_2.get_matrix(i);

      const Tensor<type, 2> i_row = tensor_1_matrix.get_submatrix_rows({i});

      const Tensor<type, 2> dot_product = dot(i_row, i_matrix);

      for(int k = 0; k < dimensions_2[1]; k++)
      {
          product(i,k) = dot_product(0,k);
      }
  }

  return product;
}


Tensor<type, 2> dot(const Tensor<type, 2>& matrix, const Tensor<type, 2>& tensor)
{

    const int order = tensor.rank();

    if(order == 2)
    {
        return dot(matrix, tensor.get_matrix(0));
    }
    else if(order > 2)
    {

        const int n = tensor.dimensions()[2];

        Tensor<type, 2> outputs(n, matrix.dimension(1));

        for(int i = 0; i < n; i ++)
        {
            const Tensor<type, 2> i_matrix = tensor.get_matrix(i);

            const Tensor<type, 2> i_row = matrix.get_submatrix_rows({i});

            Tensor<type, 2> dot_product = dot(i_row, i_matrix);

            outputs.set_row(i, dot_product.to_vector());
        }

        return outputs;
    }

    return Tensor<type, 2>();

}


/// Returns the vector norm.

double l1_norm(const Tensor<type, 1>& vector)
{
  return absolute_value(vector).calculate_sum();
}


/// Returns the vector norm.

double l2_norm(const Tensor<type, 1>& vector)
{
  const int x_size = vector.size();

  double norm = 0.0;

  for(int i = 0; i < x_size; i++) {
    norm += vector[i] *vector[i];
  }

    return sqrt(norm);
}


/// Returns the gradient of the vector norm.

Tensor<type, 1> l2_norm_gradient(const Tensor<type, 1>& vector)
{
  const int x_size = vector.size();

  Tensor<type, 1> gradient(x_size);

  const double norm = l2_norm(vector);

  if(norm == 0.0) {
    gradient.setZero();
  } else {
    gradient = vector/ norm;
  }

  return gradient;
}


/// Returns the hessian of the vector norm.

Tensor<type, 2> l2_norm_hessian(const Tensor<type, 1>& vector)
{
  const int x_size = vector.size();

  Tensor<type, 2> hessian(x_size, x_size);

  const double norm = l2_norm(vector);

  if(norm == 0.0) {
    hessian.setZero();
  } else {
    hessian = direct(vector, vector)/(norm * norm * norm);
  }

  return(hessian);
}


/// Returns the vector p-norm.

double lp_norm(const Tensor<type, 1>& vector, const double &p)
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

  const int x_size = vector.size();

  double norm = 0.0;

  for(int i = 0; i < x_size; i++) {
    norm += pow(abs(vector[i]), p);
  }

  norm = pow(norm, 1.0 / p);

  return norm;
}


/// Returns the gradient of the vector norm.

Tensor<type, 1> lp_norm_gradient(const Tensor<type, 1>& vector, const double &p)
{
#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<type, 1> calculate_p_norm_gradient(const double&) const "
              "method.\n"
           << "p value must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const int x_size = vector.size();

  Tensor<type, 1> gradient(x_size);

  const double p_norm = lp_norm(vector, p);

  if(p_norm == 0.0)
  {
    gradient.setZero();
  }
  else
  {
    for(int i = 0; i < x_size; i++)
    {
      gradient[i] =
         vector[i] * pow(abs(vector[i]), p - 2.0) / pow(p_norm, p - 1.0);
    }
  }

  return gradient;
}


/// Outer product vector*vector arithmetic operator.
/// @param other_vector vector to be multiplied to this vector.

Tensor<type, 2> direct(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
  const int x_size = x.size();

#ifdef __OPENNN_DEBUG__

  const int y_size = y.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<type, 2> direct(const Tensor<type, 1>&) method.\n"
           << "Both vector sizes must be the same.\n";

    throw logic_error(buffer.str());
  }

#endif

  Tensor<type, 2> direct(x_size, x_size);

   #pragma omp parallel for if(x_size > 1000)

  for(int i = 0; i < static_cast<int>(x_size); i++)
  {
    for(int j = 0; j < x_size; j++)
    {
      direct(static_cast<int>(i), j) = x[static_cast<int>(i)] * y[j];
    }
  }

  return direct;
}


/// Returns the determinant of a square matrix.

double determinant(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

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

      for(int row_index = 0; row_index < rows_number; row_index++)
      {
         // Calculate sub data

         Tensor<type, 2> sub_matrix(rows_number-1, columns_number-1);

         for(int i = 1; i < rows_number; i++)
         {
            int j2 = 0;

            for(int j = 0; j < columns_number; j++)
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

Tensor<type, 2> cofactor(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   Tensor<type, 2> cofactor(rows_number, columns_number);

   Tensor<type, 2> c(rows_number-1, columns_number-1);

   for(int j = 0; j < rows_number; j++)
   {
      for(int i = 0; i < rows_number; i++)
      {
         // Form the adjoint a(i,j)

         int i1 = 0;

         for(int ii = 0; ii < rows_number; ii++)
         {
            if(ii == i) continue;

            int j1 = 0;

            for(int jj = 0; jj < rows_number; jj++)
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

Tensor<type, 2> inverse(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

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
        Tensor<type, 2> inverse(1, 1, 1.0/determinant);

        return inverse;
   }

   // Calculate cofactor matrix

   const Tensor<type, 2> cofactor = OpenNN::cofactor(matrix);

   // Adjoint matrix is the transpose of cofactor matrix

   const Tensor<type, 2> adjoint = cofactor.calculate_transpose();

   // Inverse matrix is adjoint matrix divided by matrix determinant

   const Tensor<type, 2> inverse = adjoint/determinant;

   return inverse;
}


/// Calculates the eigen values of this matrix, which must be squared.
/// Returns a matrix with only one column and rows the same as this matrix with the eigenvalues.

Tensor<type, 2> eigenvalues(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__

    if(matrix.dimension(1) == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Tensor<type, 2> calculate_eigen_values() method.\n"
              << "Number of columns must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if(matrix.dimension(0) == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Tensor<type, 2> calculate_eigen_values() method.\n"
              << "Number of rows must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    if(matrix.dimension(1) != matrix.dimension(0))
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Tensor<type, 2> calculate_eigen_values() method.\n"
              << "The matrix must be squared.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Tensor<type, 2> eigenvalues(rows_number, 1);

//    const Map<MatrixXd> this_eigen((double*)this->data(), rows_number, columns_number);
//    const SelfAdjointEigenSolver<MatrixXd> matrix_eigen(this_eigen, EigenvaluesOnly);
//    Map<MatrixXd> eigenvalues_eigen(eigenvalues.data(), rows_number, 1);

//    eigenvalues_eigen = matrix_eigen.eigenvalues();

    return(eigenvalues);
}


/// Calculates the eigenvectors of this matrix, which must be squared.
/// Returns a matrix whose columns are the eigenvectors.

Tensor<type, 2> eigenvectors(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

    if(columns_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Tensor<type, 2> eigenvectors(const Tensor<type, 2>&) method.\n"
              << "Number of columns must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(rows_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Tensor<type, 2> eigenvectors(const Tensor<type, 2>&) method.\n"
              << "Number of rows must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(columns_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "Tensor<type, 2> eigenvectors(const Tensor<type, 2>&) method.\n"
              << "The matrix must be squared.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Tensor<type, 2> eigenvectors(rows_number, rows_number);

    const Map<MatrixXd> this_eigen((double*)matrix.data(), rows_number, columns_number);
    const SelfAdjointEigenSolver<MatrixXd> matrix_eigen(this_eigen, ComputeEigenvectors);
    Map<MatrixXd> eigenvectors_eigen(eigenvectors.data(), rows_number, rows_number);

    eigenvectors_eigen = matrix_eigen.eigenvectors();

    return eigenvectors;
}


/// Calculates the direct product of this matrix with another matrix.
/// This product is also known as the Kronecker product.
/// @param other_matrix Second product term.

Tensor<type, 2> direct(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   const int other_rows_number = other_matrix.dimension(0);
   const int other_columns_number = other_matrix.dimension(1);

   Tensor<type, 2> direct(rows_number*other_rows_number, columns_number*other_columns_number);

   int alpha;
   int beta;

   for(int i = 0; i < rows_number; i++)
   {
       for(int j = 0; j < columns_number; j++)
       {
           for(int k = 0; k < other_rows_number; k++)
           {
               for(int l = 0; l < other_columns_number; l++)
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

Tensor<type, 1> lp_norm(const Tensor<type, 2>& matrix, const double& p)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Tensor<type, 1> calculate_lp_norm(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Tensor<type, 1> norm(rows_number, 0.0);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < columns_number; j++)
        {
            norm[i] += pow(abs(matrix(i,j)), p);
        }

        norm[i] = pow(norm[i], 1.0 / p);
    }

    return norm;
}


/// Returns the matrix p-norm by rows.
/// The tensor must be a matrix

Tensor<type, 1> lp_norm(const Tensor<type, 2>& matrix, const double& p)
{
    if(matrix.rank() > 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double lp_norm(const Tensor<type, 2>&, const double&) method.\n"
               << "The number os dimensions of the tensor should be 2.\n";

        throw logic_error(buffer.str());
    }

    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0)
      {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Tensor<type, 1> calculate_lp_norm(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Tensor<type, 1> norm(rows_number, 0.0);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < columns_number; j++)
        {
            norm[i] += pow(abs(matrix(i,j)), p);
        }

        norm[i] = pow(norm[i], 1.0 / p);
    }

    return norm;
}


/// Returns the gradient of the matrix norm.
/// The tensor must be a matrix.

Tensor<type, 2> lp_norm_gradient(const Tensor<type, 2>& matrix, const double& p)
{
    if(matrix.rank() > 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double lp_norm(const Tensor<type, 2>&, const double&) method.\n"
               << "The number of dimensions of the tensor should be 2.\n";

        throw logic_error(buffer.str());
    }

    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Tensor<type, 2> calculate_p_norm_gradient(const double&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

      Tensor<type, 2> gradient(rows_number, columns_number);

      const Tensor<type, 1> p_norm = lp_norm(matrix, p);

      if(p_norm == 0.0)
      {
        gradient.setZero();
      }
      else
      {
        for(int i = 0; i < rows_number; i++)
        {
            for(int j = 0; j < columns_number; j++)
            {
                gradient(i,j) = matrix(i,j) * pow(abs(matrix(i,j)), p - 2.0) / pow(p_norm[i], p - 1.0);
            }
        }
      }

      return gradient;
}


/// Calculates matrix_1·matrix_2 + vector
/// The size of the output is rows_1*columns_2.

Tensor<type, 2> linear_combinations(const Tensor<type, 2>& matrix_1, const MatrixXd& matrix_2, const Tensor<type, 1>& vector)
{
    const int rows_number_1 = matrix_1.dimension(0);
    const int columns_number_1 = matrix_1.dimension(1);

    const int columns_number_2 =matrix_2.cols();

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const int rows_number_2 = matrix_2.rows();

   if(rows_number_2 != columns_number_1)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "Tensor<type, 2> linear_combinations(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 1>&) method.\n"
             << "The number of rows of matrix 2 (" << rows_number_2 << ") must be equal to the number of columns of matrix 1 (" << columns_number_1 << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Tensor<type, 2> new_matrix = dot(matrix_1, matrix_2);

   new_matrix += vector;

   return new_matrix;


}


void linear_combinations(const Tensor<type, 2>& matrix_1, const MatrixXd& matrix_2, const Tensor<type, 1>& vector, Tensor<type, 2>& new_matrix)
{
    const int rows_number_1 = matrix_1.dimension(0);
    const int columns_number_1 = matrix_1.dimension(1);

    const int columns_number_2 =matrix_2.cols();

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const int rows_number_2 = matrix_2.rows();

   if(rows_number_2 != columns_number_1)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Metrics functions.\n"
             << "void linear_combinations(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 1>&, Tensor<type, 2>&) method.\n"
             << "The number of rows of matrix 2 (" << rows_number_2 << ") must be equal to the number of columns of matrix 1 (" << columns_number_1 << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   double sum;

#pragma omp parallel for

   for(int i = 0; i < rows_number_1; i++)
   {
     for(int j = 0; j < columns_number_2; j++)
     {
        sum = 0.0;

       for(int k = 0; k < columns_number_1; k++)
       {
            sum += matrix_1(i,k)*matrix_2(k,j);
       }

       sum += vector[j];

       new_matrix(i,j) = sum;
     }
   }
}



/// Returns the distance between the elements of this vector and the elements of
/// another vector.
/// @param other_vector Other vector.

double euclidean_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>& other_vector)
{
    const int x_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const int y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double euclidean_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(int i = 0; i < x_size; i++)
    {
        error = vector[i] - other_vector[i];

        distance += error * error;
    }

    return sqrt(distance);
}


double euclidean_weighted_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>& other_vector, const Tensor<type, 1>& weights)
{

    const int x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const int y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double euclidean_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(int i = 0; i < x_size; i++) {
        error = vector[i] - other_vector[i];

        distance += error * error * weights[i];
    }

    return(sqrt(distance));
}


Tensor<type, 1> euclidean_weighted_distance_vector(const Tensor<type, 1>& vector, const Tensor<type, 1>& other_vector, const Tensor<type, 1>& weights)
{

    const int x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const int y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double euclidean_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Tensor<type, 1> distance(x_size,0.0);
    double error;

    for(int i = 0; i < x_size; i++) {
        error = vector[i] - other_vector[i];

        distance[i] = error * error * weights[i];
    }

    return(distance);
}


double manhattan_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>&other_vector)
{

    const int x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const int y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double manhattan_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(int i = 0; i < x_size; i++) {
        error = abs(vector[i] - other_vector[i]);

        distance += error;
    }

    return(distance);
}


double manhattan_weighted_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>& other_vector, const Tensor<type, 1>& weights)
{

    const int x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const int y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double manhattan_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double distance = 0.0;
    double error;

    for(int i = 0; i < x_size; i++)
    {
        error = abs(vector[i] - other_vector[i]);

        distance += error * weights[i];
    }

    return(distance);
}


Tensor<type, 1> manhattan_weighted_distance_vector(const Tensor<type, 1>& vector, const Tensor<type, 1>& other_vector, const Tensor<type, 1>& weights)
{

    const int x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const int y_size = other_vector.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double manhattan_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Tensor<type, 1> distance(x_size,0.0);
    double error;

    for(int i = 0; i < x_size; i++) {
        error = abs(vector[i] - other_vector[i]);

//        if(i==0) cout << error << endl;

        distance[i] = error * weights[i];
    }

    return(distance);
}


/// Returns the sum squared error between the elements of this vector and the
/// elements of another vector.
/// @param other_vector Other vector.

double sum_squared_error(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
  const int x_size = x.size();

#ifdef __OPENNN_DEBUG__

  const int y_size = y.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double sum_squared_error(const Tensor<type, 1>&) const method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum_squared_error = 0.0;
  double error;

  for(int i = 0; i < x_size; i++)
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

double minkowski_error(const Tensor<type, 1>& vector,
                       const Tensor<type, 1>& other_vector,
                       const double& minkowski_parameter)
{
  const int x_size = vector.size();

#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(x_size == 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double minkowski_error(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

  const int y_size = other_vector.size();

  if(y_size != x_size) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double minkowski_error(const Tensor<type, 1>&) const "
              "method.\n"
           << "Other size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

  // Control sentence

  if(minkowski_parameter < 1.0 || minkowski_parameter > 2.0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "double minkowski_error(const Tensor<type, 1>&) const "
              "method.\n"
           << "The Minkowski parameter must be comprised between 1 and 2\n";

    throw logic_error(buffer.str());
  }

#endif

  double minkowski_error = 0.0;

  for(int i = 0; i < x_size; i++)
  {
    minkowski_error +=
        pow(abs(vector[i] - other_vector[i]), minkowski_parameter);
  }

  minkowski_error = pow(minkowski_error, 1.0 / minkowski_parameter);

  return(minkowski_error);
}


double sum_squared_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const int size = x.size();

    double sum_squared_error = 0.0;

    #pragma omp parallel for reduction(+ : sum_squared_error)

    for(int i = 0; i < size; i++)
    {
        const double error = y[i] - x[i];

        sum_squared_error += error*error;
    }

    return sum_squared_error;
}


Tensor<type, 1> euclidean_distance(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_distance(const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 1> distances(rows_number);

    for(int i = 0; i < rows_number; i++)
    {
        distances[i] = euclidean_distance(matrix.get_row(i), instance);
    }

    return distances;
}


Tensor<type, 1> euclidean_distance(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 1> distances(rows_number, 0.0);
    double error;

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < columns_number; j++)
        {
            error = matrix(i,j) - other_matrix(i,j);

            distances[i] += error * error;
        }

        distances[i] = sqrt(distances[i]);
    }

    return distances;
}


Tensor<type, 1> euclidean_weighted_distance(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance, const Tensor<type, 1>& weights)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 1> distances(rows_number);

    for(int i = 0; i < rows_number; i++)
    {
        distances[i] = euclidean_weighted_distance(matrix.get_row(i), instance, weights);
    }

    return distances;
}


Tensor<type, 2> euclidean_weighted_distance_matrix(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance, const Tensor<type, 1>& weights)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 2> distances(rows_number,columns_number);

    for(int i = 0; i < rows_number; i++)
    {
        distances.set_row(i, euclidean_weighted_distance_vector(matrix.get_row(i), instance,weights));
    }

    return distances;
}


/// Calculates the distance between two rows in the matrix

double manhattan_distance(const Tensor<type, 2>& matrix, const int& first_index, const int& second_index)
{
    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_distance(const int&, const int&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Tensor<type, 1> first_row = matrix.get_row(first_index);
    const Tensor<type, 1> second_row = matrix.get_row(second_index);

    return manhattan_distance(first_row, second_row);
}


Tensor<type, 1> manhattan_distance(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_distance(const int&, const int&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 1> distances(rows_number);

    for(int i = 0; i < rows_number; i++)
    {
        distances[i] = manhattan_distance(matrix.get_row(i), instance);
    }

    return distances;
}


Tensor<type, 1> manhattan_weighted_distance(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance, const Tensor<type, 1>& weights)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 1> distances(rows_number);

    for(int i = 0; i < rows_number; i++)
    {
        distances[i] = manhattan_weighted_distance(matrix.get_row(i), instance, weights);
    }

    return distances;
}


Tensor<type, 2> manhattan_weighted_distance_matrix(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance, const Tensor<type, 1>& weights)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

     if(matrix.empty())
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 2> distances(rows_number,columns_number);

    for(int i = 0; i < rows_number; i++)
    {
        distances.set_row(i,manhattan_weighted_distance_vector(matrix.get_row(i),instance,weights));
    }

    return distances;
}


Tensor<type, 1> error_rows(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 1> error_rows(rows_number, 0.0);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < columns_number; j++)
        {
            error_rows[i] += (matrix(i,j) - other_matrix(i,j))*(matrix(i,j) - other_matrix(i,j));
        }

        error_rows[i] = sqrt(error_rows[i]);
    }

    return error_rows;
}


Tensor<type, 1> weighted_error_rows(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix, const double& weight1, const double& weight2)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 1> weighted_error_rows(rows_number, 0.0);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j =0; j < columns_number; j++)
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


double cross_entropy_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const int x_rows_number = x.dimension(0);
    const int x_columns_number = x.dimension(1);

    #ifdef __OPENNN_DEBUG__

    const int y_rows_number = y.dimension(0);

    if(y_rows_number != x_rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double cross_entropy_error(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const int y_columns_number = y.dimension(1);

    if(y_columns_number != x_columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double cross_entropy_error(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    double cross_entropy_error = 0.0;

    for(int i = 0; i < x_rows_number; i++)
    {
        for(int j = 0; j < x_columns_number; j++)
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

double minkowski_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y, const double& minkowski_parameter)
{
    const int rows_number = x.dimension(0);
    const int columns_number = x.dimension(1);

#ifdef __OPENNN_DEBUG__

    const int other_rows_number = y.dimension(0);

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const double&) method.\n"
              << "Other number of rows " << other_rows_number << " must be equal to this number of rows " << rows_number << ".\n";

       throw logic_error(buffer.str());
    }

    const int other_columns_number = y.dimension(1);

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const double&) method.\n"
              << "Other number of columns (" << other_columns_number << ") must be equal to this number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    double minkowski_error = 0.0;
    double row_minkowski_error = 0.0;

    for(int i = 0; i < rows_number; i++)
    {
        row_minkowski_error = 0.0;

        for(int j = 0; j < columns_number; j++)
        {
            row_minkowski_error += pow(abs(x(i,j) - y(i,j)), minkowski_parameter);
        }

        minkowski_error += pow(row_minkowski_error, 1.0 / minkowski_parameter);
    }

    return minkowski_error;
}


double weighted_sum_squared_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y, const double& positives_weight, const double& negatives_weight)
{
#ifdef __OPENNN_DEBUG__

    const int rows_number = x.dimension(0);
    const int columns_number = x.dimension(1);

    const int other_rows_number = y.dimension(0);

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Tensor<type, 2>&, const double&) method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const int other_columns_number = y.dimension(1);

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "double minkowski_error(const Tensor<type, 2>&, const double&) method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    double weighted_sum_squared_error = 0.0;

    double error = 0.0;

    for(int i = 0; i < x.size(); i++)
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

Tensor<type, 1> l1_norm_gradient(const Tensor<type, 1>& vector)
{
  return sign(vector);
}


/// Returns the hessian of the vector norm.

Tensor<type, 2> l1_norm_hessian(const Tensor<type, 1>& vector)
{
  const int x_size = vector.size();

  return Tensor<type, 2>(x_size, x_size, 0);
}


MatrixXd matrix_to_eigen(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    MatrixXd eigen(rows_number, columns_number);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
           eigen(i,j) = matrix(i,j);
        }
    }

    return eigen;
}


MatrixXd tensor_to_eigen(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    MatrixXd eigen(rows_number, columns_number);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
           eigen(i,j) = matrix(i,j);
        }
    }

    return eigen;
}


Tensor<type, 2> eigen_to_matrix(const MatrixXd& eigen)
{
    const int rows_number = eigen.rows();
    const int columns_number = eigen.cols();

    Tensor<type, 2> matrix(rows_number, columns_number);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
           matrix(i,j) = eigen(i,j);
        }
    }

    return matrix;
}


Tensor<type, 2> eigen_to_tensor(const MatrixXd& eigen)
{
    const int rows_number = eigen.rows();
    const int columns_number = eigen.cols();

    Tensor<type, 2> matrix(Tensor<int, 1>(rows_number, columns_number));

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
           matrix(i,j) = eigen(i,j);
        }
    }

    return matrix;
}
*/

}


