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

type sum_squared_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const auto error = y - x;

    const Eigen::array<Eigen::IndexPair<Index>, 2> product_dimensions = { Eigen::IndexPair<Index>(0, 0), Eigen::IndexPair<Index>(1, 1) };

    const Tensor<type, 0> sse = error.contract(error, product_dimensions);

    return sse(0);
}


type l2_norm(const ThreadPoolDevice& threadPoolDevice, const Tensor<type, 1>& x)
{
   Tensor<type, 0> y;

   y.device(threadPoolDevice) = x.square().sum(Eigen::array<Index, 1>({0}));

   return y(0);

 /*
  const Index x_size = vector.size();

  //vector.sum()

  type norm = 0.0;

  for(Index i = 0; i < x_size; i++) {
    norm += vector[i] *vector[i];
  }

    return sqrt(norm);
*/
}





/// Returns the vector norm.

type l1_norm(const Tensor<type, 1>& vector)
{
/*
  return absolute_value(vector).sum();
*/
    return 0;
}


/// Returns the vector norm.

type l2_norm(const Tensor<type, 1>& vector)
{
  const Index x_size = vector.size();

  type norm = 0.0;

  for(Index i = 0; i < x_size; i++) {
    norm += vector[i] *vector[i];
  }

    return sqrt(norm);
}


/// Returns the gradient of the vector norm.

Tensor<type, 1> l2_norm_gradient(const Tensor<type, 1>& vector)
{

  const Index x_size = vector.size();

  Tensor<type, 1> gradient(x_size);
/*
  const type norm = l2_norm(vector);

  if(norm == 0.0) {
    gradient.setZero();
  } else {
    gradient = vector/ norm;
  }
*/
  return gradient;
}


/// Returns the hessian of the vector norm.

Tensor<type, 2> l2_norm_hessian(const Tensor<type, 1>& vector)
{
  const Index x_size = vector.size();

  Tensor<type, 2> hessian(x_size, x_size);
/*
  const type norm = l2_norm(vector);

  if(norm == 0.0) {
    hessian.setZero();
  } else {
    hessian = direct(vector, vector)/(norm * norm * norm);
  }
*/
  return hessian;
}


/// Returns the vector p-norm.

type lp_norm(const Tensor<type, 1>& vector, const type &p)
{
#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type calculate_p_norm(const type&) method.\n"
           << "p value must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const Index x_size = vector.size();

  type norm = 0.0;

  for(Index i = 0; i < x_size; i++) {
    norm += pow(abs(vector[i]), p);
  }

  norm = pow(norm, 1.0 / p);

  return norm;
}


/// Returns the gradient of the vector norm.

Tensor<type, 1> lp_norm_gradient(const Tensor<type, 1>& vector, const type &p)
{
#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(p <= 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "Tensor<type, 1> calculate_p_norm_gradient(const type&) const "
              "method.\n"
           << "p value must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const Index x_size = vector.size();

  Tensor<type, 1> gradient(x_size);

  const type p_norm = lp_norm(vector, p);

  if(p_norm == 0.0)
  {
    gradient.setZero();
  }
  else
  {
    for(Index i = 0; i < x_size; i++)
    {
      gradient[i] =
         vector[i] * pow(abs(vector[i]), p - 2.0) / pow(p_norm, p - 1.0);
    }
  }

  return gradient;
}


/// Outer product vector*vector arithmetic operator.
/// @param vector_2 vector to be multiplied to this vector.

Tensor<type, 2> direct(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
  const Index x_size = x.size();

#ifdef __OPENNN_DEBUG__

  const Index y_size = y.size();

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

  for(Index i = 0; i < static_cast<Index>(x_size); i++)
  {
    for(Index j = 0; j < x_size; j++)
    {
      direct(i, j) = x[i] * y[j];
    }
  }

  return direct;
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

//    const Map<Tensor<type, 2>> this_eigen((type*)this->data(), rows_number, columns_number);
//    const SelfAdjointEigenSolver<Tensor<type, 2>> matrix_eigen(this_eigen, EigenvaluesOnly);
//    Map<Tensor<type, 2>> eigenvalues_eigen(eigenvalues.data(), rows_number, 1);

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
/*
    const Map<Tensor<type, 2>> this_eigen((type*)matrix.data(), rows_number, columns_number);
    const SelfAdjointEigenSolver<Tensor<type, 2>> matrix_eigen(this_eigen, ComputeEigenvectors);
    Map<Tensor<type, 2>> eigenvectors_eigen(eigenvectors.data(), rows_number, rows_number);

    eigenvectors_eigen = matrix_eigen.eigenvectors();
*/
    return eigenvectors;
}


/// Calculates the direct product of this matrix with another matrix.
/// This product is also known as the Kronecker product.
/// @param other_matrix Second product term.

Tensor<type, 2> direct(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   const Index other_rows_number = other_matrix.dimension(0);
   const Index other_columns_number = other_matrix.dimension(1);

   Tensor<type, 2> direct(rows_number*other_rows_number, columns_number*other_columns_number);

   Index alpha;
   Index beta;

   for(Index i = 0; i < rows_number; i++)
   {
       for(Index j = 0; j < columns_number; j++)
       {
           for(Index k = 0; k < other_rows_number; k++)
           {
               for(Index l = 0; l < other_columns_number; l++)
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

Tensor<type, 1> lp_norm(const Tensor<type, 2>& matrix, const type& p)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Tensor<type, 1> calculate_lp_norm(const type&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Tensor<type, 1> norm(rows_number);
/*
    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            norm[i] += pow(abs(matrix(i,j)), p);
        }

        norm[i] = pow(norm[i], 1.0 / p);
    }
*/
    return norm;
}


/// Returns the gradient of the matrix norm.
/// The tensor must be a matrix.

Tensor<type, 2> lp_norm_gradient(const Tensor<type, 2>& matrix, const type& p)
{
    if(matrix.rank() > 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "type lp_norm(const Tensor<type, 2>&, const type&) method.\n"
               << "The number of dimensions of the tensor should be 2.\n";

        throw logic_error(buffer.str());
    }

    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

      ostringstream buffer;

      if(p <= 0) {
        buffer << "OpenNN Exception: Metrics functions.\n"
               << "Tensor<type, 2> calculate_p_norm_gradient(const type&) const "
                  "method.\n"
               << "p value must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

      Tensor<type, 2> gradient(rows_number, columns_number);
/*
      const Tensor<type, 1> p_norm = lp_norm(matrix, p);

      if(p_norm == 0.0)
      {
        gradient.setZero();
      }
      else
      {
        for(Index i = 0; i < rows_number; i++)
        {
            for(Index j = 0; j < columns_number; j++)
            {
                gradient(i,j) = matrix(i,j) * pow(abs(matrix(i,j)), p - 2.0) / pow(p_norm[i], p - 1.0);
            }
        }
      }
*/
      return gradient;
}


/// Returns the distance between the elements of this vector and the elements of
/// another vector.
/// @param vector_2 Other vector.

type euclidean_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>& vector_2)
{
    const Index x_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const Index y_size = vector_2.dimension(0);

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type euclidean_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    type distance = 0.0;
    type error;

    for(Index i = 0; i < x_size; i++)
    {
        error = vector[i] - vector_2[i];

        distance += error * error;
    }

    return sqrt(distance);
}


type euclidean_weighted_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>& vector_2, const Tensor<type, 1>& weights)
{

    const Index x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const Index y_size = vector_2.dimension(0);

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type euclidean_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    type distance = 0.0;
    type error;

    for(Index i = 0; i < x_size; i++) {
        error = vector[i] - vector_2[i];

        distance += error * error * weights[i];
    }

    return(sqrt(distance));
}


Tensor<type, 1> euclidean_weighted_distance_vector(const Tensor<type, 1>& vector, const Tensor<type, 1>& vector_2, const Tensor<type, 1>& weights)
{

    const Index x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const Index y_size = vector_2.dimension(0);

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type euclidean_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Tensor<type, 1> distance(x_size);

    type error;

    for(Index i = 0; i < x_size; i++) {
        error = vector[i] - vector_2[i];

        distance[i] = error * error * weights[i];
    }

    return distance;
}


type manhattan_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>&vector_2)
{

    const Index x_size = vector.size();
#ifdef __OPENNN_DEBUG__

  const Index y_size = vector_2.dimension(0);

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type manhattan_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    type distance = 0.0;
    type error;

    for(Index i = 0; i < x_size; i++) {
        error = abs(vector[i] - vector_2[i]);

        distance += error;
    }

    return distance;
}


type manhattan_weighted_distance(const Tensor<type, 1>& vector, const Tensor<type, 1>& vector_2, const Tensor<type, 1>& weights)
{
    const Index x_size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  const Index y_size = vector_2.dimension(0);

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type manhattan_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    type distance = 0.0;
    type error;

    for(Index i = 0; i < x_size; i++)
    {
        error = abs(vector[i] - vector_2[i]);

        distance += error * weights[i];
    }

    return distance;
}


Tensor<type, 1> manhattan_weighted_distance_vector(const Tensor<type, 1>& vector, const Tensor<type, 1>& vector_2, const Tensor<type, 1>& weights)
{
    const Index x_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const Index y_size = vector_2.dimension(0);

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type manhattan_weighted_distance(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Tensor<type, 1> distance(x_size);
    type error;

    for(Index i = 0; i < x_size; i++)
    {
        error = abs(vector[i] - vector_2[i]);

        distance[i] = error * weights[i];
    }

    return distance;
}


/// Returns the sum squared error between the elements of this vector and the
/// elements of another vector.
/// @param vector_2 Other vector.

type sum_squared_error(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
  const Index x_size = x.size();

#ifdef __OPENNN_DEBUG__

  const Index y_size = y.size();

  if(y_size != x_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type sum_squared_error(const Tensor<type, 1>&) const method.\n"
           << "Size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

#endif

  type sum_squared_error = 0.0;
  type error;

  for(Index i = 0; i < x_size; i++)
  {
    error = x[i] - y[i];

    sum_squared_error += error * error;
  }

  return sum_squared_error;
}


/// Returns the Minkowski squared error between the elements of this vector and
/// the elements of another vector.
/// @param vector This vector.
/// @param vector_2 Other vector.
/// @param minkowski_parameter Minkowski exponent.

type minkowski_error(const Tensor<type, 1>& vector,
                       const Tensor<type, 1>& vector_2,
                       const type& minkowski_parameter)
{
  const Index x_size = vector.size();

#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

  if(x_size == 0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type minkowski_error(const Tensor<type, 1>&) const "
              "method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

  const Index y_size = vector_2.dimension(0);

  if(y_size != x_size) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type minkowski_error(const Tensor<type, 1>&) const "
              "method.\n"
           << "Other size must be equal to this size.\n";

    throw logic_error(buffer.str());
  }

  // Control sentence

  if(minkowski_parameter < 1.0 || minkowski_parameter > 2.0) {
    buffer << "OpenNN Exception: Metrics functions.\n"
           << "type minkowski_error(const Tensor<type, 1>&) const "
              "method.\n"
           << "The Minkowski parameter must be comprised between 1 and 2\n";

    throw logic_error(buffer.str());
  }

#endif

  type minkowski_error = 0.0;

  for(Index i = 0; i < x_size; i++)
  {
    minkowski_error +=
        pow(abs(vector[i] - vector_2[i]), minkowski_parameter);
  }

  minkowski_error = pow(minkowski_error, 1.0 / minkowski_parameter);

  return(minkowski_error);
}


Tensor<type, 1> euclidean_distance(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__

     if(matrix.dimension(0) == 0 || matrix.dimension(1) == 0)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_distance(const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 1> distances(rows_number);
/*
    for(Index i = 0; i < rows_number; i++)
    {
        distances[i] = euclidean_distance(matrix.chip(i, 0), instance);
    }
*/
    return distances;
}


Tensor<type, 1> euclidean_distance(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 1> distances(rows_number);

    type error;

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
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

     if(matrix.dimension(0) == 0 || matrix.dimension(1) == 0)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 1> distances(rows_number);
/*
    for(Index i = 0; i < rows_number; i++)
    {
        distances[i] = euclidean_weighted_distance(matrix.chip(i, 0), instance, weights);
    }
*/
    return distances;
}


Tensor<type, 2> euclidean_weighted_distance_matrix(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance, const Tensor<type, 1>& weights)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__

     if(matrix.dimension(0) == 0 || matrix.dimension(1) == 0)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "euclidean_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 2> distances(rows_number,columns_number);
/*
    for(Index i = 0; i < rows_number; i++)
    {
        distances.set_row(i, euclidean_weighted_distance_vector(matrix.chip(i, 0), instance,weights));
    }
*/
    return distances;
}


/// Calculates the distance between two rows in the matrix

type manhattan_distance(const Tensor<type, 2>& matrix, const Index& first_index, const Index& second_index)
{
    #ifdef __OPENNN_DEBUG__

     if(matrix.dimension(0) == 0 || matrix.dimension(1) == 0)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_distance(const Index&, const Index&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif
/*
    const Tensor<type, 1> first_row = matrix.get_row(first_index);
    const Tensor<type, 1> second_row = matrix.get_row(second_index);

    return manhattan_distance(first_row, second_row);

*/
    return 0;
}


Tensor<type, 1> manhattan_distance(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__

     if(matrix.dimension(0) == 0 || matrix.dimension(1) == 0)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_distance(const Index&, const Index&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }

     #endif

     Tensor<type, 1> distances(rows_number);
/*
    for(Index i = 0; i < rows_number; i++)
    {
        distances[i] = manhattan_distance(matrix.chip(i, 0), instance);
    }
*/
    return distances;
}


Tensor<type, 1> manhattan_weighted_distance(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance, const Tensor<type, 1>& weights)
{
    const Index rows_number = matrix.dimension(0);

    #ifdef __OPENNN_DEBUG__
/*
     if(matrix.dimension(0) == 0 || matrix.dimension(1) == 0)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }
*/
     #endif

     Tensor<type, 1> distances(rows_number);
/*
    for(Index i = 0; i < rows_number; i++)
    {
        distances[i] = manhattan_weighted_distance(matrix.chip(i, 0), instance, weights);
    }
*/
    return distances;
}


Tensor<type, 2> manhattan_weighted_distance_matrix(const Tensor<type, 2>& matrix, const Tensor<type, 1>& instance, const Tensor<type, 1>& weights)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #ifdef __OPENNN_DEBUG__
/*
     if(matrix.dimension(0) == 0 || matrix.dimension(1) == 0)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "manhattan_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Matrix is empty.\n";

        throw logic_error(buffer.str());
     }
*/
     #endif

     Tensor<type, 2> distances(rows_number,columns_number);
/*
    for(Index i = 0; i < rows_number; i++)
    {
        distances.set_row(i,manhattan_weighted_distance_vector(matrix.chip(i, 0),instance,weights));
    }
*/
    return distances;
}


Tensor<type, 1> error_rows(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 1> error_rows(rows_number);

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            error_rows[i] += (matrix(i,j) - other_matrix(i,j))*(matrix(i,j) - other_matrix(i,j));
        }

        error_rows[i] = sqrt(error_rows[i]);
    }

    return error_rows;
}


Tensor<type, 1> weighted_error_rows(const Tensor<type, 2>& matrix, const Tensor<type, 2>& other_matrix, const type& weight1, const type& weight2)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 1> weighted_error_rows(rows_number);

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j =0; j < columns_number; j++)
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


type cross_entropy_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const Index x_rows_number = x.dimension(0);
    const Index x_columns_number = x.dimension(1);

    #ifdef __OPENNN_DEBUG__

    const Index y_rows_number = y.dimension(0);

    if(y_rows_number != x_rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "type cross_entropy_error(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const Index y_columns_number = y.dimension(1);

    if(y_columns_number != x_columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "type cross_entropy_error(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    type cross_entropy_error = 0.0;

    for(Index i = 0; i < x_rows_number; i++)
    {
        for(Index j = 0; j < x_columns_number; j++)
        {
            const type y_value = y(i, static_cast<unsigned>(j));
            const type x_value = x(i,j);

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

type minkowski_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y, const type& minkowski_parameter)
{
    const Index rows_number = x.dimension(0);
    const Index columns_number = x.dimension(1);

#ifdef __OPENNN_DEBUG__

    const Index other_rows_number = y.dimension(0);

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "type minkowski_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&) method.\n"
              << "Other number of rows " << other_rows_number << " must be equal to this number of rows " << rows_number << ".\n";

       throw logic_error(buffer.str());
    }

    const Index other_columns_number = y.dimension(1);

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "type minkowski_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&) method.\n"
              << "Other number of columns (" << other_columns_number << ") must be equal to this number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    type minkowski_error = 0.0;
    type row_minkowski_error = 0.0;

    for(Index i = 0; i < rows_number; i++)
    {
        row_minkowski_error = 0.0;

        for(Index j = 0; j < columns_number; j++)
        {
            row_minkowski_error += pow(abs(x(i,j) - y(i,j)), minkowski_parameter);
        }

        minkowski_error += pow(row_minkowski_error, 1.0 / minkowski_parameter);
    }

    return minkowski_error;
}


type weighted_sum_squared_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y, const type& positives_weight, const type& negatives_weight)
{
#ifdef __OPENNN_DEBUG__

    const Index rows_number = x.dimension(0);
    const Index columns_number = x.dimension(1);

    const Index other_rows_number = y.dimension(0);

    if(other_rows_number != rows_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "type minkowski_error(const Tensor<type, 2>&, const type&) method.\n"
              << "Other number of rows must be equal to this number of rows.\n";

       throw logic_error(buffer.str());
    }

    const Index other_columns_number = y.dimension(1);

    if(other_columns_number != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Metrics functions.\n"
              << "type minkowski_error(const Tensor<type, 2>&, const type&) method.\n"
              << "Other number of columns must be equal to this number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    type weighted_sum_squared_error = 0.0;

    type error = 0.0;
/*
    for(Index i = 0; i < x.size(); i++)
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
                   << "type calculate_error() method.\n"
                   << "Other matrix is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }
    }
*/
    return weighted_sum_squared_error;
}


/// Returns the gradient of the vector norm.

Tensor<type, 1> l1_norm_gradient(const Tensor<type, 1>& vector)
{
/*
  return sign(vector);
*/
    return Tensor<type, 1>();
}


/// Returns the hessian of the vector norm.

Tensor<type, 2> l1_norm_hessian(const Tensor<type, 1>& vector)
{
  const Index x_size = vector.dimension(0);

  Tensor<type, 2> hessian(x_size, x_size);

  hessian.setConstant(0.0);

  return hessian;
}


}


