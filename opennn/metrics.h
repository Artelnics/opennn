//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E T R I C S   F U N C T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef METRICS_H
#define METRICS_H

#include "vector.h"
#include "matrix.h"
#include "tensor.h"
#include "functions.h"
#include <math.h>

#include "../eigen/Eigen"

using namespace std;

namespace OpenNN
{
    // Dot products

     double dot(const Vector<double>&, const Vector<double>&);

     Vector<double> dot(const Vector<double>&, const Matrix<double>&);

     Vector<double> dot(const Matrix<double>&, const Vector<double>&);

     Matrix<double> dot(const Matrix<double>&, const Matrix<double>&);

     Matrix<double> dot(const Matrix<double>&, const Tensor<double>&);

     Tensor<double> dot(const Tensor<double>&, const Matrix<double>&);

     Tensor<double> dot_2d_2d(const Tensor<double>&, const Tensor<double>&);

     Tensor<double> dot_2d_3d(const Tensor<double>&, const Tensor<double>&);

     // Direct products

     Matrix<double> direct(const Vector<double>&, const Vector<double>&);

    // DECOMPOSITIONS

     // MATRIX EIGENVALUES

     // NORMS

     double l1_norm(const Vector<double>&);

     Vector<double> l1_norm_gradient(const Vector<double>&);

     Matrix<double> l1_norm_hessian(const Vector<double>&);

     double l2_norm(const Vector<double>&);

     Vector<double> l2_norm_gradient(const Vector<double>&);

     Matrix<double> l2_norm_hessian(const Vector<double>&);

     double lp_norm(const Vector<double>&, const double &);

//     Vector<double> lp_norm_gradient(const Vector<double>&, const double &);

     Vector<double> lp_norm(const Matrix<double>&, const double& p);

     Vector<double> lp_norm(const Tensor<double>&, const double& p);

     Tensor<double> lp_norm_gradient(const Tensor<double>&, const double& p);

    // INVERTING MATICES

     double determinant(const Matrix<double>&);

     Matrix<double> cofactor(const Matrix<double>&);

     Matrix<double> inverse(const Matrix<double>&);

     // LINEAR EQUATIONS

     Matrix<double> eigenvalues(const Matrix<double>&);
     Matrix<double> eigenvectors(const Matrix<double>&);

     Matrix<double> direct(const Matrix<double>&, const Matrix<double>&);

     Tensor<double> linear_combinations(const Tensor<double>&, const Matrix<double>&, const Vector<double>&);

     // Vector distances

     double euclidean_distance(const Vector<double>&, const Vector<double>&);

     double euclidean_weighted_distance(const Vector<double>&, const Vector<double>&, const Vector<double>&);
     Vector<double> euclidean_weighted_distance_vector(const Vector<double>&, const Vector<double>&, const Vector<double>&);

     double manhattan_distance(const Vector<double>&, const Vector<double>&);
     double manhattan_weighted_distance(const Vector<double>&, const Vector<double>&, const Vector<double>&);

     Vector<double> manhattan_weighted_distance_vector(const Vector<double>&, const Vector<double>&, const Vector<double>&);

     // Matrix distances

     Vector<double> euclidean_distance(const Matrix<double>&, const Vector<double>&);
     Vector<double> euclidean_distance(const Matrix<double>&, const Matrix<double>&);

     Vector<double> euclidean_weighted_distance(const Matrix<double>&, const Vector<double>&, const Vector<double>&);
     Matrix<double> euclidean_weighted_distance_matrix(const Matrix<double>&, const Vector<double>&, const Vector<double>&);

     Vector<double> manhattan_distance(const Matrix<double>&, const Vector<double>&);
     double manhattan_distance(const Matrix<double>&, const size_t&, const size_t&);

     Vector<double> manhattan_weighted_distance(const Matrix<double>&, const Vector<double>&, const Vector<double>&);
     Matrix<double> manhattan_weighted_distance_matrix(const Matrix<double>&, const Vector<double>&, const Vector<double>&);

     // Vector errors

     double sum_squared_error(const Vector<double>&, const Vector<double>&);

     double minkowski_error(const Vector<double>&, const Vector<double>&, const double &);

     // Tensor errors

     double sum_squared_error(const Tensor<double>&, const Tensor<double>&);

     double cross_entropy_error(const Tensor<double>&, const Tensor<double>&);

     double minkowski_error(const Tensor<double>&, const Tensor<double>&, const double&);

     double weighted_sum_squared_error(const Tensor<double>&, const Tensor<double>&, const double&, const double&);

     // Error rows

     Vector<double> error_rows(const Tensor<double>&, const Tensor<double>&);

     Vector<double> weighted_error_rows(const Tensor<double>&, const Tensor<double>&, const double&, const double&);
}

#endif // __FUNCTIONS_H
