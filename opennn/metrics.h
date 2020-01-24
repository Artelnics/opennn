//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E T R I C S   F U N C T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef METRICS_H
#define METRICS_H

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#include <math.h>

#include "../eigen/Eigen/Eigen"
#include "config.h"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

using namespace std;
using namespace Eigen;

namespace OpenNN
{
    type sum_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>&);

    type l1_norm(const Tensor<type, 1>&);
    type l2_norm(const ThreadPoolDevice&, const Tensor<type, 1>&);

    Tensor<type, 1> lp_norm_gradient(const Tensor<type, 1>&, const type&);

     // Direct products

     Tensor<type, 2> direct(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // DECOMPOSITIONS

     // MATRIX EIGENVALUES

     // NORMS


     Tensor<type, 1> l1_norm_gradient(const Tensor<type, 1>&);

     Tensor<type, 2> l1_norm_hessian(const Tensor<type, 1>&);


     Tensor<type, 1> l2_norm_gradient(const Tensor<type, 1>&);

     Tensor<type, 2> l2_norm_hessian(const Tensor<type, 1>&);

     type lp_norm(const Tensor<type, 1>&, const type &);


     Tensor<type, 1> lp_norm(const Tensor<type, 2>&, const type& p);

     Tensor<type, 1> lp_norm(const Tensor<type, 2>&, const type& p);

     Tensor<type, 2> lp_norm_gradient(const Tensor<type, 2>&, const type& p);

    // INVERTING MATICES


     // LINEAR EQUATIONS

     Tensor<type, 2> eigenvalues(const Tensor<type, 2>&);
     Tensor<type, 2> eigenvectors(const Tensor<type, 2>&);

     Tensor<type, 2> direct(const Tensor<type, 2>&, const Tensor<type, 2>&);

     // Vector distances

     type euclidean_distance(const Tensor<type, 1>&, const Tensor<type, 1>&);

     type euclidean_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);
     Tensor<type, 1> euclidean_weighted_distance_vector(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     type manhattan_distance(const Tensor<type, 1>&, const Tensor<type, 1>&);
     type manhattan_weighted_distance(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     Tensor<type, 1> manhattan_weighted_distance_vector(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     // Matrix distances

     Tensor<type, 1> euclidean_distance(const Tensor<type, 2>&, const Tensor<type, 1>&);
     Tensor<type, 1> euclidean_distance(const Tensor<type, 2>&, const Tensor<type, 2>&);

     Tensor<type, 1> euclidean_weighted_distance(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 1>&);
     Tensor<type, 2> euclidean_weighted_distance_matrix(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     Tensor<type, 1> manhattan_distance(const Tensor<type, 2>&, const Tensor<type, 1>&);
     type manhattan_distance(const Tensor<type, 2>&, const Index&, const Index&);

     Tensor<type, 1> manhattan_weighted_distance(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 1>&);
     Tensor<type, 2> manhattan_weighted_distance_matrix(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     // Vector errors

     type sum_squared_error(const Tensor<type, 1>&, const Tensor<type, 1>&);

     type minkowski_error(const Tensor<type, 1>&, const Tensor<type, 1>&, const type&);

     // Tensor errors

     type cross_entropy_error(const Tensor<type, 2>&, const Tensor<type, 2>&);

     type minkowski_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&);

     type weighted_sum_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&, const type&);

     // Error rows

     Tensor<type, 1> error_rows(const Tensor<type, 2>&, const Tensor<type, 2>&);

     Tensor<type, 1> weighted_error_rows(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&, const type&);

}

#endif // METRICS_H
