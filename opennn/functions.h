//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F U N C T I O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

// System includes

#include <math.h>
#include <vector>

// OpenNN includes

#include "statistics.h"
#include "metrics.h"
#include "config.h"

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

using namespace std;
using namespace Eigen;

namespace OpenNN
{
    type random_uniform(const type & = -1.0, const type & = 1.0);

    type random_normal(const type & = 0.0, const type & = 1.0);

    Index factorial(const Index&);

    Tensor<type, 1> exponential(const Tensor<type, 1>&);

    Tensor<type, 1> logarithm(const Tensor<type, 1>&);

    Tensor<type, 1> power(const Tensor<type, 1>&, const type&);

    Tensor<type, 2> competitive(const Tensor<type, 2>&);

    Tensor<bool, 1> binary(const Tensor<type, 1>&);

    Tensor<type, 1> square_root(const Tensor<type, 1>&);

    Tensor<type, 1> cumulative(const Tensor<type, 1>&);

    Tensor<type, 1> lower_bounded(const Tensor<type, 1>&, const type &);

    Tensor<type, 1> lower_bounded(const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 1> upper_bounded(const Tensor<type, 1>&, const type &);

    Tensor<type, 1> upper_bounded(const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 1> lower_upper_bounded(const Tensor<type, 1>&, const type &, const type &);

    Tensor<type, 1> lower_upper_bounded(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 2> lower_bounded(const Tensor<type, 2>&, const type&);
    Tensor<type, 2> upper_bounded(const Tensor<type, 2>&, const type&);

    Tensor<type, 2> lower_upper_bounded(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 1>&);
    Tensor<type, 2> lower_upper_bounded(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

    Tensor<type, 1> logistic_function(const Tensor<type, 1>&, const type&, const type&);

    Tensor<type, 1> hard_sigmoid(const Tensor<type, 1>&);

    Tensor<type, 1> hyperbolic_tangent(const Tensor<type, 1>&);

    Tensor<type, 1> logistic(const Tensor<type, 1>&);

    Tensor<type, 1> linear(const Tensor<type, 1>&);

    Tensor<type, 1> threshold(const Tensor<type, 1>&);

    Tensor<type, 1> symmetric_threshold(const Tensor<type, 1>&);

    Tensor<type, 1> rectified_linear(const Tensor<type, 1>&);

    Tensor<type, 1> scaled_exponential_linear(const Tensor<type, 1>&);

    Tensor<type, 1> soft_plus(const Tensor<type, 1>&);

    Tensor<type, 1> soft_sign(const Tensor<type, 1>&);

    Tensor<type, 1> exponential_linear(const Tensor<type, 1>&);

    Tensor<type, 1> logistic_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> threshold_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> symmetric_threshold_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> linear_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> hyperbolic_tangent_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> rectified_linear_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> scaled_exponential_linear_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> soft_plus_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> soft_sign_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> hard_sigmoid_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> exponential_linear_derivatives(const Tensor<type, 1>&);

    Tensor<type, 1> softmax_derivatives(const Tensor<type, 1>&);

    // SINE FUNCTIONS

    Tensor<type, 1> sine(const Tensor<type, 1>&);

    Tensor<type, 2> sine(const Tensor<type, 2>&);


    // COSINE FUNCTIONS

    Tensor<type, 1> cosine(const Tensor<type, 1>&);

    Tensor<type, 2> cosine(const Tensor<type, 2>&);

    // LINEAR

    Tensor<type, 2> linear(const Tensor<type, 2>&);

    // HYPERBOLIC TANGENT

    Tensor<type, 2> hyperbolic_tangent(const Tensor<type, 2>&);

    // LOGISTIC

    Tensor<type, 2> logistic(const Tensor<type, 2>&);

    // BINARY

    Tensor<type, 2> binary(const Tensor<type, 2>&);

    // THRESHOLD

    Tensor<type, 2> threshold(const Tensor<type, 2>&);

    // SYMMETRIC THRESHOLD

    Tensor<type, 2> symmetric_threshold(const Tensor<type, 2>&);

    // RECTIFIED LINEAR

    Tensor<type, 2> rectified_linear(const Tensor<type, 2>&);

    // SCALED EXPONENTIAL LINEAR

    Tensor<type, 2> scaled_exponential_linear(const Tensor<type, 2>&);

    // SOFT PLUS

    Tensor<type, 2> soft_plus(const Tensor<type, 2>&);

    // SOFT SIGN

    Tensor<type, 2> soft_sign(const Tensor<type, 2>&);

    // HARD SIGMOID

    Tensor<type, 2> hard_sigmoid(const Tensor<type, 2>&);

    // EXPONENTIAL LINEAR

    Tensor<type, 2> exponential_linear(const Tensor<type, 2>&);

    // SOFTMAX

    Tensor<type, 2> softmax(const Tensor<type, 2>&);

    Tensor<type, 2> softmax_rows(const Tensor<type, 2>&);

    Tensor<type, 2> softmax_columns(const Tensor<type, 2>&);

    // LINEAR DERIVATIVES

    Tensor<type, 2> linear_derivatives(const Tensor<type, 2>&);

    // HYPERBOLIC TANGENT DERIVATIVES

    Tensor<type, 2> hyperbolic_tangent_derivatives(const Tensor<type, 2>&);

    // LOGISTIC DERIVATIVES

    Tensor<type, 2> logistic_derivatives(const Tensor<type, 2>&);

    // THRESHOLD DERIVATIVES

    Tensor<type, 2> threshold_derivatives(const Tensor<type, 2>&);

    // SYMMETRIC THRESHOLD DERIVATIVES

    Tensor<type, 2> symmetric_threshold_derivatives(const Tensor<type, 2>&);

    // RECTIFIED LINEAR DERIVATIVES

    Tensor<type, 2> rectified_linear_derivatives(const Tensor<type, 2>&);

    // SCALED EXPONENTIAL LINEAR DERIVATIVES

    Tensor<type, 2> scaled_exponential_linear_derivatives(const Tensor<type, 2>&);

    //SOFT PLUS DERIVATIVES

    Tensor<type, 2> soft_plus_derivatives(const Tensor<type, 2>&);

    // SOFT SIGN DERIVATIVES

    Tensor<type, 2> soft_sign_derivatives(const Tensor<type, 2>&);

    // HARD SIGMOID DERIVATIVES

    Tensor<type, 2> hard_sigmoid_derivatives(const Tensor<type, 2>&);

    // EXPONENTIAL LINEAR DERIVATIVES

    Tensor<type, 2> exponential_linear_derivatives(const Tensor<type, 2>&);

    // SOFTMAX DERIVATIVES

    Tensor<type, 2> softmax_derivatives(const Tensor<type, 2>&);

    Tensor<type, 1> sign(const Tensor<type, 1>&);

    Tensor<type, 1> normalized(const Tensor<type, 1>&);

    Tensor<type, 2> normalized_columns(const Tensor<type, 2>&);

    void hard_sigmoid(const Tensor<type, 2>&, Tensor<type, 2>&);
    void hyperbolic_tangent(const ThreadPoolDevice& thread_pool_device, const Tensor<type, 2>&, Tensor<type, 2>&);
    void logistic(const Tensor<type, 2>&, Tensor<type, 2>&);
    void linear(const Tensor<type, 2>&, Tensor<type, 2>&);
    void threshold(const Tensor<type, 2>&, Tensor<type, 2>&);
    void symmetric_threshold(const Tensor<type, 2>&, Tensor<type, 2>&);
    void rectified_linear(const Tensor<type, 2>&, Tensor<type, 2>&);
    void scaled_exponential_linear(const Tensor<type, 2>&, Tensor<type, 2>&);
    void soft_plus(const Tensor<type, 2>&, Tensor<type, 2>&);
    void soft_sign(const Tensor<type, 2>&, Tensor<type, 2>&);
    void exponential_linear(const Tensor<type, 2>&, Tensor<type, 2>&);
    void softmax(const Tensor<type, 2>&, Tensor<type, 2>&);
    void binary(const Tensor<type, 2>&, Tensor<type, 2>&);
    void competitive(const Tensor<type, 2>&, Tensor<type, 2>&);

    void logistic_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void symmetric_threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void hyperbolic_tangent_derivatives(const ThreadPoolDevice& thread_pool_device, const Tensor<type, 2>&, Tensor<type, 2>&);
    void rectified_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void scaled_exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void soft_plus_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void soft_sign_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void hard_sigmoid_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void exponential_linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
    void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&);
}

#endif // FUNCTIONS_H
