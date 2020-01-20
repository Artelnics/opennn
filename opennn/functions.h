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

#include <../eigen/unsupported/Eigen/CXX11/Tensor>
#include <../eigen/unsupported/Eigen/CXX11/ThreadPool>

using namespace std;
using namespace Eigen;

namespace OpenNN
{
    double random_uniform(const double & = -1.0, const double & = 1.0);

    double random_normal(const double & = 0.0, const double & = 1.0);

    int factorial(const int&);

    Tensor<double, 1> exponential(const Tensor<double, 1>&);

    Tensor<double, 1> logarithm(const Tensor<double, 1>&);

    Tensor<double, 1> power(const Tensor<double, 1>&, const double&);

    Tensor<double, 2> competitive(const Tensor<double, 2>&);

    vector<bool> binary(const Tensor<double, 1>&);

    Tensor<double, 1> square_root(const Tensor<double, 1>&);

    Tensor<double, 1> cumulative(const Tensor<double, 1>&);

    Tensor<double, 1> lower_bounded(const Tensor<double, 1>&, const double &);

    Tensor<double, 1> lower_bounded(const Tensor<double, 1>&, const Tensor<double, 1>&);

    Tensor<double, 1> upper_bounded(const Tensor<double, 1>&, const double &);

    Tensor<double, 1> upper_bounded(const Tensor<double, 1>&, const Tensor<double, 1>&);

    Tensor<double, 1> lower_upper_bounded(const Tensor<double, 1>&, const double &, const double &);

    Tensor<double, 1> lower_upper_bounded(const Tensor<double, 1>&, const Tensor<double, 1>&, const Tensor<double, 1>&);

    Tensor<double, 2> lower_bounded(const Tensor<double, 2>&, const double&);
    Tensor<double, 2> upper_bounded(const Tensor<double, 2>&, const double&);

    Tensor<double, 2> lower_upper_bounded(const Tensor<double, 2>&, const Tensor<double, 1>&, const Tensor<double, 1>&);
    Tensor<double, 2> lower_upper_bounded(const Tensor<double, 2>&, const Tensor<double, 1>&, const Tensor<double, 1>&);

    Tensor<double, 1> logistic_function(const Tensor<double, 1>&, const double&, const double&);

    Tensor<double, 1> hard_sigmoid(const Tensor<double, 1>&);

    Tensor<double, 1> hyperbolic_tangent(const Tensor<double, 1>&);

    Tensor<double, 1> logistic(const Tensor<double, 1>&);

    Tensor<double, 1> linear(const Tensor<double, 1>&);

    Tensor<double, 1> threshold(const Tensor<double, 1>&);

    Tensor<double, 1> symmetric_threshold(const Tensor<double, 1>&);

    Tensor<double, 1> rectified_linear(const Tensor<double, 1>&);

    Tensor<double, 1> scaled_exponential_linear(const Tensor<double, 1>&);

    Tensor<double, 1> soft_plus(const Tensor<double, 1>&);

    Tensor<double, 1> soft_sign(const Tensor<double, 1>&);

    Tensor<double, 1> exponential_linear(const Tensor<double, 1>&);

    Tensor<double, 1> logistic_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> threshold_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> symmetric_threshold_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> linear_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> hyperbolic_tangent_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> rectified_linear_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> scaled_exponential_linear_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> soft_plus_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> soft_sign_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> hard_sigmoid_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> exponential_linear_derivatives(const Tensor<double, 1>&);

    Tensor<double, 1> softmax_derivatives(const Tensor<double, 1>&);

    // SINE FUNCTIONS

    Tensor<double, 1> sine(const Tensor<double, 1>&);

    Tensor<double, 2> sine(const Tensor<double, 2>&);


    // COSINE FUNCTIONS

    Tensor<double, 1> cosine(const Tensor<double, 1>&);

    Tensor<double, 2> cosine(const Tensor<double, 2>&);

    // LINEAR

    Tensor<double, 2> linear(const Tensor<double, 2>&);

    // HYPERBOLIC TANGENT

    Tensor<double, 2> hyperbolic_tangent(const Tensor<double, 2>&);

    // LOGISTIC

    Tensor<double, 2> logistic(const Tensor<double, 2>&);
    Tensor<double, 2> logistic_second_derivatives(const Tensor<double, 2>&);

    // BINARY

    Tensor<double, 2> binary(const Tensor<double, 2>&);

    // THRESHOLD

    Tensor<double, 2> threshold(const Tensor<double, 2>&);

    // SYMMETRIC THRESHOLD

    Tensor<double, 2> symmetric_threshold(const Tensor<double, 2>&);

    // RECTIFIED LINEAR

    Tensor<double, 2> rectified_linear(const Tensor<double, 2>&);

    // SCALED EXPONENTIAL LINEAR

    Tensor<double, 2> scaled_exponential_linear(const Tensor<double, 2>&);

    // SOFT PLUS

    Tensor<double, 2> soft_plus(const Tensor<double, 2>&);

    // SOFT SIGN

    Tensor<double, 2> soft_sign(const Tensor<double, 2>&);

    // HARD SIGMOID

    Tensor<double, 2> hard_sigmoid(const Tensor<double, 2>&);

    // EXPONENTIAL LINEAR

    Tensor<double, 2> exponential_linear(const Tensor<double, 2>&);

    // SOFTMAX

    Tensor<double, 2> softmax(const Tensor<double, 2>&);

    Tensor<double, 2> softmax_rows(const Tensor<double, 2>&);

    Tensor<double, 2> softmax_columns(const Tensor<double, 2>&);

    // LINEAR DERIVATIVES

    Tensor<double, 2> linear_derivatives(const Tensor<double, 2>&);

    // HYPERBOLIC TANGENT DERIVATIVES

    Tensor<double, 2> hyperbolic_tangent_derivatives(const Tensor<double, 2>&);

    // LOGISTIC DERIVATIVES

    Tensor<double, 2> logistic_derivatives(const Tensor<double, 2>&);

    // THRESHOLD DERIVATIVES

    Tensor<double, 2> threshold_derivatives(const Tensor<double, 2>&);

    // SYMMETRIC THRESHOLD DERIVATIVES

    Tensor<double, 2> symmetric_threshold_derivatives(const Tensor<double, 2>&);

    // RECTIFIED LINEAR DERIVATIVES

    Tensor<double, 2> rectified_linear_derivatives(const Tensor<double, 2>&);

    // SCALED EXPONENTIAL LINEAR DERIVATIVES

    Tensor<double, 2> scaled_exponential_linear_derivatives(const Tensor<double, 2>&);

    //SOFT PLUS DERIVATIVES

    Tensor<double, 2> soft_plus_derivatives(const Tensor<double, 2>&);

    // SOFT SIGN DERIVATIVES

    Tensor<double, 2> soft_sign_derivatives(const Tensor<double, 2>&);

    // HARD SIGMOID DERIVATIVES

    Tensor<double, 2> hard_sigmoid_derivatives(const Tensor<double, 2>&);

    // EXPONENTIAL LINEAR DERIVATIVES

    Tensor<double, 2> exponential_linear_derivatives(const Tensor<double, 2>&);

    // SOFTMAX DERIVATIVES

    Tensor<double, 2> softmax_derivatives(const Tensor<double, 2>&);

    Tensor<double, 1> sign(const Tensor<double, 1>&);

    Tensor<double, 1> normalized(const Tensor<double, 1>&);

    Tensor<double, 1> absolute_value(const Tensor<double, 1>&);

    Tensor<double, 2> normalized_columns(const Tensor<double, 2>&);

    Tensor<double, 2> absolute_value(const Tensor<double, 2>&);

    void hard_sigmoid(const Tensor<double, 2>&, Tensor<double, 2>&);
    void hyperbolic_tangent(const ThreadPoolDevice& thread_pool_device, const Tensor<double, 2>&, Tensor<double, 2>&);
    void logistic(const Tensor<double, 2>&, Tensor<double, 2>&);
    void linear(const Tensor<double, 2>&, Tensor<double, 2>&);
    void threshold(const Tensor<double, 2>&, Tensor<double, 2>&);
    void symmetric_threshold(const Tensor<double, 2>&, Tensor<double, 2>&);
    void rectified_linear(const Tensor<double, 2>&, Tensor<double, 2>&);
    void scaled_exponential_linear(const Tensor<double, 2>&, Tensor<double, 2>&);
    void soft_plus(const Tensor<double, 2>&, Tensor<double, 2>&);
    void soft_sign(const Tensor<double, 2>&, Tensor<double, 2>&);
    void exponential_linear(const Tensor<double, 2>&, Tensor<double, 2>&);
    void softmax(const Tensor<double, 2>&, Tensor<double, 2>&);
    void binary(const Tensor<double, 2>&, Tensor<double, 2>&);
    void competitive(const Tensor<double, 2>&, Tensor<double, 2>&);

    void logistic_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void threshold_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void symmetric_threshold_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void linear_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void hyperbolic_tangent_derivatives(const ThreadPoolDevice& thread_pool_device, const Tensor<double, 2>&, Tensor<double, 2>&);
    void rectified_linear_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void scaled_exponential_linear_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void soft_plus_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void soft_sign_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void hard_sigmoid_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void exponential_linear_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
    void softmax_derivatives(const Tensor<double, 2>&, Tensor<double, 2>&);
}

#endif // FUNCTIONS_H
