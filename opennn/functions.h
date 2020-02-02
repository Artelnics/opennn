//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F U N C T I O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// System includes

#include <math.h>
#include <vector>

// OpenNN includes

#include "config.h"

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
//#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

using namespace std;
using namespace Eigen;

namespace OpenNN
{

/*
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
*/
}

#endif // FUNCTIONS_H
