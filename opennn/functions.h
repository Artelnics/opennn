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

// OpenNN includes

#include "vector.h"
#include "matrix.h"
#include "tensor.h"
#include "statistics.h"
#include "metrics.h"

using namespace std;

namespace OpenNN
{

    double random_uniform(const double & = -1.0, const double & = 1.0);

    double random_normal(const double & = 0.0, const double & = 1.0);

    size_t factorial(const size_t&);

    Vector<double> exponential(const Vector<double>&);

    Vector<double> logarithm(const Vector<double>&);

    Vector<double> power(const Vector<double>&, const double&);

    Tensor<double> competitive(const Tensor<double>&);

    Vector<bool> binary(const Vector<double>&);

    Vector<double> square_root(const Vector<double>&);

    Vector<double> cumulative(const Vector<double>&);

    Vector<double> lower_bounded(const Vector<double>&, const double &);

    Vector<double> lower_bounded(const Vector<double>&, const Vector<double>&);

    Vector<double> upper_bounded(const Vector<double>&, const double &);

    Vector<double> upper_bounded(const Vector<double>&, const Vector<double>&);

    Vector<double> lower_upper_bounded(const Vector<double>&, const double &, const double &);

    Vector<double> lower_upper_bounded(const Vector<double>&, const Vector<double>&, const Vector<double>&);

    Matrix<double> lower_bounded(const Matrix<double>&, const double&);
    Matrix<double> upper_bounded(const Matrix<double>&, const double&);

    Matrix<double> lower_upper_bounded(const Matrix<double>&, const Vector<double>&, const Vector<double>&);
    Tensor<double> lower_upper_bounded(const Tensor<double>&, const Vector<double>&, const Vector<double>&);

    Vector<double> hard_sigmoid(const Vector<double>&);
    Vector<double> hyperbolic_tangent(const Vector<double>&);
    Vector<double> logistic(const Vector<double>&);
    Vector<double> logistic_function(const Vector<double>&, const double&, const double&);

    Vector<double> linear(const Vector<double>&);
    Vector<double> threshold(const Vector<double>&);
    Vector<double> symmetric_threshold(const Vector<double>&);
    Vector<double> rectified_linear(const Vector<double>&);

    Vector<double> scaled_exponential_linear(const Vector<double>&);
    Vector<double> soft_plus(const Vector<double>&);
    Vector<double> soft_sign(const Vector<double>&);
    Vector<double> exponential_linear(const Vector<double>&);

    Vector<double> logistic_derivatives(const Vector<double>&);
    Vector<double> threshold_derivatives(const Vector<double>&);
    Vector<double> symmetric_threshold_derivatives(const Vector<double>&);


    Vector<double> linear_derivatives(const Vector<double>&);
    Vector<double> hyperbolic_tangent_derivatives(const Vector<double>&);
    Vector<double> rectified_linear_derivatives(const Vector<double>&);
    Vector<double> scaled_exponential_linear_derivatives(const Vector<double>&);
    Vector<double> soft_plus_derivatives(const Vector<double>&);
    Vector<double> soft_sign_derivatives(const Vector<double>&);
    Vector<double> hard_sigmoid_derivatives(const Vector<double>&);
    Vector<double> exponential_linear_derivatives(const Vector<double>&);
    Vector<double> softmax_derivatives(const Vector<double>&);


    // SINE FUNCTIONS

    Vector<double> sine(const Vector<double>&);

    Matrix<double> sine(const Matrix<double>&);


    // COSINE FUNCTIONS

    Vector<double> cosine(const Vector<double>&);

    Matrix<double> cosine(const Matrix<double>&);

    // LINEAR

    Tensor<double> linear(const Tensor<double>&);

    // HYPERBOLIC TANGENT

    Tensor<double> hyperbolic_tangent(const Tensor<double>&);

    // LOGISTIC

    Tensor<double> logistic(const Tensor<double>&);
    Tensor<double> logistic_second_derivatives(const Tensor<double>&);

    // BINARY

    Tensor<double> binary(const Tensor<double>&);

    // THRESHOLD

    Tensor<double> threshold(const Tensor<double>&);

    // SYMMETRIC THRESHOLD

    Tensor<double> symmetric_threshold(const Tensor<double>&);

    // RECTIFIED LINEAR

    Tensor<double> rectified_linear(const Tensor<double>&);

    // SCALED EXPONENTIAL LINEAR

    Tensor<double> scaled_exponential_linear(const Tensor<double>&);

    // SOFT PLUS

    Tensor<double> soft_plus(const Tensor<double>&);

    // SOFT SIGN

    Tensor<double> soft_sign(const Tensor<double>&);

    // HARD SIGMOID

    Tensor<double> hard_sigmoid(const Tensor<double>&);

    // EXPONENTIAL LINEAR

    Tensor<double> exponential_linear(const Tensor<double>&);

    // SOFTMAX

    Tensor<double> softmax(const Tensor<double>&);

    Tensor<double> softmax_rows(const Tensor<double>&);

    Matrix<double> softmax_columns(const Matrix<double>&);

    // LINEAR DERIVATIVES

    Tensor<double> linear_derivatives(const Tensor<double>&);

    // HYPERBOLIC TANGENT DERIVATIVES

    Tensor<double> hyperbolic_tangent_derivatives(const Tensor<double>&);

    // LOGISTIC DERIVATIVES

    Tensor<double> logistic_derivatives(const Tensor<double>&);

    // THRESHOLD DERIVATIVES

    Tensor<double> threshold_derivatives(const Tensor<double>&);

    // SYMMETRIC THRESHOLD DERIVATIVES

    Tensor<double> symmetric_threshold_derivatives(const Tensor<double>&);

    // RECTIFIED LINEAR DERIVATIVES

    Tensor<double> rectified_linear_derivatives(const Tensor<double>&);

    // SCALED EXPONENTIAL LINEAR DERIVATIVES

    Tensor<double> scaled_exponential_linear_derivatives(const Tensor<double>&);

    //SOFT PLUS DERIVATIVES

    Tensor<double> soft_plus_derivatives(const Tensor<double>&);

    // SOFT SIGN DERIVATIVES

    Tensor<double> soft_sign_derivatives(const Tensor<double>&);

    // HARD SIGMOID DERIVATIVES

    Tensor<double> hard_sigmoid_derivatives(const Tensor<double>&);

    // EXPONENTIAL LINEAR DERIVATIVES

    Tensor<double> exponential_linear_derivatives(const Tensor<double>&);

    // SOFTMAX DERIVATIVES

    Tensor<double> softmax_derivatives(const Tensor<double>&);

    Vector<double> sign(const Vector<double>&);

    Vector<double> normalized(const Vector<double>&);

    Vector<double> absolute_value(const Vector<double>&);

    Matrix<double> normalized_columns(const Matrix<double>&);

    Matrix<double> absolute_value(const Matrix<double>&);
}

#endif // __FUNCTIONS_H
