//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRAININGSTRATEGY_H
#define TRAININGSTRATEGY_H

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "sum_squared_error.h"
#include "mean_squared_error.h"
#include "normalized_squared_error.h"
#include "minkowski_error.h"
#include "cross_entropy_error.h"
#include "weighted_squared_error.h"

#include "optimization_algorithm.h"

#include "gradient_descent.h"
#include "conjugate_gradient.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"
#include "stochastic_gradient_descent.h"
#include "adaptive_moment_estimation.h"


namespace opennn
{

/// This class represents the concept of training strategy for a neural network in OpenNN.

///
/// A training strategy is composed of two objects:
/// <ul>
/// <li> Loss index.
/// <li> Optimization algorithm.
/// </ul> 

class TrainingStrategy
{

public:

    // Constructors

    explicit TrainingStrategy();

    explicit TrainingStrategy(NeuralNetwork*, DataSet*);

    // Enumerations

    /// Enumeration of the available error terms in OpenNN.

    enum class LossMethod
    {
        SUM_SQUARED_ERROR,
        MEAN_SQUARED_ERROR,
        NORMALIZED_SQUARED_ERROR,
        MINKOWSKI_ERROR,
        WEIGHTED_SQUARED_ERROR,
        CROSS_ENTROPY_ERROR
    };

    /// Enumeration of all the available types of optimization algorithms.

    enum class OptimizationMethod
    {
        GRADIENT_DESCENT,
        CONJUGATE_GRADIENT,
        QUASI_NEWTON_METHOD,
        LEVENBERG_MARQUARDT_ALGORITHM,
        STOCHASTIC_GRADIENT_DESCENT,
        ADAPTIVE_MOMENT_ESTIMATION
    };

    // Get methods

    DataSet* get_data_set_pointer();

    NeuralNetwork* get_neural_network_pointer() const;

    LossIndex* get_loss_index_pointer();
    OptimizationAlgorithm* get_optimization_algorithm_pointer();

    bool has_neural_network() const;
    bool has_data_set() const;

    SumSquaredError* get_sum_squared_error_pointer();
    MeanSquaredError* get_mean_squared_error_pointer();
    NormalizedSquaredError* get_normalized_squared_error_pointer();
    MinkowskiError* get_Minkowski_error_pointer();
    CrossEntropyError* get_cross_entropy_error_pointer();
    WeightedSquaredError* get_weighted_squared_error_pointer();

    GradientDescent* get_gradient_descent_pointer();
    ConjugateGradient* get_conjugate_gradient_pointer();
    QuasiNewtonMethod* get_quasi_Newton_method_pointer();
    LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer();
    StochasticGradientDescent* get_stochastic_gradient_descent_pointer();
    AdaptiveMomentEstimation* get_adaptive_moment_estimation_pointer();

    const LossMethod& get_loss_method() const;
    const OptimizationMethod& get_optimization_method() const;

    string write_loss_method() const;
    string write_optimization_method() const;

    string write_optimization_method_text() const;
    string write_loss_method_text() const;

    const bool& get_display() const;

    // Set methods

    void set();
    void set(NeuralNetwork*, DataSet*);
    void set_default() const;

    void set_threads_number(const int&);

    void set_data_set_pointer(DataSet*);
    void set_neural_network_pointer(NeuralNetwork*);

    void set_loss_index_threads_number(const int&);
    void set_optimization_algorithm_threads_number(const int&);

    void set_loss_index_pointer(LossIndex*);
    void set_loss_index_data_set_pointer(DataSet*);
    void set_loss_index_neural_network_pointer(NeuralNetwork*);

    void set_loss_method(const LossMethod&);
    void set_optimization_method(const OptimizationMethod&);

    void set_loss_method(const string&);
    void set_optimization_method(const string&);

    void set_display(const bool&);

    void set_loss_goal(const type&);
    void set_maximum_selection_failures(const Index&);
    void set_maximum_epochs_number(const int&);
    void set_display_period(const int&);

    void set_maximum_time(const type&);

    // Training methods

    TrainingResults perform_training();


    // Check methods

    void fix_forecasting();

    // Serialization methods

    void print() const;

    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;

    void save(const string&) const;
    void load(const string&);

private:

    DataSet* data_set_pointer = nullptr;

    NeuralNetwork* neural_network_pointer = nullptr;

    // Loss index

    /// Pointer to the sum squared error object wich can be used as the error term.

    SumSquaredError sum_squared_error;

    /// Pointer to the mean squared error object wich can be used as the error term.

    MeanSquaredError mean_squared_error;

    /// Pointer to the normalized squared error object wich can be used as the error term.

    NormalizedSquaredError normalized_squared_error;

    /// Pointer to the Mikowski error object wich can be used as the error term.

    MinkowskiError Minkowski_error;

    /// Pointer to the cross-entropy error object wich can be used as the error term.

    CrossEntropyError cross_entropy_error;

    /// Pointer to the weighted squared error object wich can be used as the error term.

    WeightedSquaredError weighted_squared_error;

    /// Type of loss method.

    LossMethod loss_method;

    // Optimization algorithm

    /// Gradient descent object to be used as a main optimization algorithm.

    GradientDescent gradient_descent;

    /// Conjugate gradient object to be used as a main optimization algorithm.

    ConjugateGradient conjugate_gradient;

    /// Quasi-Newton method object to be used as a main optimization algorithm.

    QuasiNewtonMethod quasi_Newton_method;

    /// Levenberg-Marquardt algorithm object to be used as a main optimization algorithm.

    LevenbergMarquardtAlgorithm Levenberg_Marquardt_algorithm;

    /// Stochastic gradient descent algorithm object to be used as a main optimization algorithm.

    StochasticGradientDescent stochastic_gradient_descent;

    /// Adaptive moment estimation algorithm object to be used as a main optimization algorithm.

    AdaptiveMomentEstimation adaptive_moment_estimation;

    /// Type of main optimization algorithm.

    OptimizationMethod optimization_method;

    /// Display messages to screen.

    bool display = true;

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn-cuda/training_strategy_cuda.h"
#endif

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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

