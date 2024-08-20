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
#include "cross_entropy_error_3d.h"
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

class TrainingStrategy
{

public:

    // Constructors

    explicit TrainingStrategy();

    explicit TrainingStrategy(NeuralNetwork*, DataSet*);

    // Enumerations

    enum class LossMethod
    {
        SUM_SQUARED_ERROR,
        MEAN_SQUARED_ERROR,
        NORMALIZED_SQUARED_ERROR,
        MINKOWSKI_ERROR,
        WEIGHTED_SQUARED_ERROR,
        CROSS_ENTROPY_ERROR,
        CROSS_ENTROPY_ERROR_3D
    };

    enum class OptimizationMethod
    {
        GRADIENT_DESCENT,
        CONJUGATE_GRADIENT,
        QUASI_NEWTON_METHOD,
        LEVENBERG_MARQUARDT_ALGORITHM,
        STOCHASTIC_GRADIENT_DESCENT,
        ADAPTIVE_MOMENT_ESTIMATION
    };

    // Get

    DataSet* get_data_set();

    NeuralNetwork* get_neural_network() const;

    LossIndex* get_loss_index();
    OptimizationAlgorithm* get_optimization_algorithm();

    bool has_neural_network() const;
    bool has_data_set() const;

    SumSquaredError* get_sum_squared_error();
    MeanSquaredError* get_mean_squared_error();
    NormalizedSquaredError* get_normalized_squared_error();
    MinkowskiError* get_Minkowski_error();
    CrossEntropyError* get_cross_entropy_error();
    WeightedSquaredError* get_weighted_squared_error();

    GradientDescent* get_gradient_descent();
    ConjugateGradient* get_conjugate_gradient();
    QuasiNewtonMethod* get_quasi_Newton_method();
    LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm();
    StochasticGradientDescent* get_stochastic_gradient_descent();
    AdaptiveMomentEstimation* get_adaptive_moment_estimation();

    const LossMethod& get_loss_method() const;
    const OptimizationMethod& get_optimization_method() const;

    string write_loss_method() const;
    string write_optimization_method() const;

    string write_optimization_method_text() const;
    string write_loss_method_text() const;

    const bool& get_display() const;

    // Set

    void set();
    void set(NeuralNetwork*, DataSet*);
    void set_default() const;

    void set_threads_number(const int&);

    void set_data_set(DataSet*);
    void set_neural_network(NeuralNetwork*);

    void set_loss_index_threads_number(const int&);
    void set_optimization_algorithm_threads_number(const int&);

    void set_loss_index(LossIndex*);
    void set_loss_index_data_set(DataSet*);
    void set_loss_index_neural_network(NeuralNetwork*);

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

    // Training

    TrainingResults perform_training();

    // Check

    void fix_forecasting();

    // Serialization

    void print() const;

    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;

    void save(const string&) const;
    void load(const string&);

private:

    DataSet* data_set = nullptr;

    NeuralNetwork* neural_network = nullptr;

    // Loss index

    SumSquaredError sum_squared_error;

    MeanSquaredError mean_squared_error;

    NormalizedSquaredError normalized_squared_error;

    MinkowskiError Minkowski_error;

    CrossEntropyError cross_entropy_error;

    CrossEntropyError3D cross_entropy_error_3d;

    WeightedSquaredError weighted_squared_error;

    LossMethod loss_method;

    // Optimization algorithm

    GradientDescent gradient_descent;

    ConjugateGradient conjugate_gradient;

    QuasiNewtonMethod quasi_Newton_method;

    LevenbergMarquardtAlgorithm Levenberg_Marquardt_algorithm;

    StochasticGradientDescent stochastic_gradient_descent;

    AdaptiveMomentEstimation adaptive_moment_estimation;

    OptimizationMethod optimization_method;

    bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/training_strategy_cuda.h"
#endif

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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

