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

#include "tinyxml2.h"

namespace OpenNN
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

    explicit TrainingStrategy(const tinyxml2::XMLDocument&);

    explicit TrainingStrategy(const string&);

   // Destructor

   virtual ~TrainingStrategy();

   // Enumerations

    /// Enumeration of available error terms in OpenNN.

    enum LossMethod
    {
       SUM_SQUARED_ERROR,
       MEAN_SQUARED_ERROR,
       NORMALIZED_SQUARED_ERROR,
       MINKOWSKI_ERROR,
       WEIGHTED_SQUARED_ERROR,
       CROSS_ENTROPY_ERROR
    };

    /// Enumeration of all the available types of optimization algorithms.

    enum OptimizationMethod
    {
       GRADIENT_DESCENT,
       CONJUGATE_GRADIENT,
       QUASI_NEWTON_METHOD,
       LEVENBERG_MARQUARDT_ALGORITHM,
       STOCHASTIC_GRADIENT_DESCENT,
       ADAPTIVE_MOMENT_ESTIMATION
    };

   // Get methods

   NeuralNetwork* get_neural_network_pointer() const;

   LossIndex* get_loss_index_pointer() const;
   OptimizationAlgorithm* get_optimization_algorithm_pointer() const;

   bool has_neural_network() const;
   bool has_data_set() const;

   bool has_loss_index() const;
   bool has_optimization_algorithm() const;

   GradientDescent* get_gradient_descent_pointer() const;
   ConjugateGradient* get_conjugate_gradient_pointer() const;
   QuasiNewtonMethod* get_quasi_Newton_method_pointer() const;
   LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer() const;
   StochasticGradientDescent* get_stochastic_gradient_descent_pointer() const;
   AdaptiveMomentEstimation* get_adaptive_moment_estimation_pointer() const;

   SumSquaredError* get_sum_squared_error_pointer() const;
   MeanSquaredError* get_mean_squared_error_pointer() const;
   NormalizedSquaredError* get_normalized_squared_error_pointer() const;
   MinkowskiError* get_Minkowski_error_pointer() const;
   CrossEntropyError* get_cross_entropy_error_pointer() const;
   WeightedSquaredError* get_weighted_squared_error_pointer() const;

   const LossMethod& get_loss_method() const;
   const OptimizationMethod& get_optimization_method() const;

   string write_loss_method() const;
   string write_optimization_method() const;

   string write_optimization_method_text() const;
   string write_loss_method_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set_default();

   void set_loss_index_pointer(LossIndex*);

   void set_loss_method(const LossMethod&);
   void set_optimization_method(const OptimizationMethod&);

   void set_loss_method(const string&);
   void set_optimization_method(const string&);

   void set_display(const bool&);

   // Pointer methods

   void destruct_optimization_algorithm();

   // Training methods

   // This method trains a neural network which has a loss index associated.

   OptimizationAlgorithm::Results perform_training() const;
   void perform_training_void() const;

   // Check methods

   bool check_forecasting() const;

   // Serialization methods

   string object_to_string() const;

   void print() const;

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);   

   void write_XML(tinyxml2::XMLPrinter&) const;
   
   void save(const string&) const;
   void load(const string&);

private:

   DataSet* data_set_pointer = nullptr;

   NeuralNetwork* neural_network_pointer = nullptr;

    // Loss index

    /// Pointer to the sum squared error object wich can be used as the error term.

    SumSquaredError* sum_squared_error_pointer = nullptr;

    /// Pointer to the mean squared error object wich can be used as the error term.

    MeanSquaredError* mean_squared_error_pointer = nullptr;

    /// Pointer to the normalized squared error object wich can be used as the error term.

    NormalizedSquaredError* normalized_squared_error_pointer = nullptr;

    /// Pointer to the Mikowski error object wich can be used as the error term.

    MinkowskiError* Minkowski_error_pointer = nullptr;

    /// Pointer to the cross entropy error object wich can be used as the error term.

    CrossEntropyError* cross_entropy_error_pointer = nullptr;

    /// Pointer to the weighted squared error object wich can be used as the error term.

    WeightedSquaredError* weighted_squared_error_pointer = nullptr;

    /// Type of loss method.

    LossMethod loss_method;

    // Optimization algorithm

    /// Pointer to a gradient descent object to be used as a main optimization algorithm.

    GradientDescent* gradient_descent_pointer = nullptr;

    /// Pointer to a conjugate gradient object to be used as a main optimization algorithm.

    ConjugateGradient* conjugate_gradient_pointer = nullptr;

    /// Pointer to a quasi-Newton method object to be used as a main optimization algorithm.

    QuasiNewtonMethod* quasi_Newton_method_pointer = nullptr;

    /// Pointer to a Levenberg-Marquardt algorithm object to be used as a main optimization algorithm.

    LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm_pointer = nullptr;

    /// Pointer to a stochastic gradient descent algorithm object to be used as a main optimization algorithm.

    StochasticGradientDescent* stochastic_gradient_descent_pointer = nullptr;

    /// Pointer to a adaptive moment estimation algorithm object to be used as a main optimization algorithm.

    AdaptiveMomentEstimation* adaptive_moment_estimation_pointer = nullptr;

    /// Type of main optimization algorithm.

    OptimizationMethod optimization_method;

    /// Display messages to screen.

    bool display;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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

