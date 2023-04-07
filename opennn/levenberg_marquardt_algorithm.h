//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEVENBERGMARQUARDTALGORITHM_H
#define LEVENBERGMARQUARDTALGORITHM_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <math.h>
#include <time.h>

// OpenNN includes

#include "config.h"
#include "tensor_utilities.h"
#include "optimization_algorithm.h"

// Eigen includes

#include "../eigen/Eigen/Dense"

namespace opennn
{

struct LevenbergMarquardtAlgorithmData;

/// Levenberg-Marquardt Algorithm will always compute the approximate Hessian matrix, which has dimensions n-by-n.

/// This concrete class represents the Levenberg-Marquardt (LM) optimization algorithm[1], use to minimize loss function.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network."
/// \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network


class LevenbergMarquardtAlgorithm : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit LevenbergMarquardtAlgorithm();

   explicit LevenbergMarquardtAlgorithm(LossIndex*);

   // Get methods

   // Stopping criteria

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;

   const Index& get_maximum_selection_failures() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   // Utilities

   const type& get_damping_parameter() const;

   const type& get_damping_parameter_factor() const;

   const type& get_minimum_damping_parameter() const;
   const type& get_maximum_damping_parameter() const;

   // Set methods

   void set_default() override;

   void set_damping_parameter(const type&);

   void set_damping_parameter_factor(const type&);

   void set_minimum_damping_parameter(const type&);
   void set_maximum_damping_parameter(const type&);

   // Stopping criteria

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);

   void set_maximum_selection_failures(const Index&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   // Training methods

   void check() const final;

   TrainingResults perform_training() final;

   void update_parameters(
           const DataSetBatch&,
           NeuralNetworkForwardPropagation&,
           LossIndexBackPropagationLM&,
           LevenbergMarquardtAlgorithmData&);

   string write_optimization_algorithm_type() const final;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const final;
   
   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;
   
private:

   // MEMBERS

   /// Initial Levenberg-Marquardt parameter.

   type damping_parameter;

   /// Minimum Levenberg-Marquardt parameter.

   type minimum_damping_parameter;

   /// Maximum Levenberg-Marquardt parameter.

   type maximum_damping_parameter;

   /// Damping parameter increase/decrease factor.

   type damping_parameter_factor;

   // Stopping criteria 

   /// Minimum loss improvement between two successive iterations. It is a stopping criterion.

   type minimum_loss_decrease;

   /// Goal value for the loss. It is a stopping criterion.

   type training_loss_goal;

   /// Maximum number of epochs at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_failures;

   /// Maximum number of epoch to perform_training. It is a stopping criterion.

   Index maximum_epochs_number;

   /// Maximum training time. It is a stopping criterion.

   type maximum_time;
};


struct LevenbergMarquardtAlgorithmData : public OptimizationAlgorithmData
{
    /// Default constructor.

    explicit LevenbergMarquardtAlgorithmData()
    {
    }

    explicit LevenbergMarquardtAlgorithmData(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method_pointer)
    {
        set(new_Levenberg_Marquardt_method_pointer);
    }

    virtual ~LevenbergMarquardtAlgorithmData() {}

    void set(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method_pointer)
    {
        Levenberg_Marquardt_algorithm = new_Levenberg_Marquardt_method_pointer;

        const LossIndex* loss_index_pointer = Levenberg_Marquardt_algorithm->get_loss_index_pointer();

        const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index parameters_number = neural_network_pointer->get_parameters_number();

        // Neural network data

        old_parameters.resize(parameters_number);

        parameters_difference.resize(parameters_number);

        potential_parameters.resize(parameters_number);
        parameters_increment.resize(parameters_number);
    }

    LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm = nullptr;

    // Neural network data

//    Tensor<type, 1> potential_parameters;

    Tensor<type, 1> old_parameters;
    Tensor<type, 1> parameters_difference;

    Tensor<type, 1> parameters_increment;

    // Loss index data

    type old_loss = type(0);

    // Optimization algorithm data

    Index epoch = 0;
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
