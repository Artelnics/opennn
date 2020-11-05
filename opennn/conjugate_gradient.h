//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONJUGATEGRADIENT_H
#define CONJUGATEGRADIENT_H

// System inlcludes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <climits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"

namespace OpenNN
{

/// In the conjugate gradient algorithms a search is performed along conjugate directions,
/// which produces generally faster convergence than a search along the steepest descent directions.

/// This concrete class represents a conjugate gradient training algorithm, based on solving sparse systems.
///
/// \cite 1 \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network
///
/// \cite 2 D.P. O'Leary "The Block Conjugate Gradient Algorithm and Related Methods."

class ConjugateGradient : public OptimizationAlgorithm
{

public:

    struct GGOptimizationData : public OptimizationData
    {
        /// Default constructor.

        explicit GGOptimizationData();

        explicit GGOptimizationData(ConjugateGradient*);

        virtual ~GGOptimizationData();

        void set(ConjugateGradient*);

        void print() const;

        ConjugateGradient* conjugate_gradient_pointer = nullptr;

        Tensor<type, 1> parameters_increment;

        Tensor<type, 1> old_gradient;

        Tensor<type, 1> old_training_direction;

        Index epoch = 0;

        type learning_rate = 0;
        type old_learning_rate = 0;

        type parameters_increment_norm = 0;

        Tensor<type, 0> training_slope;
    };

   // Enumerations

   /// Enumeration of the available training operators for obtaining the training direction.

   enum TrainingDirectionMethod{PR, FR};

   // DEFAULT CONSTRUCTOR

   explicit ConjugateGradient(); 

   explicit ConjugateGradient(LossIndex*);   

   virtual ~ConjugateGradient();

   // Get methods

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm_pointer();

   // Training operators

   const TrainingDirectionMethod& get_training_direction_method() const;
   string write_training_direction_method() const;

   // Stopping criteria

   const type& get_minimum_parameters_increment_norm() const;

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;
   const Index& get_maximum_selection_error_increases() const;
   const type& get_gradient_norm_goal() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   const bool& get_choose_best_selection() const;

   // Reserve training history

   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Set methods

   void set_default();

   void set_loss_index_pointer(LossIndex*);

   // Training operators

   void set_training_direction_method(const TrainingDirectionMethod&);
   void set_training_direction_method(const string&);

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const type&);

   void set_loss_goal(const type&);
   void set_minimum_loss_decrease(const type&);
   void set_maximum_selection_error_increases(const Index&);
   void set_gradient_norm_goal(const type&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   void set_choose_best_selection(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   void set_reserve_all_training_history(const bool&);

   // Utilities

   void set_save_period(const Index&);

   // Training direction methods

   type calculate_PR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const;
   type calculate_FR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   void calculate_PR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_FR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, Tensor<type, 1>&) const;

   void calculate_gradient_descent_training_direction(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_conjugate_gradient_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, Tensor<type, 1>&) const;

   // Training methods

   Results perform_training();

   void perform_training_void();

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const;

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   void update_epoch(
           const DataSet::Batch& batch,
           NeuralNetwork::ForwardPropagation& forward_propagation,
           LossIndex::BackPropagation& back_propagation,
           GGOptimizationData& optimization_data);

private:

   type first_learning_rate = static_cast<type>(0.01);

   /// Applied method for calculating the conjugate gradient direction.

   TrainingDirectionMethod training_direction_method = ConjugateGradient::FR;

   /// Learning rate algorithm object for one-dimensional minimization. 

   LearningRateAlgorithm learning_rate_algorithm;

   // Stopping criteria

   /// Norm of the parameters increment vector at which training stops.

   type minimum_parameters_increment_norm = 0;

   /// Minimum loss improvement between two successive iterations. It is used as a stopping criterion.

   type minimum_loss_decrease = 0;

   /// Goal value for the loss. It is used as a stopping criterion.

   type training_loss_goal = 0;

   /// Goal value for the norm of the error function gradient. It is used as a stopping criterion.

   type gradient_norm_goal = 0;

   /// Maximum number of epochs at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_error_increases;

   /// Maximum number of epochs to perform_training. It is used as a stopping criterion.

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool choose_best_selection = false;

   // TRAINING HISTORY

   /// True if the training error history vector is to be reserved, false otherwise.

   bool reserve_training_error_history;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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

