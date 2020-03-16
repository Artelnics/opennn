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
#include "optimization_algorithm.h"
#include "tinyxml2.h"

namespace OpenNN
{

/// Levenberg-Marquardt Algorithm will always compute the approximate Hessian matrix, which has dimensions n-by-n.

/// This concrete class represents a Levenberg-Marquardt Algorithm training algorithm[1], use to minimize loss function.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network."
/// \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network


class LevenbergMarquardtAlgorithm : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit LevenbergMarquardtAlgorithm();

   explicit LevenbergMarquardtAlgorithm(LossIndex*);

   explicit LevenbergMarquardtAlgorithm(const tinyxml2::XMLDocument&);

   // Destructor

   virtual ~LevenbergMarquardtAlgorithm();

   // Get methods

   // Training parameters

   const type& get_warning_parameters_norm() const;
   const type& get_warning_gradient_norm() const;

   const type& get_error_parameters_norm() const;
   const type& get_error_gradient_norm() const;

   // Stopping criteria

   const type& get_minimum_parameters_increment_norm() const;

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;
   const type& get_gradient_norm_goal() const;
   const Index& get_maximum_selection_error_increases() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   const bool& get_choose_best_selection() const;
   const bool& get_apply_early_stopping() const;

   // Reserve training history

   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Utilities

   const type& get_damping_parameter() const;

   const type& get_damping_parameter_factor() const;

   const type& get_minimum_damping_parameter() const;
   const type& get_maximum_damping_parameter() const;

   const Tensor<type, 1>& get_damping_parameter_history() const;

   // Set methods

   void set_default();

   void set_damping_parameter(const type&);

   void set_damping_parameter_factor(const type&);

   void set_minimum_damping_parameter(const type&);
   void set_maximum_damping_parameter(const type&);

   // Training parameters

   void set_warning_parameters_norm(const type&);
   void set_warning_gradient_norm(const type&);

   void set_error_parameters_norm(const type&);
   void set_error_gradient_norm(const type&);

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const type&);

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);
   void set_gradient_norm_goal(const type&);
   void set_maximum_selection_error_increases(const Index&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   void set_choose_best_selection(const bool&);
   void set_apply_early_stopping(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   /// Makes the training history of all variables to be reseved or not in memory.

   virtual void set_reserve_all_training_history(const bool&);

   // Utilities

   void set_display_period(const Index&);

   // Training methods

   void check() const;

   Results perform_training();

   void perform_training_void();

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   
   Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

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

   /// Value for the parameters norm at which a warning message is written to the screen. 

   type warning_parameters_norm;

   /// Value for the gradient norm at which a warning message is written to the screen. 

   type warning_gradient_norm;

   /// Value for the parameters norm at which the training process is assumed to fail. 
   
   type error_parameters_norm;

   /// Value for the gradient norm at which the training process is assumed to fail. 

   type error_gradient_norm;


   // Stopping criteria

   /// Norm of the parameters increment vector at which training stops.

   type minimum_parameters_increment_norm;

   /// Minimum loss improvement between two successive iterations. It is used as a stopping criterion.

   type minimum_loss_decrease;

   /// Goal value for the loss. It is used as a stopping criterion.

   type training_loss_goal;

   /// Goal value for the norm of the error function gradient. It is used as a stopping criterion.

   type gradient_norm_goal;

   /// Maximum number of iterations at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_error_increases;

   /// Maximum number of epoch to perform_training. It is used as a stopping criterion.

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool choose_best_selection;

   /// True if the selection error decrease stopping criteria has to be taken in account, false otherwise.

   bool apply_early_stopping;

   // TRAINING HISTORY

   /// True if the loss history vector is to be reserved, false otherwise.

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
