//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R      
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef QUASINEWTONMETHOD_H
#define QUASINEWTONMETHOD_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "config.h"
#include "metrics.h"
#include "loss_index.h"

#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"

#include "tinyxml2.h"

namespace OpenNN
{

///Class of optimization algorithm based on Newton's method. An approximate Hessian matrix is computed at each iteration of the algorithm based on the gradients.

/// This concrete class represents a quasi-Newton training algorithm[1], used to minimize loss function.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network." \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network


class QuasiNewtonMethod : public OptimizationAlgorithm
{

public:

   // Enumerations

   /// Enumeration of the available training operators for obtaining the approximation to the inverse hessian.

   enum InverseHessianApproximationMethod{DFP, BFGS};


   // DEFAULT CONSTRUCTOR

   explicit QuasiNewtonMethod();

   // LOSS INDEX CONSTRUCTOR

   explicit QuasiNewtonMethod(LossIndex*);

   

   explicit QuasiNewtonMethod(const tinyxml2::XMLDocument&);

   virtual ~QuasiNewtonMethod();

   // Get methods

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm_pointer();

   const InverseHessianApproximationMethod& get_inverse_hessian_approximation_method() const;
   string write_inverse_hessian_approximation_method() const;

   // Training parameters

   const type& get_warning_parameters_norm() const;
   const type& get_warning_gradient_norm() const;
   const type& get_warning_learning_rate() const;

   const type& get_error_parameters_norm() const;
   const type& get_error_gradient_norm() const;
   const type& get_error_learning_rate() const;

   const Index& get_epochs_number() const;

   // Stopping criteria

   const type& get_minimum_parameters_increment_norm() const;

   const type& get_minimum_loss_increase() const;
   const type& get_loss_goal() const;
   const type& get_gradient_norm_goal() const;
   const Index& get_maximum_selection_error_increases() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   const bool& get_return_minimum_selection_error_neural_network() const;
   const bool& get_apply_early_stopping() const;

   // Reserve training history


   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_inverse_hessian_approximation_method(const InverseHessianApproximationMethod&);
   void set_inverse_hessian_approximation_method(const string&);

   void set_display(const bool&);

   void set_default();

   // Training parameters

   void set_warning_parameters_norm(const type&);
   void set_warning_gradient_norm(const type&);
   void set_warning_learning_rate(const type&);

   void set_error_parameters_norm(const type&);
   void set_error_gradient_norm(const type&);
   void set_error_learning_rate(const type&);

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const type&);

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);
   void set_gradient_norm_goal(const type&);
   void set_maximum_selection_error_increases(const Index&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   void set_return_minimum_selection_error_neural_network(const bool&);
   void set_apply_early_stopping(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   // Utilities

   void set_display_period(const Index&);

   // Training methods

   Tensor<type, 1> calculate_gradient_descent_training_direction(const Tensor<type, 1>&) const;

   Tensor<type, 2> calculate_DFP_inverse_hessian
  (const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_BFGS_inverse_hessian
  (const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_inverse_hessian_approximation(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>&) const;
   void update_inverse_hessian_approximation(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&, const Tensor<type, 2>&) const;

   Results perform_training();
   void perform_training_void();

   // Training history methods

   void set_reserve_all_training_history(const bool&);

   string write_optimization_algorithm_type() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   string object_to_string() const;
   Tensor<string, 2> to_string_matrix() const;

private: 

   /// Training rate algorithm object. 
   /// It is used to calculate the step for the quasi-Newton training direction.

   LearningRateAlgorithm learning_rate_algorithm;

   /// Variable containing the actual method used to obtain a suitable training rate. 

   InverseHessianApproximationMethod inverse_hessian_approximation_method;

   /// Value for the parameters norm at which a warning message is written to the screen. 

   type warning_parameters_norm;

   /// Value for the gradient norm at which a warning message is written to the screen. 

   type warning_gradient_norm;

   /// Training rate value at wich a warning message is written to the screen.

   type warning_learning_rate;

   /// Value for the parameters norm at which the training process is assumed to fail. 
   
   type error_parameters_norm;

   /// Value for the gradient norm at which the training process is assumed to fail. 

   type error_gradient_norm;

   /// Training rate at wich the line minimization algorithm is assumed to be unable to bracket a minimum.

   type error_learning_rate;


   // Stopping criteria

   /// Norm of the parameters increment vector at which training stops.

   type minimum_parameters_increment_norm;

   /// Minimum loss improvement between two successive epochs. It is used as a stopping criterion.

   type minimum_loss_decrease;

   /// Goal value for the loss. It is used as a stopping criterion.

   type loss_goal;

   /// Goal value for the norm of the error function gradient. It is used as a stopping criterion.

   type gradient_norm_goal;

   /// Maximum number of epochs at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_error_increases;

   /// Maximum number of epochs to perform_training. It is used as a stopping criterion.

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool return_minimum_selection_error_neural_network;

   /// True if the selection error decrease stopping criteria has to be taken in account, false otherwise.

   bool apply_early_stopping;

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
