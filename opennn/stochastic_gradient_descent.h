//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STOCHASTICGRADIENTDESCENT_H
#define STOCHASTICGRADIENTDESCENT_H

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
#include "loss_index.h"
#include "mean_squared_error.h"
#include "optimization_algorithm.h"

namespace OpenNN
{

/// This concrete class represents the stochastic gradient descent optimization algorithm[1] for a loss index of a neural network.

/// It supports momentum, learning rate decay, and Nesterov momentum.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network." \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network


class StochasticGradientDescent : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit StochasticGradientDescent(); 

   explicit StochasticGradientDescent(LossIndex*);

   explicit StochasticGradientDescent(const tinyxml2::XMLDocument&); 

   // Destructor

   virtual ~StochasticGradientDescent();
   
   //Training operators

   const type& get_initial_learning_rate() const;
   const type& get_initial_decay() const;
   const type& get_momentum() const;
   const bool& get_nesterov() const;

   // Training parameters

   const type& get_warning_parameters_norm() const;
   const type& get_warning_gradient_norm() const;
   const type& get_error_parameters_norm() const;
   const type& get_error_gradient_norm() const;

   // Stopping criteria

   const type& get_minimum_parameters_increment_norm() const;
   const type& get_minimum_loss_increase() const;
   const type& get_loss_goal() const;
   const type& get_gradient_norm_goal() const;
   const type& get_maximum_time() const;
   const bool& get_return_minimum_selection_error_neural_network() const;
   const bool& get_apply_early_stopping() const;
   const Index& get_maximum_selection_failures() const;

   // Reserve training history

   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_default();

   void set_reserve_all_training_history(const bool&);

   //Training operators

   void set_initial_learning_rate(const type&);
   void set_initial_decay(const type&);
   void set_momentum(const type&);
   void set_nesterov(const bool&);

   // Training parameters

   void set_warning_parameters_norm(const type&);
   void set_warning_gradient_norm(const type&);
   void set_error_parameters_norm(const type&);
   void set_error_gradient_norm(const type&);
   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const type&);
   void set_minimum_loss_increase(const type&);
   void set_loss_goal(const type&);
   void set_gradient_norm_goal(const type&);
   void set_maximum_selection_error_increases(const Index&);
   void set_maximum_time(const type&);
   void set_return_minimum_selection_error_neural_network(const bool&);
   void set_apply_early_stopping(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   // Utilities

   void set_display_period(const Index&);

   // Training methods

   Results perform_training();

   void perform_training_void();

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   void update_parameters(const LossIndex::FirstOrderLoss& first_order_loss)
   {
       /*
       NeuralNetwork* neural_network_pointer = get_loss_index_pointer()->get_neural_network_pointer();

       type learning_rate_iteration

       type learning_rate;

       initial_decay > 0 ? learning_rate = initial_learning_rate * (1 / (1 + learning_rate_iteration*initial_decay)) : initial_learning_rate ;

       parameters_increment.device(thread_pool_device) = first_order_loss.gradient*static_cast<type>(-learning_rate);

       if(momentum > 0 && !nesterov)
       {
           parameters_increment.device(thread_pool_device) += last_increment*momentum;

           last_increment = parameters_increment;

           parameters.device(thread_pool_device) += parameters_increment;
       }
       else if(momentum > 0 && nesterov)
       {
           parameters_increment.device(thread_pool_device) += last_increment*momentum;

           last_increment = parameters_increment;

           nesterov_increment.device(thread_pool_device) = parameters_increment*momentum - first_order_loss.gradient*learning_rate;

           parameters.device(thread_pool_device) += nesterov_increment;
       }
       else
       {
           parameters.device(thread_pool_device) = parameters + parameters_increment;
       }

       neural_network_pointer->set_parameters(parameters);

       learning_rate_iteration++;
*/
   }


private:

   // TRAINING OPERATORS

   /// Initial learning rate

   type initial_learning_rate;

   /// Learning rate decay over each update.

   type initial_decay;

   /// Parameter that accelerates SGD in the relevant direction and dampens oscillations.

   type momentum;

   /// Boolean. Whether to apply Nesterov momentum.

   bool nesterov;

   // TRAINING PARAMETERS

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

   type loss_goal;

   /// Goal value for the norm of the error function gradient. It is used as a stopping criterion.

   type gradient_norm_goal;

   /// Maximum number of iterations at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_failures;

   /// Maximum epochs number

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool return_minimum_selection_error_neural_network;

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
