//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADAPTIVEMOMENTESTIMATION_H
#define ADAPTIVEMOMENTESTIMATION_H

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
#include <chrono>
#include <time.h>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

// OpenNN includes

#include "loss_index.h"
#include "optimization_algorithm.h"
#include "config.h"

namespace OpenNN
{

/// This concrete class represents the adaptive moment estimation(Adam) training algorithm, based on adaptative estimates of lower-order moments.

///
/// For more information visit:
///
/// \cite 1 C. Barranquero "High performance optimization algorithms for neural networks." \ref https://www.opennn.net/files/high_performance_optimization_algorithms_for_neural_networks.pdf .
///
/// \cite 2 D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980v8 (2014).

class AdaptiveMomentEstimation : public OptimizationAlgorithm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit AdaptiveMomentEstimation();

   // LOSS INDEX CONSTRUCTOR

   explicit AdaptiveMomentEstimation(LossIndex*);   

   explicit AdaptiveMomentEstimation(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~AdaptiveMomentEstimation();

   // Enumerations

    ///@todo, to remove
   /// Enumeration of Adam's variations.

   
   /// Get methods in training operators

   // Training operators

   const type& get_initial_learning_rate() const;
   const type& get_beta_1() const;
   const type& get_beta_2() const;
   const type& get_epsilon() const;


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
   const bool& get_choose_best_selection() const;
   const bool& get_apply_early_stopping() const;
   const Index& get_maximum_selection_error_increases() const;

   // Reserve training history

   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_default();

   void set_reserve_all_training_history(const bool&);

   // Training operators

   void set_initial_learning_rate(const type&);
   void set_beta_1(const type&);
   void set_beta_2(const type&);
   void set_epsilon(const type&);

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
   void set_choose_best_selection(const bool&);
   void set_apply_early_stopping(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   // Utilities

   void set_display_period(const Index&);

   // Training methods

   Results perform_training();

   /// Perform Neural Network training.

   void perform_training_void();

   /// Return the algorithm optimum for your model.

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const;

   tinyxml2::XMLDocument* to_XML() const;

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   // TRAINING OPERATORS

   /// Initial learning rate

   type initial_learning_rate;

   /// Learning rate decay over each update.

   type initial_decay;

   /// Exponential decay over gradient estimates.

   type beta_1;

   /// Exponential decay over square gradient estimates.

   type beta_2;

   /// Small number to prevent any division by zero

   type epsilon;

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

   Index maximum_selection_error_increases;

   /// Maximum number of iterations to perform_training. It is used as a stopping criterion.

   Index maximum_iterations_number;

   /// Initial batch size

   Index training_initial_batch_size;

   /// Maximum training batch size

   Index training_maximum_batch_size;

   /// Maximum epochs number

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool choose_best_selection;

   /// True if the selection error decrease stopping criteria has to be taken in account, false otherwise.

   bool apply_early_stopping;

   // TRAINING HISTORY

   /// True if the error history vector is to be reserved, false otherwise.

   bool reserve_training_error_history;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history;
};

}

#endif
