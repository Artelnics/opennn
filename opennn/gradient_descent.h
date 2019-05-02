/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R                                                  */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __GRADIENTDESCENT_H__
#define __GRADIENTDESCENT_H__

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

#include "loss_index.h"

#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"


namespace OpenNN
{

/// This concrete class represents the gradient descent optimization algorithm for
/// a loss index of a neural network.

class GradientDescent : public OptimizationAlgorithm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit GradientDescent(); 

   // LOSS INDEX CONSTRUCTOR

   explicit GradientDescent(LossIndex*);

   // XML CONSTRUCTOR

   explicit GradientDescent(const tinyxml2::XMLDocument&); 

   // DESTRUCTOR

   virtual ~GradientDescent();

   // STRUCTURES

   ///
   /// This structure contains the training results for the gradient descent. 
   ///

   struct GradientDescentResults : public OptimizationAlgorithm::OptimizationAlgorithmResults
   {
       /// Default constructor.

       GradientDescentResults()
       {
           gradient_descent_pointer = nullptr;
       }

       /// Gradient descent constructor.

       GradientDescentResults(GradientDescent* new_gradient_descent_pointer)
       {
           gradient_descent_pointer = new_gradient_descent_pointer;
       }

       /// Destructor.

       virtual ~GradientDescentResults()
       {
       }

       /// Pointer to the gradient descent object for which the training results are to be stored.

      GradientDescent* gradient_descent_pointer;

      // Training history

      /// History of the neural network parameters over the training iterations.

      Vector< Vector<double> > parameters_history;

      /// History of the parameters norm over the training iterations.

      Vector<double> parameters_norm_history;

      /// History of the loss function loss over the training iterations.

      Vector<double> loss_history;

      /// History of the selection error over the training iterations.

      Vector<double> selection_error_history;

      /// History of the loss function gradient over the training iterations.

      Vector< Vector<double> > gradient_history;

      /// History of the gradient norm over the training iterations.

      Vector<double> gradient_norm_history;

      /// History of the random search training direction over the training iterations.

      Vector< Vector<double> >  training_direction_history;

      /// History of the random search training rate over the training iterations.

      Vector<double> training_rate_history;

      /// History of the elapsed time over the training iterations.

      Vector<double> elapsed_time_history;

      // Final values

      /// Final neural network parameters vector. 

      Vector<double> final_parameters;

      /// Final neural network parameters norm. 

      double final_parameters_norm;

      /// Final loss function evaluation.

      double final_loss;

      /// Final selection error.

      double final_selection_error;

      /// Final loss function gradient. 

      Vector<double> final_gradient;

      /// Final gradient norm. 

      double final_gradient_norm;

      /// Final gradient descent training direction. 

      Vector<double> final_training_direction;

      /// Final gradient descent training rate. 

      double final_training_rate;

      /// Elapsed time of the training process. 

      double elapsed_time;

      /// Maximum number of training iterations.

      size_t iterations_number;

      /// Stopping criterion

      string stopping_criterion;

      void resize_training_history(const size_t&);

      string object_to_string() const;

      Matrix<string> write_final_results(const int& precision = 3) const;
   };

   // METHODS

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm_pointer();

   // Training parameters

   const double& get_warning_parameters_norm() const;
   const double& get_warning_gradient_norm() const;
   const double& get_warning_training_rate() const;

   const double& get_error_parameters_norm() const;
   const double& get_error_gradient_norm() const;
   const double& get_error_training_rate() const;

   // Stopping criteria

   const double& get_minimum_parameters_increment_norm() const;

   const double& get_minimum_loss_increase() const;
   const double& get_loss_goal() const;
   const double& get_gradient_norm_goal() const;
   const size_t& get_maximum_selection_error_decreases() const;

   const size_t& get_maximum_epochs_number() const;
   const double& get_maximum_time() const;

   const bool& get_return_minimum_selection_error_neural_network() const;
   const bool& get_apply_early_stopping() const;

   // Reserve training history

   const bool& get_reserve_parameters_history() const;
   const bool& get_reserve_parameters_norm_history() const;

   const bool& get_reserve_error_history() const;
   const bool& get_reserve_gradient_history() const;
   const bool& get_reserve_gradient_norm_history() const;
   const bool& get_reserve_selection_error_history() const;

   const bool& get_reserve_training_direction_history() const;
   const bool& get_reserve_training_rate_history() const;
   const bool& get_reserve_elapsed_time_history() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_learning_rate_algorithm(const LearningRateAlgorithm&);


   void set_default();

   void set_reserve_all_training_history(const bool&);


   // Training parameters

   void set_warning_parameters_norm(const double&);
   void set_warning_gradient_norm(const double&);
   void set_warning_training_rate(const double&);

   void set_error_parameters_norm(const double&);
   void set_error_gradient_norm(const double&);
   void set_error_training_rate(const double&);

   void set_maximum_epochs_number(const size_t&);

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const double&);

   void set_minimum_loss_decrease(const double&);
   void set_loss_goal(const double&);
   void set_gradient_norm_goal(const double&);
   void set_maximum_selection_error_increases(const size_t&);

   void set_maximum_time(const double&);

   void set_return_minimum_selection_error_neural_network(const bool&);
   void set_apply_early_stopping(const bool&);

   // Reserve training history

   void set_reserve_parameters_history(const bool&);
   void set_reserve_parameters_norm_history(const bool&);

   void set_reserve_error_history(const bool&);
   void set_reserve_gradient_history(const bool&);
   void set_reserve_gradient_norm_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   void set_reserve_training_direction_history(const bool&);
   void set_reserve_training_rate_history(const bool&);
   void set_reserve_elapsed_time_history(const bool&);

   // Utilities

   void set_display_period(const size_t&);

   // Training methods

   Vector<double> calculate_training_direction(const Vector<double>&) const;

   GradientDescentResults* perform_training();

   void perform_training_void();

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Matrix<string> to_string_matrix() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   // TRAINING OPERATORS

   /// Training rate algorithm object for one-dimensional minimization. 

   LearningRateAlgorithm learning_rate_algorithm;

   // TRAINING PARAMETERS

   /// Value for the parameters norm at which a warning message is written to the screen. 

   double warning_parameters_norm;

   /// Value for the gradient norm at which a warning message is written to the screen. 

   double warning_gradient_norm;   

   /// Training rate value at wich a warning message is written to the screen.

   double warning_training_rate;

   /// Value for the parameters norm at which the training process is assumed to fail. 
   
   double error_parameters_norm;

   /// Value for the gradient norm at which the training process is assumed to fail. 

   double error_gradient_norm;

   /// Training rate at wich the line minimization algorithm is assumed to be unable to bracket a minimum.

   double error_training_rate;

   // STOPPING CRITERIA

   /// Norm of the parameters increment vector at which training stops.

   double minimum_parameters_increment_norm;

   /// Minimum loss improvement between two successive iterations. It is used as a stopping criterion.

   double minimum_loss_decrease;

   /// Goal value for the loss. It is used as a stopping criterion.

   double loss_goal;

   /// Goal value for the norm of the error function gradient. It is used as a stopping criterion.

   double gradient_norm_goal;

   /// Maximum number of iterations at which the selection error increases.
   /// This is an early stopping method for improving selection.

   size_t maximum_selection_error_decreases;

   /// Initial batch size

   size_t training_initial_batch_size;

   /// Maximum training batch size

   size_t training_maximum_batch_size;

   /// Maximum epochs number

   size_t maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   double maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool return_minimum_selection_error_neural_network;

   /// True if the selection error decrease stopping criteria has to be taken in account, false otherwise.

   bool apply_early_stopping;

   // TRAINING HISTORY

   /// True if the parameters history matrix is to be reserved, false otherwise.

   bool reserve_parameters_history;

   /// True if the parameters norm history vector is to be reserved, false otherwise.

   bool reserve_parameters_norm_history;

   /// True if the loss history vector is to be reserved, false otherwise.

   bool reserve_error_history;

   /// True if the gradient history matrix is to be reserved, false otherwise.

   bool reserve_gradient_history;

   /// True if the gradient norm history vector is to be reserved, false otherwise.

   bool reserve_gradient_norm_history;

   /// True if the training direction history matrix is to be reserved, false otherwise.
   
   bool reserve_training_direction_history;

   /// True if the training rate history vector is to be reserved, false otherwise.

   bool reserve_training_rate_history;

   /// True if the elapsed time history vector is to be reserved, false otherwise.

   bool reserve_elapsed_time_history;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history;
};

}

#endif
