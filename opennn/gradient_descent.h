/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
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

#include "training_algorithm.h"
#include "training_rate_algorithm.h"


namespace OpenNN
{

/// This concrete class represents the gradient descent training algorithm for
/// a loss functional of a neural network.

class GradientDescent : public TrainingAlgorithm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit GradientDescent(void); 

   // PERFORMANCE FUNCTIONAL CONSTRUCTOR

   explicit GradientDescent(LossIndex*);

   // XML CONSTRUCTOR

   explicit GradientDescent(const tinyxml2::XMLDocument&); 


   // DESTRUCTOR

   virtual ~GradientDescent(void);

   // STRUCTURES

   ///
   /// This structure contains the training results for the gradient descent. 
   ///

   struct GradientDescentResults : public TrainingAlgorithm::TrainingAlgorithmResults
   {
       /// Default constructor.

       GradientDescentResults(void)
       {
           gradient_descent_pointer = NULL;
       }

       /// Gradient descent constructor.

       GradientDescentResults(GradientDescent* new_gradient_descent_pointer)
       {
           gradient_descent_pointer = new_gradient_descent_pointer;
       }

       /// Destructor.

       virtual ~GradientDescentResults(void)
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

      /// History of the selection loss over the training iterations.

      Vector<double> selection_loss_history;

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

      /// Final selection loss.

      double final_selection_loss;

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

      std::string stopping_criterion;

      void resize_training_history(const size_t&);

      std::string to_string(void) const;

      Matrix<std::string> write_final_results(const size_t& precision = 3) const;
   };

   // METHODS

   const TrainingRateAlgorithm& get_training_rate_algorithm(void) const;
   TrainingRateAlgorithm* get_training_rate_algorithm_pointer(void);

   // Training parameters

   const double& get_warning_parameters_norm(void) const;
   const double& get_warning_gradient_norm(void) const;
   const double& get_warning_training_rate(void) const;

   const double& get_error_parameters_norm(void) const;
   const double& get_error_gradient_norm(void) const;
   const double& get_error_training_rate(void) const;

   // Stopping criteria

   const double& get_minimum_parameters_increment_norm(void) const;

   const double& get_minimum_loss_increase(void) const;
   const double& get_loss_goal(void) const;
   const double& get_gradient_norm_goal(void) const;
   const size_t& get_maximum_selection_loss_decreases(void) const;

   const size_t& get_maximum_iterations_number(void) const;
   const double& get_maximum_time(void) const;

   const bool& get_return_minimum_selection_error_neural_network(void) const;

   // Reserve training history

   const bool& get_reserve_parameters_history(void) const;
   const bool& get_reserve_parameters_norm_history(void) const;

   const bool& get_reserve_loss_history(void) const;
   const bool& get_reserve_gradient_history(void) const;
   const bool& get_reserve_gradient_norm_history(void) const;
   const bool& get_reserve_selection_loss_history(void) const;

   const bool& get_reserve_training_direction_history(void) const;
   const bool& get_reserve_training_rate_history(void) const;
   const bool& get_reserve_elapsed_time_history(void) const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_training_rate_algorithm(const TrainingRateAlgorithm&);


   void set_default(void);

   void set_reserve_all_training_history(const bool&);


   // Training parameters

   void set_warning_parameters_norm(const double&);
   void set_warning_gradient_norm(const double&);
   void set_warning_training_rate(const double&);

   void set_error_parameters_norm(const double&);
   void set_error_gradient_norm(const double&);
   void set_error_training_rate(const double&);

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const double&);

   void set_minimum_loss_increase(const double&);
   void set_loss_goal(const double&);
   void set_gradient_norm_goal(const double&);
   void set_maximum_selection_loss_decreases(const size_t&);

   void set_maximum_iterations_number(const size_t&);
   void set_maximum_time(const double&);

   void set_return_minimum_selection_error_neural_network(const bool&);

   // Reserve training history

   void set_reserve_parameters_history(const bool&);
   void set_reserve_parameters_norm_history(const bool&);

   void set_reserve_loss_history(const bool&);
   void set_reserve_gradient_history(const bool&);
   void set_reserve_gradient_norm_history(const bool&);
   void set_reserve_selection_loss_history(const bool&);

   void set_reserve_training_direction_history(const bool&);
   void set_reserve_training_rate_history(const bool&);
   void set_reserve_elapsed_time_history(const bool&);

   // Utilities

   void set_display_period(const size_t&);

   // Training methods

   Vector<double> calculate_training_direction(const Vector<double>&) const;

   GradientDescentResults* perform_training(void);

   std::string write_training_algorithm_type(void) const;

   // Serialization methods

   Matrix<std::string> to_string_matrix(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

private:

   // TRAINING OPERATORS

   /// Training rate algorithm object for one-dimensional minimization. 

   TrainingRateAlgorithm training_rate_algorithm;

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

   double minimum_loss_increase;

   /// Goal value for the loss. It is used as a stopping criterion.

   double loss_goal;

   /// Goal value for the norm of the objective function gradient. It is used as a stopping criterion.

   double gradient_norm_goal;

   /// Maximum number of iterations at which the selection loss increases.
   /// This is an early stopping method for improving selection.

   size_t maximum_selection_loss_decreases;

   /// Maximum number of iterations to perform_training. It is used as a stopping criterion.

   size_t maximum_iterations_number;

   /// Maximum training time. It is used as a stopping criterion.

   double maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool return_minimum_selection_error_neural_network;

   // TRAINING HISTORY

   /// True if the parameters history matrix is to be reserved, false otherwise.

   bool reserve_parameters_history;

   /// True if the parameters norm history vector is to be reserved, false otherwise.

   bool reserve_parameters_norm_history;

   /// True if the loss history vector is to be reserved, false otherwise.

   bool reserve_loss_history;

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

   /// True if the selection loss history vector is to be reserved, false otherwise.

   bool reserve_selection_loss_history;
};

}

#endif
