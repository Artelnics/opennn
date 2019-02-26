/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R                        */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __LEVENBERGMARQUARDTALGORITHM_H__
#define __LEVENBERGMARQUARDTALGORITHM_H__

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

#include "optimization_algorithm.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This concrete class represents a Levenberg-Marquardt Algorithm training
/// algorithm for the sum squared error loss index for a multilayer perceptron.

class LevenbergMarquardtAlgorithm : public OptimizationAlgorithm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit LevenbergMarquardtAlgorithm();

    // LOSS INDEX CONSTRUCTOR

   explicit LevenbergMarquardtAlgorithm(LossIndex*);

   // XML CONSTRUCTOR

   explicit LevenbergMarquardtAlgorithm(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~LevenbergMarquardtAlgorithm();

   // STRUCTURES

   ///
   /// This structure contains the training results for the Levenberg-Marquardt algorithm. 
   ///

   struct LevenbergMarquardtAlgorithmResults : public OptimizationAlgorithm::OptimizationAlgorithmResults
   {
       /// Default constructor.

       LevenbergMarquardtAlgorithmResults()
       {
           Levenberg_Marquardt_algorithm_pointer = nullptr;
       }

       /// Random search constructor.

       LevenbergMarquardtAlgorithmResults(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_algorithm_pointer)
       {
           Levenberg_Marquardt_algorithm_pointer = new_Levenberg_Marquardt_algorithm_pointer;
       }

       /// Destructor.

       virtual ~LevenbergMarquardtAlgorithmResults()
       {
       }

       /// Pointer to the Levenberg-Marquardt algorithm object for which the training results are to be stored.

      LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm_pointer;

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

      /// History of the Hessian approximation over the training iterations. 

      Vector< Matrix<double> > Hessian_approximation_history;

      /// History of the damping parameter over the training iterations. 

      Vector<double> damping_parameter_history;

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

      /// Elapsed time of the training process. 

      double elapsed_time;

      /// Maximum number of training iterations.

      size_t iterations_number;

      void resize_training_history(const size_t&);
      string object_to_string() const;

      Matrix<string> write_final_results(const int& precision = 3) const;

   };

   // METHODS

   // Get methods

   // Training parameters

   const double& get_warning_parameters_norm() const;
   const double& get_warning_gradient_norm() const;

   const double& get_error_parameters_norm() const;
   const double& get_error_gradient_norm() const;

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
   const bool& get_reserve_Hessian_approximation_history() const;
   const bool& get_reserve_selection_error_history() const;

   const bool& get_reserve_elapsed_time_history() const;

   // Utilities

   const double& get_damping_parameter() const;

   const double& get_damping_parameter_factor() const;

   const double& get_minimum_damping_parameter() const;
   const double& get_maximum_damping_parameter() const;

   const bool& get_reserve_damping_parameter_history() const;

   const Vector<double>& get_damping_parameter_history() const;

   // Set methods

   void set_default();

   void set_damping_parameter(const double&);

   void set_damping_parameter_factor(const double&);

   void set_minimum_damping_parameter(const double&);
   void set_maximum_damping_parameter(const double&);

   void set_reserve_damping_parameter_history(const bool&);

   // Training parameters

   void set_warning_parameters_norm(const double&);
   void set_warning_gradient_norm(const double&);

   void set_error_parameters_norm(const double&);
   void set_error_gradient_norm(const double&);

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const double&);

   void set_minimum_loss_decrease(const double&);
   void set_loss_goal(const double&);
   void set_gradient_norm_goal(const double&);
   void set_maximum_selection_error_increases(const size_t&);

   void set_maximum_epochs_number(const size_t&);
   void set_maximum_time(const double&);

   void set_return_minimum_selection_error_neural_network(const bool&);
   void set_apply_early_stopping(const bool&);

   // Reserve training history

   void set_reserve_parameters_history(const bool&);
   void set_reserve_parameters_norm_history(const bool&);

   void set_reserve_error_history(const bool&);
   void set_reserve_gradient_history(const bool&);
   void set_reserve_gradient_norm_history(const bool&);
   void set_reserve_Hessian_approximation_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   void set_reserve_elapsed_time_history(const bool&);

   /// Makes the training history of all variables to be reseved or not in memory.

   virtual void set_reserve_all_training_history(const bool&);

   // Utilities

   void set_display_period(const size_t&);

   // Training methods

   void check() const;

   LevenbergMarquardtAlgorithmResults* perform_training();

   void perform_training_void();

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Matrix<string> to_string_matrix() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

   Vector<double> perform_Householder_QR_decomposition(const Matrix<double>&, const Vector<double>&) const;

private:

   // MEMBERS

   /// Initial Levenberg-Marquardt parameter.

   double damping_parameter;

   /// Minimum Levenberg-Marquardt parameter.

   double minimum_damping_parameter;

   /// Maximum Levenberg-Marquardt parameter.

   double maximum_damping_parameter;

   /// Damping parameter increase/decrease factor.

   double damping_parameter_factor;

   /// True if the damping parameter history vector is to be reserved, false otherwise.  

   bool reserve_damping_parameter_history;

   /// Vector containing the damping parameter history over the training iterations.

   Vector<double> damping_parameter_history;


   /// Value for the parameters norm at which a warning message is written to the screen. 

   double warning_parameters_norm;

   /// Value for the gradient norm at which a warning message is written to the screen. 

   double warning_gradient_norm;   

   /// Value for the parameters norm at which the training process is assumed to fail. 
   
   double error_parameters_norm;

   /// Value for the gradient norm at which the training process is assumed to fail. 

   double error_gradient_norm;


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

   /// Maximum number of epoch to perform_training. It is used as a stopping criterion.

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

   /// True if the Hessian history vector of matrices is to be reserved, false otherwise.

   bool reserve_Hessian_approximation_history;

   /// True if the elapsed time history vector is to be reserved, false otherwise.

   bool reserve_elapsed_time_history;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
