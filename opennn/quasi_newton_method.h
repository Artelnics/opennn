/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R                                           */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __QUASINEWTONMETHOD_H__
#define __QUASINEWTONMETHOD_H__

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

#ifdef __OPENNN_CUDA__
#include <cublas_v2.h>
#endif

// OpenNN includes

#include "loss_index.h"

#include "training_algorithm.h"
#include "training_rate_algorithm.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a quasi-Newton training algorithm for a loss functional of a neural network.
///

class QuasiNewtonMethod : public TrainingAlgorithm
{

public:

   // ENUMERATIONS

   /// Enumeration of the available training operators for obtaining the approximation to the inverse Hessian.

   enum InverseHessianApproximationMethod{DFP, BFGS};


   // DEFAULT CONSTRUCTOR

   explicit QuasiNewtonMethod(void);

   // PERFORMANCE FUNCTIONAL CONSTRUCTOR

   explicit QuasiNewtonMethod(LossIndex*);

   // XML CONSTRUCTOR

   explicit QuasiNewtonMethod(const tinyxml2::XMLDocument&);


   // DESTRUCTOR

   virtual ~QuasiNewtonMethod(void);

   // STRUCTURES

   ///
   /// This structure contains the training results for the quasi-Newton method. 
   ///

   struct QuasiNewtonMethodResults : public TrainingAlgorithm::TrainingAlgorithmResults
   {
       /// Default constructor.

       explicit QuasiNewtonMethodResults(void) : TrainingAlgorithm::TrainingAlgorithmResults()
       {
           quasi_Newton_method_pointer = NULL;
       }

       /// Quasi-Newton method constructor.

       explicit QuasiNewtonMethodResults(QuasiNewtonMethod* new_quasi_Newton_method_pointer) : TrainingAlgorithm::TrainingAlgorithmResults()
       {
           quasi_Newton_method_pointer = new_quasi_Newton_method_pointer;
       }

       /// Destructor.

       virtual ~QuasiNewtonMethodResults(void)
       {
           quasi_Newton_method_pointer = NULL;
       }

       // Members

       /// Pointer to the quasi-Newton method object for which the training results are to be stored.

      QuasiNewtonMethod* quasi_Newton_method_pointer;

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

      /// History of the inverse Hessian approximation over the training iterations. 

      Vector< Matrix<double> > inverse_Hessian_history;

      /// History of the random search training direction over the training iterations. 

      Vector< Vector<double> > training_direction_history;

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

      /// Final approximation of the inverse Hessian. 

      Matrix<double> final_inverse_Hessian_approximation;

      /// Final quasi-Newton method training direction. 

      Vector<double> final_training_direction;

      /// Final quasi-Newton method training rate. 

      double final_training_rate;

      /// Elapsed time of the training process. 

      double elapsed_time;

      /// Maximum number of training iterations.

      size_t iterations_number;

      // Methods

      QuasiNewtonMethod* get_quasi_Newton_method_pointer(void) const;

      void set_quasi_Newton_method_pointer(QuasiNewtonMethod*);

      void resize_training_history(const size_t&);

      std::string to_string(void) const;

      Matrix<std::string> write_final_results(const size_t& precision = 3) const;
   };

   // METHODS

   // Get methods

   const TrainingRateAlgorithm& get_training_rate_algorithm(void) const;
   TrainingRateAlgorithm* get_training_rate_algorithm_pointer(void);

   const InverseHessianApproximationMethod& get_inverse_Hessian_approximation_method(void) const;
   std::string write_inverse_Hessian_approximation_method(void) const;

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
   const bool& get_reserve_inverse_Hessian_history(void) const;
   const bool& get_reserve_selection_loss_history(void) const;

   const bool& get_reserve_training_direction_history(void) const;
   const bool& get_reserve_training_rate_history(void) const;
   const bool& get_reserve_elapsed_time_history(void) const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_inverse_Hessian_approximation_method(const InverseHessianApproximationMethod&);
   void set_inverse_Hessian_approximation_method(const std::string&);

   void set_default(void);

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
   void set_reserve_inverse_Hessian_history(const bool&);
   void set_reserve_selection_loss_history(const bool&);

   void set_reserve_training_direction_history(const bool&);
   void set_reserve_training_rate_history(const bool&);
   void set_reserve_elapsed_time_history(const bool&);

   // Utilities

   void set_display_period(const size_t&);

   // Training methods

   Vector<double> calculate_gradient_descent_training_direction(const Vector<double>&) const;

   Matrix<double> calculate_DFP_inverse_Hessian
   (const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) const;

   Matrix<double> calculate_BFGS_inverse_Hessian
   (const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) const;

   Matrix<double> calculate_inverse_Hessian_approximation(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) const;

#ifdef __OPENNN_CUDA__
   Matrix<double> calculate_DFP_inverse_Hessian_CUDA(double*, double*, double*, double*, double*, double*) const;
   Matrix<double> calculate_BFGS_inverse_Hessian_CUDA(double*, double*, double*, double*, double*, double*) const;
   Matrix<double> calculate_inverse_Hessian_approximation_CUDA(double*, double*, double*, double*, double*, double*) const;
#endif

   Vector<double> calculate_training_direction(const Vector<double>&, const Matrix<double>&) const;

   QuasiNewtonMethodResults* perform_training(void);

   // Training history methods

   void set_reserve_all_training_history(const bool&);

   std::string write_training_algorithm_type(void) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );


   std::string to_string(void) const;
   Matrix<std::string> to_string_matrix(void) const;


private: 

   /// Training rate algorithm object. 
   /// It is used to calculate the step for the quasi-Newton training direction.

   TrainingRateAlgorithm training_rate_algorithm;

   /// Variable containing the actual method used to obtain a suitable training rate. 

   InverseHessianApproximationMethod inverse_Hessian_approximation_method;


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

   /// True if the inverse Hessian history vector of matrices is to be reserved, false otherwise.

   bool reserve_inverse_Hessian_history;

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


// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
