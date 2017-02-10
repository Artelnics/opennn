/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R F O R M A N C E   F U N C T I O N A L   C L A S S   H E A D E R                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PERFORMANCEFUNCTIONAL_H__
#define __PERFORMANCEFUNCTIONAL_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#ifdef __OPENNN_MPI__
#include <mpi.h>
#endif
// OpenNN includes

#include "vector.h"
#include "matrix.h"
#include "numerical_differentiation.h"

#include "data_set.h"
#include "mathematical_model.h"

#include "neural_network.h"
#include "error_term.h"
#include "regularization_term.h"

#include "sum_squared_error.h"
#include "mean_squared_error.h"
#include "root_mean_squared_error.h"
#include "normalized_squared_error.h"
#include "weighted_squared_error.h"
#include "roc_area_error.h"
#include "minkowski_error.h"
#include "cross_entropy_error.h"
#include "outputs_integrals.h"

#include "neural_parameters_norm.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This abstract class represents the concept of loss functional for a neural network. 
/// A loss functional is composed of two terms: An error term and a regularization term.
/// Any derived class must implement the calculate_loss(void) method.

class LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit LossIndex(void);

   // OBJECTIVE FUNCTIONAL CONSTRUCTOR

   explicit LossIndex(ErrorTerm*);

   // NEURAL NETWORK CONSTRUCTOR

   explicit LossIndex(NeuralNetwork*);

   // NEURAL NETWORK AND DATA SET CONSTRUCTOR

   explicit LossIndex(NeuralNetwork*, DataSet*);

   // NEURAL NETWORK AND MATHEMATICAL MODEL CONSTRUCTOR

   explicit LossIndex(NeuralNetwork*, MathematicalModel*);

   // NEURAL NETWORK, MATHEMATICAL MODEL AND DATA SET CONSTRUCTOR

   explicit LossIndex(NeuralNetwork*, MathematicalModel*, DataSet*);

   // FILE CONSTRUCTOR

   explicit LossIndex(const std::string&);

   // XML CONSTRUCTOR

   explicit LossIndex(const tinyxml2::XMLDocument&);


   // COPY CONSTRUCTOR

   LossIndex(const LossIndex&);

   // DESTRUCTOR

   virtual ~LossIndex(void);

   // STRUCTURES 

   /// Performance value of the peformance function.
   /// This is a very simple structure with just one value. 

   struct ZeroOrderloss
   {
      /// Performance value.

      double loss;
   };

   /// Set of loss value and gradient vector of the peformance function. 
   /// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

   struct FirstOrderloss
   {
      /// Performance value.

      double loss;

      /// Performance function gradient vector. 

      Vector<double> gradient;
   };

   /// Set of loss value, gradient vector and Hessian matrix of the peformance function. 
   /// A method returning this structure might be implemented more efficiently than the loss, gradient and Hessian methods separately.

   struct SecondOrderloss
   {
      /// Performance value.

      double loss;

      /// Performance function gradient vector. 

	  Vector<double> gradient;

      /// Performance function Hessian matrix. 

	  Matrix<double> Hessian;
   };


   // ENUMERATIONS

   /// Enumeration of available objective types in OpenNN.

   enum ErrorType
   {
      NO_ERROR,
      SUM_SQUARED_ERROR,
      MEAN_SQUARED_ERROR,
      ROOT_MEAN_SQUARED_ERROR,
      NORMALIZED_SQUARED_ERROR,
      MINKOWSKI_ERROR,
      WEIGHTED_SQUARED_ERROR,
      ROC_AREA_ERROR,
      CROSS_ENTROPY_ERROR,
      USER_ERROR
   };

   /// Enumeration of available regularization types in OpenNN.

   enum RegularizationType
   {
      NO_REGULARIZATION,
      NEURAL_PARAMETERS_NORM,
      OUTPUTS_INTEGRALS,
      USER_REGULARIZATION
   };


   // METHODS

   // Check methods

   void check_neural_network(void) const;

   void check_error_terms(void) const;

   // Get methods

   /// Returns a pointer to the neural network associated to the loss functional.

   inline NeuralNetwork* get_neural_network_pointer(void) const 
   {
      #ifdef __OPENNN_DEBUG__

      if(!neural_network_pointer)
      {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: LossIndex class.\n"
                  << "NeuralNetwork* get_neural_network_pointer(void) const method.\n"
                  << "Neural network pointer is NULL.\n";

           throw std::logic_error(buffer.str());
      }

      #endif

      return(neural_network_pointer);
   }

   /// Returns a pointer to the mathematical model associated to the loss functional.

   inline MathematicalModel* get_mathematical_model_pointer(void) const
   {
        #ifdef __OPENNN_DEBUG__

        if(!mathematical_model_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "MathematicalModel* get_mathematical_model_pointer(void) const method.\n"
                    << "MathematicalModel pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif

      return(mathematical_model_pointer);
   }

   /// Returns a pointer to the data set associated to the loss functional.

   inline DataSet* get_data_set_pointer(void) const
   {
        #ifdef __OPENNN_DEBUG__

        if(!data_set_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "DataSet* get_data_set_pointer(void) const method.\n"
                    << "DataSet pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif

      return(data_set_pointer);
   }

   bool has_neural_network(void) const;
   bool has_mathematical_model(void) const;
   bool has_data_set(void) const;

   bool has_selection(void) const;

   bool is_sum_squared_terms(void) const;

   // Objective terms

   SumSquaredError* get_sum_squared_error_pointer(void) const;
   MeanSquaredError* get_mean_squared_error_pointer(void) const;
   RootMeanSquaredError* get_root_mean_squared_error_pointer(void) const;
   NormalizedSquaredError* get_normalized_squared_error_pointer(void) const;
   MinkowskiError* get_Minkowski_error_pointer(void) const;
   CrossEntropyError* get_cross_entropy_error_pointer(void) const;
   WeightedSquaredError* get_weighted_squared_error_pointer(void) const;
   RocAreaError* get_roc_area_error_pointer(void) const;
   ErrorTerm* get_user_error_pointer(void) const;

   // Regularization terms

   NeuralParametersNorm* get_neural_parameters_norm_pointer(void) const;
   OutputsIntegrals* get_outputs_integrals_pointer(void) const;
   RegularizationTerm* get_user_regularization_pointer(void) const;

   // Functional type methods

   const ErrorType& get_error_type(void) const;
   const RegularizationType& get_regularization_type(void) const;

   std::string write_error_type(void) const;
   std::string write_regularization_type(void) const;

   std::string write_error_type_text(void) const;
   std::string write_regularization_type_text(void) const;

   // Serialization methods

   const bool& get_display(void) const;

   // Set methods

   void set_neural_network_pointer(NeuralNetwork*);

   void set_mathematical_model_pointer(MathematicalModel*);
   void set_data_set_pointer(DataSet*);

   void set_user_error_pointer(ErrorTerm*);
   void set_user_regularization_pointer(RegularizationTerm*);

   void set_default(void);

#ifdef __OPENNN_MPI__
   void set_MPI(DataSet*, NeuralNetwork*, const LossIndex*);
#endif

   // Functionals methods 

   void set_error_type(const ErrorType&);
   void set_regularization_type(const RegularizationType&);

   void set_error_type(const std::string&);
   void set_regularization_type(const std::string&);
   
   void destruct_error_term(void);
   void destruct_regularization_term(void);

   void destruct_all_terms(void);

   void set_display(const bool&);

   // Loss index methods

   double calculate_error(void) const;
   double calculate_regularization(void) const;

#ifdef __OPENNN_MPI__
   double calculate_error_MPI(void) const;
#endif

   double calculate_error(const Vector<double>&) const;
   double calculate_regularization(const Vector<double>&) const;

#ifdef __OPENNN_MPI__
   double calculate_error_MPI(const Vector<double>&) const;
#endif

   Vector<double> calculate_error_terms(void) const;
   Matrix<double> calculate_error_terms_Jacobian(void) const;

   Vector<double> calculate_error_gradient(void) const;
   Vector<double> calculate_regularization_gradient(void) const;

#ifdef __OPENNN_MPI__
   Vector<double> calculate_error_gradient_MPI(void) const;
#endif

   Vector<double> calculate_error_gradient(const Vector<double>&) const;
   Vector<double> calculate_regularization_gradient(const Vector<double>&) const;

#ifdef __OPENNN_MPI__
   Vector<double> calculate_error_gradient_MPI(const Vector<double>&) const;
#endif

   Matrix<double> calculate_error_Hessian(void) const;
   Matrix<double> calculate_regularization_Hessian(void) const;

   Matrix<double> calculate_error_Hessian(const Vector<double>&) const;
   Matrix<double> calculate_regularization_Hessian(const Vector<double>&) const;

   double calculate_loss(void) const;
   Vector<double> calculate_gradient(void) const;
   Matrix<double> calculate_Hessian(void) const;

   double calculate_loss(const Vector<double>&) const;
   Vector<double> calculate_gradient(const Vector<double>&) const;
   Matrix<double> calculate_Hessian(const Vector<double>&) const;

   virtual Matrix<double> calculate_inverse_Hessian(void) const;

   virtual Vector<double> calculate_vector_dot_Hessian(const Vector<double>&) const;

   Vector<double> calculate_terms(void) const;
   Matrix<double> calculate_terms_Jacobian(void) const;

   virtual ZeroOrderloss calculate_zero_order_loss(void) const;
   virtual FirstOrderloss calculate_first_order_loss(void) const;
   virtual SecondOrderloss calculate_second_order_loss(void) const;

   double calculate_selection_error(void) const;
   //double calculate_selection_regularization(void) const;
   //double calculate_selection_constraints(void) const;

#ifdef __OPENNN_MPI__
   double calculate_selection_error_MPI(void) const;
#endif

   virtual double calculate_selection_loss(void) const;

   // Taylor approximation methods

   double calculate_zero_order_Taylor_approximation(const Vector<double>&) const;
   double calculate_first_order_Taylor_approximation(const Vector<double>&) const;
   double calculate_second_order_Taylor_approximation(const Vector<double>&) const;

   // Directional loss

   double calculate_loss(const Vector<double>&, const double&) const;
   double calculate_loss_derivative(const Vector<double>&, const double&) const;
   double calculate_loss_second_derivative(const Vector<double>&, const double&) const;

   // Serialization methods

   virtual tinyxml2::XMLDocument* to_XML(void) const;   
   virtual void from_XML(const tinyxml2::XMLDocument&);
   
   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   // virtual void read_XML(   );

   virtual std::string to_string(void) const;
   virtual void save(const std::string&) const;   
   virtual void load(const std::string&);   

   virtual std::string write_information(void);   

   void print(void) const;


private:

   /// Pointer to a neural network object.

   NeuralNetwork* neural_network_pointer;

   /// Pointer to a data set object.

   DataSet* data_set_pointer;

   /// Pointer to a mathematical model object.

   MathematicalModel* mathematical_model_pointer;

   /// Type of error term.

   ErrorType error_type;

   /// Type of regularization term.

   RegularizationType regularization_type;

   // Error terms

   /// Pointer to the sum squared error object wich can be used as the error term.

   SumSquaredError* sum_squared_error_pointer;

   /// Pointer to the mean squared error object wich can be used as the error term.

   MeanSquaredError* mean_squared_error_pointer;

   /// Pointer to the root mean squared error object wich can be used as the error term.

   RootMeanSquaredError* root_mean_squared_error_pointer;

   /// Pointer to the normalized squared error object wich can be used as the error term.

   NormalizedSquaredError* normalized_squared_error_pointer;

   /// Pointer to the Mikowski error object wich can be used as the error term.

   MinkowskiError* Minkowski_error_pointer;

   /// Pointer to the cross entropy error object wich can be used as the error term.

   CrossEntropyError* cross_entropy_error_pointer;

   /// Pointer to the weighted squared error object wich can be used as the error term.

   WeightedSquaredError* weighted_squared_error_pointer;

   /// Pointer to the ROC area error object wich can be used as the error term.

   RocAreaError* roc_area_error_pointer;

   /// Pointer to the user error term object wich can be used as objective.

   ErrorTerm* user_error_pointer;

   // Regularization terms

   /// Pointer to the neural parameters norm object wich can be used as the regularization term.

   NeuralParametersNorm* neural_parameters_norm_pointer;

   /// Pointer to the sum outputs integrals object wich can be used as the regularization term.

   OutputsIntegrals* outputs_integrals_pointer;

   /// Pointer to a user error term to be used for regularization.

   RegularizationTerm* user_regularization_pointer;

   // Sparsity terms

   //SparsityTerm

   /// Display messages to screen. 

   bool display;

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
