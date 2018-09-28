/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E R R O R   T E R M   C L A S S   H E A D E R                                                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __ERRORTERM_H__
#define __ERRORTERM_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

// OpenNN includes

#include "vector.h"
#include "matrix.h"
#include "numerical_differentiation.h"

#include "data_set.h"

#include "neural_network.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{
/// This class represents the concept of error term. 
/// A error term is a summand in the loss functional expression. 
/// Any derived class must implement the calculate_loss() method.

class ErrorTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit ErrorTerm();

   // NEURAL NETWORK CONSTRUCTOR

   explicit ErrorTerm(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit ErrorTerm(DataSet*);

   // NEURAL NETWORK AND DATA SET CONSTRUCTOR

   explicit ErrorTerm(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit ErrorTerm(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   ErrorTerm(const ErrorTerm&);

   // DESTRUCTOR

   virtual ~ErrorTerm();

   // ASSIGNMENT OPERATOR

   virtual ErrorTerm& operator =(const ErrorTerm&);

   // EQUAL TO OPERATOR

   virtual bool operator ==(const ErrorTerm&) const;

   // STRUCTURES

   /// This structure contains the zero order loss quantities of a error term. 
   /// This only includes the loss itself.

   struct ZerothOrderPerformance
   {
      /// Error term evaluation.

      double loss;
   };


   /// This structure contains the first order loss quantities of a error term. 
   /// This includes the loss itself and the gradient vector.

   struct FirstOrderPerformance
   {
      /// Error term loss. 

      double loss;

      /// Error term gradient vector. 

      Vector<double> gradient;
   };


   /// This structure contains the second order loss quantities of a error term. 
   /// This includes the loss itself, the gradient vector and the Hessian matrix.

   struct SecondOrderPerformance
   {
      /// Peformance term loss. 

      double loss;

      /// Error term gradient vector. 

      Vector<double> gradient;

	  /// Error term Hessian matrix. 

      Matrix<double> Hessian;
   };


   ///
   /// This structure contains the zero order evaluation of the terms function.
   ///

   struct ZerothOrderTerms
   {
      /// Subterms loss vector.

      Vector<double> terms;
   };

   /// Set of subterms vector and subterms Jacobian matrix of the error term. 
   /// A method returning this structure might be more efficient than calculating the error terms and the terms Jacobian separately.

   struct FirstOrderTerms
   {
      /// Subterms loss vector. 

      Vector<double> terms;

      /// Subterms Jacobian matrix. 

      Matrix<double> Jacobian;
   };


   // METHODS

   // Get methods

   /// Returns a pointer to the neural network object associated to the error term.

   inline NeuralNetwork* get_neural_network_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!neural_network_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: ErrorTerm class.\n"
                    << "NeuralNetwork* get_neural_network_pointer() const method.\n"
                    << "Neural network pointer is NULL.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return(neural_network_pointer);
   }


   /// Returns a pointer to the data set object associated to the error term.

   inline DataSet* get_data_set_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!data_set_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: ErrorTerm class.\n"
                    << "DataSet* get_data_set_pointer() const method.\n"
                    << "DataSet pointer is NULL.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return(data_set_pointer);
   }


   /// Returns a pointer to the numerical differentiation object used in this error term object. 

   inline NumericalDifferentiation* get_numerical_differentiation_pointer() const
   {
        #ifdef __OPENNN_DEBUG__

        if(!numerical_differentiation_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: ErrorTerm class.\n"
                    << "NumericalDifferentiation* get_numerical_differentiation_pointer() const method.\n"
                    << "Numerical differentiation pointer is NULL.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return(numerical_differentiation_pointer);
   }

   const bool& get_display() const;

   bool has_neural_network() const;
   bool has_data_set() const;
   bool has_numerical_differentiation() const;


   // Set methods

   virtual void set();
   virtual void set(NeuralNetwork*);
   virtual void set(DataSet*);
   virtual void set(NeuralNetwork*, DataSet*);

   void set(const ErrorTerm&);

   virtual void set_neural_network_pointer(NeuralNetwork*);

   virtual void set_data_set_pointer(DataSet*);

   void set_numerical_differentiation_pointer(NumericalDifferentiation*);

   virtual void set_default();

   void set_display(const bool&);

   // Pointer methods

   void construct_numerical_differentiation();
   void delete_numerical_differentiation_pointer();

   // Checking methods

   virtual void check() const;

   // Layers delta methods
   
   Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) const;
   Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) const;   

   // Interlayers Delta methods

   double calculate_loss_output_combinations(const Vector<double>& combinations) const;

   //Matrix< Matrix <double> > calculate_interlayers_Delta() const;

   Matrix<double> calculate_output_interlayers_Delta(const Vector<double>&, const Vector<double>&, const Vector<double>&, const Matrix<double>&) const;
   Matrix<double> calculate_interlayers_Delta(const size_t& ,const size_t& , const Vector<double>& , const Vector<double>&, const Vector<double>& , const Vector<double>&, const Vector< Vector<double> >&, const Vector<double>& , const Matrix<double>&, const Matrix<double>&, const Vector< Vector<double> >&) const;

   Matrix< Matrix <double> > calculate_interlayers_Delta(const Vector< Vector<double> >&, const Vector< Vector<double> >&, const Matrix< Matrix<double> >&, const Vector<double>&, const Matrix<double>&, const Vector< Vector<double> >&) const;

   // Point objective function methods

   Vector<double> calculate_point_gradient(const Vector<double>&, const Vector< Vector<double> >&, const Vector< Vector<double> >&) const;
   Vector<double> calculate_point_gradient(const Vector< Matrix<double> >&, const Vector< Vector<double> >&) const;

   Matrix<double> calculate_point_Hessian(const Vector< Vector<double> >&, const Vector< Vector< Vector<double> > >&, const Matrix< Matrix<double> >&, const Vector< Vector<double> >&, const Matrix< Matrix<double> >&) const;
   Matrix<double> calculate_single_hidden_layer_point_Hessian(const Vector< Vector<double> >&,
                                                              const Vector< Vector<double> >&,
                                                              const Vector< Vector< Vector<double> > >&,
                                                              const Vector< Vector<double> >&,
                                                              const Matrix<double>&) const;

   // Objective methods

   /// Returns the loss value of the error term.

   virtual double calculate_error() const = 0;

   /// Returns the default loss of a error term for a given set of neural network parameters. 

   virtual double calculate_error(const Vector<double>&) const = 0;

   /// Returns an loss of the error term for selection purposes.  

   virtual double calculate_selection_error() const
   {
      return(0.0);
   }

   /// Returns the error term gradient.

   virtual Vector<double> calculate_output_gradient(const Vector<double>&, const Vector<double>&) const
   {
        Vector<double> output_gradient;

        return(output_gradient);
   }

   virtual Vector<double> calculate_gradient() const;
   virtual Vector<double> calculate_gradient(const double&, const double&) const;

   virtual Vector<double> calculate_gradient(const Vector<double>&) const;

   /// Returns the error term Hessian.

   virtual Matrix<double> calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const
   {
         Matrix<double> output_Hessian;

         return(output_Hessian);
   }

   virtual Matrix<double> calculate_Hessian() const; 
   virtual Matrix<double> calculate_Hessian(const Vector<double>&) const;

   virtual Matrix<double> calculate_Hessian_one_layer() const;
   virtual Matrix<double> calculate_Hessian_two_layers() const;

   virtual Vector<double> calculate_terms() const;
   virtual Vector<double> calculate_terms(const Vector<double>&) const;

   virtual Matrix<double> calculate_terms_Jacobian() const;

   virtual ErrorTerm::FirstOrderTerms calculate_first_order_terms() const;

   virtual string write_error_term_type() const;

   virtual string write_information() const;

   // Serialization methods

   virtual string object_to_string() const;

   virtual tinyxml2::XMLDocument* to_XML() const;   
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   //virtual void read_XML(   );

   size_t calculate_Kronecker_delta(const size_t&, const size_t&) const;

protected:

   /// Pointer to a multilayer perceptron object.

   NeuralNetwork* neural_network_pointer;

   /// Pointer to a data set object.

   DataSet* data_set_pointer;

   /// Numerical differentiation object.

   NumericalDifferentiation* numerical_differentiation_pointer;

   /// Display messages to screen. 

   bool display;  
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
