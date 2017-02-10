/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R E G U L A R I Z A T I O N   T E R M   C L A S S   H E A D E R                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __REGULARIZATIONTERM_H__
#define __REGULARIZATIONTERM_H__

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
#include "mathematical_model.h"

#include "neural_network.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{
/// This class represents the concept of error term. 
/// A error term is a summand in the loss functional expression. 
/// Any derived class must implement the calculate_loss(void) method.

class RegularizationTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit RegularizationTerm(void);

   // NEURAL NETWORK CONSTRUCTOR

   explicit RegularizationTerm(NeuralNetwork*);

   // XML CONSTRUCTOR

   explicit RegularizationTerm(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   RegularizationTerm(const RegularizationTerm&);

   // DESTRUCTOR

   virtual ~RegularizationTerm(void);

   // ASSIGNMENT OPERATOR

   virtual RegularizationTerm& operator = (const RegularizationTerm&);

   // EQUAL TO OPERATOR

   virtual bool operator == (const RegularizationTerm&) const;

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

   inline NeuralNetwork* get_neural_network_pointer(void) const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!neural_network_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: RegularizationTerm class.\n"
                    << "NeuralNetwork* get_neural_network_pointer(void) const method.\n"
                    << "Neural network pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif

      return(neural_network_pointer);
   }


   /// Returns a pointer to the numerical differentiation object used in this error term object. 

   inline NumericalDifferentiation* get_numerical_differentiation_pointer(void) const
   {
        #ifdef __OPENNN_DEBUG__

        if(!numerical_differentiation_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: RegularizationTerm class.\n"
                    << "NumericalDifferentiation* get_numerical_differentiation_pointer(void) const method.\n"
                    << "Numerical differentiation pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif

      return(numerical_differentiation_pointer);
   }

   const bool& get_display(void) const;

   bool has_neural_network(void) const;
   bool has_numerical_differentiation(void) const;


   // Set methods

   virtual void set(void);
   virtual void set(NeuralNetwork*);

   void set(const RegularizationTerm&);

   virtual void set_neural_network_pointer(NeuralNetwork*);

   void set_numerical_differentiation_pointer(NumericalDifferentiation*);

   virtual void set_default(void);

   void set_display(const bool&);

   // Pointer methods

   void construct_numerical_differentiation(void);
   void delete_numerical_differentiation_pointer(void);

   // Checking methods

   virtual void check(void) const;

   // Objective methods

   /// Returns the loss value of the error term.

   virtual double calculate_regularization(void) const = 0;

   /// Returns the default loss of a error term for a given set of neural network parameters. 

   virtual double calculate_regularization(const Vector<double>&) const = 0;

   virtual Vector<double> calculate_gradient(void) const; 

   virtual Vector<double> calculate_gradient(const Vector<double>&) const;

   virtual Matrix<double> calculate_Hessian(void) const; 

   virtual Matrix<double> calculate_Hessian(const Vector<double>&) const;

   virtual std::string write_error_term_type(void) const;

   virtual std::string write_information(void) const;

   // Serialization methods

   virtual std::string to_string(void) const;

   virtual tinyxml2::XMLDocument* to_XML(void) const;   
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   //virtual void read_XML(   );

   size_t calculate_Kronecker_delta(const size_t&, const size_t&) const;

protected:

   /// Pointer to a neural network object.

   NeuralNetwork* neural_network_pointer;

   /// Numerical differentiation object.

   NumericalDifferentiation* numerical_differentiation_pointer;

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
