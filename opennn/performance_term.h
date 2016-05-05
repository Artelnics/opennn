/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R F O R M A N C E   T E R M   C L A S S   H E A D E R                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PERFORMANCETERM_H__
#define __PERFORMANCETERM_H__

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
/// This class represents the concept of performance term. 
/// A performance term is a summand in the performance functional expression. 
/// Any derived class must implement the calculate_performance(void) method.

class PerformanceTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit PerformanceTerm(void);

   // NEURAL NETWORK CONSTRUCTOR

   explicit PerformanceTerm(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit PerformanceTerm(DataSet*);

   // MATHEMATICAL MODEL CONSTRUCTOR

   explicit PerformanceTerm(MathematicalModel*);

   // NEURAL NETWORK AND DATA SET CONSTRUCTOR

   explicit PerformanceTerm(NeuralNetwork*, DataSet*);

   // NEURAL NETWORK AND MATHEMATICAL MODEL CONSTRUCTOR

   explicit PerformanceTerm(NeuralNetwork*, MathematicalModel*);

   // NEURAL NETWORK, MATHEMATICAL MODEL AND DATA SET CONSTRUCTOR

   explicit PerformanceTerm(NeuralNetwork*, MathematicalModel*, DataSet*);

   // XML CONSTRUCTOR

   explicit PerformanceTerm(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   PerformanceTerm(const PerformanceTerm&);

   // DESTRUCTOR

   virtual ~PerformanceTerm(void);

   // ASSIGNMENT OPERATOR

   virtual PerformanceTerm& operator = (const PerformanceTerm&);

   // EQUAL TO OPERATOR

   virtual bool operator == (const PerformanceTerm&) const;

   // STRUCTURES

   /// This structure contains the zero order performance quantities of a performance term. 
   /// This only includes the performance itself.

   struct ZerothOrderPerformance
   {
      /// Performance term evaluation.

      double performance;
   };


   /// This structure contains the first order performance quantities of a performance term. 
   /// This includes the performance itself and the gradient vector.

   struct FirstOrderPerformance
   {
      /// Performance term performance. 

      double performance;

      /// Performance term gradient vector. 

      Vector<double> gradient;
   };


   /// This structure contains the second order performance quantities of a performance term. 
   /// This includes the performance itself, the gradient vector and the Hessian matrix.

   struct SecondOrderPerformance
   {
      /// Peformance term performance. 

      double performance;

      /// Performance term gradient vector. 

      Vector<double> gradient;

	  /// Performance term Hessian matrix. 

      Matrix<double> Hessian;
   };


   ///
   /// This structure contains the zero order evaluation of the terms function.
   ///

   struct ZerothOrderTerms
   {
      /// Subterms performance vector.

      Vector<double> terms;
   };

   /// Set of subterms vector and subterms Jacobian matrix of the performance term. 
   /// A method returning this structure might be more efficient than calculating the objective terms and the terms Jacobian separately.

   struct FirstOrderTerms
   {
      /// Subterms performance vector. 

      Vector<double> terms;

      /// Subterms Jacobian matrix. 

      Matrix<double> Jacobian;
   };


   // METHODS

   // Get methods

   /// Returns a pointer to the neural network object associated to the performance term.

   inline NeuralNetwork* get_neural_network_pointer(void) const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!neural_network_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceTerm class.\n"
                    << "NeuralNetwork* get_neural_network_pointer(void) const method.\n"
                    << "Neural network pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif

      return(neural_network_pointer);
   }


   /// Returns a pointer to the mathematical model object associated to the performance term.

   inline MathematicalModel* get_mathemtaical_model_pointer(void) const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!mathematical_model_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceTerm class.\n"
                    << "MathematicalModel* get_mathematical_model_pointer(void) const method.\n"
                    << "MathematicalModel pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif


      return(mathematical_model_pointer);
   }


   /// Returns a pointer to the data set object associated to the performance term.

   inline DataSet* get_data_set_pointer(void) const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!data_set_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceTerm class.\n"
                    << "DataSet* get_data_set_pointer(void) const method.\n"
                    << "DataSet pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif

      return(data_set_pointer);
   }


   /// Returns a pointer to the numerical differentiation object used in this performance term object. 

   inline NumericalDifferentiation* get_numerical_differentiation_pointer(void) const
   {
        #ifdef __OPENNN_DEBUG__

        if(!numerical_differentiation_pointer)
        {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceTerm class.\n"
                    << "NumericalDifferentiation* get_numerical_differentiation_pointer(void) const method.\n"
                    << "Numerical differentiation pointer is NULL.\n";

             throw std::logic_error(buffer.str());
        }

        #endif

      return(numerical_differentiation_pointer);
   }

   const bool& get_display(void) const;

   bool has_neural_network(void) const;
   bool has_mathematical_model(void) const;
   bool has_data_set(void) const;
   bool has_numerical_differentiation(void) const;


   // Set methods

   virtual void set(void);
   virtual void set(NeuralNetwork*);
   virtual void set(DataSet*);
   virtual void set(MathematicalModel*);
   virtual void set(NeuralNetwork*, DataSet*);
   virtual void set(NeuralNetwork*, MathematicalModel*);
   virtual void set(NeuralNetwork*, MathematicalModel*, DataSet*);

   void set(const PerformanceTerm&);

   virtual void set_neural_network_pointer(NeuralNetwork*);

   virtual void set_mathematical_model_pointer(MathematicalModel*);
   virtual void set_data_set_pointer(DataSet*);

   void set_numerical_differentiation_pointer(NumericalDifferentiation*);

   virtual void set_default(void);

   void set_display(const bool&);

   // Pointer methods

   void construct_numerical_differentiation(void);
   void delete_numerical_differentiation_pointer(void);

   // Checking methods

   virtual void check(void) const;

   // Layers delta methods
   
   Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) const;
   Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) const;   

   // Interlayers Delta methods

   double calculate_performance_output_combinations(const Vector<double>& combinations) const;

   //Matrix< Matrix <double> > calculate_interlayers_Delta(void) const;

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

   /// Returns the performance value of the performance term.

   virtual double calculate_performance(void) const = 0;

   /// Returns the default performance of a performance term for a given set of neural network parameters. 

   virtual double calculate_performance(const Vector<double>&) const = 0;

   /// Returns an performance of the performance term for selection purposes.  

   virtual double calculate_selection_performance(void) const
   {
      return(0.0);
   }

   /// Returns the performance term gradient.

   virtual Vector<double> calculate_output_gradient(const Vector<double>&, const Vector<double>&) const
   {
        Vector<double> output_gradient;
        return(output_gradient);
   }

   virtual Vector<double> calculate_gradient(void) const; 

   virtual Vector<double> calculate_gradient(const Vector<double>&) const;

   /// Returns the performance term Hessian.

   virtual Matrix<double> calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const
   {
         Matrix<double> output_Hessian;
         return(output_Hessian);
   }

   virtual Matrix<double> calculate_Hessian(void) const; 

   virtual Matrix<double> calculate_Hessian(const Vector<double>&) const;

   virtual Vector<double> calculate_terms(void) const;
   virtual Vector<double> calculate_terms(const Vector<double>&) const;

   virtual Matrix<double> calculate_terms_Jacobian(void) const;

   virtual PerformanceTerm::FirstOrderTerms calculate_first_order_terms(void) const;

   virtual std::string write_performance_term_type(void) const;

   virtual std::string write_information(void) const;

   // Serialization methods

   virtual std::string to_string(void) const;

   virtual tinyxml2::XMLDocument* to_XML(void) const;   
   virtual void from_XML(const tinyxml2::XMLDocument&);

   size_t calculate_Kronecker_delta(const size_t&, const size_t&) const;

protected:

   /// Pointer to a multilayer perceptron object.

   NeuralNetwork* neural_network_pointer;

   /// Pointer to a data set object.

   DataSet* data_set_pointer;

   /// Pointer to a mathematical model object.

   MathematicalModel* mathematical_model_pointer;

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
