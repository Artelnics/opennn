/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R O B A B I L I S T I C   L A Y E R   C L A S S   H E A D E R                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PROBABILISTICLAYER_H__
#define __PROBABILISTICLAYER_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This class represents a layer of probabilistic neurons.
/// The neural network defined in OpenNN includes a probabilistic layer for those problems 
/// when the outptus are to be interpreted as probabilities. 

class ProbabilisticLayer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit ProbabilisticLayer(void);

   // PROBABILISTIC NEURONS NUMBER CONSTRUCTOR

   explicit ProbabilisticLayer(const size_t&);

   // COPY CONSTRUCTOR

   ProbabilisticLayer(const ProbabilisticLayer&);

   // DESTRUCTOR

   virtual ~ProbabilisticLayer(void);

   // ASSIGNMENT OPERATOR

   ProbabilisticLayer& operator = (const ProbabilisticLayer&);

   // EQUAL TO OPERATOR

   bool operator == (const ProbabilisticLayer&) const;

   // ENUMERATIONS

   /// Enumeration of available methods for interpreting variables as probabilities.  
   
   enum ProbabilisticMethod{Binary, Probability, Competitive, Softmax, NoProbabilistic};

   // GET METHODS

   const size_t& get_probabilistic_neurons_number(void) const;

   const double& get_decision_threshold(void) const;

   const ProbabilisticMethod& get_probabilistic_method(void) const;
   std::string write_probabilistic_method(void) const;
   std::string write_probabilistic_method_text(void) const;


   const bool& get_display(void) const;

   // SET METHODS

   void set(void);
   void set(const size_t&);
   void set(const ProbabilisticLayer&);

   void set_probabilistic_neurons_number(const size_t&);

   void set_decision_threshold(const double&);

   void set_probabilistic_method(const ProbabilisticMethod&);
   void set_probabilistic_method(const std::string&);

   virtual void set_default(void);

   // Display messages

   void set_display(const bool&);

   // Pruning and growing

   void prune_probabilistic_neuron(void);

   // Initialization methods

   void initialize_random(void);

   // Probabilistic post-processing

   Vector<double> calculate_outputs(const Vector<double>&) const;
   Matrix<double> calculate_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_Hessian_form(const Vector<double>&) const;

   Vector<double> calculate_binary_output(const Vector<double>&) const;
   Matrix<double> calculate_binary_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_binary_Hessian_form(const Vector<double>&) const;

   Vector<double> calculate_probability_output(const Vector<double>&) const;
   Matrix<double> calculate_probability_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_probability_Hessian_form(const Vector<double>&) const;

   Vector<double> calculate_competitive_output(const Vector<double>&) const;
   Matrix<double> calculate_competitive_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_competitive_Hessian_form(const Vector<double>&) const;

   Vector<double> calculate_softmax_output(const Vector<double>&) const;
   Matrix<double> calculate_softmax_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_softmax_Hessian_form(const Vector<double>&) const;

   Vector<double> calculate_no_probabilistic_output(const Vector<double>&) const;
   Matrix<double> calculate_no_probabilistic_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_no_probabilistic_Hessian_form(const Vector<double>&) const;

   // Expression methods

   std::string write_binary_expression(const Vector<std::string>&, const Vector<std::string>&) const;
   std::string write_probability_expression(const Vector<std::string>&, const Vector<std::string>&) const;
   std::string write_competitive_expression(const Vector<std::string>&, const Vector<std::string>&) const;
   std::string write_softmax_expression(const Vector<std::string>&, const Vector<std::string>&) const;
   std::string write_no_probabilistic_expression(const Vector<std::string>&, const Vector<std::string>&) const;

   std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const;

   // Serialization methods

   std::string to_string(void) const;

   virtual tinyxml2::XMLDocument* to_XML(void) const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   //virtual void read_XML(   );

protected:

   // MEMBERS

   /// Number of probabilistic neurons in the layer. 

   size_t probabilistic_neurons_number;

   /// Decision threshold.

   double decision_threshold;

   /// Probabilistic processing method.

   ProbabilisticMethod probabilistic_method;

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

