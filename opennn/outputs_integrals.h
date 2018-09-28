/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O U T P U T S   I N T E G R A L S   C L A S S   H E A D E R                                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __OUTPUTSINTEGRALS_H__
#define __OUTPUTSINTEGRALS_H__

// System includes

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>

// OpenNN includes

#include "regularization_term.h"
#include "numerical_integration.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the outputs integrals error term. 
/// It is defined as the weighted sum of the integrals of the neural network outputs.
/// The neural network here must have only one input. 
/// This error term might be used in optimal control as an objective or a regularization terms. 

class OutputsIntegrals : public RegularizationTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit OutputsIntegrals();

   // NEURAL NETWORK CONSTRUCTOR

   explicit OutputsIntegrals(NeuralNetwork*);

   // XML CONSTRUCTOR

   explicit OutputsIntegrals(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~OutputsIntegrals();    

   // METHODS

   // Get methods

   const NumericalIntegration& get_numerical_integration() const;
   NumericalIntegration* get_numerical_integration_pointer();

   const Vector<double>& get_outputs_integrals_weights() const;
   const double& get_output_integral_weight(const size_t&) const;

   // Set methods

   void set_numerical_integration(const NumericalIntegration&);

   void set_outputs_integrals_weights(const Vector<double>&);
   void set_output_integral_weight(const size_t&, const double&);

   void set_default();

   // Checking methods

   void check() const;

   // Regularization methods

   double calculate_regularization() const;
   double calculate_regularization(const Vector<double>&) const;

   Vector<double> calculate_gradient() const;

   Matrix<double> calculate_Hessian() const;

   string write_error_term_type() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

private:

   /// Object for numerical integration of functions. 

   NumericalIntegration numerical_integration;

   /// Weigth for each output integral. 

   Vector<double> outputs_integrals_weights;

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
