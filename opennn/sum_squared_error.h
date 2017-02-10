/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S U M   S Q U A R E D   E R R O R   C L A S S   H E A D E R                                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __SUMSQUAREDERROR_H__
#define __SUMSQUAREDERROR_H__

// System includes

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>

// OpenNN includes

#include "error_term.h"
#include "data_set.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This class represents the sum squared peformance term functional. 
/// This is used as the error term in data modeling problems, such as function regression, 
/// classification or time series prediction.

class SumSquaredError : public ErrorTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit SumSquaredError(void);

   // NEURAL NETWORK CONSTRUCTOR

   explicit SumSquaredError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit SumSquaredError(DataSet*);

   // GENERAL CONSTRUCTOR

   explicit SumSquaredError(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit SumSquaredError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   SumSquaredError(const SumSquaredError&);

   // DESTRUCTOR

   virtual ~SumSquaredError(void);    

   // METHODS

   // Get methods

   // Set methods

   // Checking methods

   void check(void) const;

   // loss methods

   double calculate_error(void) const;

   double calculate_selection_error(void) const;

   Vector<double> calculate_output_gradient(const Vector<double>&, const Vector<double>&) const;

   Matrix<double> calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const;

   double calculate_error(const Vector<double>&) const;

   Matrix<double> calculate_single_hidden_layer_Hessian(void) const;

   double calculate_loss_combination(const size_t&, const Vector<double>&) const;
   double calculate_loss_combinations(const size_t&, const Vector<double>&, const size_t&, const Vector<double>&) const;

   // Objective terms methods

   Vector<double> calculate_terms(void) const;
   Vector<double> calculate_terms(const Vector<double>&) const;

   Matrix<double> calculate_terms_Jacobian(void) const;

   ErrorTerm::FirstOrderTerms calculate_first_order_terms(void) const;

   // Squared errors methods

   Vector<double> calculate_squared_errors(void) const;

   std::string write_error_term_type(void) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );

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
