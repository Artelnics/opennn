/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C R O S S   E N T R O P Y   E R R O R   C L A S S   H E A D E R                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __CROSSENTROPYERROR_H__
#define __CROSSENTROPYERROR_H__

// System includes

#include <iostream>
#include <fstream>
#include <math.h>

// OpenNN includes

#include "error_term.h"
#include "data_set.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"


namespace OpenNN
{

/// This class represents the cross entropy error term. 
/// This functional is used in classification problems.

class CrossEntropyError : public ErrorTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit CrossEntropyError(void);

   // NEURAL NETWORK CONSTRUCTOR

   explicit CrossEntropyError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit CrossEntropyError(DataSet*);

   // GENERAL CONSTRUCTOR

   explicit CrossEntropyError(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit CrossEntropyError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   CrossEntropyError(const CrossEntropyError&);

   // DESTRUCTOR

   virtual ~CrossEntropyError(void);

   // ASSIGNMENT OPERATOR

   CrossEntropyError& operator = (const CrossEntropyError&);

   // EQUAL TO OPERATOR

   bool operator == (const CrossEntropyError&) const;

   // METHODS

   // Checking methods

   void check(void) const;

   // loss methods

   double calculate_error(void) const;
   double calculate_error(const Vector<double>&) const;

   double calculate_minimum_loss(void) const;

   double calculate_selection_error(void) const;
   double calculate_minimum_selection_error(void) const;

   double calculate_error_unnormalized(void) const;
   double calculate_error_unnormalized(const Vector<double>&) const;

   double calculate_minimum_loss_unnormalized(void) const;

   double calculate_selection_error_unnormalized(void) const;
   double calculate_minimum_selection_error_unnormalized(void) const;

   Vector<double> calculate_output_gradient(const Vector<double> &, const Vector<double> &) const;
   Matrix<double> calculate_output_Hessian(const Vector<double> &, const Vector<double> &) const;

   Vector<double> calculate_output_gradient_unnormalized(const Vector<double> &, const Vector<double> &) const;

   Vector<double> calculate_gradient_unnormalized(void) const;

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
