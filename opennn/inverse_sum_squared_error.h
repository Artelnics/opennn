/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N V E R S E   S U M   S Q U A R E D   E R R O R   C L A S S   H E A D E R                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INVERSESUMSQUAREDERROR_H__
#define __INVERSESUMSQUAREDERROR_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>

// OpenNN includes

#include "mathematical_model.h"
#include "data_set.h"

#include "performance_term.h"

namespace OpenNN
{

/// This class represents the concept of sum squared error for inverse problems. 
/// It measures the difference between the outputs from a mathematical model and the targets in a data set. 
/// This performance term is used in inverse problems.

class InverseSumSquaredError : public PerformanceTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit InverseSumSquaredError(void);

   // NEURAL NETWORK CONSTRUCTOR

   explicit InverseSumSquaredError(NeuralNetwork*);

   // NEURAL NETWORK, MATHEMATICAL MODEL AND DATA SET CONSTRUCTOR

   explicit InverseSumSquaredError(NeuralNetwork*, MathematicalModel*, DataSet*);

   // XML CONSTRUCTOR

   explicit InverseSumSquaredError(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~InverseSumSquaredError(void);    

   // ENUMERATIONS

   /// Enumeration of the different methods for putting the unknowns into the mathematical model. 

   enum UnknownsMethod{LookUpTable, IndependentParametersValues};

   // STRUCTURES

   // METHODS

   // Get methods

   const UnknownsMethod& get_unknowns_method(void) const;
   std::string write_unknowns_method(void) const;

   // Set methods

   void set_unknowns_method(const UnknownsMethod&);
   void set_unknowns_method(const std::string&);

   void set_default(void);

   // Checking methods

   void check(void) const;

   // Objective methods

   double calculate_performance(void) const;   
   double calculate_performance(const Vector<double>&) const;
   double calculate_selection_performance(void) const;

   std::string write_performance_term_type(void) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;   
   void from_XML(const tinyxml2::XMLDocument&);   

private:

   // MEMBERS

   /// Variable containing the method for putting the unknowns into the mathematical model. 

   UnknownsMethod unknowns_method;
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
