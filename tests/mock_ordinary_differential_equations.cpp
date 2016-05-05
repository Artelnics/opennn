/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   http://flood.sourceforge.net                                                                               */
/*                                                                                                              */
/*   M O C K   O R D I N A R Y   D I F F E R E N T I A L   E Q U A T I O N S   C L A S S                        */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   E-mail: roberto-lopez@users.sourceforge.net                                                                */ 
/*                                                                                                              */
/****************************************************************************************************************/

// System includes

#include <iostream>     
#include <fstream>     
#include <cmath>     


#include "mock_ordinary_differential_equations.h"


namespace OpenNN
{

// DEFAULT CONSTRUCTOR

MockOrdinaryDifferentialEquations::MockOrdinaryDifferentialEquations(void) : OrdinaryDifferentialEquations()
{
   set_default();
}


// DESTRUCTOR

MockOrdinaryDifferentialEquations::~MockOrdinaryDifferentialEquations(void) 
{
}


// METHODS


// void set_default(void) method

void MockOrdinaryDifferentialEquations::set_default(void)
{
   dependent_variables_number = 1;
}


// Vector<double> calculate_dependent_variables_dots(const NeuralNetwork&, const Vector<double>&) const method

Vector<double> MockOrdinaryDifferentialEquations::calculate_dependent_variables_dots(const NeuralNetwork&, const Vector<double>&) const
{
   Vector<double> dependent_variables_dots(dependent_variables_number, 0.0);

   return(dependent_variables_dots);
}

}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2012 Roberto Lopez 
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
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
