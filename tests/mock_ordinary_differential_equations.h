/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   http://flood.sourceforge.net                                                                               */
/*                                                                                                              */
/*   M O C K   O R D I N A R Y   D I F F E R E N T I A L   E Q U A T I O N S   C L A S S   H E A D E R          */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   E-mail: roberto-lopez@users.sourceforge.net                                                                */ 
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MOCKORDINARYDIFFERENTIALEQUATIONS_H__
#define __MOCKORDINARYDIFFERENTIALEQUATIONS_H__

#include "../opennn/opennn.h"

namespace OpenNN
{

class MockOrdinaryDifferentialEquations : public OrdinaryDifferentialEquations
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MockOrdinaryDifferentialEquations(void);

   // DESTRUCTOR

   virtual ~MockOrdinaryDifferentialEquations(void);


   // METHODS

   // Get methods

   // Set methods

   void set_default(void);

   // Mathematical model methods

   Vector<double> calculate_dependent_variables_dots(const NeuralNetwork&, const Vector<double>&) const;

};

}

#endif


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
