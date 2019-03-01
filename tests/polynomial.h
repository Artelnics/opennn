/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   http://flood.sourceforge.net                                                                               */
/*                                                                                                              */
/*   P O L Y N O M I A L   C L A S S   H E A D E R                                                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   E-mail: roberto-lopez@users.sourceforge.net                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


#ifndef __POLYNOMIAL_H__
#define __POLYNOMIAL_H__

#include "../opennn/opennn.h"

using namespace OpenNN;

class Polynomial : public LossIndex
{

public:

   // GENERAL CONSTRUCTOR

   explicit Polynomial(NeuralNetwork*);

   // DEFAULT CONSTRUCTOR

   explicit Polynomial(void);

   // DESTRUCTOR

   virtual ~Polynomial(void);


   // METHODS

   // Get methods

   const Vector<double>& get_coefficients(void) const;

   // Set methods

   void set_coefficients(const Vector<double>&);

   // Output methods

   double calculate_outputs(double) const;

   // Objective function methods

   double calculate_evaluation(void) const;
   double calculate_evaluation(const Vector<double>&) const;
   
private:

   Vector<double> coefficients;
};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2012 Roberto Lopez 
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
