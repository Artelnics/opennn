/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   http://flood.sourceforge.net                                                                               */
/*                                                                                                              */
/*   P O L Y N O M I A L   C L A S S                                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   E-mail: roberto-lopez@users.sourceforge.net                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "polynomial.h"

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>


using namespace OpenNN;


// GENERAL CONSTRUCTOR

Polynomial::Polynomial(NeuralNetwork* new_neural_network_pointer)       
{
}


// DEFAULT CONSTRUCTOR

Polynomial::Polynomial(void)
{
}


// DESTRUCTOR

Polynomial::~Polynomial(void)
{
}


// METHODS

// const Vector<double>& get_coefficients(void) const method

const Vector<double>& Polynomial::get_coefficients(void) const
{
   return(coefficients);
}


// void set_coefficients(const Vector<double>&) method

void Polynomial::set_coefficients(const Vector<double>& new_coefficients)
{
   coefficients = new_coefficients;
}


// double calculate_outputs(double) const method

double Polynomial::calculate_outputs(double inputs) const
{
   unsigned int size = coefficients.size();

   double outputs = 0.0;

   for(unsigned int i = 0; i < size; i++)
   {
      outputs += coefficients[i]*pow(inputs,(int)i);
   }

   return(outputs);
}


// double calculate_evaluation(void) const method

double Polynomial::calculate_evaluation(void) const
{
/*
   const IndependentParameters& independent_parameters = neural_network_pointer->get_independent_parameters();

   double inputs = independent_parameters.get_parameter(0);

   double objective = calculate_outputs(inputs);

   return(objective);
*/

   return(0.0);
}


// double calculate_evaluation(const Vector<double>&) const method

double Polynomial::calculate_evaluation(const Vector<double>&) const
{
   return(0.0);
}


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
