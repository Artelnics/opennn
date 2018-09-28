/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O C K   P E R F O R M A N C E   T E R M   C L A S S   H E A D E R                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MOCKErrorTerm_H__
#define __MOCKErrorTerm_H__


// OpenNN includes

#include "../opennn/opennn.h"


using namespace OpenNN;


class MockErrorTerm : public ErrorTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MockErrorTerm();

   // GENERAL CONSTRUCTOR

   explicit MockErrorTerm(NeuralNetwork*);

   explicit MockErrorTerm(NeuralNetwork*, DataSet*);


   // DESTRUCTOR

   virtual ~MockErrorTerm(); 

    // ENUMERATIONS

    enum Expression{SumSquaredParameters, OutputIntegral};

   // METHODS

    const Expression& get_expression() const;
    void set_expression(const Expression&);

    void set_default();

    // Sum squared parameters methods

    double calculate_sum_squared_parameters() const;
    Vector<double> calculate_sum_squared_parameters_gradient() const;
    Matrix<double> calculate_sum_squared_parameters_Hessian() const;

    Vector<double> calculate_sum_squared_parameters_terms() const;
    Matrix<double> calculate_sum_squared_parameters_terms_Jacobian() const;

    double calculate_sum_squared_parameters(const Vector<double>&) const;
    Vector<double> calculate_sum_squared_parameters_gradient(const Vector<double>&) const;
    Matrix<double> calculate_sum_squared_parameters_Hessian(const Vector<double>&) const;

   // Output integral methods

   double calculate_output_integral_integrand(const double&) const;
   double calculate_output_integral() const;
   
   Vector<double> calculate_output_integral_integrand_gradient(const double&) const;
   Vector<double> calculate_output_integral_gradient() const;

   Matrix<double> calculate_output_integral_integrand_Hessian(const double&) const;
   Matrix<double> calculate_output_integral_Hessian() const;

   double calculate_output_integral(const Vector<double>&) const;
   Vector<double> calculate_output_integral_gradient(const Vector<double>&) const;
   Matrix<double> calculate_output_integral_Hessian(const Vector<double>&) const;


   // Performance methods

   double calculate_error() const;
   Vector<double> calculate_gradient() const;
   Matrix<double> calculate_Hessian() const;

   Vector<double> calculate_terms() const;
   Matrix<double> calculate_terms_Jacobian() const;

   double calculate_error(const Vector<double>&) const;
   Vector<double> calculate_gradient(const Vector<double>&) const;
   Matrix<double> calculate_Hessian(const Vector<double>&) const;


private:

   Expression expression;

   NumericalIntegration numerical_integration;


};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Roberto Lopez
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
