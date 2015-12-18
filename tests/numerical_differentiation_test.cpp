/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   N U M E R I C A L   D I F F E R E N T I A T I O N   T E S T   C L A S S                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "polynomial.h"

#include "numerical_differentiation_test.h"


using namespace OpenNN;


NumericalDifferentiationTest::NumericalDifferentiationTest(void) : UnitTesting() 
{
}


NumericalDifferentiationTest::~NumericalDifferentiationTest(void)
{
}


void NumericalDifferentiationTest::test_constructor(void)
{
   message += "test_constructor\n";
}


void NumericalDifferentiationTest::test_destructor(void)
{
   message += "test_destructor\n";
}


void NumericalDifferentiationTest::test_calculate_forward_differences_derivative(void)
{
   message += "test_calculate_forward_differences_derivative\n";

   NumericalDifferentiation nd;

   double d;
   double x;

   // Test

   x = 0.0;
   d = nd.calculate_forward_differences_derivative(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d == 1.0, LOG);

}


void NumericalDifferentiationTest::test_calculate_central_differences_derivative(void)
{
   message += "test_calculate_central_differences_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d;

   // Test

   x = 0.0;
   d = nd.calculate_central_differences_derivative(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_derivative(void)
{
   message += "test_calculate_derivative\n";

   NumericalDifferentiation nd;

   double d;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   d = nd.calculate_derivative(*this, &NumericalDifferentiationTest::f1, 0.0);

   assert_true(d == 1.0, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d = nd.calculate_derivative(*this, &NumericalDifferentiationTest::f1, 0.0);

   assert_true(d == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_second_derivative(void)
{
   message += "test_calculate_forward_differences_second_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d2;

   // Test

   x = 0.0;
   d2 = nd.calculate_forward_differences_second_derivative(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d2 == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_second_derivative(void)
{
   message += "test_calculate_central_differences_second_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d2;

   // Test

   x = 0.0;
   d2 = nd.calculate_central_differences_second_derivative(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(fabs(d2) <= 1.0e-6, LOG);
}


void NumericalDifferentiationTest::test_calculate_second_derivative(void)
{
   message += "test_calculate_second_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d2;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x = 0.0;
   d2 = nd.calculate_second_derivative(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(fabs(d2) <= 1.0e-6, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x = 0.0;
   d2 = nd.calculate_second_derivative(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(fabs(d2) <= 1.0e-6, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_gradient(void)
{
   message += "test_calculate_forward_differences_gradient\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Vector<double> g;
	   
   // Test

   x.set(2, 0.0);

   g = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(g.size() == 2, LOG);
   assert_true(g == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_gradient(void)
{
   message += "test_calculate_central_differences_gradient\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Vector<double> g;
	   
   // Test

   x.set(2, 0.0);

   g = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(g.size() == 2, LOG);
   assert_true(g == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_gradient(void)
{
   message += "test_calculate_gradient\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Vector<double> g;
	   
   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x.set(2, 0.0);

   g = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(g.size() == 2, LOG);
   assert_true(g == 1.0, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x.set(2, 0.0);

   g = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(g.size() == 2, LOG);
   assert_true(g == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_Hessian(void)
{
   message += "test_calculate_forward_differences_Hessian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Matrix<double> H;
	   
   // Test

   x.set(2, 0.0);
   H = nd.calculate_forward_differences_Hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.get_rows_number() == 2, LOG);
   assert_true(H.get_columns_number() == 2, LOG);
   assert_true(H == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_Hessian(void)
{
   message += "test_calculate_central_differences_Hessian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Matrix<double> H;
	   
   // Test

   x.set(2, 0.0);
   H = nd.calculate_central_differences_Hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.get_rows_number() == 2, LOG);
   assert_true(H.get_columns_number() == 2, LOG);
   assert_true(H == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_Hessian(void)
{
   message += "test_calculate_Hessian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Matrix<double> H;
	   
   Matrix<double> forward;
   Matrix<double> central;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x.set(2, 0.0);

   H = nd.calculate_Hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.get_rows_number() == 2, LOG);
   assert_true(H.get_columns_number() == 2, LOG);
   assert_true(H == 0.0, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x.set(2, 0.0);

   H = nd.calculate_Hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.get_rows_number() == 2, LOG);
   assert_true(H.get_columns_number() == 2, LOG);
   assert_true(H == 0.0, LOG);

   // Test

   x.set(4);
   x.randomize_normal();

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   forward = nd.calculate_Hessian(*this, &NumericalDifferentiationTest::f2, x);

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   central = nd.calculate_Hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true((forward-central).calculate_absolute_value() < 1.0e-3, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_Jacobian(void)
{
   message += "test_calculate_forward_differences_Jacobian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Matrix<double> J;

   Matrix<double> J_true;

   // Test

   x.set(2, 0.0);

   J = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f3, x);

   J_true.set(2, 2);
   J_true.initialize_identity();

   assert_true(J == J_true, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_Jacobian(void)
{
   message += "test_calculate_central_differences_Jacobian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Matrix<double> J;

   Matrix<double> J_true;

   // Test

   x.set(2, 0.0);

   J = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f3, x);

   J_true.set_identity(2);

   assert_true(J == J_true, LOG);
}


void NumericalDifferentiationTest::test_calculate_Jacobian(void)
{
   message += "test_calculate_Jacobian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Matrix<double> J;

   Matrix<double> J_true;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x.set(2, 0.0);

   J = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f3, x);

   J_true.set_identity(2);

   assert_true(J == J_true, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x.set(2, 0.0);

   J = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f3, x);

   J_true.set_identity(2);

   assert_true(J == J_true, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_Hessian_form(void)
{
   message += "test_calculate_forward_differences_Hessian_form\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Vector< Matrix<double> > H;

   // Test

   x.set(2, 0.0);
   H = nd.calculate_forward_differences_Hessian_form(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].get_rows_number() == 2, LOG);
   assert_true(H[0].get_columns_number() == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].get_rows_number() == 2, LOG);
   assert_true(H[1].get_columns_number() == 2, LOG);
   assert_true(H[1] == 0.0, LOG);
}

 
void NumericalDifferentiationTest::test_calculate_central_differences_Hessian_form(void)
{
   message += "test_calculate_central_differences_Hessian_form\n";

   NumericalDifferentiation nd;

   Vector<double> x(2, 0.0);

   Vector< Matrix<double> > Hessian = nd.calculate_central_differences_Hessian_form(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(Hessian.size() == 2, LOG);

   assert_true(Hessian[0].get_rows_number() == 2, LOG);
   assert_true(Hessian[0].get_columns_number() == 2, LOG);
   assert_true(Hessian[0] == 0.0, LOG);

   assert_true(Hessian[1].get_rows_number() == 2, LOG);
   assert_true(Hessian[1].get_columns_number() == 2, LOG);
   assert_true(Hessian[1] == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_Hessian_form(void)
{
   message += "test_calculate_Hessian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Vector< Matrix<double> > H;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x.set(2, 0.0);

   H = nd.calculate_Hessian_form(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].get_rows_number() == 2, LOG);
   assert_true(H[0].get_columns_number() == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].get_rows_number() == 2, LOG);
   assert_true(H[1].get_columns_number() == 2, LOG);
   assert_true(H[1] == 0.0, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x.set(2, 0.0);

   H = nd.calculate_Hessian_form(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].get_rows_number() == 2, LOG);
   assert_true(H[0].get_columns_number() == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].get_rows_number() == 2, LOG);
   assert_true(H[1].get_columns_number() == 2, LOG);
   assert_true(H[1] == 0.0, LOG);
}


void NumericalDifferentiationTest::run_test_case(void)
{
   message += "Running numerical differentiation test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Derivative methods

   test_calculate_forward_differences_derivative();
   test_calculate_central_differences_derivative();
   test_calculate_derivative();

   test_calculate_forward_differences_derivative();
   test_calculate_central_differences_derivative();
   test_calculate_derivative();

   // Second derivative methods

   test_calculate_forward_differences_second_derivative();
   test_calculate_central_differences_second_derivative();
   test_calculate_second_derivative();

   test_calculate_forward_differences_second_derivative();
   test_calculate_central_differences_second_derivative();
   test_calculate_second_derivative();

   // Gradient methods

   test_calculate_forward_differences_gradient();
   test_calculate_central_differences_gradient();
   test_calculate_gradient();

   // Hessian methods

   test_calculate_forward_differences_Hessian();
   test_calculate_central_differences_Hessian();
   test_calculate_Hessian();

   // Jacobian methods

   test_calculate_forward_differences_Jacobian();
   test_calculate_central_differences_Jacobian();
   test_calculate_Jacobian();

   // Hessian methods

   test_calculate_forward_differences_Hessian();
   test_calculate_central_differences_Hessian();
   test_calculate_Hessian();

   message += "End of numerical differentiation test case.\n";
}


// double f1(const double&) const  method

double NumericalDifferentiationTest::f1(const double& x) const 
{
   return(x);
}


// double f2(const Vector<double>&) const method

double NumericalDifferentiationTest::f2(const Vector<double>& x) const
{
   return(x.calculate_sum());
}


// Vector<double> f3(const Vector<double>&) const method

Vector<double> NumericalDifferentiationTest::f3(const Vector<double>& x) const
{ 
   return(x);
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2015 Roberto Lopez.
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
