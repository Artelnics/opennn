/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N U M E R I C A L   D I F F E R E N T I A T I O N   T E S T   C L A S S                                    */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "polynomial.h"

#include "numerical_differentiation_test.h"


using namespace OpenNN;


NumericalDifferentiationTest::NumericalDifferentiationTest() : UnitTesting() 
{
}


NumericalDifferentiationTest::~NumericalDifferentiationTest()
{
}


void NumericalDifferentiationTest::test_constructor()
{
   message += "test_constructor\n";
}


void NumericalDifferentiationTest::test_destructor()
{
   message += "test_destructor\n";
}


void NumericalDifferentiationTest::test_calculate_forward_differences_derivatives()
{
   message += "test_calculate_forward_differences_derivative\n";

   NumericalDifferentiation nd;

   double d;
   double x;

   // Test

   x = 0.0;
   d = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d == 1.0, LOG);

}


void NumericalDifferentiationTest::test_calculate_central_differences_derivatives()
{
   message += "test_calculate_central_differences_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d;

   // Test

   x = 0.0;
   d = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_derivatives()
{
   message += "test_calculate_derivative\n";

   NumericalDifferentiation nd;

   double d;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   d = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f1, 0.0);

   assert_true(d == 1.0, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f1, 0.0);

   assert_true(d == 1.0, LOG);

}


void NumericalDifferentiationTest::test_calculate_forward_differences_second_derivatives()
{
   message += "test_calculate_forward_differences_second_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d2;

   // Test

   x = 0.0;
   d2 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d2 == 0.0, LOG);

   // Test

   Matrix<double> M;

   Vector<double> x1(5);
   Vector<double> x2(3);

   const size_t dummy_1 = 0;
   const size_t dummy_2 = 0;

   x1.randomize_normal();
   x2.randomize_normal();

   M = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f7, dummy_1, x1, dummy_2, x2);

   assert_true(M.get_rows_number() == 5, LOG);
   assert_true(M.get_columns_number() == 3, LOG);

   // Test

   x1.set(5, 1.0);
   x2.set(5, 1.0);

   M = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f7, dummy_1, x1, dummy_2, x2);

   assert_true(M.get_rows_number() == 5, LOG);
   assert_true(M.get_columns_number() == 5, LOG);
   assert_true(M.to_vector().is_constant(), LOG);

   // Test

   x1.set(9);
   x2.set(15);

   M = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f7, dummy_1, x1, dummy_2, x2);

   assert_true(M.get_rows_number() == 9, LOG);
   assert_true(M.get_columns_number() == 15, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_second_derivatives()
{
   message += "test_calculate_central_differences_second_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d2;

   // Test

   x = 0.0;
   d2 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(fabs(d2) <= 1.0e-6, LOG);
}


void NumericalDifferentiationTest::test_calculate_second_derivatives()
{
   message += "test_calculate_second_derivative\n";

   NumericalDifferentiation nd;

   double x;
   double d2;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x = 0.0;
   d2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(fabs(d2) <= 1.0e-6, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x = 0.0;
   d2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(fabs(d2) <= 1.0e-6, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_gradient()
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


void NumericalDifferentiationTest::test_calculate_central_differences_gradient()
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


void NumericalDifferentiationTest::test_calculate_gradient()
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


void NumericalDifferentiationTest::test_calculate_forward_differences_Hessian()
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


void NumericalDifferentiationTest::test_calculate_central_differences_Hessian()
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


void NumericalDifferentiationTest::test_calculate_Hessian()
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


void NumericalDifferentiationTest::test_calculate_forward_differences_Jacobian()
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


void NumericalDifferentiationTest::test_calculate_central_differences_Jacobian()
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


void NumericalDifferentiationTest::test_calculate_Jacobian()
{
   message += "test_calculate_Jacobian\n";

   NumericalDifferentiation nd;

   size_t dummy;

   Vector<double> x;
   Matrix<double> J;

   Matrix<double> J_true;

   Matrix<double> J_fd;
   Matrix<double> J_cd;


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

   // Test

   x.set(2, 1.23);

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   J_fd = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f8, dummy, dummy, x);

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   J_cd = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f8, dummy, dummy, x);

   assert_true((J_fd-J_cd).calculate_absolute_value().calculate_maximum() < 0.05, LOG);

   cout << J_fd << endl;
   cout << J_cd << endl;

}


void NumericalDifferentiationTest::test_calculate_forward_differences_Hessian_form()
{
   message += "test_calculate_forward_differences_Hessian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Vector< Matrix<double> > H;

   // Test

   x.set(2, 0.0);
   H = nd.calculate_forward_differences_Hessian(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].get_rows_number() == 2, LOG);
   assert_true(H[0].get_columns_number() == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].get_rows_number() == 2, LOG);
   assert_true(H[1].get_columns_number() == 2, LOG);
   assert_true(H[1] == 0.0, LOG);
}

 
void NumericalDifferentiationTest::test_calculate_central_differences_Hessian_form()
{
   message += "test_calculate_central_differences_Hessian\n";

   NumericalDifferentiation nd;

   Vector<double> x(2, 0.0);

   Vector< Matrix<double> > Hessian = nd.calculate_central_differences_Hessian(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(Hessian.size() == 2, LOG);

   assert_true(Hessian[0].get_rows_number() == 2, LOG);
   assert_true(Hessian[0].get_columns_number() == 2, LOG);
   assert_true(Hessian[0] == 0.0, LOG);

   assert_true(Hessian[1].get_rows_number() == 2, LOG);
   assert_true(Hessian[1].get_columns_number() == 2, LOG);
   assert_true(Hessian[1] == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_Hessian_form()
{
   message += "test_calculate_Hessian\n";

   NumericalDifferentiation nd;

   Vector<double> x;
   Vector< Matrix<double> > H;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x.set(2, 0.0);

   H = nd.calculate_Hessian(*this, &NumericalDifferentiationTest::f3, x);

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

   H = nd.calculate_Hessian(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].get_rows_number() == 2, LOG);
   assert_true(H[0].get_columns_number() == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].get_rows_number() == 2, LOG);
   assert_true(H[1].get_columns_number() == 2, LOG);
   assert_true(H[1] == 0.0, LOG);
}


void NumericalDifferentiationTest::run_test_case()
{
   message += "Running numerical differentiation test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Derivative methods

   test_calculate_forward_differences_derivatives();
   test_calculate_central_differences_derivatives();
   test_calculate_derivatives();

   test_calculate_forward_differences_derivatives();
   test_calculate_central_differences_derivatives();
   test_calculate_derivatives();

   // Second derivative methods

   test_calculate_forward_differences_second_derivatives();
   test_calculate_central_differences_second_derivatives();
   test_calculate_second_derivatives();

   test_calculate_forward_differences_second_derivatives();
   test_calculate_central_differences_second_derivatives();
   test_calculate_second_derivatives();

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


double NumericalDifferentiationTest::f1(const double& x) const 
{
   return(x);
}


double NumericalDifferentiationTest::f2(const Vector<double>& x) const
{
   return(x.calculate_sum());
}


Vector<double> NumericalDifferentiationTest::f3(const Vector<double>& x) const
{ 
   return(x);
}


double NumericalDifferentiationTest::f7(const size_t&, const Vector<double>& x, const size_t&, const Vector<double>& y) const
{
   return(x.assemble(y).calculate_L2_norm());
}


Vector<double> NumericalDifferentiationTest::f8(const size_t&, const size_t&, const Vector<double>& x) const
{
    return x*x*(x+1.0);
}



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
