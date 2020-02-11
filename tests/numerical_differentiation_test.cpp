//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D I F F E R E N T I A T I O N   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "numerical_differentiation_test.h"


NumericalDifferentiationTest::NumericalDifferentiationTest() : UnitTesting() 
{
}


NumericalDifferentiationTest::~NumericalDifferentiationTest()
{
}

/*
void NumericalDifferentiationTest::test_constructor()
{
   cout << "test_constructor\n";
}


void NumericalDifferentiationTest::test_destructor()
{
   cout << "test_destructor\n";
}


void NumericalDifferentiationTest::test_calculate_forward_differences_derivatives()
{
   cout << "test_calculate_forward_differences_derivative\n";

   NumericalDifferentiation nd;

   type d;
   type x;

   // Test

   x = 0.0;
   d = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d == 1.0, LOG);

}


void NumericalDifferentiationTest::test_calculate_central_differences_derivatives()
{
   cout << "test_calculate_central_differences_derivative\n";

   NumericalDifferentiation nd;

   type x;
   type d;

   // Test

   x = 0.0;
   d = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_derivatives()
{
   cout << "test_calculate_derivative\n";

   NumericalDifferentiation nd;

   type d;

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
   cout << "test_calculate_forward_differences_second_derivative\n";

   NumericalDifferentiation nd;

   type x;
   type d2;

   // Test

   x = 0.0;
   d2 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(d2 == 0.0, LOG);

   // Test

   Tensor<type, 2> matrix;

   Tensor<type, 1> x1(5);
   Tensor<type, 1> x2(3);

   const Index dummy_1 = 0;
   const Index dummy_2 = 0;

   x1.setRandom();
   x2.setRandom();

   matrix = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f7, dummy_1, x1, dummy_2, x2);

   assert_true(matrix.dimension(0) == 5, LOG);
   assert_true(matrix.dimension(1) == 3, LOG);

   // Test

   x1.set(5, 1.0);
   x2.set(5, 1.0);

   matrix = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f7, dummy_1, x1, dummy_2, x2);

   assert_true(matrix.dimension(0) == 5, LOG);
   assert_true(matrix.dimension(1) == 5, LOG);
   assert_true(matrix.to_vector().is_constant(), LOG);

   // Test

   x1.set(9);
   x2.set(15);

   matrix = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f7, dummy_1, x1, dummy_2, x2);

   assert_true(matrix.dimension(0) == 9, LOG);
   assert_true(matrix.dimension(1) == 15, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_second_derivatives()
{
   cout << "test_calculate_central_differences_second_derivative\n";

   NumericalDifferentiation nd;

   type x;
   type d2;

   // Test

   x = 0.0;
   d2 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(abs(d2) <= 1.0e-6, LOG);
}


void NumericalDifferentiationTest::test_calculate_second_derivatives()
{
   cout << "test_calculate_second_derivative\n";

   NumericalDifferentiation nd;

   type x;
   type d2;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x = 0.0;
   d2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(abs(d2) <= 1.0e-6, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x = 0.0;
   d2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);

   assert_true(abs(d2) <= 1.0e-6, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_gradient()
{
   cout << "test_calculate_forward_differences_gradient\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 1> g;
	   
   // Test

   x.set(2, 0.0);

   g = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(g.size() == 2, LOG);
   assert_true(g == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_gradient()
{
   cout << "test_calculate_central_differences_gradient\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 1> g;
	   
   // Test

   x.set(2, 0.0);

   g = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(g.size() == 2, LOG);
   assert_true(g == 1.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_training_loss_gradient()
{
   cout << "test_calculate_training_loss_gradient\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 1> g;
	   
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


void NumericalDifferentiationTest::test_calculate_forward_differences_hessian()
{
   cout << "test_calculate_forward_differences_hessian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 2> H;
	   
   // Test

   x.set(2, 0.0);
   H = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(H == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_hessian()
{
   cout << "test_calculate_central_differences_hessian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 2> H;
	   
   // Test

   x.set(2, 0.0);
   H = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(H == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_hessian()
{
   cout << "test_calculate_hessian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 2> H;
	   
   Tensor<type, 2> forward;
   Tensor<type, 2> central;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x.set(2, 0.0);

   H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(H == 0.0, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x.set(2, 0.0);

   H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(H == 0.0, LOG);

   // Test

   x.set(4);
   x.setRandom();

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   forward = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2, x);

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   central = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(absolute_value(forward-central) < 1.0e-3, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_Jacobian()
{
   cout << "test_calculate_forward_differences_Jacobian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 2> J;

   Tensor<type, 2> J_true;

   // Test

   x.set(2, 0.0);

   J = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f3, x);

   J_true.set(2, 2);
   J_true.initialize_identity();

   assert_true(J == J_true, LOG);
}


void NumericalDifferentiationTest::test_calculate_central_differences_Jacobian()
{
   cout << "test_calculate_central_differences_Jacobian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 2> J;

   Tensor<type, 2> J_true;

   // Test

   x.set(2, 0.0);

   J = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f3, x);

   J_true.set_identity(2);

   assert_true(J == J_true, LOG);
}


void NumericalDifferentiationTest::test_calculate_Jacobian()
{
   cout << "test_calculate_Jacobian\n";

   NumericalDifferentiation nd;

   Index dummy;

   Tensor<type, 1> x;
   Tensor<type, 2> J;

   Tensor<type, 2> J_true;

   Tensor<type, 2> J_fd;
   Tensor<type, 2> J_cd;


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

//   assert_true(absolute_value(maximum((J_fd-J_cd))) < 0.05, LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_hessian_form()
{
   cout << "test_calculate_forward_differences_hessian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<Tensor<type, 2>, 1> H;

   // Test

   x.set(2, 0.0);
   H = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].dimension(0) == 2, LOG);
   assert_true(H[0].dimension(1) == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].dimension(0) == 2, LOG);
   assert_true(H[1].dimension(1) == 2, LOG);
   assert_true(H[1] == 0.0, LOG);
}

 
void NumericalDifferentiationTest::test_calculate_central_differences_hessian_form()
{
   cout << "test_calculate_central_differences_hessian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x(2, 0.0);

   Tensor<Tensor<type, 2>, 1> hessian = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(hessian.size() == 2, LOG);

   assert_true(hessian[0].dimension(0) == 2, LOG);
   assert_true(hessian[0].dimension(1) == 2, LOG);
   assert_true(hessian[0] == 0.0, LOG);

   assert_true(hessian[1].dimension(0) == 2, LOG);
   assert_true(hessian[1].dimension(1) == 2, LOG);
   assert_true(hessian[1] == 0.0, LOG);
}


void NumericalDifferentiationTest::test_calculate_hessian_form()
{
   cout << "test_calculate_hessian\n";

   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<Tensor<type, 2>, 1> H;

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   x.set(2, 0.0);

   H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].dimension(0) == 2, LOG);
   assert_true(H[0].dimension(1) == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].dimension(0) == 2, LOG);
   assert_true(H[1].dimension(1) == 2, LOG);
   assert_true(H[1] == 0.0, LOG);

   // Test

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   x.set(2, 0.0);

   H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f3, x);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].dimension(0) == 2, LOG);
   assert_true(H[0].dimension(1) == 2, LOG);
   assert_true(H[0] == 0.0, LOG);

   assert_true(H[1].dimension(0) == 2, LOG);
   assert_true(H[1].dimension(1) == 2, LOG);
   assert_true(H[1] == 0.0, LOG);
}
*/

void NumericalDifferentiationTest::run_test_case()
{
   cout << "Running numerical differentiation test case...\n";
/*
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
   test_calculate_training_loss_gradient();

   // hessian methods

   test_calculate_forward_differences_hessian();
   test_calculate_central_differences_hessian();
   test_calculate_hessian();

   // Jacobian methods

   test_calculate_forward_differences_Jacobian();
   test_calculate_central_differences_Jacobian();
   test_calculate_Jacobian();

   // hessian methods

   test_calculate_forward_differences_hessian();
   test_calculate_central_differences_hessian();
   test_calculate_hessian();
*/
   cout << "End of numerical differentiation test case.\n";
}

/*
type NumericalDifferentiationTest::f1(const type& x) const
{
   return x;
}


type NumericalDifferentiationTest::f2(const Tensor<type, 1>& x) const
{
   return x.sum();
}


Tensor<type, 1> NumericalDifferentiationTest::f3(const Tensor<type, 1>& x) const
{ 
   return x;
}


type NumericalDifferentiationTest::f7(const Index&, const Tensor<type, 1>& x, const Index&, const Tensor<type, 1>& y) const
{
   return l2_norm(x.assemble(y));
}


Tensor<type, 1> NumericalDifferentiationTest::f8(const Index&, const Index&, const Tensor<type, 1>& x) const
{
    return x*x*(x+1.0);
}
*/


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
