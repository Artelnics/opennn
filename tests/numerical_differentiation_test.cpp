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


void NumericalDifferentiationTest::test_constructor()
{
   cout << "test_constructor\n";

   NumericalDifferentiation nd;
   nd.set_numerical_differentiation_method(OpenNN::NumericalDifferentiation::ForwardDifferences);

   NumericalDifferentiation nd_1(nd);

   assert_true(nd_1.get_numerical_differentiation_method() ==OpenNN::NumericalDifferentiation::ForwardDifferences, LOG);
}

void NumericalDifferentiationTest::test_destructor()
{
   cout << "test_destructor\n";
}


void NumericalDifferentiationTest::test_set_get_methods()
{
   cout << "test_set_methods\n";

   NumericalDifferentiation nd;
   NumericalDifferentiation nd_1;

   // Test 1

   nd.set_numerical_differentiation_method(OpenNN::NumericalDifferentiation::ForwardDifferences);
   nd.set_precision_digits(9);
   nd.set_display(true);

   assert_true(nd.get_numerical_differentiation_method() == OpenNN::NumericalDifferentiation::ForwardDifferences, LOG);
   assert_true(nd.get_precision_digits() == 9, LOG);
   assert_true(nd.get_display() == true, LOG);

   // Test 2

   nd_1.set(nd);

   assert_true(nd_1.get_numerical_differentiation_method() == OpenNN::NumericalDifferentiation::ForwardDifferences, LOG);
   assert_true(nd_1.get_precision_digits() == 9, LOG);
   assert_true(nd_1.get_display() == true, LOG);

   // Test 3

   nd.set_numerical_differentiation_method(OpenNN::NumericalDifferentiation::ForwardDifferences);
   assert_true(nd.get_numerical_differentiation_method() == OpenNN::NumericalDifferentiation::ForwardDifferences, LOG);

   nd.set_numerical_differentiation_method(OpenNN::NumericalDifferentiation::CentralDifferences);
   assert_true(nd.get_numerical_differentiation_method() == OpenNN::NumericalDifferentiation::CentralDifferences, LOG);

   nd.set_numerical_differentiation_method("ForwardDifferences");
   assert_true(nd.get_numerical_differentiation_method() == OpenNN::NumericalDifferentiation::ForwardDifferences, LOG);

   nd.set_numerical_differentiation_method("CentralDifferences");
   assert_true(nd.get_numerical_differentiation_method() == OpenNN::NumericalDifferentiation::CentralDifferences, LOG);

   // Test 4

   nd.set_default();

   assert_true(nd.get_numerical_differentiation_method() == OpenNN::NumericalDifferentiation::CentralDifferences, LOG);
   assert_true(nd.get_precision_digits() == 6, LOG);
   assert_true(nd.get_display() == true, LOG);
}

void NumericalDifferentiationTest::test_calculate_methods()
{
   cout << "test_calculate_methods\n";

   NumericalDifferentiation nd;
   NumericalDifferentiation nd_1;

   // Test 1

   nd.set_precision_digits(9);

//   assert_true(nd.calculate_eta() == 1e-9, LOG);
   assert_true(abs(nd.calculate_h(5) - static_cast<type>(0.000189)) < static_cast<type>(1e-5), LOG);

   // Test 2

   Tensor<type, 1> input(5);
   input.setValues({0,1,2,3,4});

   nd.set_precision_digits(3);

   assert_true(abs(nd.calculate_h(input)(0) - static_cast<type>(0.031)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(nd.calculate_h(input)(1) - static_cast<type>(0.063)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(nd.calculate_h(input)(2) - static_cast<type>(0.094)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(nd.calculate_h(input)(3) - static_cast<type>(0.126)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(nd.calculate_h(input)(4) - static_cast<type>(0.158)) < static_cast<type>(1e-3), LOG);

   // Test 3

   Tensor<type, 2> input_2d(2,2);
   input_2d.setValues({{0,1},{2,3}});

   assert_true(abs(nd.calculate_h(input_2d)(0,0) - static_cast<type>(0.031)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(nd.calculate_h(input_2d)(0,1) - static_cast<type>(0.063)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(nd.calculate_h(input_2d)(1,0) - static_cast<type>(0.094)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(nd.calculate_h(input_2d)(1,1) - static_cast<type>(0.126)) < static_cast<type>(1e-3), LOG);

   // Test 4

   Tensor<type, 1> input_4_0(4);
   Tensor<type, 1> input_4_1(4);
   input_4_0.setValues({1,2,3,4});
   input_4_1.setValues({1,4,9,16});

   assert_true(nd.calculate_backward_differences_derivatives(input_4_0, input_4_1)(0) - 0 < static_cast<type>(1e-5), LOG);
   assert_true(nd.calculate_backward_differences_derivatives(input_4_0, input_4_1)(1) - 3 < static_cast<type>(1e-5), LOG);
   assert_true(nd.calculate_backward_differences_derivatives(input_4_0, input_4_1)(2) - 5 < static_cast<type>(1e-5), LOG);
   assert_true(nd.calculate_backward_differences_derivatives(input_4_0, input_4_1)(3) - 7 < static_cast<type>(1e-5), LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_derivatives()
{
   cout << "test_calculate_forward_differences_derivative\n";

   NumericalDifferentiation nd;

   // Test 1

   type x = 1;

   type d1 = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f1, x);
   type d1_1 = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
   type d1_2 = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

   assert_true(abs(d1 - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_1 - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_2 - 3) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,2});

   Tensor<type,1> d2 = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(abs(d2(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 3

   Tensor<type,2>x_2d(1,2);
   x_2d.setValues({{1,2}});

   Tensor<type,2> d3 = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f3, x_2d);

   assert_true(abs(d3(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(0,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 4

   Index dummy_index = 3;

   Tensor<type,1> d4 = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_1d);

   assert_true(abs(d4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 12) < static_cast<type>(1e-2), LOG);

/*
   // Test 5

   Tensor<type,1> d5 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f5, dummy_index, x_1d);

   assert_true(abs(d5(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 12) < static_cast<type>(1e-2), LOG);
*/
}

void NumericalDifferentiationTest::test_calculate_central_differences_derivatives()
{
   cout << "test_calculate_central_differences_derivative\n";

   NumericalDifferentiation nd;

   // Test 1

   type x = 1;

   type d1 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f1, x);
   type d1_1 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
   type d1_2 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

   assert_true(abs(d1 - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_1 - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_2 - 3) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,2});

   Tensor<type,1> d2 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(abs(d2(0) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(d2(1) - 4) < static_cast<type>(1e-3), LOG);

   // Test 3

   Tensor<type,2>x_2d(1,2);
   x_2d.setValues({{1,2}});

   Tensor<type,2> d3 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f3, x_2d);

   assert_true(abs(d3(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(0,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 4

   Index dummy_index = 3;

   Tensor<type,1> d4 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_1d);

   assert_true(abs(d4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 12) < static_cast<type>(1e-2), LOG);

/*
   // Test 5

   Tensor<type,1> d5 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f5, dummy_index, x_1d);

   assert_true(abs(d5(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 12) < static_cast<type>(1e-2), LOG);
*/
}

void NumericalDifferentiationTest::test_calculate_derivatives()
{
   cout << "test_calculate_derivative\n";

   NumericalDifferentiation nd;

   // Test 1_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   type d = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f1, 0);

   assert_true(abs(d - 1) < static_cast<type>(1e-2), LOG);

   // Test 1_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f1, 0);

   assert_true(abs(d - 1) < static_cast<type>(1e-2), LOG);

   // Test 2_0

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   Tensor<type, 1> d_2 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(abs(d_2(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_2(1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 2_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d_2 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(abs(d_2(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_2(1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 3_0

   Tensor<type,2>x_2d(1,2);
   x_2d.setValues({{1,2}});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   Tensor<type, 2> d_3 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f3, x_2d);

   assert_true(abs(d_3(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_3(0,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 3_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d_3 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f3, x_2d);

   assert_true(abs(d_3(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_3(0,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 4_0

   Index dummy_index = 3;

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   Tensor<type, 1> d_4 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_1d);

   assert_true(abs(d_4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_4(1) - 12) < static_cast<type>(1e-2), LOG);

   // Test 4_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d_4 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_1d);

   assert_true(abs(d_4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_4(1) - 12) < static_cast<type>(1e-2), LOG);

/*
   // Test 5_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   Tensor<type,1> d5 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f5, dummy_index, x_1d);

   assert_true(abs(d5(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 12) < static_cast<type>(1e-2), LOG);

   // Test 5_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   Tensor<type,1> d5 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f5, dummy_index, x_1d);

   assert_true(abs(d5(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 12) < static_cast<type>(1e-2), LOG);
*/
}


void NumericalDifferentiationTest::test_calculate_forward_differences_second_derivatives()
{
   cout << "test_calculate_forward_differences_second_derivative\n";

   NumericalDifferentiation nd;

   // Test 1

   type x = 1;

   type d1 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);
   type d1_1 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
   type d1_2 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

   assert_true(abs(d1 - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_1 - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(d1_2 - 6) < static_cast<type>(1e-2), LOG);

   // Test 2
/*
   Tensor<type,1>x_2(2);
   x_2.setValues({1,2});

   Tensor<type,1> d2 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f2, x_2);

   assert_true(abs(d2(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 2) < static_cast<type>(1e-2), LOG);
*/

   // Test 3
/*
   Tensor<type,1>x_3(2);
   x_3.setValues({1,2});

   Index dummy_index = 1;

   Tensor<type,1> d3 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_3);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 2) < static_cast<type>(1e-2), LOG);
*/

   // Test
/*
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
   assert_true(matrix.dimension(1) == 15, LOG);*/
}

void NumericalDifferentiationTest::test_calculate_central_differences_second_derivatives()
{
   cout << "test_calculate_central_differences_second_derivative\n";
   NumericalDifferentiation nd;

   // Test 1

   type x = 1;

   type d1 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);
   type d1_1 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
   type d1_2 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

   assert_true(abs(d1 - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_1 - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(d1_2 - 6) < static_cast<type>(1e-1), LOG);

   // Test 2
/*
   Tensor<type,1>x_2(2);
   x_2.setValues({1,2});

   Tensor<type,1> d2 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f2, x_2);

   assert_true(abs(d2(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 2) < static_cast<type>(1e-2), LOG);
*/

   // Test 3
/*
   Tensor<type,1>x_3(2);
   x_3.setValues({1,2});

   Index dummy_index = 1;

   Tensor<type,1> d3 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_3);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 2) < static_cast<type>(1e-2), LOG);
*/
}

void NumericalDifferentiationTest::test_calculate_second_derivatives()
{
   cout << "test_calculate_second_derivative\n";

   NumericalDifferentiation nd;

   // Test 1_0

   type x = 1;

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   type d1 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);
   type d1_1 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
   type d1_2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

   assert_true(abs(d1 - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_1 - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(d1_2 - 6) < static_cast<type>(1e-2), LOG);

   // Test 1_1

   x = 1;

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d1 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);
   d1_1 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
   d1_2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

   assert_true(abs(d1 - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1_1 - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(d1_2 - 6) < static_cast<type>(1e-1), LOG);

   // Test 2_0
/*
   Tensor<type,1>x_2(2);
   x_2.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   Tensor<type,1> d2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f2, x_2);

   assert_true(abs(d2(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 2) < static_cast<type>(1e-2), LOG);
*/

   // Test 2_1
/*
   Tensor<type,1>x_2(2);
   x_2.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d2 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f2, x_2);

   assert_true(abs(d2(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 2) < static_cast<type>(1e-2), LOG);
*/

   // Test 3_0
/*
   Tensor<type,1>x_3(2);
   x_3.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   Index dummy_index = 1;

   Tensor<type,1> d3 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_3);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 2) < static_cast<type>(1e-2), LOG);
*/

   // Test 3_1
/*
   Tensor<type,1>x_3(2);
   x_3.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   Index dummy_index = 1;

   d3 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f4, dummy_index, x_3);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 2) < static_cast<type>(1e-2), LOG);
*/
}



void NumericalDifferentiationTest::test_calculate_forward_differences_gradient()
{
   cout << "test_calculate_forward_differences_gradient\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,2});

   Tensor<type, 1> d1 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f6, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type, 1> d2 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f6_, x_1d);

   assert_true(abs(d2(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 3

   Tensor<type,1>x3_1d(2);
   x3_1d.setValues({1,2});

   Tensor<type,1>dummy(2);
   dummy.setValues({2,3});

   Tensor<type, 1> d3 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f8, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 4

   Tensor<type, 1> d4 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f7, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 5

   Tensor<Index,1>dummy_5(2);
   dummy_5.setValues({2,3});

   Tensor<type, 1> d5 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f9, dummy_5, x3_1d);

   assert_true(abs(d5(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 3) < static_cast<type>(1e-2), LOG);
}

void NumericalDifferentiationTest::test_calculate_central_differences_gradient()
{
   cout << "test_calculate_central_differences_gradient\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,2});

   Tensor<type, 1> d1 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f6, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type, 1> d2 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f6_, x_1d);

   assert_true(abs(d2(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 3

   Tensor<type,1>x3_1d(2);
   x3_1d.setValues({1,2});

   Tensor<type,1>dummy(2);
   dummy.setValues({2,3});

   Tensor<type, 1> d3 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f8, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 4

   Tensor<type, 1> d4 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f7, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 5

   Tensor<Index,1>dummy_5(2);
   dummy_5.setValues({2,3});

   Tensor<type, 1> d5 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f9, dummy_5, x3_1d);

   assert_true(abs(d5(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 3) < static_cast<type>(1e-2), LOG);
}

void NumericalDifferentiationTest::test_calculate_training_loss_gradient()
{
   cout << "test_calculate_training_loss_gradient\n";

   NumericalDifferentiation nd;

   // Test 1_0

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d1 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f6, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 1_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d1 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f6, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d2 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f6_, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d2 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f6_, x_1d);

   assert_true(abs(d2(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 3_=

   Tensor<type,1>x3_1d(2);
   x3_1d.setValues({1,2});

   Tensor<type,1>dummy(2);
   dummy.setValues({2,3});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d3 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f8, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 3_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d3 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f8, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 4_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d4 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f7, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 4_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d4 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f7, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 5_0

   Tensor<Index,1>dummy_5(2);
   dummy_5.setValues({2,3});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d5 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f9, dummy_5, x3_1d);

   assert_true(abs(d5(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 5_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d5 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f9, dummy_5, x3_1d);

   assert_true(abs(d5(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 3) < static_cast<type>(1e-2), LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_hessian()
{
   cout << "test_calculate_forward_differences_gradient\n";
   /*
   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 2> H;

   // Test

   x.set(2, 0.0);
   H = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(H == 0.0, LOG);*/

}

void NumericalDifferentiationTest::test_calculate_central_differences_hessian()
{
   cout << "test_calculate_central_differences_gradient\n";
   /*
   NumericalDifferentiation nd;

   Tensor<type, 1> x;
   Tensor<type, 2> H;

   // Test

   x.set(2, 0.0);
   H = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f2, x);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(H == 0.0, LOG);*/
}

void NumericalDifferentiationTest::test_calculate_hessian()
{
   cout << "test_calculate_training_loss_gradient\n";
   /*
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

   assert_true(absolute_value(forward-central) < 1.0e-3, LOG);*/
}



/*
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

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   test_set_get_methods();
   test_calculate_methods();

   // Derivative methods

   test_calculate_forward_differences_derivatives();
   test_calculate_central_differences_derivatives();
   test_calculate_derivatives();


   // Second derivative methods

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
/*
   // Jacobian methods

   test_calculate_forward_differences_Jacobian();
   test_calculate_central_differences_Jacobian();
   test_calculate_Jacobian();

   // hessian methods

   test_calculate_forward_differences_hessian_form();
   test_calculate_central_differences_hessian_form();
   test_calculate_hessian_form();
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
