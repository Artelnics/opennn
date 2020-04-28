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

   Tensor<type,1> d4 = nd.calculate_forward_differences_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_1d);

   assert_true(abs(d4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 12) < static_cast<type>(1e-2), LOG);

/*
   // Test 5

   Tensor<type,1> d5 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f3_1, dummy_index, x_1d);

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

   Tensor<type,1> d4 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_1d);

   assert_true(abs(d4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 12) < static_cast<type>(1e-2), LOG);

/*
   // Test 5

   Tensor<type,1> d5 = nd.calculate_central_differences_derivatives(*this, &NumericalDifferentiationTest::f3_1, dummy_index, x_1d);

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

   Tensor<type, 1> d_4 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_1d);

   assert_true(abs(d_4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_4(1) - 12) < static_cast<type>(1e-2), LOG);

   // Test 4_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   d_4 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_1d);

   assert_true(abs(d_4(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d_4(1) - 12) < static_cast<type>(1e-2), LOG);

/*
   // Test 5_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);

   Tensor<type,1> d5 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f3_1, dummy_index, x_1d);

   assert_true(abs(d5(0) - 6) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 12) < static_cast<type>(1e-2), LOG);

   // Test 5_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   Tensor<type,1> d5 = nd.calculate_derivatives(*this, &NumericalDifferentiationTest::f3_1, dummy_index, x_1d);

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

   Tensor<type,1> d3 = nd.calculate_forward_differences_second_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_3);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 2) < static_cast<type>(1e-2), LOG);
*/
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

   Tensor<type,1> d3 = nd.calculate_central_differences_second_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_3);

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

   Tensor<type,1> d3 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_3);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 2) < static_cast<type>(1e-2), LOG);
*/

   // Test 3_1
/*
   Tensor<type,1>x_3(2);
   x_3.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

   Index dummy_index = 1;

   d3 = nd.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_3);

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

   Tensor<type, 1> d1 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f4, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type, 1> d2 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f4_1, x_1d);

   assert_true(abs(d2(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 3

   Tensor<type,1>x3_1d(2);
   x3_1d.setValues({1,2});

   Tensor<type,1>dummy(2);
   dummy.setValues({2,3});

   Tensor<type, 1> d3 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f4_3, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 4

   Tensor<type, 1> d4 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f4_2, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 5

   Tensor<Index,1>dummy_5(2);
   dummy_5.setValues({2,3});

   Tensor<type, 1> d5 = nd.calculate_forward_differences_gradient(*this, &NumericalDifferentiationTest::f4_4, dummy_5, x3_1d);

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

   Tensor<type, 1> d1 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f4, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type, 1> d2 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f4_1, x_1d);

   assert_true(abs(d2(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 3

   Tensor<type,1>x3_1d(2);
   x3_1d.setValues({1,2});

   Tensor<type,1>dummy(2);
   dummy.setValues({2,3});

   Tensor<type, 1> d3 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f4_3, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 4

   Tensor<type, 1> d4 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f4_2, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 5

   Tensor<Index,1>dummy_5(2);
   dummy_5.setValues({2,3});

   Tensor<type, 1> d5 = nd.calculate_central_differences_gradient(*this, &NumericalDifferentiationTest::f4_4, dummy_5, x3_1d);

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
   Tensor<type, 1> d1 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 1_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d1 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d2 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_1, x_1d);

   assert_true(abs(d1(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d1(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 2_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d2 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_1, x_1d);

   assert_true(abs(d2(0) - 1) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d2(1) - 1) < static_cast<type>(1e-2), LOG);

   // Test 3_=

   Tensor<type,1>x3_1d(2);
   x3_1d.setValues({1,2});

   Tensor<type,1>dummy(2);
   dummy.setValues({2,3});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d3 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_3, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 3_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d3 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_3, dummy, x3_1d);

   assert_true(abs(d3(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d3(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 4_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d4 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_2, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 4_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d4 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_2, 2, x3_1d);

   assert_true(abs(d4(0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d4(1) - 8) < static_cast<type>(1e-2), LOG);

   // Test 5_0

   Tensor<Index,1>dummy_5(2);
   dummy_5.setValues({2,3});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 1> d5 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_4, dummy_5, x3_1d);

   assert_true(abs(d5(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 3) < static_cast<type>(1e-2), LOG);

   // Test 5_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   d5 = nd.calculate_gradient(*this, &NumericalDifferentiationTest::f4_4, dummy_5, x3_1d);

   assert_true(abs(d5(0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(d5(1) - 3) < static_cast<type>(1e-2), LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_hessian()
{
   cout << "test_calculate_forward_differences_hessian\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   Tensor<type, 2> H = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f4_5, x_1d);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(abs(H(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(0,1) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,0) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 2

   Tensor<type,1>dummy(2);
   dummy.setValues({1,2});

   Tensor<type, 2> H2 = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f4_6, dummy, x_1d);

   assert_true(H2.dimension(0) == 2, LOG);
   assert_true(H2.dimension(1) == 2, LOG);
   assert_true(abs(H2(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(0,1) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 3

   Tensor<type, 2> H3 = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f4_7, 2, x_1d);

   assert_true(H3.dimension(0) == 2, LOG);
   assert_true(H3.dimension(1) == 2, LOG);
   assert_true(abs(H3(0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(0,1) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,0) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,1) - 4) < static_cast<type>(1e-1), LOG);
}

void NumericalDifferentiationTest::test_calculate_central_differences_hessian()
{
   cout << "test_calculate_central_differences_hessian\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   Tensor<type, 2> H = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f4_5, x_1d);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(abs(H(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(0,1) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,0) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 2

   Tensor<type,1>dummy(2);
   dummy.setValues({1,2});

   Tensor<type, 2> H2 = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f4_6, dummy, x_1d);

   assert_true(H2.dimension(0) == 2, LOG);
   assert_true(H2.dimension(1) == 2, LOG);
   assert_true(abs(H2(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(0,1) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 3

   Tensor<type, 2> H3 = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f4_7, 2, x_1d);

   assert_true(H3.dimension(0) == 2, LOG);
   assert_true(H3.dimension(1) == 2, LOG);
   assert_true(abs(H3(0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(0,1) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,0) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,1) - 4) < static_cast<type>(1e-1), LOG);
}

void NumericalDifferentiationTest::test_calculate_hessian()
{
   cout << "test_calculate_hessian\n";

   NumericalDifferentiation nd;

   // Test 1_0

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 2> H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f4_5, x_1d);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(abs(H(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(0,1) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,0) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 1_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f4_5, x_1d);

   assert_true(H.dimension(0) == 2, LOG);
   assert_true(H.dimension(1) == 2, LOG);
   assert_true(abs(H(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(0,1) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,0) - 1) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 2_0

   Tensor<type,1>dummy(2);
   dummy.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 2> H2 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f4_6, dummy, x_1d);

   assert_true(H2.dimension(0) == 2, LOG);
   assert_true(H2.dimension(1) == 2, LOG);
   assert_true(abs(H2(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(0,1) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 2_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   H2 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f4_6, dummy, x_1d);

   assert_true(H2.dimension(0) == 2, LOG);
   assert_true(H2.dimension(1) == 2, LOG);
   assert_true(abs(H2(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(0,1) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2(1,1) - 0) < static_cast<type>(1e-1), LOG);

   // Test 3_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   Tensor<type, 2> H3 = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f4_7, 2, x_1d);

   assert_true(H3.dimension(0) == 2, LOG);
   assert_true(H3.dimension(1) == 2, LOG);
   assert_true(abs(H3(0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(0,1) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,0) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,1) - 4) < static_cast<type>(1e-1), LOG);

   // Test 3_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   H3 = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f4_7, 2, x_1d);

   assert_true(H3.dimension(0) == 2, LOG);
   assert_true(H3.dimension(1) == 2, LOG);
   assert_true(abs(H3(0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(0,1) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,0) - 8) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3(1,1) - 4) < static_cast<type>(1e-1), LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_Jacobian()
{
   cout << "test_calculate_forward_differences_Jacobian\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   Tensor<type, 2> J = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(J.dimension(0) == 2, LOG);
   assert_true(J.dimension(1) == 2, LOG);
   assert_true(abs(J(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(1,1) - 2) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type,1>dummy(2);
   dummy.setValues({1,2});

   Tensor<type, 2> J2 = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(J2.dimension(0) == 2, LOG);
   assert_true(J2.dimension(1) == 2, LOG);
   assert_true(abs(J2(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(1,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 3

   Tensor<type, 2> J3 = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f2_1, 2, x_1d);

   assert_true(J3.dimension(0) == 2, LOG);
   assert_true(J3.dimension(1) == 2, LOG);
   assert_true(abs(J3(0,0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(1,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 4

   Tensor<type,1>dummy_vec(2);
   dummy_vec.setValues({-1,-2});

   Tensor<type, 2> J4 = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f5, 1, dummy_vec, x_1d);

   assert_true(J4.dimension(0) == 2, LOG);
   assert_true(J4.dimension(1) == 2, LOG);
   assert_true(abs(J4(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J4(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J4(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J4(1,1) + 2) < static_cast<type>(1e-1), LOG);

   // Test 5

   Tensor<type, 2> J5 = nd.calculate_forward_differences_Jacobian(*this, &NumericalDifferentiationTest::f5_1, 2, 5, x_1d);

   assert_true(J5.dimension(0) == 2, LOG);
   assert_true(J5.dimension(1) == 2, LOG);
   assert_true(abs(J5(0,0) - 10) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(1,1) - 10) < static_cast<type>(1e-1), LOG);
}

void NumericalDifferentiationTest::test_calculate_central_differences_Jacobian()
{
   cout << "test_calculate_central_differences_Jacobian\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   Tensor<type, 2> J = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(J.dimension(0) == 2, LOG);
   assert_true(J.dimension(1) == 2, LOG);
   assert_true(abs(J(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(1,1) - 2) < static_cast<type>(1e-2), LOG);

   // Test 2

   Tensor<type,1>dummy(2);
   dummy.setValues({1,2});

   Tensor<type, 2> J2 = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(J2.dimension(0) == 2, LOG);
   assert_true(J2.dimension(1) == 2, LOG);
   assert_true(abs(J2(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(1,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 3

   Tensor<type, 2> J3 = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f2_1, 2, x_1d);

   assert_true(J3.dimension(0) == 2, LOG);
   assert_true(J3.dimension(1) == 2, LOG);
   assert_true(abs(J3(0,0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(1,1) - 4) < static_cast<type>(1e-2), LOG);

   // Test 4

   Tensor<type,1>dummy_vec(2);
   dummy_vec.setValues({-1,-2});

   Tensor<type, 2> J4 = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f5, 1, dummy_vec, x_1d);

   assert_true(J4.dimension(0) == 2, LOG);
   assert_true(J4.dimension(1) == 2, LOG);
   assert_true(abs(J4(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J4(0,1) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J4(1,0) - 0) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J4(1,1) + 2) < static_cast<type>(1e-1), LOG);

   // Test 5

   Tensor<type, 2> J5 = nd.calculate_central_differences_Jacobian(*this, &NumericalDifferentiationTest::f5_1, 2, 5, x_1d);

   assert_true(J5.dimension(0) == 2, LOG);
   assert_true(J5.dimension(1) == 2, LOG);
   assert_true(abs(J5(0,0) - 10) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(1,1) - 10) < static_cast<type>(1e-1), LOG);
}

void NumericalDifferentiationTest::test_calculate_Jacobian()
{
   cout << "test_calculate_Jacobian\n";

   NumericalDifferentiation nd;

   // Test 1_0

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 2> J = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(J.dimension(0) == 2, LOG);
   assert_true(J.dimension(1) == 2, LOG);
   assert_true(abs(J(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 1_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   J = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(J.dimension(0) == 2, LOG);
   assert_true(J.dimension(1) == 2, LOG);
   assert_true(abs(J(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 2_0

   Tensor<type,1>dummy(2);
   dummy.setValues({1,2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 2> J2 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(J2.dimension(0) == 2, LOG);
   assert_true(J2.dimension(1) == 2, LOG);
   assert_true(abs(J2(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 2_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   J2 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(J2.dimension(0) == 2, LOG);
   assert_true(J2.dimension(1) == 2, LOG);
   assert_true(abs(J2(0,0) - 2) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J2(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 3_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 2> J3 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2_1, 2, x_1d);

   assert_true(J3.dimension(0) == 2, LOG);
   assert_true(J3.dimension(1) == 2, LOG);
   assert_true(abs(J3(0,0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 3_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   J3 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2_1, 2, x_1d);

   assert_true(J3.dimension(0) == 2, LOG);
   assert_true(J3.dimension(1) == 2, LOG);
   assert_true(abs(J3(0,0) - 4) < static_cast<type>(1e-2), LOG);
   assert_true(abs(J3(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 4_0

   Tensor<type,1>dummy_vec(2);
   dummy_vec.setValues({-1,-2});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 2> J4 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f5, 1, dummy_vec, x_1d);

   assert_true(J4.dimension(0) == 2, LOG);
   assert_true(J4.dimension(1) == 2, LOG);
   assert_true(abs(J4(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J4(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 4_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   J4 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f5, 1, dummy_vec, x_1d);

   assert_true(J4.dimension(0) == 2, LOG);
   assert_true(J4.dimension(1) == 2, LOG);
   assert_true(abs(J4(0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J4(1,0) - 0) < static_cast<type>(1e-2), LOG);

   // Test 5

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<type, 2> J5 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f5_1, 2, 5, x_1d);

   assert_true(J5.dimension(0) == 2, LOG);
   assert_true(J5.dimension(1) == 2, LOG);
   assert_true(abs(J5(0,0) - 10) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(1,1) - 10) < static_cast<type>(1e-1), LOG);

   // Test 5

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   J5 = nd.calculate_Jacobian(*this, &NumericalDifferentiationTest::f5_1, 2, 5, x_1d);

   assert_true(J5.dimension(0) == 2, LOG);
   assert_true(J5.dimension(1) == 2, LOG);
   assert_true(abs(J5(0,0) - 10) < static_cast<type>(1e-1), LOG);
   assert_true(abs(J5(1,0) - 0) < static_cast<type>(1e-1), LOG);
}


void NumericalDifferentiationTest::test_calculate_forward_differences_hessian_form()
{
   cout << "test_calculate_forward_differences_hessian_form\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   Tensor<Tensor<type, 2>, 1> H = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].dimension(0) == 2, LOG);
   assert_true(H[0].dimension(1) == 2, LOG);
   assert_true(abs(H[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H[1].dimension(0) == 2, LOG);
   assert_true(H[1].dimension(1) == 2, LOG);
   assert_true(abs(H[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 2

   Tensor<type,1>dummy(2);
   dummy.setValues({1,1});

   Tensor<Tensor<type, 2>, 1> H2 = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(H2.size() == 2, LOG);

   assert_true(H2[0].dimension(0) == 2, LOG);
   assert_true(H2[0].dimension(1) == 2, LOG);
   assert_true(abs(H2[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H2[1].dimension(0) == 2, LOG);
   assert_true(H2[1].dimension(1) == 2, LOG);
   assert_true(abs(H2[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 3

   Tensor<Tensor<type, 2>, 1> H3 = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f2_1, -1, x_1d);

   assert_true(H3.size() == 2, LOG);

   assert_true(H3[0].dimension(0) == 2, LOG);
   assert_true(H3[0].dimension(1) == 2, LOG);
   assert_true(abs(H3[0](0,0) + 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H3[1].dimension(0) == 2, LOG);
   assert_true(H3[1].dimension(1) == 2, LOG);
   assert_true(abs(H3[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](1,1) + 2) < static_cast<type>(1e-1), LOG);

   // Test 4

   Tensor<Tensor<type, 2>, 1> H4 = nd.calculate_forward_differences_hessian(*this, &NumericalDifferentiationTest::f5, 1, dummy, x_1d);

   assert_true(H4.size() == 2, LOG);

   assert_true(H4[0].dimension(0) == 2, LOG);
   assert_true(H4[0].dimension(1) == 2, LOG);
   assert_true(abs(H4[0](0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H4[1].dimension(0) == 2, LOG);
   assert_true(H4[1].dimension(1) == 2, LOG);
   assert_true(abs(H4[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](1,1) - 4) < static_cast<type>(1e-1), LOG);
}

void NumericalDifferentiationTest::test_calculate_central_differences_hessian_form()
{
   cout << "test_calculate_central_differences_hessian_form\n";

   NumericalDifferentiation nd;

   // Test 1

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   Tensor<Tensor<type, 2>, 1> H = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].dimension(0) == 2, LOG);
   assert_true(H[0].dimension(1) == 2, LOG);
   assert_true(abs(H[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H[1].dimension(0) == 2, LOG);
   assert_true(H[1].dimension(1) == 2, LOG);
   assert_true(abs(H[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 2

   Tensor<type,1>dummy(2);
   dummy.setValues({1,1});

   Tensor<Tensor<type, 2>, 1> H2 = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(H2.size() == 2, LOG);

   assert_true(H2[0].dimension(0) == 2, LOG);
   assert_true(H2[0].dimension(1) == 2, LOG);
   assert_true(abs(H2[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H2[1].dimension(0) == 2, LOG);
   assert_true(H2[1].dimension(1) == 2, LOG);
   assert_true(abs(H2[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 3

   Tensor<Tensor<type, 2>, 1> H3 = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f2_1, -1, x_1d);

   assert_true(H3.size() == 2, LOG);

   assert_true(H3[0].dimension(0) == 2, LOG);
   assert_true(H3[0].dimension(1) == 2, LOG);
   assert_true(abs(H3[0](0,0) + 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H3[1].dimension(0) == 2, LOG);
   assert_true(H3[1].dimension(1) == 2, LOG);
   assert_true(abs(H3[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](1,1) + 2) < static_cast<type>(1e-1), LOG);

   // Test 4

   Tensor<Tensor<type, 2>, 1> H4 = nd.calculate_central_differences_hessian(*this, &NumericalDifferentiationTest::f5, 1, dummy, x_1d);

   assert_true(H4.size() == 2, LOG);

   assert_true(H4[0].dimension(0) == 2, LOG);
   assert_true(H4[0].dimension(1) == 2, LOG);
   assert_true(abs(H4[0](0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](1,1) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H4[1].dimension(0) == 2, LOG);
   assert_true(H4[1].dimension(1) == 2, LOG);
   assert_true(abs(H4[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](1,1) - 4) < static_cast<type>(1e-1), LOG);
}

void NumericalDifferentiationTest::test_calculate_hessian_form()
{
   cout << "test_calculate_hessian_form\n";

   NumericalDifferentiation nd;

   // Test 1_0

   Tensor<type,1>x_1d(2);
   x_1d.setValues({1,1});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<Tensor<type, 2>, 1> H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].dimension(0) == 2, LOG);
   assert_true(H[0].dimension(1) == 2, LOG);
   assert_true(abs(H[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H[1].dimension(0) == 2, LOG);
   assert_true(H[1].dimension(1) == 2, LOG);
   assert_true(abs(H[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 1_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   H = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2, x_1d);

   assert_true(H.size() == 2, LOG);

   assert_true(H[0].dimension(0) == 2, LOG);
   assert_true(H[0].dimension(1) == 2, LOG);
   assert_true(abs(H[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H[1].dimension(0) == 2, LOG);
   assert_true(H[1].dimension(1) == 2, LOG);
   assert_true(abs(H[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 2_0

   Tensor<type,1>dummy(2);
   dummy.setValues({1,1});

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<Tensor<type, 2>, 1> H2 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(H2.size() == 2, LOG);

   assert_true(H2[0].dimension(0) == 2, LOG);
   assert_true(H2[0].dimension(1) == 2, LOG);
   assert_true(abs(H2[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H2[1].dimension(0) == 2, LOG);
   assert_true(H2[1].dimension(1) == 2, LOG);
   assert_true(abs(H2[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 2_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   H2 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

   assert_true(H2.size() == 2, LOG);

   assert_true(H2[0].dimension(0) == 2, LOG);
   assert_true(H2[0].dimension(1) == 2, LOG);
   assert_true(abs(H2[0](0,0) - 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H2[1].dimension(0) == 2, LOG);
   assert_true(H2[1].dimension(1) == 2, LOG);
   assert_true(abs(H2[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H2[1](1,1) - 2) < static_cast<type>(1e-1), LOG);

   // Test 3_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<Tensor<type, 2>, 1> H3 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2_1, -1, x_1d);

   assert_true(H3.size() == 2, LOG);

   assert_true(H3[0].dimension(0) == 2, LOG);
   assert_true(H3[0].dimension(1) == 2, LOG);
   assert_true(abs(H3[0](0,0) + 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H3[1].dimension(0) == 2, LOG);
   assert_true(H3[1].dimension(1) == 2, LOG);
   assert_true(abs(H3[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](1,1) + 2) < static_cast<type>(1e-1), LOG);

   // Test 3_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   H3 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f2_1, -1, x_1d);

   assert_true(H3.size() == 2, LOG);

   assert_true(H3[0].dimension(0) == 2, LOG);
   assert_true(H3[0].dimension(1) == 2, LOG);
   assert_true(abs(H3[0](0,0) + 2) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H3[1].dimension(0) == 2, LOG);
   assert_true(H3[1].dimension(1) == 2, LOG);
   assert_true(abs(H3[1](0,1) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H3[1](1,1) + 2) < static_cast<type>(1e-1), LOG);

   // Test 4_0

   nd.set_numerical_differentiation_method(NumericalDifferentiation::ForwardDifferences);
   Tensor<Tensor<type, 2>, 1> H4 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f5, 1, dummy, x_1d);

   assert_true(H4.size() == 2, LOG);

   assert_true(H4[0].dimension(0) == 2, LOG);
   assert_true(H4[0].dimension(1) == 2, LOG);
   assert_true(abs(H4[0](0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H4[1].dimension(0) == 2, LOG);
   assert_true(H4[1].dimension(1) == 2, LOG);
   assert_true(abs(H4[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](1,0) - 0) < static_cast<type>(1e-1), LOG);

   // Test 4_1

   nd.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
   H4 = nd.calculate_hessian(*this, &NumericalDifferentiationTest::f5, 1, dummy, x_1d);

   assert_true(H4.size() == 2, LOG);

   assert_true(H4[0].dimension(0) == 2, LOG);
   assert_true(H4[0].dimension(1) == 2, LOG);
   assert_true(abs(H4[0](0,0) - 4) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[0](1,0) - 0) < static_cast<type>(1e-1), LOG);

   assert_true(H4[1].dimension(0) == 2, LOG);
   assert_true(H4[1].dimension(1) == 2, LOG);
   assert_true(abs(H4[1](0,0) - 0) < static_cast<type>(1e-1), LOG);
   assert_true(abs(H4[1](1,0) - 0) < static_cast<type>(1e-1), LOG);
}

void NumericalDifferentiationTest::test_calculate_central_differences_gradient_matrix()
{
    cout << "test_calculate_central_differences_gradient_matrix\n";

    NumericalDifferentiation nd;

    Tensor<type,2>x_2d(2,2);
    x_2d.setValues({{1,2},{-1,-2}});
/*
    Tensor<type, 2> d1 = nd.calculate_central_differences_gradient_matrix(*this, &NumericalDifferentiationTest::f2_2,-1, x_2d);

    cout << "central_differences_gradient_matrix" << endl;
    cout << d1 << endl;

    assert_true(abs(d1(0,0) - 1) < static_cast<type>(1e-2), LOG);
    assert_true(abs(d1(0,1) - 1) < static_cast<type>(1e-2), LOG);
*/
}

void NumericalDifferentiationTest::test_calculate_central_differences_hessian_matrices()
{
    cout << "test_calculate_central_differences_hessian_matrices\n";

    NumericalDifferentiation nd;

    Tensor<type,1>dummy(2);
    dummy.setValues({-1,-1});

    Tensor<type,1>x_1d(2);
    x_1d.setValues({3,4});
/*
    Tensor<type, 1> H = nd.calculate_central_differences_hessian_matrices(*this, &NumericalDifferentiationTest::f5, 2, dummy, x_1d);

    cout << "H" << endl;
    cout << H << endl;

    assert_true(abs(H(0) - 1) < static_cast<type>(1e-2), LOG);
    assert_true(abs(H(1) - 1) < static_cast<type>(1e-2), LOG);
*/
}

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

   test_calculate_central_differences_gradient_matrix();

   // hessian methods

   test_calculate_forward_differences_hessian();
   test_calculate_central_differences_hessian();
   test_calculate_hessian();

   // Jacobian methods

   test_calculate_forward_differences_Jacobian();
   test_calculate_central_differences_Jacobian();
   test_calculate_Jacobian();

   // hessian methods

   test_calculate_forward_differences_hessian_form();
   test_calculate_central_differences_hessian_form();
   test_calculate_hessian_form();

   test_calculate_central_differences_hessian_matrices();

   cout << "End of numerical differentiation test case.\n";
}



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
