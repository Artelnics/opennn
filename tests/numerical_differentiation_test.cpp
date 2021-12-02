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

    NumericalDifferentiation nd_1(numerical_differentiation);

    assert_true(nd_1.get_precision_digits()== 6, LOG);
}


void NumericalDifferentiationTest::test_destructor()
{
    cout << "test_destructor\n";

    NumericalDifferentiation* nd_1 = new NumericalDifferentiation(numerical_differentiation);

    delete nd_1;
}


void NumericalDifferentiationTest::test_set_get_methods()
{
    cout << "test_set_methods\n";

    // Test

    numerical_differentiation.set_precision_digits(9);
    numerical_differentiation.set_display(true);

    assert_true(numerical_differentiation.get_precision_digits() == 9, LOG);
    assert_true(numerical_differentiation.get_display(), LOG);

    // Test

    numerical_differentiation.set_default();

    assert_true(numerical_differentiation.get_precision_digits() == 6, LOG);
    assert_true(numerical_differentiation.get_display(), LOG);
}


void NumericalDifferentiationTest::test_calculate_methods()
{
    cout << "test_calculate_methods\n";

    // Test

    numerical_differentiation.set_precision_digits(9);

    assert_true(numerical_differentiation.calculate_eta() == type(1e-9), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(type(5)) - static_cast<type>(0.000189)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    Tensor<type, 1> input(5);
    input.setValues({ type(0),type(1),type(2),type(3),type(4)});

    numerical_differentiation.set_precision_digits(3);

    assert_true(abs(numerical_differentiation.calculate_h(input)(0) - static_cast<type>(0.031)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(input)(1) - static_cast<type>(0.063)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(input)(2) - static_cast<type>(0.094)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(input)(3) - static_cast<type>(0.126)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(input)(4) - static_cast<type>(0.158)) < static_cast<type>(1e-3), LOG);

    // Test

    Tensor<type, 2> input_2d(2,2);

    input_2d.setValues({{type(0),type(1)},{type(2),type(3)}});

    assert_true(abs(numerical_differentiation.calculate_h(input_2d)(0,0) - static_cast<type>(0.031)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(input_2d)(0,1) - static_cast<type>(0.063)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(input_2d)(1,0) - static_cast<type>(0.094)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(numerical_differentiation.calculate_h(input_2d)(1,1) - static_cast<type>(0.126)) < static_cast<type>(1e-3), LOG);

}


void NumericalDifferentiationTest::test_calculate_derivatives()
{
    cout << "test_calculate_derivatives\n";

    // Test

    type x = type(1);

    type derivatives = numerical_differentiation.calculate_derivatives(*this, &NumericalDifferentiationTest::f1, x);
    type derivatives_1 = numerical_differentiation.calculate_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
    type derivatives_2 = numerical_differentiation.calculate_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

    assert_true(abs(derivatives - type(1)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(derivatives_1 - type(2)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(derivatives_2 - type(3)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 1> x_1d(2);
    x_1d.setValues({type(1),type(2)});

    Tensor<type, 1> d2 = numerical_differentiation.calculate_derivatives(*this, &NumericalDifferentiationTest::f2, x_1d);

    assert_true(abs(d2(0) - type(2)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(d2(1) - type(4)) < static_cast<type>(1e-3), LOG);

    // Test

    Tensor<type,2>x_2d(1,2);
    x_2d.setValues({{type(1),type(2)}});

    Tensor<type, 2> d3 = numerical_differentiation.calculate_derivatives(*this, &NumericalDifferentiationTest::f3, x_2d);

    assert_true(abs(d3(0,0) - type(2)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(d3(0,1) - type(4)) < static_cast<type>(1e-2), LOG);

    // Test

    Index dummy_index = 3;

    Tensor<type, 1> d4 = numerical_differentiation.calculate_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_1d);

    assert_true(abs(d4(0) - type(6)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(d4(1) - type(12)) < static_cast<type>(1e-2), LOG);

    // Test 5

//    Tensor<type, 1> d5 = nunmerical_differentiation.calculate_derivatives(*this, &NumericalDifferentiationTest::f3_1, dummy_index, x_1d);

//    assert_true(abs(d5(0) - type(6)) < static_cast<type>(1e-2), LOG);
//    assert_true(abs(d5(1) - type(12)) < static_cast<type>(1e-2), LOG);
}


void NumericalDifferentiationTest::test_calculate_second_derivatives()
{
    cout << "test_calculate_second_derivative\n";

    // Test

    type x = type(1);

    type derivatives = numerical_differentiation.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1, x);
    type derivatives_1 = numerical_differentiation.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1_1, x);
    type derivatives_2 = numerical_differentiation.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f1_2, x);

    assert_true(abs(derivatives - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(derivatives_1 - type(2)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(derivatives_2 - 6) < static_cast<type>(1e-1), LOG);

    // Test

    Tensor<type, 1> x_2(2);
    x_2.setValues({type(1),type(2)});

    Tensor<type, 1> d2 = numerical_differentiation.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f2, x_2);

    assert_true(abs(d2(0) - type(2)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(d2(1) - type(2)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 1> x_3(2);
    x_3.setValues({type(1),type(2)});

    Index dummy_index = 1;

    Tensor<type, 1> d3 = numerical_differentiation.calculate_second_derivatives(*this, &NumericalDifferentiationTest::f2_1, dummy_index, x_3);

    assert_true(abs(d3(0) - type(2)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(d3(1) - type(2)) < static_cast<type>(1e-2), LOG);
}


void NumericalDifferentiationTest::test_calculate_gradient()
{
    cout << "test_calculate_gradient\n";

    // Test

    Tensor<type, 1> x_1d(2);
    x_1d.setValues({type(1),type(2)});

    Tensor<type, 1> derivatives = numerical_differentiation.calculate_gradient(*this, &NumericalDifferentiationTest::f4, x_1d);

    assert_true(abs(derivatives(0) - type(1)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(derivatives(1) - type(1)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 1> d2 = numerical_differentiation.calculate_gradient(*this, &NumericalDifferentiationTest::f4_1, x_1d);

    assert_true(abs(d2(0) - type(1)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(d2(1) - type(1)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 1> x3_1d(2);
    x3_1d.setValues({type(1),type(2)});

    Tensor<type, 1>dummy(2);
    dummy.setValues({type(2),type(3)});

//    Tensor<type, 1> d3 = numerical_differentiation.calculate_gradient(*this, &NumericalDifferentiationTest::f4_3, dummy, x3_1d);

//    assert_true(abs(d3(0) - type(2)) < static_cast<type>(1e-2), LOG);
//    assert_true(abs(d3(1) - type(3)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 1> d4 = numerical_differentiation.calculate_gradient(*this, &NumericalDifferentiationTest::f4_2, 2, x3_1d);

    assert_true(abs(d4(0) - type(4)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(d4(1) - type(8)) < static_cast<type>(1e-2), LOG);

    // Test 5

//   Tensor<type,1>dummy_5(2);
//   dummy_5.setValues({type(2),type(3)});

//   Tensor<type, 1> d5 = numerical_differentiation.calculate_gradient(*this, &NumericalDifferentiationTest::f4_4, dummy_5, x3_1d);

//   assert_true(abs(d5(0) - type(2)) < static_cast<type>(1e-2), LOG);
//   assert_true(abs(d5(1) - type(3)) < static_cast<type>(1e-2), LOG);
}


void NumericalDifferentiationTest::test_calculate_gradient_matrix()
{
//    cout << "test_calculate_gradient_matrix\n";

//    Tensor<type,2>x_2d(2,2);
//    x_2d.setValues({{1,2},{-1,-2}});

//    Tensor<type, 2> derivatives = numerical_differentiation.calculate_gradient_matrix(*this, &NumericalDifferentiationTest::f2_2,-1, x_2d);

//    assert_true(abs(derivatives(0,0) - type(1)) < static_cast<type>(1e-2), LOG);
//    assert_true(abs(derivatives(0,1) - type(1)) < static_cast<type>(1e-2), LOG);
}


void NumericalDifferentiationTest::test_calculate_hessian()
{
    cout << "test_calculate_hessian\n";

    // Test

    Tensor<type, 1> x_1d(2);
    x_1d.setValues({type(1),type(1)});

    Tensor<type, 2> H = numerical_differentiation.calculate_hessian(*this, &NumericalDifferentiationTest::f4_5, x_1d);

    assert_true(H.dimension(0) == 2, LOG);
    assert_true(H.dimension(1) == 2, LOG);
    assert_true(abs(H(0,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H(0,1) - type(1)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H(1,0) - type(1)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H(1,1) - type(0)) < static_cast<type>(1e-1), LOG);

    // Test

    Tensor<type, 1>dummy(2);
    dummy.setValues({type(1),type(2)});

    Tensor<type, 2> H2 = numerical_differentiation.calculate_hessian(*this, &NumericalDifferentiationTest::f4_6, dummy, x_1d);

    assert_true(H2.dimension(0) == 2, LOG);
    assert_true(H2.dimension(1) == 2, LOG);
    assert_true(abs(H2(0,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2(0,1) - type(2)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2(1,0) - type(2)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2(1,1) - type(0)) < static_cast<type>(1e-1), LOG);

    // Test

    Tensor<type, 2> H3 = numerical_differentiation.calculate_hessian(*this, &NumericalDifferentiationTest::f4_7, 2, x_1d);

    assert_true(H3.dimension(0) == 2, LOG);
    assert_true(H3.dimension(1) == 2, LOG);
    assert_true(abs(H3(0,0) - type(4)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3(0,1) - type(8)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3(1,0) - type(8)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3(1,1) - type(4)) < static_cast<type>(1e-1), LOG);
}


void NumericalDifferentiationTest::test_calculate_hessian_form()
{
    cout << "test_calculate_hessian_form\n";

    // Test

    Tensor<type, 1> x_1d(2);
    x_1d.setValues({type(1),type(1)});

    Tensor<Tensor<type, 2>, 1> H = numerical_differentiation.calculate_hessian(*this, &NumericalDifferentiationTest::f2, x_1d);

    assert_true(H.size() == 2, LOG);

    assert_true(H[0].dimension(0) == 2, LOG);
    assert_true(H[0].dimension(1) == 2, LOG);
    assert_true(abs(H[0](0,0) - type(2)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H[0](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H[0](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H[0](1,1) - type(0)) < static_cast<type>(1e-1), LOG);

    assert_true(H[1].dimension(0) == 2, LOG);
    assert_true(H[1].dimension(1) == 2, LOG);
    assert_true(abs(H[1](0,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H[1](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H[1](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H[1](1,1) - type(2)) < static_cast<type>(1e-1), LOG);

    // Test

    Tensor<type, 1>dummy(2);
    dummy.setValues({type(1),type(1)});

    Tensor<Tensor<type, 2>, 1> H2
            = numerical_differentiation.calculate_hessian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

    assert_true(H2.size() == 2, LOG);

    assert_true(H2[0].dimension(0) == 2, LOG);
    assert_true(H2[0].dimension(1) == 2, LOG);
    assert_true(abs(H2[0](0,0) - type(2)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2[0](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2[0](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2[0](1,1) - type(0)) < static_cast<type>(1e-1), LOG);

    assert_true(H2[1].dimension(0) == 2, LOG);
    assert_true(H2[1].dimension(1) == 2, LOG);
    assert_true(abs(H2[1](0,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2[1](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2[1](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H2[1](1,1) - type(2)) < static_cast<type>(1e-1), LOG);

    // Test

    Tensor<Tensor<type, 2>, 1> H3
            = numerical_differentiation.calculate_hessian(*this, &NumericalDifferentiationTest::f2_1, -1, x_1d);

    assert_true(H3.size() == 2, LOG);

    assert_true(H3[0].dimension(0) == 2, LOG);
    assert_true(H3[0].dimension(1) == 2, LOG);
    assert_true(abs(H3[0](0,0) + type(2)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3[0](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3[0](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3[0](1,1) - type(0)) < static_cast<type>(1e-1), LOG);

    assert_true(H3[1].dimension(0) == 2, LOG);
    assert_true(H3[1].dimension(1) == 2, LOG);
    assert_true(abs(H3[1](0,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3[1](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3[1](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H3[1](1,1) + type(2)) < static_cast<type>(1e-1), LOG);

    // Test

    Tensor<Tensor<type, 2>, 1> H4
            = numerical_differentiation.calculate_hessian(*this, &NumericalDifferentiationTest::f5, 1, dummy, x_1d);

    assert_true(H4.size() == 2, LOG);

    assert_true(H4[0].dimension(0) == 2, LOG);
    assert_true(H4[0].dimension(1) == 2, LOG);
    assert_true(abs(H4[0](0,0) - type(4)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H4[0](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H4[0](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H4[0](1,1) - type(0)) < static_cast<type>(1e-1), LOG);

    assert_true(H4[1].dimension(0) == 2, LOG);
    assert_true(H4[1].dimension(1) == 2, LOG);
    assert_true(abs(H4[1](0,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H4[1](0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H4[1](1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(H4[1](1,1) - type(4)) < static_cast<type>(1e-1), LOG);
}


void NumericalDifferentiationTest::test_calculate_hessian_matrices()
{
    cout << "test_calculate_hessian_matrices\n";

    Tensor<type, 1>dummy(2);
    dummy.setValues({type(-1),type(-1)});

    Tensor<type, 1> x_1d(2);
    x_1d.setValues({type(3),type(4)});

//    Tensor<type, 1> H = numerical_differentiation.calculate_hessian_matrices(*this, &NumericalDifferentiationTest::f5, 2, dummy, x_1d);

//    assert_true(abs(H(0) - type(1)) < static_cast<type>(1e-2), LOG);
//    assert_true(abs(H(1) - type(1)) < static_cast<type>(1e-2), LOG);
}

void NumericalDifferentiationTest::test_calculate_Jacobian()
{
    cout << "test_calculate_Jacobian\n";

    // Test

    Tensor<type, 1> x_1d(2);
    x_1d.setValues({type(1),type(1)});

    Tensor<type, 2> J = numerical_differentiation.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2, x_1d);

    assert_true(J.dimension(0) == 2, LOG);
    assert_true(J.dimension(1) == 2, LOG);
    assert_true(abs(J(0,0) - type(2)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J(0,1) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J(1,0) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J(1,1) - type(2)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 1>dummy(2);
    dummy.setValues({type(1),type(2)});

    Tensor<type, 2> J2 = numerical_differentiation.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2_2, dummy, x_1d);

    assert_true(J2.dimension(0) == 2, LOG);
    assert_true(J2.dimension(1) == 2, LOG);
    assert_true(abs(J2(0,0) - type(2)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J2(0,1) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J2(1,0) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J2(1,1) - type(4)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 2> J3 = numerical_differentiation.calculate_Jacobian(*this, &NumericalDifferentiationTest::f2_1, 2, x_1d);

    assert_true(J3.dimension(0) == 2, LOG);
    assert_true(J3.dimension(1) == 2, LOG);
    assert_true(abs(J3(0,0) - type(4)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J3(0,1) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J3(1,0) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J3(1,1) - type(4)) < static_cast<type>(1e-2), LOG);

    // Test

    Tensor<type, 1>dummy_vec(2);
    dummy_vec.setValues({ type(-1),type(-2)});

    Tensor<type, 2> J4 = numerical_differentiation.calculate_Jacobian(*this, &NumericalDifferentiationTest::f5, 1, dummy_vec, x_1d);

    assert_true(J4.dimension(0) == 2, LOG);
    assert_true(J4.dimension(1) == 2, LOG);
    assert_true(abs(J4(0,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(J4(0,1) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J4(1,0) - type(0)) < static_cast<type>(1e-2), LOG);
    assert_true(abs(J4(1,1) + type(2)) < static_cast<type>(1e-1), LOG);

    // Test 5

    Tensor<type, 2> J5 = numerical_differentiation.calculate_Jacobian(*this, &NumericalDifferentiationTest::f5_1, 2, 5, x_1d);

    assert_true(J5.dimension(0) == 2, LOG);
    assert_true(J5.dimension(1) == 2, LOG);
    assert_true(abs(J5(0,0) - type(10)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(J5(0,1) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(J5(1,0) - type(0)) < static_cast<type>(1e-1), LOG);
    assert_true(abs(J5(1,1) - type(10)) < static_cast<type>(1e-1), LOG);
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

    test_calculate_derivatives();

    // Second derivative methods

    test_calculate_second_derivatives();

    // Gradient methods

    test_calculate_gradient();

    test_calculate_gradient_matrix();

    // hessian methods

    test_calculate_hessian();

    test_calculate_hessian_form();

    test_calculate_hessian_matrices();

    // Jacobian methods

    test_calculate_Jacobian();

    cout << "End of numerical differentiation test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
