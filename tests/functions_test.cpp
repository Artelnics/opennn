//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F U N C T I O N S   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   E-mail: artelnics@artelnics.com

#include "functions_test.h"

FunctionsTest::FunctionsTest() : UnitTesting()
{
}


FunctionsTest::~FunctionsTest()
{
}


void FunctionsTest::test_constructor()
{
   cout << "test_constructor\n";

}


void FunctionsTest::test_destructor()
{
   cout << "test_destructor\n";
}


void FunctionsTest::test_factorial()
{
    cout << "tets_factorial\n";

    //Trivial case

    double solution = 1;

    assert_true(abs(factorial(0) - solution) <= 10e-3, LOG);

    //Random numbers

    solution = 120;

    assert_true(abs(factorial(5) - solution) <= 10e-3, LOG);

    solution = 3628800;

    assert_true(abs(factorial(10) - solution) <= 10e-3, LOG);

}


void FunctionsTest::test_exponential()
{
    cout << "test_exponential\n";

    // Trivial case

    const Vector<double> exp = exponential({0});
    assert_true( abs(exp[0] - 1) <= 10e-3, LOG);

    //Random numbers

    const Vector<double> solution({0.000911882,162754.7914});
    const Vector<double> exp2 = exponential({-7, 12});

    assert_true(abs(exp2[0] - solution[0]) <= 10e-3, LOG);
    assert_true(abs(exp2[1] - solution[1]) <= 10e-3, LOG);
}


void FunctionsTest::test_logarithm()
{
    cout << "test_logarithm\n";

    // Trivial case

    const Vector<double> log = logarithm({0});
    //assert_true( abs(log[0]) <= 10e-3, LOG);

    //Random numbers

    const Vector<double> solution({2.079181246,0.954242509});
    const Vector<double> log2 = logarithm({10, 9});

//    assert_true(abs(log2[0] - solution[0]) <= 10e-3, LOG);
//    assert_true(abs(log2[1] - solution[1]) <= 10e-3, LOG);
}


void FunctionsTest::test_power()
{
    cout << "test_power\n";

    // Trivial case

     Vector<double> pow = power({1},0);
    assert_true( abs(pow[0] - 1) <= 10e-3, LOG);

    //Random numbers

    const Vector<double> solution({10.95445115,3.99925E-06,169});
    const Vector<double> pow2= power({120},0.5);
    const Vector<double> pow3= power({63},-3);
    const Vector<double> pow4= power({13},2);

    assert_true(abs(pow2[0] - solution[0]) <= 10e-3, LOG);
    assert_true(abs(pow3[0] - solution[1]) <= 10e-3, LOG);
    assert_true(abs(pow4[0] - solution[2]) <= 10e-3, LOG);
}


void FunctionsTest::test_binary()
{
    cout << "test_binary\n";

    const Vector<double> vector({0.3, 1.7, 0.1, 5});

    const Vector<bool> solution({0, 1, 0, 1});

    assert_true(binary(vector) == solution, LOG);
}


void FunctionsTest::test_square_root()
{
    cout << "test_square_root\n";

    const Vector<double> vector({ 100, 25, 36});
    const Vector<double> solution({10, 5, 6});

    assert_true( square_root(vector) - solution == 0, LOG);
}


void FunctionsTest::test_cumulative()
{
    cout << "test_cumulative\n";

    const Vector<double> vector({1, 2, 3, 4});
    const Vector<double> solution({1, 3, 6, 10});

    assert_true(cumulative(vector) == solution, LOG);
}


void FunctionsTest::test_lower_bounded()
{
    cout << "test_lower_bounded\n";

    //Vector-number case

    const Vector<double> vector({4, 5, 6, 7, 8, 12, 15});
    const Vector<double> solution({7, 7, 7, 7, 8, 12, 15});

    assert_true(lower_bounded(vector, 7) == solution, LOG);

    // Vector-vector case

    const Vector<double> vector2({8, 6, 9, 4, 3, 12, 4});
    const Vector<double> solution2({8, 6, 9, 7, 8, 12, 15});

    assert_true(lower_bounded(vector,vector2) == solution2, LOG);

    //Matrix-number case

    Matrix<double> matrix(3,3);

    matrix.set_column(0, {5, 7, 13});
    matrix.set_column(1, {4, 6, 21});
    matrix.set_column(2, {61, 2, -7});

    Matrix<double> sol(3,3);

    sol.set_column(0, {10, 10, 13});
    sol.set_column(1, {10, 10, 21});
    sol.set_column(2, {61, 10, 10});

    assert_true( lower_bounded(matrix,10) == sol, LOG);
}


void FunctionsTest::test_upper_bounded()
{
    cout << "test_upper_bounded\n";

    //Vector-number case

    const Vector<double> vector({4, 5, 6, 7, 8, 12, 15});
    const Vector<double> solution({4, 5, 6, 7, 7, 7, 7});

    assert_true(upper_bounded(vector, 7) == solution, LOG);

    //Vector-vector case

    const Vector<double> vector2({8, 6, 9, 4, 3, 12, 4});
    const Vector<double> solution2({4, 5, 6, 4, 3, 12, 4});

    assert_true(upper_bounded(vector,vector2) == solution2, LOG);

    //Matrix-number case

    Matrix<double> matrix(3,3);

    matrix.set_column(0, {5, 7, 13});
    matrix.set_column(1, {4, 6, 21});
    matrix.set_column(2, {61, 2, -7});

    Matrix<double> sol(3,3);

    sol.set_column(0, {5, 7, 10});
    sol.set_column(1, {4, 6, 10});
    sol.set_column(2, {10, 2, -7});

    assert_true( upper_bounded(matrix,10) == sol, LOG);
}


void FunctionsTest::test_lower_upper_bounded()
{
    cout << "test_lower_upper_bounded\n";

    // Vector-number case

    const Vector<double> vector({4, 5, 6, 7, 8, 12, 15});
    const Vector<double> solution({5, 5, 6, 7, 8, 12, 13});

    assert_true(lower_upper_bounded(vector, 5,13) == solution, LOG);

    //Vector-vector case

    const Vector<double> vector2({5, 5, 5, 5, 5, 5, 5});
    const Vector<double> solution2({5, 5, 6, 7, 8, 12, 12});
    const Vector<double> vector3({12, 12, 12, 12, 12, 12, 12});

    assert_true(lower_upper_bounded(vector,vector2,vector3) == solution2, LOG);

    //Matrix case

    Matrix<double> matrix(3,3);

    matrix.set_column(0, {5, 7, 13});
    matrix.set_column(1, {4, 6, 21});
    matrix.set_column(2, {61, 2, -7});

    Matrix<double> sol(3,3);

    sol.set_column(0, {5, 7, 10});
    sol.set_column(1, {4, 6, 10});
    sol.set_column(2, {10, 3, 3});

    assert_true( lower_upper_bounded(matrix,3,10) == sol, LOG);
}


void FunctionsTest::test_threshold()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(Vector<size_t>({4,1}));
    tensor2.set(Vector<size_t>({4,1}));

    tensor1[0] = -1;
    tensor1[1] = 1;
    tensor1[2] = 2;
    tensor1[3] = -2;

    tensor2 = threshold(tensor1);

    assert_true(abs(tensor2[0] - 0) < 0.001, LOG);
    assert_true(abs(tensor2[1] - 1) < 0.001, LOG);
    assert_true(abs(tensor2[2] - 1) < 0.001, LOG);
    assert_true(abs(tensor2[3] - 0) < 0.001, LOG);

}

void FunctionsTest::test_symmetric_threshold()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(4);
    tensor2.set(4);

    tensor1[0] = -1;
    tensor1[1] = 1;
    tensor1[2] = 2;
    tensor1[3] = -2;

    tensor2 = symmetric_threshold(tensor1);

    assert_true(abs(tensor2[0] - -1) < 0.001, LOG);
    assert_true(abs(tensor2[1] - 1) < 0.001, LOG);
    assert_true(abs(tensor2[2] - 1) < 0.001, LOG);
    assert_true(abs(tensor2[3] - -1) < 0.001, LOG);

}

void FunctionsTest::test_threshold_second_derivatives()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(2);
    tensor2.set(2);

    tensor1[0] = 2;
    tensor1[1] = -2;

    tensor2 = threshold_derivatives(tensor1);

    assert_true(abs(tensor1[0] - 0) < 0.000001, LOG);
    assert_true(abs(tensor1[1] - 0) < 0.000001, LOG);

}

void FunctionsTest::test_symmetric_threshold_derivatives()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(2);
    tensor2.set(2);

    tensor1[0] = 2;
    tensor1[1] = -2;

    tensor2 = threshold_derivatives(tensor1);

    assert_true(abs(tensor2[0] - 0) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - 0) < 0.000001, LOG);

}


void FunctionsTest::test_logistic()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(4);
    tensor2.set(4);

    tensor1[0] = -1;
    tensor1[1] = 1;
    tensor1[2] = 2;
    tensor1[3] = -2;

    tensor2 = logistic(tensor1);

    assert_true(abs(tensor2[0] - 0.268941) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - 0.731059) < 0.000001, LOG);
    assert_true(abs(tensor2[2] - 0.880797) < 0.000001, LOG);
    assert_true(abs(tensor2[3] - 0.119203) < 0.000001, LOG);

}


void FunctionsTest::test_hyperbolic_tangent()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(4);
    tensor2.set(4);

    tensor1[0] = -1;
    tensor1[1] = 1;
    tensor1[2] = 2;
    tensor1[3] = -2;

    tensor2 = hyperbolic_tangent(tensor1);

    assert_true(abs(tensor2[0] - -0.761594) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - 0.761594) < 0.000001, LOG);
    assert_true(abs(tensor2[2] - 0.964028) < 0.000001, LOG);
    assert_true(abs(tensor2[3] - -0.964028) < 0.000001, LOG);

}


void FunctionsTest::test_hyperbolic_tangent_derivatives()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(4);
    tensor2.set(4);

    tensor1[0] = -1;
    tensor1[1] = 1;
    tensor1[2] = 2;
    tensor1[3] = -2;

    tensor2 = hyperbolic_tangent_derivatives(tensor1);

    assert_true(abs(tensor2[0] - 0.419974) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - 0.419974) < 0.000001, LOG);
    assert_true(abs(tensor2[2] - 0.070651) < 0.000001, LOG);
    assert_true(abs(tensor2[3] - 0.070651) < 0.000001, LOG);

}

void FunctionsTest::test_logistic_derivatives()
{

    Tensor<double> tensor1;
    Vector<double> tensor2;

    tensor1.set(4);
    tensor2.set(4);

    tensor1[0] = -1;
    tensor1[1] = 1;
    tensor1[2] = 2;
    tensor1[3] = -2;

    tensor2 = logistic_derivatives(tensor1);

    assert_true(abs(tensor2[0] - 0.196612) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - 0.196612) < 0.000001, LOG);
    assert_true(abs(tensor2[2] - 0.104994) < 0.000001, LOG);
    assert_true(abs(tensor2[3] - 0.104994) < 0.000001, LOG);

}

void FunctionsTest::test_logistic_second_derivatives()
{

    Tensor<double> tensor1;
    Vector<double> tensor2;

    tensor1.set(4);
    tensor2.set(4);

    tensor1[0] = -1;
    tensor1[1] = 1;
    tensor1[2] = 2;
    tensor1[3] = -2;

    tensor2 = logistic_second_derivatives(tensor1);

    assert_true(abs(tensor2[0] - 0.090858) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - -0.090858) < 0.000001, LOG);
    assert_true(abs(tensor2[2] - -0.079963) < 0.000001, LOG);
    assert_true(abs(tensor2[3] - 0.079963) < 0.000001, LOG);

}

void FunctionsTest::test_threshold_derivatives()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(2);
    tensor2.set(2);

    tensor1[0] = 2;
    tensor1[1] = -2;

    tensor2 = threshold_derivatives(tensor1);

    assert_true(abs(tensor2[0] - 0) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - 0) < 0.000001, LOG);

}

void FunctionsTest::test_symmetric_threshold_second_derivatives()
{

    Tensor<double> tensor1;
    Tensor<double> tensor2;

    tensor1.set(2);
    tensor2.set(2);

    tensor1[0] = 2;
    tensor1[1] = -2;

    tensor2 = threshold_derivatives(tensor1);

    assert_true(abs(tensor2[0] - 0) < 0.000001, LOG);
    assert_true(abs(tensor2[1] - 0) < 0.000001, LOG);

}


void FunctionsTest::run_test_case()
{
   cout << "Running functions test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   test_factorial();
   test_exponential();
   test_logarithm();
   test_power();
   test_binary();
   test_square_root();
   test_cumulative();
   test_lower_bounded();
   test_lower_upper_bounded();

   // Mathematic function

//   test_threshold();
//   test_symmetric_threshold();
//   test_logistic();
//   test_hyperbolic_tangent();

//   test_hyperbolic_tangent_derivatives();
//   test_logistic_derivatives();
//   test_logistic_second_derivatives();
//   test_threshold_derivatives();
//   test_threshold_second_derivatives();
//   test_symmetric_threshold_derivatives();
//   test_symmetric_threshold_second_derivatives();


   cout << "End of functions test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C); 2005-2019 Artificial Intelligence Techniques, SL.
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
