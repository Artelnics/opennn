//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D I F F E R E N T I A T I O N   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NUMERICALDIFFERENTIATIONTEST_H
#define NUMERICALDIFFERENTIATIONTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class NumericalDifferentiationTest : public UnitTesting
{

public:

    explicit NumericalDifferentiationTest();

    virtual ~NumericalDifferentiationTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    void test_calculate_methods();

    // Derivative methods

    void test_calculate_derivatives();

    // Second derivative methods

    void test_calculate_second_derivatives();

    // Gradient methods

    void test_calculate_gradient();

    void test_calculate_gradient_matrix();

    // hessian methods

    void test_calculate_hessian();

    void test_calculate_hessian_form();

    void test_calculate_hessian_matrices();

    // Jacobian methods

    void test_calculate_Jacobian();

    // Unit testing methods

    void run_test_case();

private:

    NumericalDifferentiation numerical_differentiation;

    // Constant methods

    type f1(const type& var_x) const
    {
        return var_x;
    }

    type f1_1(const type& var_x) const
    {
        return pow(var_x, type(2));
    }

    type f1_2(const type& var_x) const
    {
        return pow(var_x, type(3));
    }

    Tensor<type, 1> f2(const Tensor<type, 1>& vector_x) const
    {
        return vector_x.square();
    }

    Tensor<type, 1> f2_1(const Index& constant, const Tensor<type, 1>& vector_x) const
    {
        return type(constant)*vector_x.square();
    }

    Tensor<type, 1> f2_2(const Index& constant, const Tensor<type, 2>& matrix_x) const
    {
        Tensor<type, 1> y;

        return  type(constant)*matrix_x.maximum();
    }

    Tensor<type, 1> f2_2(const Tensor<type, 1>& dummy, const Tensor<type, 1>& vector_x) const
    {
        return dummy*vector_x.square();
    }

    Tensor<type, 2> f3(const Tensor<type, 2>& vector_x) const
    {
        return vector_x.square();
    }


    Tensor<type, 2> f3_1(const Tensor<type, 1>& vector_x, const Tensor<type, 1>&) const
    {
        Tensor<type, 2> matrix;

        return matrix;
    }

    type f4(const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> sum_ = vector_x.sum();

        return sum_(0);
    }


    type f4_1(const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> sum_ = vector_x.sum();

        return sum_(0);
    }

    type f4_2(const Index& dummy, const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> sum_ = type(dummy)*vector_x.square().sum();

        return sum_(0);
    }

    type f4_3(const Tensor<type, 1>& dummy, const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> sum_ = (vector_x * dummy).sum();

        return sum_(0);
    }

    type f4_4(const Tensor<Index,1>& dummy, const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> sum_ = (vector_x * dummy).sum();

        return sum_(0);
    }

    type f4_5(const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> prod_ = vector_x.prod();

        return prod_(0);
    }

    type f4_6(const Tensor<type, 1>& dummy, const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> sum_ = (vector_x * dummy).prod();

        return sum_(0);
    }

    type f4_7(const Index& dummy, const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 0> prod_ = type(dummy) * vector_x.square().prod();

        return prod_(0);
    }

    Tensor<type, 1> f5(const Index& dummy_int, const Tensor<type, 1>& dummy_vec, const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 1> prod_ = type(dummy_int) * vector_x.square() + dummy_vec * vector_x.square();

        return prod_;
    }

    Tensor<type, 1> f5_1(const Index& dummy_int_1, const Index& dummy_int_2, const Tensor<type, 1>& vector_x) const
    {
        Tensor<type, 1> func_(vector_x.size());

        for(Index i = 0; i < vector_x.size(); i++)
        {
            if(vector_x(i) == type(0))
            {
                func_(i) = type(dummy_int_1) * vector_x(i) * vector_x(i);
            }
            else
            {
                func_(i) = type(dummy_int_2) * vector_x(i) * vector_x(i);
            }
        }

        return func_;
    }

};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
