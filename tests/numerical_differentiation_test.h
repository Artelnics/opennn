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

#include "unit_testing.h"

using namespace OpenNN;


class NumericalDifferentiationTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit NumericalDifferentiationTest();

   virtual ~NumericalDifferentiationTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   void test_set_get_methods();
   void test_calculate_methods();

   // Derivative methods

   void test_calculate_forward_differences_derivatives();
   void test_calculate_central_differences_derivatives();
   void test_calculate_derivatives();

   // Second derivative methods

   void test_calculate_forward_differences_second_derivatives();
   void test_calculate_central_differences_second_derivatives();
   void test_calculate_second_derivatives();

   // Gradient methods

   void test_calculate_forward_differences_gradient();
   void test_calculate_central_differences_gradient();
   void test_calculate_training_loss_gradient();

   void test_calculate_central_differences_gradient_matrix();

   // hessian methods

   void test_calculate_forward_differences_hessian();
   void test_calculate_central_differences_hessian();
   void test_calculate_hessian();

   // Jacobian methods

   void test_calculate_forward_differences_Jacobian();
   void test_calculate_central_differences_Jacobian();
   void test_calculate_Jacobian();

   // hessian methods

   void test_calculate_forward_differences_hessian_form();
   void test_calculate_central_differences_hessian_form();
   void test_calculate_hessian_form();

   void test_calculate_central_differences_hessian_matrices();

   // Unit testing methods

   void run_test_case();

private:

   // Constant methods


   type f1(const type& var_x) const
   {
       return var_x;
   }

   type f1_1(const type& var_x) const
   {
       return pow(var_x,2);
   }

   type f1_2(const type& var_x) const
   {
       return pow(var_x,3);
   }

   Tensor<type,1> f2(const Tensor<type,1>& vector_x) const
   {
       return vector_x.square();
   }

   Tensor<type,1> f2_1(const Index& cte, const Tensor<type,1>& vector_x) const
   {
       return cte*vector_x.square();
   }

   Tensor<type,1> f2_2(const Index& cte, const Tensor<type,2>& matrix_x) const
   {
       return cte*matrix_x.maximum();
   }

   Tensor<type,1> f2_2(const Tensor<type,1>& dummy, const Tensor<type,1>& vector_x) const
   {
       return dummy*vector_x.square();
   }

   Tensor<type,2> f3(const Tensor<type,2>& vector_x) const
   {
       return vector_x.square();
   }


   Tensor<type,2> f3_1(const Tensor<type,1>& vector_x, const Tensor<type,1>&) const
   {
       Tensor<type,2> matrix;

       return matrix;
   }


   type f4(const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> sum_ = vector_x.sum();

       return sum_(0);
   }


   type f4_1(const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> sum_ = vector_x.sum();

       return sum_(0);
   }

   type f4_2(const Index& dummy, const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> sum_ = dummy*vector_x.square().sum();

       return sum_(0);
   }

   type f4_3(const Tensor<type,1>& dummy, const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> sum_ = (vector_x * dummy).sum();

       return sum_(0);
   }

   type f4_4(const Tensor<Index,1>& dummy, const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> sum_ = (vector_x * dummy).sum();

       return sum_(0);
   }

   type f4_5(const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> prod_ = vector_x.prod();

       return prod_(0);
   }

   type f4_6(const Tensor<type,1>& dummy, const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> sum_ = (vector_x * dummy).prod();

       return sum_(0);
   }

   type f4_7(const Index& dummy, const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 0> prod_ = dummy*vector_x.square().prod();

       return prod_(0);
   }

   Tensor<type,1> f5(const Index& dummy_int, const Tensor<type,1>& dummy_vec, const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 1> prod_ = dummy_int*vector_x.square() + dummy_vec*vector_x.square();

       return prod_;
   }

   Tensor<type,1> f5_1(const Index& dummy_int_1, const Index& dummy_int_2, const Tensor<type,1>& vector_x) const
   {
       Tensor<type, 1> func_(vector_x.size());

       for(Index i = 0; i < vector_x.size(); i++)
       {
       if(vector_x(i) == 0)
       {
           func_(i) = dummy_int_1 * vector_x(i) * vector_x(i);
       }
       else
       {
           func_(i) = dummy_int_2 * vector_x(i) * vector_x(i);
       }
       }

       return func_;
   }

};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques, SL.
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
