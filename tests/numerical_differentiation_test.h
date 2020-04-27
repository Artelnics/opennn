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

//   void test_calculate_forward_differences_derivatives();
//   void test_calculate_central_differences_derivatives();
//   void test_calculate_derivatives();

   // Second derivative methods

   void test_calculate_forward_differences_second_derivatives();
   void test_calculate_central_differences_second_derivatives();
   void test_calculate_second_derivatives();

//   void test_calculate_forward_differences_second_derivatives();
//   void test_calculate_central_differences_second_derivatives();
//   void test_calculate_second_derivatives();

   // Gradient methods

   void test_calculate_forward_differences_gradient();
   void test_calculate_central_differences_gradient();
   void test_calculate_training_loss_gradient();

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


   Tensor<type,1> f2(const Tensor<type,1>& vect_x) const
   {
       return vect_x.square();
   }

   Tensor<type,2> f3(const Tensor<type,2>& vect_x) const
   {
       return vect_x.square();
   }

   Tensor<type,1> f4(const Index& cte, const Tensor<type,1>& vect_x) const
   {
       return cte*vect_x.square();
   }

/*
   Tensor<type,2> f5(const Tensor<type,1>& vect_x, const Tensor<type,1>& vect_x) const
   {
       return cte*vect_x.square();
   }
*/

   type f6(const Tensor<type,1>& vect_x) const
   {
       Tensor<type, 0> sum_ = vect_x.sum();

       return sum_(0);
   }

   type f6_(const Tensor<type,1>& vect_x) const
   {
       Tensor<type, 0> sum_ = vect_x.sum();

       return sum_(0);
   }

   type f7(const Index& dummy, const Tensor<type,1>& vect_x) const
   {
       Tensor<type, 0> sum_ = dummy*vect_x.square().sum();

       return sum_(0);
   }

   type f8(const Tensor<type,1>& dummy, const Tensor<type,1>& vect_x) const
   {
       Tensor<type, 0> sum_ = (vect_x * dummy).sum();

       return sum_(0);
   }

   type f9(const Tensor<Index,1>& dummy, const Tensor<type,1>& vect_x) const
   {
       Tensor<type, 0> sum_ = (vect_x * dummy).sum();

       return sum_(0);
   }

   /*   type f2(const Tensor<type, 1>&) const;
   Tensor<type, 1> f3(const Tensor<type, 1>&) const;
   type f7(const Index&, const Tensor<type, 1>&, const Index&, const Tensor<type, 1>&) const;

   // Non constant methods

   type f4(const type&);
   type f5(const Tensor<type, 1>&);
   Tensor<type, 1> f6(const Tensor<type, 1>&);

   Tensor<type, 1> f8(const Index&, const Index&, const Tensor<type, 1>&) const;
*/
};


#endif


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
