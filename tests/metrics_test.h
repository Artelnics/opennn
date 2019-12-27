//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E T R I C S   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef METRICSTEST_H
#define METRICSTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class MetricsTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit MetricsTest();

   virtual ~MetricsTest();

   // Constructor and destructor methods

   void test_constructor();

   void test_destructor();

   // NORMS

   void test_l1_norm();

   void test_l1_norm_gradient();

    void test_l1_norm_hessian();

   void test_l2_norm();

   void test_l2_norm_gradient();

   void test_l2_norm_hessian();

   void test_Lp_norm();

   void test_Lp_norm_gradient();

   void test_lp_norm();

   void test_lp_norm_gradient();

   void test_linear_combinations();


   // INVERTING MATICES

   void test_determinant();

   void test_cofactor();

   void test_inverse();


   // LINEAR EQUATIONS

   void test_eigenvalues();

   void test_eigenvectors();

   void test_direct();

   // Vector distances

   void test_euclidean_distance();

   void test_euclidean_weighted_distance();

   void test_euclidean_weighted_distance_vector();

   void test_manhattan_distance();

   void test_manhattan_weighted_distance();

   void test_manhattan_weighted_distance_vector();

   // Unit testing methods

   void run_test_case();
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

