/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N U M E R I C A L   I N T E G R A T I O N   T E S T   C L A S S                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "unit_testing.h"

#include "polynomial.h"

#include "numerical_integration_test.h"


using namespace OpenNN;


NumericalIntegrationTest::NumericalIntegrationTest() : UnitTesting() 
{
}


NumericalIntegrationTest::~NumericalIntegrationTest()
{
}


void NumericalIntegrationTest::test_constructor()
{
   message += "test_constructor\n";
}


void NumericalIntegrationTest::test_destructor()
{
   message += "test_destructor\n";
}


void NumericalIntegrationTest::test_calculate_trapezoid_integral()
{
   message += "test_calculate_trapezoid_integral\n";

   NumericalIntegration ni;

   // Case 1

   Vector<double> x_1(0, 1, 10);
   Vector<double> y_1(0, 1, 10);

   double integral_1 = ni.calculate_trapezoid_integral(x_1, y_1);

   assert_true(integral_1 == 50.0, LOG);

}


void NumericalIntegrationTest::test_calculate_Simpson_integral()
{
   message += "test_calculate_Simpson_integral\n";

   NumericalIntegration ni;

   // Case 1

   Vector<double> x_1(0.0, 1, 10.0);
   Vector<double> y_1(0.0, 1, 10.0);

   double integral_1 = ni.calculate_Simpson_integral(x_1, y_1);

   assert_true(integral_1 == 50.0, LOG);

}


void NumericalIntegrationTest::run_test_case()
{
   message += "Running numerical integration test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Integration methods

   test_calculate_trapezoid_integral();
   test_calculate_Simpson_integral();

   message += "End of numerical integration test case.\n";
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
