/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D I N A R Y   D I F F E R E N T I A L   E Q U A T I O N S   T E S T   C L A S S                        */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "ordinary_differential_equations_test.h"

using namespace OpenNN;


OrdinaryDifferentialEquationsTest::OrdinaryDifferentialEquationsTest() : UnitTesting() 
{   
}


OrdinaryDifferentialEquationsTest::~OrdinaryDifferentialEquationsTest()
{
}


void OrdinaryDifferentialEquationsTest::test_constructor()
{
   message += "test_constructor\n";
}


void OrdinaryDifferentialEquationsTest::test_destructor()
{
   message += "test_destructor\n";
}


void OrdinaryDifferentialEquationsTest::test_get_points_number()
{
   message += "test_get_points_number\n";
}


void OrdinaryDifferentialEquationsTest::test_get_tolerance()
{
   message += "test_get_tolerance\n";

}


void OrdinaryDifferentialEquationsTest::test_get_initial_size()
{
   message += "test_get_initial_size\n";

}


void OrdinaryDifferentialEquationsTest::test_get_warning_size()
{
   message += "test_get_warning_size\n";
}


void OrdinaryDifferentialEquationsTest::test_get_error_size()
{
   message += "test_get_error_size\n";
}


void OrdinaryDifferentialEquationsTest::test_get_display()
{
   message += "test_get_display\n";
}


void OrdinaryDifferentialEquationsTest::test_set_default()
{
   message += "test_set_default\n";
}


void OrdinaryDifferentialEquationsTest::test_set_points_number()
{
   message += "test_set_points_number\n";
}


void OrdinaryDifferentialEquationsTest::test_set_tolerance()
{
   message += "test_set_tolerance\n";
}


void OrdinaryDifferentialEquationsTest::test_set_initial_size()
{
   message += "test_set_initial_size\n";
}


void OrdinaryDifferentialEquationsTest::test_set_warning_size()
{
   message += "test_set_warning_size\n";
}


void OrdinaryDifferentialEquationsTest::test_set_error_size()
{
   message += "test_set_error_size\n";
}


void OrdinaryDifferentialEquationsTest::test_set_display()
{
   message += "test_set_display\n";
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_integral_1()
{
   message += "test_calculate_Runge_Kutta_integral_1\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   size_t points_number = 11;

   ode.set_points_number(points_number);

   Vector<double> x(points_number);
   Vector<double> y(points_number); 

   ode.calculate_Runge_Kutta_solution(*this, 
   x, y,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0);
   
   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y[points_number-1] == 0.0, LOG);

   ode.calculate_Runge_Kutta_solution(*this, 
   x, y,
   &OrdinaryDifferentialEquationsTest::calculate_x_dot,
   nn, 0.0, 1.0, 
   0.0);

   assert_true(x[points_number-1] == 1.0, LOG);
//   assert_true(y[points_number-1] == 0.5, LOG);

   ode.calculate_Runge_Kutta_solution(*this, 
   x, y,
   &OrdinaryDifferentialEquationsTest::calculate_x_squared_dot,
   nn, 0.0, 1.0, 
   0.0);

   assert_true(x[points_number-1] == 1.0, LOG);
//   assert_true(y[points_number-1] == 1.0/3.0, LOG);
*/
}

// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_integral_2()
{
   message += "test_calculate_Runge_Kutta_integral_2\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   size_t points_number = 11;

   ode.set_points_number(points_number);

   Vector<double> x(points_number);
   Vector<double> y1(points_number); 
   Vector<double> y2(points_number); 

   ode.calculate_Runge_Kutta_solution(*this, 
   x, y1, y2,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0, 0.0);
   
   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y1[points_number-1] == 0.0, LOG);
   assert_true(y2[points_number-1] == 0.0, LOG);
*/
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_integral_3()
{
   message += "test_calculate_Runge_Kutta_integral_3\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   size_t points_number = 11;

   ode.set_points_number(points_number);

   Vector<double> x(points_number);
   Vector<double> y1(points_number); 
   Vector<double> y2(points_number); 
   Vector<double> y3(points_number); 

   ode.calculate_Runge_Kutta_solution(*this, 
   x, y1, y2, y3,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0, 0.0, 0.0);
   
   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y1[points_number-1] == 0.0, LOG);
   assert_true(y2[points_number-1] == 0.0, LOG);
   assert_true(y3[points_number-1] == 0.0, LOG);
*/
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_integral_4()
{
   message += "test_calculate_Runge_Kutta_integral_4\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   size_t points_number = 11;

   ode.set_points_number(points_number);

   Vector<double> x(points_number);
   Vector<double> y1(points_number); 
   Vector<double> y2(points_number); 
   Vector<double> y3(points_number); 
   Vector<double> y4(points_number); 

   ode.calculate_Runge_Kutta_solution(*this, 
   x, y1, y2, y3, y4,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0, 0.0, 0.0, 0.0);
   
   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y1[points_number-1] == 0.0, LOG);
   assert_true(y2[points_number-1] == 0.0, LOG);
   assert_true(y3[points_number-1] == 0.0, LOG);
   assert_true(y4[points_number-1] == 0.0, LOG);
*/
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_integral_5()
{
   message += "test_calculate_Runge_Kutta_integral_5\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   size_t points_number = 11;

   ode.set_points_number(points_number);

   Vector<double> x(points_number);
   Vector<double> y1(points_number); 
   Vector<double> y2(points_number); 
   Vector<double> y3(points_number); 
   Vector<double> y4(points_number); 
   Vector<double> y5(points_number); 

   ode.calculate_Runge_Kutta_solution(*this, 
   x, y1, y2, y3, y4, y5,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0, 0.0, 0.0, 0.0, 0.0);
   
   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y1[points_number-1] == 0.0, LOG);
   assert_true(y2[points_number-1] == 0.0, LOG);
   assert_true(y3[points_number-1] == 0.0, LOG);
   assert_true(y4[points_number-1] == 0.0, LOG);
   assert_true(y5[points_number-1] == 0.0, LOG);
*/
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_Fehlberg_integral_1()
{
   message += "test_calculate_Runge_Kutta_Fehlberg_integral_1\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   Vector<double> x;
   Vector<double> y; 

   size_t points_number = ode.calculate_Runge_Kutta_Fehlberg_solution(*this, 
   x, y,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0);

   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y[points_number-1] == 0.0, LOG);
*/
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_Fehlberg_integral_2()
{
   message += "test_calculate_Runge_Kutta_Fehlberg_integral_2\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   Vector<double> x;
   Vector<double> y1; 
   Vector<double> y2; 

   size_t points_number = ode.calculate_Runge_Kutta_Fehlberg_solution(*this, 
   x, y1, y2,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0, 0.0);

   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y1[points_number-1] == 0.0, LOG);
   assert_true(y2[points_number-1] == 0.0, LOG);
*/
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_Fehlberg_integral_3()
{
   message += "test_calculate_Runge_Kutta_Fehlberg_integral_3\n";

//   OrdinaryDifferentialEquations ode;
//   NeuralNetwork nn;

//   Vector<double> x;
//   Vector<double> y1; 
//   Vector<double> y2; 
//   Vector<double> y3; 

//   size_t points_number = ode.calculate_Runge_Kutta_Fehlberg_solution(*this, 
//   x, y1, y2, y3,
//   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
//   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
//   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
//   nn, 0.0, 1.0, 
//   0.0, 0.0, 0.0);

//   assert_true(x[points_number-1] == 1.0, LOG);
//   assert_true(y1[points_number-1] == 0.0, LOG);
//   assert_true(y2[points_number-1] == 0.0, LOG);
//   assert_true(y3[points_number-1] == 0.0, LOG);

}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_Fehlberg_integral_4()
{
   message += "test_calculate_Runge_Kutta_Fehlberg_integral_4\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   Vector<double> x;
   Vector<double> y1; 
   Vector<double> y2; 
   Vector<double> y3; 
   Vector<double> y4; 

   size_t points_number = ode.calculate_Runge_Kutta_Fehlberg_solution(*this, 
   x, y1, y2, y3, y4,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0, 0.0, 0.0, 0.0);

   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y1[points_number-1] == 0.0, LOG);
   assert_true(y2[points_number-1] == 0.0, LOG);
   assert_true(y3[points_number-1] == 0.0, LOG);
   assert_true(y4[points_number-1] == 0.0, LOG);
*/
}


// @todo Update tests.

void OrdinaryDifferentialEquationsTest::test_calculate_Runge_Kutta_Fehlberg_integral_5()
{
   message += "test_calculate_Runge_Kutta_Fehlberg_integral_5\n";
/*
   OrdinaryDifferentialEquations ode;
   NeuralNetwork nn;

   Vector<double> x;
   Vector<double> y1; 
   Vector<double> y2; 
   Vector<double> y3; 
   Vector<double> y4; 
   Vector<double> y5; 

   size_t points_number = ode.calculate_Runge_Kutta_Fehlberg_solution(*this, 
   x, y1, y2, y3, y4, y5,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   &OrdinaryDifferentialEquationsTest::calculate_zero_dot,
   nn, 0.0, 1.0, 
   0.0, 0.0, 0.0, 0.0, 0.0);

   assert_true(x[points_number-1] == 1.0, LOG);
   assert_true(y1[points_number-1] == 0.0, LOG);
   assert_true(y2[points_number-1] == 0.0, LOG);
   assert_true(y3[points_number-1] == 0.0, LOG);
   assert_true(y4[points_number-1] == 0.0, LOG);
   assert_true(y5[points_number-1] == 0.0, LOG);
*/
}


double OrdinaryDifferentialEquationsTest::calculate_zero_dot(const NeuralNetwork&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_x_dot(const NeuralNetwork&, const double& x, const double&) const
{
   return(x);
}


double OrdinaryDifferentialEquationsTest::calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&, const double&) const
{
   return(0.0);
}


double OrdinaryDifferentialEquationsTest::calculate_x_squared_dot(const NeuralNetwork&, const double& x, const double&) const
{
   return(x*x);
}


void OrdinaryDifferentialEquationsTest::test_to_XML()   
{
   message += "test_to_XML\n";
}


void OrdinaryDifferentialEquationsTest::test_from_XML()   
{
   message += "test_from_XML\n";
}


void OrdinaryDifferentialEquationsTest::run_test_case()
{
   message += "Running ordinary differential equations test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_points_number();

   test_get_tolerance();
   test_get_initial_size();
   test_get_warning_size();
   test_get_error_size();

   test_get_display();

   // Set methods

   test_set_default();

   test_set_points_number();

   test_set_tolerance();
   test_set_initial_size();
   test_set_warning_size();
   test_set_error_size();

   test_set_display();

   // Runge-Kutta methods

   test_calculate_Runge_Kutta_integral_1();
   test_calculate_Runge_Kutta_integral_2();
   test_calculate_Runge_Kutta_integral_3();
   test_calculate_Runge_Kutta_integral_4();
   test_calculate_Runge_Kutta_integral_5();

   // Runge-Kutta-Fehlberg methods

   test_calculate_Runge_Kutta_Fehlberg_integral_1();
   test_calculate_Runge_Kutta_Fehlberg_integral_2();
   test_calculate_Runge_Kutta_Fehlberg_integral_3();
   test_calculate_Runge_Kutta_Fehlberg_integral_4();
   test_calculate_Runge_Kutta_Fehlberg_integral_5();

   message += "End of ordinary differential equations test case.\n";
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
