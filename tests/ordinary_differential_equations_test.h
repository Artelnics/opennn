/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D I N A R Y   D I F F E R E N T I A L   E Q U A T I O N S   T E S T   C L A S S   H E A D E R          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   E-mail: robertolopez@artelnics.com                                                                         */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __ORDINARYDIFFERENTIALEQUATIONSTEST_H__
#define __ORDINARYDIFFERENTIALEQUATIONSTEST_H__

// Unit testing includes

#include "unit_testing.h"


using namespace OpenNN;


class OrdinaryDifferentialEquationsTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit OrdinaryDifferentialEquationsTest();

   // DESTRUCTOR

   virtual ~OrdinaryDifferentialEquationsTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_points_number();

   void test_get_tolerance();
   void test_get_initial_size();
   void test_get_warning_size();
   void test_get_error_size();

   void test_get_display();

   // Set methods

   void test_set_default();

   void test_set_points_number();

   void test_set_tolerance();
   void test_set_initial_size();
   void test_set_warning_size();
   void test_set_error_size();

   void test_set_display();

   // Runge-Kutta methods

   void test_calculate_Runge_Kutta_integral_1();
   void test_calculate_Runge_Kutta_integral_2();
   void test_calculate_Runge_Kutta_integral_3();
   void test_calculate_Runge_Kutta_integral_4();
   void test_calculate_Runge_Kutta_integral_5();

   // Runge-Kutta-Fehlberg methods

   void test_calculate_Runge_Kutta_Fehlberg_integral_1();
   void test_calculate_Runge_Kutta_Fehlberg_integral_2();
   void test_calculate_Runge_Kutta_Fehlberg_integral_3();
   void test_calculate_Runge_Kutta_Fehlberg_integral_4();
   void test_calculate_Runge_Kutta_Fehlberg_integral_5();

   double calculate_zero_dot(const NeuralNetwork&, const double&, const double&) const;
   double calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&) const;
   double calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&) const;
   double calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&) const;
   double calculate_zero_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&, const double&) const;

   double calculate_x_dot(const NeuralNetwork&, const double&, const double&) const;
   double calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&) const;
   double calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&) const;
   double calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&) const;
   double calculate_x_dot(const NeuralNetwork&, const double&, const double&, const double&, const double&, const double&, const double&) const;

   double calculate_x_squared_dot(const NeuralNetwork&, const double&, const double&) const;

   // Serialization methods

   void test_to_XML();   
   void test_from_XML();   

   void test_save();
   void test_load();

   // Unit testing methods

   void run_test_case();
};


#endif


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
