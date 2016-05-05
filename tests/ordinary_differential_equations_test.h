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

   explicit OrdinaryDifferentialEquationsTest(void);

   // DESTRUCTOR

   virtual ~OrdinaryDifferentialEquationsTest(void);

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Get methods

   void test_get_points_number(void);

   void test_get_tolerance(void);
   void test_get_initial_size(void);
   void test_get_warning_size(void);
   void test_get_error_size(void);

   void test_get_display(void);

   // Set methods

   void test_set_default(void);

   void test_set_points_number(void);

   void test_set_tolerance(void);
   void test_set_initial_size(void);
   void test_set_warning_size(void);
   void test_set_error_size(void);

   void test_set_display(void);

   // Runge-Kutta methods

   void test_calculate_Runge_Kutta_integral_1(void);
   void test_calculate_Runge_Kutta_integral_2(void);
   void test_calculate_Runge_Kutta_integral_3(void);
   void test_calculate_Runge_Kutta_integral_4(void);
   void test_calculate_Runge_Kutta_integral_5(void);

   // Runge-Kutta-Fehlberg methods

   void test_calculate_Runge_Kutta_Fehlberg_integral_1(void);
   void test_calculate_Runge_Kutta_Fehlberg_integral_2(void);
   void test_calculate_Runge_Kutta_Fehlberg_integral_3(void);
   void test_calculate_Runge_Kutta_Fehlberg_integral_4(void);
   void test_calculate_Runge_Kutta_Fehlberg_integral_5(void);

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

   void test_to_XML(void);   
   void test_from_XML(void);   

   void test_save(void);
   void test_load(void);

   // Unit testing methods

   void run_test_case(void);
};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
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
