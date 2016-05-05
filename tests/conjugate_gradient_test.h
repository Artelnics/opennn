/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O N J U G A T E   G R A D I E N T   T E S T   C L A S S   H E A D E R                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __CONJUGATEGRADIENTTEST_H__
#define __CONJUGATEGRADIENTTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class ConjugateGradientTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit ConjugateGradientTest(void); 


   // DESTRUCTOR

   virtual ~ConjugateGradientTest(void);


   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Get methods

   void test_get_training_direction_method(void);
   void test_get_training_direction_method_name(void);

   // Set methods

   void test_set_training_direction_method(void);

   // Training methods

   void test_calculate_PR_parameter(void);
   void test_calculate_FR_parameter(void);

   void test_calculate_FR_training_direction(void);
   void test_calculate_PR_training_direction(void);

   void test_calculate_training_direction(void);

   void test_perform_training(void);

   // Training history methods

   void test_set_reserve_all_training_history(void);

   // Serialization methods

   void test_to_XML(void);   
   void test_from_XML(void);

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

