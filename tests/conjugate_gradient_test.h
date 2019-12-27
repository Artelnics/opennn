//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   T E S T   C L A S S   H E A D E R        
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONJUGATEGRADIENTTEST_H
#define CONJUGATEGRADIENTTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class ConjugateGradientTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   

   explicit ConjugateGradientTest(); 


   

   virtual ~ConjugateGradientTest();


   

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_training_direction_method();
   void test_get_training_direction_method_name();

   // Set methods

   void test_set_training_direction_method();

   // Training methods

   void test_calculate_PR_parameter();
   void test_calculate_FR_parameter();

   void test_calculate_FR_training_direction();
   void test_calculate_PR_training_direction();

   void test_calculate_training_direction();

   void test_perform_training();

   // Training history methods

   void test_set_reserve_all_training_history();

   // Serialization methods

   void test_to_XML();   
   void test_from_XML();

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

