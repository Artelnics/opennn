//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEARNINGRATEALGORITHMTEST_H
#define LEARNINGRATEALGORITHMTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class LearningRateAlgorithmTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit LearningRateAlgorithmTest(); 

   virtual ~LearningRateAlgorithmTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_loss_index_pointer();

   // Training operators

   void test_get_learning_rate_method();
   void test_get_learning_rate_method_name();

   // Training parameters

   void test_get_loss_tolerance();

   void test_get_warning_learning_rate();

   void test_get_error_learning_rate();

   // Utilities

   void test_get_display();

   // Set methods

   void test_set();
   void test_set_default();

   void test_set_loss_index_pointer();

   // Training operators

   void test_set_learning_rate_method();

   // Training parameters

   void test_set_loss_tolerance();

   void test_set_warning_learning_rate();

   void test_set_error_learning_rate();

    // Utilities

   void test_set_display();

   // Training methods

   void test_calculate_directional_point();

   void test_calculate_bracketing_triplet();

   void test_calculate_fixed_directional_point();
   void test_calculate_golden_section_directional_point();
   void test_calculate_Brent_method_directional_point();
   
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

