//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   T E S T   C L A S S   H E A D E R 
//
//   Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRAININGSTRATEGYTEST_H
#define TRAININGSTRATEGYTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class TrainingStrategyTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   

   explicit TrainingStrategyTest(); 


   

   virtual ~TrainingStrategyTest();


   

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_loss_index_pointer();

   // Utilities

   void test_get_display();

   // Set methods

   void test_set();
   void test_set_default();

   void test_set_loss_index_pointer();

    // Utilities

   void test_set_display();

   // Training methods

   void test_perform_training();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   void test_print();
   void test_save();
   void test_load();

   // Results methods

   void test_results_constructor();
   void test_results_destructor();

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

