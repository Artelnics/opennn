//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N I T   T E S T I N G   C L A S S   H E A D E R                     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef UNITTESTING_H
#define UNITTESTING_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;

class UnitTesting
{

public:

   explicit UnitTesting();   

   virtual ~UnitTesting();

   // Get methods

   size_t get_tests_count() const;
   size_t get_tests_passed_count() const;
   size_t get_tests_failed_count() const;

   bool get_numerical_differentiation_tests() const;
   size_t get_random_tests_number() const;

   const bool& get_display() const;

   // Set methods

   void set_tests_count(const size_t&);
   void set_tests_passed_count(const size_t&);
   void set_tests_failed_count(const size_t&);

   void set_numerical_differentiation_tests(const bool&);
   void set_random_tests_number(const size_t&);

   void set_message(const string&);

   void set_display(const bool&);

   // Unit testing methods

   void assert_true(const bool&, const string&);
   void assert_false(const bool&, const string&);
   
   // Test case methods

   /// This method runs all the methods contained in the test case. 

   virtual void run_test_case() = 0;

   void print_results();

protected:

   /// Number of performed tests.

   size_t tests_count;

   /// Number of tests which have passed the test case.
 
   size_t tests_passed_count;

   /// Number of tests which have failed the test case.

   size_t tests_failed_count;

   /// True if test using numerical differentiation are to be performed.

   bool numerical_differentiation_tests;

   /// Number of iterations in random tests loops.

   size_t random_tests_number;

   /// True if messages from this class are to be displayed, false otherwise.

   bool display;
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
