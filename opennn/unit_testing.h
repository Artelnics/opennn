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

using namespace opennn;

class UnitTesting
{

public:

   explicit UnitTesting();

   virtual ~UnitTesting();

   // Get methods

   Index get_tests_count() const;
   Index get_tests_passed_count() const;
   Index get_tests_failed_count() const;

   Index get_random_tests_number() const;

   const bool& get_display() const;

   // Set methods

   void set_tests_count(const Index&);
   void set_tests_passed_count(const Index&);
   void set_tests_failed_count(const Index&);

   void set_random_tests_number(const Index&);

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

   Index tests_count = 0;

   /// Number of tests which have passed the test case.
 
   Index tests_passed_count = 0;

   /// Number of tests which have failed the test case.

   Index tests_failed_count = 0;

   /// Number of iterations in random tests loops.

   Index random_tests_number = 0;

   /// True if messages from this class are displayed and false otherwise.

   bool display = true;

   const int n = omp_get_max_threads();
   ThreadPool* thread_pool = new ThreadPool(n);
   ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, n);

};

#endif

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
