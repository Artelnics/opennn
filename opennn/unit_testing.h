//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N I T   T E S T I N G   C L A S S   H E A D E R                     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef UNITTESTING_H
#define UNITTESTING_H

#include <string>

#include "config.h"

namespace opennn
{

class UnitTesting
{

public:

   explicit UnitTesting();

   virtual ~UnitTesting();

   // Get

   Index get_tests_count() const;
   Index get_tests_passed_count() const;
   Index get_tests_failed_count() const;

   Index get_random_tests_number() const;

   const bool& get_display() const;

   // Set

   void set_tests_count(const Index&);
   void set_tests_passed_count(const Index&);
   void set_tests_failed_count(const Index&);

   void set_random_tests_number(const Index&);

   void set_message(const string&);

   void set_display(const bool&);

   // Unit testing

   void assert_true(const bool&, const string&);
   void assert_false(const bool&, const string&);
   
   // Test case

   virtual void run_test_case() = 0;

   void print_results();

protected:

   Index tests_count = 0;

   Index tests_passed_count = 0;

   Index tests_failed_count = 0;

   Index random_tests_number = 0;

   bool display = true;

   const int n = omp_get_max_threads();
   ThreadPool* thread_pool = new ThreadPool(n);
   ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, n);

};

}
#endif

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
