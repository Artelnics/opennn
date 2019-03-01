/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   U N I T   T E S T I N G   C L A S S                                                                        */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include"unit_testing.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

UnitTesting::UnitTesting()
{
   message = "";

   tests_count = 0;
   tests_passed_count = 0;
   tests_failed_count = 0;

   numerical_differentiation_tests = true;
   random_tests_number = 0;

   display = true;
}


// DESTRUCTOR
 
/// Destructor.

UnitTesting::~UnitTesting()
{ 
}


// METHODS

// size_t get_tests_count() const method

/// Returns the number of tests which have been performed by the test case. 

size_t UnitTesting::get_tests_count() const
{
   return(tests_count);
}


// size_t get_tests_passed_count() const method

/// Returns the number of tests which have passed the test case. 

size_t UnitTesting::get_tests_passed_count() const
{
   return(tests_passed_count);
}


// size_t get_tests_failed_count() const method

/// Returns the number of tests which have failed the test case. 

size_t UnitTesting::get_tests_failed_count() const
{
   return(tests_failed_count);
}


// size_t get_random_tests_number() const method

/// Returns the number of iterations for loops of random tests. 

size_t UnitTesting::get_random_tests_number() const
{
   return(random_tests_number);
}


// bool get_numerical_differentiation_tests() const method

bool UnitTesting::get_numerical_differentiation_tests() const
{
   return(numerical_differentiation_tests);
}


// string& get_message() method

/// Returns a reference to the test case information message. 

string& UnitTesting::get_message() 
{
   return(message);
}


// const bool& get_display() const method

/// Returns the display messages to the screen value of this object. 

const bool& UnitTesting::get_display() const
{
   return(display);
}


// void set_tests_count(size_t) method

/// Sets a new value for the number of tests performed by the test case. 
/// @param new_tests_count Number of tests performed. 

void UnitTesting::set_tests_count(const size_t& new_tests_count)
{
   tests_count = new_tests_count;
}


// void set_tests_passed_count(size_t) method

/// Sets a new value for the number of tests which have passed the test case. 
/// @param new_tests_passed_count Number of tests passed. 

void UnitTesting::set_tests_passed_count(const size_t& new_tests_passed_count)
{
   tests_passed_count = new_tests_passed_count;
}


// void set_tests_failed_count(size_t) method

/// Sets a new value for the number of tests which have failed the test case. 
/// @param new_tests_failed_count Number of tests failed. 

void UnitTesting::set_tests_failed_count(const size_t& new_tests_failed_count)
{
   tests_failed_count = new_tests_failed_count;
}


// void set_numerical_differentiation_tests(bool) method

void UnitTesting::set_numerical_differentiation_tests(const bool& new_numerical_differentiation_tests)
{
   numerical_differentiation_tests = new_numerical_differentiation_tests;
}


// void set_random_tests_number(size_t) method

/// Sets a new value for the number of iterations in loops of random tests. 
/// @param new_random_tests_number Number of random tests in each loop. 

void UnitTesting::set_random_tests_number(const size_t& new_random_tests_number)
{
   random_tests_number = new_random_tests_number;
}


// void set_message(const string&) method

/// Sets a new test case information message. 
/// @param new_message Information message. 

void UnitTesting::set_message(const string& new_message)
{
   message = new_message;
}


// void set_display(const bool&) method

/// Sets a new display value to this object.
/// @param new_display Display value. 

void UnitTesting::set_display(const bool& new_display)
{
   display = new_display;
}


// void assert_true(bool, string) method

/// Checks that a condition is true.
/// It increases the number of tests by one.
/// It increases the number of tests passed by one if the condition is true.
/// It increases the number of tests failed by one if the condition is false.
/// It appends to the information message an error message is the condition is not satisfied. 
/// @param condition Expression of the condition to be tested. 
/// @param error_message Error message to be appended to the information message, 
/// typically the file name and the line where the condition has been tested. 

void UnitTesting::assert_true(const bool& condition, const string& error_message)
{
   tests_count++;

   if(condition)
   {
      tests_passed_count++;
   }
   else
   {
      message += "void assert_true(bool, const string&) method failed\n";
      message += error_message;
      tests_failed_count++;
   }
}


// void assert_false(bool) method

/// Checks that a condition is false.
/// It increases the number of tests by one.
/// It increases the number of tests passed by one if the condition is false.
/// It increases the number of tests failed by one if the condition is true.
/// It appends to the information message an error message is the condition is not satisfied. 
/// @param condition Expression of the condition to be tested. 
/// @param error_message Error message to be appended to the information message, 
/// typically the file name and the line where the condition has been tested. 

void UnitTesting::assert_false(const bool& condition, const string& error_message)
{
   tests_count++;

   if(!condition)
   {
      tests_passed_count++;
   }
   else
   {
      message += "void assert_false(bool, const string&) method failed\n";
      message += error_message;
      tests_failed_count++;
   }
}


// void print_results() method

/// Prints the test case results to the screen: 
/// <ul>
/// <li> Information message.
/// <li> Number of tests performed.
/// <li> Number of tests passed.
/// <li> Number of tests failed.
/// <li> Concluding remarks.
/// </ul> 

void UnitTesting::print_results()
{
   run_test_case();

   cout << message << endl;

   cout << "Tests run: " << tests_count << endl;
   cout << "Tests passed: " << tests_passed_count << endl;
   cout << "Tests failed: " << tests_failed_count << endl;

   if(tests_failed_count == 0)
   {
      cout << "Test case OK." << endl;
   }
   else
   {
      cout << "Test case NOT OK: " << tests_failed_count << " tests failed."  << endl;
   } 
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
