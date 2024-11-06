//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N I T   T E S T I N G   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include"unit_testing.h"

#include <iostream>
#include <string>

namespace opennn
{

UnitTesting::UnitTesting()
{
}


Index UnitTesting::get_tests_count() const
{
    return tests_count;
}


Index UnitTesting::get_tests_passed_count() const
{
    return tests_passed_count;
}


Index UnitTesting::get_tests_failed_count() const
{
    return tests_failed_count;
}


Index UnitTesting::get_random_tests_number() const
{
    return random_tests_number;
}


const bool& UnitTesting::get_display() const
{
    return display;
}


void UnitTesting::set_tests_count(const Index& new_tests_count)
{
    tests_count = new_tests_count;
}


void UnitTesting::set_tests_passed_count(const Index& new_tests_passed_count)
{
    tests_passed_count = new_tests_passed_count;
}


void UnitTesting::set_tests_failed_count(const Index& new_tests_failed_count)
{
    tests_failed_count = new_tests_failed_count;
}


void UnitTesting::set_random_tests_number(const Index& new_random_tests_number)
{
    random_tests_number = new_random_tests_number;
}


void UnitTesting::set_display(const bool& new_display)
{
    display = new_display;
}


void UnitTesting::assert_true(const bool& condition, const string& error_message)
{
    tests_count++;

    if(condition)
    {
        tests_passed_count++;
    }
    else
    {
        cout << "void assert_true(bool, const string&) method failed\n";
        cout << error_message;
        tests_failed_count++;
    }
}


void UnitTesting::assert_false(const bool& condition, const string& error_message)
{
    tests_count++;

    if(!condition)
    {
        tests_passed_count++;
    }
    else
    {
        cout << "void assert_false(bool, const string&) method failed\n";
        cout << error_message;
        tests_failed_count++;
    }
}


void UnitTesting::print_results()
{
    run_test_case();

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

}

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
