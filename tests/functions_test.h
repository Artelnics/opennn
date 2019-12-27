//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N    T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef FUNCTIONSTEST_H
#define FUNCTIONSTEST_H

// Unit testing includes

#include "unit_testing.h"


using namespace OpenNN;

class FunctionsTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

    // DEFAULT CONSTRUCTOR

    explicit FunctionsTest();

    virtual ~FunctionsTest();

    void test_constructor();
    void test_destructor();

    void test_factorial();
    void test_exponential();
    void test_logarithm();
    void test_power();
    void test_binary();
    void test_square_root();
    void test_cumulative();
    void test_lower_bounded();
    void test_upper_bounded();
    void test_lower_upper_bounded();

    // Mathematics function

    void test_threshold();
    void test_symmetric_threshold();
    void test_logistic();
    void test_hyperbolic_tangent();

    void test_hyperbolic_tangent_derivatives();
    void test_logistic_derivatives();
    void test_logistic_second_derivatives();
    void test_threshold_derivatives();
    void test_threshold_second_derivatives();
    void test_symmetric_threshold_derivatives();
    void test_symmetric_threshold_second_derivatives();

    // Unit tseting method

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
