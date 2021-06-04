//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CORRELATIONS_TEST_H
#define CORRELATIONS_TEST_H

// Unit testing includes

#include "unit_testing.h"

class CorrelationsTest : public UnitTesting
{

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

    // DEFAULT CONSTRUCTOR

    explicit CorrelationsTest();

    virtual ~CorrelationsTest();

    // Correlation methods

    void test_linear_correlation();

    void test_linear_regression();

    void test_logistic_correlation();

    void test_logarithmic_correlation();

    void test_exponential_correlation();

    void test_power_correlation();

    // Time series correlation methods

    void test_autocorrelations();
    void test_cross_correlations();

    // Contingency table

    // Unit testing methods

    void run_test_case();

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
