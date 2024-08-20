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

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/sum_squared_error.h"
#include "../opennn/learning_rate_algorithm.h"

namespace opennn
{

class LearningRateAlgorithmTest : public UnitTesting 
{

public:

    explicit LearningRateAlgorithmTest();

    virtual ~LearningRateAlgorithmTest();

    // Constructor and destructor

    void test_constructor();

    void test_destructor();

    // Training

    void test_calculate_bracketing_triplet();

    void test_calculate_golden_section_directional_point();

    void test_calculate_Brent_method_directional_point();

    // Unit testing

    void run_test_case();

private:

    DataSet data_set;

    NeuralNetwork neural_network;

    SumSquaredError sum_squared_error;

    LearningRateAlgorithm learning_rate_algorithm;
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
