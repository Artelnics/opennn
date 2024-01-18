//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADAPTIVEMOMENTESTIMATIONTEST_H
#define ADAPTIVEMOMENTESTIMATIONTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

namespace opennn
{

class AdaptiveMomentEstimationTest : public UnitTesting
{

public:

    explicit AdaptiveMomentEstimationTest();

    virtual ~AdaptiveMomentEstimationTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    // Training methods

    void test_perform_training();

    // Unit testing methods

    void run_test_case();

private:

    Index samples_number;
    Index inputs_number;
    Index outputs_number;
    Index neurons_number;

    DataSet data_set;

    NeuralNetwork neural_network;

    SumSquaredError sum_squared_error;

    AdaptiveMomentEstimation adaptive_moment_estimation;

    TrainingResults training_results;

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
