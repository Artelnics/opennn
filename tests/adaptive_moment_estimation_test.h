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
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/mean_squared_error.h"

namespace opennn
{

class AdaptiveMomentEstimationTest : public UnitTesting
{

public:

    explicit AdaptiveMomentEstimationTest();

    void test_constructor();

    // Training

    void test_perform_training();

    // Unit testing

    void run_test_case();

private:

    Index samples_number = 0;
    Index inputs_number = 0;
    Index outputs_number = 0;
    Index neurons_number = 0;

    DataSet data_set;

    NeuralNetwork neural_network;

    MeanSquaredError mean_squared_error;

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
