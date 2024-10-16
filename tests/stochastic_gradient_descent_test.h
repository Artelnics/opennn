//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STOCHASTICGRADIENTDESCENTTEST_H
#define STOCHASTICGRADIENTDESCENTTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/sum_squared_error.h"
#include "../opennn/stochastic_gradient_descent.h"

namespace opennn
{

class StochasticGradientDescentTest : public UnitTesting
{

public:

    explicit StochasticGradientDescentTest();

    void test_constructor();

    // Training

    void test_perform_training();

    void test_transformer_training();

    // Serialization

    void test_to_XML();

    // Unit testing

    void run_test_case();

private:

    Index samples_number;
    Index inputs_number;
    Index outputs_number;

    DataSet data_set;

    NeuralNetwork neural_network;

    SumSquaredError sum_squared_error;

    StochasticGradientDescent stochastic_gradient_descent;

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
