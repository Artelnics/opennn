//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "long_short_term_memory_layer_test.h"

LongShortTermMemoryLayerTest::LongShortTermMemoryLayerTest() : UnitTesting()
{
}


LongShortTermMemoryLayerTest::~LongShortTermMemoryLayerTest()
{
}


void LongShortTermMemoryLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    // Test

    long_short_term_memory_layer.set();

    assert_true(long_short_term_memory_layer.get_forget_weights().dimension(1) == 0, LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    timesteps = 1;

    long_short_term_memory_layer.set(inputs_number, neurons_number, timesteps);

    assert_true(long_short_term_memory_layer.get_parameters_number() == 12, LOG);

    // Test

    inputs_number = 2;
    neurons_number = 3;
    timesteps = 1;

    long_short_term_memory_layer.set(inputs_number, neurons_number, timesteps);

    assert_true(long_short_term_memory_layer.get_parameters_number() == 72, LOG);
}


void LongShortTermMemoryLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    LongShortTermMemoryLayer* lstm_layer = new LongShortTermMemoryLayer;

    delete lstm_layer;
}


void LongShortTermMemoryLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    Index samples_number;
    Index inputs_number;
    Index neurons_number;

    pair<type*, dimensions> inputs_pair;

    Tensor<type, 1> parameters;
    Tensor<type, 2> inputs;
    bool is_training = true;

    // Test

    samples_number = 1;
    inputs_number = 1;
    neurons_number = 1;
    timesteps = 1;

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    long_short_term_memory_layer.set(inputs_number, neurons_number, timesteps);
    long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::ActivationFunction::HyperbolicTangent);
    long_short_term_memory_layer.set_parameters_constant(type(1));

    long_short_term_layer_forward_propagation.set(samples_number, &long_short_term_memory_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    long_short_term_memory_layer.forward_propagate(tensor_wrapper(inputs_pair), &long_short_term_layer_forward_propagation, is_training);

    assert_true(long_short_term_layer_forward_propagation.outputs.rank() == 2, LOG);
    assert_true(long_short_term_layer_forward_propagation.outputs.dimension(0) == 1, LOG);
    assert_true(long_short_term_layer_forward_propagation.outputs.dimension(1) == inputs.dimension(1), LOG);

}


void LongShortTermMemoryLayerTest::run_test_case()
{
    cout << "Running long short-term memory layer test case...\n";

    // Constructor and destructor

    test_constructor();
    test_destructor();

    // Forward propagate

    test_forward_propagate();

    cout << "End of long short-term memory layer test case.\n\n";
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
