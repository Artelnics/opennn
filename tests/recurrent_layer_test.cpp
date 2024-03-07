//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "recurrent_layer_test.h"

RecurrentLayerTest::RecurrentLayerTest() : UnitTesting()
{
}


RecurrentLayerTest::~RecurrentLayerTest()
{
}


void RecurrentLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    Index inputs_number;
    Index neurons_number;

    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> recurrent_initializer;
    Tensor<type, 1> biases;

    // Test

    inputs_number = 1;
    neurons_number = 1;

    recurrent_layer.set(inputs_number, neurons_number);

    assert_true(recurrent_layer.get_parameters_number() == 3, LOG);
    assert_true(recurrent_layer.get_biases_number() == 1, LOG);

    // Test

    inputs_number = 2;
    neurons_number = 3;

    recurrent_layer.set(inputs_number, neurons_number);

    assert_true(recurrent_layer.get_parameters_number() == 18, LOG);
    assert_true(recurrent_layer.get_biases_number() == 3, LOG);
}


void RecurrentLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    RecurrentLayer* recurrent_layer = new RecurrentLayer;
    delete recurrent_layer;
}


void RecurrentLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activations_derivatives\n";

    Tensor<type, 1> combinations;
    Tensor<type, 1> activations;
    Tensor<type, 1> activations_derivatives;

    Tensor<type, 1> numerical_activation_derivative;
    Tensor<type, 0> maximum_difference;

}


void RecurrentLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";
    
    neurons_number = 4;
    samples_number = 2;
    inputs_number = 3;
    bool is_training = false;

    Tensor<type, 2> outputs;

    Tensor<type, 1> parameters;
    Tensor<type, 2> new_weights;
    Tensor<type, 2> new_recurrent_weights;
    Tensor<type, 1> new_biases;

    pair<type*, dimensions> inputs_pair;

    recurrent_layer.set(inputs_number, neurons_number);

    recurrent_layer.set_activation_function(RecurrentLayer::ActivationFunction::HyperbolicTangent);

    Tensor<type, 2> inputs(samples_number, inputs_number);

    recurrent_layer.set_parameters_constant(type(1));
    inputs.setConstant(type(1));

    recurrent_layer_forward_propagation.set(samples_number, &recurrent_layer);

    Tensor<type*, 1> inputs_data(1);
    inputs_data(0) = inputs.data();

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    recurrent_layer.forward_propagate(tensor_wrapper(inputs_pair), &recurrent_layer_forward_propagation, is_training);

    outputs = recurrent_layer_forward_propagation.outputs;
    
    // Test

    samples_number = 3;
    inputs_number = 2;
    neurons_number = 2;

    recurrent_layer.set(inputs_number, neurons_number);

    inputs.resize(samples_number,inputs_number);
    inputs.setConstant(type(1));

    recurrent_layer.set_activation_function("SoftPlus");
    recurrent_layer.set_timesteps(3);

    new_weights.resize(2,2);
    new_weights.setConstant(type(1));
    new_recurrent_weights.resize(2,2);
    new_recurrent_weights.setConstant(type(1));
    new_biases.resize(2);
    new_biases.setConstant(type(1));

    recurrent_layer.set_biases(new_biases);
    recurrent_layer.set_input_weights(new_weights);
    recurrent_layer.set_recurrent_weights(new_recurrent_weights);

    parameters = recurrent_layer.get_parameters();

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};
    
    recurrent_layer.forward_propagate(tensor_wrapper(inputs_pair), &recurrent_layer_forward_propagation, is_training);

    outputs = recurrent_layer_forward_propagation.outputs;
}


void RecurrentLayerTest::run_test_case()
{
    cout << "Running recurrent layer test case...\n";

    // Constructor and destructor

    test_constructor();

    test_destructor();

    // Activation Derivatives

    test_calculate_activations_derivatives();

    // Forward propagate

    test_forward_propagate();

    cout << "End of recurrent layer test case.\n\n";
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
