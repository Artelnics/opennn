//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer_test.h"


ScalingLayerTest::ScalingLayerTest() : UnitTesting()
{
    scaling_layer.set_display(false);

}


ScalingLayerTest::~ScalingLayerTest()
{
}


void ScalingLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    ScalingLayer2D scaling_layer_1;

    assert_true(scaling_layer_1.get_type() == Layer::Type::Scaling2D, LOG);
    assert_true(scaling_layer_1.get_neurons_number() == 0, LOG);

    ScalingLayer2D scaling_layer_2(3);

    assert_true(scaling_layer_2.get_descriptives().size() == 3, LOG);
    assert_true(scaling_layer_2.get_scaling_methods().size() == 3, LOG);

    descriptives.resize(2);

    ScalingLayer2D scaling_layer_3(descriptives);

    assert_true(scaling_layer_3.get_descriptives().size() == 2, LOG);
}


void ScalingLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    ScalingLayer2D* scaling_layer = new ScalingLayer2D;
    delete scaling_layer;
}


void ScalingLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<Descriptives,1> inputs_descriptives;

    pair<type*, dimensions> inputs_pair;

    // Test
    
    inputs_number = 1;
    samples_number = 1;
    bool is_training = true;

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::NoScaling);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};
    
    scaling_layer.forward_propagate(inputs_pair,
                                    &scaling_layer_forward_propagation,
                                    is_training);
    
    outputs = scaling_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    
    // Test

    inputs_number = 3 + rand()%10;
    samples_number = 1;

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::NoScaling);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    scaling_layer.forward_propagate(inputs_pair,
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;


    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1) - inputs(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(2) - inputs(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    
    // Test

    inputs_number = 1;
    samples_number = 1;

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MinimumMaximum);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    scaling_layer.forward_propagate(inputs_pair,
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0) - inputs(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    
    // Test

    inputs_number = 3;
    samples_number = 3;

    inputs.resize(samples_number,inputs_number);

    inputs.setValues({{type(1),type(1),type(1)},
                    {type(2),type(2),type(2)},
                    {type(3),type(3),type(3)}});

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MinimumMaximum);

    Tensor<Index, 1> all_indices;
    initialize_sequential(all_indices, 0, 1, inputs_number-1);
    
    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    scaling_layer.forward_propagate(inputs_pair,
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    assert_true(abs(outputs(0,0) - type(-1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1,0) - type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(2,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    
    // Test

    inputs_number = 2;
    samples_number = 2;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::MeanStandardDeviation);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({{type(0),type(0)},
                      {type(2),type(2)}});

    initialize_sequential(all_indices, 0, 1, inputs_number-1);

    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    scaling_layer.forward_propagate(inputs_pair,
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    type scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation - inputs_descriptives(0).mean / inputs_descriptives(0).standard_deviation;

    assert_true(abs(outputs(0, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test
    
    inputs_number = 2;
    samples_number = 2;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::StandardDeviation);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({ {type(0),type(0)},
                      {type(2),type(2)} });

    initialize_sequential(all_indices, 0, 1, inputs_number - 1);

    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    scaling_layer.forward_propagate(inputs_pair,
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == inputs_number, LOG);
    assert_true(outputs.dimension(1) == samples_number, LOG);

    scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation;

    assert_true(abs(outputs(0, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 2 + rand()%10;
    samples_number = 1;

    scaling_layer.set(inputs_number);
    scaling_layer.set_display(false);
    scaling_layer.set_scalers(Scaler::StandardDeviation);

    inputs.resize(samples_number, inputs_number);
    inputs.setRandom();

    initialize_sequential(all_indices, 0, 1, inputs_number - 1);

    inputs_descriptives = opennn::descriptives(inputs, all_indices, all_indices);
    scaling_layer.set_descriptives(inputs_descriptives);

    scaling_layer_forward_propagation.set(samples_number, &scaling_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    scaling_layer.forward_propagate(inputs_pair,
                                    &scaling_layer_forward_propagation,
                                    is_training);

    outputs = scaling_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_number, LOG);

    scaled_input = inputs(0, 0) / inputs_descriptives(0).standard_deviation;
    assert_true(abs(outputs(0, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);

    scaled_input = inputs(1, 0) / inputs_descriptives(1).standard_deviation;
    assert_true(abs(outputs(1, 0) - scaled_input) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ScalingLayerTest::run_test_case()
{
    cout << "Running scaling layer test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Scaling and unscaling

    test_forward_propagate();

    cout << "End of scaling layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library sl free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library sl distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
