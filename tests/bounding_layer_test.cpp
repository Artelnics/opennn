//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   T E S T   C L A S S                     
//
//   Artificial Intelligence Techniques SL
//   E-mail: artelnics@artelnics.com                                       

#include "bounding_layer_test.h"


BoundingLayerTest::BoundingLayerTest() : UnitTesting()
{
}


BoundingLayerTest::~BoundingLayerTest()
{
}


void BoundingLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    BoundingLayer bounding_layer_1;

    assert_true(bounding_layer_1.get_neurons_number() == 0, LOG);
}


void BoundingLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    BoundingLayer* bounding_layer_1 = new BoundingLayer;

    delete bounding_layer_1;
}


void BoundingLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    BoundingLayer bounding_layer;
    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<Index, 1> inputs_dimensions;
    Tensor<Index, 1> outputs_dimensions;

    BoundingLayerForwardPropagation bounding_layer_forward_propagation;

    // Test

    Index samples_number = 1;
    Index inputs_number = 1;
    bool switch_train = false;

    bounding_layer.set(inputs_number);

    bounding_layer.set_lower_bound(0, type(-1.0));
    bounding_layer.set_upper_bound(0, type(1));
    bounding_layer.set_bounding_method("Bounding");

    inputs.resize(1, 1);
    inputs(0) = type(-2.0);
    inputs_dimensions = get_dimensions(inputs);

    bounding_layer_forward_propagation.set(samples_number, &bounding_layer);
    bounding_layer.forward_propagate(inputs.data(), inputs_dimensions, &bounding_layer_forward_propagation, switch_train);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs(0) - type(-1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs(0) = type(2.0);

    bounding_layer_forward_propagation.set(samples_number, &bounding_layer);
    bounding_layer.forward_propagate(inputs.data(), inputs_dimensions, &bounding_layer_forward_propagation, switch_train);

    TensorMap<Tensor<type, 2>> outputs_2(bounding_layer_forward_propagation.outputs_data,
                                       bounding_layer_forward_propagation.outputs_dimensions(0),
                                       bounding_layer_forward_propagation.outputs_dimensions(1));

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs(0) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
}


void BoundingLayerTest::run_test_case()
{
    cout << "Running bounding layer test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Lower and upper bounds

    test_calculate_outputs();

    cout << "End of bounding layer test case.\n\n";
}


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
