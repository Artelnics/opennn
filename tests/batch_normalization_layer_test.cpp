//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M A L I Z A T I O N   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch_normalization_layer_test.h"

BatchNormalizationLayerTest::BatchNormalizationLayerTest() : UnitTesting()
{

}


BatchNormalizationLayerTest::~BatchNormalizationLayerTest()
{

}


void BatchNormalizationLayerTest::test_perform_inputs_normalization()
{

    const int inputs_number = 3;
    const int batch_number = 3;

    Tensor<type, 2> inputs(inputs_number,batch_number);

    inputs.setValues({{1,2,3},
                      {4,5,6},
                      {7,8,9}});

    BatchNormalizationLayer *batch_norm_layer;
    batch_norm_layer->set(inputs_number);

    BatchNormalizationLayerForwardPropagation forward_propagation;
    forward_propagation.set(batch_number, batch_norm_layer);

    Tensor<type, 2> inputs_normalized = batch_norm_layer->perform_inputs_normalization(inputs, &forward_propagation);

    assert_true(forward_propagation.mean[0] == 4, LOG);
    assert_true(forward_propagation.variance[0] == 6, LOG);

    assert_true(forward_propagation.mean[1] == 5, LOG);
    assert_true(forward_propagation.variance[1] == 6, LOG);

    assert_true(forward_propagation.mean[2] == 6, LOG);
    assert_true(forward_propagation.variance[2] == 6, LOG);

    assert_true(abs(inputs_normalized(0) - type(-1.22474)) < type(0.00001), LOG);
    assert_true(abs(inputs_normalized(1) - type(0)) < type(0.00001), LOG);
    assert_true(abs(inputs_normalized(2) - type(1.22474)) < type(0.00001), LOG);

    assert_true(abs(inputs_normalized(3) - type(-1.22474)) < type(0.00001), LOG);
    assert_true(abs(inputs_normalized(4) - type(0)) < type(0.00001), LOG);
    assert_true(abs(inputs_normalized(5) - type(1.22474)) < type(0.00001), LOG);

    assert_true(abs(inputs_normalized(6) - type(-1.22474)) < type(0.00001), LOG);
    assert_true(abs(inputs_normalized(7) - type(0)) < type(0.00001), LOG);
    assert_true(abs(inputs_normalized(8) - type(1.22474)) < type(0.00001), LOG);

}

//Tensor<type, 2> inputs_normalized = perform_inputs_normalization(inputs, batch_norm_layer_forward_propagation);

//void BatchNormalizationLayerTest::test_calculate_combinations()
//{

//}

//void PerceptronLayerTest::test_forward_propagate()
//{
//    cout << "test_forward_propagate\n";


//    assert_true(forward_propagation.combinations.rank() == 2, LOG);
//    assert_true(forward_propagation.combinations.dimension(0) == samples_number, LOG);
//    assert_true(forward_propagation.combinations.dimension(1) == neurons_number, LOG);
//    assert_true(abs(forward_propagation.combinations(0,0) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
//    assert_true(abs(forward_propagation.combinations(0,1) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
//    assert_true(abs(outputs_2(0,0) - static_cast<type>(0.99505)) < static_cast<type>(1e-3), LOG);
//    assert_true(abs(outputs_2(0,1) - static_cast<type>(0.99505)) < static_cast<type>(1e-3), LOG);
//    assert_true(abs(forward_propagation.activations_derivatives(0,0) - static_cast<type>(0.00986)) < static_cast<type>(1e-3), LOG);
//    assert_true(abs(forward_propagation.activations_derivatives(0,1) - static_cast<type>(0.00986)) < static_cast<type>(1e-3), LOG);

//}


void BatchNormalizationLayerTest::run_test_case()
{
    cout << "Running perceptron layer test case...\n";

    // Constructor and destructor

    test_perform_inputs_normalization();

    // Set methods

//    test_set();
//    test_set_default();

//    // Perceptron layer parameters

//    test_set_biases();
//    test_set_synaptic_weights();
//    test_set_perceptrons_number();

//    // Inputs

//    test_set_inputs_number();

//    // Activation functions

//    test_set_activation_function();

//    // Parameters methods

//    test_set_parameters();

//    // Parameters initialization methods

//    test_set_parameters_constant();
//    test_set_parameters_random();

//    // Combinations

//    test_calculate_combinations();

//    // Activation

//    test_calculate_activations();
//    test_calculate_activations_derivatives();

//    // Outputs

//    test_calculate_outputs();

//    // Forward propagate

//    test_forward_propagate();

//    cout << "End of perceptron layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2023 Artificial Intelligence Techniques, SL.
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
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
