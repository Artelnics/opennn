//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "probabilistic_layer_3d_test.h"

#include "../opennn/tensors.h"

namespace opennn
{

ProbabilisticLayer3DTest::ProbabilisticLayer3DTest() : UnitTesting()
{
}


ProbabilisticLayer3DTest::~ProbabilisticLayer3DTest()
{
}


void ProbabilisticLayer3DTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    ProbabilisticLayer3D probabilistic_layer_3d_1;

    assert_true(
        probabilistic_layer_3d_1.get_inputs_number() == 0 &&
        probabilistic_layer_3d_1.get_inputs_depth() == 0 &&
        probabilistic_layer_3d_1.get_neurons_number() == 0 &&
        probabilistic_layer_3d_1.get_biases_number() == 0 &&
        probabilistic_layer_3d_1.get_synaptic_weights_number() == 0 &&
        probabilistic_layer_3d_1.get_parameters_number() == 0, LOG);

    // Probabilistic neurons number constructor

    ProbabilisticLayer3D probabilistic_layer_3d_2;

    probabilistic_layer_3d_2.set_neurons_number(0);

    assert_true(
        probabilistic_layer_3d_2.get_inputs_number() == 0 &&
        probabilistic_layer_3d_2.get_inputs_depth() == 0 &&
        probabilistic_layer_3d_2.get_neurons_number() == 0 &&
        probabilistic_layer_3d_2.get_biases_number() == 0 &&
        probabilistic_layer_3d_2.get_synaptic_weights_number() == 0 &&
        probabilistic_layer_3d_2.get_parameters_number() == 0, LOG);

    ProbabilisticLayer3D probabilistic_layer_3d_3;

    probabilistic_layer_3d_3.set_neurons_number(3);

    assert_true(
        probabilistic_layer_3d_3.get_inputs_number() == 0 &&
        probabilistic_layer_3d_3.get_inputs_depth() == 0 &&
        probabilistic_layer_3d_3.get_neurons_number() == 3 &&
        probabilistic_layer_3d_3.get_biases_number() == 3 &&
        probabilistic_layer_3d_3.get_synaptic_weights_number() == 0 &&
        probabilistic_layer_3d_3.get_parameters_number() == 3, LOG);


    ProbabilisticLayer3D probabilistic_layer_3d_4(1, 2, 3);

    assert_true(
        probabilistic_layer_3d_4.get_inputs_number() == 1 &&
        probabilistic_layer_3d_4.get_inputs_depth() == 2 &&
        probabilistic_layer_3d_4.get_neurons_number() == 3 &&
        probabilistic_layer_3d_4.get_biases_number() == 3 &&
        probabilistic_layer_3d_4.get_synaptic_weights_number() == 6 &&
        probabilistic_layer_3d_4.get_parameters_number() == 9, LOG);
}


void ProbabilisticLayer3DTest::test_destructor()
{
    cout << "test_destructor\n";

    ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D;

    delete probabilistic_layer_3d;
}


void ProbabilisticLayer3DTest::test_calculate_combinations()
{
    cout << "test_calculate_combinations\n";

    Tensor<type, 1> biases(1);
    Tensor<type, 2> synaptic_weights(1, 1);
    biases.setConstant(type(1));
    synaptic_weights.setConstant(type(2));

    Tensor<type, 3> inputs(1, 1, 1);
    inputs.setConstant(type(3));

    Tensor<type, 3> combinations(1, 1, 1);

    probabilistic_layer_3d.set(1, 1, 1);

    probabilistic_layer_3d.set_synaptic_weights(synaptic_weights);
    probabilistic_layer_3d.set_biases(biases);

    probabilistic_layer_3d.calculate_combinations(inputs, combinations);

    assert_true(
        combinations.rank() == 3 &&
        combinations.dimension(0) == 1 &&
        combinations.dimension(1) == 1 &&
        combinations.dimension(2) == 1, LOG);
    assert_true(abs(combinations(0, 0, 0) - type(7)) < type(1e-5), LOG);
}


void ProbabilisticLayer3DTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";
/*
    Tensor<type, 3> combinations;
    Tensor<type, 3> activations;
    Tensor<type, 4> activations_derivatives;

    {
        // Test 1

        samples_number = 1;
        inputs_number = 1;
        inputs_depth = 1;
        neurons_number = 1;

        probabilistic_layer_3d.set(inputs_number, inputs_depth, neurons_number);

        combinations.resize(samples_number, inputs_number, neurons_number);
        combinations.setConstant(type(0));

        activations.resize(samples_number, inputs_number, neurons_number);
        activations.setConstant(type(0));

        activations_derivatives.resize(samples_number, inputs_number, neurons_number, neurons_number);
        activations_derivatives.setConstant(type(0));

        probabilistic_layer_3d.calculate_activations_derivatives(combinations,
                                                                 activations,
                                                                 activations_derivatives);

        assert_true(
            activations_derivatives.rank() == 4 &&
            activations_derivatives.dimension(0) == samples_number &&
            activations_derivatives.dimension(1) == inputs_number &&
            activations_derivatives.dimension(2) == neurons_number &&
            activations_derivatives.dimension(3) == neurons_number, LOG);

        assert_true(activations_derivatives(0, 0, 0, 0) == type(0), LOG);
    }


    {
        // Test 2

        samples_number = 1;
        inputs_number = 1;
        inputs_depth = 1;
        neurons_number = 3;

        probabilistic_layer_3d.set(inputs_number, inputs_depth, neurons_number);

        combinations.resize(samples_number, inputs_number, neurons_number);
        combinations.setValues({ {{1}, {2}, {3}} });

        activations.resize(samples_number, inputs_number, neurons_number);
        activations.setConstant(type(0));

        activations_derivatives.resize(samples_number, inputs_number, neurons_number, neurons_number);
        activations_derivatives.setConstant(type(0));

        probabilistic_layer_3d.calculate_activations_derivatives(combinations,
                                                                 activations,
                                                                 activations_derivatives);

        assert_true(
            activations_derivatives.rank() == 4 &&
            activations_derivatives.dimension(0) == samples_number &&
            activations_derivatives.dimension(1) == inputs_number &&
            activations_derivatives.dimension(2) == neurons_number &&
            activations_derivatives.dimension(3) == neurons_number, LOG);

        assert_true(check_softmax_derivatives(activations, activations_derivatives), LOG);
        
        /*
        assert_true(
            abs(activations_derivatives(0, 0, 0, 0) - (activations(0, 0, 0) - activations(0, 0, 0) * activations(0, 0, 0))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 0, 1) - (-activations(0, 0, 0) * activations(0, 0, 1))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 0, 2) - (-activations(0, 0, 0) * activations(0, 0, 2))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 1, 0) - (-activations(0, 0, 1) * activations(0, 0, 0))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 1, 1) - (activations(0, 0, 1) - activations(0, 0, 1) * activations(0, 0, 1))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 1, 2) - (-activations(0, 0, 1) * activations(0, 0, 2))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 2, 0) - (-activations(0, 0, 2) * activations(0, 0, 0))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 2, 1) - (-activations(0, 0, 2) * activations(0, 0, 1))) < type(1e-3) &&
            abs(activations_derivatives(0, 0, 2, 2) - (activations(0, 0, 2) - activations(0, 0, 2) * activations(0, 0, 2))) < type(1e-3), LOG);
        
    }

    {
        // Test 3

        samples_number = 2;
        inputs_number = 2;
        inputs_depth = 2;
        neurons_number = 3;

        probabilistic_layer_3d.set(inputs_number, inputs_depth, neurons_number);

        combinations.resize(samples_number, inputs_number, neurons_number);
        combinations.setRandom();

        activations.resize(samples_number, inputs_number, neurons_number);
        activations.setConstant(type(0));

        activations_derivatives.resize(samples_number, inputs_number, neurons_number, neurons_number);
        activations_derivatives.setConstant(type(0));

        probabilistic_layer_3d.calculate_activations_derivatives(combinations,
                                                                 activations,
                                                                 activations_derivatives);

        assert_true(
            activations_derivatives.rank() == 4 &&
            activations_derivatives.dimension(0) == samples_number &&
            activations_derivatives.dimension(1) == inputs_number &&
            activations_derivatives.dimension(2) == neurons_number &&
            activations_derivatives.dimension(3) == neurons_number, LOG);

        assert_true(check_softmax_derivatives(activations, activations_derivatives), LOG);
    }*/
}

bool ProbabilisticLayer3DTest::check_softmax_derivatives(Tensor<type, 3>& activations, Tensor<type, 4>& activations_derivatives) const
{
    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < inputs_number; j++)
        {
            for(Index k = 0; k < neurons_number; k++)
            {
                for(Index v = 0; v < neurons_number; v++)
                {
                    if(k == v)
                    {
                        if(activations_derivatives(i, j, k, v) != activations(i, j, k) - activations(i, j, k) * activations(i, j, k))
                            return false;
                    }
                    else
                    {
                        if(activations_derivatives(i, j, k, v) != -activations(i, j, k) * activations(i, j, v))
                            return false;
                    }
                }
            }
        }
    }

    return true;
}


void ProbabilisticLayer3DTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    bool is_training = true;

    {
        // Test 1

        samples_number = 5;
        inputs_number = 2;
        inputs_depth = 3;
        neurons_number = 4;

        probabilistic_layer_3d.set(inputs_number, inputs_depth, neurons_number);

        probabilistic_layer_3d.set_parameters_constant(type(1));

        Tensor<type, 3> inputs(samples_number, inputs_number, inputs_depth);
        inputs.setConstant(type(1));
        
        //Forward propagate

        probabilistic_layer_3d_forward_propagation.set(samples_number, &probabilistic_layer_3d);
        
        probabilistic_layer_3d.forward_propagate(tensor_wrapper(to_pair(inputs)),
                                                 &probabilistic_layer_3d_forward_propagation,
                                                 is_training);
        
        Tensor<type, 3> outputs = probabilistic_layer_3d_forward_propagation.outputs;

        bool correct_outputs = true;

        for(Index i = 0; i < outputs.size(); i++)
            if(abs(outputs(i) - type(0.5) > type(1e-3))) correct_outputs = false;

        assert_true(
            outputs.dimension(0) == samples_number &&
            outputs.dimension(1) == inputs_number &&
            outputs.dimension(2) == neurons_number, LOG);

        assert_true(correct_outputs, LOG);
        
    }

    {
        // Test 2

        samples_number = 1;
        inputs_number = 1;
        inputs_depth = 3;
        neurons_number = 4;
        
        probabilistic_layer_3d.set(inputs_number, inputs_depth, neurons_number);

        Tensor<type, 1> biases(neurons_number);
        Tensor<type, 2> synaptic_weights(inputs_depth, neurons_number);

        biases.setConstant(type(1));

        synaptic_weights.setValues({ {type(1),type(-1),type(0),type(1)},
                                    {type(2),type(-2),type(0),type(2)},
                                    {type(3),type(-3),type(0),type(3)} });
        
        probabilistic_layer_3d.set_synaptic_weights(synaptic_weights);
        probabilistic_layer_3d.set_biases(biases);

        Tensor<type, 3> inputs(samples_number, inputs_number, inputs_depth);
        inputs.setConstant(type(1));
        
        //Forward propagate

        probabilistic_layer_3d_forward_propagation.set(samples_number, &probabilistic_layer_3d);

        probabilistic_layer_3d.forward_propagate(tensor_wrapper(to_pair(inputs)),
                                                 &probabilistic_layer_3d_forward_propagation,
                                                 is_training);

        Tensor<type, 3> outputs = probabilistic_layer_3d_forward_propagation.outputs;

        Tensor<type, 1> combination_solution(4);
        combination_solution.setValues({ type(7),type(-5),type(1),type(7) });

        Tensor<type, 0> softmax_sum = combination_solution.exp().sum();
        Tensor<type, 1> softmax_solution = combination_solution.exp() / softmax_sum(0);

        bool correct_outputs = true;
        
        for(Index i = 0; i < outputs.size(); i++)
            if(abs(outputs(i) - softmax_solution(i) > type(1e-3))) correct_outputs = false;

        assert_true(
            outputs.dimension(0) == samples_number &&
            outputs.dimension(1) == inputs_number &&
            outputs.dimension(2) == neurons_number, LOG);

        assert_true(correct_outputs, LOG);
    }
}


void ProbabilisticLayer3DTest::run_test_case()
{
    cout << "Running probabilistic layer test case...\n";

    // Constructor and destructor

    test_constructor();
    test_destructor();

    // Forward propagate

    test_calculate_combinations();
    test_calculate_activations();

    test_forward_propagate();

    cout << "End of probabilistic layer test case.\n\n";
}

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
