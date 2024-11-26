#include "pch.h"

/*

void ProbabilisticLayer3DTest::test_constructor()
{
    cout << "test_constructor\n";

    ProbabilisticLayer3D probabilistic_layer_3d_1;

    EXPECT_EQ(
        probabilistic_layer_3d_1.get_inputs_number() == 0 &&
        probabilistic_layer_3d_1.get_inputs_depth() == 0 &&
        probabilistic_layer_3d_1.get_neurons_number() == 0 &&
        probabilistic_layer_3d_1.get_parameters_number() == 0);

    // Probabilistic neurons number constructor

    ProbabilisticLayer3D probabilistic_layer_3d_2;

    probabilistic_layer_3d_2.set_output_dimensions(0);

    EXPECT_EQ(
        probabilistic_layer_3d_2.get_inputs_number() == 0 &&
        probabilistic_layer_3d_2.get_inputs_depth() == 0 &&
        probabilistic_layer_3d_2.get_neurons_number() == 0 &&
        probabilistic_layer_3d_2.get_parameters_number() == 0);

    ProbabilisticLayer3D probabilistic_layer_3d_3;

    probabilistic_layer_3d_3.set_output_dimensions(3);

    EXPECT_EQ(
        probabilistic_layer_3d_3.get_inputs_number() == 0 &&
        probabilistic_layer_3d_3.get_inputs_depth() == 0 &&
        probabilistic_layer_3d_3.get_neurons_number() == 3 &&
        probabilistic_layer_3d_3.get_parameters_number() == 3);


    ProbabilisticLayer3D probabilistic_layer_3d_4(1, 2, 3);

    EXPECT_EQ(
        probabilistic_layer_3d_4.get_inputs_number() == 1 &&
        probabilistic_layer_3d_4.get_inputs_depth() == 2 &&
        probabilistic_layer_3d_4.get_neurons_number() == 3 &&
        probabilistic_layer_3d_4.get_parameters_number() == 9);

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

    EXPECT_EQ(
        combinations.rank() == 3 &&
        combinations.dimension(0) == 1 &&
        combinations.dimension(1) == 1 &&
        combinations.dimension(2) == 1);
    EXPECT_EQ(abs(combinations(0, 0, 0) - type(7)) < type(1e-5));
}


void ProbabilisticLayer3DTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";

    Tensor<type, 3> combinations;
    Tensor<type, 3> activations;
    Tensor<type, 4> activation_derivatives;

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

        activation_derivatives.resize(samples_number, inputs_number, neurons_number, neurons_number);
        activation_derivatives.setConstant(type(0));

        probabilistic_layer_3d.calculate_activations_derivatives(combinations,
                                                                 activations,
                                                                 activation_derivatives);

        EXPECT_EQ(
            activation_derivatives.rank() == 4 &&
            activation_derivatives.dimension(0) == samples_number &&
            activation_derivatives.dimension(1) == inputs_number &&
            activation_derivatives.dimension(2) == neurons_number &&
            activation_derivatives.dimension(3) == neurons_number);

        EXPECT_EQ(activation_derivatives(0, 0, 0, 0) == type(0));
    }


    {
        // Test 2

        samples_number = 1;
        inputs_number = 1;
        inputs_depth = 1;
        neurons_number = 3;

        probabilistic_layer_3d.set(inputs_number, inputs_depth, neurons_number);

        combinations.resize(samples_number, inputs_number, neurons_number);
        combinations.setValues({{{1}, {2}, {3}}});

        activations.resize(samples_number, inputs_number, neurons_number);
        activations.setConstant(type(0));

        activation_derivatives.resize(samples_number, inputs_number, neurons_number, neurons_number);
        activation_derivatives.setConstant(type(0));

        probabilistic_layer_3d.calculate_activations_derivatives(combinations,
                                                                 activations,
                                                                 activation_derivatives);

        EXPECT_EQ(
            activation_derivatives.rank() == 4 &&
            activation_derivatives.dimension(0) == samples_number &&
            activation_derivatives.dimension(1) == inputs_number &&
            activation_derivatives.dimension(2) == neurons_number &&
            activation_derivatives.dimension(3) == neurons_number);

        EXPECT_EQ(check_softmax_derivatives(activations, activation_derivatives));
        
        /*
        EXPECT_EQ(
            abs(activation_derivatives(0, 0, 0, 0) - (activations(0, 0, 0) - activations(0, 0, 0) * activations(0, 0, 0))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 0, 1) - (-activations(0, 0, 0) * activations(0, 0, 1))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 0, 2) - (-activations(0, 0, 0) * activations(0, 0, 2))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 1, 0) - (-activations(0, 0, 1) * activations(0, 0, 0))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 1, 1) - (activations(0, 0, 1) - activations(0, 0, 1) * activations(0, 0, 1))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 1, 2) - (-activations(0, 0, 1) * activations(0, 0, 2))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 2, 0) - (-activations(0, 0, 2) * activations(0, 0, 0))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 2, 1) - (-activations(0, 0, 2) * activations(0, 0, 1))) < type(1e-3) &&
            abs(activation_derivatives(0, 0, 2, 2) - (activations(0, 0, 2) - activations(0, 0, 2) * activations(0, 0, 2))) < type(1e-3));
        
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

        activation_derivatives.resize(samples_number, inputs_number, neurons_number, neurons_number);
        activation_derivatives.setConstant(type(0));

        probabilistic_layer_3d.calculate_activations_derivatives(combinations,
                                                                 activations,
                                                                 activation_derivatives);

        EXPECT_EQ(
            activation_derivatives.rank() == 4 &&
            activation_derivatives.dimension(0) == samples_number &&
            activation_derivatives.dimension(1) == inputs_number &&
            activation_derivatives.dimension(2) == neurons_number &&
            activation_derivatives.dimension(3) == neurons_number);

        EXPECT_EQ(check_softmax_derivatives(activations, activation_derivatives));
    }
}

bool ProbabilisticLayer3DTest::check_softmax_derivatives(Tensor<type, 3>& activations, Tensor<type, 4>& activation_derivatives) const
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
                        if(activation_derivatives(i, j, k, v) != activations(i, j, k) - activations(i, j, k) * activations(i, j, k))
                            return false;
                    }
                    else
                    {
                        if(activation_derivatives(i, j, k, v) != -activations(i, j, k) * activations(i, j, v))
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

        EXPECT_EQ(
            outputs.dimension(0) == samples_number &&
            outputs.dimension(1) == inputs_number &&
            outputs.dimension(2) == neurons_number);

        EXPECT_EQ(correct_outputs);
       
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
                                    {type(3),type(-3),type(0),type(3)}});

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

        EXPECT_EQ(
            outputs.dimension(0) == samples_number &&
            outputs.dimension(1) == inputs_number &&
            outputs.dimension(2) == neurons_number);

        EXPECT_EQ(correct_outputs);

    }
}

}
*/