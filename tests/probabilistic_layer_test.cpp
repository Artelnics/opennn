#include "pch.h"

#include "../opennn/probabilistic_layer.h"

TEST(ProbabilisticLayerTest, DefaultConstructor)
{
    ProbabilisticLayer probabilistic_layer;

    EXPECT_EQ(probabilistic_layer.get_input_dimensions(), dimensions{0});
    EXPECT_EQ(probabilistic_layer.get_output_dimensions(), dimensions{0});
}


TEST(ProbabilisticLayerTest, GeneralConstructor)
{
    ProbabilisticLayer probabilistic_layer({1}, {2});

    EXPECT_EQ(probabilistic_layer.get_input_dimensions(), dimensions{ 1 });
    EXPECT_EQ(probabilistic_layer.get_output_dimensions(), dimensions{ 2 });
    EXPECT_EQ(probabilistic_layer.get_parameters_number(), 4);
}


/*

void ProbabilisticLayerTest::test_calculate_combinations()
{

    Tensor<type, 1> biases(1);
    Tensor<type, 2> synaptic_weights(1, 1);

    biases.setConstant(type(1));
    synaptic_weights.setConstant(type(2));

    Tensor<type, 2> inputs(1, 1);
    inputs.setConstant(type(3));
    
    Tensor<type, 2> combinations(1, 1);
    probabilistic_layer.set(1, 1);

    probabilistic_layer.set_synaptic_weights(synaptic_weights);
    probabilistic_layer.set_biases(biases);

    probabilistic_layer.calculate_combinations(inputs, combinations);

    EXPECT_EQ(combinations.rank() == 2
             && combinations.dimension(0) == 1
             && combinations.dimension(1) == 1);

    EXPECT_EQ(abs(combinations(0, 0) - type(7)) < type(1e-5));
}


void ProbabilisticLayerTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";

    // Test

    samples_number = 1;
    inputs_number = 1;
    neurons_number = 1;

    probabilistic_layer.set(inputs_number, neurons_number);

    combinations.resize(samples_number, neurons_number);
    combinations.setConstant({ type(-1.55) });

    activations.resize(samples_number, neurons_number);
    activation_derivatives.resize(samples_number, neurons_number);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Logistic);

    probabilistic_layer.calculate_activations(combinations,
                                              activation_derivatives);

    EXPECT_EQ(abs(activations(0, 0) - type(0.175)) < type(1e-2));

    EXPECT_EQ(
        activation_derivatives.rank() == 2 &&
        activation_derivatives.dimension(0) == inputs_number &&
        activation_derivatives.dimension(1) == neurons_number);

    EXPECT_EQ(abs(activation_derivatives(0, 0) - type(0.1444)) < type(1e-3));

}


void ProbabilisticLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    inputs_number = 2;
    neurons_number = 2;
    samples_number = 5;

    probabilistic_layer.set(inputs_number, neurons_number);

    probabilistic_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    //Forward propagate

    probabilistic_layer_forward_propagation.set(samples_number, &probabilistic_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    input_pairs.first = inputs.data();
    input_pairs.second = {{samples_number, inputs_number}};

    probabilistic_layer.forward_propagate({input_pairs},
                                          &probabilistic_layer_forward_propagation,
                                          is_training);

    outputs = probabilistic_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs.dimension(0) == samples_number);
    EXPECT_EQ(outputs.dimension(1) == neurons_number );
    EXPECT_EQ(abs(outputs(0,0) - type(0.5)) < type(1e-3));
    EXPECT_EQ(abs(outputs(0,1) - type(0.5)) < type(1e-3));
    
    // Test 1

    inputs_number = 3;
    neurons_number = 4;
    samples_number = 1;

    probabilistic_layer.set(inputs_number, neurons_number);

    Tensor<type, 2> inputs_test_1_tensor(samples_number,inputs_number);
    inputs_test_1_tensor.setConstant(type(1));

    synaptic_weights.resize(3, 4);
    biases.resize( 4);

    biases.setConstant(type(1));

    synaptic_weights.setValues({{type(1),type(-1),type(0),type(1)},
                                {type(2),type(-2),type(0),type(2)},
                                {type(3),type(-3),type(0),type(3)}});

    probabilistic_layer.set_synaptic_weights(synaptic_weights);
    probabilistic_layer.set_biases(biases);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    probabilistic_layer_forward_propagation.set(samples_number, &probabilistic_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    probabilistic_layer.forward_propagate({input_pairs},
                                          &probabilistic_layer_forward_propagation,
                                          is_training);

    Tensor<type, 1> perceptron_sol(4);
    perceptron_sol.setValues({ type(7),type(-5),type(1),type(7)});

    Tensor<type,0> div = perceptron_sol.exp().sum();
    Tensor<type, 1> sol_ = perceptron_sol.exp() / div(0);

    TensorMap<Tensor<type, 2>> outputs_test_1 = probabilistic_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs_test_1.rank() == 2);
    EXPECT_EQ(outputs_test_1.dimension(0) == 1);
    EXPECT_EQ(outputs_test_1.dimension(1) == 4);
    EXPECT_EQ(abs(outputs_test_1(0, 0) - sol_(0)) < type(1e-3));
    EXPECT_EQ(abs(outputs_test_1(1, 0) - sol_(1)) < type(1e-3));
    EXPECT_EQ(abs(outputs_test_1(2, 0) - sol_(2)) < type(1e-3));
    EXPECT_EQ(abs(outputs_test_1(3, 0) - sol_(3)) < type(1e-3));

    // Test 2

    inputs_number = 3;
    neurons_number = 4;
    samples_number = 1;
    is_training = false;

    probabilistic_layer.set(inputs_number, neurons_number);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Competitive);

    probabilistic_layer_forward_propagation.set(samples_number,
                                                &probabilistic_layer);

    Tensor<type, 2> inputs_test_2_tensor(samples_number,inputs_number);
    inputs_test_2_tensor.setConstant(type(1));

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    probabilistic_layer.forward_propagate({input_pairs},
                                          &probabilistic_layer_forward_propagation,
                                          is_training);

    Tensor<type, 2> outputs_test_2 = probabilistic_layer_forward_propagation.outputs;

    EXPECT_EQ(outputs_test_2.dimension(0) == 1);
    EXPECT_EQ(outputs_test_2.dimension(1) == 4);
    EXPECT_EQ(Index(outputs_test_2(0,0)) == 1);
    EXPECT_EQ(Index(outputs_test_2(0,1)) == 0);
    EXPECT_EQ(Index(outputs_test_2(0,2)) == 0);
    EXPECT_EQ(Index(outputs_test_2(0,3)) == 0);

    // Test 3

    inputs_number = 2;
    neurons_number = 4;
    samples_number = 1;
    is_training = true;

    biases.resize( 4);
    biases.setValues({type(9),type(-8),type(7),type(-6)});

    synaptic_weights.resize(2, 4);

    synaptic_weights.resize(2, 4);

    synaptic_weights.setValues({{type(-11), type(12), type(-13), type(14)},
                                {type(21), type(-22), type(23), type(-24)}});

    probabilistic_layer.set_synaptic_weights(synaptic_weights);
    probabilistic_layer.set_biases(biases);

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    probabilistic_layer.set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    probabilistic_layer.forward_propagate({input_pairs}, &probabilistic_layer_forward_propagation, is_training);

    outputs = probabilistic_layer_forward_propagation.outputs;

    Tensor<type, 1>perceptron_sol_3(4);
    perceptron_sol.setValues({type(19),type(-18),type(17),type(-16)});

    div = perceptron_sol.exp().sum();
    sol_ = perceptron_sol.exp() / div(0);
    
    EXPECT_EQ(outputs.dimension(0) == 1);
    EXPECT_EQ(outputs.dimension(1) == 4);
    EXPECT_EQ(abs(outputs(0,0) - sol_(0)) < type(1e-3));
    EXPECT_EQ(abs(outputs(1,0) - sol_(1)) < type(1e-3));
    EXPECT_EQ(abs(outputs(2,0) - sol_(2)) < type(1e-3));
    EXPECT_EQ(abs(outputs(3,0) - sol_(3)) < type(1e-3));
    
    // Test  4

    samples_number = 1;
    inputs_number = 3;
    neurons_number = 2;

    probabilistic_layer.set(inputs_number, neurons_number);
    probabilistic_layer.set_parameters_constant(type(0));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(0));

    probabilistic_layer_forward_propagation.set(samples_number, &probabilistic_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    probabilistic_layer.forward_propagate({input_pairs}, &probabilistic_layer_forward_propagation, is_training);

    outputs = probabilistic_layer_forward_propagation.outputs;
    
    EXPECT_EQ(outputs.dimension(0) == 1);
    EXPECT_EQ(outputs.dimension(1) == 2);
    EXPECT_EQ(abs(outputs(0,0) - type(0.5)) < type(NUMERIC_LIMITS_MIN));

}

}
*/
