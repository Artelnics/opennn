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


TEST(ProbabilisticLayerTest, CalculateCombinations)
{

    ProbabilisticLayer probabilistic_layer({ 1 }, { 1 });

    EXPECT_EQ(probabilistic_layer.get_input_dimensions(), dimensions{ 1 });
    EXPECT_EQ(probabilistic_layer.get_output_dimensions(), dimensions{ 1 });
    EXPECT_EQ(probabilistic_layer.get_parameters_number(), 2);

    Tensor<type, 2> inputs(1, 1);
    inputs.setConstant(type(1));

    Tensor<type, 2> combinations(1, 1);
    probabilistic_layer.set({ 1 }, { 1 });
    probabilistic_layer.set_parameters_constant(type(1));

    probabilistic_layer.calculate_combinations(inputs, combinations);

    EXPECT_EQ(combinations.rank(), 2);
    EXPECT_EQ(combinations.dimension(0), 1);
    EXPECT_EQ(combinations.dimension(1), 1);
    EXPECT_NEAR(combinations(0, 0),  type(2), NUMERIC_LIMITS_MIN);

}


TEST(ProbabilisticLayerTest, CalculateActivations)
{
    ProbabilisticLayer probabilistic_layer({ 1 }, { 1 });

    Tensor<type, 2> activations(1, 1);
    activations.setConstant({ type(-1.55) });
    Tensor<type, 2> activation_derivatives(1, 1);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Activation::Logistic);

    probabilistic_layer.calculate_activations(activations,activation_derivatives);

    EXPECT_NEAR(abs(activations(0, 0)), type(0.175), type(1e-2));

    EXPECT_EQ(
        activation_derivatives.rank() == 2 &&
            activation_derivatives.dimension(0) == 1 &&
            activation_derivatives.dimension(1) == 1,true);

    EXPECT_NEAR(abs(activation_derivatives(0, 0)), type(0.1444), type(1e-3));

    activations.setConstant({ type(-1.55) });
    probabilistic_layer.set_activation_function(ProbabilisticLayer::Activation::Softmax);
    probabilistic_layer.calculate_activations(activations,activation_derivatives);

    EXPECT_NEAR(abs(activations(0, 0)), type(1), type(1e-2));

    activations.setConstant({ type(-1.55) });
    probabilistic_layer.set_activation_function(ProbabilisticLayer::Activation::Competitive);
    probabilistic_layer.calculate_activations(activations,activation_derivatives);

    EXPECT_NEAR(abs(activations(0, 0)), type(1), type(1e-2));

}

TEST(ProbabilisticLayerTest, ForwardPropagate)
{
    //Test softmax

    Index inputs_number = 2;
    Index neurons_number = 2;
    Index samples_number = 5;

    ProbabilisticLayer probabilistic_layer({ inputs_number }, { neurons_number });

    probabilistic_layer.set_parameters_constant(type(1));

    Tensor<type, 2> inputs(samples_number, inputs_number);
    inputs.setConstant(type(1));

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Activation::Softmax);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<ProbabilisticLayerForwardPropagation>(samples_number, &probabilistic_layer);

    pair<type*, dimensions> input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    bool is_training = true;

    probabilistic_layer.forward_propagate({ input_pairs },
                                          forward_propagation,
                                          is_training);

    pair<type*, dimensions> output_pair = forward_propagation->get_outputs_pair();

    Tensor<type, 2> outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), samples_number);
    EXPECT_EQ(outputs.dimension(1), neurons_number);
    EXPECT_NEAR(abs(outputs(0, 0)), type(0.5), type(1e-3));
    EXPECT_NEAR(abs(outputs(0, 1)), type(0.5), type(1e-3));

    //Test competitive

    inputs_number = 3;
    neurons_number = 4;
    samples_number = 1;

    probabilistic_layer.set({inputs_number}, {neurons_number});

    inputs(samples_number,inputs_number);
    inputs.setConstant(type(1));

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Activation::Competitive);

    forward_propagation = make_unique<ProbabilisticLayerForwardPropagation>(samples_number, &probabilistic_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    probabilistic_layer.set_parameters_constant(type(1));

    probabilistic_layer.forward_propagate({ input_pairs },
                                          forward_propagation,
                                          is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), 1);
    EXPECT_EQ(outputs.dimension(1), 4);
    EXPECT_EQ(Index(outputs(0,0)), 1);
    EXPECT_EQ(Index(outputs(0,1)), 0);
    EXPECT_EQ(Index(outputs(0,2)), 0);
    EXPECT_EQ(Index(outputs(0,3)), 0);

    //Test logistic

    inputs_number = 3;
    neurons_number = 4;
    samples_number = 1;
    Tensor<type, 2> activation_derivatives;

    probabilistic_layer.set({inputs_number}, {neurons_number});

    inputs(samples_number,inputs_number);
    inputs.setConstant(type(1));

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Activation::Logistic);

    forward_propagation = make_unique<ProbabilisticLayerForwardPropagation>(samples_number, &probabilistic_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    probabilistic_layer.forward_propagate({ input_pairs },
                                          forward_propagation,
                                          is_training);

    output_pair = forward_propagation->get_outputs_pair();

    outputs = tensor_map_2(output_pair);

    EXPECT_EQ(outputs.dimension(0), 1);
    EXPECT_EQ(outputs.dimension(1), 4);
    EXPECT_NEAR(outputs(0,0), 0.5, type(0.1));
    EXPECT_NEAR(outputs(0,1), 0.5, type(0.1));
    EXPECT_NEAR(outputs(0,2), 0.5, type(0.1));
    EXPECT_NEAR(outputs(0,3), 0.5, type(0.1));

}
