#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/perceptron_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

TEST(Dense2dTest, DefaultConstructor)
{
    Dense2d perceptron_layer;

    EXPECT_EQ(perceptron_layer.get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(perceptron_layer.get_input_dimensions(), dimensions{ 0 });
    EXPECT_EQ(perceptron_layer.get_output_dimensions(), dimensions{ 0 });
}


TEST(Dense2dTest, GeneralConstructor)
{
    Dense2d perceptron_layer({10}, {3}, Dense2d::Activation::Linear);
    
    EXPECT_EQ(perceptron_layer.get_activation_function_string(), "Linear");
    EXPECT_EQ(perceptron_layer.get_input_dimensions(), dimensions{ 10 });
    EXPECT_EQ(perceptron_layer.get_output_dimensions(), dimensions{ 3 });
    EXPECT_EQ(perceptron_layer.get_parameters_number(), 33);
}


TEST(Dense2dTest, Combinations)
{
    
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index outputs_number = get_random_index(1, 10);

    Dense2d perceptron_layer({inputs_number}, {outputs_number});
    perceptron_layer.set_parameters_random();

    Tensor<type, 2> inputs(samples_number, inputs_number);
    inputs.setRandom();

    Tensor<type, 2> combinations(samples_number, outputs_number);

    perceptron_layer.calculate_combinations(inputs, combinations);

    EXPECT_EQ(combinations.dimension(0), samples_number);
    EXPECT_EQ(combinations.dimension(1), outputs_number);

}


TEST(Dense2dTest, Activations)
{
    Dense2d perceptron_layer({ 1 }, { 1 });
    perceptron_layer.set_parameters_random();

    Tensor<type, 2> activations(1, 1);
    Tensor<type, 2> activation_derivatives(1, 1);

    perceptron_layer.set_activation_function(Dense2d::Activation::Logistic);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(0.731), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.196), 0.001);
    
    perceptron_layer.set_activation_function(Dense2d::Activation::HyperbolicTangent);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(0.761), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(0.41997), 0.001);
    
    perceptron_layer.set_activation_function(Dense2d::Activation::Linear);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);
    
    perceptron_layer.set_activation_function(Dense2d::Activation::RectifiedLinear);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);
    
    perceptron_layer.set_activation_function(Dense2d::Activation::ExponentialLinear);
    activations.setConstant(type(1));
    perceptron_layer.calculate_activations(activations, activation_derivatives);

    EXPECT_NEAR(activations(0, 0), type(1), 0.001);
    EXPECT_NEAR(activation_derivatives(0, 0), type(1), 0.001);
}


TEST(Dense2dTest, ForwardPropagate)
{
    const Index batch_size = 2;
    const Index inputs_number = 2;
    const Index neurons_number = 2;
    const Index outputs_number = 2;

    bool is_training = true;

    Tensor<type, 2> inputs(batch_size, inputs_number);
    inputs.setConstant(1);

    Dense2d dense2d_layer(
        {inputs_number},
        {outputs_number},
        Dense2d::Activation::Linear);
    dense2d_layer.set_parameters_random();

    unique_ptr<LayerForwardPropagation> forward_propagate =
        make_unique<Dense2dForwardPropagation>(batch_size, &dense2d_layer);

    dense2d_layer.forward_propagate(
        { { inputs.data(), dimensions{batch_size, inputs_number} } },
        forward_propagate,
        is_training);

    Dense2dForwardPropagation* forward_ptr =
        static_cast<Dense2dForwardPropagation*>(forward_propagate.get());

<<<<<<< HEAD
    pair<type*, dimensions> output_pair = forward_propagation->get_output_pair();
=======
    const Tensor<type, 2>& outputs = forward_ptr->outputs;
    const Tensor<type, 2>& activation_derivatives = forward_ptr->activation_derivatives;
>>>>>>> f2ec3fa0d (wip)

    EXPECT_EQ(dense2d_layer.get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(dense2d_layer.get_input_dimensions(), dimensions({inputs_number}));
    EXPECT_EQ(dense2d_layer.get_output_dimensions(), dimensions({outputs_number}));

    EXPECT_NEAR(abs(activation_derivatives(0,0)), type(1), type(1e-3));
}

