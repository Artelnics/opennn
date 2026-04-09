#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/recurrent_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

TEST(RecurrentLayerTest, DefaultConstructor)
{
    Recurrent recurrent_layer;

    EXPECT_EQ(recurrent_layer.get_inputs_number(), 0);
    EXPECT_EQ(recurrent_layer.get_outputs_number(), 0);
}


TEST(RecurrentLayerTest, GeneralConstructor)
{
    const Index inputs_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);
    const Index time_steps = random_integer(1, 10);

    Recurrent recurrent_layer({ time_steps, inputs_number }, { neurons_number });

    const Index parameters_number = neurons_number + (inputs_number + neurons_number) * neurons_number;

    // Aligned parameter count >= raw parameter count
    EXPECT_GE(recurrent_layer.get_parameters_number(), parameters_number);
    EXPECT_EQ(recurrent_layer.get_input_shape(), Shape({ time_steps, inputs_number }));
    EXPECT_EQ(recurrent_layer.get_output_shape(), Shape({ neurons_number }));
}


TEST(RecurrentLayerTest, ForwardPropagate)
{
    Index outputs_number = 8;
    Index samples_number = 3;
    Index inputs_number = 8;
    Index time_steps = 3;

    // Test HyperbolicTangent
    {
        NeuralNetwork neural_network;
        auto layer = make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{outputs_number});
        layer->set_activation_function("HyperbolicTangent");
        neural_network.add_layer(std::move(layer));
        neural_network.compile();

        // Set all parameters to 0.1
        VectorR& params = neural_network.get_parameters();
        params.setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, time_steps, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, true);

        TensorView outputs_view = forward_propagation.get_outputs();

        EXPECT_EQ(outputs_view.shape[0], samples_number);
    }

    // Test Sigmoid
    {
        NeuralNetwork neural_network;
        auto layer = make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{outputs_number});
        layer->set_activation_function("Sigmoid");
        neural_network.add_layer(std::move(layer));
        neural_network.compile();

        VectorR& params = neural_network.get_parameters();
        params.setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, time_steps, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, true);

        TensorView outputs_view = forward_propagation.get_outputs();

        EXPECT_EQ(outputs_view.shape[0], samples_number);
    }

    // Test Linear
    {
        NeuralNetwork neural_network;
        auto layer = make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{outputs_number});
        layer->set_activation_function("Linear");
        neural_network.add_layer(std::move(layer));
        neural_network.compile();

        VectorR& params = neural_network.get_parameters();
        params.setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, time_steps, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, true);

        TensorView outputs_view = forward_propagation.get_outputs();

        EXPECT_EQ(outputs_view.shape[0], samples_number);
    }
}
