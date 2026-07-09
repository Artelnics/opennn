#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/recurrent_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

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

    EXPECT_GE(recurrent_layer.get_parameters_number(), parameters_number);
    EXPECT_EQ(recurrent_layer.get_input_shape(), Shape({ time_steps, inputs_number }));
    EXPECT_EQ(recurrent_layer.get_output_shape(), Shape({ neurons_number }));
}


TEST(RecurrentLayerTest, ForwardPropagateValues)
{
    const Index samples_number = 1;
    const Index inputs_number = 2;
    const Index outputs_number = 2;
    const Index time_steps = 3;

    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{outputs_number}, "Identity"));
        neural_network.compile();

        VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, time_steps, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        const TensorView outputs_view = forward_propagation.get_outputs();

        ASSERT_EQ(outputs_view.shape.rank, 2);
        EXPECT_EQ(outputs_view.shape[0], samples_number);
        EXPECT_EQ(outputs_view.shape[1], outputs_number);

        const float* output_data = outputs_view.as<type>();

        type h = type(0.1) * type(1) * type(inputs_number) + type(0.1);
        for (Index t = 1; t < time_steps; ++t)
            h = type(0.1) * type(1) * type(inputs_number) + type(0.1) + h * type(0.1) * type(outputs_number);

        for (Index j = 0; j < outputs_number; ++j)
            EXPECT_NEAR(output_data[j], h, 1.0e-5f);
    }

    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{outputs_number}, "Tanh"));
        neural_network.compile();

        VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, time_steps, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        const TensorView outputs_view = forward_propagation.get_outputs();

        const float* output_data = outputs_view.as<type>();

        type h = std::tanh(type(0.1) * type(1) * type(inputs_number) + type(0.1));
        for (Index t = 1; t < time_steps; ++t)
            h = std::tanh(type(0.1) * type(1) * type(inputs_number) + type(0.1) + h * type(0.1) * type(outputs_number));

        for (Index j = 0; j < outputs_number; ++j)
            EXPECT_NEAR(output_data[j], h, 1.0e-5f);
    }
}


TEST(RecurrentLayerTest, ReturnSequences)
{
    const Index samples_number = 1;
    const Index inputs_number = 2;
    const Index outputs_number = 2;
    const Index time_steps = 3;

    auto layer = make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{outputs_number}, "Identity");
    layer->set_return_sequences(true);

    EXPECT_EQ(layer->get_output_shape(), Shape({time_steps, outputs_number}));

    NeuralNetwork neural_network;
    neural_network.add_layer(std::move(layer));
    neural_network.compile();

    VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(type(0.1));

    Tensor3 inputs(samples_number, time_steps, inputs_number);
    inputs.setConstant(type(1));

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, time_steps, inputs_number}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView outputs_view = forward_propagation.get_outputs();

    ASSERT_EQ(outputs_view.shape.rank, 3);
    EXPECT_EQ(outputs_view.shape[0], samples_number);
    EXPECT_EQ(outputs_view.shape[1], time_steps);
    EXPECT_EQ(outputs_view.shape[2], outputs_number);

    const float* output_data = outputs_view.as<type>();

    type h = type(0.1) * type(1) * type(inputs_number) + type(0.1);
    for (Index t = 0; t < time_steps; ++t)
    {
        if (t > 0)
            h = type(0.1) * type(1) * type(inputs_number) + type(0.1) + h * type(0.1) * type(outputs_number);

        for (Index j = 0; j < outputs_number; ++j)
            EXPECT_NEAR(output_data[t * outputs_number + j], h, 1.0e-5f);
    }
}


TEST(RecurrentLayerTest, BackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index time_steps = 3;
    const Index inputs_number = 4;
    const Index outputs_number = 3;
    const Index targets_number = 2;

    const Shape input_shape{time_steps, inputs_number};

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Recurrent>(input_shape, Shape{outputs_number}, "Tanh"));
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{outputs_number}, Shape{targets_number}, "Identity"));

    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
