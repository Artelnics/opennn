#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/long_short_term_memory_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

TEST(LongShortTermMemoryLayerTest, DefaultConstructor)
{
    LongShortTermMemory lstm_layer;

    EXPECT_EQ(lstm_layer.get_inputs_number(), 0);
    EXPECT_EQ(lstm_layer.get_outputs_number(), 0);
}


TEST(LongShortTermMemoryLayerTest, GeneralConstructor)
{
    const Index inputs_number   = random_integer(1, 10);
    const Index neurons_number  = random_integer(1, 10);
    const Index time_steps      = random_integer(1, 10);

    LongShortTermMemory lstm_layer({ time_steps, inputs_number }, { neurons_number });

    const Index parameters_number =
        4 * neurons_number * (1 + inputs_number + neurons_number);

    EXPECT_GE(lstm_layer.get_parameters_number(), parameters_number);
    EXPECT_EQ(lstm_layer.get_input_shape(),  Shape({ time_steps, inputs_number }));
    EXPECT_EQ(lstm_layer.get_output_shape(), Shape({ neurons_number }));
}


TEST(LongShortTermMemoryLayerTest, ReturnSequencesOutputShape)
{
    const Index inputs_number  = 4;
    const Index neurons_number = 6;
    const Index time_steps     = 5;

    LongShortTermMemory lstm_layer({ time_steps, inputs_number }, { neurons_number });

    EXPECT_FALSE(lstm_layer.get_return_sequences());
    EXPECT_EQ(lstm_layer.get_output_shape(), Shape({ neurons_number }));

    lstm_layer.set_return_sequences(true);
    EXPECT_TRUE(lstm_layer.get_return_sequences());
    EXPECT_EQ(lstm_layer.get_output_shape(), Shape({ time_steps, neurons_number }));
}


TEST(LongShortTermMemoryLayerTest, ForwardPropagate)
{
    const Index outputs_number = 8;
    const Index samples_number = 3;
    const Index inputs_number  = 8;
    const Index time_steps     = 3;

    const vector<string> cell_activations = {"Tanh", "Sigmoid", "Identity"};

    for (const string& act : cell_activations)
    {
        NeuralNetwork neural_network;
        auto layer = make_unique<LongShortTermMemory>(
            Shape{time_steps, inputs_number}, Shape{outputs_number});
        layer->set_activation_function(act);
        layer->set_recurrent_activation_function("Sigmoid");
        neural_network.add_layer(std::move(layer));
        neural_network.compile();

        VectorMap(neural_network.get_parameters_data(),
                  neural_network.get_parameters_size()).setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = {
            TensorView(inputs.data(),
                       {samples_number, time_steps, inputs_number})
        };
        neural_network.forward_propagate(input_views, forward_propagation, true);

        TensorView outputs_view = forward_propagation.get_outputs();
        EXPECT_EQ(outputs_view.shape[0], samples_number)
            << "cell activation=" << act;
        EXPECT_EQ(outputs_view.shape[1], outputs_number)
            << "cell activation=" << act;
    }
}


TEST(LongShortTermMemoryLayerTest, ForwardPropagateValues)
{
    const Index samples_number = 2;
    const Index inputs_number  = 2;
    const Index outputs_number = 2;
    const Index time_steps     = 2;

    const type expected_t0 = type(0.09524119);
    const type expected_t1 = type(0.15569832);

    {
        NeuralNetwork neural_network;
        auto layer = make_unique<LongShortTermMemory>(
            Shape{time_steps, inputs_number}, Shape{outputs_number});
        layer->set_activation_function("Tanh");
        layer->set_recurrent_activation_function("Sigmoid");
        neural_network.add_layer(std::move(layer));
        neural_network.compile();

        VectorMap(neural_network.get_parameters_data(),
                  neural_network.get_parameters_size()).setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = {
            TensorView(inputs.data(), {samples_number, time_steps, inputs_number})
        };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        const TensorView outputs_view = forward_propagation.get_outputs();
        ASSERT_EQ(outputs_view.shape.rank, size_t(2));
        EXPECT_EQ(outputs_view.shape[0], samples_number);
        EXPECT_EQ(outputs_view.shape[1], outputs_number);

        const float* output_data = outputs_view.as<type>();
        for (Index i = 0; i < samples_number * outputs_number; ++i)
            EXPECT_NEAR(output_data[i], expected_t1, type(1.0e-5));
    }

    {
        NeuralNetwork neural_network;
        auto layer = make_unique<LongShortTermMemory>(
            Shape{time_steps, inputs_number}, Shape{outputs_number});
        layer->set_activation_function("Tanh");
        layer->set_recurrent_activation_function("Sigmoid");
        layer->set_return_sequences(true);
        neural_network.add_layer(std::move(layer));
        neural_network.compile();

        VectorMap(neural_network.get_parameters_data(),
                  neural_network.get_parameters_size()).setConstant(type(0.1));

        Tensor3 inputs(samples_number, time_steps, inputs_number);
        inputs.setConstant(type(1));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = {
            TensorView(inputs.data(), {samples_number, time_steps, inputs_number})
        };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        const TensorView outputs_view = forward_propagation.get_outputs();
        ASSERT_EQ(outputs_view.shape.rank, size_t(3));
        EXPECT_EQ(outputs_view.shape[0], samples_number);
        EXPECT_EQ(outputs_view.shape[1], time_steps);
        EXPECT_EQ(outputs_view.shape[2], outputs_number);

        const float* output_data = outputs_view.as<type>();
        for (Index b = 0; b < samples_number; ++b)
            for (Index h = 0; h < outputs_number; ++h)
            {
                const Index base = (b * time_steps + 0) * outputs_number + h;
                EXPECT_NEAR(output_data[base], expected_t0, type(1.0e-5));
                const Index base1 = (b * time_steps + 1) * outputs_number + h;
                EXPECT_NEAR(output_data[base1], expected_t1, type(1.0e-5));
            }
    }
}


TEST(LongShortTermMemoryLayerTest, BackPropagate)
{
    const Index samples_number = 4;
    const Index inputs_number  = 3;
    const Index targets_number = 2;
    const Index time_steps     = 3;

    TabularDataset dataset(samples_number, {time_steps, inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<LongShortTermMemory>(
        Shape{time_steps, inputs_number}, Shape{targets_number}));
    neural_network.compile();

    VectorMap(neural_network.get_parameters_data(),
              neural_network.get_parameters_size()).setConstant(type(0.05));

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}


TEST(LongShortTermMemoryLayerTest, BackPropagateReturnSequences)
{
    const Index samples_number = 4;
    const Index inputs_number  = 3;
    const Index neurons_number = 2;
    const Index time_steps     = 3;

    const Shape input_shape{time_steps, inputs_number};

    TabularDataset dataset(samples_number, input_shape, {time_steps, neurons_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    auto layer = make_unique<LongShortTermMemory>(input_shape, Shape{neurons_number});
    layer->set_return_sequences(true);
    neural_network.add_layer(std::move(layer));
    neural_network.compile();

    VectorMap(neural_network.get_parameters_data(),
              neural_network.get_parameters_size()).setConstant(type(0.05));

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}


TEST(LongShortTermMemoryLayerTest, UnsupportedActivationThrows)
{
    LongShortTermMemory lstm_layer(Shape{3, 4}, Shape{5});

    EXPECT_THROW(lstm_layer.set_activation_function("Softmax"), std::runtime_error);
    EXPECT_THROW(lstm_layer.set_recurrent_activation_function("Softmax"), std::runtime_error);
}


TEST(LongShortTermMemoryLayerTest, ForgetBiasInitialisedToOne)
{
    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<LongShortTermMemory>(Shape{3, 2}, Shape{4}));
    neural_network.compile();
    neural_network.set_parameters_glorot();

    const auto* lstm = dynamic_cast<const LongShortTermMemory*>(neural_network.get_layer(0).get());
    ASSERT_NE(lstm, nullptr);

    const TensorView& bf = lstm->get_forget_bias();
    ASSERT_NE(bf.data, nullptr);
    ASSERT_GT(bf.shape.size(), 0);
    const float* bf_data = bf.as<float>();
    for (Index i = 0; i < bf.shape.size(); ++i)
        EXPECT_FLOAT_EQ(bf_data[i], 1.0f);
}
