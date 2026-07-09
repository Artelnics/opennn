#include "pch.h"

#include "opennn/neural_network.h"
#include "opennn/scaling_layer.h"
#include "opennn/statistics.h"

using namespace opennn;

TEST(ScalingLayerTest, DefaultConstructor)
{
    Scaling scaling_layer_2d({0});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{0});
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{0});
}


TEST(ScalingLayerTest, GeneralConstructor)
{
    Scaling scaling_layer_2d({1});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_name(), "Scaling");
}


TEST(ScalingLayerTest, ForwardPropagate)
{
    Index inputs_number = 3;
    Index samples_number = 2;

    const type TOLERANCE = type(1.0e-4);

    // Test None: scaling layer just copies input to output
    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling>(Shape{inputs_number}));
        neural_network.compile();

        Scaling* layer = static_cast<Scaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("None");

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setConstant(type(10));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();
        EXPECT_NEAR(output_view.as<type>()[0], 10.0, TOLERANCE);
    }

    // Test: scaling layer copies input regardless of scaler setting
    // (actual scaling is done by Optimizer::set_scaling() before training)
    {
        inputs_number = 1;
        samples_number = 3;

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling>(Shape{inputs_number}));
        neural_network.compile();

        Scaling* layer = static_cast<Scaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("MinimumMaximum");

        Tensor2 inputs(samples_number, inputs_number);
        inputs.setValues({{type(2)},{type(4)},{type(6)}});

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();
        // Scaling layer just copies, so output == input
        EXPECT_NEAR(output_view.as<type>()[0], type(2), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[2], type(6), TOLERANCE);
    }
}


TEST(ScalingLayerTest, ForwardPropagateScalesValues)
{
    const Index samples_number = 3;
    const type TOLERANCE = type(1.0e-4);

    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling>(Shape{1}));
        neural_network.compile();

        Scaling* layer = static_cast<Scaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("MeanStandardDeviation");
        layer->set_descriptives({Descriptives(type(2), type(6), type(4), type(2))});

        Tensor2 inputs(samples_number, 1);
        inputs.setValues({{type(2)}, {type(4)}, {type(6)}});

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, 1}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();
        EXPECT_NEAR(output_view.as<type>()[0], type(-1), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[1], type(0), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[2], type(1), TOLERANCE);
    }

    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling>(Shape{1}));
        neural_network.compile();

        Scaling* layer = static_cast<Scaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("MinimumMaximum");
        layer->set_descriptives({Descriptives(type(2), type(6), type(4), type(2))});

        Tensor2 inputs(samples_number, 1);
        inputs.setValues({{type(2)}, {type(4)}, {type(6)}});

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, 1}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();
        EXPECT_NEAR(output_view.as<type>()[0], type(-1), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[1], type(0), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[2], type(1), TOLERANCE);
    }
}


TEST(ScalingLayerTest, ForwardPropagateScalerModes)
{
    const Index samples_number = 3;
    const type TOLERANCE = type(1.0e-4);

    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling>(Shape{1}));
        neural_network.compile();

        Scaling* layer = static_cast<Scaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("StandardDeviation");
        layer->set_descriptives({Descriptives(type(2), type(6), type(4), type(2))});

        Tensor2 inputs(samples_number, 1);
        inputs.setValues({{type(2)}, {type(4)}, {type(6)}});

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, 1}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();
        EXPECT_NEAR(output_view.as<type>()[0], type(1), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[1], type(2), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[2], type(3), TOLERANCE);
    }

    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling>(Shape{1}));
        neural_network.compile();

        Scaling* layer = static_cast<Scaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("Logarithm");

        Tensor2 inputs(samples_number, 1);
        inputs.setValues({{type(1)}, {type(2)}, {type(4)}});

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, 1}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();
        EXPECT_NEAR(output_view.as<type>()[0], type(0), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[1], type(0.6931472), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[2], type(1.3862944), TOLERANCE);
    }

    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling>(Shape{1}));
        neural_network.compile();

        Scaling* layer = static_cast<Scaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("ImageMinMax");

        Tensor2 inputs(samples_number, 1);
        inputs.setValues({{type(0)}, {type(255)}, {type(510)}});

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(inputs.data(), {samples_number, 1}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();
        EXPECT_NEAR(output_view.as<type>()[0], type(0), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[1], type(1), TOLERANCE);
        EXPECT_NEAR(output_view.as<type>()[2], type(2), TOLERANCE);
    }
}
