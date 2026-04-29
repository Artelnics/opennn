#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/statistics.h"

using namespace opennn;

TEST(ScalingLayerTest, DefaultConstructor)
{
    Scaling<2> scaling_layer_2d({0});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{0});
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{0});
}


TEST(ScalingLayerTest, GeneralConstructor)
{
    Scaling<2> scaling_layer_2d({1});

    EXPECT_EQ(scaling_layer_2d.get_input_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_output_shape(), Shape{ 1 });
    EXPECT_EQ(scaling_layer_2d.get_name(), "Scaling2d");
}


TEST(ScalingLayerTest, ForwardPropagate)
{
    Index inputs_number = 3;
    Index samples_number = 2;

    const type TOLERANCE = type(1.0e-4);

    // Test None: scaling layer just copies input to output
    {
        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Scaling<2>>(Shape{inputs_number}));
        neural_network.compile();

        Scaling<2>* layer = static_cast<Scaling<2>*>(neural_network.get_layer(0).get());
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
        neural_network.add_layer(make_unique<Scaling<2>>(Shape{inputs_number}));
        neural_network.compile();

        Scaling<2>* layer = static_cast<Scaling<2>*>(neural_network.get_layer(0).get());
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
