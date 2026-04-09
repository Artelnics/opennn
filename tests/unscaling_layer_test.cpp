#include "pch.h"

#include "../opennn/statistics.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/unscaling_layer.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

TEST(UnscalingTest, DefaultConstructor)
{
    Unscaling unscaling_layer;

    EXPECT_EQ(unscaling_layer.get_name(), "Unscaling");
    EXPECT_EQ(unscaling_layer.get_input_shape(), Shape{ 0 });
    EXPECT_EQ(unscaling_layer.get_output_shape(), Shape{ 0 });
}

TEST(UnscalingTest, GeneralConstructor)
{
    Unscaling unscaling_layer({ 3 });

    EXPECT_EQ(unscaling_layer.get_input_shape(), Shape{ 3 });
    EXPECT_EQ(unscaling_layer.get_output_shape(), Shape{ 3 });
    EXPECT_EQ(unscaling_layer.get_name(), "Unscaling");
}

TEST(UnscalingTest, ForwardPropagate)
{
    Index inputs_number;
    Index samples_number;
    const type TOLERANCE = type(1e-4);

    // Test Unscaling Scaler::None
    {
        inputs_number = 3;
        samples_number = 1;

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Unscaling>(Shape{inputs_number}));
        neural_network.compile();

        Unscaling* layer = static_cast<Unscaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("None");

        Tensor2 data_to_unscale(samples_number, inputs_number);
        data_to_unscale.setConstant(type(10));

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(data_to_unscale.data(), {samples_number, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();

        EXPECT_NEAR(output_view.data[0], type(10), TOLERANCE);
        EXPECT_NEAR(output_view.data[1], type(10), TOLERANCE);
        EXPECT_NEAR(output_view.data[2], type(10), TOLERANCE);
    }

    // Test Unscaling with MinimumMaximum
    {
        inputs_number = 1;
        samples_number = 3;

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Unscaling>(Shape{inputs_number}));
        neural_network.compile();

        Unscaling* layer = static_cast<Unscaling*>(neural_network.get_layer(0).get());
        layer->set_scalers("MinimumMaximum");

        MatrixR original_data(samples_number, inputs_number);
        original_data << type(2), type(4), type(6);

        vector<Descriptives> actual_descriptives = descriptives(original_data);
        layer->set_descriptives(actual_descriptives);

        // Feed scaled values: for MinMax with range [-1,1] and min=2,max=6:
        //   scaled = -1 + 2*(x-2)/(6-2) => x=2 -> -1, x=4 -> 0, x=6 -> 1
        MatrixR scaled_data(samples_number, inputs_number);
        scaled_data << type(-1), type(0), type(1);

        ForwardPropagation forward_propagation(samples_number, &neural_network);
        vector<TensorView> input_views = { TensorView(scaled_data.data(), {samples_number, inputs_number}) };
        neural_network.forward_propagate(input_views, forward_propagation, false);

        TensorView output_view = forward_propagation.get_outputs();

        EXPECT_NEAR(output_view.data[0], original_data(0, 0), TOLERANCE);
        EXPECT_NEAR(output_view.data[1], original_data(1, 0), TOLERANCE);
        EXPECT_NEAR(output_view.data[2], original_data(2, 0), TOLERANCE);
    }
}
