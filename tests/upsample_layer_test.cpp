#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/upsample_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

class UpsampleLayerTest : public ::testing::Test
{
protected:
    const Index height = 2;
    const Index width = 3;
    const Index channels = 2;
    const Index scale_factor = 2;
    const Shape input_shape{ height, width, channels };
};


TEST_F(UpsampleLayerTest, Constructor)
{
    Upsample upsample_layer(input_shape, scale_factor, "upsample_test");

    EXPECT_EQ(upsample_layer.get_name(), "Upsample");
    EXPECT_EQ(upsample_layer.get_input_shape(), input_shape);

    const Shape output_shape = upsample_layer.get_output_shape();

    ASSERT_EQ(output_shape.rank, 3);
    EXPECT_EQ(output_shape[0], height * scale_factor);
    EXPECT_EQ(output_shape[1], width * scale_factor);
    EXPECT_EQ(output_shape[2], channels);
}


TEST_F(UpsampleLayerTest, GeneralConstructorLabel)
{
    Upsample upsample_layer(input_shape, scale_factor, "upsample_test");

    EXPECT_EQ(upsample_layer.get_name(), "Upsample");
    EXPECT_EQ(upsample_layer.get_label(), "upsample_test");
    EXPECT_EQ(upsample_layer.get_input_shape(), input_shape);

    const Shape output_shape = upsample_layer.get_output_shape();

    ASSERT_EQ(output_shape.rank, 3);
    EXPECT_EQ(output_shape[0], height * scale_factor);
    EXPECT_EQ(output_shape[1], width * scale_factor);
    EXPECT_EQ(output_shape[2], channels);
}


TEST_F(UpsampleLayerTest, EmptyInputShapeOutputEmpty)
{
    Upsample upsample_layer;

    EXPECT_TRUE(upsample_layer.get_output_shape().empty());
}


TEST_F(UpsampleLayerTest, ForwardPropagateNearestNeighborReplication)
{
    const Index batch_size = 1;
    const Index single_channel = 1;
    const Index in_h = 2;
    const Index in_w = 2;
    const Index scale = 2;
    const Shape shape{ in_h, in_w, single_channel };

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(shape, scale, "upsample_test"));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, in_h, in_w, single_channel);
    inputs_data(0, 0, 0, 0) = 1.0f;
    inputs_data(0, 0, 1, 0) = 2.0f;
    inputs_data(0, 1, 0, 0) = 3.0f;
    inputs_data(0, 1, 1, 0) = 4.0f;

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(),
        { batch_size, in_h, in_w, single_channel }) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 4);
    ASSERT_EQ(output_view.shape[1], in_h * scale);
    ASSERT_EQ(output_view.shape[2], in_w * scale);

    const Index out_w = in_w * scale;
    const float* out = output_view.as<type>();

    const float expected[4][4] = {
        { 1.0f, 1.0f, 2.0f, 2.0f },
        { 1.0f, 1.0f, 2.0f, 2.0f },
        { 3.0f, 3.0f, 4.0f, 4.0f },
        { 3.0f, 3.0f, 4.0f, 4.0f }
    };

    for (Index oh = 0; oh < in_h * scale; ++oh)
        for (Index ow = 0; ow < out_w; ++ow)
            EXPECT_NEAR(out[oh * out_w + ow], expected[oh][ow], 1e-6f);
}


TEST_F(UpsampleLayerTest, ForwardPropagateChannelsPreserved)
{
    const Index batch_size = 1;
    const Index in_h = 1;
    const Index in_w = 1;
    const Index ch = 3;
    const Index scale = 3;
    const Shape shape{ in_h, in_w, ch };

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(shape, scale, "upsample_test"));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, in_h, in_w, ch);
    inputs_data(0, 0, 0, 0) = 10.0f;
    inputs_data(0, 0, 0, 1) = 20.0f;
    inputs_data(0, 0, 0, 2) = 30.0f;

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(),
        { batch_size, in_h, in_w, ch }) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape[1], in_h * scale);
    ASSERT_EQ(output_view.shape[2], in_w * scale);
    ASSERT_EQ(output_view.shape[3], ch);

    const float* out = output_view.as<type>();
    const Index spatial = (in_h * scale) * (in_w * scale);

    for (Index s = 0; s < spatial; ++s)
    {
        EXPECT_NEAR(out[s * ch + 0], 10.0f, 1e-6f);
        EXPECT_NEAR(out[s * ch + 1], 20.0f, 1e-6f);
        EXPECT_NEAR(out[s * ch + 2], 30.0f, 1e-6f);
    }
}


TEST_F(UpsampleLayerTest, UpsampleBackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index in_h = 2;
    const Index in_w = 2;
    const Index ch = 2;
    const Index kernels_number = 2;
    const Index scale = 2;
    const Index targets_number = 2;

    const Shape spatial_shape{ in_h, in_w, ch };

    TabularDataset dataset(samples_number, spatial_shape, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Convolutional>(spatial_shape,
                                                        Shape{ 3, 3, ch, kernels_number },
                                                        "Identity",
                                                        Shape{ 1, 1 },
                                                        "Same"),
                             { -1 });
    const Index conv_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Upsample>(neural_network.get_layer(conv_index)->get_output_shape(), scale, "upsample_test"),
                             { conv_index });
    const Index upsample_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(upsample_index)->get_output_shape()),
                             { upsample_index });

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{ targets_number }, "Identity"));

    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
