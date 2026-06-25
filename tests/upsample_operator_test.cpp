#include "pch.h"
#include "numerical_derivatives.h"

#include "../opennn/tensor_types.h"
#include "../opennn/upsample_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/loss.h"

using namespace opennn;

class UpsampleOperatoreratorTest : public ::testing::Test
{
protected:
    const Index height = 3;
    const Index width = 2;
    const Index channels = 2;
    const Index scale_factor = 2;
    const Shape input_shape{ height, width, channels };
};


TEST_F(UpsampleOperatoreratorTest, Constructor)
{
    Upsample upsample_layer(input_shape, scale_factor);

    EXPECT_EQ(upsample_layer.get_name(), "Upsample");
    EXPECT_EQ(upsample_layer.get_input_shape(), input_shape);
    EXPECT_EQ(upsample_layer.get_output_shape(),
              Shape({ height * scale_factor, width * scale_factor, channels }));
}


TEST_F(UpsampleOperatoreratorTest, OutputShapeDependsOnScaleFactor)
{
    Upsample upsample_layer(input_shape, 3);

    EXPECT_EQ(upsample_layer.get_output_shape(),
              Shape({ height * 3, width * 3, channels }));

    upsample_layer.set_scale_factor(1);
    EXPECT_EQ(upsample_layer.get_output_shape(), input_shape);
}


TEST_F(UpsampleOperatoreratorTest, EmptyInputShapeGivesEmptyOutputShape)
{
    Upsample upsample_layer;

    EXPECT_TRUE(upsample_layer.get_input_shape().empty());
    EXPECT_TRUE(upsample_layer.get_output_shape().empty());
}


TEST_F(UpsampleOperatoreratorTest, ForwardOutputShape)
{
    const Index batch_size = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(input_shape, scale_factor));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setConstant(0.0f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(),
                                                  { batch_size, height, width, channels }) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], height * scale_factor);
    EXPECT_EQ(output_view.shape[2], width * scale_factor);
    EXPECT_EQ(output_view.shape[3], channels);
}


TEST_F(UpsampleOperatoreratorTest, ForwardConstantReplication)
{
    const Index batch_size = 1;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(input_shape, scale_factor));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setConstant(2.5f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(),
                                                  { batch_size, height, width, channels }) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], 2.5f, 1e-6f);
}


TEST_F(UpsampleOperatoreratorTest, ForwardPixelReplication)
{
    const Index batch_size = 1;
    const Index out_h = height * scale_factor;
    const Index out_w = width * scale_factor;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(input_shape, scale_factor));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);

    for (Index ih = 0; ih < height; ++ih)
        for (Index iw = 0; iw < width; ++iw)
            for (Index c = 0; c < channels; ++c)
                inputs_data(0, ih, iw, c) = type((ih * width + iw) * channels + c);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(),
                                                  { batch_size, height, width, channels }) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const type* dst = output_view.as<type>();

    for (Index oh = 0; oh < out_h; ++oh)
        for (Index ow = 0; ow < out_w; ++ow)
            for (Index c = 0; c < channels; ++c)
            {
                const Index ih = oh / scale_factor;
                const Index iw = ow / scale_factor;
                const type expected = inputs_data(0, ih, iw, c);
                const Index out_idx = (oh * out_w + ow) * channels + c;
                EXPECT_NEAR(dst[out_idx], expected, 1e-6f);
            }
}


TEST_F(UpsampleOperatoreratorTest, ForwardBatchIndependence)
{
    const Index batch_size = 2;
    const Index out_h = height * scale_factor;
    const Index out_w = width * scale_factor;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(input_shape, scale_factor));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setZero();

    for (Index iw = 0; iw < width; ++iw)
        for (Index c = 0; c < channels; ++c)
            inputs_data(1, 0, iw, c) = 7.0f;

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(),
                                                  { batch_size, height, width, channels }) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const type* dst = output_view.as<type>();

    const Index sample_size = out_h * out_w * channels;

    for (Index i = 0; i < sample_size; ++i)
        EXPECT_NEAR(dst[i], 0.0f, 1e-6f);

    type second_sum = 0.0f;
    for (Index i = sample_size; i < 2 * sample_size; ++i)
        second_sum += dst[i];

    EXPECT_NEAR(second_sum, 7.0f * scale_factor * width * scale_factor * channels, 1e-4f);
}


TEST_F(UpsampleOperatoreratorTest, ScaleFactorOneIsIdentity)
{
    const Index batch_size = 1;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(input_shape, 1));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);

    for (Index ih = 0; ih < height; ++ih)
        for (Index iw = 0; iw < width; ++iw)
            for (Index c = 0; c < channels; ++c)
                inputs_data(0, ih, iw, c) = type((ih * width + iw) * channels + c) + 0.5f;

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(),
                                                  { batch_size, height, width, channels }) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.size(), inputs_data.size());

    const type* dst = output_view.as<type>();
    const type* src = inputs_data.data();

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(dst[i], src[i], 1e-6f);
}


TEST_F(UpsampleOperatoreratorTest, BackPropagateDeltaAccumulation)
{
    const Index samples_number = 5;
    const Index targets_number = 1;

    TabularDataset dataset(samples_number, input_shape, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Upsample>(input_shape, scale_factor));

    const Shape flatten_input_shape = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten>(flatten_input_shape));

    const Shape dense_input_shape = neural_network.get_layer(1)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense>(dense_input_shape, dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
