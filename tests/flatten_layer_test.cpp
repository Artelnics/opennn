#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/flatten_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

class FlattenLayerTest : public ::testing::Test
{
protected:
    const Index height = 4;
    const Index width = 4;
    const Index channels = 3;
    const Shape input_shape{ height, width, channels };

    unique_ptr<Flatten> flatten_layer;

    void SetUp() override
    {
        flatten_layer = make_unique<Flatten>(input_shape);
    }
};


TEST_F(FlattenLayerTest, DefaultConstructor)
{
    Flatten default_flatten;

    EXPECT_EQ(default_flatten.get_name(), "Flatten");
    EXPECT_EQ(default_flatten.get_output_shape(), Shape{ 0 });
}


TEST_F(FlattenLayerTest, Constructor)
{
    EXPECT_EQ(flatten_layer->get_input_shape(), input_shape);
    EXPECT_EQ(flatten_layer->get_output_shape(), Shape{ height * width * channels });
    EXPECT_EQ(flatten_layer->get_name(), "Flatten");
}


TEST_F(FlattenLayerTest, ForwardPropagate)
{
    const Index batch_size = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Flatten>(input_shape));
    neural_network.compile();

    Tensor4 inputs_data(batch_size, height, width, channels);
    inputs_data.setConstant(1.23f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_data.data(), {batch_size, height, width, channels}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const Shape& output_dims = output_view.shape;

    ASSERT_EQ(output_dims.rank, 2);
    EXPECT_EQ(output_dims[0], batch_size);
    EXPECT_EQ(output_dims[1], height * width * channels);

    for(Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], 1.23f, 1e-6f);
}


TEST_F(FlattenLayerTest, FlattenBackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index embedding_dimension = 4;

    const Shape network_input_shape{sequence_length, embedding_dimension};
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, network_input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<opennn::Dense>(network_input_shape, Shape{embedding_dimension}, "Identity"),
                             {-1});
    const Index dense_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(dense_index)->get_output_shape()),
                             {dense_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
