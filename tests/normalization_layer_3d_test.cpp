#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/random_utilities.h"
#include "../opennn/normalization_layer_3d.h"
#include "../opennn/neural_network.h"

using namespace opennn;


struct Normalization3dLayerConfig {
    Index batch_size;
    Index sequence_length;
    Index embedding_dimension;
    string test_name;
};

class Normalization3dLayerTest : public ::testing::TestWithParam<Normalization3dLayerConfig> {};

INSTANTIATE_TEST_SUITE_P(Normalization3dTests, Normalization3dLayerTest, ::testing::Values(
                                                                             Normalization3dLayerConfig{ 2, 5, 8, "Small" },
                                                                             Normalization3dLayerConfig{ 4, 10, 16, "Medium" },
                                                                             Normalization3dLayerConfig{ 1, 32, 64, "SingleBatch" },
                                                                             Normalization3dLayerConfig{ 8, 4, 12, "OddDimensions" }
                                                                             ));


TEST(Normalization3dTest, DefaultConstructor)
{
    Normalization3d normalization_3d;

    EXPECT_EQ(normalization_3d.get_input_shape().rank, 2);
    EXPECT_EQ(normalization_3d.get_input_shape()[0], 0);
    EXPECT_EQ(normalization_3d.get_output_shape().rank, 2);
    EXPECT_EQ(normalization_3d.get_output_shape()[0], 0);
}


TEST(Normalization3dTest, GeneralConstructor)
{
    const Index sequence_length = 15;
    const Index embedding_dimension = 32;

    Normalization3d normalization_3d({sequence_length, embedding_dimension});

    EXPECT_EQ(normalization_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(normalization_3d.get_embedding_dimension(), embedding_dimension);
    EXPECT_EQ(normalization_3d.get_input_shape()[0], sequence_length);
    EXPECT_EQ(normalization_3d.get_input_shape()[1], embedding_dimension);
}


TEST_P(Normalization3dLayerTest, ForwardPropagate)
{
    Normalization3dLayerConfig parameters = GetParam();
    const Index batch_size = parameters.batch_size;
    const Index seq = parameters.sequence_length;
    const Index dim = parameters.embedding_dimension;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Normalization3d>(Shape{seq, dim}, parameters.test_name));
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 inputs_tensor(batch_size, seq, dim);
    for (Index i = 0; i < inputs_tensor.size(); ++i) {
        inputs_tensor.data()[i] = static_cast<type>(random_normal(0.0, 5.0));
    }

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_tensor.data(), {batch_size, seq, dim}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], seq);
    EXPECT_EQ(output_view.shape[2], dim);
}


TEST_P(Normalization3dLayerTest, BackPropagate)
{
    Normalization3dLayerConfig parameters = GetParam();
    const Index batch_size = parameters.batch_size;
    const Index seq = parameters.sequence_length;
    const Index dim = parameters.embedding_dimension;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Normalization3d>(Shape{seq, dim}, parameters.test_name));
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 inputs_tensor(batch_size, seq, dim);
    for (Index i = 0; i < inputs_tensor.size(); ++i)
        inputs_tensor.data()[i] = static_cast<type>(random_normal(0.0, 5.0));

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs_tensor.data(), {batch_size, seq, dim}) };
    neural_network.forward_propagate(input_views, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();

    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], seq);
    EXPECT_EQ(output_view.shape[2], dim);
}
