#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/multihead_attention_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"
#include <cmath>

using namespace opennn;

namespace
{

void set_identity_projections(Layer* layer, Index embedding_dimension)
{
    vector<TensorView>& parameter_views = layer->get_parameter_views();

    for (size_t projection = 0; projection < 4; ++projection)
    {
        VectorMap bias = parameter_views[2 * projection].as_vector();
        bias.setZero();

        MatrixMap weights = parameter_views[2 * projection + 1].as_matrix();
        weights.setZero();
        for (Index diagonal = 0; diagonal < embedding_dimension; ++diagonal)
            weights(diagonal, diagonal) = type(1.0);
    }
}

}


TEST(MultiHeadAttentionTest, DefaultConstructors)
{
    MultiHeadAttention mha_self;
    EXPECT_EQ(mha_self.get_query_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_source_sequence_length(), 0);
    EXPECT_EQ(mha_self.get_embedding_dimension(), 0);
}


TEST(MultiHeadAttentionTest, GeneralConstructors)
{
    MultiHeadAttention mha_self_config({ 10, 32 }, 4);
    EXPECT_EQ(mha_self_config.get_query_sequence_length(), 10);
    EXPECT_EQ(mha_self_config.get_source_sequence_length(), 10);
    EXPECT_EQ(mha_self_config.get_embedding_dimension(), 32);
    EXPECT_EQ(mha_self_config.get_heads_number(), 4);

    MultiHeadAttention mha_cross({ 5, 16 }, { 8, 16 }, 2);
    EXPECT_EQ(mha_cross.get_query_sequence_length(), 5);
    EXPECT_EQ(mha_cross.get_source_sequence_length(), 8);
    EXPECT_EQ(mha_cross.get_embedding_dimension(), 16);
    EXPECT_EQ(mha_cross.get_heads_number(), 2);
}


TEST(MultiHeadAttentionTest, GeneralConstructorOutputAndInputShape)
{
    MultiHeadAttention mha_self({ 10, 32 }, 4);
    EXPECT_EQ(mha_self.get_input_shape(), (Shape{ 10, 32 }));
    EXPECT_EQ(mha_self.get_output_shape(), (Shape{ 10, 32 }));

    MultiHeadAttention mha_cross({ 5, 16 }, { 8, 16 }, 2);
    EXPECT_EQ(mha_cross.get_input_shape(), (Shape{ 5, 16 }));
    EXPECT_EQ(mha_cross.get_output_shape(), (Shape{ 5, 16 }));
    EXPECT_EQ(mha_cross.get_source_sequence_length(), 8);
}


TEST(MultiHeadAttentionTest, ForwardSelfAttentionMatchesHandComputed)
{
    const Index batch_size = 1;
    const Index sequence_length = 2;
    const Index embedding_dimension = 2;
    const Index heads_number = 1;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<MultiHeadAttention>(Shape{ sequence_length, embedding_dimension }, heads_number));
    neural_network.compile();

    set_identity_projections(neural_network.get_layer(0).get(), embedding_dimension);

    Tensor3 input_data(batch_size, sequence_length, embedding_dimension);
    input_data(0, 0, 0) = type(1.0);  input_data(0, 0, 1) = type(0.0);
    input_data(0, 1, 0) = type(0.0);  input_data(0, 1, 1) = type(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), { batch_size, sequence_length, embedding_dimension }) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], sequence_length);
    EXPECT_EQ(output_view.shape[2], embedding_dimension);

    const float scale = type(1.0) / sqrt(type(embedding_dimension));
    const float exp_score = exp(scale);
    const float high = exp_score / (exp_score + type(1.0));
    const float low = type(1.0) / (exp_score + type(1.0));

    const float* output_data = output_view.as<type>();

    EXPECT_NEAR(output_data[0], high, 1.0e-5f);
    EXPECT_NEAR(output_data[1], low,  1.0e-5f);
    EXPECT_NEAR(output_data[2], low,  1.0e-5f);
    EXPECT_NEAR(output_data[3], high, 1.0e-5f);
}


TEST(MultiHeadAttentionTest, CausalMaskForward)
{
    const Index batch_size = 1;
    const Index sequence_length = 2;
    const Index embedding_dimension = 2;
    const Index heads_number = 1;

    auto mha = make_unique<MultiHeadAttention>(Shape{ sequence_length, embedding_dimension }, heads_number);
    mha->set(sequence_length, sequence_length, embedding_dimension, heads_number, true);

    NeuralNetwork neural_network;
    neural_network.add_layer(move(mha));
    neural_network.compile();

    set_identity_projections(neural_network.get_layer(0).get(), embedding_dimension);

    Tensor3 input_data(batch_size, sequence_length, embedding_dimension);
    input_data(0, 0, 0) = type(1.0);  input_data(0, 0, 1) = type(0.0);
    input_data(0, 1, 0) = type(0.0);  input_data(0, 1, 1) = type(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), { batch_size, sequence_length, embedding_dimension }) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    const float* output_data = output_view.as<type>();

    EXPECT_NEAR(output_data[0], type(1.0), 1.0e-5f);
    EXPECT_NEAR(output_data[1], type(0.0), 1.0e-5f);

    const float scale = type(1.0) / sqrt(type(embedding_dimension));
    const float exp_score = exp(scale);
    const float high = exp_score / (exp_score + type(1.0));
    const float low = type(1.0) / (exp_score + type(1.0));

    EXPECT_NEAR(output_data[2], low,  1.0e-5f);
    EXPECT_NEAR(output_data[3], high, 1.0e-5f);
}


TEST(MultiHeadAttentionTest, CrossAttentionForwardOrGradient)
{
    const Index batch_size = 1;
    const Index query_sequence_length = 1;
    const Index source_sequence_length = 2;
    const Index embedding_dimension = 2;
    const Index heads_number = 1;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<MultiHeadAttention>(Shape{ query_sequence_length, embedding_dimension },
                                                             Shape{ source_sequence_length, embedding_dimension },
                                                             heads_number),
                             { -1, -2 });
    neural_network.compile();

    EXPECT_EQ(neural_network.get_layer(0)->get_output_shape(), (Shape{ query_sequence_length, embedding_dimension }));

    set_identity_projections(neural_network.get_layer(0).get(), embedding_dimension);

    Tensor3 query_data(batch_size, query_sequence_length, embedding_dimension);
    query_data(0, 0, 0) = type(1.0);  query_data(0, 0, 1) = type(0.0);

    Tensor3 source_data(batch_size, source_sequence_length, embedding_dimension);
    source_data(0, 0, 0) = type(1.0);  source_data(0, 0, 1) = type(0.0);
    source_data(0, 1, 0) = type(0.0);  source_data(0, 1, 1) = type(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = {
        TensorView(query_data.data(),  { batch_size, query_sequence_length, embedding_dimension }),
        TensorView(source_data.data(), { batch_size, source_sequence_length, embedding_dimension })
    };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 3);
    EXPECT_EQ(output_view.shape[1], query_sequence_length);
    EXPECT_EQ(output_view.shape[2], embedding_dimension);

    const float scale = type(1.0) / sqrt(type(embedding_dimension));
    const float exp_score = exp(scale);
    const float high = exp_score / (exp_score + type(1.0));
    const float low = type(1.0) / (exp_score + type(1.0));

    const float* output_data = output_view.as<type>();

    EXPECT_NEAR(output_data[0], high, 1.0e-5f);
    EXPECT_NEAR(output_data[1], low,  1.0e-5f);
}


TEST(MultiHeadAttentionTest, BackwardGradientMatchesNumerical)
{
    const Index samples_number = 4;
    const Index sequence_length = 3;
    const Index heads_number = 2;
    const Index head_dimension = 2;
    const Index embedding_dimension = heads_number * head_dimension;

    const Shape input_shape{ sequence_length, embedding_dimension };

    TabularDataset dataset(samples_number, input_shape, { sequence_length * embedding_dimension });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<MultiHeadAttention>(input_shape, heads_number));
    neural_network.add_layer(make_unique<Flatten>(neural_network.get_output_shape()));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
