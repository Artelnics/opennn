#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include <utility>

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/layers/pooling_layer_3d.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/normalization_layer_3d.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/loss.h"

using namespace opennn;

TEST(Pool3dOperatoreratorTest, MaxConstructorShapes)
{
    const Shape input_shape{3, 4};
    Pooling3d layer(input_shape, PoolingMethod::MaxPooling, "max_pool_3d");

    EXPECT_EQ(layer.get_name(), "Pooling3d");
    EXPECT_EQ(layer.get_input_shape(), input_shape);
    EXPECT_EQ(layer.get_sequence_length(), 3);
    EXPECT_EQ(layer.get_input_features(), 4);
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::MaxPooling);
    EXPECT_EQ(layer.get_output_shape(), Shape{4});
}

TEST(Pool3dOperatoreratorTest, AverageConstructorShapes)
{
    const Shape input_shape{5, 2};
    Pooling3d layer(input_shape, PoolingMethod::AveragePooling, "avg_pool_3d");

    EXPECT_EQ(layer.get_input_features(), 2);
    EXPECT_EQ(layer.get_sequence_length(), 5);
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::AveragePooling);
    EXPECT_EQ(layer.get_output_shape(), Shape{2});
}

TEST(Pool3dOperatoreratorTest, SetPoolingMethodFromString)
{
    Pooling3d layer({3, 4}, PoolingMethod::MaxPooling, "pool");
    layer.set_pooling_method("AveragePooling");
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::AveragePooling);
    layer.set_pooling_method("MaxPooling");
    EXPECT_EQ(layer.get_pooling_method(), PoolingMethod::MaxPooling);
}

TEST(Pool3dOperatoreratorTest, ForwardMaxValuesAndShape)
{
    const Index batch_size = 2;
    const Index seq = 3;
    const Index feat = 4;
    const Shape input_shape{seq, feat};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(input_shape, PoolingMethod::MaxPooling, "max_pool"));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2, 2, 2}},
                          {{8, 7, 6, 5}, {4, 3, 2, 1}, {0, 0, 0, 0}}});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.get_shape().get_rank(), 2);
    EXPECT_EQ(output_view.get_shape()[0], batch_size);
    EXPECT_EQ(output_view.get_shape()[1], feat);

    const float expected[8] = {5, 6, 7, 8, 8, 7, 6, 5};
    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], expected[i], 1e-6f);
}

TEST(Pool3dOperatoreratorTest, ForwardAverageValuesWithPadding)
{
    const Index batch_size = 2;
    const Index seq = 3;
    const Index feat = 4;
    const Shape input_shape{seq, feat};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(input_shape, PoolingMethod::AveragePooling, "avg_pool"));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2, 2, 2}},
                          {{8, 7, 6, 5}, {4, 3, 2, 1}, {0, 0, 0, 0}}});

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.get_shape().get_rank(), 2);
    EXPECT_EQ(output_view.get_shape()[0], batch_size);
    EXPECT_EQ(output_view.get_shape()[1], feat);

    const float expected[8] = {8.0f / 3.0f, 10.0f / 3.0f, 4.0f, 14.0f / 3.0f,
                               6.0f, 5.0f, 4.0f, 3.0f};
    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], expected[i], 1e-5f);
}

TEST(Pool3dOperatoreratorTest, ForwardAverageAllPaddingIsZero)
{
    const Index batch_size = 1;
    const Index seq = 2;
    const Index feat = 3;
    const Shape input_shape{seq, feat};

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(input_shape, PoolingMethod::AveragePooling, "avg_pool"));
    neural_network.compile();

    Tensor3 input_data(batch_size, seq, feat);
    input_data.setZero();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(), {batch_size, seq, feat}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    for (Index i = 0; i < output_view.size(); ++i)
        EXPECT_NEAR(output_view.as<type>()[i], 0.0f, 1e-7f);
}

TEST(Pool3dOperatoreratorTest, BackPropagateMaxGradient)
{
    const Index samples_number = 6;
    const Index seq = 4;
    const Index feat = 3;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, Shape{seq, feat}, Shape{targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(Shape{seq, feat}, PoolingMethod::MaxPooling, "max_pool"));
    const Shape pool_output_shape = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense>(pool_output_shape, dataset.get_target_shape()));
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

TEST(Pool3dOperatoreratorTest, BackPropagateAverageGradient)
{
    const Index samples_number = 6;
    const Index seq = 4;
    const Index feat = 3;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, Shape{seq, feat}, Shape{targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Pooling3d>(Shape{seq, feat}, PoolingMethod::AveragePooling, "avg_pool"));
    const Shape pool_output_shape = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense>(pool_output_shape, dataset.get_target_shape()));
    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    const type max_abs_diff = (gradient - numerical_gradient).array().abs().maxCoeff();
    const type gradient_scale = max(type(1), numerical_gradient.array().abs().maxCoeff());
    EXPECT_LT(max_abs_diff / gradient_scale, type(2.0e-2));
}

namespace
{

const Index padded_sequence_length = 8;
const Index padded_embedding_dimension = 6;
const Index padded_vocabulary_size = 16;

// Token id 0 is the padding token, so a prefix of nonzero ids is a shorter
// sequence -- the same convention the Embedding exports its lengths from.
void write_padded_token_ids(MatrixR& data, const vector<Index>& valid_lengths)
{
    for (Index sample = 0; sample < Index(valid_lengths.size()); ++sample)
        for (Index position = 0; position < padded_sequence_length; ++position)
            data(sample, position) = position < valid_lengths[size_t(sample)]
                ? type(1 + (sample * padded_sequence_length + position) % (padded_vocabulary_size - 1))
                : type(0);
}

void add_padded_average_pooling_stack(NeuralNetwork& network)
{
    auto embedding = make_unique<Embedding>(Shape{padded_vocabulary_size, padded_sequence_length},
                                            padded_embedding_dimension, "embedding");
    embedding->set_add_positional_encoding(true);
    embedding->set_export_valid_lengths(true);
    network.add_layer(std::move(embedding), {-1});

    network.add_layer(make_unique<Normalization3d>(
                          Shape{padded_sequence_length, padded_embedding_dimension},
                          "normalization"), {0});

    network.add_layer(make_unique<Pooling3d>(Shape{padded_sequence_length, padded_embedding_dimension},
                                             PoolingMethod::AveragePooling, "avg_pool"), {1});
}

// A normalization starts at gamma = 1, beta = 0, which maps the zero row the
// Embedding writes for a padding token back to zero, leaving the padding
// visible to the scan after all. Training moves the shift off zero, and that is
// the state these tests need: the padded rows are then ordinary values and the
// exported lengths are the only thing that still knows they are padding.
void shift_normalization(NeuralNetwork& network)
{
    network.get_layer(1)->get_parameter_views()[1].as_vector().setConstant(type(0.25));
}

}

// Average pooling recovers the sequence length by scanning for all-zero token
// rows, which holds only while a padded row is still the zero row the Embedding
// wrote. One layer with a nonzero shift in between -- a normalization is the
// usual one -- replaces that row with its own bias, and the scan then counts
// padding as data and divides by the wrong number. The Embedding already
// exports the exact lengths, and attention already consumes them; this states
// that pooling has to average over those rows and no others.
TEST(Pool3dOperatoreratorTest, AverageIgnoresPaddingBehindANonzeroShift)
{
    const vector<Index> valid_lengths{padded_sequence_length, 5, 2, 1};
    const Index batch_size = Index(valid_lengths.size());

    NeuralNetwork neural_network;
    add_padded_average_pooling_stack(neural_network);
    neural_network.compile();
    neural_network.set_parameters_random();
    shift_normalization(neural_network);

    MatrixR token_ids(batch_size, padded_sequence_length);
    write_padded_token_ids(token_ids, valid_lengths);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(token_ids.data(), {batch_size, padded_sequence_length}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    // The reference is built from what pooling actually receives, so it states
    // the averaging rule alone and does not reimplement the normalization.
    const type* pooled_input = forward_propagation.inputs[2][0].as<type>();

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.get_shape().get_rank(), 2);
    ASSERT_EQ(output_view.get_shape()[0], batch_size);
    ASSERT_EQ(output_view.get_shape()[1], padded_embedding_dimension);

    for (Index sample = 0; sample < batch_size; ++sample)
    {
        const Index valid_length = valid_lengths[size_t(sample)];

        for (Index feature = 0; feature < padded_embedding_dimension; ++feature)
        {
            type sum = 0;
            for (Index position = 0; position < valid_length; ++position)
                sum += pooled_input[(sample * padded_sequence_length + position) * padded_embedding_dimension + feature];

            EXPECT_NEAR(output_view.as<type>()[sample * padded_embedding_dimension + feature],
                        sum / type(valid_length), type(1.0e-5))
                << "sample " << sample << " (valid length " << valid_length << "), feature " << feature;
        }
    }
}

// The backward pass re-derives the sequence lengths from its own copy of the
// same scan, so the fix has to land in both directions or the gradient stops
// being the gradient of the forward pass. Note what the padded rows here are
// and are not: the Embedding writes an explicit zero row for the padding token
// id, weights and positional encoding both skipped, so nothing upstream of the
// normalization can move them. It is the normalization's own shift that makes
// them nonzero, and from there they are ordinary values that the average has no
// way to tell from data.
TEST(Pool3dOperatoreratorTest, AverageGradientAgreesWithItsForwardOnPaddedBatches)
{
    const vector<Index> valid_lengths{padded_sequence_length, 5, 2, 1};
    const Index samples_number = Index(valid_lengths.size());
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, Shape{padded_sequence_length}, Shape{targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    MatrixR data = dataset.get_data();
    write_padded_token_ids(data, valid_lengths);
    dataset.set_data(std::move(data));

    NeuralNetwork neural_network;
    add_padded_average_pooling_stack(neural_network);
    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_layer(2)->get_output_shape(),
                                                        dataset.get_target_shape()), {2});
    neural_network.compile();
    neural_network.set_parameters_random();
    shift_normalization(neural_network);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    // This passes today, and is meant to: both directions read the same wrong
    // count, so they agree with each other. What it pins is that they go on
    // agreeing. A forward pass that averaged over the exported lengths while the
    // backward still counted nonzero rows would divide the same deltas by
    // different numbers -- 8 against 1 for the shortest sequence here -- so this
    // is what catches a fix applied to one direction only.
    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    const type max_abs_diff = (gradient - numerical_gradient).array().abs().maxCoeff();
    const type gradient_scale = max(type(1), numerical_gradient.array().abs().maxCoeff());
    EXPECT_LT(max_abs_diff / gradient_scale, type(2.0e-2));
}
