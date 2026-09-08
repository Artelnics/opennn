#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include <utility>

#include "opennn/core/tensor_types.h"
#include "opennn/core/configuration.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/network/layers/embedding_layer.h"
#include "opennn/network/layers/flatten_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/network.h"
#include "opennn/training/loss.h"

using namespace opennn;

TEST(LearnedPositionalTest, EmbeddingGradientCheck)
{
    const Index samples_number  = 6;
    const Index vocabulary_size = 12;
    const Index sequence_length = 5;
    const Index embedding_dim   = 8;
    const Index targets_number  = 3;
    const Index flattened       = sequence_length * embedding_dim;

    TabularDataset dataset(samples_number, { sequence_length }, { targets_number });
    dataset.set_data_integer(vocabulary_size);
    dataset.set_sample_roles("Training");

    auto embedding = make_unique<Embedding>(Shape{vocabulary_size, sequence_length}, embedding_dim);
    embedding->set_learned_positional(true);
    ASSERT_TRUE(embedding->get_learned_positional());

    Network network;
    network.add_layer(std::move(embedding));
    network.add_layer(make_unique<Flatten>(network.get_layer(0)->get_output_shape()));
    network.add_layer(make_unique<opennn::Dense>(Shape{flattened}, Shape{targets_number}));
    network.compile();
    network.set_parameters_random();

    EXPECT_GT(network.get_parameters_number(),
              vocabulary_size * embedding_dim + flattened * targets_number + targets_number);

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-2));
}
