#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/dataset.h"
#include "../opennn/language_dataset.h"
#include "../opennn/image_dataset.h"
#include "../opennn/dense_layer.h"
#include "../opennn/pooling_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/loss.h"
#include "../opennn/standard_networks.h"
#include "../opennn/recurrent_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/multihead_attention_layer.h"
#include <iomanip>

using namespace opennn;

TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    Loss loss;

    EXPECT_EQ(loss.get_neural_network(), nullptr);
    EXPECT_EQ(loss.get_dataset(), nullptr);
}


TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;
    Loss loss(&neural_network, &dataset);

    EXPECT_NE(loss.get_neural_network(), nullptr);
    EXPECT_NE(loss.get_dataset(), nullptr);
}


TEST(MeanSquaredErrorTest, BackPropagateDense2d)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{ inputs_number }, Shape{ dataset.get_target_shape()}));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = loss.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const VectorR gradient = loss.calculate_gradient();
    const VectorR numerical_gradient = loss.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}


TEST(MeanSquaredErrorTest, BackPropagateRecurrent)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = random_integer(3, 10);
    const Index time_steps = random_integer(1, 10);

    Dataset dataset(samples_number, {time_steps, inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Recurrent>(Shape{time_steps, inputs_number}, Shape{targets_number}));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = loss.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const VectorR gradient = loss.calculate_gradient();
    const VectorR numerical_gradient = loss.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}


TEST(MeanSquaredErrorTest, BackPropagateConvolutional)
{
    const Index samples_number = 6;
    const Index targets_number = 1;

    const Shape input_shape = { 21, 21, 3 };
    const Shape kernel_shape = { 3, 3, 3, 1 };

    ImageDataset dataset(samples_number, { input_shape }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(input_shape, kernel_shape));
    const Shape flatten_layer_input_dimensions = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten<4>>(flatten_layer_input_dimensions));
    const Shape dense_layer_input_dimensions = neural_network.get_layer(1)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense<2>>(dense_layer_input_dimensions, dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = loss.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const VectorR gradient = loss.calculate_gradient();
    const VectorR numerical_gradient = loss.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}


TEST(MeanSquaredErrorTest, BackPropagatePooling)
{
    const Index samples_number = 6;
    const Index targets_number = 1;

    const Shape input_shape = { 21, 21, 3 };
    const Shape kernel_shape = { 3, 3, 3, 1 };

    ImageDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Convolutional>(input_shape, kernel_shape));
    const Shape conv_output_dimensions = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Pooling>(conv_output_dimensions));
    const Shape pool_output_dimensions = neural_network.get_layer(1)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten<4>>(pool_output_dimensions));
    const Shape flatten_output_dimensions = neural_network.get_layer(2)->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense<2>>(flatten_output_dimensions, dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = loss.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const VectorR gradient = loss.calculate_gradient();
    const VectorR numerical_gradient = loss.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}


TEST(MeanSquaredErrorTest, BackPropagateEmbedding)
{
    const Index samples_number = random_integer(5, 10);
    const Index inputs_number = random_integer(10, 20);
    const Index targets_number = random_integer(3, 10);

    const Index embeding_dim = inputs_number;
    const Index sequence_length = random_integer(1, 10);
    const Index flattened_size = sequence_length * embeding_dim;

    Dataset dataset(samples_number, { sequence_length }, { targets_number });
    dataset.set_data_integer(inputs_number);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Embedding>(Shape{ inputs_number, sequence_length }, embeding_dim));
    const Shape flatten_layer_input_dimensions = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten<3>>(Shape{ flatten_layer_input_dimensions }));
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{ flattened_size }, Shape{ targets_number }));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = loss.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const VectorR gradient = loss.calculate_gradient();

    const VectorR numerical_gradient = loss.calculate_numerical_gradient();

    // Embedding + Dense with large flattened sizes produces large gradients;
    // absolute tolerance must account for gradient magnitude.
    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(5.0e-2)), true);
}


TEST(MeanSquaredErrorTest, BackPropagateMultiheadAttention)
{
    const Index batch_size = random_integer(1, 10);
    const Index sequence_length = random_integer(3, 10);
    const Index heads_number = random_integer(1, 10);
    const Index head_dimension = random_integer(1, 10);
    const Index embedding_dimension = heads_number * head_dimension;

    const Shape sample_input_dimensions = { sequence_length, embedding_dimension };

    const Shape sample_target_shape = { sequence_length * embedding_dimension };

    Dataset dataset(batch_size, sample_input_dimensions, sample_target_shape);
    dataset.set_data_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<MultiHeadAttention>(dataset.get_input_shape(), heads_number));
    neural_network.add_layer(make_unique<Flatten<3>>(neural_network.get_output_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = loss.calculate_numerical_error();
    const VectorR analytical_gradient = loss.calculate_gradient();
    const VectorR numerical_gradient = loss.calculate_numerical_gradient();

    EXPECT_GE(error, 0.0) << "MSE must be positive";
    EXPECT_TRUE(are_equal(analytical_gradient, numerical_gradient, type(1.0e-3)));
}
