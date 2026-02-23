#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/dataset.h"
#include "../opennn/language_dataset.h"
#include "../opennn/image_dataset.h"
#include "../opennn/dense_layer.h"
#include "../opennn/pooling_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/standard_networks.h"
#include "../opennn/recurrent_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/multihead_attention_layer.h"

using namespace opennn;

TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    MeanSquaredError mean_squared_error;

    EXPECT_EQ(mean_squared_error.has_neural_network(), false);
    EXPECT_EQ(mean_squared_error.has_dataset(), false);
}


TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;
    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    EXPECT_EQ(mean_squared_error.has_neural_network(), true);
    EXPECT_EQ(mean_squared_error.has_dataset(), true);
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

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const Tensor1 gradient = mean_squared_error.calculate_gradient();
    const Tensor1 numerical_gradient = mean_squared_error.calculate_numerical_gradient();

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

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const Tensor1 gradient = mean_squared_error.calculate_gradient();
    const Tensor1 numerical_gradient = mean_squared_error.calculate_numerical_gradient();

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
    const Shape flatten_layer_input_dimensions = neural_network.get_layer(0).get()->get_output_shape();
    neural_network.add_layer(make_unique<Flatten<4>>(flatten_layer_input_dimensions));
    const Shape dense_layer_input_dimensions = neural_network.get_layer(1).get()->get_output_shape();
    neural_network.add_layer(make_unique<opennn::Dense<2>>(dense_layer_input_dimensions, dataset.get_target_shape()));

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();
    EXPECT_GE(error, 0);
    
    const Tensor1 gradient = mean_squared_error.calculate_gradient();
    const Tensor1 numerical_gradient = mean_squared_error.calculate_numerical_gradient();

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

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const Tensor1 gradient = mean_squared_error.calculate_gradient();
    const Tensor1 numerical_gradient = mean_squared_error.calculate_numerical_gradient();

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
    const Shape flatten_layer_input_dimensions = neural_network.get_layer(0).get()->get_output_shape();
    neural_network.add_layer(make_unique<Flatten<3>>(Shape{ flatten_layer_input_dimensions }));
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{ flattened_size }, Shape{ targets_number }));

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();
    EXPECT_GE(error, 0);

    const Tensor1 gradient = mean_squared_error.calculate_gradient();
    const Tensor1 numerical_gradient = mean_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
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

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();
    const Tensor1 analytical_gradient = mean_squared_error.calculate_gradient();
    const Tensor1 numerical_gradient = mean_squared_error.calculate_numerical_gradient();

    EXPECT_GE(error, 0.0) << "MSE must be positive";
    EXPECT_TRUE(are_equal(analytical_gradient, numerical_gradient, type(1.0e-3)));
}


TEST(MeanSquaredErrorTest, BackPropagateLm)
{
    const Index samples_number = random_integer(2, 10);
    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = random_integer(1, 10);
    const Index neurons_number = random_integer(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices("Training"),
        dataset.get_variable_indices("Input"),
        dataset.get_variable_indices("Target"));

    ApproximationNetwork neural_network({ inputs_number }, { neurons_number }, { dataset.get_target_shape()});

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_inputs(), forward_propagation, true);

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    BackPropagation back_propagation(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    BackPropagationLM back_propagation_lm(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

    Tensor2 numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();
    Tensor1 numerical_gradient = mean_squared_error.calculate_numerical_gradient();
    Tensor2 numerical_hessian = mean_squared_error.calculate_numerical_hessian();

    EXPECT_NEAR(back_propagation_lm.error, back_propagation.error, type(1.0e-3));
    EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian), true);
    EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1e-2)), true);
    EXPECT_EQ(are_equal(back_propagation_lm.hessian, numerical_hessian, type(1.0e-1)), true);
}
