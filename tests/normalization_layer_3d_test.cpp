#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/random_utilities.h"
#include "opennn/normalization_layer_3d.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;


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

    Normalization3d normalization_3d({sequence_length, embedding_dimension}, "encoder_norm");

    EXPECT_EQ(normalization_3d.get_name(), "Normalization3d");
    EXPECT_EQ(normalization_3d.get_label(), "encoder_norm");
    EXPECT_EQ(normalization_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(normalization_3d.get_embedding_dimension(), embedding_dimension);
    EXPECT_EQ(normalization_3d.get_input_shape()[0], sequence_length);
    EXPECT_EQ(normalization_3d.get_input_shape()[1], embedding_dimension);
    EXPECT_EQ(normalization_3d.get_output_shape()[0], sequence_length);
    EXPECT_EQ(normalization_3d.get_output_shape()[1], embedding_dimension);
}


TEST(Normalization3dTest, ForwardMatchesHandComputedLayerNorm)
{
    const Index batch_size = 1;
    const Index seq = 1;
    const Index dim = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Normalization3d>(Shape{seq, dim}, "norm"));
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 inputs(batch_size, seq, dim);
    inputs.data()[0] = type(1);
    inputs.data()[1] = type(2);
    inputs.data()[2] = type(3);
    inputs.data()[3] = type(4);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(inputs.data(), {batch_size, seq, dim}) };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    const float* output_data = output_view.as<type>();

    const type inv_std = type(1) / std::sqrt(type(1.25));

    EXPECT_NEAR(output_data[0], type(-1.5) * inv_std, 1.0e-4f);
    EXPECT_NEAR(output_data[1], type(-0.5) * inv_std, 1.0e-4f);
    EXPECT_NEAR(output_data[2], type(0.5)  * inv_std, 1.0e-4f);
    EXPECT_NEAR(output_data[3], type(1.5)  * inv_std, 1.0e-4f);
}


TEST(Normalization3dTest, BackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index embedding_dimension = 4;

    const Shape input_shape{sequence_length, embedding_dimension};
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Normalization3d>(input_shape, "norm"), {-1});
    const Index norm_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(norm_index)->get_output_shape()),
                             {norm_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(2.0e-3));
}


TEST(Normalization3dTest, FusedResidualAddForward)
{
    const Index batch_size = 1;
    const Index seq = 1;
    const Index dim = 4;

    auto norm = make_unique<Normalization3d>(Shape{seq, dim}, "fused_norm");
    norm->set_fuse_add(true);

    NeuralNetwork neural_network;
    neural_network.add_layer(move(norm), {-1, -2});
    neural_network.compile();
    neural_network.set_parameters_random();

    Tensor3 main_input(batch_size, seq, dim);
    Tensor3 residual_input(batch_size, seq, dim);

    main_input.data()[0]     = type(1);
    main_input.data()[1]     = type(2);
    main_input.data()[2]     = type(3);
    main_input.data()[3]     = type(4);

    residual_input.data()[0] = type(1);
    residual_input.data()[1] = type(0);
    residual_input.data()[2] = type(1);
    residual_input.data()[3] = type(0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = {
        TensorView(main_input.data(), {batch_size, seq, dim}),
        TensorView(residual_input.data(), {batch_size, seq, dim})
    };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();
    const float* output_data = output_view.as<type>();

    EXPECT_NEAR(output_data[0], type(-1), 1.0e-4f);
    EXPECT_NEAR(output_data[1], type(-1), 1.0e-4f);
    EXPECT_NEAR(output_data[2], type(1),  1.0e-4f);
    EXPECT_NEAR(output_data[3], type(1),  1.0e-4f);
}


TEST(Normalization3dTest, FusedResidualAddGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index sequence_length = 3;
    const Index embedding_dimension = 4;

    const Shape input_shape{sequence_length, embedding_dimension};
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<opennn::Dense>(input_shape, Shape{embedding_dimension}, "Identity"),
                             {-1});
    const Index dense_index = neural_network.get_layers_number() - 1;

    auto norm = make_unique<Normalization3d>(input_shape, "fused_norm");
    norm->set_fuse_add(true);
    neural_network.add_layer(move(norm), {dense_index, -1});
    const Index norm_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(norm_index)->get_output_shape()),
                             {norm_index});

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(), Shape{targets_number}, "Identity"));

    neural_network.compile();
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(2.0e-3));
}
