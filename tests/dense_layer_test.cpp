#include "pch.h"
#include "numerical_derivatives.h"

#include "../opennn/tensor_types.h"
#include "../opennn/layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/loss.h"

using namespace opennn;


TEST(Dense2dTest, DefaultConstructor)
{
    opennn::Dense dense_layer;

    EXPECT_EQ(dense_layer.get_name(), "Dense");
}


TEST(Dense2dTest, GeneralConstructor)
{
    opennn::Dense dense_layer({10}, {3}, "Identity");

    EXPECT_EQ(dense_layer.get_activation_function(), ActivationOp::Function::Identity);

    EXPECT_EQ(dense_layer.get_input_shape()[0], 10);
    EXPECT_EQ(dense_layer.get_output_shape()[0], 3);
    EXPECT_GE(dense_layer.get_parameters_number(), 33);
}


TEST(Dense2dTest, ForwardValuesMatchHandComputed)
{
    const Index batch_size = 2;
    const Index inputs_number = 3;
    const Index outputs_number = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity"));
    neural_network.compile();

    Layer* dense_layer = neural_network.get_layer(0).get();
    vector<TensorView>& parameter_views = dense_layer->get_parameter_views();

    VectorMap bias = parameter_views[0].as_vector();
    bias(0) = type(0.5);
    bias(1) = type(-1.0);

    MatrixMap weights = parameter_views[1].as_matrix();
    weights(0, 0) = type(1.0);  weights(0, 1) = type(2.0);
    weights(1, 0) = type(3.0);  weights(1, 1) = type(4.0);
    weights(2, 0) = type(5.0);  weights(2, 1) = type(6.0);

    MatrixR input_data(batch_size, inputs_number);
    input_data(0, 0) = type(1.0);  input_data(0, 1) = type(0.0);  input_data(0, 2) = type(-1.0);
    input_data(1, 0) = type(2.0);  input_data(1, 1) = type(1.0);  input_data(1, 2) = type(0.0);

    const MatrixR result = neural_network.calculate_outputs(input_data);

    ASSERT_EQ(result.rows(), batch_size);
    ASSERT_EQ(result.cols(), outputs_number);

    EXPECT_NEAR(result(0, 0), type(-3.5), 1.0e-5f);
    EXPECT_NEAR(result(0, 1), type(-5.0), 1.0e-5f);
    EXPECT_NEAR(result(1, 0), type(5.5),  1.0e-5f);
    EXPECT_NEAR(result(1, 1), type(7.0),  1.0e-5f);
}


TEST(Dense2dTest, BackwardGradientMatchesNumerical)
{
    const Index samples_number = 5;
    const Index inputs_number = 3;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{targets_number}, "Sigmoid"));
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


TEST(Dense3dTest, GeneralConstructor)
{
    const Index sequence_length = 5;
    const Index input_embedding = 4;
    const Index output_embedding = 3;

    opennn::Dense dense_layer({sequence_length, input_embedding}, {output_embedding}, "Identity");

    EXPECT_EQ(dense_layer.get_activation_function(), ActivationOp::Function::Identity);

    EXPECT_EQ(dense_layer.get_input_shape()[0], sequence_length);
    EXPECT_EQ(dense_layer.get_input_shape()[1], input_embedding);

    EXPECT_EQ(dense_layer.get_output_shape()[0], sequence_length);
    EXPECT_EQ(dense_layer.get_output_shape()[1], output_embedding);

    EXPECT_GE(dense_layer.get_parameters_number(), input_embedding * output_embedding + output_embedding);
}


TEST(Dense3dTest, SequenceForwardValuesMatchHandComputed)
{
    const Index batch_size = 2;
    const Index sequence_length = 2;
    const Index input_embedding = 2;
    const Index output_embedding = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{sequence_length, input_embedding},
                                                        Shape{output_embedding}, "Identity"));
    neural_network.compile();

    Layer* dense_layer = neural_network.get_layer(0).get();
    vector<TensorView>& parameter_views = dense_layer->get_parameter_views();

    VectorMap bias = parameter_views[0].as_vector();
    bias(0) = type(1.0);
    bias(1) = type(-1.0);

    MatrixMap weights = parameter_views[1].as_matrix();
    weights(0, 0) = type(1.0);  weights(0, 1) = type(0.0);
    weights(1, 0) = type(0.0);  weights(1, 1) = type(2.0);

    Tensor3 input_data(batch_size, sequence_length, input_embedding);
    input_data(0, 0, 0) = type(1.0);  input_data(0, 0, 1) = type(2.0);
    input_data(0, 1, 0) = type(3.0);  input_data(0, 1, 1) = type(4.0);
    input_data(1, 0, 0) = type(-1.0); input_data(1, 0, 1) = type(0.0);
    input_data(1, 1, 0) = type(0.5);  input_data(1, 1, 1) = type(-2.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, sequence_length, input_embedding}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 3);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], sequence_length);
    EXPECT_EQ(output_view.shape[2], output_embedding);

    const float* output_data = output_view.as<type>();

    EXPECT_NEAR(output_data[0], type(2.0),  1.0e-5f);
    EXPECT_NEAR(output_data[1], type(3.0),  1.0e-5f);
    EXPECT_NEAR(output_data[2], type(4.0),  1.0e-5f);
    EXPECT_NEAR(output_data[3], type(7.0),  1.0e-5f);

    EXPECT_NEAR(output_data[4], type(0.0),  1.0e-5f);
    EXPECT_NEAR(output_data[5], type(-1.0), 1.0e-5f);
    EXPECT_NEAR(output_data[6], type(1.5),  1.0e-5f);
    EXPECT_NEAR(output_data[7], type(-5.0), 1.0e-5f);
}


TEST(ActivationTest, LeakyReLUForwardPassesPositiveAndScalesNegative)
{
    vector<float> buffer = { -2.0f, -0.5f, 0.0f, 0.5f, 2.0f };
    TensorView view(buffer.data(), {Index(buffer.size())});

    activation_forward(view, ActivationFunction::LeakyReLU);

    EXPECT_FLOAT_EQ(buffer[0], -0.2f);
    EXPECT_FLOAT_EQ(buffer[1], -0.05f);
    EXPECT_FLOAT_EQ(buffer[2],  0.0f);
    EXPECT_FLOAT_EQ(buffer[3],  0.5f);
    EXPECT_FLOAT_EQ(buffer[4],  2.0f);
}


TEST(ActivationTest, LeakyReLUBackwardGatesByOutputSign)
{
    vector<float> outputs = { -0.2f, -0.05f, 0.0f, 0.5f, 2.0f };
    vector<float> delta   = {  1.0f,   2.0f, 3.0f, 4.0f, 5.0f };

    TensorView outputs_view(outputs.data(), {Index(outputs.size())});
    TensorView delta_view  (delta.data(),   {Index(delta.size())});

    activation_backward(outputs_view, delta_view, ActivationFunction::LeakyReLU);

    EXPECT_FLOAT_EQ(delta[0], 0.1f);
    EXPECT_FLOAT_EQ(delta[1], 0.2f);
    EXPECT_FLOAT_EQ(delta[2], 3.0f);
    EXPECT_FLOAT_EQ(delta[3], 4.0f);
    EXPECT_FLOAT_EQ(delta[4], 5.0f);
}


TEST(Dense, BatchNormForwardOrGradient)
{
    const Index samples_number = 6;
    const Index inputs_number = 3;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, {inputs_number}, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{targets_number},
                                                        "Identity", true));
    neural_network.compile();
    neural_network.set_parameters_random();

    EXPECT_TRUE(static_cast<opennn::Dense*>(neural_network.get_layer(0).get())->get_batch_normalization());

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    const type max_abs_diff = (gradient - numerical_gradient).array().abs().maxCoeff();
    const type gradient_scale = max(type(1), numerical_gradient.array().abs().maxCoeff());

    EXPECT_LT(max_abs_diff / gradient_scale, type(1.0e-2));
}
