#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/tensor_types.h"
#include "opennn/batch_norm_operator.h"
#include "opennn/dense_layer.h"
#include "opennn/neural_network.h"
#include "opennn/tabular_dataset.h"
#include "opennn/loss.h"

using namespace opennn;


TEST(BatchNormalizationOperatoreratorTest, DefaultIsInactive)
{
    BatchNormalizationOperator batch_norm;

    EXPECT_FALSE(batch_norm.active());
    EXPECT_EQ(batch_norm.features, 0);
    EXPECT_TRUE(batch_norm.parameter_specs().empty());
    EXPECT_TRUE(batch_norm.state_specs().empty());
}


TEST(BatchNormalizationOperatoreratorTest, SetActivatesAndStoresMomentum)
{
    BatchNormalizationOperator batch_norm;
    batch_norm.set(7, 0.25f);

    EXPECT_TRUE(batch_norm.active());
    EXPECT_EQ(batch_norm.features, 7);
    EXPECT_FLOAT_EQ(batch_norm.momentum, 0.25f);
}


TEST(BatchNormalizationOperatoreratorTest, SetRejectsInvalidMomentum)
{
    BatchNormalizationOperator batch_norm;

    EXPECT_ANY_THROW(batch_norm.set(4, 1.0f));
    EXPECT_ANY_THROW(batch_norm.set(4, -0.1f));
}


TEST(BatchNormalizationOperatoreratorTest, ParameterAndStateSpecsMatchFeatures)
{
    const Index features = 5;

    BatchNormalizationOperator batch_norm;
    batch_norm.set(features);

    const vector<TensorSpec> parameter_specs = batch_norm.parameter_specs();
    const vector<TensorSpec> state_specs = batch_norm.state_specs();

    ASSERT_EQ(parameter_specs.size(), 2u);
    ASSERT_EQ(state_specs.size(), 2u);

    EXPECT_EQ(parameter_specs[0].shape[0], features);
    EXPECT_EQ(parameter_specs[1].shape[0], features);
    EXPECT_EQ(state_specs[0].shape[0], features);
    EXPECT_EQ(state_specs[1].shape[0], features);

    EXPECT_EQ(parameter_specs[0].dtype, Type::FP32);
}


TEST(BatchNormalizationOperatoreratorTest, LinkAndInitDefaults)
{
    const Index features = 4;

    VectorR gamma_storage(features);
    VectorR beta_storage(features);
    VectorR running_mean_storage(features);
    VectorR running_variance_storage(features);

    gamma_storage.setConstant(3.0f);
    beta_storage.setConstant(9.0f);
    running_mean_storage.setConstant(7.0f);
    running_variance_storage.setConstant(5.0f);

    BatchNormalizationOperator batch_norm;
    batch_norm.set(features);

    vector<TensorView> parameter_views = {
        TensorView(gamma_storage.data(), {features}),
        TensorView(beta_storage.data(), {features})
    };
    vector<TensorView> state_views = {
        TensorView(running_mean_storage.data(), {features}),
        TensorView(running_variance_storage.data(), {features})
    };

    batch_norm.link_parameters(parameter_views);
    batch_norm.link_states(state_views);
    batch_norm.init_defaults();

    EXPECT_FLOAT_EQ(gamma_storage.maxCoeff(), 1.0f);
    EXPECT_FLOAT_EQ(gamma_storage.minCoeff(), 1.0f);
    EXPECT_FLOAT_EQ(beta_storage.cwiseAbs().maxCoeff(), 0.0f);
    EXPECT_FLOAT_EQ(running_mean_storage.cwiseAbs().maxCoeff(), 0.0f);
    EXPECT_FLOAT_EQ(running_variance_storage.minCoeff(), 1.0f);
    EXPECT_FLOAT_EQ(running_variance_storage.maxCoeff(), 1.0f);
}


TEST(BatchNormalizationOperatoreratorTest, DenseEnablesBatchNormalization)
{
    opennn::Dense dense({6}, {4}, "Identity", true);

    EXPECT_TRUE(dense.get_batch_normalization());
}


TEST(BatchNormalizationOperatoreratorTest, DenseDisabledByDefault)
{
    opennn::Dense dense({6}, {4}, "Identity");

    EXPECT_FALSE(dense.get_batch_normalization());
}


TEST(BatchNormalizationOperatoreratorTest, ForwardTrainingNormalizesPerFeature)
{
    const Index batch_size = 32;
    const Index inputs_number = 5;
    const Index outputs_number = 3;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity", true));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setRandom();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, inputs_number}) };
    neural_network.forward_propagate(inputs, forward_propagation, true);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 2);
    ASSERT_EQ(output_view.shape[0], batch_size);
    ASSERT_EQ(output_view.shape[1], outputs_number);

    const MatrixMap output = output_view.as_flat_matrix();

    const VectorR feature_mean = output.colwise().mean();

    const Index N = output.rows();
    const VectorR feature_variance =
        (output.rowwise() - feature_mean.transpose()).array().square().colwise().sum() / float(N);

    EXPECT_LT(feature_mean.cwiseAbs().maxCoeff(), 1.0e-4f);

    EXPECT_LT((feature_variance.array() - 1.0f).abs().maxCoeff(), 1.0e-3f);
}


TEST(BatchNormalizationOperatoreratorTest, ForwardInferenceUsesRunningStatistics)
{
    const Index batch_size = 8;
    const Index inputs_number = 4;
    const Index outputs_number = 3;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity", true));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setRandom();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, inputs_number}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape[0], batch_size);
    ASSERT_EQ(output_view.shape[1], outputs_number);

    const MatrixMap output = output_view.as_flat_matrix();

    EXPECT_TRUE(output.allFinite());
}


TEST(BatchNormalizationOperatoreratorTest, InferenceIsDeterministicAcrossRows)
{
    const Index batch_size = 5;
    const Index inputs_number = 4;
    const Index outputs_number = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity", true));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setConstant(0.5f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, inputs_number}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    const MatrixMap output = forward_propagation.get_outputs().as_flat_matrix();

    for (Index column = 0; column < output.cols(); ++column)
    {
        const float reference = output(0, column);
        for (Index row = 1; row < output.rows(); ++row)
            EXPECT_NEAR(output(row, column), reference, 1.0e-5f);
    }
}


TEST(BatchNormalizationOperatoreratorTest, GradientMatchesFiniteDifferences)
{
    const Index samples_number = 16;
    const Index inputs_number = 4;
    const Index targets_number = 2;

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ inputs_number },
                                                        dataset.get_target_shape(),
                                                        "Identity",
                                                        true));
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
