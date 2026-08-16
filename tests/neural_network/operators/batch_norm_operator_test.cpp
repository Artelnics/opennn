#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/batch_norm_operator.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/training_strategy/loss.h"

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

    ASSERT_EQ(output_view.get_shape().get_rank(), 2);
    ASSERT_EQ(output_view.get_shape()[0], batch_size);
    ASSERT_EQ(output_view.get_shape()[1], outputs_number);

    const MatrixMap output = output_view.as_flat_matrix();

    const VectorR feature_mean = output.colwise().mean();

    const Index N = output.rows();
    const VectorR feature_variance =
        (output.rowwise() - feature_mean.transpose()).array().square().colwise().sum() / float(N);

    EXPECT_LT(feature_mean.cwiseAbs().maxCoeff(), 1.0e-4f);

    // Batch norm divides by sqrt(var + BN_EPSILON), so the normalized variance is
    // var/(var + BN_EPSILON) and falls short of 1 by about BN_EPSILON/var. This layer's
    // pre-normalization feature variances are O(1e-3), which puts the shortfall near
    // 2e-3. The bound tracks BN_EPSILON: raise it if that constant grows again.
    EXPECT_LT((feature_variance.array() - 1.0f).abs().maxCoeff(), 5.0e-3f);
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

    ASSERT_EQ(output_view.get_shape()[0], batch_size);
    ASSERT_EQ(output_view.get_shape()[1], outputs_number);

    const MatrixMap output = output_view.as_flat_matrix();

    EXPECT_TRUE(output.allFinite());
}

// Batch norm only means anything if inference reproduces training. Drive the
// running statistics onto a fixed batch, and the two modes must then agree: the
// batch statistics they each normalize by are the same numbers. This caught an
// inference epsilon of 1e-2 against training's 1e-5, which shrank every channel
// whose variance was near or below it - here to 50-66% of its trained value.
TEST(BatchNormalizationOperatoreratorTest, InferenceMatchesTrainingOnConvergedStatistics)
{
    const Index batch_size = 32, inputs_number = 5, outputs_number = 4;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(
        Shape{inputs_number}, Shape{outputs_number}, "Identity", true));
    neural_network.compile();
    neural_network.set_parameters_random();

    MatrixR input_data(batch_size, inputs_number);
    input_data.setRandom();

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, inputs_number}) };

    for (int i = 0; i < 2000; ++i)
        neural_network.forward_propagate(inputs, forward_propagation, true);

    const MatrixR training_output = forward_propagation.get_outputs().as_flat_matrix();

    neural_network.forward_propagate(inputs, forward_propagation, false);
    const MatrixR inference_output = forward_propagation.get_outputs().as_flat_matrix();

    const float scale = training_output.cwiseAbs().maxCoeff();
    ASSERT_GT(scale, 0.0f);

    const float divergence = (inference_output - training_output).cwiseAbs().maxCoeff() / scale;

    EXPECT_LT(divergence, 1.0e-2f);
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

// The running variance takes the BIASED batch variance, with no Bessel
// correction. Most frameworks apply M/(M-1) here and OpenNN does not, so the
// convention is easy to "correct" by mistake - and since inference reproduces
// training only when both sides agree, changing it silently rescales the output
// of every model already saved to disk. Nothing else in the suite pinned it.
TEST(BatchNormalizationOperatoreratorTest, RunningVarianceUsesBiasedEstimate)
{
    const Index batch_size = 8;
    const Index features   = 3;
    const float momentum   = 0.1f;   // Dense's batch-norm default.

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{features}, Shape{features}, "Identity", true));
    neural_network.compile();
    neural_network.set_parameters_random();

    // Scaled up so the two conventions separate well clear of float noise: they
    // differ by momentum * variance * (M/(M-1) - 1), which at a Glorot-scale
    // variance of ~3e-3 is only 5e-5 and cannot be told apart reliably.
    MatrixR input_data(batch_size, features);
    input_data.setRandom();
    input_data *= 50.0f;

    // The state buffer is aligned per spec, so the running variances are not
    // simply the second block. Locate them: init_defaults is the only thing that
    // writes exactly 1.0, so those slots are the variances.
    const VectorMap states_before(neural_network.get_states_data(), neural_network.get_states_size());
    vector<Index> variance_slots;
    for (Index i = 0; i < states_before.size(); ++i)
        if (states_before(i) == 1.0f) variance_slots.push_back(i);

    ASSERT_EQ(Index(variance_slots.size()), features)
        << "could not locate the running variances in the state buffer";

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {batch_size, features}) };
    neural_network.forward_propagate(inputs, forward_propagation, true);

    // Recover the variance the forward pass actually normalized by, rather than
    // recomputing it from the inputs: slot 2 holds inverse_variance, which is
    // rsqrt(var + BN_EPSILON) over the batch. That variance is biased by
    // definition - it is what normalization divides by - so comparing the running
    // statistic against it isolates the one question this test asks, and does so
    // without depending on the parameter layout or on what the layer computes.
    // Slot 3 is inverse_variance: the layer's forward slots carry one leading
    // entry ahead of get_forward_specs, so the two per-channel statistic slots
    // land at 2 (mean) and 3, not 1 and 2.
    constexpr float bn_epsilon = 1.0e-5f;
    const VectorMap inverse_variance = forward_propagation.slots[0][3].as_vector();
    ASSERT_EQ(inverse_variance.size(), features);

    const VectorMap states(neural_network.get_states_data(), neural_network.get_states_size());

    for (Index i = 0; i < features; ++i)
    {
        const float normalizing_variance =
            1.0f / (inverse_variance(i) * inverse_variance(i)) - bn_epsilon;

        // Running variance starts at 1, so one update leaves
        // (1 - momentum) + momentum * variance.
        const float observed = states(variance_slots[size_t(i)]);
        const float biased   = (1.0f - momentum) + momentum * normalizing_variance;
        const float bessel   = (1.0f - momentum) + momentum * normalizing_variance
                             * float(batch_size) / float(batch_size - 1);

        EXPECT_NEAR(observed, biased, 1.0e-3f);
        ASSERT_GT(abs(bessel - biased), 1.0e-2f) << "variance too small to tell the conventions apart";
        EXPECT_GT(abs(observed - bessel), 1.0e-2f) << "running variance is Bessel-corrected";
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
