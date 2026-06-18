#include "pch.h"
#include "../opennn/tensor_types.h"
#include "../opennn/tensor_operations.h"
#include "../opennn/dropout_operator.h"
#include "../opennn/dense_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/random_utilities.h"

using namespace opennn;


TEST(DropoutOperatorTest, ActiveFlagReflectsRate)
{
    DropoutOp dropout;

    EXPECT_FALSE(dropout.active());

    dropout.set_rate(0.5f);
    EXPECT_TRUE(dropout.active());

    dropout.set_rate(0.0f);
    EXPECT_FALSE(dropout.active());
}


TEST(DropoutOperatorTest, SetRateRejectsOutOfRange)
{
    DropoutOp dropout;

    EXPECT_THROW(dropout.set_rate(-0.1f), std::exception);
    EXPECT_THROW(dropout.set_rate(1.0f), std::exception);
    EXPECT_THROW(dropout.set_rate(1.5f), std::exception);

    EXPECT_NO_THROW(dropout.set_rate(0.999f));
}


TEST(DropoutForwardTest, ZeroRateIsIdentity)
{
    VectorR values(5);
    values << 1.0f, -2.0f, 3.5f, 0.25f, -0.75f;
    const VectorR original = values;

    TensorView output(values.data(), { Index(values.size()) });
    Buffer mask;

    dropout_forward(output, mask, 0.0f);

    for (Index i = 0; i < values.size(); ++i)
        EXPECT_FLOAT_EQ(values[i], original[i]);

    EXPECT_TRUE(mask.empty());
}


TEST(DropoutForwardTest, MaskValuesAreZeroOrKeepScale)
{
    set_seed(123u);

    const Index element_count = 4096;
    const float rate = 0.5f;
    const float keep_scale = 1.0f / (1.0f - rate);

    VectorR values(element_count);
    values.setConstant(1.0f);
    TensorView output(values.data(), { element_count });
    Buffer mask;

    dropout_forward(output, mask, rate);

    const float* mask_values = mask.as<float>();

    for (Index i = 0; i < element_count; ++i)
    {
        const bool is_zero = mask_values[i] == 0.0f;
        const bool is_scale = std::abs(mask_values[i] - keep_scale) < 1e-5f;
        EXPECT_TRUE(is_zero || is_scale);

        if (is_zero)
            EXPECT_FLOAT_EQ(values[i], 0.0f);
        else
            EXPECT_FLOAT_EQ(values[i], keep_scale);
    }
}


TEST(DropoutForwardTest, InvertedScalingPreservesMeanApproximately)
{
    set_seed(7u);

    const Index element_count = 100000;
    const float rate = 0.3f;
    const float input_value = 2.0f;

    VectorR values(element_count);
    values.setConstant(input_value);
    TensorView output(values.data(), { element_count });
    Buffer mask;

    dropout_forward(output, mask, rate);

    double sum = 0.0;
    for (Index i = 0; i < element_count; ++i)
        sum += values[i];

    const float output_mean = float(sum / double(element_count));

    EXPECT_NEAR(output_mean, input_value, 0.05f);
}


TEST(DropoutForwardTest, DroppedFractionMatchesRate)
{
    set_seed(42u);

    const Index element_count = 100000;
    const float rate = 0.4f;

    VectorR values(element_count);
    values.setConstant(1.0f);
    TensorView output(values.data(), { element_count });
    Buffer mask;

    dropout_forward(output, mask, rate);

    const float* mask_values = mask.as<float>();

    Index dropped = 0;
    for (Index i = 0; i < element_count; ++i)
        if (mask_values[i] == 0.0f) ++dropped;

    const float dropped_fraction = float(dropped) / float(element_count);

    EXPECT_NEAR(dropped_fraction, rate, 0.02f);
}


TEST(DropoutBackwardTest, AppliesSameMaskAsForward)
{
    set_seed(11u);

    const Index element_count = 256;
    const float rate = 0.5f;

    VectorR values(element_count);
    values.setConstant(1.0f);
    TensorView output(values.data(), { element_count });
    Buffer mask;

    dropout_forward(output, mask, rate);

    vector<float> mask_snapshot(element_count);
    for (Index i = 0; i < element_count; ++i)
        mask_snapshot[i] = mask.as<float>()[i];

    VectorR delta(element_count);
    for (Index i = 0; i < element_count; ++i)
        delta[i] = 3.0f;

    TensorView delta_view(delta.data(), { element_count });
    dropout_backward(delta_view, mask, rate);

    for (Index i = 0; i < element_count; ++i)
        EXPECT_FLOAT_EQ(delta[i], 3.0f * mask_snapshot[i]);
}


TEST(DropoutBackwardTest, ZeroRateLeavesDeltaUnchanged)
{
    VectorR delta(4);
    delta << 1.0f, 2.0f, 3.0f, 4.0f;
    const VectorR original = delta;

    TensorView delta_view(delta.data(), { Index(delta.size()) });
    Buffer mask;

    dropout_backward(delta_view, mask, 0.0f);

    for (Index i = 0; i < delta.size(); ++i)
        EXPECT_FLOAT_EQ(delta[i], original[i]);
}


TEST(DropoutLayerTest, InferencePassIsIdentityWithDropout)
{
    const Index batch_size = 8;
    const Index inputs_number = 6;
    const Index outputs_number = 5;

    NeuralNetwork dropout_network;
    auto dense_with_dropout = make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity");
    dense_with_dropout->set_dropout_rate(0.5f);
    dropout_network.add_layer(std::move(dense_with_dropout));
    dropout_network.compile();

    NeuralNetwork reference_network;
    reference_network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity"));
    reference_network.compile();

    set_seed(99u);
    dropout_network.set_parameters_random();

    const Index parameters_size = dropout_network.get_parameters_size();
    ASSERT_EQ(parameters_size, reference_network.get_parameters_size());

    VectorR parameters(parameters_size);
    const float* parameters_data = dropout_network.get_parameters_data();
    for (Index i = 0; i < parameters_size; ++i)
        parameters[i] = parameters_data[i];

    reference_network.set_parameters(parameters);

    MatrixR input(batch_size, inputs_number);
    set_random_uniform(input, -1.0f, 1.0f);

    const MatrixR dropout_output = dropout_network.calculate_outputs(input);
    const MatrixR reference_output = reference_network.calculate_outputs(input);

    ASSERT_EQ(dropout_output.rows(), reference_output.rows());
    ASSERT_EQ(dropout_output.cols(), reference_output.cols());

    for (Index r = 0; r < dropout_output.rows(); ++r)
        for (Index c = 0; c < dropout_output.cols(); ++c)
            EXPECT_NEAR(dropout_output(r, c), reference_output(r, c), 1e-5f);
}


TEST(DropoutLayerTest, TrainingPassDiffersFromInference)
{
    const Index batch_size = 32;
    const Index inputs_number = 8;
    const Index outputs_number = 8;

    NeuralNetwork neural_network;
    auto dense = make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "Identity");
    dense->set_dropout_rate(0.5f);
    neural_network.add_layer(std::move(dense));
    neural_network.compile();

    set_seed(5u);
    neural_network.set_parameters_random();

    MatrixR input(batch_size, inputs_number);
    input.setConstant(type(1.0));

    ForwardPropagation training_propagation(batch_size, &neural_network);
    vector<TensorView> training_views = { TensorView(input.data(), {batch_size, inputs_number}) };
    neural_network.forward_propagate(training_views, training_propagation, true);

    const TensorView training_output = training_propagation.get_outputs();

    Index zero_count = 0;
    for (Index i = 0; i < training_output.size(); ++i)
        if (training_output.as<type>()[i] == 0.0f) ++zero_count;

    EXPECT_GT(zero_count, 0);
    EXPECT_LT(zero_count, training_output.size());
}
