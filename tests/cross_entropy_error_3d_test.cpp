#include "pch.h"
#include "opennn/tensor_types.h"
#include "opennn/error_functions.h"
#include "opennn/loss.h"
#include "opennn/dense_layer.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "gtest/gtest.h"

using namespace opennn;

namespace
{

void fill_logits(Tensor3& logits)
{
    logits(0, 0, 0) = type(0.1); logits(0, 0, 1) = type(0.2); logits(0, 0, 2) = type(0.6); logits(0, 0, 3) = type(0.1);
    logits(0, 1, 0) = type(0.25); logits(0, 1, 1) = type(0.25); logits(0, 1, 2) = type(0.25); logits(0, 1, 3) = type(0.25);
    logits(0, 2, 0) = type(0.1); logits(0, 2, 1) = type(0.1); logits(0, 2, 2) = type(0.2); logits(0, 2, 3) = type(0.6);
}

}


TEST(CrossEntropyError3DTest, DefaultConstructor)
{
    NeuralNetwork neural_network;
    TabularDataset dataset;

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy3d);

    EXPECT_TRUE(loss.get_neural_network() != nullptr);
    EXPECT_TRUE(loss.get_dataset() != nullptr);
}


TEST(CrossEntropyError3DTest, ForwardIgnoresPaddingAndCountsTokens)
{
    const Index batch = 1;
    const Index sequence_length = 3;
    const Index vocabulary_size = 4;

    Tensor3 logits(batch, sequence_length, vocabulary_size);
    fill_logits(logits);

    VectorR targets(sequence_length);
    targets << type(2), type(0), type(1);

    TensorView input_view(logits.data(), { batch, sequence_length, vocabulary_size });
    TensorView target_view(targets.data(), { sequence_length });

    float error = 0;
    Index active_tokens = 0;
    Index correct_tokens = 0;

    cross_entropy_3d(input_view, target_view, error, active_tokens, correct_tokens);

    EXPECT_EQ(active_tokens, 2);
    EXPECT_EQ(correct_tokens, 1);

    const float expected_error =
        (-std::log(0.6f + EPSILON) - std::log(0.1f + EPSILON)) / 2.0f;
    EXPECT_NEAR(error, expected_error, 1.0e-5f);

    const float accuracy = float(correct_tokens) / float(active_tokens);
    EXPECT_NEAR(accuracy, 0.5f, 1.0e-6f);
}


TEST(CrossEntropyError3DTest, GradientMatchesFormulaAndZerosPadding)
{
    const Index batch = 1;
    const Index sequence_length = 3;
    const Index vocabulary_size = 4;

    Tensor3 logits(batch, sequence_length, vocabulary_size);
    fill_logits(logits);

    VectorR targets(sequence_length);
    targets << type(2), type(0), type(1);

    Tensor3 gradients(batch, sequence_length, vocabulary_size);
    gradients.setConstant(type(-999));

    TensorView input_view(logits.data(), { batch, sequence_length, vocabulary_size });
    TensorView target_view(targets.data(), { sequence_length });
    TensorView gradient_view(gradients.data(), { batch, sequence_length, vocabulary_size });

    cross_entropy_3d_gradient(input_view, target_view, gradient_view, 2);

    EXPECT_NEAR(gradients(0, 0, 0), type(0.05),  1.0e-5);
    EXPECT_NEAR(gradients(0, 0, 1), type(0.10),  1.0e-5);
    EXPECT_NEAR(gradients(0, 0, 2), type(-0.20), 1.0e-5);
    EXPECT_NEAR(gradients(0, 0, 3), type(0.05),  1.0e-5);

    EXPECT_NEAR(gradients(0, 1, 0), type(0), 1.0e-6);
    EXPECT_NEAR(gradients(0, 1, 1), type(0), 1.0e-6);
    EXPECT_NEAR(gradients(0, 1, 2), type(0), 1.0e-6);
    EXPECT_NEAR(gradients(0, 1, 3), type(0), 1.0e-6);

    EXPECT_NEAR(gradients(0, 2, 0), type(0.05),  1.0e-5);
    EXPECT_NEAR(gradients(0, 2, 1), type(-0.45), 1.0e-5);
    EXPECT_NEAR(gradients(0, 2, 2), type(0.10),  1.0e-5);
    EXPECT_NEAR(gradients(0, 2, 3), type(0.30),  1.0e-5);
}


TEST(CrossEntropyError3DTest, AllPaddingGivesZeroLossAndGradient)
{
    const Index batch = 1;
    const Index sequence_length = 3;
    const Index vocabulary_size = 4;

    Tensor3 logits(batch, sequence_length, vocabulary_size);
    fill_logits(logits);

    VectorR targets(sequence_length);
    targets.setZero();

    Tensor3 gradients(batch, sequence_length, vocabulary_size);
    gradients.setConstant(type(-999));

    TensorView input_view(logits.data(), { batch, sequence_length, vocabulary_size });
    TensorView target_view(targets.data(), { sequence_length });
    TensorView gradient_view(gradients.data(), { batch, sequence_length, vocabulary_size });

    float error = type(-1);
    Index active_tokens = -1;
    Index correct_tokens = -1;

    cross_entropy_3d(input_view, target_view, error, active_tokens, correct_tokens);

    EXPECT_EQ(active_tokens, 0);
    EXPECT_EQ(correct_tokens, 0);
    EXPECT_EQ(error, type(0));

    cross_entropy_3d_gradient(input_view, target_view, gradient_view, active_tokens);

    for (Index s = 0; s < sequence_length; ++s)
        for (Index v = 0; v < vocabulary_size; ++v)
            EXPECT_NEAR(gradients(0, s, v), type(0), 1.0e-6);
}
