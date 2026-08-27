#include "tests/pch.h"
#include "opennn/core/tensor_types.h"
#include "opennn/training_strategy/error_functions.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "gtest/gtest.h"

using namespace opennn;

namespace
{

// Logits, not probabilities: CrossEntropyError3d fuses the softmax, so these rows are fed
// to it raw. They happen to sum to 1, which is harmless and keeps the fixture readable.
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

    // logsumexp(row) - row[target], averaged over the two non-padding tokens.
    // Both rows share the same exponentials, so both have logsumexp = 1.658963691:
    //   token 0: 1.658963691 - 0.6 = 1.058963691
    //   token 2: 1.658963691 - 0.1 = 1.558963691
    const float expected_error = 1.308963691f;
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

    // 0.5 * softmax(row), with 0.5 subtracted at the target class.
    EXPECT_NEAR(gradients(0, 0, 0), type(0.105176975),  1.0e-5);
    EXPECT_NEAR(gradients(0, 0, 1), type(0.116238534),  1.0e-5);
    EXPECT_NEAR(gradients(0, 0, 2), type(-0.326592484), 1.0e-5);
    EXPECT_NEAR(gradients(0, 0, 3), type(0.105176975),  1.0e-5);

    EXPECT_NEAR(gradients(0, 1, 0), type(0), 1.0e-6);
    EXPECT_NEAR(gradients(0, 1, 1), type(0), 1.0e-6);
    EXPECT_NEAR(gradients(0, 1, 2), type(0), 1.0e-6);
    EXPECT_NEAR(gradients(0, 1, 3), type(0), 1.0e-6);

    EXPECT_NEAR(gradients(0, 2, 0), type(0.105176975),  1.0e-5);
    EXPECT_NEAR(gradients(0, 2, 1), type(-0.394823025), 1.0e-5);
    EXPECT_NEAR(gradients(0, 2, 2), type(0.116238534),  1.0e-5);
    EXPECT_NEAR(gradients(0, 2, 3), type(0.173407516),  1.0e-5);
}

// Softmax cross-entropy depends on differences between logits, never on their absolute
// level, so adding a constant to a whole row must leave both loss and gradient untouched.
// This is the property the max subtraction exists to preserve; without it the exponentials
// overflow long before the mathematics would.
TEST(CrossEntropyError3DTest, IsInvariantToAConstantShiftOfTheLogits)
{
    const Index batch = 1;
    const Index sequence_length = 3;
    const Index vocabulary_size = 4;

    VectorR targets(sequence_length);
    targets << type(2), type(0), type(1);
    TensorView target_view(targets.data(), { sequence_length });

    const auto measure = [&](float shift, float& error, Tensor3& gradients)
    {
        Tensor3 logits(batch, sequence_length, vocabulary_size);
        fill_logits(logits);
        for (Index s = 0; s < sequence_length; ++s)
            for (Index v = 0; v < vocabulary_size; ++v)
                logits(0, s, v) += type(shift);

        TensorView input_view(logits.data(), { batch, sequence_length, vocabulary_size });
        TensorView gradient_view(gradients.data(), { batch, sequence_length, vocabulary_size });

        Index active_tokens = 0;
        Index correct_tokens = 0;
        cross_entropy_3d(input_view, target_view, error, active_tokens, correct_tokens);
        cross_entropy_3d_gradient(input_view, target_view, gradient_view, active_tokens);
    };

    float plain_error = 0;
    float shifted_error = 0;
    Tensor3 plain_gradients(batch, sequence_length, vocabulary_size);
    Tensor3 shifted_gradients(batch, sequence_length, vocabulary_size);

    measure(0.0f, plain_error, plain_gradients);

    // Large enough that exp() of the raw logits would overflow to infinity.
    measure(200.0f, shifted_error, shifted_gradients);

    EXPECT_TRUE(std::isfinite(shifted_error));
    EXPECT_NEAR(shifted_error, plain_error, 1.0e-5f);

    for (Index s = 0; s < sequence_length; ++s)
        for (Index v = 0; v < vocabulary_size; ++v)
        {
            EXPECT_TRUE(std::isfinite(float(shifted_gradients(0, s, v))));
            EXPECT_NEAR(shifted_gradients(0, s, v), plain_gradients(0, s, v), 1.0e-5);
        }
}

// The gradient of the fused loss is softmax(logits) - onehot(target), so over the tokens it
// counts it must sum to zero per row: the mass it adds to the non-target classes is exactly
// what it removes from the target. A row that does not sum to zero is a dropped or
// double-counted term, which a tolerance-based comparison can easily miss.
TEST(CrossEntropyError3DTest, GradientRowsSumToZeroOverCountedTokens)
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

    for (Index s = 0; s < sequence_length; ++s)
    {
        float row_sum = 0;
        for (Index v = 0; v < vocabulary_size; ++v)
            row_sum += float(gradients(0, s, v));

        EXPECT_NEAR(row_sum, 0.0f, 1.0e-6f);
    }
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
