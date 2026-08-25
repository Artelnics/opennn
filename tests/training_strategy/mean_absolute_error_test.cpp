#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include "opennn/core/tensor_types.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/error_functions.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

TEST(MeanAbsoluteErrorTest, ValueAndFeatureReduction)
{
    MatrixR outputs(2, 3);
    outputs << 1.0f, 4.0f, 2.0f,
               3.0f, 1.0f, 8.0f;

    MatrixR targets(2, 3);
    targets << 1.0f, 2.0f, 5.0f,
               4.0f, 1.0f, 2.0f;

    float error = 0.0f;
    mean_absolute_error(TensorView(outputs.data(), {2, 3}),
                        TensorView(targets.data(), {2, 3}),
                        error,
                        nullptr);

    EXPECT_FLOAT_EQ(error, 2.0f);
}

TEST(MeanAbsoluteErrorTest, GradientSignAndZeroResidual)
{
    MatrixR outputs(1, 3);
    outputs << 1.0f, 4.0f, 2.0f;

    MatrixR targets(1, 3);
    targets << 1.0f, 2.0f, 5.0f;

    MatrixR deltas = MatrixR::Zero(1, 3);
    mean_absolute_error_gradient(TensorView(outputs.data(), {1, 3}),
                                 TensorView(targets.data(), {1, 3}),
                                 TensorView(deltas.data(), {1, 3}));

    EXPECT_FLOAT_EQ(deltas(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(deltas(0, 1), 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(deltas(0, 2), -1.0f / 3.0f);
}

TEST(MeanAbsoluteErrorTest, BackPropagate)
{
    TabularDataset dataset(4, {2}, {2});

    MatrixR data(4, 4);
    data << -0.8f, -0.4f,  0.6f,  0.9f,
             0.2f,  0.7f, -0.7f, -0.3f,
             0.9f, -0.1f,  0.1f,  0.5f,
            -0.3f,  0.5f,  0.8f, -0.8f;
    dataset.set_data(data);
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({2}, {3}, {2});

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanAbsoluteError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), 2.0e-3f);
}

TEST(MeanAbsoluteErrorTest, StringSelection)
{
    Loss loss;
    loss.set_error("MeanAbsoluteError");

    EXPECT_EQ(loss.get_error(), Loss::Error::MeanAbsoluteError);
    EXPECT_EQ(loss.get_name(), "MeanAbsoluteError");
}

TEST(MeanAbsoluteErrorTest, TrainingStrategySerialization)
{
    TabularDataset dataset(2, { 1 }, { 1 });
    ApproximationNetwork neural_network({ 1 }, { 2 }, { 1 });
    TrainingStrategy strategy(&neural_network, &dataset);
    strategy.set_loss("MeanAbsoluteError");

    const filesystem::path path =
        filesystem::temp_directory_path() / "opennn_mean_absolute_error.json";
    strategy.save(path);

    TrainingStrategy loaded(&neural_network, &dataset);
    loaded.load(path);

    ASSERT_NE(loaded.get_loss(), nullptr);
    EXPECT_EQ(loaded.get_loss()->get_error(), Loss::Error::MeanAbsoluteError);

    error_code error;
    filesystem::remove(path, error);
}
