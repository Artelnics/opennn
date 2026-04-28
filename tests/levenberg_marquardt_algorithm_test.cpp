#include "pch.h"

#include "../opennn/dense_layer.h"
#include "../opennn/loss.h"
#include "../opennn/dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/levenberg_marquardt_algorithm.h"

using namespace opennn;

TEST(LevenbergMarquardtAlgorithmTest, DefaultConstructor)
{
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm;

    EXPECT_EQ(levenberg_marquardt_algorithm.get_loss() == nullptr, true);
}


TEST(LevenbergMarquardtAlgorithmTest, GeneralConstructor)
{
    Loss loss;

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&loss);

    EXPECT_EQ(levenberg_marquardt_algorithm.get_loss() != nullptr, true);
}


TEST(LevenbergMarquardtAlgorithmTest, Train)
{
    const Index samples = 20;
    const Index inputs_n = 1;
    const Index outputs_n = 1;

    Dataset dataset(samples, {inputs_n}, {outputs_n});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_n}, Shape{outputs_n}, "Linear"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");

    LevenbergMarquardtAlgorithm lm(&loss);
    lm.set_display(false);
    lm.set_maximum_epochs(10);

    TrainingResults results = lm.train();

    EXPECT_GE(results.get_epochs_number(), 1);
    EXPECT_GE(results.get_training_error(), type(0));
}


TEST(LevenbergMarquardtAlgorithmTest, TrainReducesError)
{
    const Index samples = 30;
    const Index inputs_n = 2;
    const Index outputs_n = 1;

    Dataset dataset(samples, {inputs_n}, {outputs_n});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    // Use raw Dense network (no Scaling/Unscaling) — LM Jacobian only handles Dense<2>
    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_n}, Shape{3}, "HyperbolicTangent"));
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{3}, Shape{outputs_n}, "Linear"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    loss.set_regularization("None");

    const type initial_error = loss.calculate_numerical_error();

    LevenbergMarquardtAlgorithm lm(&loss);
    lm.set_display(false);
    lm.set_maximum_epochs(50);

    TrainingResults results = lm.train();

    EXPECT_LE(results.get_training_error(), initial_error + type(0.1));
}
