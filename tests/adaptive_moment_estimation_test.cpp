#include "pch.h"

#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/loss.h"
#include "../opennn/language_dataset.h"
#include "gtest/gtest.h"

using namespace opennn;

TEST(AdaptiveMomentEstimationTest, DefaultConstructor)
{
    AdaptiveMomentEstimation adaptive_moment_estimation;

    EXPECT_EQ(adaptive_moment_estimation.get_loss() == nullptr, true);
}


TEST(AdaptiveMomentEstimationTest, GeneralConstructor)
{
    Loss loss;
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);

    EXPECT_TRUE(adaptive_moment_estimation.get_loss() != nullptr);
}


TEST(AdaptiveMomentEstimationTest, TrainEmpty)
{
    AdaptiveMomentEstimation adaptive_moment_estimation;

    const TrainingResults training_results = adaptive_moment_estimation.train();

    EXPECT_EQ(adaptive_moment_estimation.get_loss() == nullptr, true);
}


TEST(AdaptiveMomentEstimationTest, TrainApproximation)
{
    Dataset dataset(1, {1}, {1});
    dataset.set_data_constant(type(1));

    ApproximationNetwork neural_network({1}, {1}, {1});
    // neural_network.set_parameters_constant(type(1));

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);
    adaptive_moment_estimation.set_maximum_epochs(1);
    adaptive_moment_estimation.set_display(false);

    const TrainingResults training_results = adaptive_moment_estimation.train();

    EXPECT_LE(training_results.get_epochs_number(), 1);
}

// @todo
TEST(AdaptiveMomentEstimationTest, TrainTransformer)
{
    // constexpr Index batch_size = 1;
    // constexpr Index input_length = 2;
    // constexpr Index decoder_length = 3;
    // constexpr Index input_shape = 5;
    // constexpr Index context_dimension = 6;
    // constexpr Index depth = 4;
    // constexpr Index dense_depth = 6;
    // constexpr Index heads_number = 4;
    // constexpr Index layers_number = 1;
    // constexpr type training_loss_goal = static_cast<type>(0.05);

    // const vector<string> input_vocab = {"[PAD]", "[UNK]", "hello", "world"};
    // const vector<string> target_vocab = {"[PAD]", "[UNK]", "bonjour", "monde"};

    // LanguageDataset language_dataset;
    // language_dataset.set_input_vocabulary(input_vocab);
    // language_dataset.set_target_vocabulary(target_vocab);

    // vector<vector<string>> input_sequences = {{"hello", "world"}};
    // vector<vector<string>> target_sequences = {{"bonjour", "monde", "[PAD]"}};

    // language_dataset.encode_input_data(input_sequences);
    // language_dataset.encode_target_data(target_sequences);
    // language_dataset.set(LanguageDataset::"Training");

    // opennn::Transformer transformer({
    //     input_length,
    //     decoder_length,
    //     input_shape,
    //     context_dimension,
    //     depth,
    //     dense_depth,
    //     heads_number,
    //     layers_number
    // });

    // Loss cross_entropy_loss(&transformer, &language_dataset);
    // cross_entropy_loss.set_error(Loss::Error::CrossEntropy);
    // AdaptiveMomentEstimation adaptive_moment_estimation(&cross_entropy_loss);

    // adaptive_moment_estimation.set_display(true);
    // adaptive_moment_estimation.set_display_period(100);
    // adaptive_moment_estimation.set_loss_goal(training_loss_goal);
    // adaptive_moment_estimation.set_maximum_epochs(1000);
    // adaptive_moment_estimation.set_maximum_time(1000.0);
    // const TrainingResults training_results = adaptive_moment_estimation.train();

    // EXPECT_EQ(training_results.get_training_error(), training_loss_goal);
}


// @todo
TEST(AdaptiveMomentEstimationTest, PerformTrainingLossError)
{
    const Index samples_number = 10;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();

    ApproximationNetwork neural_network({inputs_number}, {}, {outputs_number});
    // neural_network.set_parameters_constant(-1);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);

    adaptive_moment_estimation.set_maximum_epochs(1);
    adaptive_moment_estimation.set_display(false);

    TrainingResults training_results = adaptive_moment_estimation.train();
    const type error1 = training_results.get_training_error();

    adaptive_moment_estimation.set_maximum_epochs(50);
    // neural_network.set_parameters_constant(-1);

    training_results = adaptive_moment_estimation.train();
    const type error2 = training_results.get_training_error();

    EXPECT_LT(error2, error1);
}


TEST(AdaptiveMomentEstimationTest, PerformTrainingLossGoal)
{
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();

    ApproximationNetwork neural_network({inputs_number}, {}, {outputs_number});

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);

    const type training_loss_goal = type(0.05);

    adaptive_moment_estimation.set_loss_goal(training_loss_goal);
    adaptive_moment_estimation.set_maximum_epochs(10000);
    adaptive_moment_estimation.set_maximum_time(1000.0);
    adaptive_moment_estimation.set_display(false);

    TrainingResults training_results = adaptive_moment_estimation.train();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);


}

