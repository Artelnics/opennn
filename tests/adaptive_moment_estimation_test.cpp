#include "pch.h"

#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/mean_squared_error.h"

#include "../opennn/transformer.h"
#include "../opennn/language_dataset.h"
#include "../opennn/cross_entropy_error_3d.h"
#include "gtest/gtest.h"

using namespace opennn;

TEST(AdaptiveMomentEstimationTest, DefaultConstructor)
{
    AdaptiveMomentEstimation adaptive_moment_estimation;

    EXPECT_EQ(adaptive_moment_estimation.has_loss_index(), false);
}


TEST(AdaptiveMomentEstimationTest, GeneralConstructor)
{
    MeanSquaredError mean_squared_error;
    AdaptiveMomentEstimation adaptive_moment_estimation(&mean_squared_error);

    EXPECT_TRUE(adaptive_moment_estimation.has_loss_index());
}


TEST(AdaptiveMomentEstimationTest, TrainEmpty)
{
    AdaptiveMomentEstimation adaptive_moment_estimation;

    const TrainingResults training_results = adaptive_moment_estimation.perform_training();

    EXPECT_EQ(adaptive_moment_estimation.has_loss_index(), false);
}


TEST(AdaptiveMomentEstimationTest, TrainApproximation)
{
    Dataset dataset(1, {1}, {1});
    dataset.set_data_constant(type(1));
    
    ApproximationNetwork neural_network({1}, {1}, {1});
    // neural_network.set_parameters_constant(type(1));

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    AdaptiveMomentEstimation adaptive_moment_estimation(&mean_squared_error);
    adaptive_moment_estimation.set_maximum_epochs_number(1);
    adaptive_moment_estimation.set_display(false);

    const TrainingResults training_results = adaptive_moment_estimation.perform_training();

    EXPECT_LE(training_results.get_epochs_number(), 1);
}

// @todo
TEST(AdaptiveMomentEstimationTest, TrainTransformer)
{
    // constexpr Index batch_size = 1;
    // constexpr Index input_length = 2;
    // constexpr Index decoder_length = 3;
    // constexpr Index input_dimensions = 5;
    // constexpr Index context_dimension = 6;
    // constexpr Index depth = 4;
    // constexpr Index perceptron_depth = 6;
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
    // language_dataset.set(LanguageDataset::SampleUse::Training);

    // opennn::Transformer transformer({
    //     input_length,
    //     decoder_length,
    //     input_dimensions,
    //     context_dimension,
    //     depth,
    //     perceptron_depth,
    //     heads_number,
    //     layers_number
    // });

    // CrossEntropyError3d cross_entropy_error_3d(&transformer, &language_dataset);
    // AdaptiveMomentEstimation adaptive_moment_estimation(&cross_entropy_error_3d);

    // adaptive_moment_estimation.set_display(true);
    // adaptive_moment_estimation.set_display_period(100);
    // adaptive_moment_estimation.set_loss_goal(training_loss_goal);
    // adaptive_moment_estimation.set_maximum_epochs_number(1000);
    // adaptive_moment_estimation.set_maximum_time(1000.0);
    // const TrainingResults training_results = adaptive_moment_estimation.perform_training();

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

    MeanSquaredError loss(&neural_network, &dataset);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);

    adaptive_moment_estimation.set_maximum_epochs_number(1);

    TrainingResults training_results = adaptive_moment_estimation.perform_training();
    const type error1 = training_results.get_training_error();

    adaptive_moment_estimation.set_maximum_epochs_number(50);
    // neural_network.set_parameters_constant(-1);

    training_results = adaptive_moment_estimation.perform_training();
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
    neural_network.set_parameters_random();

    MeanSquaredError loss(&neural_network, &dataset);
    AdaptiveMomentEstimation adaptive_moment_estimation(&loss);

    const type training_loss_goal = type(0.05);

    adaptive_moment_estimation.set_loss_goal(training_loss_goal);
    adaptive_moment_estimation.set_maximum_epochs_number(10000);
    adaptive_moment_estimation.set_maximum_time(1000.0);

    TrainingResults training_results = adaptive_moment_estimation.perform_training();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);
}

