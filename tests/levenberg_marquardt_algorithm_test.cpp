#include "pch.h"

#include "../opennn/perceptron_layer.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/levenberg_marquardt_algorithm.h"

using namespace opennn;

TEST(LevenbergMarquardtAlgorithmTest, DefaultConstructor)
{
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm;

    EXPECT_EQ(levenberg_marquardt_algorithm.has_loss_index(), false);
}


TEST(LevenbergMarquardtAlgorithmTest, GeneralConstructor)
{
    MeanSquaredError mean_squared_error;

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&mean_squared_error);

    EXPECT_EQ(levenberg_marquardt_algorithm.has_loss_index(), true);
}


TEST(LevenbergMarquardtAlgorithmTest, TrainEmpty)
{
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm;

    levenberg_marquardt_algorithm.perform_training();

//    EXPECT_EQ(levenberg_marquardt_algorithm.has_loss_index(), true);
}


TEST(LevenbergMarquardtAlgorithmTest, Train)
{
    /*
    const Index samples_number = get_random_index(1, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index outputs_number = get_random_index(1, 10);

    DataSet data_set(1, { 1 }, { 1 });
    //data_set.set_data_constant(type(1));
    data_set.set_data_random();
    data_set.set(DataSet::SampleUse::Training);

    //NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {1}, {1}, {1});
    //neural_network.set_parameters_random();
    //neural_network.set_parameters_constant(type(1));

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<PerceptronLayer>(dimensions{ 1 },
        dimensions{ 1 },
        PerceptronLayer::ActivationFunction::Linear));

    MeanSquaredError mean_squared_error(&neural_network, &data_set);
    
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&mean_squared_error);
    levenberg_marquardt_algorithm.set_maximum_epochs_number(1);
    //levenberg_marquardt_algorithm.set_display(false);

    //TrainingResults training_results = levenberg_marquardt_algorithm.perform_training();

    //system("pause");
/*
    EXPECT_LE(training_results.get_epochs_number(), 1);

    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_constant(-1);

    levenberg_marquardt_algorithm.set_maximum_epochs_number(1);

    training_results = levenberg_marquardt_algorithm.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error < old_error);

    // Test

    old_error = error;

    levenberg_marquardt_algorithm.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);

    training_results = levenberg_marquardt_algorithm.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error <= old_error);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(training_loss_goal);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0);
    levenberg_marquardt_algorithm.set_maximum_epochs_number(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);

    training_results = levenberg_marquardt_algorithm.perform_training();

    EXPECT_EQ(training_results.get_training_error() <= training_loss_goal);

    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(type(0));
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(minimum_loss_decrease);
    levenberg_marquardt_algorithm.set_maximum_epochs_number(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);

    training_results = levenberg_marquardt_algorithm.perform_training();

    EXPECT_EQ(levenberg_marquardt_algorithm.get_minimum_loss_decrease() <= minimum_loss_decrease);

    EXPECT_EQ(levenberg_marquardt_algorithm.has_loss_index(), true);
*/
}
