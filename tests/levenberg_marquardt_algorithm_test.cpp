#include "pch.h"

#include "../opennn/mean_squared_error.h"
#include "../opennn/levenberg_marquardt_algorithm.h"


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


/*
void LevenbergMarquardtAlgorithmTest::test_perform_training()
{
    type old_error = numeric_limits<float>::max();

    Index samples_number;
    Index inputs_number;
    Index outputs_number;

    type error;

    // Test

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;

    data_set.set(1,1,1);
    data_set.set_data_constant(type(1));

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_constant(type(1));

    levenberg_marquardt_algorithm.set_maximum_epochs_number(1);
    levenberg_marquardt_algorithm.set_display(false);
    /*
    training_results = levenberg_marquardt_algorithm.perform_training();

    assert_true(training_results.get_epochs_number() <= 1, LOG);
    
    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_constant(-1);

    levenberg_marquardt_algorithm.set_maximum_epochs_number(1);
    
    training_results = levenberg_marquardt_algorithm.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);
    
    // Test

    old_error = error;

    levenberg_marquardt_algorithm.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);
    
    training_results = levenberg_marquardt_algorithm.perform_training();
    error = training_results.get_training_error();

    assert_true(error <= old_error, LOG);
    
    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(training_loss_goal);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0);
    levenberg_marquardt_algorithm.set_maximum_epochs_number(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);
    /*
    training_results = levenberg_marquardt_algorithm.perform_training();

    assert_true(training_results.get_training_error() <= training_loss_goal, LOG);
    
    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(type(0));
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(minimum_loss_decrease);
    levenberg_marquardt_algorithm.set_maximum_epochs_number(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);
    
    training_results = levenberg_marquardt_algorithm.perform_training();

    assert_true(levenberg_marquardt_algorithm.get_minimum_loss_decrease() <= minimum_loss_decrease, LOG);
    
}
}
*/
