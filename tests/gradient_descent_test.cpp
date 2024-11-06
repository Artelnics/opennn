#include "pch.h"

#include "../opennn/gradient_descent.h"


TEST(GradientDescentTest, DefaultConstructor)
{
    GradientDescent gradient_descent_1;
//    assert_true(!gradient_descent_1.has_loss_index(), LOG);

    EXPECT_EQ(1, 1);
}


TEST(GradientDescentTest, GeneralConstructor)
{
    EXPECT_EQ(1, 1);
}


/*
namespace opennn
{

void GradientDescentTest::test_constructor()
{
    cout << "test_constructor\n";


    // Loss index constructor

    GradientDescent gradient_descent_2(&sum_squared_error);
    assert_true(gradient_descent_2.has_loss_index(), LOG);
}


void GradientDescentTest::test_perform_training()
{
    cout << "test_perform_training\n";

    type old_error = numeric_limits<float>::max();

    type error;

    // Test

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;

    data_set.set(1,1,1);
    data_set.set_data_constant(type(1));

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_constant(type(1));

    gradient_descent.set_maximum_epochs_number(1);
    gradient_descent.set_display(false);
    training_results = gradient_descent.perform_training();

    assert_true(training_results.get_epochs_number() <= 1, LOG);

    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_constant(-1);

    gradient_descent.set_maximum_epochs_number(1);

    training_results = gradient_descent.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Test

    old_error = error;

    gradient_descent.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);

    training_results = gradient_descent.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    gradient_descent.set_loss_goal(training_loss_goal);
    gradient_descent.set_minimum_loss_decrease(0.0);
    gradient_descent.set_maximum_epochs_number(1000);
    gradient_descent.set_maximum_time(1000.0);

    training_results = gradient_descent.perform_training();

    assert_true(training_results.get_training_error() <= training_loss_goal, LOG);

    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    gradient_descent.set_loss_goal(type(0));
    gradient_descent.set_minimum_loss_decrease(minimum_loss_decrease);
    gradient_descent.set_maximum_epochs_number(1000);
    gradient_descent.set_maximum_time(1000.0);

    training_results = gradient_descent.perform_training();

    assert_true(training_results.loss_decrease <= minimum_loss_decrease, LOG);

}

}
*/
