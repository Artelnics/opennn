#include "pch.h"

#include "../opennn/transformer.h"
#include "../opennn/cross_entropy_error_3d.h"
#include "../opennn/language_data_set.h"
#include "../opennn/stochastic_gradient_descent.h"
#include "../opennn/mean_squared_error.h"

TEST(StochasticGradientDescentTest, DefaultConstructor)
{
    StochasticGradientDescent adaptive_moment_estimation;

    EXPECT_TRUE(!adaptive_moment_estimation.has_loss_index());
}


TEST(StochasticGradientDescentTest, GeneralConstructor)
{
    MeanSquaredError mean_squared_error;
    StochasticGradientDescent adaptive_moment_estimation(&mean_squared_error);

    EXPECT_TRUE(adaptive_moment_estimation.has_loss_index());
}


TEST(StochasticGradientDescentTest, Train)
{
    MeanSquaredError mean_squared_error;
    StochasticGradientDescent adaptive_moment_estimation(&mean_squared_error);

    type old_error = numeric_limits<float>::max();

    type error = 0;


    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    DataSet data_set(samples_number, {inputs_number}, {outputs_number});
    data_set.set_data_constant(type(1));
    /*
    neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number }, {}, { outputs_number });
    neural_network.set_parameters_constant(type(1));

    stochastic_gradient_descent.set_maximum_epochs_number(1);
    stochastic_gradient_descent.set_display(false);
    training_results = stochastic_gradient_descent.perform_training();

    EXPECT_EQ(training_results.get_epochs_number() <= 1);

    // Test

    data_set.set(1, 1, 1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number }, {}, { outputs_number });
    neural_network.set_parameters_constant(-1);

    stochastic_gradient_descent.set_maximum_epochs_number(1);

    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error < old_error);

    // Test

    old_error = error;

    stochastic_gradient_descent.set_maximum_epochs_number(10000);
    neural_network.set_parameters_constant(-1);

    stochastic_gradient_descent.set_display(true);
    stochastic_gradient_descent.set_display_period(1000);
    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error <= old_error);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    stochastic_gradient_descent.set_loss_goal(training_loss_goal);
    stochastic_gradient_descent.set_maximum_epochs_number(1000);
    stochastic_gradient_descent.set_maximum_time(1000.0);

    training_results = stochastic_gradient_descent.perform_training();

    EXPECT_EQ(training_results.get_training_error() <= training_loss_goal);
*/
}


TEST(StochasticGradientDescentTest, TrainTransformer)
{
    type old_error = numeric_limits<float>::max();

    type error = 0;

    Index context_length = 0;
    Index context_dimension = 0;
    Index input_dimensions = 0;

    LanguageDataSet language_data_set;

    Index depth;
    Index perceptron_depth;
    Index heads_number;
    Index layers_number;

    Transformer transformer;

    CrossEntropyError3D cross_entropy_error_3d(&transformer, &language_data_set);
/*
    stochastic_gradient_descent.set_loss_index(&cross_entropy_error_3d);

    samples_number = 1;

    inputs_number = 2;
    context_length = 3;
    input_dimensions = 5;
    context_dimension = 6;

    depth = 4;
    perceptron_depth = 6;
    heads_number = 4;
    layers_number = 1;

    data_set.set_data_random_language_model(samples_number, inputs_number, context_length, input_dimensions, context_dimension);

    transformer.set({ inputs_number, context_length, input_dimensions, context_dimension,
                        depth, perceptron_depth, heads_number, layers_number });

    stochastic_gradient_descent.set_loss_goal(NUMERIC_LIMITS_MIN);

    // Test

    stochastic_gradient_descent.set_maximum_epochs_number(1);
    stochastic_gradient_descent.set_display(false);

    training_results = stochastic_gradient_descent.perform_training();

    EXPECT_EQ(training_results.get_epochs_number() <= 1);

    // Test

    transformer.set_parameters_constant(-1);

    stochastic_gradient_descent.set_maximum_epochs_number(1);

    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error < old_error);

    // Test

    old_error = error;

    stochastic_gradient_descent.set_maximum_epochs_number(10000);
    transformer.set_parameters_constant(-1);

    stochastic_gradient_descent.set_display(true);
    stochastic_gradient_descent.set_display_period(1000);

    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error <= old_error);
*/
}
