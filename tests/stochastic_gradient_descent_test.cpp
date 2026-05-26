#include "pch.h"
#include "../opennn/tabular_dataset.h"

#include "../opennn/language_dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/loss.h"
#include "../opennn/stochastic_gradient_descent.h"

using namespace opennn;

TEST(StochasticGradientDescentTest, DefaultConstructor)
{
    StochasticGradientDescent adaptive_moment_estimation;

    EXPECT_TRUE(adaptive_moment_estimation.get_loss() == nullptr);
}


TEST(StochasticGradientDescentTest, GeneralConstructor)
{
    Loss loss;
    StochasticGradientDescent adaptive_moment_estimation(&loss);

    EXPECT_TRUE(adaptive_moment_estimation.get_loss() != nullptr);
}


TEST(StochasticGradientDescentTest, Train)
{
    type old_error = numeric_limits<type>::max();

    type error = 0;

    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    TabularDataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_constant(type(1));

    ApproximationNetwork neural_network({inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    StochasticGradientDescent stochastic_gradient_descent(&loss);
    stochastic_gradient_descent.set_maximum_epochs(1);
    stochastic_gradient_descent.set_display(false);

    TrainingResults training_results = stochastic_gradient_descent.train();

    EXPECT_LE(training_results.get_epochs_number(), 1);

    // Test

    dataset.set_data_random();

    neural_network.set_parameters_random();

    training_results = stochastic_gradient_descent.train();
    error = training_results.get_training_error();

    EXPECT_LE(error, old_error);

    // Test

    old_error = error;

    stochastic_gradient_descent.set_maximum_epochs(10000);
    stochastic_gradient_descent.set_display(true);
    stochastic_gradient_descent.set_display_period(1000);

    neural_network.set_parameters_random();

    training_results = stochastic_gradient_descent.train();
    error = training_results.get_training_error();

    EXPECT_LE(error, old_error);

    // Loss goal

    type training_loss_goal = type(0.1);

    stochastic_gradient_descent.set_loss_goal(training_loss_goal);
    stochastic_gradient_descent.set_maximum_epochs(1000);
    stochastic_gradient_descent.set_maximum_time(1000);

    training_results = stochastic_gradient_descent.train();

    EXPECT_LE(training_results.get_training_error(), training_loss_goal);
}


TEST(StochasticGradientDescentTest, TrainTransformer)
{

    type old_error = numeric_limits<type>::max();

    type error = 0;

    LanguageDataset language_dataset;

    Index samples_number = 1;

    Index inputs_number = 2;
    Index context_length = 3;
    Index input_shape = 5;
    Index context_dimension = 6;

    Index depth = 4;
    Index dense_depth = 6;
    Index heads_number = 4;
    Index layers_number = 1;

    Transformer transformer(inputs_number,
                            context_length,
                            input_shape,
                            context_dimension,
                            depth,
                            heads_number,
                            dense_depth,
                            layers_number);

    Loss cross_entropy_loss(&transformer, &language_dataset);
    cross_entropy_loss.set_error(Loss::Error::CrossEntropy);

    StochasticGradientDescent stochastic_gradient_descent(&cross_entropy_loss);
/*
    language_dataset.set_data_random_language_model(samples_number, inputs_number, context_length, input_shape, context_dimension);

    transformer.set({ inputs_number, context_length, input_shape, context_dimension,
                        depth, dense_depth, heads_number, layers_number });

    stochastic_gradient_descent.set_loss_goal(EPSILON);

    // Test

    stochastic_gradient_descent.set_maximum_epochs(1);
    stochastic_gradient_descent.set_display(false);

    training_results = stochastic_gradient_descent.train();

//    EXPECT_EQ(training_results.get_epochs_number() <= 1);

    // Test

    transformer.set_parameters_constant(-1);

    stochastic_gradient_descent.set_maximum_epochs(1);

    training_results = stochastic_gradient_descent.train();
    error = training_results.get_training_error();

//    EXPECT_EQ(error < old_error);

    // Test

    old_error = error;

    stochastic_gradient_descent.set_maximum_epochs(10000);
    transformer.set_parameters_constant(-1);

    stochastic_gradient_descent.set_display(true);
    stochastic_gradient_descent.set_display_period(1000);

    training_results = stochastic_gradient_descent.train();
    error = training_results.get_training_error();

//    EXPECT_EQ(error <= old_error);
*/
}
