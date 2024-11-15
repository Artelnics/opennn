#include "pch.h"

#include "../opennn/transformer.h"
#include "../opennn/cross_entropy_error_3d.h"
#include "../opennn/language_data_set.h"


/*
void StochasticGradientDescentTest::test_constructor()
{
    cout << "test_constructor\n";

    StochasticGradientDescent stochastic_gradient_descent_1;
    assert_true(!stochastic_gradient_descent_1.has_loss_index(), LOG);

    // Loss index constructor

    StochasticGradientDescent stochastic_gradient_descent_2(&sum_squared_error);
    assert_true(stochastic_gradient_descent_2.has_loss_index(), LOG);
}


void StochasticGradientDescentTest::test_perform_training()
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

    stochastic_gradient_descent.set_maximum_epochs_number(1);
    stochastic_gradient_descent.set_display(false);
    training_results = stochastic_gradient_descent.perform_training();

    assert_true(training_results.get_epochs_number() <= 1, LOG);

    // Test

    data_set.set(1,1,1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_constant(-1);

    stochastic_gradient_descent.set_maximum_epochs_number(1);

    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Test

    old_error = error;

    stochastic_gradient_descent.set_maximum_epochs_number(10000);
    neural_network.set_parameters_constant(-1);

    stochastic_gradient_descent.set_display(true);
    stochastic_gradient_descent.set_display_period(1000);
    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    assert_true(error <= old_error, LOG);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    stochastic_gradient_descent.set_loss_goal(training_loss_goal);
    stochastic_gradient_descent.set_maximum_epochs_number(1000);
    stochastic_gradient_descent.set_maximum_time(1000.0);

    training_results = stochastic_gradient_descent.perform_training();

    assert_true(training_results.get_training_error() <= training_loss_goal, LOG);
}


void StochasticGradientDescentTest::test_transformer_training()
{
    cout << "\ntest_transformer_training\n";

    type old_error = numeric_limits<float>::max();

    type error;

    Index context_length;
    Index context_dimension;
    Index input_dimensions;

    LanguageDataSet data_set;

    Index depth;
    Index perceptron_depth;
    Index heads_number;
    Index layers_number;

    Transformer transformer;

    CrossEntropyError3D cross_entropy_error_3d;

    cross_entropy_error_3d.set(&transformer, &data_set);

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

    assert_true(training_results.get_epochs_number() <= 1, LOG);

    // Test

    transformer.set_parameters_constant(-1);

    stochastic_gradient_descent.set_maximum_epochs_number(1);

    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    assert_true(error < old_error, LOG);

    // Test

    old_error = error;

    stochastic_gradient_descent.set_maximum_epochs_number(10000);
    transformer.set_parameters_constant(-1);
    
    stochastic_gradient_descent.set_display(true);
    stochastic_gradient_descent.set_display_period(1000);

    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();

    assert_true(error <= old_error, LOG);
}


void StochasticGradientDescentTest::test_to_XML()
{
    cout << "test_to_XML\n";

    XMLPrinter file_stream;

    stochastic_gradient_descent.to_XML(file_stream);
}

}
*/