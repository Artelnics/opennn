//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   T E S T   C L A S S    
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "stochastic_gradient_descent_test.h"

#include "../opennn/transformer.h"
#include "../opennn/cross_entropy_error_3d.h"
#include "../opennn/language_data_set.h"

namespace opennn
{

StochasticGradientDescentTest::StochasticGradientDescentTest() : UnitTesting()
{
    sum_squared_error.set(&neural_network, &data_set);

    stochastic_gradient_descent.set_loss_index(&sum_squared_error);

    stochastic_gradient_descent.set_display(false);
}


StochasticGradientDescentTest::~StochasticGradientDescentTest()
{
}


void StochasticGradientDescentTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    StochasticGradientDescent stochastic_gradient_descent_1;
    assert_true(!stochastic_gradient_descent_1.has_loss_index(), LOG);

    // Loss index constructor

    StochasticGradientDescent stochastic_gradient_descent_2(&sum_squared_error);
    assert_true(stochastic_gradient_descent_2.has_loss_index(), LOG);
}


void StochasticGradientDescentTest::test_destructor()
{
    cout << "test_destructor\n";

    StochasticGradientDescent* stochastic_gradient_descent = new StochasticGradientDescent;
    delete stochastic_gradient_descent;
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
    Index number_of_layers;

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
    number_of_layers = 1;

    data_set.set_data_random_language_model(samples_number, inputs_number, context_length, input_dimensions, context_dimension);

    transformer.set({ inputs_number, context_length, input_dimensions, context_dimension,
                        depth, perceptron_depth, heads_number, number_of_layers });

    stochastic_gradient_descent.set_loss_goal(NUMERIC_LIMITS_MIN);

    // Test

    stochastic_gradient_descent.set_maximum_epochs_number(1);
    stochastic_gradient_descent.set_display(false);
/*
    training_results = stochastic_gradient_descent.perform_training();
*/
    assert_true(training_results.get_epochs_number() <= 1, LOG);


    // Test

    transformer.set_parameters_constant(-1);

    stochastic_gradient_descent.set_maximum_epochs_number(1);
/*
    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();
*/
    assert_true(error < old_error, LOG);

    // Test

    old_error = error;

    stochastic_gradient_descent.set_maximum_epochs_number(10000);
    transformer.set_parameters_constant(-1);
    
    stochastic_gradient_descent.set_display(true);
    stochastic_gradient_descent.set_display_period(1000);
/*
    training_results = stochastic_gradient_descent.perform_training();
    error = training_results.get_training_error();
*/
    assert_true(error <= old_error, LOG);

}


void StochasticGradientDescentTest::test_to_XML()
{
    cout << "test_to_XML\n";

    tinyxml2::XMLPrinter file_stream;

    stochastic_gradient_descent.to_XML(file_stream);
}


void StochasticGradientDescentTest::run_test_case()
{
    cout << "Running stochastic gradient descent test case...\n";

    // Constructor and destructor

    test_constructor();

    test_destructor();

    // Training

    test_perform_training();

    test_transformer_training();

    // Serialization

    test_to_XML();

    cout << "End of stochastic gradient descent test case.\n\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
