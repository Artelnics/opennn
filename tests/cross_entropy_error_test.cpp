//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_test.h"

CrossEntropyErrorTest::CrossEntropyErrorTest() : UnitTesting() 
{
    cross_entropy_error.set(&neural_network, &data_set);
    cross_entropy_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


CrossEntropyErrorTest::~CrossEntropyErrorTest()
{
}


void CrossEntropyErrorTest::test_constructor()
{
    cout << "test_constructor\n";

    CrossEntropyError cross_entropy_error;
}


void CrossEntropyErrorTest::test_destructor()
{
    cout << "test_destructor\n";

    CrossEntropyError* cross_entropy_error = new CrossEntropyError;

    delete cross_entropy_error;
}


void CrossEntropyErrorTest::test_back_propagate()
{
    cout << "test_back_propagate\n";

    // Empty test does not work
    // cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool switch_train = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation, switch_train);

        // Loss index

        back_propagation.set(samples_number, &cross_entropy_error);
        cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_differentiation_gradient = cross_entropy_error.calculate_numerical_differentiation_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_differentiation_gradient, type(1.0e-3)), LOG);
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1;
        neurons_number = 1 + rand()%10;
        bool switch_train = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_binary_random();
        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation, switch_train);

        // Loss index

        back_propagation.set(samples_number, &cross_entropy_error);
        cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_differentiation_gradient = cross_entropy_error.calculate_numerical_differentiation_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_differentiation_gradient, type(1.0e-2)), LOG);
    }
}


void CrossEntropyErrorTest::run_test_case()
{
    cout << "Running cross-entropy error test case...\n";

    // Test constructor

    test_constructor();
    test_destructor();

    // Back-propagation methods

    test_back_propagate();

    cout << "End of cross-entropy error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
