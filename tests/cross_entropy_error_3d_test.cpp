//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_3d_test.h"

CrossEntropyError3DTest::CrossEntropyError3DTest() : UnitTesting()
{
    cross_entropy_error_3d.set(&neural_network, &data_set);
    cross_entropy_error_3d.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


CrossEntropyError3DTest::~CrossEntropyError3DTest()
{
}


void CrossEntropyError3DTest::test_constructor()
{
    cout << "test_constructor\n";

    CrossEntropyError3D cross_entropy_error_3d;
}


void CrossEntropyError3DTest::test_destructor()
{
    cout << "test_destructor\n";

    CrossEntropyError3D* cross_entropy_error_3d = new CrossEntropyError3D;

    delete cross_entropy_error_3d;
}


void CrossEntropyError3DTest::test_back_propagate()
{
    cout << "test_back_propagate\n";


    // Test approximation all zero
    {/* set data and LossIndex::calculate_output_delta()
        samples_number = 1;
        inputs_number = 1;
        inputs_dimension = 1;
        depth = 1;
        
        // Data set

        Tensor<type, 2> data(samples_number, inputs_number + inputs_number + 1);
        data.setValues({ {1, 1, 1} });
        data_set.set_data(data);

        data_set.set_raw_variable_use(0, DataSet::VariableUse::Input);
        data_set.set_raw_variable_use(1, DataSet::VariableUse::Target);
        data_set.set_raw_variable_use(2, DataSet::VariableUse::Target);

        cout << "Targets: " << endl << data_set.get_target_data() << endl;

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();

        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);
        
        // Neural network

        neural_network.set();
        
        EmbeddingLayer* embedding_layer = new EmbeddingLayer(inputs_dimension, inputs_number, depth);
        neural_network.add_layer(embedding_layer);

        ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D(inputs_number, depth, inputs_dimension + 1);
        neural_network.add_layer(probabilistic_layer_3d);
        
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);
        
        // Loss index

        back_propagation.set(samples_number, &cross_entropy_error_3d);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);
        
        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
        assert_true(back_propagation.gradient.size() == neural_network.get_parameters_number(), LOG);

        cout << "Gradient: " << endl << back_propagation.gradient << endl;

        assert_true(is_zero(back_propagation.gradient), LOG);*/
    }
    /*
    // Test approximation all random
    {
        samples_number = type(1) + rand() % 5;
        inputs_number = type(1) + rand() % 5;
        outputs_number = type(1) + rand() % 5;
        neurons_number = type(1) + rand() % 5;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number, neurons_number, outputs_number });
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &cross_entropy_error_3d);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }


    {
        samples_number = 10;
        inputs_number = 8;
        outputs_number = 9;
        neurons_number = 12;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number, neurons_number, outputs_number });
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &cross_entropy_error_3d);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);
    }*/
}


void CrossEntropyError3DTest::test_calculate_gradient_transformer()
{/*
    cout << "Running calculate gradient transformer test case...\n";

    // Test
    {
        data_set.set();
        neural_network.set();

        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

        assert_true(numerical_gradient.size() == 0, LOG);
    }

    // Test
    {
        data_set.set();
        neural_network.set();

        samples_number = 1;
        inputs_number = 1;

        inputs_dimension = 1;
        depth = 1;

        Tensor<type, 3> inputs(samples_number, inputs_number, inputs_dimension);
        inputs.setConstant(2);

        Tensor<type, 3> targets(samples_number, inputs_number, depth);
        targets.setConstant(1);
        
        ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D(inputs_number, inputs_dimension, depth);
        neural_network.add_layer(probabilistic_layer_3d);
        
        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient(inputs, targets);

        cout << "Numerical gradient:" << endl << numerical_gradient << endl;
        
        forward_propagation.set(samples_number, &neural_network);
        back_propagation.set(samples_number, &cross_entropy_error_3d);

        cross_entropy_error_3d.calculate_output_delta(get_pair(targets), forward_propagation, back_propagation);
        
        cross_entropy_error_3d.back_propagate(get_pair(inputs), get_pair(targets), forward_propagation, back_propagation);

        cout << "Analitical gradient:" << endl << back_propagation.gradient << endl;
        
    }

    cout << "End of calculate gradient transformer test case...\n";*/
}


void CrossEntropyError3DTest::run_test_case()
{
    cout << "Running cross-entropy error test case...\n";
    
    // Test constructor

    test_constructor();
    test_destructor();
    
    // Transformer test
    
    test_calculate_gradient_transformer();
    
    // Back-propagation methods

    test_back_propagate();
    
    cout << "End of cross-entropy error test case.\n\n";
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
