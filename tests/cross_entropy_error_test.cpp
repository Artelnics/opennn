//   OpenNN: Open Neural Networks Library
//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_test.h"

#include "../opennn/tensors.h"

namespace opennn
{

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

    cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);
    
    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {}, {outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, true);

        // Loss index

        back_propagation.set(samples_number, &cross_entropy_error);
        cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1;
        neurons_number = 1 + rand()%10;
        bool is_training = true;

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

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &cross_entropy_error);
        cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);
    }

    // Test classification convolutional layer
    {
        bool is_training = true;

        const Index kernel_height = 1;
        const Index kernel_width = 1;
        const Index kernel_channels = 1;
        const Index kernels_number = 1;

        // Data set

        image_data_set.set_display(false);
        image_data_set.set_data_source_path("data/conv_test");
/*
        image_data_set.read_bmp();
        image_data_set.scale_input_variables();

        samples_number = image_data_set.get_samples_number();

        training_samples_indices = image_data_set.get_training_samples_indices();
        input_variables_indices = image_data_set.get_input_variables_indices();
        target_variables_indices = image_data_set.get_target_variables_indices();

        batch.set(samples_number, &image_data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.delete_layers();

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(image_data_set.get_input_dimensions(), { kernel_height,kernel_width,kernel_channels,kernels_number });
        neural_network.add_layer(convolutional_layer);

        ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer->get_output_dimensions(), { 1,1,kernel_channels,kernels_number });
        neural_network.add_layer(convolutional_layer_2);

        FlattenLayer* flatten_layer = new FlattenLayer(convolutional_layer_2->get_output_dimensions());
        neural_network.add_layer(flatten_layer);

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(flatten_layer->get_output_dimensions(),image_data_set.get_target_dimensions());
        neural_network.add_layer(probabilistic_layer);

        neural_network.set_parameters_constant(type(0));
        //neural_network.set_parameters_random();
/*
         { // debug
            image_data_set.set_display(true);
            image_data_set.print();
            cout << image_data_set.get_data() << endl;
            neural_network.print();
        }

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        cross_entropy_error.set_data_set(&image_data_set);
        cross_entropy_error.set_neural_network(&neural_network);

        back_propagation.set(samples_number, &cross_entropy_error);
        cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

        // Numerical gradient

        numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

        // Assert

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == image_data_set.get_target_dimensions()[0], LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
       */
    }
/*
    // Test classification pooling layer
    {
        bool is_training = true;

        const Index pool_height = 1;
        const Index pool_width = 1;

        // Data set

        image_data_set.set_display(false);
        image_data_set.set_data_source_path("data/conv_test");
        image_data_set.read_bmp();
        image_data_set.scale_input_variables();

        samples_number = image_data_set.get_samples_number();

        training_samples_indices = image_data_set.get_training_samples_indices();
        input_variables_indices = image_data_set.get_input_variables_indices();
        target_variables_indices = image_data_set.get_target_variables_indices();

        batch.set(samples_number, &image_data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.delete_layers();

        PoolingLayer* pooling_layer = new PoolingLayer(image_data_set.get_input_dimensions(), { pool_height, pool_width });
        neural_network.add_layer(pooling_layer);

        PoolingLayer* pooling_layer_2 = new PoolingLayer(pooling_layer->get_output_dimensions(), { 1, 1 });
        neural_network.add_layer(pooling_layer_2);

        FlattenLayer* flatten_layer = new FlattenLayer(pooling_layer_2->get_output_dimensions());
        neural_network.add_layer(flatten_layer);

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(flatten_layer->get_output_dimensions(), image_data_set.get_target_dimensions());
        neural_network.add_layer(probabilistic_layer);

        //neural_network.set_parameters_constant(type(0));
        neural_network.set_parameters_random();

        { // debug
            image_data_set.set_display(true);
            image_data_set.print();
            cout << image_data_set.get_data() << endl;
            neural_network.print();
        }

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        cross_entropy_error.set_data_set(&image_data_set);
        cross_entropy_error.set_neural_network(&neural_network);

        back_propagation.set(samples_number, &cross_entropy_error);
        cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

        // Numerical gradient

        numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

        // Assert

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == image_data_set.get_target_dimensions()[0], LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);

    }

    // Test classification convolutional and pooling layers mix
    {
        bool is_training = true;

        const Index kernel_height = 1;
        const Index kernel_width = 1;
        const Index kernel_channels = 1;
        const Index kernels_number = 1;

        const Index pool_height = 2;
        const Index pool_width = 2;

        // Data set

        image_data_set.set_display(false);
        image_data_set.set_data_source_path("data/conv_test");
        image_data_set.read_bmp();
        image_data_set.scale_input_variables();

        samples_number = image_data_set.get_samples_number();

        training_samples_indices = image_data_set.get_training_samples_indices();
        input_variables_indices = image_data_set.get_input_variables_indices();
        target_variables_indices = image_data_set.get_target_variables_indices();

        batch.set(samples_number, &image_data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.delete_layers();

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(image_data_set.get_input_dimensions(), { kernel_height,kernel_width,kernel_channels,kernels_number });
        neural_network.add_layer(convolutional_layer);

        PoolingLayer* pooling_layer = new PoolingLayer(convolutional_layer->get_output_dimensions(), { pool_height, pool_width });
        neural_network.add_layer(pooling_layer);

        FlattenLayer* flatten_layer = new FlattenLayer(pooling_layer->get_output_dimensions());
        neural_network.add_layer(flatten_layer);

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(flatten_layer->get_output_dimensions(), image_data_set.get_target_dimensions());
        neural_network.add_layer(probabilistic_layer);

        //neural_network.set_parameters_constant(type(0));
        neural_network.set_parameters_random();

        { // debug
            image_data_set.set_display(true);
            image_data_set.print();
            cout << image_data_set.get_data() << endl;
            neural_network.print();
        }

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        cross_entropy_error.set_data_set(&image_data_set);
        cross_entropy_error.set_neural_network(&neural_network);

        back_propagation.set(samples_number, &cross_entropy_error);
        cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

        // Numerical gradient

        numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

        // Assert

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == image_data_set.get_target_dimensions()[0], LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);
    }
*/
}


void CrossEntropyErrorTest::run_test_case()
{
    cout << "Running cross-entropy error test case...\n";

    test_constructor();
    test_destructor();

    test_back_propagate();

    cout << "End of cross-entropy error test case.\n\n";
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
