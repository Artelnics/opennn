//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   T E S T   C L A S S             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "mean_squared_error_test.h"


MeanSquaredErrorTest::MeanSquaredErrorTest() : UnitTesting() 
{
    mean_squared_error.set(&neural_network, &data_set);

    mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


MeanSquaredErrorTest::~MeanSquaredErrorTest()
{
}


void MeanSquaredErrorTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default

    MeanSquaredError mean_squared_error_1;

    assert_true(!mean_squared_error_1.has_neural_network(), LOG);
    assert_true(!mean_squared_error_1.has_data_set(), LOG);

    // Neural network and data set

    MeanSquaredError mean_squared_error_2(&neural_network, &data_set);

    assert_true(mean_squared_error_2.has_neural_network(), LOG);
    assert_true(mean_squared_error_2.has_data_set(), LOG);
}


void MeanSquaredErrorTest::test_destructor()
{
    cout << "test_destructor\n";

    // Default

    MeanSquaredError* mean_squared_error_1 = new MeanSquaredError;

    delete mean_squared_error_1;

    // Neural network and data set

    MeanSquaredError* mean_squared_error_2 = new MeanSquaredError(&neural_network, &data_set);

    delete mean_squared_error_2;
}


void MeanSquaredErrorTest::test_back_propagate()
{
    cout << "test_back_propagate\n";

    // Empty test does not work
    // mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    // Test approximation all zero
//    {
//        samples_number = 1;
//        inputs_number = 1;
//        outputs_number = 1;
//        neurons_number = 1;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_constant(type(0));

//        data_set.set_training();

//        training_samples_indices = data_set.get_training_samples_indices();

//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
//        neural_network.set_parameters_constant(type(0));

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
//        assert_true(back_propagation.gradient.size() == inputs_number+inputs_number*neurons_number+outputs_number+outputs_number*neurons_number, LOG);

//        assert_true(is_zero(back_propagation.gradient) , LOG);
//    }

//    // Test approximation all random
//    {
//        samples_number = 1 + rand()%5;
//        inputs_number = 1 + rand()%5;
//        outputs_number = 1 + rand()%5;
//        neurons_number = 1 + rand()%5;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_random();

//        data_set.set_training();

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
//        neural_network.set_parameters_random();

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
//    }

//    // Test binary classification trivial
//    {
//        inputs_number = 1;
//        outputs_number = 1;
//        samples_number = 1;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_constant(type(0));

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, outputs_number});
//        neural_network.set_parameters_constant(type(0));

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation.errors.dimension(0) == 1, LOG);
//        assert_true(back_propagation.errors.dimension(1) == 1, LOG);
//        assert_true(back_propagation.error - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-3)), LOG);

//    }

//    // Test binary classification random samples, inputs, outputs, neurons
//    {
//        samples_number = 1 + rand()%10;
//        inputs_number = 1 + rand()%10;
//        outputs_number = 1 + rand()%10;
//        neurons_number = 1 + rand()%10;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_binary_random();
//        data_set.set_training();

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
//        neural_network.set_parameters_random();

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation.error >= 0, LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
//    }


//    // Test forecasting trivial
//    {
//        inputs_number = 1;
//        outputs_number = 1;
//        samples_number = 1;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_constant(type(0));

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, outputs_number});
//        neural_network.set_parameters_constant(type(0));

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);


//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation.error < type(1e-1), LOG);
//        assert_true(is_zero(back_propagation.gradient, type(1e-1)), LOG);
//    }

//    // Test forecasting random samples, inputs, outputs, neurons
//    {
//        samples_number = 1 + rand()%10;
//        inputs_number = 1 + rand()%10;
//        outputs_number = 1 + rand()%10;
//        neurons_number = 1 + rand()%10;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_random();
//        data_set.set_training();

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, neurons_number, outputs_number});
//        neural_network.set_parameters_random();

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation.error >= type(0), LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-1)), LOG);
//    }
//}


//void MeanSquaredErrorTest::test_back_propagate_lm()
//{
//    cout << "test_back_propagate_lm\n";

//    // Test approximation random samples, inputs, outputs, neurons
//    {
//        samples_number = 1 + rand()%10;
//        inputs_number = 1 + rand()%10;
//        outputs_number = 1 + rand()%10;
//        neurons_number = 1 + rand()%10;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_random();
//        data_set.set_training();

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
//        neural_network.set_parameters_random();

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);


//        // not running in  visual studio
//        /*
//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
//        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-1)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-1)), LOG);
//        */
//    }

//    // Test binary classification random samples, inputs, outputs, neurons
//    {
//        samples_number = 1 + rand()%10;
//        inputs_number = 1 + rand()%10;
//        outputs_number = 1 + rand()%10;
//        neurons_number = 1 + rand()%10;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_binary_random();
//        data_set.set_training();

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
//        neural_network.set_parameters_random();

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        // not running in visual studio
//        /*
//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
//        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
//        */
//    }

//    // Test multiple classification random samples, inputs, outputs, neurons
//    {
//        samples_number = 1 + rand()%10;
//        inputs_number = 1 + rand()%10;
//        outputs_number = 1 + rand()%10;
//        neurons_number = 1 + rand()%10;

//        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_random();
//        data_set.set_training();

//        training_samples_indices = data_set.get_training_samples_indices();
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
//        neural_network.set_parameters_random();

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        // not running in visual studio
//        /*
//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
//        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
//        */
//    }

    // Test convolutional
    {
        const Index input_images = 1;
        const Index input_kernels = 3;

        const Index channels = 3;

        const Index rows_input = 4;
        const Index cols_input = 4;
        const Index rows_kernel = 3;
        const Index cols_kernel = 3;

        //set dimensions

        Tensor<type,4> input_batch(rows_input, cols_input, channels, input_images);
        Tensor<type,4> kernel(rows_kernel, cols_kernel, channels, input_kernels);
        Tensor<type,1> bias(input_kernels);

        const Index inputs_number_convolution = (rows_input)*(cols_input)*channels*input_images;
        const Index output_number_convolution = (rows_input - rows_kernel + 1)*(cols_input - cols_kernel + 1)*input_kernels*input_images;

        //set values

        input_batch.setConstant(1.);

        input_batch.chip(0,3).chip(0,2).setConstant(2.);
        input_batch.chip(0,3).chip(1,2).setConstant(3.);
        input_batch.chip(0,3).chip(2,2).setConstant(4.);

        kernel.chip(0,3).setConstant(type(1./3.));
        kernel.chip(1,3).setConstant(type(1./9.));
        kernel.chip(2,3).setConstant(type(1./27.));

        bias.setValues({0, 0, 0});

        neural_network.set(NeuralNetwork::ProjectType::ImageClassification,
                           {inputs_number_convolution, output_number_convolution, 1});

        ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(neural_network.get_layer_pointer(0));
        FlattenLayer* flatten_layer = static_cast<FlattenLayer*>(neural_network.get_layer_pointer(1));
        PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));


        //set_dims //this should be inside nn contructor.
        convolutional_layer->set(input_batch, kernel, bias);
        convolutional_layer->set(input_batch, kernel, bias);

        //set dims //this should be inside nn contructor.
        flatten_layer->set(convolutional_layer->get_outputs_dimensions());

        //set values
        convolutional_layer->set_synaptic_weights(kernel);
        convolutional_layer->set_biases(bias);

        perceptron_layer->set_synaptic_weights_constant(1.);
        perceptron_layer->set_biases_constant(0);

        //start

        batch.inputs_4d = input_batch;

        forward_propagation.set(input_images, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);
        forward_propagation.print();

        cout<<"parameters"<<endl;
        cout<<neural_network.get_parameters()<<endl;

        ///@todo
        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
    }
}


void MeanSquaredErrorTest::test_calculate_gradient_convolutional_network()
{
    cout << "test_calculate_gradient_convolutional_network\n";

    const Index images_number = 2;

    Tensor<Index, 1> inputs_dimensions(3);
    inputs_dimensions(0) = 2;
    inputs_dimensions(1) = 2;
    inputs_dimensions(2) = 2;

    Tensor<type, 2> data(images_number,9);

    // Image 1

    data(0,0) = 1;
    data(0,1) = 5;
    data(0,2) = 2;
    data(0,3) = 6;

    data(0,4) = 3;
    data(0,5) = 7;
    data(0,6) = 4;
    data(0,7) = 8;

    data(0,8) = 0; // Target

    // Image 2

    data(1,0) = 9;
    data(1,1) = 13;
    data(1,2) = 10;
    data(1,3) = 14;

    data(1,4) = 11;
    data(1,5) = 15;
    data(1,6) = 12;
    data(1,7) = 16;

    data(1,8) = 1; // Target

    DataSet data_set(images_number,1,1);
//    data_set.set_data_constant(3.1416);
    data_set.set_input_variables_dimensions(inputs_dimensions);
    data_set.set_data(data); // 2d data

    const Tensor<Index, 1> training_samples_indices = data_set.get_training_samples_indices();
    const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

    DataSetBatch batch(images_number, &data_set);

    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//    batch.print();

    Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
    convolutional_layer_inputs_dimensions(0) = 2;
    convolutional_layer_inputs_dimensions(1) = 2;
    convolutional_layer_inputs_dimensions(2) = 2;
    convolutional_layer_inputs_dimensions(3) = images_number;

    NeuralNetwork neural_network;

    Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
    convolutional_layer_kernels_dimensions(0) = 1;
    convolutional_layer_kernels_dimensions(1) = 1;
    convolutional_layer_kernels_dimensions(2) = images_number;
    convolutional_layer_kernels_dimensions(3) = 1;

    ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

    convolutional_layer->set_synaptic_weights_constant(0.5);
    convolutional_layer->set_biases_constant(0);

    Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
    flatten_layer_inputs_dimensions(0) = 2;
    flatten_layer_inputs_dimensions(1) = 2;
    flatten_layer_inputs_dimensions(2) = 1;
    flatten_layer_inputs_dimensions(3) = 2;

    FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

    PerceptronLayer* perceptron_layer = new PerceptronLayer(4, 1);

    perceptron_layer->set_synaptic_weights_constant(1);
    perceptron_layer->set_biases_constant(0);
    perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    neural_network.add_layer(convolutional_layer);
    neural_network.add_layer(flatten_layer);
    neural_network.add_layer(perceptron_layer);

    NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

    neural_network.forward_propagate(batch, forward_propagation);
    forward_propagation.print();

    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    LossIndexBackPropagation back_propagation(1, &mean_squared_error);

//    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();

    cout << "Numerical gradient: " << gradient_numerical_differentiation << endl;
}



void MeanSquaredErrorTest::run_test_case()
{
    cout << "Running mean squared error test case...\n";

//    test_constructor();
//    test_destructor();

    // Back propagate methods

//    test_back_propagate();
    test_calculate_gradient_convolutional_network();
//    test_back_propagate_lm();

    cout << "End of mean squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lemser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lemser General Public License for more details.

// You should have received a copy of the GNU Lemser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
