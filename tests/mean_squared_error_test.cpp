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
    {
        samples_number = 1;
        inputs_number = 1;
        outputs_number = 1;
        neurons_number = 1;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();

        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
        assert_true(back_propagation.gradient.size() == inputs_number+inputs_number*neurons_number+outputs_number+outputs_number*neurons_number, LOG);

        assert_true(is_zero(back_propagation.gradient) , LOG);
    }

    // Test approximation all random
    {
        samples_number = 1 + rand()%5;
        inputs_number = 1 + rand()%5;
        outputs_number = 1 + rand()%5;
        neurons_number = 1 + rand()%5;

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

        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
    }

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

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.errors.dimension(0) == 1, LOG);
        assert_true(back_propagation.errors.dimension(1) == 1, LOG);
        assert_true(back_propagation.error - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-3)), LOG);

    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
    }


    // Test forecasting trivial
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

        neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error < type(1e-1), LOG);
        assert_true(is_zero(back_propagation.gradient, type(1e-1)), LOG);
    }

    // Test forecasting random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Forecasting, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= type(0), LOG);

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-1)), LOG);
    }
    // Test convolutional
   {
       const Index input_images = 3;
       const Index input_kernels = 3;

       const Index channels = 3;

       const Index rows_input = 3;
       const Index cols_input = 3;
       const Index rows_kernel = 2;
       const Index cols_kernel = 2;

       data_set.set_data_file_name("E:/opennn/blank/test-6px-python-bmp/");

       data_set.read_bmp();

       const Index samples_number = data_set.get_training_samples_number();

       const Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

       DataSetBatch batch(samples_number, &data_set);

       batch.fill(samples_indices, input_variables_indices, target_variables_indices);

       Eigen::array<Eigen::Index, 4> extents = {0, 0, 0, 0};
       Eigen::array<Eigen::Index, 4> offsets = {batch.inputs_4d.dimension(0),
                                                batch.inputs_4d.dimension(1)-1, //padding
                                                batch.inputs_4d.dimension(2),
                                                batch.inputs_4d.dimension(3)};

       //remove padding
       Tensor<float, 4> new_batch = batch.inputs_4d.slice(extents, offsets);
       batch.inputs_4d = new_batch;

       //set dimensions

       Tensor<type,4> kernel(rows_kernel, cols_kernel, channels, input_kernels);
       Tensor<type,1> bias(input_kernels);

       const Index inputs_number_convolution = (rows_input)*(cols_input)*channels;
       const Index output_number_convolution = (rows_input - rows_kernel + 1)*(cols_input - cols_kernel + 1)*input_kernels;

       //set values

       kernel.chip(0,3).setConstant(type(1./3.));
       kernel.chip(1,3).setConstant(type(1./9.));
       kernel.chip(2,3).setConstant(type(1./27.));

//       bias.setValues({0, 0, 0});

       neural_network.set(NeuralNetwork::ProjectType::ImageClassification,
                          {inputs_number_convolution, output_number_convolution, 1});

       ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(neural_network.get_layer_pointer(0));
       FlattenLayer* flatten_layer = static_cast<FlattenLayer*>(neural_network.get_layer_pointer(1));
       PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));

       //set_dims //this should be inside nn contructor.

       cout<<batch.inputs_4d<<endl;
       system("pause");
       convolutional_layer->set(batch.inputs_4d, kernel, bias);

       /*
       //set dims //this should be inside nn contructor.
       flatten_layer->set(convolutional_layer->get_outputs_dimensions());


       //set values
       convolutional_layer->set_synaptic_weights(kernel);
       convolutional_layer->set_biases(bias);

       perceptron_layer->set_synaptic_weights_constant(1.);
       perceptron_layer->set_biases_constant(0);

       //start

       forward_propagation.set(input_images, &neural_network);
       neural_network.forward_propagate(batch, forward_propagation);
       forward_propagation.print();

       cout<<"parameters"<<endl;
       cout<<neural_network.get_parameters()<<endl;

       // create Dataset object to load data.
       gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
       */
   }
}


void MeanSquaredErrorTest::test_back_propagate_lm()
{
    cout << "test_back_propagate_lm\n";

    // Test approximation random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);


        // not running in  visual studio
        /*
        back_propagation_lm.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-1)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-1)), LOG);
        */
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // not running in visual studio
        /*
        back_propagation_lm.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
        */
    }

    // Test multiple classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // not running in visual studio
        /*
        back_propagation_lm.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();
        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
        */
    }
}


void MeanSquaredErrorTest::run_test_case()
{
    cout << "Running mean squared error test case...\n";

    test_constructor();
    test_destructor();

    // Back propagate methods

    test_back_propagate();

    test_back_propagate_lm();

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
