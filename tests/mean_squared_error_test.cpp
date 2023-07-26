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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.errors.dimension(0) == 1, LOG);
        assert_true(back_propagation.errors.dimension(1) == 1, LOG);
        assert_true(back_propagation.error - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-3)), LOG);

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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);
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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= type(0), LOG);

//        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-1)), LOG);
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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

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

    // Test convolutional
    {

//        samples_number = 1;
//        inputs_number = 1;
//        outputs_number = 1;
//        neurons_number = 1;
//        bool is_training = true;

//        const Index channels = 3;

//        const Index rows_input = 4;
//        const Index cols_input = 4;
//        const Index rows_kernel = 3;
//        const Index cols_kernel = 3;

//        //set dimensions

//        Tensor<type,4> input_batch(rows_input, cols_input, channels, input_images);
//        Tensor<type,4> kernel(rows_kernel, cols_kernel, channels, input_kernels);
//        Tensor<type,1> bias(input_kernels);

//        const Index inputs_number_convolution = (rows_input)*(cols_input)*channels*input_images;
//        const Index output_number_convolution = (rows_input - rows_kernel + 1)*(cols_input - cols_kernel + 1)*input_kernels*input_images;

//        //set values

//        input_batch.setConstant(1.);

//        input_batch.chip(0,3).chip(0,2).setConstant(2.);
//        input_batch.chip(0,3).chip(1,2).setConstant(3.);
//        input_batch.chip(0,3).chip(2,2).setConstant(4.);

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation, is_training);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);

//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();

//        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
//        assert_true(back_propagation.gradient.size() == inputs_number+inputs_number*neurons_number+outputs_number+outputs_number*neurons_number, LOG);

//        assert_true(is_zero(back_propagation.gradient) , LOG);

    }

    // Test approximation all random
    {
        samples_number = 1 + rand()%5;
        inputs_number = 1 + rand()%5;
        outputs_number = 1 + rand()%5;
        neurons_number = 1 + rand()%5;
        bool is_training = true;

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
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_differentiation_gradient, type(1.0e-3)), LOG);
    }

    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(1));

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.errors.dimension(0) == 1, LOG);
        assert_true(back_propagation.errors.dimension(1) == 1, LOG);
        assert_true(back_propagation.error - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_differentiation_gradient, type(1.0e-3)), LOG);
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
        bool is_training = true;
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_differentiation_gradient, type(1.0e-2)), LOG);
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
        bool is_training = true;
        neural_network.forward_propagate(batch, forward_propagation, is_training);

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
        bool is_training = true;
        neural_network.forward_propagate(batch, forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error >= type(0), LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_differentiation_gradient, type(1.0e-1)), LOG);
    }

    // Test convolutional
   if(false)
    {

//       const Index input_images = 3;
//       const Index input_kernels = 3;

//       const Index channels = 3;

//       const Index rows_input = 3;
//       const Index cols_input = 3;
//       const Index rows_kernel = 2;
//       const Index cols_kernel = 2;

//       data_set.set_data_file_name("../../blank/test-6px-python-bmp/");

//       data_set.read_bmp();

//       const Index samples_number = data_set.get_training_samples_number();

//       const Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
//       const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//       const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

//       DataSetBatch batch(samples_number, &data_set);

//       batch.fill(samples_indices, input_variables_indices, target_variables_indices);

//       Eigen::array<Eigen::Index, 4> extents = {0, 0, 0, 0};
//       Eigen::array<Eigen::Index, 4> offsets = {batch.inputs_4d.dimension(0),
//                                                batch.inputs_4d.dimension(1)-1, //padding
//                                                batch.inputs_4d.dimension(2),
//                                                batch.inputs_4d.dimension(3)};

//       //remove padding
//       Tensor<float, 4> new_batch = batch.inputs_4d.slice(extents, offsets);
//       batch.inputs_4d = new_batch;

//       //set dimensions

//       Tensor<type,4> kernel(rows_kernel, cols_kernel, channels, input_kernels);
//       Tensor<type,1> bias(input_kernels);

//       const Index inputs_number_convolution = (rows_input)*(cols_input)*channels;
//       const Index output_number_convolution = (rows_input - rows_kernel + 1)*(cols_input - cols_kernel + 1)*input_kernels;

//       //set values

//       kernel.chip(0,3).setConstant(type(1./3.));
//       kernel.chip(1,3).setConstant(type(1./9.));
//       kernel.chip(2,3).setConstant(type(1./27.));

////       bias.setValues({0, 0, 0});

//       neural_network.set(NeuralNetwork::ProjectType::ImageClassification,
//                          {inputs_number_convolution, output_number_convolution, 1});

//       ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(neural_network.get_layer_pointer(0));
//       FlattenLayer* flatten_layer = static_cast<FlattenLayer*>(neural_network.get_layer_pointer(1));
//       PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));

//       //set_dims //this should be inside nn contructor.

//       cout<<batch.inputs_4d<<endl;
//       getchar();
//       convolutional_layer->set(batch.inputs_4d, kernel, bias);

//       //set dims //this should be inside nn contructor.
//       flatten_layer->set(convolutional_layer->get_outputs_dimensions());


//       //set values
//       convolutional_layer->set_synaptic_weights(kernel);
//       convolutional_layer->set_biases(bias);

//       perceptron_layer->set_synaptic_weights_constant(1.);
//       perceptron_layer->set_biases_constant(0);

//       //start

//       forward_propagation.set(input_images, &neural_network);
//       neural_network.forward_propagate(batch, forward_propagation);
//       forward_propagation.print();

//       cout<<"parameters"<<endl;
//       cout<<neural_network.get_parameters()<<endl;

//       // create Dataset object to load data.
//       numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();
   }

}


void MeanSquaredErrorTest::test_calculate_gradient_convolutional_network()
{
    cout << "test_calculate_gradient_convolutional_network\n";

//    const Index images_number = 2;

//    Tensor<Index, 1> inputs_dimensions(3);
//    inputs_dimensions(0) = 2;
//    inputs_dimensions(1) = 2;
//    inputs_dimensions(2) = 2;

//    Tensor<type, 2> data(images_number,9);

//    // Image 1

//    data(0,0) = 1;
//    data(0,1) = 5;
//    data(0,2) = 2;
//    data(0,3) = 6;

//    data(0,4) = 3;
//    data(0,5) = 7;
//    data(0,6) = 4;
//    data(0,7) = 8;

//    data(0,8) = 0; // Target

//    // Image 2

//    data(1,0) = 9;
//    data(1,1) = 13;
//    data(1,2) = 10;
//    data(1,3) = 14;
//    forward_propagation.set(samples_number, &neural_network);
//    bool is_training = true;
//    neural_network.forward_propagate(batch, forward_propagation, is_training);

//    data(1,4) = 11;
//    data(1,5) = 15;
//    data(1,6) = 12;
//    data(1,7) = 16;

//    data(1,8) = 1; // Target

//    DataSet data_set(images_number,1,1);
//    //    data_set.set_data_constant(3.1416);
//    data_set.set_input_variables_dimensions(inputs_dimensions);
//    data_set.set_data(data); // 2d data

//    const Tensor<Index, 1> training_samples_indices = data_set.get_training_samples_indices();
//    const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//    const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

//    DataSetBatch batch(images_number, &data_set);

//    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//    //    batch.print();

//    Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
//    convolutional_layer_inputs_dimensions(0) = 2;
//    convolutional_layer_inputs_dimensions(1) = 2;
//    convolutional_layer_inputs_dimensions(2) = 2;
//    convolutional_layer_inputs_dimensions(3) = images_number;
        // not running in  visual studio
//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();
//        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, numerical_differentiation_gradient, type(1.0e-1)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-1)), LOG);
//    }

//    NeuralNetwork neural_network;

//    Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
//    convolutional_layer_kernels_dimensions(0) = 1;
//    convolutional_layer_kernels_dimensions(1) = 1;
//    convolutional_layer_kernels_dimensions(2) = images_number;
//    convolutional_layer_kernels_dimensions(3) = 1;

//    ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

//    convolutional_layer->set_synaptic_weights_constant(0.5);
//    convolutional_layer->set_biases_constant(0);

//    Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
//    flatten_layer_inputs_dimensions(0) = 2;
//    flatten_layer_inputs_dimensions(1) = 2;
//    flatten_layer_inputs_dimensions(2) = 1;
//    flatten_layer_inputs_dimensions(3) = 2;

//    FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

//    PerceptronLayer* perceptron_layer = new PerceptronLayer(4, 1);

//    perceptron_layer->set_synaptic_weights_constant(1);
//    perceptron_layer->set_biases_constant(0);
//    perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
//    forward_propagation.set(samples_number, &neural_network);
//    bool is_training = true;
//    neural_network.forward_propagate(batch, forward_propagation, is_training);

//    neural_network.add_layer(convolutional_layer);
//    neural_network.add_layer(flatten_layer);
//    neural_network.add_layer(perceptron_layer);

//    NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

//    neural_network.forward_propagate(batch, forward_propagation);
//    //    forward_propagation.print();

//    MeanSquaredError mean_squared_error(&neural_network, &data_set);

//    LossIndexBackPropagation back_propagation(2, &mean_squared_error);

//    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//    //    cout << "Gradient: " << back_propagation.gradient << endl;

        // not running in visual studio

//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();
//        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, numerical_differentiation_gradient, type(1.0e-2)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
//    }

//    gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();


//    cout << "Numerical gradient: " << gradient_numerical_differentiation << endl;

        // Data set

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
//        bool is_training = true;
//        neural_network.forward_propagate(batch, forward_propagation, is_training);

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        MeanSquaredError.back_propagate(batch, forward_propagation, back_propagation);
        // not running in visual studio

//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        numerical_differentiation_gradient = mean_squared_error.calculate_numerical_differentiation_gradient();
//        jacobian_numerical_differentiation = mean_squared_error.calculate_jacobian_numerical_differentiation();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, numerical_differentiation_gradient, type(1.0e-2)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, jacobian_numerical_differentiation, type(1.0e-2)), LOG);
//    }

//}
//    {
//        const Index images_number = 2;

//        Tensor<Index, 1> inputs_dimensions(3);
//        inputs_dimensions(0) = 2;
//        inputs_dimensions(1) = 2;
//        inputs_dimensions(2) = 2;

//        Tensor<type, 2> data(images_number,9);

//        // Image 1

//        data(0,0) = 1;
//        data(0,1) = 5;
//        data(0,2) = 2;
//        data(0,3) = 6;

//        data(0,4) = 3;
//        data(0,5) = 7;
//        data(0,6) = 4;
//        data(0,7) = 8;

//        data(0,8) = 0; // Target

//        // Image 2

//        data(1,0) = 9;
//        data(1,1) = 13;
//        data(1,2) = 10;
//        data(1,3) = 14;

//        data(1,4) = 11;
//        data(1,5) = 15;
//        data(1,6) = 12;
//        data(1,7) = 16;

//        data(1,8) = 1; // Target

//        DataSet data_set(images_number,1,1);
//        data_set.set_input_variables_dimensions(inputs_dimensions);
//        data_set.set_data(data); // 2d data

//        const Tensor<Index, 1> training_samples_indices = data_set.get_training_samples_indices();
//        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

//        DataSetBatch batch(images_number, &data_set);

//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        //    batch.print();

//        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
//        convolutional_layer_inputs_dimensions(0) = 2;
//        convolutional_layer_inputs_dimensions(1) = 2;
//        convolutional_layer_inputs_dimensions(2) = 2;
//        convolutional_layer_inputs_dimensions(3) = images_number;

//        NeuralNetwork neural_network;

//        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
//        convolutional_layer_kernels_dimensions(0) = 1;
//        convolutional_layer_kernels_dimensions(1) = 1;
//        convolutional_layer_kernels_dimensions(2) = images_number;
//        convolutional_layer_kernels_dimensions(3) = 1;

//        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

//        Tensor<type, 4> kernels(1,1,2,1);

//        kernels(0) = 0.5;
//        kernels(1) = 0.7;

//        convolutional_layer->set_synaptic_weights(kernels);

//        //    convolutional_layer->set_synaptic_weights_constant(0.5);
//        convolutional_layer->set_biases_constant(0);

//        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
//        flatten_layer_inputs_dimensions(0) = 2;
//        flatten_layer_inputs_dimensions(1) = 2;
//        flatten_layer_inputs_dimensions(2) = 1;
//        flatten_layer_inputs_dimensions(3) = 2;

//        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

//        PerceptronLayer* perceptron_layer = new PerceptronLayer(4, 2);

//        Tensor<type, 2> synaptic_weights(4,2);
//        synaptic_weights(0,0) = 1;
//        synaptic_weights(1,0) = 1;
//        synaptic_weights(2,0) = 1;
//        synaptic_weights(3,0) = 1;

//        synaptic_weights(0,1) = 2;
//        synaptic_weights(1,1) = 2;
//        synaptic_weights(2,1) = 2;
//        synaptic_weights(3,1) = 2;

//        //    perceptron_layer->set_synaptic_weights_constant(1);
//        perceptron_layer->set_synaptic_weights(synaptic_weights);
//        perceptron_layer->set_biases_constant(0);
//        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

//        cout << "Perceptron layer weights: " << perceptron_layer->get_synaptic_weights() << endl;

//        neural_network.add_layer(convolutional_layer);
//        neural_network.add_layer(flatten_layer);
//        neural_network.add_layer(perceptron_layer);

//        NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

//        neural_network.forward_propagate(batch, forward_propagation);
//        forward_propagation.print();

//        system("pause");

//        MeanSquaredError mean_squared_error(&neural_network, &data_set);

//        LossIndexBackPropagation back_propagation(2, &mean_squared_error);

//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        //    cout << "Gradient: " << back_propagation.gradient << endl;

//        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();

//        cout << "Numerical gradient: " << gradient_numerical_differentiation << endl;
//    }

    // Inputs 3x3x2x2; Filters: 1x1x2; Perceptrons: 2 --> Test ok
/*
    {
        const Index images_number = 2;

        Tensor<Index, 1> inputs_dimensions(3);
        inputs_dimensions(0) = 2; // Channels number
        inputs_dimensions(1) = 3; // rows number
        inputs_dimensions(2) = 3; // columns number

        Tensor<type, 2> data(images_number,19);

        // Image 1

        data(0,0) = 1;
        data(0,1) = 10;
        data(0,2) = 2;
        data(0,3) = 11;
        data(0,4) = 3;
        data(0,5) = 12;
        data(0,6) = 4;
        data(0,7) = 13;
        data(0,8) = 5;
        data(0,9) = 14;
        data(0,10) = 6;
        data(0,11) = 15;
        data(0,12) = 7;
        data(0,13) = 16;
        data(0,14) = 8;
        data(0,15) = 17;
        data(0,16) = 9;
        data(0,17) = 18;

        data(0,18) = 0; // Target

        // Image 2

        data(1,0) = 19;
        data(1,1) = 28;
        data(1,2) = 20;
        data(1,3) = 29;
        data(1,4) = 21;
        data(1,5) = 30;
        data(1,6) = 22;
        data(1,7) = 31;
        data(1,8) = 23;
        data(1,9) = 32;
        data(1,10) = 24;
        data(1,11) = 33;
        data(1,12) = 25;
        data(1,13) = 34;
        data(1,14) = 26;
        data(1,15) = 35;
        data(1,16) = 27;
        data(1,17) = 36;

        data(1,18) = 1; // Target

        DataSet data_set(images_number,1,1);
        data_set.set_input_variables_dimensions(inputs_dimensions);
        data_set.set_data(data); // 2d data

        const Tensor<Index, 1> training_samples_indices = data_set.get_training_samples_indices();
        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        DataSetBatch batch(images_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        batch.print();

        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
        convolutional_layer_inputs_dimensions(0) = 3;
        convolutional_layer_inputs_dimensions(1) = 3;
        convolutional_layer_inputs_dimensions(2) = 2;
        convolutional_layer_inputs_dimensions(3) = images_number;

        NeuralNetwork neural_network;

        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
        convolutional_layer_kernels_dimensions(0) = 1;
        convolutional_layer_kernels_dimensions(1) = 1;
        convolutional_layer_kernels_dimensions(2) = images_number;
        convolutional_layer_kernels_dimensions(3) = 1;

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

        Tensor<type, 4> kernels(1,1,2,1);

        kernels(0) = 0.5;
        kernels(1) = 0.7;

        convolutional_layer->set_synaptic_weights(kernels);

    //    convolutional_layer->set_synaptic_weights_constant(0.5);
        convolutional_layer->set_biases_constant(0);

        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
        flatten_layer_inputs_dimensions(0) = 3;
        flatten_layer_inputs_dimensions(1) = 3;
        flatten_layer_inputs_dimensions(2) = 1;
        flatten_layer_inputs_dimensions(3) = 2;

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

        PerceptronLayer* perceptron_layer = new PerceptronLayer(9, 2);

        Tensor<type, 2> synaptic_weights(9,2);
        synaptic_weights(0,0) = 1;
        synaptic_weights(1,0) = 1;
        synaptic_weights(2,0) = 1;
        synaptic_weights(3,0) = 1;
        synaptic_weights(4,0) = 1;
        synaptic_weights(5,0) = 1;
        synaptic_weights(6,0) = 1;
        synaptic_weights(7,0) = 1;
        synaptic_weights(8,0) = 1;

        synaptic_weights(0,1) = 2;
        synaptic_weights(1,1) = 2;
        synaptic_weights(2,1) = 2;
        synaptic_weights(3,1) = 2;
        synaptic_weights(4,1) = 2;
        synaptic_weights(5,1) = 2;
        synaptic_weights(6,1) = 2;
        synaptic_weights(7,1) = 2;
        synaptic_weights(8,1) = 2;
    //    perceptron_layer->set_synaptic_weights_constant(1);
        perceptron_layer->set_synaptic_weights(synaptic_weights);
        perceptron_layer->set_biases_constant(0);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        cout << "Perceptron layer weights: " << perceptron_layer->get_synaptic_weights() << endl;

        neural_network.add_layer(convolutional_layer);
        neural_network.add_layer(flatten_layer);
        neural_network.add_layer(perceptron_layer);

        NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

        neural_network.forward_propagate(batch, forward_propagation);
//        forward_propagation.print();

//        system("pause");

        MeanSquaredError mean_squared_error(&neural_network, &data_set);

        LossIndexBackPropagation back_propagation(2, &mean_squared_error);

        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    //    cout << "Gradient: " << back_propagation.gradient << endl;

        gradient_numerical_differentiation = mean_squared_error.calculate_gradient_numerical_differentiation();

        cout << "Numerical gradient: " << endl << gradient_numerical_differentiation << endl;

    }

    */

    // Inputs 3x3x2x2; Filters: 2x2x2; Perceptrons: 1 --> Working (4-Jan-23)
/*
    {
        bool is_training = true;

        const Index input_channels_number = 2;
        const Index input_rows_number = 3;
        const Index input_columns_number = 3;

        const Index images_number = 2;

        Tensor<Index, 1> inputs_dimensions(3);
        inputs_dimensions(0) = input_channels_number; // Channels number
        inputs_dimensions(1) = input_rows_number; // rows number
        inputs_dimensions(2) = input_columns_number; // columns number

        Tensor<type, 2> data(images_number,19);

        // Image 1

        data(0,0) = 1;
        data(0,1) = 10;
        data(0,2) = 2;
        data(0,3) = 11;
        data(0,4) = 3;
        data(0,5) = 12;
        data(0,6) = 4;
        data(0,7) = 13;
        data(0,8) = 5;
        data(0,9) = 14;
        data(0,10) = 6;
        data(0,11) = 15;
        data(0,12) = 7;
        data(0,13) = 16;
        data(0,14) = 8;
        data(0,15) = 17;
        data(0,16) = 9;
        data(0,17) = 18;

        data(0,18) = 0; // Target

        // Image 2

        data(1,0) = 19;
        data(1,1) = 28;
        data(1,2) = 20;
        data(1,3) = 29;
        data(1,4) = 21;
        data(1,5) = 30;
        data(1,6) = 22;
        data(1,7) = 31;
        data(1,8) = 23;
        data(1,9) = 32;
        data(1,10) = 24;
        data(1,11) = 33;
        data(1,12) = 25;
        data(1,13) = 34;
        data(1,14) = 26;
        data(1,15) = 35;
        data(1,16) = 27;
        data(1,17) = 36;

        data(1,18) = 1; // Target

        DataSet data_set(images_number,1,1);
        data_set.set_input_variables_dimensions(inputs_dimensions);

        data_set.set_data(data); // 2d data

        const Tensor<Index, 1> training_samples_indices = data_set.get_training_samples_indices();
//        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        Tensor<Index, 1> input_variables_indices(18);
        input_variables_indices(0) = 0;
        input_variables_indices(1) = 1;
        input_variables_indices(2) = 2;
        input_variables_indices(3) = 3;
        input_variables_indices(4) = 4;
        input_variables_indices(5) = 5;
        input_variables_indices(6) = 6;
        input_variables_indices(7) = 7;
        input_variables_indices(8) = 8;
        input_variables_indices(9) = 9;
        input_variables_indices(10) = 10;
        input_variables_indices(11) = 11;
        input_variables_indices(12) = 12;
        input_variables_indices(13) = 13;
        input_variables_indices(14) = 14;
        input_variables_indices(15) = 15;
        input_variables_indices(16) = 16;
        input_variables_indices(17) = 17;

        Tensor<Index, 1> target_variables_indices(1);
        target_variables_indices(0) = 18;

        DataSetBatch batch(images_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
        convolutional_layer_inputs_dimensions(0) = input_rows_number;
        convolutional_layer_inputs_dimensions(1) = input_columns_number;
        convolutional_layer_inputs_dimensions(2) = input_channels_number;
        convolutional_layer_inputs_dimensions(3) = images_number;

        NeuralNetwork neural_network;

        const Index kernels_rows_number = 2;
        const Index kernels_columns_number = 2;
        const Index kernels_channels_number = input_channels_number;
        const Index kernels_number = 2;

        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
        convolutional_layer_kernels_dimensions(0) = kernels_rows_number;
        convolutional_layer_kernels_dimensions(1) = kernels_columns_number;
        convolutional_layer_kernels_dimensions(2) = kernels_number;
        convolutional_layer_kernels_dimensions(3) = kernels_channels_number;

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

        Tensor<type, 4> kernels(kernels_rows_number,kernels_columns_number,images_number,kernels_channels_number);

        kernels(0,0,0,0) = static_cast<type>(0.5);
        kernels(0,1,0,0) = static_cast<type>(0.5);
        kernels(1,0,0,0) = static_cast<type>(0.5);
        kernels(1,1,0,0) = static_cast<type>(0.5);

        kernels(0,0,0,1) = static_cast<type>(0.5);
        kernels(0,1,0,1) = static_cast<type>(0.5);
        kernels(1,0,0,1) = static_cast<type>(0.5);
        kernels(1,1,0,1) = static_cast<type>(0.5);

        kernels(0,0,1,0) = static_cast<type>(0.7);
        kernels(0,1,1,0) = static_cast<type>(0.7);
        kernels(1,0,1,0) = static_cast<type>(0.7);
        kernels(1,1,1,0) = static_cast<type>(0.7);

        kernels(0,0,1,1) = static_cast<type>(0.7);
        kernels(0,1,1,1) = static_cast<type>(0.7);
        kernels(1,0,1,1) = static_cast<type>(0.7);
        kernels(1,1,1,1) = static_cast<type>(0.7);

        convolutional_layer->set_synaptic_weights(kernels);

        convolutional_layer->set_biases_constant(0);

        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
        flatten_layer_inputs_dimensions(0) = input_rows_number-kernels_rows_number+1;
        flatten_layer_inputs_dimensions(1) = input_columns_number-kernels_columns_number+1;
        flatten_layer_inputs_dimensions(2) = kernels_number;
        flatten_layer_inputs_dimensions(3) = images_number;

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

        const Index perceptron_layer_inputs_number = flatten_layer_inputs_dimensions(0)*flatten_layer_inputs_dimensions(1)*flatten_layer_inputs_dimensions(2);
        const Index perceptrons_number = 1;

        PerceptronLayer* perceptron_layer = new PerceptronLayer(perceptron_layer_inputs_number, perceptrons_number);

        Tensor<type, 2> synaptic_weights(perceptron_layer_inputs_number,perceptrons_number);

        for(Index i = 0; i < perceptron_layer_inputs_number; i++)
        {
            for(Index j = 0; j < perceptrons_number; j++)
            {
                synaptic_weights(i,j) = static_cast<type>(j+1);
            }
        }

//        synaptic_weights(0,0) = 1;
//        synaptic_weights(1,0) = 1;
//        synaptic_weights(2,0) = 1;
//        synaptic_weights(3,0) = 1;

//        synaptic_weights(0,1) = 2;
//        synaptic_weights(1,1) = 2;
//        synaptic_weights(2,1) = 2;
//        synaptic_weights(3,1) = 2;

        perceptron_layer->set_synaptic_weights(synaptic_weights);
        perceptron_layer->set_biases_constant(0);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        neural_network.add_layer(convolutional_layer);
        neural_network.add_layer(flatten_layer);
        neural_network.add_layer(perceptron_layer);

        NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

        neural_network.forward_propagate(batch, forward_propagation, is_training);
//        forward_propagation.print();

//        system("pause");

        MeanSquaredError mean_squared_error(&neural_network, &data_set);

        LossIndexBackPropagation back_propagation(images_number, &mean_squared_error);

        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        cout << "Gradient: " << endl << back_propagation.gradient << endl;

        const Tensor<type,1> gradient_numerical_differentiation = mean_squared_error.calculate_numerical_differentiation_gradient();

        cout << "Numerical gradient: " << endl << gradient_numerical_differentiation << endl;
    }
*/

    // Inputs 3x3x2x2; Filters: 2x2x2; Perceptrons: 1

    if(true)
    {
        bool is_training = true;

        const Index input_channels_number = 2;
        const Index input_rows_number = 3;
        const Index input_columns_number = 3;

        const Index images_number = 1;

        Tensor<Index, 1> inputs_dimensions(3);
        inputs_dimensions(0) = input_channels_number; // Channels number
        inputs_dimensions(1) = input_rows_number; // rows number
        inputs_dimensions(2) = input_columns_number; // columns number

        Tensor<type, 2> data(images_number,19);

        // Image 1

        data(0,0) = 17;
        data(0,1) = 8;
        data(0,2) = 16;
        data(0,3) = 7;
        data(0,4) = 15;
        data(0,5) = 6;
        data(0,6) = 14;
        data(0,7) = 5;
        data(0,8) = 13;
        data(0,9) = 4;
        data(0,10) = 12;
        data(0,11) = 3;
        data(0,12) = 11;
        data(0,13) = 2;
        data(0,14) = 10;
        data(0,15) = 1;
        data(0,16) = 9;
        data(0,17) = 0;

        data(0,18) = 1; // Target

        // Image 2

//        data(1,0) = 19;
//        data(1,1) = 20;
//        data(1,2) = 21;
//        data(1,3) = 22;
//        data(1,4) = 23;
//        data(1,5) = 24;
//        data(1,6) = 25;
//        data(1,7) = 26;
//        data(1,8) = 27;
//        data(1,9) = 28;
//        data(1,10) = 29;
//        data(1,11) = 30;
//        data(1,12) = 31;
//        data(1,13) = 32;
//        data(1,14) = 33;
//        data(1,15) = 34;
//        data(1,16) = 35;
//        data(1,17) = 36;

//        data(1,18) = 0; // Target

        DataSet data_set(images_number,1,1);

        data_set.set_data(data); // 2d data
//        data_set.set_data_random();

        cout << "Data: " << endl << data_set.get_data() << endl;

        system("pause");

        data_set.set_input_variables_dimensions(inputs_dimensions);

        data_set.set_training();

        const Tensor<Index, 1> training_samples_indices = data_set.get_training_samples_indices();
//        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        Tensor<Index, 1> input_variables_indices(18);
        input_variables_indices(0) = 0;
        input_variables_indices(1) = 1;
        input_variables_indices(2) = 2;
        input_variables_indices(3) = 3;
        input_variables_indices(4) = 4;
        input_variables_indices(5) = 5;
        input_variables_indices(6) = 6;
        input_variables_indices(7) = 7;
        input_variables_indices(8) = 8;
        input_variables_indices(9) = 9;
        input_variables_indices(10) = 17;
        input_variables_indices(11) = 16;
        input_variables_indices(12) = 15;
        input_variables_indices(13) = 14;
        input_variables_indices(14) = 13;
        input_variables_indices(15) = 12;
        input_variables_indices(16) = 11;
        input_variables_indices(17) = 10;


        Tensor<Index, 1> target_variables_indices(1);
        target_variables_indices(0) = 18;

        DataSetBatch batch(images_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        batch.print();
//        system("pause");

        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
        convolutional_layer_inputs_dimensions(0) = input_rows_number;
        convolutional_layer_inputs_dimensions(1) = input_columns_number;
        convolutional_layer_inputs_dimensions(2) = input_channels_number;
        convolutional_layer_inputs_dimensions(3) = images_number;

        NeuralNetwork neural_network;

        const Index kernels_rows_number = 2;
        const Index kernels_columns_number = 2;
        const Index kernels_channels_number = input_channels_number;
        const Index kernels_number = 2;

        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
        convolutional_layer_kernels_dimensions(0) = kernels_rows_number;
        convolutional_layer_kernels_dimensions(1) = kernels_columns_number;
        convolutional_layer_kernels_dimensions(2) = kernels_number;
        convolutional_layer_kernels_dimensions(3) = kernels_channels_number;

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

        Tensor<type, 4> kernels(kernels_rows_number, kernels_columns_number, kernels_channels_number, kernels_number);

        kernels(0,0,0,0) = static_cast<type>(0.5);
        kernels(0,1,0,0) = static_cast<type>(0.5);
        kernels(1,0,0,0) = static_cast<type>(0.5);
        kernels(1,1,0,0) = static_cast<type>(0.5);

        kernels(0,0,0,1) = static_cast<type>(0.5);
        kernels(0,1,0,1) = static_cast<type>(0.5);
        kernels(1,0,0,1) = static_cast<type>(0.5);
        kernels(1,1,0,1) = static_cast<type>(0.5);

        kernels(0,0,1,0) = static_cast<type>(0.7);
        kernels(0,1,1,0) = static_cast<type>(0.7);
        kernels(1,0,1,0) = static_cast<type>(0.7);
        kernels(1,1,1,0) = static_cast<type>(0.7);

        kernels(0,0,1,1) = static_cast<type>(0.7);
        kernels(0,1,1,1) = static_cast<type>(0.7);
        kernels(1,0,1,1) = static_cast<type>(0.7);
        kernels(1,1,1,1) = static_cast<type>(0.7);

        convolutional_layer->set_synaptic_weights(kernels);

        convolutional_layer->set_biases_constant(0);

//        convolutional_layer->set_parameters_random();

        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
        flatten_layer_inputs_dimensions(0) = input_rows_number-kernels_rows_number+1;
        flatten_layer_inputs_dimensions(1) = input_columns_number-kernels_columns_number+1;
        flatten_layer_inputs_dimensions(2) = kernels_number;
        flatten_layer_inputs_dimensions(3) = images_number;

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

        const Index perceptron_layer_inputs_number = flatten_layer_inputs_dimensions(0)*flatten_layer_inputs_dimensions(1)*flatten_layer_inputs_dimensions(2);
        const Index perceptrons_number = 1;

        PerceptronLayer* perceptron_layer = new PerceptronLayer(perceptron_layer_inputs_number, perceptrons_number);

        Tensor<type, 2> synaptic_weights(perceptron_layer_inputs_number,perceptrons_number);

        for(Index i = 0; i < perceptron_layer_inputs_number; i++)
        {
            for(Index j = 0; j < perceptrons_number; j++)
            {
                synaptic_weights(i,j) = static_cast<type>(1);//static_cast<type>(j+1);
            }
        }

//        cout << "Synaptic weights: " << synaptic_weights << endl;

//        perceptron_layer->set_parameters_random();

        synaptic_weights(0,0) = 1;
        synaptic_weights(1,0) = 1;
        synaptic_weights(2,0) = 1;
        synaptic_weights(3,0) = 1;

        synaptic_weights(0,1) = 2;
        synaptic_weights(1,1) = 2;
        synaptic_weights(2,1) = 2;
        synaptic_weights(3,1) = 2;

        perceptron_layer->set_synaptic_weights(synaptic_weights);
        perceptron_layer->set_biases_constant(0);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

//        perceptron_layer->set_parameters_random();

        neural_network.add_layer(convolutional_layer);
        neural_network.add_layer(flatten_layer);
        neural_network.add_layer(perceptron_layer);

        NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

        neural_network.forward_propagate(batch, forward_propagation, is_training);
//        forward_propagation.print();


//        system("pause");

        MeanSquaredError mean_squared_error(&neural_network, &data_set);

        LossIndexBackPropagation back_propagation(images_number, &mean_squared_error);

        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        cout << "Gradient: " << endl << back_propagation.gradient << endl;

        const Tensor<type,1> gradient_numerical_differentiation = mean_squared_error.calculate_numerical_differentiation_gradient();

        cout << "Numerical gradient: " << endl << gradient_numerical_differentiation << endl;

        cout << "Gradient   ;    Numerical gradient " << endl;

        for(Index i = 0; i < back_propagation.gradient.size(); i++)
        {
            cout << back_propagation.gradient(i) << " ; " << gradient_numerical_differentiation(i) <<  " ; " <<
                    std::abs((back_propagation.gradient(i) - gradient_numerical_differentiation(i))/gradient_numerical_differentiation(i)*100)
                 << "%" << endl;

        }
    }
}



void MeanSquaredErrorTest::run_test_case()
{
    cout << "Running mean squared error test case...\n";
/*
    test_constructor();
    test_destructor();
*/
    // Convolutional network methods

    test_calculate_gradient_convolutional_network();

    // Back propagate methods
/*
    test_back_propagate();
    test_back_propagate_lm();
*/
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
