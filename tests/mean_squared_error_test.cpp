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

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, 1.0e-3), LOG);

    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1;
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

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, 1.0e-2), LOG);
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

        assert_true(back_propagation.error < type(NUMERIC_LIMITS_MIN), LOG);
        assert_true(is_zero(back_propagation.gradient), LOG);
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

        assert_true(are_equal(back_propagation.gradient, gradient_numerical_differentiation, type(1.0e-2)), LOG);

    }
}


void MeanSquaredErrorTest::test_back_propagate_lm()
{
    cout << "test_back_propagate_lm\n";

}



void MeanSquaredErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   Index samples_number;
   Index inputs_number;
   Index targets_number;

   Index neurons_number;

   Tensor<type, 2> data;

   Tensor<type, 1> parameters;

   // Test

   samples_number = 1;
   inputs_number = 1;
   targets_number = 1;

   data_set.set(samples_number, inputs_number, targets_number);
   data_set.set_data_constant(type(1));
   data_set.set_training();
   batch.set(samples_number, &data_set);

   neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, targets_number});
   neural_network.set_parameters_constant(type(0));

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   mean_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error < type(NUMERIC_LIMITS_MIN), LOG);

   // Test

   samples_number = 2;
   inputs_number = 2;
   targets_number = 2;

   data_set.set(samples_number, inputs_number, targets_number);

   data.resize(2, 4);
   
   data.setValues({{type(1), type(2), type(3),type(4)},{type(1), type(2), type(3),type(4)}});
   
   data_set.set_data(data);

   batch.set(samples_number, &data_set);

   neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);

   mean_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(abs(back_propagation.error - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

   assert_true(back_propagation.error == type(1.0), LOG);

}


void MeanSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   Index samples_number;

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_variables_indices;
   Tensor<Index, 1> target_variables_indices;

   Index inputs_number;
   Index outputs_number;
   Index neurons_number;

   RecurrentLayer* recurrent_layer = new RecurrentLayer;

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer;

   PerceptronLayer* perceptron_layer_1 = new PerceptronLayer();
   PerceptronLayer* perceptron_layer_2 = new PerceptronLayer();

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   // Trivial test

   samples_number = 10;
   inputs_number = 1;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_constant(type(0));
   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set_parameters_constant(type(0));

   perceptron_layer_1->set(inputs_number, outputs_number);
   neural_network.add_layer(perceptron_layer_1);

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);
   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   cout << "Error gradient 1:\n" << error_gradient << endl;
   cout << "Error gradient 2:\n" << numerical_error_gradient << endl;

   assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
   assert_true(all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(),
                      [](type i) { return (i-static_cast<type>(0))<NUMERIC_LIMITS_MIN; }), LOG);
   assert_true(are_equal(error_gradient, numerical_error_gradient), LOG);

   neural_network.set();

   // Test perceptron
/*
   samples_number = 10;
   inputs_number = 3;
   outputs_number = 5;
   neurons_number = 6;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();
   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   perceptron_layer_1 = new PerceptronLayer(inputs_number, neurons_number);
   //perceptron_layer_1->set_default();
   //perceptron_layer_1->set(inputs_number, neurons_number);
   perceptron_layer_2->set(neurons_number, outputs_number);

   neural_network.add_layer(perceptron_layer_1);
   neural_network.add_layer(perceptron_layer_2);

   neural_network.set_parameters_random();

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   cout << "Error gradient 1: " << error_gradient << endl;
   cout << "Error gradient 2: " << numerical_error_gradient << endl;

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   // Test perceptron and binary probabilistic

   neural_network.set();

   samples_number = 3;
   inputs_number = 3;
   neurons_number = 4;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_binary_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});

   neural_network.set_parameters_random();

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);


   // Test perceptron and multiple probabilistic

   neural_network.set();

   samples_number = 3;
   inputs_number = 3;
   neurons_number = 2;
   outputs_number = 3;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_binary_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});

   neural_network.set_parameters_random();

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);


   // Test lstm

   neural_network.set();

   samples_number = 4;
   inputs_number = 3;
   outputs_number = 2;
   neurons_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   long_short_term_memory_layer->set(inputs_number, neurons_number);

   neural_network.add_layer(long_short_term_memory_layer);

   neural_network.set_parameters_random();

   long_short_term_memory_layer->set_timesteps(2);

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   neural_network.set();

   // Test recurrent
   samples_number = 4;
   inputs_number = 2;
   outputs_number = 1;
   neurons_number = 3;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   recurrent_layer->set(inputs_number, neurons_number);

   neural_network.add_layer(recurrent_layer);

   neural_network.set_parameters_random();

   recurrent_layer->set_timesteps(2);

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   neural_network.set();

   // Test recurrent and perceptron

   samples_number = 4;
   inputs_number = 2;
   outputs_number = 1;
   neurons_number = 3;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   recurrent_layer->set(inputs_number, neurons_number);
   perceptron_layer_1->set(neurons_number, outputs_number);

   neural_network.add_layer(recurrent_layer);
   neural_network.add_layer(perceptron_layer_1);

   neural_network.set_parameters_random();

   recurrent_layer->set_timesteps(2);

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   neural_network.set();

   // Test recurrent and binary probabilistic

   samples_number = 4;
   inputs_number = 3;
   neurons_number = 4;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_binary_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   recurrent_layer->set(inputs_number, neurons_number);
   probabilistic_layer->set(neurons_number, outputs_number);

   neural_network.add_layer(recurrent_layer);
   neural_network.add_layer(probabilistic_layer);

   neural_network.set_parameters_random();

   recurrent_layer->set_timesteps(2);

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   neural_network.set();

   // Test recurrent and multiple probabilistic

   samples_number = 3;
   inputs_number = 3;
   neurons_number = 2;
   outputs_number = 3;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_binary_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   recurrent_layer->set(inputs_number, neurons_number);
   probabilistic_layer->set(neurons_number, outputs_number);

   neural_network.add_layer(recurrent_layer);
   neural_network.add_layer(probabilistic_layer);

   neural_network.set_parameters_random();

   recurrent_layer->set_timesteps(2);

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   // Test Perceptron LM

   samples_number = 2;
   inputs_number = 2;
   neurons_number = 3;
   outputs_number = 4;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();
   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});

   neural_network.set_parameters_random();

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   //       back_propagation_lm.set(samples_number, &mean_squared_error);
   //       mean_squared_error.back_propagate(batch,forward_propagation, back_propagation_lm);

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
   assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   neural_network.set();

   // Test probabilistic (binary) LM

   samples_number = 2;
   inputs_number = 2;
   neurons_number = 3;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_binary_random();

   data_set.set_training();

   batch.set(samples_number, &data_set);

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});

   neural_network.set_parameters_random();

   forward_propagation.set(samples_number, &neural_network);
   back_propagation.set(samples_number, &mean_squared_error);
   back_propagation_lm.set(samples_number, &mean_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   //       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
   assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   neural_network.set();

   // Test probabilistic (multiple) LM

   samples_number = 2;
   inputs_number = 2;
   neurons_number = 3;
   outputs_number = 3;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_binary_random();

   data_set.set_training();

   samples_indices = data_set.get_training_samples_indices();
   input_variables_indices = data_set.get_input_variables_indices();
   target_variables_indices = data_set.get_target_variables_indices();

   neural_network.set(NeuralNetwork::ProjectType::Classification, {inputs_number, neurons_number, outputs_number});

   neural_network.set_parameters_random();

   batch.set(samples_number, &data_set);
   batch.fill(samples_indices, input_variables_indices, target_variables_indices);

   forward_propagation.set(samples_number, &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   back_propagation.set(samples_number, &mean_squared_error);
   mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

   back_propagation_lm.set(samples_number, &mean_squared_error);
   //       mean_squared_error.back_propagate(batch, forward_propagation, back_propagation_lm);

   numerical_error_gradient = mean_squared_error.calculate_gradient_numerical_differentiation();

   assert_true(are_equal(back_propagation.gradient, back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
   assert_true(are_equal(back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);

   */
}


void MeanSquaredErrorTest::run_test_case()
{
   cout << "Running mean squared error test case...\n";

   test_constructor();

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
