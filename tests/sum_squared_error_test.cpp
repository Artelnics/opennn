//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   T E S T   C L A S S               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "sum_squared_error_test.h"

SumSquaredErrorTest::SumSquaredErrorTest() : UnitTesting() 
{
}


SumSquaredErrorTest::~SumSquaredErrorTest() 
{
}


void SumSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   SumSquaredError sum_squared_error_1;

   assert_true(!sum_squared_error_1.has_neural_network(), LOG);
   assert_true(!sum_squared_error_1.has_data_set(), LOG);

   // Neural network and data set

   NeuralNetwork neural_network_2;
   DataSet data_set;

   NeuralNetwork neural_network_3;
   SumSquaredError sum_squared_error_4(&neural_network_2, &data_set);

   assert_true(sum_squared_error_4.has_neural_network(), LOG);
   assert_true(sum_squared_error_4.has_data_set(), LOG);
}


void SumSquaredErrorTest::test_destructor()
{
   cout << "test_destructor\n";
}


void SumSquaredErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   DataSet data_set;
   Tensor<type, 2> data;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> inputs_indices;
   Tensor<Index,1> targets_indices;

   DataSetBatch batch;

   NeuralNetwork neural_network;

   Index inputs_number;
   Index target_number;

   Tensor<Index, 1> architecture;
   Tensor<type, 1> parameters;

   NeuralNetworkForwardPropagation forward_propagation;

   SumSquaredError sum_squared_error(&neural_network, &data_set);
   sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

   LossIndexBackPropagation training_back_propagation;

   // Test 0

   //Dataset

   data_set.set(1, 2, 2);
   data_set.initialize_data(0.0);
   data_set.set_training();

   batch.set(1, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

   // Neural network

   inputs_number = 2;
   target_number = 2;

   architecture.resize(2);
   architecture.setValues({inputs_number,target_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, training_back_propagation);

   assert_true(training_back_propagation.error == 0.0, LOG);

   // Test 1

   //Dataset

   data_set.set(1, 2, 2);
   data_set.initialize_data(1.0);
   data_set.set_training();

   batch.set(1, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

   // Neural network

   neural_network.set_parameters_constant(0.0);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, training_back_propagation);

   assert_true(training_back_propagation.error == 2.0, LOG);

   // Test 2

   //Dataset

   data_set.set(1, 1, 1);
   data_set.initialize_data(0.0);
   data_set.set_training();

   batch.set(1, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

   // Neural network

   neural_network.set();

   RecurrentLayer* recurrent_layer = new RecurrentLayer(1, 1);
   neural_network.add_layer(recurrent_layer);

   PerceptronLayer* perceptron_layer = new PerceptronLayer(1,1);
   neural_network.add_layer(perceptron_layer);

   neural_network.set_parameters_constant(0.0);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, training_back_propagation);

   assert_true(training_back_propagation.error == 0.0, LOG);

   // Test 3

   //Dataset

   data.resize(9,5);
   data.setValues({{-1,-1,1,-1,3},
                       {-1,0,1,0,2},
                       {-1,1,1,1,2},
                       {0,-1,1,0,2},
                       {0,0,1,1,1},
                       {0,1,1,2,2},
                       {1,-2,1,1,3},
                       {1,0,1,2,2},
                       {1,1,1,3,3}});
   data_set.set(data);
   data_set.set(9, 3, 2);
   data_set.set_training();

   batch.set(9, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

    // Neural network

   neural_network.set();

   architecture.setValues({3,1,2});
   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_constant(1.0);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_error(batch, forward_propagation, training_back_propagation);

   assert_true(training_back_propagation.error - 8.241 < 1e-3, LOG);
}


void SumSquaredErrorTest::test_calculate_output_delta()
{
   cout << "test_calculate_output_delta\n";

   DataSet data_set;
   Tensor<type, 2> data;

   Tensor<Index,1> training_samples_indices;
   Tensor<Index,1> inputs_indices;
   Tensor<Index,1> targets_indices;

   DataSetBatch batch;

   NeuralNetwork neural_network;

   Tensor<type, 1> parameters;

   Index inputs_number;
   Index target_number;
   Tensor<Index, 1> architecture;

   NeuralNetworkForwardPropagation forward_propagation;

   SumSquaredError sum_squared_error(&neural_network, &data_set);
   sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

   LossIndexBackPropagation training_back_propagation;

   Tensor<type, 1> numerical_gradient;

   // Test 0

   //Dataset

   data_set.set(1, 2, 2);
   data_set.initialize_data(0.0);
   data_set.set_training();

   batch.set(1, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

        // Neural network

   inputs_number = 2;
   target_number = 2;
   architecture.resize(2);
   architecture.setValues({inputs_number,target_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_constant(0.0);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, training_back_propagation);

//   assert_true(training_back_propagation.output_delta(0) == 0.0, LOG);
//   assert_true(training_back_propagation.output_delta(1) == 0.0, LOG);

   // Test 1

   //Dataset

   data_set.set(1, 2, 2);
   data_set.initialize_data(1.0);
   data_set.set_training();

   batch.set(1, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

   // Neural network

   neural_network.set_parameters_constant(0.0);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, training_back_propagation);

   numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//   assert_true(abs(training_back_propagation_1.output_delta(0)-numerical_gradient(4)) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(training_back_propagation_1.output_delta(1)-numerical_gradient(5)) < static_cast<type>(1e-3), LOG);

   // Test 2_1 / Perceptron

   //Dataset

   Index samples_number = 3;
   inputs_number = 1;
   Index hidden_neurons = 1;
   Index outputs_number = 2;

   data.resize(samples_number,inputs_number+outputs_number);
   data.setValues({{-1,-1,3},{-1,0,2},{-1,1,2},});

   data_set.set(data);
   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_training();

   batch.set(3, &data_set);

   training_samples_indices = data_set.get_training_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

   // Neural network

   neural_network.set();

   PerceptronLayer* perceptron_layer = new PerceptronLayer(hidden_neurons,outputs_number);
   neural_network.add_layer(perceptron_layer);

   parameters.resize(16);
   parameters.setValues({1,1,2,0, 1,2,1,1, 1,2,1,0, 1,1,2,1});
   neural_network.set_parameters(parameters);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, training_back_propagation);

   numerical_gradient.resize(neural_network.get_parameters_number());
   numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//   assert_true(abs(training_back_propagation_2.output_delta(0,1) + static_cast<type>(4.476)) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(training_back_propagation_2.output_delta(1,0) + static_cast<type>(1.523)) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(training_back_propagation_2.output_delta(2,1) + static_cast<type>(2.476)) < static_cast<type>(1e-3), LOG);

   // Test 2_2 / Recurrent

        // Neural network

   neural_network.set();

   RecurrentLayer* recurrent_layer = new RecurrentLayer(inputs_number, outputs_number);
   recurrent_layer->initialize_hidden_states(0.0);
   recurrent_layer->set_timesteps(10);
   neural_network.add_layer(recurrent_layer);

   neural_network.set_parameters(parameters);

   forward_propagation.set(data_set.get_training_samples_number(), &neural_network);
   training_back_propagation.set(data_set.get_training_samples_number(), &sum_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   sum_squared_error.calculate_output_delta(batch, forward_propagation, training_back_propagation);

//   assert_true(abs(training_back_propagation_2_2.output_delta(0,1) + 6) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(training_back_propagation_2_2.output_delta(1,0) + 0) < static_cast<type>(1e-3), LOG);
//   assert_true(abs(training_back_propagation_2_2.output_delta(2,1) + 4) < static_cast<type>(1e-3), LOG);

}


void SumSquaredErrorTest::test_calculate_Jacobian_gradient()
{
   cout << "test_calculate_Jacobian_gradient\n";

   NeuralNetwork neural_network;

   Tensor<type, 1> parameters;

   DataSet data_set;
   Tensor<type, 2> data;

   SumSquaredError sum_squared_error(&neural_network, &data_set);
   sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

   // Test 0

        //Dataset

   Index inputs_number = 2;
   Index target_number = 3;

   data_set.set(1, inputs_number, target_number);
   data_set.initialize_data(0.0);
   data_set.set_training();

   DataSetBatch batch(1, &data_set);

   Tensor<Index,1> training_samples_indices = data_set.get_training_samples_indices();
   Tensor<Index,1> inputs_indices = data_set.get_input_variables_indices();
   Tensor<Index,1> targets_indices = data_set.get_target_variables_indices();

   batch.fill(training_samples_indices, inputs_indices, targets_indices);

        // Neural network

   Tensor<Index, 1> architecture(2);
   architecture.setValues({inputs_number,target_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   NeuralNetworkForwardPropagation forward_propagation(data_set.get_training_samples_number(), &neural_network);
   neural_network.forward_propagate(batch, forward_propagation);

   LossIndexBackPropagation training_back_propagation(data_set.get_training_samples_number(), &sum_squared_error);
   sum_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

   LossIndexBackPropagationLM loss_index_back_propagation_lm(training_samples_indices.size(), &sum_squared_error);
//   sum_squared_error.calculate_squared_errors_jacobian(batch, forward_propagation, training_back_propagation, loss_index_back_propagation_lm);
//   sum_squared_error.calculate_gradient(batch, forward_propagation, loss_index_back_propagation_lm);

   assert_true(loss_index_back_propagation_lm.gradient(0) == 0.0, LOG);
   assert_true(loss_index_back_propagation_lm.gradient(1) == 0.0, LOG);
}


void SumSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   DataSet data_set;
   NeuralNetwork neural_network;
   SumSquaredError sum_squared_error(&neural_network, &data_set);

   Tensor<Index, 1> architecture;

   Tensor<type, 1> parameters;
   Tensor<type, 1> gradient;
   Tensor<type, 1> numerical_gradient;
   Tensor<type, 1> error;

   Index inputs_number;
   Index outputs_number;
   Index samples_number;
   Index hidden_neurons;

   // Test lstm
{
    samples_number = 10;
    inputs_number = 3;
    hidden_neurons = 2;
    outputs_number = 4;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    neural_network.set();

    LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer(inputs_number, hidden_neurons);

    long_short_term_memory_layer->set_timesteps(8);

    neural_network.add_layer(long_short_term_memory_layer);

    PerceptronLayer* perceptron_layer = new PerceptronLayer(hidden_neurons,outputs_number);

    neural_network.add_layer(perceptron_layer);

    neural_network.set_parameters_random();

    parameters = neural_network.get_parameters();

    numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//    gradient = sum_squared_error.calculate_error_gradient();

//    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);
}

   neural_network.set();

   // Test recurrent
{
//    samples_number = 5;
//    inputs_number = 3;
//    hidden_neurons = 7;
//    outputs_number = 3;

//    data_set.set(samples_number, inputs_number, outputs_number);

//    data_set.set_data_random();

//    data_set.set_training();

//    neural_network.set();

//    RecurrentLayer* recurrent_layer = new RecurrentLayer(inputs_number, hidden_neurons);

//    recurrent_layer->initialize_hidden_states(0.0);
//    recurrent_layer->set_timesteps(10);

//    neural_network.add_layer(recurrent_layer);

//    PerceptronLayer* perceptron_layer = new PerceptronLayer(hidden_neurons,outputs_number);

//    neural_network.add_layer(perceptron_layer);

//    neural_network.set_parameters_random();

//    parameters = neural_network.get_parameters();

//    numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//    gradient = sum_squared_error.calculate_error_gradient();

//    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);
}

   // Test perceptron

   neural_network.set();
{
    samples_number = 5;
    inputs_number = 2;
    hidden_neurons = 7;
    outputs_number = 4;

    data_set.set(samples_number, inputs_number, outputs_number);

    data_set.set_data_random();

    data_set.set_training();

    PerceptronLayer* perceptron_layer_1 = new PerceptronLayer(inputs_number, hidden_neurons);

    neural_network.add_layer(perceptron_layer_1);

    PerceptronLayer* perceptron_layer_2 = new PerceptronLayer(hidden_neurons, outputs_number);

    neural_network.add_layer(perceptron_layer_2);

//    numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//    gradient = sum_squared_error.calculate_error_gradient();

//    assert_true(numerical_gradient-gradient < 1.0e-3, LOG);
}

   // Test convolutional
{
   samples_number = 5;
   inputs_number = 147;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);
//   data_set.set_input_variables_dimensions(Tensor<Index, 1>({3,7,7}));
//   data_set.set_target_variables_dimensions(Tensor<Index, 1>({1}));
   data_set.set_data_random();
   data_set.set_training();

   const type parameters_minimum = -100.0;
   const type parameters_maximum = 100.0;

//   ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer({3,7,7}, {2,2,2});
//   Tensor<type, 2> filters_1({2,3,2,2}, 0);
//   filters_1.setRandom(parameters_minimum,parameters_maximum);
//   convolutional_layer_1->set_synaptic_weights(filters_1);
//   Tensor<type, 1> biases_1(2, 0);
//   biases_1.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_1->set_biases(biases_1);

//   ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer_1->get_outputs_dimensions(), {2,2,2});
//   convolutional_layer_2->set_padding_option(OpenNN::ConvolutionalLayer::Same);
//   Tensor<type, 2> filters_2({2,2,2,2}, 0);
//   filters_2.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_2->set_synaptic_weights(filters_2);
//   Tensor<type, 1> biases_2(2, 0);
//   biases_2.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_2->set_biases(biases_2);

//   PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_2->get_outputs_dimensions(), {2,2});

//   ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_1->get_outputs_dimensions(), {1,2,2});
//   convolutional_layer_3->set_padding_option(OpenNN::ConvolutionalLayer::Same);
//   Tensor<type, 2> filters_3({1,2,2,2}, 0);
//   filters_3.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_3->set_synaptic_weights(filters_3);
//   Tensor<type, 1> biases_3(1, 0);
//   biases_3.setRandom(parameters_minimum, parameters_maximum);
//   convolutional_layer_3->set_biases(biases_3);

//   PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_3->get_outputs_dimensions(), {2,2});
//   pooling_layer_2->set_pooling_method(PoolingLayer::MaxPooling);

//   PoolingLayer* pooling_layer_3 = new PoolingLayer(pooling_layer_2->get_outputs_dimensions(), {2,2});
//   pooling_layer_3->set_pooling_method(PoolingLayer::MaxPooling);

//   PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3->get_outputs_dimensions().calculate_product(), 3, OpenNN::PerceptronLayer::ActivationFunction::Linear);
//   perceptron_layer->set_parameters_random(parameters_minimum, parameters_maximum);

//   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), outputs_number);
//   probabilistic_layer->set_parameters_random(parameters_minimum, parameters_maximum);

//   neural_network.set();
//   neural_network.add_layer(convolutional_layer_1);
//   neural_network.add_layer(convolutional_layer_2);
//   neural_network.add_layer(pooling_layer_1);
//   neural_network.add_layer(convolutional_layer_3);
//   neural_network.add_layer(pooling_layer_2);
//   neural_network.add_layer(pooling_layer_3);
//   neural_network.add_layer(perceptron_layer);
//   neural_network.add_layer(probabilistic_layer);

//   numerical_gradient = sum_squared_error.calculate_gradient_numerical_differentiation();

//   gradient = sum_squared_error.calculate_error_gradient();

//   assert_true(absolute_value(numerical_gradient - gradient) < 1e-3, LOG);
}
}


void SumSquaredErrorTest::test_calculate_squared_errors()
{
   cout << "test_calculate_squared_errors\n";
}


void SumSquaredErrorTest::test_calculate_squared_errors_jacobian()
{   
   cout << "test_calculate_squared_errors_jacobian\n";

//   NumericalDifferentiation nd;

//   NeuralNetwork neural_network;
//   Tensor<Index, 1> architecture;
//   Tensor<type, 1> parameters;

//   DataSet data_set;

//   SumSquaredError sum_squared_error(&neural_network, &data_set);

//   Tensor<type, 1> gradient;

//   Tensor<type, 1> terms;
//   Tensor<type, 2> terms_Jacobian;
//   Tensor<type, 2> numerical_squared_errors_jacobian;

//   Tensor<Index, 1> samples;

//   Tensor<type, 2> inputs;
//   Tensor<type, 2> targets;

//   Tensor<type, 2> outputs;
//   Tensor<type, 2> output_delta;

//   Tensor<Tensor<type, 2>, 1> layers_activations;

//   Tensor<Tensor<type, 2>, 1> layers_activations_derivatives;

//   Tensor<Tensor<type, 2>, 1> layers_delta;

   // Test

//   architecture.setValues({1, 1, 1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);

//   neural_network.set_parameters_constant(0.0);

//   data_set.set(1, 1, 1);

//   data_set.initialize_data(0.0);

//   samples.set(1,0);
   //samples.initialize_sequential();

//   inputs = data_set.get_input_data(samples);
//   targets = data_set.get_target_data(samples);

//   outputs = neural_network.calculate_outputs(inputs);
//   output_delta = sum_squared_error.calculate_output_delta(outputs, targets);

//   Tensor<LayerForwardPropagation, 1> forward_propagation = neural_network.forward_propagate(inputs);

//   layers_delta = sum_squared_error.calculate_layers_delta(forward_propagation, output_delta);

//   terms_Jacobian = sum_squared_error.calculate_squared_errors_jacobian(inputs, forward_propagation, layers_delta);

//   assert_true(terms_Jacobian.dimension(0) == data_set.get_samples_number(), LOG);
//   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test 

//   architecture.setValues({3, 4, 2});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_constant(0.0);

//   data_set.set(3, 2, 5);
//   sum_squared_error.set(&neural_network, &data_set);
//   data_set.initialize_data(0.0);

//   terms_Jacobian = sum_squared_error.calculate_squared_errors_jacobian();

//   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
//   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

//   architecture.resize(3);
//   architecture[0] = 5;
//   architecture[1] = 1;
//   architecture[2] = 2;

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_constant(0.0);

//   data_set.set(5, 2, 3);
//   sum_squared_error.set(&neural_network, &data_set);
//   data_set.initialize_data(0.0);

//   terms_Jacobian = sum_squared_error.calculate_squared_errors_jacobian();

//   assert_true(terms_Jacobian.dimension(0) == data_set.get_training_samples_number(), LOG);
//   assert_true(terms_Jacobian.dimension(1) == neural_network.get_parameters_number(), LOG);
//   assert_true(terms_Jacobian == 0.0, LOG);

   // Test

//   architecture.setValues({1, 1, 1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();
//   parameters = neural_network.get_parameters();

//   data_set.set(1, 1, 1);
//   data_set.set_data_random();

//   terms_Jacobian = sum_squared_error.calculate_squared_errors_jacobian();
//   numerical_squared_errors_jacobian = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_training_terms, parameters);

//   assert_true(absolute_value(terms_Jacobian-numerical_squared_errors_jacobian) < 1.0e-3, LOG);

   // Test

//   architecture.setValues({2, 2, 2});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();
//   parameters = neural_network.get_parameters();

//   data_set.set(2, 2, 2);
//   data_set.set_data_random();

//   terms_Jacobian = sum_squared_error.calculate_squared_errors_jacobian();
//   numerical_squared_errors_jacobian = nd.calculate_Jacobian(sse, &SumSquaredError::calculate_training_terms, parameters);

//   assert_true(absolute_value(terms_Jacobian-numerical_squared_errors_jacobian) < 1.0e-3, LOG);

   // Test

//   architecture.setValues({2, 2, 2});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();

//   data_set.set(2, 2, 2);
//   data_set.set_data_random();

//   gradient = sum_squared_error.calculate_gradient();

//   terms = sum_squared_error.calculate_training_error_terms();
//   terms_Jacobian = sum_squared_error.calculate_squared_errors_jacobian();

//   assert_true(absolute_value((terms_Jacobian.calculate_transpose()).dot(terms)*2.0 - gradient) < 1.0e-3, LOG);

}


void SumSquaredErrorTest::test_to_XML()   
{
    cout << "test_to_XML\n";

    SumSquaredError sum_squared_error;

//    tinyxml2::XMLDocument* document;

    // Test

//    document = sum_squared_error.to_XML();

//    assert_true(document != nullptr, LOG);

//    delete document;
}


void SumSquaredErrorTest::test_from_XML()
{
    cout << "test_from_XML\n";

    SumSquaredError sum_squared_error1;
    SumSquaredError sum_squared_error2;

   tinyxml2::XMLDocument* document;

   // Test

//   sum_squared_error1.set_display(false);

//   document = sum_squared_error1.to_XML();

//   sum_squared_error2.from_XML(*document);

//   delete document;

//   assert_true(!sum_squared_error2.get_display(), LOG);
}


void SumSquaredErrorTest::run_test_case()
{
   cout << "Running sum squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();


   // Get methods

   // Set methods

   // Error methods

   test_calculate_error();

   test_calculate_output_delta();

   test_calculate_error_gradient();

   test_calculate_Jacobian_gradient();

   // Squared errors methods

   test_calculate_squared_errors();

   test_calculate_squared_errors_jacobian();

   //Serialization methods

    test_to_XML();

    test_from_XML();

   cout << "End of sum squared error test case.\n\n";
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
