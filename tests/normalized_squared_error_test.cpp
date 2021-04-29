//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques, S.L. (Artelnics)
//   artelnics@artelnics.com

#include "normalized_squared_error_test.h"

NormalizedSquaredErrorTest::NormalizedSquaredErrorTest() : UnitTesting()
{
    normalized_squared_error.set(&neural_network, &data_set);
}


NormalizedSquaredErrorTest::~NormalizedSquaredErrorTest()
{
}


void NormalizedSquaredErrorTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default

   NormalizedSquaredError normalized_squared_error_1;

   assert_true(!normalized_squared_error_1.has_neural_network(), LOG);
   assert_true(!normalized_squared_error_1.has_data_set(), LOG);

   // Neural network and data set

   NeuralNetwork neural_network_3;
   DataSet data_set_3;
   NormalizedSquaredError nse3(&neural_network_3, &data_set_3);

   assert_true(nse3.has_neural_network(), LOG);
   assert_true(nse3.has_data_set(), LOG);
}


void NormalizedSquaredErrorTest::test_destructor()
{
   cout << "test_destructor\n";
}


void NormalizedSquaredErrorTest::test_calculate_normalization_coefficient()
{
   cout << "test_calculate_normalization_coefficient\n";

   Index samples_number = 4;
   Index inputs_number = 4;
   Index outputs_number = 4;

   Tensor<type, 1> targets_mean(outputs_number);
   Tensor<type, 2> targets(samples_number, outputs_number);

   // Test

   data_set.generate_random_data(samples_number, inputs_number+outputs_number);

   Tensor<string, 1> uses(8);
   uses.setValues({"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"});

   data_set.set_columns_uses(uses);

   targets = data_set.get_target_data();

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, outputs_number});
   neural_network.set_parameters_random();

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();

   type normalization_coefficient = normalized_squared_error.calculate_normalization_coefficient(targets, targets_mean);

   assert_true(normalization_coefficient > 0, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_error()
{
   cout << "test_calculate_error\n";

   DataSet data_set;

   Tensor<type, 2> data;
   Index samples_number;

   Tensor<Index,1> batch_samples_indices;
   Tensor<Index,1> inputs_indices;
   Tensor<Index,1> targets_indices;

   Index batch_samples_number;

   DataSetBatch batch;

   Index inputs_number;
   Index hidden_neurons;
   Index outputs_number;
   
   Tensor<type, 1> parameters;

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation back_propagation;

   // Test

   data_set.set(1, 1, 1);

   neural_network.set(NeuralNetwork::Approximation, {1, 2});

   samples_number = 1;
   inputs_number = 1;
   outputs_number = 1;
   hidden_neurons = 1;

   data.resize(2, 2);
   data(0,0) = -1.0;
   data(0,1) = -1.0;
   data(1,0) = 1.0;
   data(1,1) = 1.0;

   data_set.set_data(data);
   data_set.set_training();

   batch.set(1, &data_set);

   batch_samples_indices = data_set.get_used_samples_indices();
   inputs_indices = data_set.get_input_variables_indices();
   targets_indices = data_set.get_target_variables_indices();

   batch.fill(batch_samples_indices, inputs_indices, targets_indices);
   batch_samples_number = batch.get_samples_number();

   forward_propagation.set(batch_samples_number, &neural_network);

   back_propagation.set(batch_samples_number, &normalized_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   normalized_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   // Test

   samples_number = 7;
   inputs_number = 8;
   outputs_number = 5;
   hidden_neurons = 3;

   neural_network.set(NeuralNetwork::Approximation, {inputs_number, hidden_neurons, outputs_number});
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();

   normalized_squared_error.set_normalization_coefficient();

//   assert_true(abs(normalized_squared_error.calculate_error() - normalized_squared_error.calculate_training_error(parameters)) < 1.0e-3, LOG);

}


/// @todo This test method does not work if the number of samples is equal to 1

void NormalizedSquaredErrorTest::test_calculate_error_gradient()
{
   cout << "test_calculate_error_gradient\n";

   DataSet data_set;
   Index samples_number;

   DataSetBatch batch;

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_indices;
   Tensor<Index, 1> target_indices;

   Index inputs_number;
   Index outputs_number;
   Index hidden_neurons;
   Index hidden_neurons_2;

   RecurrentLayer* recurrent_layer = new RecurrentLayer();

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

   PerceptronLayer* perceptron_layer_1 = new PerceptronLayer();
   PerceptronLayer* perceptron_layer_2 = new PerceptronLayer();

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

   NeuralNetworkForwardPropagation forward_propagation;

   LossIndexBackPropagation training_back_propagation;

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   // Trivial test
   {
       samples_number = 10;
       inputs_number = 1;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_constant(0.0);
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       perceptron_layer_1->set(inputs_number, outputs_number);
       neural_network.add_layer(perceptron_layer_1);

       neural_network.set_parameters_constant(0.0);

       normalized_squared_error.set_normalization_coefficient(1.0);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       training_back_propagation.set(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);
       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
       assert_true(all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(),
                          [](type i) { return (i-static_cast<type>(0))<numeric_limits<type>::min(); }), LOG);
   }

   neural_network.set();

   // Test perceptron

   {
       samples_number = 10;
       inputs_number = 3;
       outputs_number = 5;

       const Index neurons_number = 6;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       perceptron_layer_1->set(inputs_number, neurons_number);
       perceptron_layer_2->set(neurons_number, outputs_number);

       neural_network.add_layer(perceptron_layer_1);
       neural_network.add_layer(perceptron_layer_2);

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient(1.0);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       training_back_propagation.set(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);
       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test perceptron and binary probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       hidden_neurons = 4;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();
       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       training_back_propagation.set(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test perceptron and multiple probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       hidden_neurons = 2;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();
       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       forward_propagation.set(samples_number, &neural_network);
       training_back_propagation.set(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, training_back_propagation);

       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test lstm

   {
       samples_number = 4;
       inputs_number = 3;
       outputs_number = 2;
       hidden_neurons = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       long_short_term_memory_layer->set(inputs_number, hidden_neurons);

       neural_network.add_layer(long_short_term_memory_layer);

       neural_network.set_parameters_random();

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       normalized_squared_error.set_normalization_coefficient();

       long_short_term_memory_layer->set_timesteps(2);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent
   {
       samples_number = 4;
       inputs_number = 2;
       outputs_number = 1;
       hidden_neurons = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);

       neural_network.add_layer(recurrent_layer);

       neural_network.set_parameters_random();

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       normalized_squared_error.set_normalization_coefficient();

       recurrent_layer->set_timesteps(2);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and perceptron

   {
       samples_number = 4;
       inputs_number = 2;
       outputs_number = 1;
       hidden_neurons = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_random();

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);
       perceptron_layer_1->set(hidden_neurons, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(perceptron_layer_1);

       neural_network.set_parameters_random();

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       normalized_squared_error.set_normalization_coefficient();

       recurrent_layer->set_timesteps(2);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and binary probabilistic
   {
       samples_number = 4;
       inputs_number = 3;
       hidden_neurons = 4;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);
       probabilistic_layer->set(hidden_neurons, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(probabilistic_layer);

       neural_network.set_parameters_random();

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       normalized_squared_error.set_normalization_coefficient();

       recurrent_layer->set_timesteps(2);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test recurrent and multiple probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       hidden_neurons = 2;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       recurrent_layer->set(inputs_number, hidden_neurons);
       probabilistic_layer->set(hidden_neurons, outputs_number);

       neural_network.add_layer(recurrent_layer);
       neural_network.add_layer(probabilistic_layer);

       neural_network.set_parameters_random();

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       normalized_squared_error.set_normalization_coefficient();

       recurrent_layer->set_timesteps(2);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(error_gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test Perceptron LM
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons = 3;
       outputs_number = 4;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Approximation, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);
       LossIndexBackPropagationLM loss_index_back_propagation_lm(samples_number, &normalized_squared_error);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, loss_index_back_propagation_lm);

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, loss_index_back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(loss_index_back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test probabilistic (binary) LM

   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons = 3;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);
       LossIndexBackPropagationLM loss_index_back_propagation_lm(samples_number, &normalized_squared_error);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, loss_index_back_propagation_lm);

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, loss_index_back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(loss_index_back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   neural_network.set();

   // Test probabilistic (multiple) LM
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons = 3;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);
       LossIndexBackPropagationLM loss_index_back_propagation_lm(samples_number, &normalized_squared_error);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, loss_index_back_propagation_lm);

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();

       assert_true(are_equal(back_propagation.gradient, loss_index_back_propagation_lm.gradient, static_cast<type>(1.0e-3)), LOG);
       assert_true(are_equal(loss_index_back_propagation_lm.gradient, numerical_error_gradient, static_cast<type>(1.0e-3)), LOG);
   }

   // Test convolutional
   {
       // @todo

       /*
       neural_network.set();

       samples_number = 2;

       Index channels_number = 1;
       Index rows_number = 3;
       Index columns_number = 3;

       Index kernels_number = 2;
       Index kernels_rows_number = 2;
       Index kernels_columns_number = 2;

       inputs_number = channels_number*rows_number*columns_number;
       outputs_number = kernels_number*kernels_rows_number*kernels_columns_number;

       Tensor<Index, 1> input_variables_dimensions(4);
       input_variables_dimensions[0] = samples_number;
       input_variables_dimensions[1] = channels_number;
       input_variables_dimensions[2] = rows_number;
       input_variables_dimensions[3] = columns_number;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_input_variables_dimensions(input_variables_dimensions);
       data_set.set_data_constant(0.5);
       data_set.set_training();

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       DataSetBatch batch(samples_number, &data_set);
       batch.fill(samples_indices, input_indices, target_indices);

       Tensor<Index, 1> kernels_dimensions(4);
       kernels_dimensions(0) = kernels_number;
       kernels_dimensions(1) = channels_number;
       kernels_dimensions(2) = kernels_rows_number;
       kernels_dimensions(3) = kernels_columns_number;

       ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer(input_variables_dimensions, kernels_dimensions);
       convolutional_layer_1->set_parameters_constant(static_cast<type>(0.7));
       convolutional_layer_1->set_activation_function(ConvolutionalLayer::ActivationFunction::HyperbolicTangent);

       neural_network.add_layer(convolutional_layer_1);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
       normalized_squared_error.set_normalization_coefficient(1);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);

       neural_network.forward_propagate(batch, forward_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       numerical_error_gradient = normalized_squared_error.calculate_gradient_numerical_differentiation();
       */
   }
}



///@todo check for classification

void NormalizedSquaredErrorTest::test_calculate_squared_errors()
{
   cout << "test_calculate_squared_errors\n";

   Index samples_number;
   Index inputs_number;
   Index hidden_neurons_number;
   Index outputs_number;

   // Test

   samples_number = 7;
   inputs_number = 6;
   hidden_neurons_number = 5;
   outputs_number = 7;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();
   data_set.set_training();

   DataSetBatch batch(samples_number, &data_set);

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_indices, target_indices);

   neural_network.set(NeuralNetwork::Approximation, {inputs_number,hidden_neurons_number,outputs_number});
   neural_network.set_parameters_random();

   normalized_squared_error.set_normalization_coefficient();

   NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
   LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);
   LossIndexBackPropagationLM loss_index_back_propagation_lm(samples_number, &normalized_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   normalized_squared_error.calculate_squared_errors(batch, forward_propagation, loss_index_back_propagation_lm);
   normalized_squared_error.calculate_error(batch, forward_propagation, loss_index_back_propagation_lm);

   normalized_squared_error.calculate_errors(batch, forward_propagation, back_propagation);
   normalized_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(static_cast<type>(abs(loss_index_back_propagation_lm.error - back_propagation.error)) < static_cast<type>(1.0e-3), LOG);
}


void NormalizedSquaredErrorTest::test_calculate_squared_errors_jacobian()
{
   cout << "test_calculate_squared_errors_jacobian\n";

   Tensor<Index, 1> samples_indices;
   Tensor<Index, 1> input_indices;
   Tensor<Index, 1> target_indices;

   DataSetBatch batch;

   Index samples_number;
   Index inputs_number;
   Index hidden_neurons_number;
   Index outputs_number;

   // Test Perceptron
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 4;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Approximation, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);
       LossIndexBackPropagationLM loss_index_back_propagation_lm(samples_number, &normalized_squared_error);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, loss_index_back_propagation_lm);

       Tensor<type, 2> numerical_squared_errors_jacobian;

       numerical_squared_errors_jacobian = normalized_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(loss_index_back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, 1.0e-3), LOG);
   }

   // Test probabilistic (binary)

   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);
       LossIndexBackPropagationLM loss_index_back_propagation_lm(samples_number, &normalized_squared_error);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, loss_index_back_propagation_lm);

       Tensor<type, 2> numerical_squared_errors_jacobian;

       numerical_squared_errors_jacobian = normalized_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(loss_index_back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, static_cast<type>(1e-3)), LOG);
   }

   // Test probabilistic (multiple)
   {
       samples_number = 2;
       inputs_number = 2;
       hidden_neurons_number = 3;
       outputs_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       data_set.set_training();

       batch.set(samples_number, &data_set);

       samples_indices = data_set.get_training_samples_indices();
       input_indices = data_set.get_input_variables_indices();
       target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       neural_network.set(NeuralNetwork::Classification, {inputs_number, hidden_neurons_number, outputs_number});

       neural_network.set_parameters_random();

       normalized_squared_error.set_normalization_coefficient();

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation back_propagation(samples_number, &normalized_squared_error);
       LossIndexBackPropagationLM loss_index_back_propagation_lm(samples_number, &normalized_squared_error);

       normalized_squared_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       neural_network.forward_propagate(batch, forward_propagation);
       normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

       normalized_squared_error.back_propagate(batch, forward_propagation, loss_index_back_propagation_lm);

       Tensor<type, 2> numerical_squared_errors_jacobian;

       numerical_squared_errors_jacobian = normalized_squared_error.calculate_Jacobian_numerical_differentiation();

       assert_true(are_equal(loss_index_back_propagation_lm.squared_errors_jacobian, numerical_squared_errors_jacobian, static_cast<type>(1e-3)), LOG);
   }
}


void NormalizedSquaredErrorTest::test_to_XML()
{
   cout << "test_to_XML\n";
}


void NormalizedSquaredErrorTest::test_from_XML()
{
   cout << "test_from_XML\n";
}


void NormalizedSquaredErrorTest::run_test_case()
{
   cout << "Running normalized squared error test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();
   test_calculate_normalization_coefficient();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_error();

   test_calculate_error_gradient();

   // Squared errors methods

   test_calculate_squared_errors();

   test_calculate_squared_errors_jacobian();

   // Squared errors methods

   test_calculate_squared_errors();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   cout << "End of normalized squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lenser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lenser General Public License for more details.

// You should have received a copy of the GNU Lenser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
