//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques, S.L. (Artelnics)
//   artelnics@artelnics.com

#include "normalized_squared_error_test.h"
#include <omp.h>

NormalizedSquaredErrorTest::NormalizedSquaredErrorTest(void) : UnitTesting()
{
}


NormalizedSquaredErrorTest::~NormalizedSquaredErrorTest(void)
{
}

void NormalizedSquaredErrorTest::test_constructor(void) // @todo
{
   cout << "test_constructor\n";

   // Default

   NormalizedSquaredError normalized_squared_error_1;

   assert_true(normalized_squared_error_1.has_neural_network() == false, LOG);
   assert_true(normalized_squared_error_1.has_data_set() == false, LOG);

   // Neural network and data set

   NeuralNetwork neural_network_3;
   DataSet data_set_3;
   NormalizedSquaredError nse3(&neural_network_3, &data_set_3);

   assert_true(nse3.has_neural_network() == true, LOG);
   assert_true(nse3.has_data_set() == true, LOG);
}


void NormalizedSquaredErrorTest::test_destructor(void) // @todo
{
   cout << "test_destructor\n";
}

void NormalizedSquaredErrorTest::test_calculate_normalization_coefficient(void) // @todo
{
   cout << "test_calculate_normalization_coefficient\n";

   NeuralNetwork neural_network;
   DataSet data_set;
   NormalizedSquaredError nse(&neural_network, &data_set);

   Index samples_number = 4;
   Index inputs_number = 4;
   Index outputs_number = 4;    //targets_number or means_number

   Tensor<type, 1> targets_mean(outputs_number);
   Tensor<type, 2> targets(samples_number, outputs_number);

   // Test

   data_set.generate_random_data(samples_number, inputs_number+outputs_number);

   Tensor<string, 1> uses(8);
   uses.setValues({"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"});

   data_set.set_columns_uses(uses);

   targets = data_set.get_target_data();
   //targets_mean = data_set.calculate_training_targets_mean();

   Tensor<Index, 1> architecture(2);
   architecture.setValues({inputs_number, outputs_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

//   data_set.set(samples_number, inputs_number, outputs_number);
//   data_set.set_data_random();

   type normalization_coefficient = nse.calculate_normalization_coefficient(targets, targets_mean);

   assert_true(normalization_coefficient > 0, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_error(void) // @todo
{
   cout << "test_calculate_error\n";

   Tensor<Index, 1> architecture(2);
   architecture.setValues({1, 2});
   Tensor<type, 1> parameters;

   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);

   DataSet data_set(1, 1, 1);

   Index samples_number;
   samples_number = 1;
   Index inputs_number;
   inputs_number = 1;
   Index outputs_number;
   outputs_number = 1;
   Index hidden_neurons;
   hidden_neurons = 1;

   Tensor<type, 2> new_data(2, 2);
   new_data(0,0) = -1.0;
   new_data(0,1) = -1.0;
   new_data(1,0) = 1.0;
   new_data(1,1) = 1.0;

   data_set.set_data(new_data);
   data_set.set_training();

   NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);
   DataSet::Batch batch(1, &data_set);

   Tensor<Index,1> batch_samples_indices = data_set.get_used_samples_indices();
   Tensor<Index,1> inputs_indices = data_set.get_input_variables_indices();
   Tensor<Index,1> targets_indices = data_set.get_target_variables_indices();

   batch.fill(batch_samples_indices, inputs_indices, targets_indices);
   Index batch_samples_number = batch.get_samples_number();

   NeuralNetwork::ForwardPropagation forward_propagation(batch_samples_number, &neural_network);

   LossIndex::BackPropagation back_propagation(batch_samples_number, &normalized_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   normalized_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   // Test

//   samples_number = 7;
//   inputs_number = 8;
//   outputs_number = 5;
//   hidden_neurons = 3;

//   architecture.setValues({inputs_number, hidden_neurons, outputs_number});

//   neural_network.set(NeuralNetwork::Approximation, architecture);
//   neural_network.set_parameters_random();

//   parameters = neural_network.get_parameters();

//   data_set.set(samples_number, inputs_number, outputs_number);
//   data_set.set_data_random();

//   normalized_squared_error.set_normalization_coefficient();

//   assert_true(abs(normalized_squared_error.calculate_error() - normalized_squared_error.calculate_training_error(parameters)) < 1.0e-3, LOG);

}


void NormalizedSquaredErrorTest::test_calculate_error_gradient(void) // @todo
{
   cout << "test_calculate_error_gradient\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

   Tensor<type, 1> error_gradient;
   Tensor<type, 1> numerical_error_gradient;

   Index samples_number;
   Index inputs_number;
   Index outputs_number;
   Index hidden_neurons;

   ScalingLayer* scaling_layer = new ScalingLayer();

   RecurrentLayer* recurrent_layer = new RecurrentLayer();

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

   PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer();
   PerceptronLayer* output_perceptron_layer = new PerceptronLayer();

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

   // Test trivial
{
   samples_number = 10;
   inputs_number = 1;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.initialize_data(0.0);
   data_set.set_training();

   DataSet::Batch batch(samples_number, &data_set);

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_indices, target_indices);

   hidden_perceptron_layer->set(inputs_number, outputs_number);
   neural_network.add_layer(hidden_perceptron_layer);

   neural_network.set_parameters_constant(0.0);

   nse.set_normalization_coefficient(1.0);

   nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

   NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);
   LossIndex::BackPropagation training_back_propagation(samples_number, &nse);

   neural_network.forward_propagate(batch, forward_propagation);

   nse.back_propagate(batch, forward_propagation, training_back_propagation);
   error_gradient = training_back_propagation.gradient;

   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation(&nse);

   assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
   assert_true(std::all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(), [](type i) { return (i-static_cast<type>(0))<std::numeric_limits<type>::min(); }), LOG);
}

   neural_network.set();
/*
   // Test perceptron and probabilistic
{

   samples_number = 3;
   inputs_number = 1;
   hidden_neurons = 1;
   outputs_number = 3;

   data_set.set(samples_number, inputs_number, outputs_number);

//   data_set.set_data_random();

   Tensor<type,2> data(3,4);

   data(0,0) = static_cast<type>(0.2);
   data(0,1) = 1;
   data(0,2) = 0;
   data(0,3) = 0;
   data(1,0) = static_cast<type>(0.3);
   data(1,1) = 0;
   data(1,2) = 1;
   data(1,3) = 0;
   data(2,0) = static_cast<type>(0.5);
   data(2,1) = 0;
   data(2,2) = 0;
   data(2,3) = 1;

   data_set.set_data(data);

   Tensor<string, 1> columns_uses(4);
   columns_uses(0) = "Input";
   columns_uses(1) = "Target";
   columns_uses(2) = "Target";
   columns_uses(3) = "Target";

   data_set.set_columns_uses(columns_uses);

   data_set.set_training();

   DataSet::Batch batch(samples_number, &data_set);

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_indices, target_indices);

   hidden_perceptron_layer->set(inputs_number, hidden_neurons);
   output_perceptron_layer->set(hidden_neurons, outputs_number);
   probabilistic_layer->set(outputs_number, outputs_number);

   neural_network.add_layer(hidden_perceptron_layer);
   neural_network.add_layer(output_perceptron_layer);
  neural_network.add_layer(probabilistic_layer);

   Tensor<type,2> perceptron_weights(1,3);
   perceptron_weights(0,0) = static_cast<type>(-0.318137);
   perceptron_weights(0,1) = static_cast<type>(-0.0745666);
   perceptron_weights(0,2) = static_cast<type>(-0.0732468);

   Tensor<type,2> perceptron_biases(1,3);
   perceptron_biases(0,0) = static_cast<type>(1.88443);
   perceptron_biases(0,1) = static_cast<type>(0.776795);
   perceptron_biases(0,2) = static_cast<type>(0.0126074);

   Tensor<type,2> probabilistic_weights(3,3);
   probabilistic_weights(0,0) = static_cast<type>(-0.290916);
   probabilistic_weights(0,1) = static_cast<type>(2.1764);
   probabilistic_weights(0,2) = static_cast<type>(-1.71237);
   probabilistic_weights(1,0) = static_cast<type>(-0.147688);
   probabilistic_weights(1,1) = static_cast<type>(1.71663);
   probabilistic_weights(1,2) = static_cast<type>(0.349156);
   probabilistic_weights(2,0) = static_cast<type>(-0.302181);
   probabilistic_weights(2,1) = static_cast<type>(1.18804);
   probabilistic_weights(2,2) = static_cast<type>(0.754033);

   Tensor<type,2> probabilistic_biases(1,3);
   probabilistic_biases(0,0) = static_cast<type>(1.95245);
   probabilistic_biases(0,1) = static_cast<type>(0.68821);
   probabilistic_biases(0,2) = static_cast<type>(1.75451);

//   neural_network.set_parameters_random();

   hidden_perceptron_layer->set_synaptic_weights(perceptron_weights);
   hidden_perceptron_layer->set_biases(perceptron_biases);

   probabilistic_layer->set_synaptic_weights(probabilistic_weights);
   probabilistic_layer->set_biases(probabilistic_biases);

   nse.set_normalization_coefficient();
   nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

   NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);
   LossIndex::BackPropagation training_back_propagation(samples_number, &nse);

   neural_network.forward_propagate(batch, forward_propagation);

   nse.back_propagate(batch, forward_propagation, training_back_propagation);

   error_gradient = training_back_propagation.gradient;

   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation(&nse);

   const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

   assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
}
*/
   neural_network.set();

   // Test lstm
/*
{
   samples_number = 4;
   inputs_number = 2;
   outputs_number = 3;
   hidden_neurons = 4;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

   DataSet::Batch batch(samples_number, &data_set);

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_indices, target_indices);

   long_short_term_memory_layer->set(inputs_number, hidden_neurons);

   neural_network.add_layer(long_short_term_memory_layer);

   neural_network.set_parameters_random();

   nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

   nse.set_normalization_coefficient();

   long_short_term_memory_layer->set_timesteps(2);

   NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);

   LossIndex::BackPropagation back_propagation(samples_number, &nse);

   neural_network.forward_propagate(batch, forward_propagation);

   nse.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation(&nse);

   const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

   assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
}
*/
   neural_network.set();
/*
   // Test recurrent
{
   samples_number = 4;
   inputs_number = 2;
   outputs_number = 3;
   hidden_neurons = 4;

   data_set.set(samples_number, inputs_number, outputs_number);

   data_set.set_data_random();

   data_set.set_training();

   DataSet::Batch batch(samples_number, &data_set);

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_indices, target_indices);

   recurrent_layer->set(inputs_number, hidden_neurons);

   neural_network.add_layer(recurrent_layer);

   neural_network.set_parameters_random();

   nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

   nse.set_normalization_coefficient();

   recurrent_layer->set_timesteps(2);

   NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);

   LossIndex::BackPropagation back_propagation(samples_number, &nse);

   neural_network.forward_propagate(batch, forward_propagation);

   nse.back_propagate(batch, forward_propagation, back_propagation);

   error_gradient = back_propagation.gradient;

   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation(&nse);

   const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

   assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
}
*/
   // Test convolutional
{
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
//   data_set.set_data_random();
   data_set.initialize_data(0.5);
   data_set.set_training();

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   DataSet::Batch batch(samples_number, &data_set);
   batch.fill(samples_indices, input_indices, target_indices);

   cout << "Inputs4d: " << batch.inputs_4d << endl;

   Tensor<Index, 1> kernels_dimensions(4);
   kernels_dimensions(0) = kernels_number;
   kernels_dimensions(1) = channels_number;
   kernels_dimensions(2) = kernels_rows_number;
   kernels_dimensions(3) = kernels_columns_number;

   ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer(input_variables_dimensions, kernels_dimensions);
   convolutional_layer_1->set_parameters_constant(static_cast<type>(0.7));
   convolutional_layer_1->set_activation_function(ConvolutionalLayer::ActivationFunction::HyperbolicTangent);


   neural_network.add_layer(convolutional_layer_1);

   nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
   nse.set_normalization_coefficient(1);

   NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);

   LossIndex::BackPropagation back_propagation(samples_number, &nse);

   neural_network.forward_propagate(batch, forward_propagation);

   nse.back_propagate(batch, forward_propagation, back_propagation);



//   cout << "Combinations4d: " << forward_propagation.layers(0).combinations_4d << endl;
//   cout << "Combinations2d: " << forward_propagation.layers(0).combinations_2d << endl;

//   cout << "Activations4d:  " << forward_propagation.layers(0).activations_4d << endl;
//   cout << "Activations2d:  " << forward_propagation.layers(0).activations_2d << endl;

//   cout << "ActivationsDerivatives4d:  " << forward_propagation.layers(0).activations_derivatives_4d << endl;
//   cout << "ActivationsDerivatives2d:  " << forward_propagation.layers(0).activations_derivatives_2d << endl;

//   cout << "Error: " << back_propagation.error << endl;

   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation(&nse);

   cout << "numerical error gradient: " << numerical_error_gradient << endl;






   /*
   ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer_1->get_outputs_dimensions(), {2,2,2});
   convolutional_layer_2->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<type, 2> filters_2({2,2,2,2}, 0);
   filters_2.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_synaptic_weights(filters_2);
   Tensor<type, 1> biases_2(2, 0);
   biases_2.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_2->set_biases(biases_2);

   PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_2->get_outputs_dimensions(), {2,2});

   ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_1->get_outputs_dimensions(), {1,2,2});
   convolutional_layer_3->set_padding_option(OpenNN::ConvolutionalLayer::Same);
   Tensor<type, 2> filters_3({1,2,2,2}, 0);
   filters_3.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_3->set_synaptic_weights(filters_3);
   Tensor<type, 1> biases_3(1, 0);
   biases_3.setRandom(parameters_minimum, parameters_maximum);
   convolutional_layer_3->set_biases(biases_3);

   PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_3->get_outputs_dimensions(), {2,2});
   pooling_layer_2->set_pooling_method(PoolingLayer::MaxPooling);

   PoolingLayer* pooling_layer_3 = new PoolingLayer(pooling_layer_2->get_outputs_dimensions(), {2,2});
   pooling_layer_3->set_pooling_method(PoolingLayer::MaxPooling);

   PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3->get_outputs_dimensions().calculate_product(), 3, OpenNN::PerceptronLayer::ActivationFunction::Linear);
   perceptron_layer->set_parameters_random(parameters_minimum, parameters_maximum);

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), outputs_number);
   probabilistic_layer->set_parameters_random(parameters_minimum, parameters_maximum);

   neural_network.set();
   neural_network.add_layer(convolutional_layer_1);
   neural_network.add_layer(convolutional_layer_2);
   neural_network.add_layer(pooling_layer_1);
   neural_network.add_layer(convolutional_layer_3);
   neural_network.add_layer(pooling_layer_2);
   neural_network.add_layer(pooling_layer_3);
   neural_network.add_layer(perceptron_layer);
   neural_network.add_layer(probabilistic_layer);

   numerical_error_gradient = nse.calculate_error_gradient_numerical_differentiation();

   error_gradient = nse.calculate_error_gradient();

   assert_true(absolute_value(numerical_error_gradient - error_gradient) < 1e-3, LOG);
*/
}
}


void NormalizedSquaredErrorTest::test_calculate_error_terms(void) // @todo
{
   cout << "test_calculate_error_terms\n";

   NeuralNetwork neural_network;
   Tensor<Index, 1> architecture;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

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

   DataSet::Batch batch(samples_number, &data_set);

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_indices, target_indices);

   architecture.resize(3);
   architecture[0] = inputs_number;
   architecture[1] = hidden_neurons_number;
   architecture[2] = outputs_number;

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

   const Index parameters_number = neural_network.get_parameters_number();

   nse.set_normalization_coefficient();

   NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);
   LossIndex::BackPropagation back_propagation(samples_number, &nse);
   LossIndex::SecondOrderLoss second_order_loss(parameters_number, samples_number);

   neural_network.forward_propagate(batch, forward_propagation);

   nse.calculate_error(batch, forward_propagation, back_propagation);

   nse.calculate_error_terms(batch, forward_propagation, second_order_loss);

   assert_true(abs(second_order_loss.error - back_propagation.error) < 1.0e-3, LOG);
}


void NormalizedSquaredErrorTest::test_calculate_error_terms_Jacobian(void) // @todo
{
   cout << "test_calculate_error_terms_Jacobian\n";

   NeuralNetwork neural_network;
   Tensor<Index, 1> architecture;

   DataSet data_set;

   NormalizedSquaredError nse(&neural_network, &data_set);

   Index samples_number;
   Index inputs_number;
   Index hidden_neurons_number;
   Index outputs_number;

   // Test

   samples_number = 2;
   inputs_number = 2;
   hidden_neurons_number = 1;
   outputs_number = 1;

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();
   data_set.set_training();

   DataSet::Batch batch(samples_number, &data_set);

   Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
   const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
   const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

   batch.fill(samples_indices, input_indices, target_indices);

   architecture.resize(3);
   architecture[0] = inputs_number;
   architecture[1] = hidden_neurons_number;
   architecture[2] = outputs_number;

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_random();

   const Index parameters_number = neural_network.get_parameters_number();

   nse.set_normalization_coefficient();

   NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);
   LossIndex::BackPropagation back_propagation(samples_number, &nse);
   LossIndex::SecondOrderLoss second_order_loss(parameters_number, samples_number);

   neural_network.forward_propagate(batch, forward_propagation);
   nse.back_propagate(batch, forward_propagation, back_propagation);

   nse.calculate_error_terms_Jacobian(batch, forward_propagation, back_propagation, second_order_loss);

   nse.calculate_error(batch, forward_propagation, back_propagation);

   nse.calculate_error_terms(batch, forward_propagation, second_order_loss);

   assert_true(abs(second_order_loss.error - back_propagation.error) < 1.0e-3, LOG);

   nse.calculate_error_terms_Jacobian(batch, forward_propagation, back_propagation, second_order_loss);

   Tensor<type, 2> numerical_Jacobian_terms;

   forward_propagation.print();
   numerical_Jacobian_terms = nse.calculate_Jacobian_numerical_differentiation(&nse);

   const Tensor<type, 2> difference = second_order_loss.error_terms_Jacobian-numerical_Jacobian_terms;

   assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
}


void NormalizedSquaredErrorTest::test_calculate_squared_errors(void) // @todo
{
    cout << "test_calculate_squared_errors\n";

//    NeuralNetwork neural_network;

//    DataSet data_set;

//    NormalizedSquaredError nse(&neural_network, &data_set);

//    Tensor<type, 1> squared_errors;

//    // Test

//    architecture.setValues({1,1});

//    neural_network.set(NeuralNetwork::Approximation, architecture);
//    neural_network.set_parameters_random();

//    data_set.set(2, 1, 1);
//    data_set.set_data_random();

//    squared_errors = nse.calculate_squared_errors();

//    assert_true(squared_errors.size() == 2, LOG);
}


void NormalizedSquaredErrorTest::test_to_XML(void) // @todo
{
   cout << "test_to_XML\n";
}


void NormalizedSquaredErrorTest::test_from_XML(void) // @todo
{
   cout << "test_from_XML\n";
}


void NormalizedSquaredErrorTest::run_test_case(void) // @todo
{
   cout << "Running normalized squared error test case...\n";

   // Constructor and destructor methods

 /*  test_constructor();
   test_destructor();
   test_calculate_normalization_coefficient();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_error();*/
   test_calculate_error_gradient();

   // Error terms methods
/*
   test_calculate_error_terms();

   test_calculate_error_terms_Jacobian();

   // Squared errors methods

   test_calculate_squared_errors();

   // Serialization methods

   test_to_XML();
   test_from_XML();
*/
   cout << "End of normalized squared error test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques SL.
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
