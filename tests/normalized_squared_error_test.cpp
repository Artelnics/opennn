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

   assert_true(!normalized_squared_error_1.has_neural_network(), LOG);
   assert_true(!normalized_squared_error_1.has_data_set(), LOG);

   // Neural network and data set

   NeuralNetwork neural_network_3;
   DataSet data_set_3;
   NormalizedSquaredError nse3(&neural_network_3, &data_set_3);

   assert_true(nse3.has_neural_network(), LOG);
   assert_true(nse3.has_data_set(), LOG);
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
   Index outputs_number = 4;

   Tensor<type, 1> targets_mean(outputs_number);
   Tensor<type, 2> targets(samples_number, outputs_number);

   // Test

   data_set.generate_random_data(samples_number, inputs_number+outputs_number);

   Tensor<string, 1> uses(8);
   uses.setValues({"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"});

   data_set.set_columns_uses(uses);

   targets = data_set.get_target_data();
//   targets_mean = data_set.calculate_training_targets_mean();

   Tensor<Index, 1> architecture(2);
   architecture.setValues({inputs_number, outputs_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();

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
   DataSetBatch batch(1, &data_set);

   Tensor<Index,1> batch_samples_indices = data_set.get_used_samples_indices();
   Tensor<Index,1> inputs_indices = data_set.get_input_variables_indices();
   Tensor<Index,1> targets_indices = data_set.get_target_variables_indices();

   batch.fill(batch_samples_indices, inputs_indices, targets_indices);
   Index batch_samples_number = batch.get_samples_number();

   NeuralNetworkForwardPropagation forward_propagation(batch_samples_number, &neural_network);

   LossIndexBackPropagation back_propagation(batch_samples_number, &normalized_squared_error);

   neural_network.forward_propagate(batch, forward_propagation);

   normalized_squared_error.calculate_error(batch, forward_propagation, back_propagation);

   assert_true(back_propagation.error == 0.0, LOG);

   // Test

   samples_number = 7;
   inputs_number = 8;
   outputs_number = 5;
   hidden_neurons = 3;

   architecture.setValues({inputs_number, hidden_neurons, outputs_number});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_random();

   parameters = neural_network.get_parameters();

   data_set.set(samples_number, inputs_number, outputs_number);
   data_set.set_data_random();

   normalized_squared_error.set_normalization_coefficient();

//   assert_true(abs(normalized_squared_error.calculate_error() - normalized_squared_error.calculate_training_error(parameters)) < 1.0e-3, LOG);
}


/// @todo This test method does not work if the number of samples is equal to 1

void NormalizedSquaredErrorTest::test_calculate_error_gradient(void)
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

   RecurrentLayer* recurrent_layer = new RecurrentLayer();

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

   PerceptronLayer* hidden_perceptron_layer = new PerceptronLayer();
   PerceptronLayer* output_perceptron_layer = new PerceptronLayer();

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

   // Trivial test
   {
       samples_number = 10;
       inputs_number = 1;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.initialize_data(0.0);
       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       hidden_perceptron_layer->set(inputs_number, outputs_number);
       neural_network.add_layer(hidden_perceptron_layer);

       neural_network.set_parameters_constant(0.0);

       nse.set_normalization_coefficient(1.0);

       nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation training_back_propagation(samples_number, &nse);

       neural_network.forward_propagate(batch, forward_propagation);

       nse.back_propagate(batch, forward_propagation, training_back_propagation);
       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = nse.calculate_gradient_numerical_differentiation(&nse);

       assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
       assert_true(std::all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(),
                               [](type i) { return (i-static_cast<type>(0))<std::numeric_limits<type>::min(); }), LOG);
   }

   neural_network.set();

   // Test perceptron

   {
       samples_number = 10;
       inputs_number = 1;
       outputs_number = 1;

       const Index neurons_number = 3;

       data_set.set(samples_number, inputs_number, outputs_number);
       data_set.set_data_random();
       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       batch.fill(samples_indices, input_indices, target_indices);

       hidden_perceptron_layer->set(inputs_number, neurons_number);
       output_perceptron_layer->set(neurons_number, outputs_number);

       neural_network.add_layer(hidden_perceptron_layer);
       neural_network.add_layer(output_perceptron_layer);

       neural_network.set_parameters_random();

       nse.set_normalization_coefficient(1.0);

       nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation training_back_propagation(samples_number, &nse);

       neural_network.forward_propagate(batch, forward_propagation);

       nse.back_propagate(batch, forward_propagation, training_back_propagation);
       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = nse.calculate_gradient_numerical_differentiation(&nse);

       assert_true((error_gradient.dimension(0) == neural_network.get_parameters_number()) , LOG);
       assert_true(std::all_of(error_gradient.data(), error_gradient.data()+error_gradient.size(),
                               [](type i) { return (i-static_cast<type>(0)) < std::numeric_limits<type>::min(); }), LOG);
   }

   // Test perceptron and binary probabilistic
   {
       samples_number = 3;
       inputs_number = 3;
       hidden_neurons = 4;
       outputs_number = 1;

       data_set.set(samples_number, inputs_number, outputs_number);

       data_set.set_data_binary_random();

       const Index columns_number = inputs_number+1;

       Tensor<DataSet::Column, 1> columns(columns_number);

       for(Index i = 0; i < columns_number-1; i++)
       {
           columns(i).name = "input_" + std::to_string(i+1);
           columns(i).column_use = DataSet::Input;
           columns(i).type = DataSet::Numeric;
       }

       Tensor<DataSet::VariableUse, 1> categories_uses(2);
       categories_uses.setConstant(DataSet::VariableUse::Target);

       Tensor<string, 1> categories(2);
       categories(0) = "category_0";
       categories(1) = "category_1";

       columns(columns_number-1).name = "target";
       columns(columns_number-1).column_use = DataSet::Target;
       columns(columns_number-1).type = DataSet::Binary;
       columns(columns_number-1).categories = categories;
       columns(columns_number-1).categories_uses = categories_uses;

       data_set.set_columns(columns);

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

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

       neural_network.set_parameters_random();

       nse.set_normalization_coefficient();
       nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation training_back_propagation(samples_number, &nse);

       neural_network.forward_propagate(batch, forward_propagation);

       nse.back_propagate(batch, forward_propagation, training_back_propagation);

       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = nse.calculate_gradient_numerical_differentiation(&nse);

       const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

       assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
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

       const Index columns_number = inputs_number+1;

       Tensor<DataSet::Column, 1> columns(columns_number);

       for(Index i = 0; i < columns_number-1; i++)
       {
           columns(i).name = "input_" + std::to_string(i+1);
           columns(i).column_use = DataSet::Input;
           columns(i).type = DataSet::Numeric;
       }

       Tensor<DataSet::VariableUse, 1> categories_uses(outputs_number);
       categories_uses.setConstant(DataSet::VariableUse::Target);

       Tensor<string, 1> categories(outputs_number);

       for(Index i = 0; i < outputs_number; i++) categories(i) = "category_" + std::to_string(i+1);

       columns(columns_number-1).name = "target";
       columns(columns_number-1).column_use = DataSet::Target;
       columns(columns_number-1).type = DataSet::Categorical;
       columns(columns_number-1).categories = categories;
       columns(columns_number-1).categories_uses = categories_uses;

       data_set.set_columns(columns);

       data_set.set_training();

       DataSetBatch batch(samples_number, &data_set);

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

       neural_network.set_parameters_random();

       nse.set_normalization_coefficient();
       nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
       LossIndexBackPropagation training_back_propagation(samples_number, &nse);

       neural_network.forward_propagate(batch, forward_propagation);

       nse.back_propagate(batch, forward_propagation, training_back_propagation);

       error_gradient = training_back_propagation.gradient;

       numerical_error_gradient = nse.calculate_gradient_numerical_differentiation(&nse);

       const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

       assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
   }

   neural_network.set();

   // Test lstm

   {
       samples_number = 4;
       inputs_number = 2;
       outputs_number = 4;
       hidden_neurons = 3;

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

       nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       nse.set_normalization_coefficient();

       long_short_term_memory_layer->set_timesteps(2);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &nse);

       neural_network.forward_propagate(batch, forward_propagation);

       nse.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = nse.calculate_gradient_numerical_differentiation(&nse);

       const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

       assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
   }

   neural_network.set();

   // Test recurrent
   {
       samples_number = 4;
       inputs_number = 1;
       outputs_number = 1;
       hidden_neurons = 2;

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

       nse.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

       nse.set_normalization_coefficient();

       recurrent_layer->set_timesteps(2);

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &nse);

       neural_network.forward_propagate(batch, forward_propagation);

       nse.back_propagate(batch, forward_propagation, back_propagation);

       error_gradient = back_propagation.gradient;

       numerical_error_gradient = nse.calculate_gradient_numerical_differentiation(&nse);

       const Tensor<type, 1> difference = error_gradient-numerical_error_gradient;

       assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
   }

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
       data_set.initialize_data(0.5);
       data_set.set_training();

       Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
       const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
       const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

       DataSetBatch batch(samples_number, &data_set);
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

       NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);

       LossIndexBackPropagation back_propagation(samples_number, &nse);

       neural_network.forward_propagate(batch, forward_propagation);

       nse.back_propagate(batch, forward_propagation, back_propagation);

       numerical_error_gradient = nse.calculate_gradient_numerical_differentiation(&nse);
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

   DataSetBatch batch(samples_number, &data_set);

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

   NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
   LossIndexBackPropagation back_propagation(samples_number, &nse);
   LossIndexBackPropagationLM loss_index_back_propagation_lm(parameters_number, samples_number);

   neural_network.forward_propagate(batch, forward_propagation);

   nse.calculate_error(batch, forward_propagation, back_propagation);

   nse.calculate_squared_errors(batch, forward_propagation, loss_index_back_propagation_lm);

   assert_true(abs(loss_index_back_propagation_lm.error - back_propagation.error) < 1.0e-3, LOG);
}


/// @todo

void NormalizedSquaredErrorTest::test_calculate_error_terms_Jacobian(void)
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

   DataSetBatch batch(samples_number, &data_set);

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

   NeuralNetworkForwardPropagation forward_propagation(samples_number, &neural_network);
   LossIndexBackPropagation back_propagation(samples_number, &nse);
   LossIndexBackPropagationLM loss_index_back_propagation_lm(parameters_number, samples_number);

   neural_network.forward_propagate(batch, forward_propagation);
   nse.back_propagate(batch, forward_propagation, back_propagation);

//   nse.calculate_squared_errors_Jacobian(batch, forward_propagation, loss_index_back_propagation_lm);

   nse.calculate_error(batch, forward_propagation, back_propagation);

   nse.calculate_squared_errors(batch, forward_propagation, loss_index_back_propagation_lm);

   assert_true(abs(loss_index_back_propagation_lm.error - back_propagation.error) < 1.0e-3, LOG);

//   nse.calculate_error_terms_Jacobian(batch, forward_propagation, loss_index_back_propagation_lm);

   Tensor<type, 2> numerical_Jacobian_terms;

   forward_propagation.print();
   numerical_Jacobian_terms = nse.calculate_Jacobian_numerical_differentiation(&nse);

   const Tensor<type, 2> difference = loss_index_back_propagation_lm.squared_errors_Jacobian-numerical_Jacobian_terms;

   assert_true(std::all_of(difference.data(), difference.data()+difference.size(), [](type i) { return (i)<static_cast<type>(1.0e-3); }), LOG);
}


/// @todo

void NormalizedSquaredErrorTest::test_calculate_squared_errors(void)
{
    cout << "test_calculate_squared_errors\n";

    NeuralNetwork neural_network;

    DataSet data_set;

    NormalizedSquaredError nse(&neural_network, &data_set);

    Tensor<Index, 1> architecture;
    Tensor<type, 1> squared_errors;

    // Test

    architecture.setValues({1,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_random();

    data_set.set(2, 1, 1);
    data_set.set_data_random();

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

   test_constructor();
   test_destructor();
   test_calculate_normalization_coefficient();

   // Get methods

   // Set methods

   // Error methods

   test_calculate_error();
   test_calculate_error_gradient();

   // Error terms methods

   test_calculate_error_terms();

   test_calculate_error_terms_Jacobian();

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
