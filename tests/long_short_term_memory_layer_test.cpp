//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y    L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "long_short_term_memory_layer_test.h"


LongShortTermMemoryLayerTest::LongShortTermMemoryLayerTest() : UnitTesting()
{
}


LongShortTermMemoryLayerTest::~LongShortTermMemoryLayerTest()
{
}


void LongShortTermMemoryLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    LongShortTermMemoryLayer long_short_term_memory_layer;
    size_t inputs_number;
    size_t neurons_number;

    Matrix<double> synaptic_weights;
    Matrix<double> recurrent_initializer;
    Vector<double> biases;

    // Test

    long_short_term_memory_layer.set();

    assert_true(long_short_term_memory_layer.get_forget_weights().get_columns_number() == 0, LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;

    long_short_term_memory_layer.set(inputs_number, neurons_number);

    assert_true(long_short_term_memory_layer.get_parameters_number() == 12, LOG);

    assert_true(long_short_term_memory_layer.get_biases().get_columns_number() == 4, LOG);
    assert_true(long_short_term_memory_layer.get_biases().get_rows_number()
             == long_short_term_memory_layer.get_neurons_number(), LOG);

    assert_true(long_short_term_memory_layer.get_weights().get_dimensions_number() == 3, LOG);

    assert_true(long_short_term_memory_layer.get_weights().get_dimension(0)
             == long_short_term_memory_layer.get_inputs_number(), LOG);
    assert_true(long_short_term_memory_layer.get_weights().get_dimension(1)
             == long_short_term_memory_layer.get_neurons_number(), LOG);
    assert_true(long_short_term_memory_layer.get_weights().get_dimension(2) == 4, LOG);

    //Test

    inputs_number = 2;
    neurons_number = 3;

    long_short_term_memory_layer.set(inputs_number, neurons_number);

    assert_true(long_short_term_memory_layer.get_parameters_number() == 72, LOG);


    assert_true(long_short_term_memory_layer.get_biases().get_columns_number() == 4, LOG);
    assert_true(long_short_term_memory_layer.get_biases().get_rows_number()
             == long_short_term_memory_layer.get_neurons_number(), LOG);

    assert_true(long_short_term_memory_layer.get_weights().get_dimensions_number() == 3, LOG);

    assert_true(long_short_term_memory_layer.get_recurrent_weights().get_dimension(0)
             == long_short_term_memory_layer.get_neurons_number(), LOG);
    assert_true(long_short_term_memory_layer.get_recurrent_weights().get_dimension(1)
             == long_short_term_memory_layer.get_neurons_number(), LOG);
    assert_true(long_short_term_memory_layer.get_recurrent_weights().get_dimension(2) == 4, LOG);

}


void LongShortTermMemoryLayerTest::test_destructor()
{
   cout << "test_destructor\n";

}

void LongShortTermMemoryLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   LongShortTermMemoryLayer long_short_term_memory_layer_1;

   long_short_term_memory_layer_1.set(4,3);

   LongShortTermMemoryLayer long_short_term_memory_layer_2 = long_short_term_memory_layer_1;

   assert_true(long_short_term_memory_layer_1.get_inputs_number() == 4, LOG);
   assert_true(long_short_term_memory_layer_1.get_neurons_number() == 3, LOG);
}

void LongShortTermMemoryLayerTest::test_get_inputs_number()
{
   cout << "test_get_inputs_number\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   size_t inputs_number;
   size_t neurons_number;


   // Test

   long_short_term_memory_layer.set(0,0);
   assert_true(long_short_term_memory_layer.get_inputs_number() == 0, LOG);

   // Test

   inputs_number = 2;
   neurons_number = 3;

   long_short_term_memory_layer.set(inputs_number, neurons_number);
   assert_true(long_short_term_memory_layer.get_inputs_number() == inputs_number, LOG);
   assert_true(long_short_term_memory_layer.get_neurons_number() == neurons_number, LOG);
}

void LongShortTermMemoryLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   size_t inputs_number;
   size_t neurons_number;


   // Test

   long_short_term_memory_layer.set();
   assert_true(long_short_term_memory_layer.get_neurons_number() == 0, LOG);

   // Test

   inputs_number = 2;
   neurons_number = 3;

   long_short_term_memory_layer.set(inputs_number, neurons_number);
   assert_true(long_short_term_memory_layer.get_neurons_number() == 3, LOG);
}


void LongShortTermMemoryLayerTest::test_get_activation_function()
{
   cout << "test_get_activation_function\n";

   LongShortTermMemoryLayer long_short_term_memory_layer(1, 1);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::Logistic);
   assert_true(long_short_term_memory_layer.get_activation_function() == LongShortTermMemoryLayer::Logistic, LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::HyperbolicTangent);
   assert_true(long_short_term_memory_layer.get_activation_function() == LongShortTermMemoryLayer::HyperbolicTangent, LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::Threshold);
   assert_true(long_short_term_memory_layer.get_activation_function() == LongShortTermMemoryLayer::Threshold, LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::SymmetricThreshold);
   assert_true(long_short_term_memory_layer.get_activation_function() == LongShortTermMemoryLayer::SymmetricThreshold, LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::Linear);
   assert_true(long_short_term_memory_layer.get_activation_function() == LongShortTermMemoryLayer::Linear, LOG);
}

void LongShortTermMemoryLayerTest::test_write_activation_function()
{
   cout << "test_write_activation_function\n";

   LongShortTermMemoryLayer long_short_term_memory_layer(1, 1);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::Logistic);
   assert_true(long_short_term_memory_layer.write_activation_function() == "Logistic", LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::HyperbolicTangent);
   assert_true(long_short_term_memory_layer.write_activation_function() == "HyperbolicTangent", LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::Threshold);
   assert_true(long_short_term_memory_layer.write_activation_function() == "Threshold", LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::SymmetricThreshold);
   assert_true(long_short_term_memory_layer.write_activation_function() == "SymmetricThreshold", LOG);

   long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::Linear);
   assert_true(long_short_term_memory_layer.write_activation_function() == "Linear", LOG);
}


void LongShortTermMemoryLayerTest::test_get_recurrent_activation_function()
{
   cout << "test_get_activation_function\n";

   LongShortTermMemoryLayer long_short_term_memory_layer(1, 1);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::Logistic);
   assert_true(long_short_term_memory_layer.get_recurrent_activation_function() == LongShortTermMemoryLayer::Logistic, LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::HyperbolicTangent);
   assert_true(long_short_term_memory_layer.get_recurrent_activation_function() == LongShortTermMemoryLayer::HyperbolicTangent, LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::Threshold);
   assert_true(long_short_term_memory_layer.get_recurrent_activation_function() == LongShortTermMemoryLayer::Threshold, LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::SymmetricThreshold);
   assert_true(long_short_term_memory_layer.get_recurrent_activation_function() == LongShortTermMemoryLayer::SymmetricThreshold, LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::Linear);
   assert_true(long_short_term_memory_layer.get_recurrent_activation_function() == LongShortTermMemoryLayer::Linear, LOG);
}


void LongShortTermMemoryLayerTest::test_write_recurrent_activation_function()
{
   cout << "test_write_activation_function\n";

   LongShortTermMemoryLayer long_short_term_memory_layer(1, 1);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::Logistic);
   assert_true(long_short_term_memory_layer.write_recurrent_activation_function() == "Logistic", LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::HyperbolicTangent);
   assert_true(long_short_term_memory_layer.write_recurrent_activation_function() == "HyperbolicTangent", LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::Threshold);
   assert_true(long_short_term_memory_layer.write_recurrent_activation_function() == "Threshold", LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::SymmetricThreshold);
   assert_true(long_short_term_memory_layer.write_recurrent_activation_function() == "SymmetricThreshold", LOG);

   long_short_term_memory_layer.set_recurrent_activation_function(LongShortTermMemoryLayer::Linear);
   assert_true(long_short_term_memory_layer.write_recurrent_activation_function() == "Linear", LOG);
}

void LongShortTermMemoryLayerTest::test_get_parameters_number()
{
   cout << "test_get_parameters_number\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   // Test

   long_short_term_memory_layer.set(1, 1);

   assert_true(long_short_term_memory_layer.get_parameters_number() == 12, LOG);

   // Test

   long_short_term_memory_layer.set(3, 1);

   assert_true(long_short_term_memory_layer.get_parameters_number() == 20, LOG);

   // Test

   long_short_term_memory_layer.set(2, 4);

   assert_true(long_short_term_memory_layer.get_parameters_number() == 28 * 4 , LOG);

   // Test

   long_short_term_memory_layer.set(4, 2);

   assert_true(long_short_term_memory_layer.get_parameters_number() == 28 * 2, LOG);

}

void LongShortTermMemoryLayerTest::test_get_biases()
{
   cout << "test_get_biases\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   size_t inputs_number;
   size_t neurons_number;

   //Test

   neurons_number = 3;
   inputs_number = 2;

   long_short_term_memory_layer.set(inputs_number,neurons_number);

   long_short_term_memory_layer.initialize_biases(0.0);

   assert_true(long_short_term_memory_layer.get_biases().size() == 12, LOG);
   assert_true(long_short_term_memory_layer.get_biases().get_rows_number() == neurons_number, LOG);
   assert_true(long_short_term_memory_layer.get_biases().get_columns_number() == 4, LOG);
   
   //Test

   neurons_number = 3;
   inputs_number = 2;

   long_short_term_memory_layer.set(inputs_number,neurons_number);

   long_short_term_memory_layer.set_forget_biases({0,1,2});
   long_short_term_memory_layer.set_input_biases({3,4,5});
   long_short_term_memory_layer.set_state_biases({5,7,8});
   long_short_term_memory_layer.set_output_biases({9,10,11});

   assert_true(long_short_term_memory_layer.get_forget_biases() == long_short_term_memory_layer.get_biases().get_column(0), LOG);
   assert_true(long_short_term_memory_layer.get_input_biases() == long_short_term_memory_layer.get_biases().get_column(1), LOG);
   assert_true(long_short_term_memory_layer.get_state_biases() == long_short_term_memory_layer.get_biases().get_column(2), LOG);
   assert_true(long_short_term_memory_layer.get_output_biases() == long_short_term_memory_layer.get_biases().get_column(3), LOG);

}

void LongShortTermMemoryLayerTest::test_get_weights()
{
   cout << "test_get_synaptic_weights\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Tensor<double> weights;

   //Test

   long_short_term_memory_layer.set(3,2);

   long_short_term_memory_layer.initialize_parameters(0.0);

   weights = long_short_term_memory_layer.get_weights();

   assert_true(weights.get_dimensions_number() == 3, LOG);
   assert_true(weights.get_dimension(0) == 3, LOG);
   assert_true(weights.get_dimension(1) == 2, LOG);
   assert_true(weights.get_dimension(2) == 4, LOG);
   assert_true(weights == 0.0, LOG);

   //Test

   long_short_term_memory_layer.set(3, 2);

   long_short_term_memory_layer.randomize_parameters_uniform();

   assert_true(long_short_term_memory_layer.get_forget_weights() == long_short_term_memory_layer.get_weights().get_matrix(0), LOG);
   assert_true(long_short_term_memory_layer.get_input_weights() == long_short_term_memory_layer.get_weights().get_matrix(1), LOG);
   assert_true(long_short_term_memory_layer.get_state_weights() == long_short_term_memory_layer.get_weights().get_matrix(2), LOG);
   assert_true(long_short_term_memory_layer.get_output_weights() == long_short_term_memory_layer.get_weights().get_matrix(3), LOG);
}

void LongShortTermMemoryLayerTest::test_get_recurrent_weights()
{
   cout << "test_get_recurrent_weights\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Tensor<double> recurrent_weights;

   //Test

   long_short_term_memory_layer.set(3,2);

   long_short_term_memory_layer.initialize_parameters(0.0);

   recurrent_weights = long_short_term_memory_layer.get_recurrent_weights();

   assert_true(recurrent_weights.get_dimensions_number() == 3, LOG);
   assert_true(recurrent_weights.get_dimension(0) == 2, LOG);
   assert_true(recurrent_weights.get_dimension(1) == 2, LOG);
   assert_true(recurrent_weights.get_dimension(2) == 4, LOG);
   assert_true(recurrent_weights == 0.0, LOG);

   //Test

   long_short_term_memory_layer.set(3, 2);

   long_short_term_memory_layer.randomize_parameters_uniform();

   assert_true(long_short_term_memory_layer.get_forget_recurrent_weights() == long_short_term_memory_layer.get_recurrent_weights().get_matrix(0), LOG);
   assert_true(long_short_term_memory_layer.get_input_recurrent_weights() == long_short_term_memory_layer.get_recurrent_weights().get_matrix(1), LOG);
   assert_true(long_short_term_memory_layer.get_state_recurrent_weights() == long_short_term_memory_layer.get_recurrent_weights().get_matrix(2), LOG);
   assert_true(long_short_term_memory_layer.get_output_recurrent_weights() == long_short_term_memory_layer.get_recurrent_weights().get_matrix(3), LOG);
}


void LongShortTermMemoryLayerTest::test_set_biases()
{
   cout << "test_set_biases\n";

    LongShortTermMemoryLayer long_short_term_memory_layer;

    Matrix<double> biases;

    // Test

    long_short_term_memory_layer.set(1, 1);


    biases.set(1, 4, 0.0);

    long_short_term_memory_layer.set_forget_biases(biases.get_column(0));
    long_short_term_memory_layer.set_input_biases(biases.get_column(1));
    long_short_term_memory_layer.set_state_biases(biases.get_column(2));
    long_short_term_memory_layer.set_output_biases(biases.get_column(3));

    assert_true(long_short_term_memory_layer.get_biases() == biases, LOG);
}


void LongShortTermMemoryLayerTest::test_set_weights()
{
   cout << "test_set_synaptic_weights\n";

    LongShortTermMemoryLayer long_short_term_memory_layer(1, 2);

    Tensor<double> weights({1, 2, 4},  4.0);

    long_short_term_memory_layer.set_forget_weights(weights.get_matrix(0));
    long_short_term_memory_layer.set_input_weights(weights.get_matrix(1));
    long_short_term_memory_layer.set_state_weights(weights.get_matrix(2));
    long_short_term_memory_layer.set_output_weights(weights.get_matrix(3));
    long_short_term_memory_layer.get_weights();

    assert_true(long_short_term_memory_layer.get_weights() == weights, LOG);
    assert_true(long_short_term_memory_layer.get_weights() == 4.0, LOG);
}


void LongShortTermMemoryLayerTest::test_set_recurrent_weights()
{
   cout << "test_set_synaptic_weights\n";

    LongShortTermMemoryLayer long_short_term_memory_layer(1, 2);

    Tensor<double> recurrent_weights({2, 2, 4},  0.0);

    long_short_term_memory_layer.set_forget_recurrent_weights(recurrent_weights.get_matrix(0));
    long_short_term_memory_layer.set_input_recurrent_weights(recurrent_weights.get_matrix(1));
    long_short_term_memory_layer.set_state_recurrent_weights(recurrent_weights.get_matrix(2));
    long_short_term_memory_layer.set_output_recurrent_weights(recurrent_weights.get_matrix(3));

    assert_true(long_short_term_memory_layer.get_recurrent_weights() == recurrent_weights, LOG);
    assert_true(long_short_term_memory_layer.get_recurrent_weights() == 0.0, LOG);
}

void LongShortTermMemoryLayerTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

    LongShortTermMemoryLayer long_short_term_memory_layer;
    Matrix<double> biases;
    Tensor<double> weights;
    Tensor<double> recurrent_weights;

    Matrix<double> new_biases;
    Tensor<double> new_weights;
    Tensor<double> new_recurrent_weights;

    long_short_term_memory_layer.set(2, 3);

    biases.set(3, 4, 1.0);
    long_short_term_memory_layer.set_forget_biases(biases.get_column(0));
    long_short_term_memory_layer.set_input_biases(biases.get_column(1));
    long_short_term_memory_layer.set_state_biases(biases.get_column(2));
    long_short_term_memory_layer.set_output_biases(biases.get_column(3));

    weights.set({2, 3, 4}, 6.0);

    long_short_term_memory_layer.set_forget_weights(weights.get_matrix(0));
    long_short_term_memory_layer.set_input_weights(weights.get_matrix(1));
    long_short_term_memory_layer.set_state_weights(weights.get_matrix(2));
    long_short_term_memory_layer.set_output_weights(weights.get_matrix(3));

    recurrent_weights.set({3, 3, 4}, 2.0);

    long_short_term_memory_layer.set_forget_recurrent_weights(recurrent_weights.get_matrix(0));
    long_short_term_memory_layer.set_input_recurrent_weights(recurrent_weights.get_matrix(1));
    long_short_term_memory_layer.set_state_recurrent_weights(recurrent_weights.get_matrix(2));
    long_short_term_memory_layer.set_output_recurrent_weights(recurrent_weights.get_matrix(3));

    size_t new_inputs_number = 6;

    long_short_term_memory_layer.set_inputs_number(new_inputs_number);

    new_biases = long_short_term_memory_layer.get_biases();
    new_weights = long_short_term_memory_layer.get_weights();
    new_recurrent_weights = long_short_term_memory_layer.get_recurrent_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(weights != new_weights, LOG);
    assert_true(recurrent_weights == new_recurrent_weights, LOG);
}


void LongShortTermMemoryLayerTest::test_set_parameters()
{
  cout << "test_set_parameters\n";

    LongShortTermMemoryLayer long_short_term_memory_layer(1, 1);

    Vector<double> parameters(12);

    parameters.initialize_sequential();

    long_short_term_memory_layer.set_parameters(parameters);

    assert_true(long_short_term_memory_layer.get_parameters() == parameters, LOG);
}


void LongShortTermMemoryLayerTest::test_initialize_parameters()
{
   cout << "test_initialize_parameters\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Vector<double> parameters;

   // Test

   long_short_term_memory_layer.set(3, 2);
   long_short_term_memory_layer.initialize_parameters(0.0);

   parameters = long_short_term_memory_layer.get_parameters();

   assert_true(parameters == 0.0, LOG);
   assert_true(parameters.size() == 48, LOG);
}


void LongShortTermMemoryLayerTest::test_initialize_biases()
{
   cout << "test_initialize_biases\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Vector<double> forget_biases;
   Vector<double> input_biases;
   Vector<double> state_biases;
   Vector<double> output_biases;

   // Test

   long_short_term_memory_layer.set(3, 2);

   long_short_term_memory_layer.initialize_forget_biases(0.0);
   forget_biases = long_short_term_memory_layer.get_forget_biases();

   long_short_term_memory_layer.initialize_input_biases(1.0);
   input_biases = long_short_term_memory_layer.get_input_biases();

   long_short_term_memory_layer.initialize_state_biases(2.0);
   state_biases = long_short_term_memory_layer.get_state_biases();

   long_short_term_memory_layer.initialize_output_biases(3.0);
   output_biases = long_short_term_memory_layer.get_output_biases();

   assert_true(forget_biases == 0.0, LOG);
   assert_true(forget_biases.size() == 2, LOG);

   assert_true(input_biases == 1.0, LOG);
   assert_true(input_biases.size() == 2, LOG);

   assert_true(state_biases == long_short_term_memory_layer.get_biases().get_column(2), LOG);
   assert_true(output_biases == long_short_term_memory_layer.get_biases().get_column(3), LOG);
}


void LongShortTermMemoryLayerTest::test_initialize_weights()
{
   cout << "test_initialize_synaptic_weights\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Matrix<double> forget_weights;
   Matrix<double> input_weights;
   Matrix<double> state_weights;
   Matrix<double> output_weights;

   // Test

   long_short_term_memory_layer.set(3, 2);

   long_short_term_memory_layer.initialize_forget_weights(0.0);
   forget_weights = long_short_term_memory_layer.get_forget_weights();

   long_short_term_memory_layer.initialize_input_weights(1.0);
   input_weights = long_short_term_memory_layer.get_input_weights();

   long_short_term_memory_layer.initialize_state_weights(2.0);
   state_weights = long_short_term_memory_layer.get_state_weights();

   long_short_term_memory_layer.initialize_output_weights(3.0);
   output_weights = long_short_term_memory_layer.get_output_weights();

   assert_true(forget_weights == 0.0, LOG);
   assert_true(forget_weights.size() == 6, LOG);

   assert_true(input_weights == 1.0, LOG);
   assert_true(input_weights.size() == 6, LOG);

   assert_true(state_weights == long_short_term_memory_layer.get_weights().get_matrix(2), LOG);
   assert_true(output_weights == long_short_term_memory_layer.get_weights().get_matrix(3), LOG);
}


void LongShortTermMemoryLayerTest::test_initialize_recurrent_weights()
{
   cout << "test_initialize_recurrent_weights\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Matrix<double> forget_recurrent_weights;
   Matrix<double> input_recurrent_weights;
   Matrix<double> state_recurrent_weights;
   Matrix<double> output_recurrent_weights;

   // Test

   long_short_term_memory_layer.set(3, 2);

   long_short_term_memory_layer.initialize_forget_recurrent_weights(0.0);
   forget_recurrent_weights = long_short_term_memory_layer.get_forget_recurrent_weights();

   long_short_term_memory_layer.initialize_input_recurrent_weights(1.0);
   input_recurrent_weights = long_short_term_memory_layer.get_input_recurrent_weights();

   long_short_term_memory_layer.initialize_state_recurrent_weights(2.0);
   state_recurrent_weights = long_short_term_memory_layer.get_state_recurrent_weights();

   long_short_term_memory_layer.initialize_output_recurrent_weights(3.0);
   output_recurrent_weights = long_short_term_memory_layer.get_output_recurrent_weights();

   assert_true(forget_recurrent_weights == 0.0, LOG);
   assert_true(forget_recurrent_weights.size() == 4, LOG);

   assert_true(input_recurrent_weights == 1.0, LOG);
   assert_true(input_recurrent_weights.size() == 4, LOG);

   assert_true(state_recurrent_weights == long_short_term_memory_layer.get_recurrent_weights().get_matrix(2), LOG);
   assert_true(output_recurrent_weights == long_short_term_memory_layer.get_recurrent_weights().get_matrix(3), LOG);
}

void LongShortTermMemoryLayerTest::test_randomize_parameters_uniform()
{
   cout << "test_randomize_parameters_uniform\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;
   Vector<double> parameters;

   // Test

   long_short_term_memory_layer.set(1,1);

   long_short_term_memory_layer.randomize_parameters_uniform();
   parameters = long_short_term_memory_layer.get_parameters();

   assert_true(parameters >= -1.0, LOG);
   assert_true(parameters <= 1.0, LOG);

}


void LongShortTermMemoryLayerTest::test_randomize_parameters_normal()
{
   cout << "test_randomize_parameters_normal\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;
   Vector<double> parameters;

   // Test

   long_short_term_memory_layer.set(1, 1);

   long_short_term_memory_layer.randomize_parameters_normal(1.0, 0.0);
   parameters = long_short_term_memory_layer.get_parameters();

   assert_true(parameters == 1.0, LOG);
}


void LongShortTermMemoryLayerTest::test_calculate_parameters_norm()
{
   cout << "test_calculate_parameters_norm\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Vector<double> parameters;

   double parameters_norm;

   // Test

   long_short_term_memory_layer.set(1, 2);
   long_short_term_memory_layer.initialize_parameters(0.0);
   parameters=long_short_term_memory_layer.get_parameters();

   parameters_norm = long_short_term_memory_layer.calculate_parameters_norm();

   assert_true(parameters == 0.0, LOG);
   assert_true(parameters.size() == 32, LOG);
   assert_true(parameters_norm == 0.0, LOG);

   // Test

   long_short_term_memory_layer.set(4, 2);

   long_short_term_memory_layer.initialize_biases(2.0);
   long_short_term_memory_layer.initialize_biases(1.0);
   long_short_term_memory_layer.initialize_recurrent_weights(-0.5);

   parameters = long_short_term_memory_layer.get_parameters();

   parameters_norm = long_short_term_memory_layer.calculate_parameters_norm();

   assert_true(abs(parameters_norm - l2_norm(parameters)) < 1.0e-6, LOG);


   // Test

   long_short_term_memory_layer.set(4, 2);

   parameters.set(56, 1.0);

   long_short_term_memory_layer.set_parameters(parameters);

   parameters_norm = long_short_term_memory_layer.calculate_parameters_norm();

   assert_true(abs(parameters_norm - l2_norm(parameters)) < 1.0e-6, LOG);
}

void LongShortTermMemoryLayerTest::test_get_parameters()
{
   cout << "test_get_parameters\n";
   LongShortTermMemoryLayer long_short_term_memory_layer;

   long_short_term_memory_layer.set(1,2);



   long_short_term_memory_layer.initialize_forget_weights(1.0);
   long_short_term_memory_layer.initialize_state_weights(3.0);
   long_short_term_memory_layer.initialize_input_weights(2.0);
   long_short_term_memory_layer.initialize_output_weights(4.0);
   long_short_term_memory_layer.initialize_forget_recurrent_weights(5.0);
   long_short_term_memory_layer.initialize_output_recurrent_weights(8.0);
   long_short_term_memory_layer.initialize_state_recurrent_weights(7.0);
   long_short_term_memory_layer.initialize_input_recurrent_weights(6.0);
   long_short_term_memory_layer.initialize_forget_biases(9.0);
   long_short_term_memory_layer.initialize_input_biases(10.0);
   long_short_term_memory_layer.initialize_state_biases(11.0);
   long_short_term_memory_layer.initialize_output_biases(12.0);

//   cout<< "parameters.  "<< long_short_term_memory_layer.get_parameters()<<endl;
//   cout<< "weight.  "<< long_short_term_memory_layer.get_weights()<<endl;
//   cout<< "recurrent_weight.  "<< long_short_term_memory_layer.get_recurrent_weights()<<endl;
//   cout<< "biases "<< long_short_term_memory_layer.get_biases()<<endl;

}

void LongShortTermMemoryLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   LongShortTermMemoryLayer long_short_term_memory_layer;

   Tensor<double> inputs;
   Tensor<double> outputs;

   Vector<double> parameters;

   Tensor<double> weights;
   Tensor<double> recurrent_weights;
   Matrix<double> biases;

   size_t instances = 3;

    //Test

   long_short_term_memory_layer.set(3,2);
   long_short_term_memory_layer.set_timesteps(2);

   long_short_term_memory_layer.set_activation_function("HyperbolicTangent");
   long_short_term_memory_layer.set_recurrent_activation_function("HardSigmoid");

   inputs.set({instances, 3},1.0);

   weights.set(Vector<size_t>({3,2,4}));
   recurrent_weights.set(Vector<size_t>({2,2,4}));
   biases.set(2,4);

   weights.initialize_sequential();
   recurrent_weights.initialize_sequential();
   biases.initialize_sequential();

   parameters = weights.to_vector().assemble(recurrent_weights.to_vector()).assemble(biases.to_vector());

   long_short_term_memory_layer.set_forget_weights(weights.get_matrix(0));
   long_short_term_memory_layer.set_input_weights(weights.get_matrix(1));
   long_short_term_memory_layer.set_state_weights(weights.get_matrix(2));
   long_short_term_memory_layer.set_output_weights(weights.get_matrix(3));

   long_short_term_memory_layer.set_forget_recurrent_weights(recurrent_weights.get_matrix(0));
   long_short_term_memory_layer.set_input_recurrent_weights(recurrent_weights.get_matrix(1));
   long_short_term_memory_layer.set_state_recurrent_weights(recurrent_weights.get_matrix(2));
   long_short_term_memory_layer.set_output_recurrent_weights(recurrent_weights.get_matrix(3));

   long_short_term_memory_layer.set_forget_biases(biases.get_column(0));
   long_short_term_memory_layer.set_input_biases(biases.get_column(1));
   long_short_term_memory_layer.set_state_biases(biases.get_column(2));
   long_short_term_memory_layer.set_output_biases(biases.get_column(3));

   assert_true(long_short_term_memory_layer.calculate_outputs(inputs) == long_short_term_memory_layer.calculate_outputs(inputs, parameters), LOG);

   assert_true(long_short_term_memory_layer.calculate_outputs(inputs, biases, weights, recurrent_weights) == long_short_term_memory_layer.calculate_outputs(inputs, parameters), LOG);

   assert_true(long_short_term_memory_layer.calculate_outputs(inputs) == long_short_term_memory_layer.calculate_outputs(inputs,biases, weights, recurrent_weights), LOG);

   //Test

   long_short_term_memory_layer.set(1, 2 );

   long_short_term_memory_layer.set_timesteps(3);

   inputs.set({3,1}, 1.0);

   long_short_term_memory_layer.initialize_biases(0.0);
   long_short_term_memory_layer.initialize_weights(1.0);
   long_short_term_memory_layer.initialize_recurrent_weights(1.0);

   long_short_term_memory_layer.set_activation_function("HyperbolicTangent");
   long_short_term_memory_layer.set_recurrent_activation_function("HardSigmoid");

   cout<< "outputs:  "<< long_short_term_memory_layer.calculate_outputs(inputs)<<endl;
}


void LongShortTermMemoryLayerTest::run_test_case()
{
   cout << "Running long short term memory layer test case...\n";

   // Constructor and destructor

   test_constructor();

   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Inputs and perceptrons

   test_get_inputs_number();
   test_get_neurons_number();

   //Activation funcions

   test_get_activation_function();
   test_write_activation_function();

   test_get_recurrent_activation_function();
   test_write_recurrent_activation_function();

   // Parameters

   test_get_parameters_number();
   test_get_parameters();

   test_get_biases();
   test_get_weights();
   test_get_recurrent_weights();

   // lstm layer parameters

   test_set_biases();

   test_set_weights();

   test_set_recurrent_weights();

   // Inputs

   test_set_inputs_number();

   // Parameters methods

   test_set_parameters();

   // Parameters initialization methods

   test_initialize_parameters();

   test_initialize_biases();

   test_initialize_weights();
   test_initialize_recurrent_weights();

   test_randomize_parameters_uniform();

   test_randomize_parameters_normal();

   // Parameters norm

   test_calculate_parameters_norm();

   // Calculate outputs

   test_calculate_outputs();


   cout << "End of long short term memory layer test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
