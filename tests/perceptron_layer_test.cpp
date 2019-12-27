//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   T E S T   C L A S S                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer_test.h"


PerceptronLayerTest::PerceptronLayerTest() : UnitTesting()
{
}


PerceptronLayerTest::~PerceptronLayerTest()
{
}


void PerceptronLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    PerceptronLayer perceptron_layer_1;

    assert_true(perceptron_layer_1.get_inputs_number() == 0, LOG);
    assert_true(perceptron_layer_1.get_neurons_number() == 0, LOG);
    assert_true(perceptron_layer_1.get_type() == Layer::Perceptron, LOG);

    // ARCHITECTURE CONSTRUCTOR

    PerceptronLayer perceptron_layer_2(10, 3, PerceptronLayer::Linear);
    assert_true(perceptron_layer_2.get_activation_function() == PerceptronLayer::Linear, LOG);
    assert_true(perceptron_layer_2.get_inputs_number() == 10, LOG);
    assert_true(perceptron_layer_2.get_neurons_number() == 3, LOG);
    assert_true(perceptron_layer_2.get_biases().size() == 3, LOG);
    assert_true(perceptron_layer_2.get_parameters_number() == 33, LOG);

    // Copy constructor

    perceptron_layer_1.set(1, 2);

    PerceptronLayer perceptron_layer_3(perceptron_layer_1);

    assert_true(perceptron_layer_3.get_inputs_number() == 1, LOG);
    assert_true(perceptron_layer_3.get_neurons_number() == 2, LOG);
}


void PerceptronLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void PerceptronLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   PerceptronLayer perceptron_layer_1;
   PerceptronLayer perceptron_layer_2 = perceptron_layer_1;

   assert_true(perceptron_layer_2.get_inputs_number() == 0, LOG);
   assert_true(perceptron_layer_2.get_neurons_number() == 0, LOG);
}


void PerceptronLayerTest::test_get_inputs_number()
{
   cout << "test_get_inputs_number\n";

   PerceptronLayer perceptron_layer;

   // Test

   perceptron_layer.set();
   assert_true(perceptron_layer.get_inputs_number() == 0, LOG);

   // Test

   perceptron_layer.set(1, 1);
   assert_true(perceptron_layer.get_inputs_number() == 1, LOG);
}


void PerceptronLayerTest::test_get_neurons_number()
{
   cout << "test_get_size\n";

   PerceptronLayer perceptron_layer(1, 1);

   assert_true(perceptron_layer.get_neurons_number() == 1, LOG);
}


void PerceptronLayerTest::test_get_activation_function()
{
   cout << "test_get_activation_function\n";

   PerceptronLayer perceptron_layer(1, 1);
   
   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::Logistic, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::HyperbolicTangent, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::Threshold, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::SymmetricThreshold, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::Linear, LOG);
}


void PerceptronLayerTest::test_write_activation_function()
{
   cout << "test_write_activation_function\n";

   PerceptronLayer perceptron_layer(1, 1);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   assert_true(perceptron_layer.write_activation_function() == "Logistic", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   assert_true(perceptron_layer.write_activation_function() == "HyperbolicTangent", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   assert_true(perceptron_layer.write_activation_function() == "Threshold", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   assert_true(perceptron_layer.write_activation_function() == "SymmetricThreshold", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   assert_true(perceptron_layer.write_activation_function() == "Linear", LOG);
}


void PerceptronLayerTest::test_get_parameters_number()
{      
   cout << "test_get_parameters_number\n";

   PerceptronLayer perceptron_layer;

   // Test

   perceptron_layer.set(1, 1);

   assert_true(perceptron_layer.get_parameters_number() == 2, LOG);

   // Test

   perceptron_layer.set(3, 1);

   assert_true(perceptron_layer.get_parameters_number() == 4, LOG);

   // Test

   perceptron_layer.set(2, 4);

   assert_true(perceptron_layer.get_parameters_number() == 12, LOG);

   // Test

   perceptron_layer.set(4, 2);

   assert_true(perceptron_layer.get_parameters_number() == 10, LOG);

}


void PerceptronLayerTest::test_set()
{
   cout << "test_set\n";
}


void PerceptronLayerTest::test_set_default()
{
   cout << "test_set_default\n";
}


void PerceptronLayerTest::test_get_biases()
{
   cout << "test_get_biases\n";

   PerceptronLayer perceptron_layer;
   Vector<double> biases;

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.initialize_parameters(0.0);

   biases = perceptron_layer.get_biases();

   assert_true(biases.size() == 1, LOG);
   assert_true(biases[0] == 0.0, LOG);
}


void PerceptronLayerTest::test_get_synaptic_weights()
{
   cout << "test_get_synaptic_weights\n";

   PerceptronLayer perceptron_layer;

   Matrix<double> synaptic_weights;

   // Test

   perceptron_layer.set(1, 1);

   perceptron_layer.initialize_parameters(0.0);

   synaptic_weights = perceptron_layer.get_synaptic_weights();

   assert_true(synaptic_weights.get_rows_number() == 1, LOG);
   assert_true(synaptic_weights.get_columns_number() == 1, LOG);
   assert_true(synaptic_weights == 0.0, LOG);
}


void PerceptronLayerTest::test_get_parameters()
{
   cout << "test_get_parameters\n";

   PerceptronLayer perceptron_layer;
   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.initialize_parameters(1.0);

   parameters = perceptron_layer.get_parameters();

   assert_true(parameters.size() ==2, LOG);
   assert_true(parameters == 1.0, LOG);

   // Test

     perceptron_layer.set(2, 4);

    biases.set(4);
    biases[0] = 0.85;
    biases[1] = -0.25;
    biases[2] = 0.29;
    biases[3] = -0.77;

    perceptron_layer.set_biases(biases);

    synaptic_weights.set(4, 2);

    synaptic_weights(0,0) = -0.04;
    synaptic_weights(0,1) = 0.87;

    synaptic_weights(1,0) = 0.25;
    synaptic_weights(1,1) = -0.27;

    synaptic_weights(2,0) = -0.57;
    synaptic_weights(2,1) = 0.15;

    synaptic_weights(3,0) = 0.96;
    synaptic_weights(3,1) = -0.48;

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    parameters = perceptron_layer.get_parameters();

    assert_true(parameters.size() == 12, LOG);
    assert_true(abs(biases[0] - 0.85) < numeric_limits<double>::min(), LOG);
    assert_true(abs(parameters[8] - 0.85) < numeric_limits<double>::epsilon(), LOG);
    assert_true(abs(parameters[7] - -0.48) < numeric_limits<double>::epsilon(), LOG);
}


void PerceptronLayerTest::test_get_perceptrons_parameters()
{
    cout << "test_get_perceptrons_parameters\n";

     PerceptronLayer perceptron_layer;
     Vector<double> biases;
     Matrix<double> synaptic_weights;
     Vector<double>  perceptrons_parameters;
     Vector<double> vector;

     vector.set(3);

     perceptron_layer.set(2, 4);

     biases.set(4);
     biases[0] = 0.85;
     biases[1] = -0.25;
     biases[2] = 0.29;
     biases[3] = -0.77;

     perceptron_layer.set_biases(biases);

     synaptic_weights.set(4, 2);

     synaptic_weights(0,0) = -0.04;
     synaptic_weights(0,1) = 0.87;

     synaptic_weights(1,0) = 0.25;
     synaptic_weights(1,1) = -0.27;

     synaptic_weights(2,0) = -0.57;
     synaptic_weights(2,1) = 0.15;

     synaptic_weights(3,0) = 0.96;
     synaptic_weights(3,1) = -0.48;

     perceptron_layer.set_synaptic_weights(synaptic_weights);

     perceptrons_parameters = perceptron_layer.get_parameters();

     vector[0] = 0.85;
     vector[1] = -0.04;
     vector[2] = 0.87;

     assert_true(perceptrons_parameters.size() == 12 , LOG);
     assert_true(abs(perceptrons_parameters[8] - vector[0])  < numeric_limits<double>::min(), LOG);
}


void PerceptronLayerTest::test_set_biases()
{
   cout << "test_set_biases\n";

    PerceptronLayer perceptron_layer;

    Vector<double> biases;

    // Test

    perceptron_layer.set(1, 1);

    biases.set(1, 0.0);

    perceptron_layer.set_biases(biases);

    assert_true(perceptron_layer.get_biases() == biases, LOG);
}


void PerceptronLayerTest::test_set_synaptic_weights()
{
   cout << "test_set_synaptic_weights\n";

    PerceptronLayer perceptron_layer(1, 2);

    Matrix<double> synaptic_weights(2, 1, 0.0);

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    assert_true(perceptron_layer.get_synaptic_weights() == synaptic_weights, LOG);
    assert_true(perceptron_layer.get_synaptic_weights() == 0.0, LOG);
}


void PerceptronLayerTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

    PerceptronLayer perceptron_layer;
    Vector<double> biases;
    Matrix<double> synaptic_weights;
    Vector<double> new_biases;
    Matrix<double> new_synaptic_weights;

    perceptron_layer.set(2, 2);

    biases.set(2);
    biases[0] = 0.85;
    biases[1] = -0.25;

    perceptron_layer.set_biases(biases);

    synaptic_weights.set(2, 2);

    synaptic_weights(0,0) = -0.04;
    synaptic_weights(0,1) = 0.87;

    synaptic_weights(1,0) = -0.27;
    synaptic_weights(1,1) = -0.57;

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    size_t new_inputs_number = 6;

    perceptron_layer.set_inputs_number(new_inputs_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
}


void PerceptronLayerTest::test_set_perceptrons_number()
{
   cout << "test_set_perceptrons_number\n";

    PerceptronLayer perceptron_layer;
    Vector<double> biases;
    Matrix<double> synaptic_weights;
    Vector<double> new_biases;
    Matrix<double> new_synaptic_weights;

    perceptron_layer.set(3, 2);

    biases.set(2);
    biases[0] = 0.85;
    biases[1] = -0.25;

    perceptron_layer.set_biases(biases);

    synaptic_weights.set(2, 3);

    synaptic_weights(0,0) = -0.04;
    synaptic_weights(0,1) = 0.87;
    synaptic_weights(0,2) = 0.25;

    synaptic_weights(1,0) = -0.27;
    synaptic_weights(1,1) = -0.57;
    synaptic_weights(1,2) = 0.15;

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    size_t new_perceptrons_number = 1;

    perceptron_layer.set_neurons_number(new_perceptrons_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() != new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
}


void PerceptronLayerTest::test_set_parameters()
{
  cout << "test_set_parameters\n";

    PerceptronLayer perceptron_layer(1, 1);

    Vector<double> parameters(2.0,1,3.0);

    perceptron_layer.set_parameters(parameters);

    assert_true(perceptron_layer.get_parameters() == parameters, LOG);
}


void PerceptronLayerTest::test_get_display()
{
   cout << "test_get_display\n";
}


void PerceptronLayerTest::test_set_size()
{
   cout << "test_set_size\n";
}


void PerceptronLayerTest::test_set_activation_function()
{
   cout << "test_set_activation_function\n";
}


void PerceptronLayerTest::test_set_display()
{
   cout << "test_set_display\n";
}


void PerceptronLayerTest::test_grow_inputs()
{
   cout << "test_grow_inputs\n";

     PerceptronLayer perceptron_layer;

     // Test

     perceptron_layer.set();
     perceptron_layer.grow_input();
     assert_true(perceptron_layer.get_inputs_number() == 0, LOG);
     assert_true(perceptron_layer.get_neurons_number() == 0, LOG);

     // Test

     perceptron_layer.set(2, 1);
     perceptron_layer.grow_input();

     assert_true(perceptron_layer.get_inputs_number() == 3, LOG);
     assert_true(perceptron_layer.get_neurons_number() == 1, LOG);
}


void PerceptronLayerTest::test_grow_perceptrons()
{
   cout << "test_grow_perceptrons\n";

     PerceptronLayer perceptron_layer;

     // Test

     perceptron_layer.set(1, 1);
     perceptron_layer.grow_perceptrons(4);

     assert_true(perceptron_layer.get_inputs_number() == 1, LOG);
     assert_true(perceptron_layer.get_neurons_number() == 5, LOG);
}


void PerceptronLayerTest::test_prune_input()
{
   cout << "test_prune_input\n";

    PerceptronLayer perceptron_layer;

     // Test

     perceptron_layer.set(2, 1);
     perceptron_layer.prune_input(0);

     assert_true(perceptron_layer.get_inputs_number() == 1, LOG);
     assert_true(perceptron_layer.get_neurons_number() == 1, LOG);
}


void PerceptronLayerTest::test_prune_neuron()
{
   cout << "test_prune_neuron\n";

    PerceptronLayer perceptron_layer;

    // Test

    perceptron_layer.set(1, 2);
    perceptron_layer.prune_neuron(0);

    assert_true(perceptron_layer.get_inputs_number() == 1, LOG);
    assert_true(perceptron_layer.get_neurons_number() == 1, LOG);
}


void PerceptronLayerTest::test_initialize_parameters()
{
   cout << "test_initialize_parameters\n";

   PerceptronLayer perceptron_layer;

   Vector<double> parameters;

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.initialize_parameters(0.0);

   parameters = perceptron_layer.get_parameters();

   assert_true(parameters == 0.0, LOG);
}


void PerceptronLayerTest::test_initialize_biases()
{
   cout << "test_initialize_biases\n";
}


void PerceptronLayerTest::test_initialize_synaptic_weights()
{
   cout << "test_initialize_synaptic_weights\n";
}


void PerceptronLayerTest::test_randomize_parameters_uniform()
{
   cout << "test_randomize_parameters_uniform\n";

   PerceptronLayer perceptron_layer;
   Vector<double> parameters;

   // Test

   perceptron_layer.set(1,1);

   perceptron_layer.randomize_parameters_uniform();
   parameters = perceptron_layer.get_parameters();
   
   assert_true(parameters >= -1.0, LOG);
   assert_true(parameters <= 1.0, LOG);   
}


void PerceptronLayerTest::test_randomize_parameters_normal()
{
   cout << "test_randomize_parameters_normal\n";

   PerceptronLayer perceptron_layer;
   Vector<double> parameters;

   // Test

   perceptron_layer.set(1, 1);

   perceptron_layer.randomize_parameters_normal(1.0, 0.0);
   parameters = perceptron_layer.get_parameters();

   assert_true(parameters == 1.0, LOG);
}


void PerceptronLayerTest::test_calculate_parameters_norm()
{
   cout << "test_calculate_parameters_norm\n";

   PerceptronLayer perceptron_layer;
   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   double parameters_norm;

   // Test

   perceptron_layer.set(1, 2);
   perceptron_layer.initialize_parameters(0.0);
   parameters=perceptron_layer.get_parameters();

   parameters_norm = perceptron_layer.calculate_parameters_norm();
   assert_true(parameters== 0.0, LOG);
   assert_true(parameters.size() == 4, LOG);

   assert_true(parameters_norm == 0.0, LOG);

   // Test

   perceptron_layer.set(4, 2);

   biases.set(2, 2.0);
   perceptron_layer.set_biases(biases);

   synaptic_weights.set(2, 4, -1.0);
   perceptron_layer.set_synaptic_weights(synaptic_weights);

   parameters = perceptron_layer.get_parameters();

   parameters_norm = perceptron_layer.calculate_parameters_norm();
   assert_true(biases == 2.0, LOG);
   assert_true(synaptic_weights == -1.0, LOG);
   assert_true(abs(parameters_norm - l2_norm(parameters)) < 1.0e-6, LOG);

   // Test

   perceptron_layer.set(4, 2);

   parameters.set(10, 1.0);
   parameters[0] = 0.41;
   parameters[1] = -0.68;
   parameters[2] = 0.14;
   parameters[3] = -0.50;
   parameters[4] = 0.52;
   parameters[5] = -0.70;
   parameters[6] = 0.85;
   parameters[7] = -0.18;
   parameters[8] = -0.65;
   parameters[9] = 0.05;

   perceptron_layer.set_parameters(parameters);

   parameters_norm = perceptron_layer.calculate_parameters_norm();

   assert_true(abs(parameters_norm - l2_norm(parameters)) < 1.0e-6, LOG);
}


void PerceptronLayerTest::test_calculate_combinations()
{
   cout << "test_calculate_combinations\n";

   PerceptronLayer perceptron_layer;

   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   Tensor<double> inputs;
   Tensor<double> combinations;

   // Test

    perceptron_layer.set(1,1);
    perceptron_layer.initialize_biases(1.0);
    perceptron_layer.initialize_synaptic_weights(2.0);

    inputs.set({1,1}, 3.0);

    combinations = perceptron_layer.calculate_combinations(inputs);

    assert_true(combinations.get_dimensions_number() == 2, LOG);
    assert_true(combinations.get_dimension(0) == 1, LOG);
    assert_true(combinations.get_dimension(1) == 1, LOG);
    assert_true(combinations(0,0) == 7.0, LOG);

   // Test

   perceptron_layer.set(2, 2);
   perceptron_layer.initialize_parameters(1);

   inputs.set(Vector<size_t>({1,2}));
   inputs.initialize(1);

   combinations = perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.get_dimensions_number() == 2, LOG);
   assert_true(combinations.get_dimension(0) == 1, LOG);
   assert_true(combinations.get_dimension(1) == 2, LOG);
   assert_true(combinations(0,0) == 3.0, LOG);

   //Test

   perceptron_layer.set(3,4);

   synaptic_weights.set(3,4,1.0);
   biases.set(4,2.0);
   inputs.set({2,3},0.5);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   combinations=perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.get_dimensions_number() == 2, LOG);
   assert_true(combinations.get_dimension(0) == 2, LOG);
   assert_true(combinations.get_dimension(1) == 4, LOG);
   assert_true(combinations(0,0) == 3.5, LOG);

   // Test

   perceptron_layer.set(2, 4);
   perceptron_layer.initialize_biases(1);
   synaptic_weights.set(2,4,1.0);
   biases.set(4,1.0);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   inputs.set(Vector<size_t>({1,2}));
   inputs(0,0) = 0.5;
   inputs(0,1) = 0.5;

   combinations = perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.get_dimensions_number() == 2, LOG);
   assert_true(combinations.get_dimension(0) == 1, LOG);
   assert_true(combinations.get_dimension(1) == 4, LOG);
   assert_true(combinations(0,0) == 2.0, LOG);

   //Test

   perceptron_layer.set(3, 4);

   biases.set(4);
   biases.initialize_sequential();

   synaptic_weights.set(3,4);
   synaptic_weights.initialize_sequential();

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   inputs.set(Vector<size_t>({2,3}));
   inputs.initialize_sequential();

   combinations = perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.get_dimensions_number() == 2, LOG);
   assert_true(combinations.get_dimension(0) == 2, LOG);
   assert_true(combinations.get_dimension(1) == 4, LOG);

   // Test

   perceptron_layer.set(1, 1);

   inputs.set(Vector<size_t>({2,1}));
   inputs.randomize_normal();

   biases.set(1);
   biases.initialize_sequential();

   synaptic_weights.set(1,1);
   synaptic_weights.initialize_sequential();

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   parameters = perceptron_layer.get_parameters();

   assert_true(perceptron_layer.calculate_combinations(inputs) == perceptron_layer.calculate_combinations(inputs, parameters), LOG);
}


void PerceptronLayerTest::test_calculate_activations()
{
   cout << "test_calculate_activations\n";

   PerceptronLayer perceptron_layer;

   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   Tensor<double> inputs;
   Tensor<double> activations;
   Tensor<double> combinations;

   // Test

   perceptron_layer.set(1,1);

   biases.set(1,1.0);
   synaptic_weights.set(1,1,1.0);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);

   inputs.set({1,1},1.0);

   combinations = perceptron_layer.calculate_combinations(inputs);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.get_dimensions_number() == 2, LOG);
   assert_true(activations.get_dimension(0) == 1, LOG);
   assert_true(activations.get_dimension(1) == 1, LOG);
   assert_true(abs(activations.calculate_sum() - 2.0) < numeric_limits<double>::min(), LOG);

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.initialize_parameters(2);

   inputs.set({2,1},2);

   combinations = perceptron_layer.calculate_combinations(inputs);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.get_dimensions_number() == 2, LOG);
   assert_true(activations.get_dimension(0) == 2, LOG);
   assert_true(activations.get_dimension(1) == 1, LOG);
   assert_true(activations(0,0) == 6.0, LOG);

   // Test

   perceptron_layer.set(2, 2);
   parameters.set(6,0.0);

   combinations.set({1,2},0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activations= perceptron_layer.calculate_activations(combinations);

   assert_true(activations.get_dimensions_number() == 2, LOG);
   assert_true(activations.get_dimension(0) == 1, LOG);
   assert_true(activations.get_dimension(1) == 2, LOG);
   assert_true(activations(0,0) == 0.0, LOG);

   // Test

   perceptron_layer.set(1, 2);
   parameters.set(4);
   perceptron_layer.initialize_parameters(0.0);

   combinations.set({2,2},0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.get_dimensions_number() == 2, LOG);
   assert_true(activations.get_dimension(0) == 2, LOG);
   assert_true(activations.get_dimension(1) == 2, LOG);
   assert_true(activations == 1.0 , LOG);

   // Test

   perceptron_layer.set(1, 2);
   perceptron_layer.initialize_parameters(0.0);

   combinations.set({2,2}, -2.0);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.get_dimensions_number() == 2, LOG);
   assert_true(activations.get_dimension(0) == 2, LOG);
   assert_true(activations.get_dimension(1) == 2, LOG);
   assert_true(activations == -1.0, LOG);

   // Test

   perceptron_layer.set(1, 2);
   perceptron_layer.initialize_parameters(0.0);

   combinations.set({2,2},4.0);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.get_dimensions_number() == 2, LOG);
   assert_true(activations.get_dimension(0) == 2, LOG);
   assert_true(activations.get_dimension(1) == 2, LOG);
   assert_true(activations == 4.0, LOG);

   // Test

   perceptron_layer.set(3, 2);

   parameters.set(8,1.0);

   perceptron_layer.set_parameters(parameters);

   inputs.set({1,3},0.5);

   combinations = perceptron_layer.calculate_combinations(inputs);
   assert_true(combinations.get_dimensions_number() == 2, LOG);
   assert_true(combinations.get_dimension(0) == 1, LOG);
   assert_true(combinations.get_dimension(1) == 2, LOG);
   assert_true(combinations(0,0) == 2.5, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations.get_dimensions_number() == 2, LOG);
   assert_true(activations.get_dimension(0) == 1, LOG);
   assert_true(activations.get_dimension(1) == 2, LOG);
   assert_true(activations(0,0) == 1.0, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations(0,0) == 1.0, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(abs(activations(0,0) - 1.0/(1.0+exp(-2.5))) < numeric_limits<double>::min(), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations(0,0) == tanh(2.5), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations(0,0) == 2.5, LOG);
}


void PerceptronLayerTest::test_calculate_activations_derivatives()
{
   cout << "test_calculate_activation_derivative\n";

   NumericalDifferentiation numerical_differentiation;

   PerceptronLayer perceptron_layer;
   Vector<double> parameters;         
   Tensor<double> inputs;
   Tensor<double> combinations;
   Tensor<double> activations_derivatives;
   Tensor<double> numerical_activation_derivative;

   numerical_differentiation_tests = true;

   // Test

   perceptron_layer.set(1, 1);
   combinations.set({1,1}, 0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
   assert_true(activations_derivatives.get_dimensions_number() == 2, LOG);
   assert_true(activations_derivatives.get_dimension(0) == 1, LOG);
   assert_true(activations_derivatives.get_dimension(1) == 1, LOG);
   assert_true(activations_derivatives == 0.25, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
   assert_true(activations_derivatives.get_dimensions_number() == 2, LOG);
   assert_true(activations_derivatives.get_dimension(0) == 1, LOG);
   assert_true(activations_derivatives.get_dimension(1) == 1, LOG);
   assert_true(activations_derivatives == 1.0, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
   assert_true(activations_derivatives.get_dimensions_number() == 2, LOG);
   assert_true(activations_derivatives.get_dimension(0) == 1, LOG);
   assert_true(activations_derivatives.get_dimension(1) == 1, LOG);
   assert_true(activations_derivatives == 1.0, LOG);

   // Test

   if(numerical_differentiation_tests)
   {
      perceptron_layer.set(2, 4);

      combinations.set(Vector<size_t>({1,4}));
      combinations(0,0) = 1.56;
      combinations(0,2) = -0.68;
      combinations(0,2)= 0.91;
      combinations(0,3) = -1.99;

      perceptron_layer.set_activation_function(PerceptronLayer::Logistic);

      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);

      assert_true(absolute_value(absolute_value(activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);


      perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);

      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);

      assert_true(absolute_value(activations_derivatives - numerical_activation_derivative) < 1.0e-3, LOG);


      perceptron_layer.set_activation_function(PerceptronLayer::Linear);

      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);

      assert_true(absolute_value(activations_derivatives - numerical_activation_derivative) < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      perceptron_layer.set(4, 2);

      parameters.set(10);
      parameters[0] = 0.41;
      parameters[1] = -0.68; 
      parameters[2] = 0.14; 
      parameters[3] = -0.50; 
      parameters[4] = 0.52; 
      parameters[5] = -0.70; 
      parameters[6] = 0.85; 
      parameters[7] = -0.18; 
      parameters[8] = -0.65; 
      parameters[9] = 0.05; 

      perceptron_layer.set_parameters(parameters);

      inputs.set(Vector<size_t>({1,4}));
      inputs[0] = 0.85;
      inputs[1] = -0.25;
      inputs[2] = 0.29;
      inputs[3] = -0.77;

      combinations = perceptron_layer.calculate_combinations(inputs);

      perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::Linear);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
      assert_true(absolute_value((activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);
   }
}


void PerceptronLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_activation_derivative\n";

   PerceptronLayer perceptron_layer;

   Vector<double> parameters;
   Tensor<double> inputs;
   Tensor<double> outputs;
   Vector<double> potential_outputs;

   // Test 

   perceptron_layer.set(3, 2);
   perceptron_layer.initialize_parameters(0.0);

   inputs.set(Vector<size_t>({1,3}));
   inputs.initialize(0.0);

   outputs = perceptron_layer.calculate_outputs(inputs);

   assert_true(outputs.get_dimensions_number() == 2, LOG);
   assert_true(outputs.get_dimension(0) == 1, LOG);
   assert_true(outputs.get_dimension(1) == 2, LOG);
   assert_true(outputs == 0.0, LOG);

   // Test

   perceptron_layer.set(4, 2);

   parameters.set(10);
   parameters[0] = 0.41;
   parameters[1] = -0.68; 
   parameters[2] = 0.14; 
   parameters[3] = -0.50; 
   parameters[4] = 0.52; 
   parameters[5] = -0.70; 
   parameters[6] = 0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] = 0.05; 

   perceptron_layer.set_parameters(parameters);

   inputs.set(Vector<size_t>({1,4}));
   inputs[0] = 0.85;
   inputs[1] = -0.25;
   inputs[2] = 0.29;
   inputs[3] = -0.77;

   outputs = perceptron_layer.calculate_outputs(inputs);

   assert_true(outputs.get_dimensions_number() == 2, LOG);
   assert_true(outputs.get_dimension(0) == 1, LOG);
   assert_true(outputs.get_dimension(1) == 2, LOG);

   // Test

   inputs.set(Vector<size_t>({1,1}));
   inputs.initialize((3.0));

   perceptron_layer.set(1, 1);

   perceptron_layer.initialize_parameters(2.0);

   outputs = perceptron_layer.calculate_outputs(inputs);

   parameters.set(2, 1.0);

   potential_outputs = perceptron_layer.calculate_outputs(inputs, parameters);

   assert_true(outputs != potential_outputs, LOG);

   // Test

   perceptron_layer.set(1, 1);

   inputs.set(Vector<size_t>({1,1}));
   inputs.randomize_normal();

   parameters = perceptron_layer.get_parameters();

   assert_true(perceptron_layer.calculate_outputs(inputs) == perceptron_layer.calculate_outputs(inputs, parameters), LOG);
}


void PerceptronLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";
}


void PerceptronLayerTest::run_test_case()
{
   cout << "Running perceptron layer test case...\n";

   // Constructor and destructor

   test_constructor();

   test_destructor();

   // Assignment operators

   test_assignment_operator();

   // Get methods

   // Inputs and perceptrons

   test_get_inputs_number();

   test_get_neurons_number();

   // Parameters

   test_get_parameters_number();

   test_get_biases();

   test_get_synaptic_weights();

   test_get_parameters();

   test_get_perceptrons_parameters();

   // Activation functions

   test_get_activation_function();

   test_write_activation_function();

   // Display messages

   test_get_display();

   // Set methods

   test_set();

   test_set_default();

   // Perceptron layer parameters

   test_set_biases();

   test_set_synaptic_weights();

   test_set_perceptrons_number();

   // Inputs

   test_set_inputs_number();

   // Activation functions

   test_set_activation_function();

   // Parameters methods

   test_set_parameters();

   // Display messages

   test_set_display();

   // Growing and pruning

   test_grow_inputs();

   test_grow_perceptrons();

   test_prune_input();

   test_prune_neuron();

   // Parameters initialization methods

   test_initialize_parameters();

   test_initialize_biases();

   test_initialize_synaptic_weights();

   test_randomize_parameters_uniform();

   test_randomize_parameters_normal();

   // Parameters initialization methods

   test_initialize_parameters();

   test_randomize_parameters_uniform();

   test_randomize_parameters_normal();

   // Parameters norm

   test_calculate_parameters_norm();      

   // Combination

   test_calculate_combinations();

   // Activation

   test_calculate_activations();

   test_calculate_activations_derivatives();

   // Outputs

   test_calculate_outputs();

  // Expression methods

   test_write_expression();

   cout << "End of perceptron layer test case.\n";
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
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
