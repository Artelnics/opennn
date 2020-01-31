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
   Tensor<type, 2> biases;

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.set_parameters_constant(0.0);
   biases = perceptron_layer.get_biases();

   assert_true(biases.size() == 1, LOG);
   assert_true(biases(0) == static_cast<type>(0.0), LOG);

   // Test

   cout << "test_get_biases_with_parameters\n";

   PerceptronLayer perceptron_layer_2;
   Tensor<type, 2> biases_2(1, 4);
   Tensor<type, 2> synaptic_weights(2, 4);
   Tensor<type, 1> parameters(12);
   perceptron_layer.set(2, 4);




//   biases.resize(1, 4);


   biases_2(0,0) = static_cast<type>(0.85);
   biases_2(0,1) = -static_cast<type>(0.25);
   biases_2(0,2) = static_cast<type>(0.29);
   biases_2(0,3) = -static_cast<type>(0.77);

   synaptic_weights(0,0) = -static_cast<type>(0.04);
   synaptic_weights(1,0) = static_cast<type>(0.87);
   synaptic_weights(0,1) = static_cast<type>(0.25);
   synaptic_weights(1,1) = -static_cast<type>(0.27);
   synaptic_weights(0,2) = -static_cast<type>(0.57);
   synaptic_weights(1,2) = static_cast<type>(0.15);
   synaptic_weights(0,3) = static_cast<type>(0.96);
   synaptic_weights(1,3) = -static_cast<type>(0.48);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases_2);

   parameters = perceptron_layer.get_parameters();
   biases = perceptron_layer.get_biases(parameters);
/*
   assert_true(biases.size() == 4, LOG);
   assert_true(biases(0,0) == static_cast<type>(0.85), LOG);
   assert_true(biases(0,3) == -static_cast<type>(0.77), LOG);
*/
}


void PerceptronLayerTest::test_get_synaptic_weights()
{
   cout << "test_get_synaptic_weights\n";

   PerceptronLayer perceptron_layer;
   Tensor<type, 2> synaptic_weights;

   // Test

   perceptron_layer.set(1, 1);

   perceptron_layer.set_parameters_constant(0.0);

   synaptic_weights = perceptron_layer.get_synaptic_weights();

   assert_true(synaptic_weights.dimension(0) == 1, LOG);
   assert_true(synaptic_weights.dimension(1) == 1, LOG);
   assert_true(synaptic_weights(0,0) == static_cast<type>(0.0), LOG);

   // Test

   cout << "test_get_synaptic_weight_with_parameters\n";

   PerceptronLayer perceptron_layer_2;
   Tensor<type, 2> biases_2;

   Tensor<type, 1> parameters;

   perceptron_layer_2.set(2, 4);

   biases_2.resize(1, 4);
   synaptic_weights.resize(2, 4);

   biases_2(0,0) = static_cast<type>(0.85);
   biases_2(0,1) = -static_cast<type>(0.25);
   biases_2(0,2) = static_cast<type>(0.29);
   biases_2(0,3) = -static_cast<type>(0.77);

   synaptic_weights(0,0) = -static_cast<type>(0.04);
   synaptic_weights(1,0) = static_cast<type>(0.87);
   synaptic_weights(0,1) = static_cast<type>(0.25);
   synaptic_weights(1,1) = -static_cast<type>(0.27);
   synaptic_weights(0,2) = -static_cast<type>(0.57);
   synaptic_weights(1,2) = static_cast<type>(0.15);
   synaptic_weights(0,3) = static_cast<type>(0.96);
   synaptic_weights(1,3) = -static_cast<type>(0.48);

   perceptron_layer_2.set_synaptic_weights(synaptic_weights);
   perceptron_layer_2.set_biases(biases_2);

   parameters = perceptron_layer_2.get_parameters();

   Tensor<type,2> test_synaptic_weights = perceptron_layer_2.get_synaptic_weights(parameters);


   assert_true(test_synaptic_weights.size() == 8, LOG);
   assert_true(test_synaptic_weights(0,0) == -static_cast<type>(0.04), LOG);
   assert_true(test_synaptic_weights(1,3) == -static_cast<type>(0.48), LOG);

}


void PerceptronLayerTest::test_get_parameters()
{

   cout << "test_get_parameters\n";

   PerceptronLayer perceptron_layer;
   Tensor<type, 2> synaptic_weights;
   Tensor<type, 1> parameters;

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.set_parameters_constant(1.0);

   parameters = perceptron_layer.get_parameters();

   assert_true(parameters.size() == 2, LOG);
   assert_true(parameters(0) == static_cast<type>(1.0), LOG);

   // Test

   PerceptronLayer perceptron_layer_2;
   Tensor<type, 2> biases_2(1, 4);
   Tensor<type, 2> synaptic_weights_2(2, 4);


    perceptron_layer_2.set(2, 4);

    biases_2(0,0) = static_cast<type>(0.85);
    biases_2(0,1) = -static_cast<type>(0.25);
    biases_2(0,2) = static_cast<type>(0.29);
    biases_2(0,3) = -static_cast<type>(0.77);

    synaptic_weights_2(0,0) = -static_cast<type>(0.04);
    synaptic_weights_2(1,0) = static_cast<type>(0.87);
    synaptic_weights_2(0,1) = static_cast<type>(0.25);
    synaptic_weights_2(1,1) = -static_cast<type>(0.27);
    synaptic_weights_2(0,2) = -static_cast<type>(0.57);
    synaptic_weights_2(1,2) = static_cast<type>(0.15);
    synaptic_weights_2(0,3) = static_cast<type>(0.96);
    synaptic_weights_2(1,3) = -static_cast<type>(0.48);

    perceptron_layer_2.set_synaptic_weights(synaptic_weights_2);
    perceptron_layer_2.set_biases(biases_2);

    Tensor<type,1>new_parameters = perceptron_layer_2.get_parameters();

    assert_true(new_parameters.size() == 12, LOG);
    assert_true(abs(new_parameters(0) - static_cast<type>(-0.04)) < static_cast<type>(1e-4), LOG);
    assert_true(abs(new_parameters(8) - static_cast<type>(0.85)) < static_cast<type>(1e-4), LOG);
    assert_true(abs(new_parameters(7) - static_cast<type>(-0.48)) < static_cast<type>(1e-5), LOG);

    }


void PerceptronLayerTest::test_get_perceptrons_parameters()
{

    cout << "test_get_perceptrons_parameters\n";
/*
     PerceptronLayer perceptron_layer;
     Tensor<type, 1> biases;
     Tensor<type, 2> synaptic_weights;
     Tensor<type, 1>  perceptrons_parameters;
     Tensor<type, 1> vector;

     vector.resize(3);

     perceptron_layer.set(2, 4);

     biases.resize(4);
     biases[0] = 0.85;
     biases[1] = -0.25;
     biases[2] = 0.29;
     biases[3] = -0.77;

     perceptron_layer.set_biases(biases);

     synaptic_weights.resize(4, 2);

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
//     assert_true(abs(perceptrons_parameters[8] - vector[0])  < numeric_limits<type>::min(), LOG);
*/
}


void PerceptronLayerTest::test_set_biases()
{

   cout << "test_set_biases\n";

    PerceptronLayer perceptron_layer;

    Tensor<type, 2> biases(1, 4);

    // Test

    perceptron_layer.set(1, 4);

    biases.setConstant(4);

    perceptron_layer.set_biases(biases);

    assert_true(perceptron_layer.get_biases().size() == 4, LOG);

}


void PerceptronLayerTest::test_set_synaptic_weights()
{
   cout << "test_set_synaptic_weights\n";

//    PerceptronLayer perceptron_layer(1, 2);

//    Tensor<type, 2> synaptic_weights(2, 1);

//    synaptic_weights.setConstant(0.0);

//    perceptron_layer.set_synaptic_weights(synaptic_weights);

//    assert_true(perceptron_layer.get_synaptic_weights() == synaptic_weights, LOG);
//    assert_true(perceptron_layer.get_synaptic_weights() == 0.0, LOG);
}


void PerceptronLayerTest::test_set_inputs_number()
{
    /*
   cout << "test_set_inputs_number\n";

    PerceptronLayer perceptron_layer;
    Tensor<type, 1> biases;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 1> new_biases;
    Tensor<type, 2> new_synaptic_weights;

    perceptron_layer.set(2, 2);

    biases.resize(2);
    biases[0] = 0.85;
    biases[1] = -0.25;

    perceptron_layer.set_biases(biases);

    synaptic_weights.resize(2, 2);

    synaptic_weights(0,0) = -0.04;
    synaptic_weights(0,1) = 0.87;

    synaptic_weights(1,0) = -0.27;
    synaptic_weights(1,1) = -0.57;

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_inputs_number = 6;

    perceptron_layer.set_inputs_number(new_inputs_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
    */
}


void PerceptronLayerTest::test_set_perceptrons_number()
{
   cout << "test_set_perceptrons_number\n";
/*
    PerceptronLayer perceptron_layer;
    Tensor<type, 1> biases;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 1> new_biases;
    Tensor<type, 2> new_synaptic_weights;

    perceptron_layer.set(3, 2);

    biases.resize(2);
    biases[0] = 0.85;
    biases[1] = -0.25;

    perceptron_layer.set_biases(biases);

    synaptic_weights.resize(2, 3);

    synaptic_weights(0,0) = -0.04;
    synaptic_weights(0,1) = 0.87;
    synaptic_weights(0,2) = 0.25;

    synaptic_weights(1,0) = -0.27;
    synaptic_weights(1,1) = -0.57;
    synaptic_weights(1,2) = 0.15;

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_perceptrons_number = 1;

    perceptron_layer.set_neurons_number(new_perceptrons_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() != new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
    */
}


void PerceptronLayerTest::test_set_parameters()
{
  cout << "test_set_parameters\n";

    PerceptronLayer perceptron_layer(1, 2);

    Tensor<type, 1> parameters(4);

    parameters.setValues({1,1,2,2});

    perceptron_layer.set_parameters(parameters);

    cout<<perceptron_layer.get_synaptic_weights();

    assert_true(perceptron_layer.get_synaptic_weights()(0) == 1, LOG);
    assert_true(perceptron_layer.get_biases()(0) == 2, LOG);
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


void PerceptronLayerTest::test_set_parameters_constant()
{
   cout << "test_set_parameters_constant\n";

   PerceptronLayer perceptron_layer;

   Tensor<type, 1> parameters;

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.set_parameters_constant(0.0);

   parameters = perceptron_layer.get_parameters();

//   assert_true(parameters == 0.0, LOG);
}


void PerceptronLayerTest::test_initialize_biases()
{
   cout << "test_initialize_biases\n";
}


void PerceptronLayerTest::test_initialize_synaptic_weights()
{
   cout << "test_initialize_synaptic_weights\n";
}


void PerceptronLayerTest::test_set_parameters_random()
{
   cout << "test_set_parameters_random\n";
/*
   PerceptronLayer perceptron_layer;
   Tensor<type, 1> parameters;;

   // Test

   perceptron_layer.set(1,1);

   perceptron_layer.set_parameters_random();
   parameters = perceptron_layer.get_parameters();
   
//   assert_true(parameters >= -1.0, LOG);
//   assert_true(parameters <= 1.0, LOG);
*/
}


void PerceptronLayerTest::test_calculate_parameters_norm()
{
   cout << "test_calculate_parameters_norm\n";
/*
   PerceptronLayer perceptron_layer;
   Tensor<type, 1> biases;
   Tensor<type, 2> synaptic_weights;
   Tensor<type, 1> parameters;

   type parameters_norm;

   // Test

   perceptron_layer.set(1, 2);
   perceptron_layer.set_parameters_constant(0.0);
   parameters=perceptron_layer.get_parameters();

   parameters_norm = perceptron_layer.calculate_parameters_norm();
//   assert_true(parameters== 0.0, LOG);
   assert_true(parameters.size() == 4, LOG);

   assert_true(parameters_norm == 0.0, LOG);

   // Test

   perceptron_layer.set(4, 2);

   biases.resize(2);
   biases.setConstant(2.0);

   perceptron_layer.set_biases(biases);

   synaptic_weights.resize(2, 4);
   synaptic_weights.setConstant(-1.0);

   perceptron_layer.set_synaptic_weights(synaptic_weights);

   parameters = perceptron_layer.get_parameters();

   parameters_norm = perceptron_layer.calculate_parameters_norm();
//   assert_true(biases == 2.0, LOG);
//   assert_true(synaptic_weights == -1.0, LOG);
//   assert_true(abs(parameters_norm - l2_norm(parameters)) < 1.0e-6, LOG);

   // Test

   perceptron_layer.set(4, 2);

   parameters.resize(10);
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

//   assert_true(abs(parameters_norm - l2_norm(parameters)) < 1.0e-6, LOG);
*/
}


void PerceptronLayerTest::test_calculate_combinations()
{

   cout << "test_calculate_combinations\n";

   PerceptronLayer perceptron_layer;

   Tensor<type, 2> biases(1,1);
   Tensor<type, 2> synaptic_weights(1,1);
   Tensor<type, 1> parameters(1);

   Tensor<type, 2> inputs(1,1);
   Tensor<type, 2> combinations(1,1);

   // Test

   perceptron_layer.set(1,1);
   perceptron_layer.initialize_biases(1.0);
   perceptron_layer.initialize_synaptic_weights(2.0);

   inputs.resize(1,1);

   inputs.setConstant(3.0);

   combinations = perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.rank() == 2, LOG);
   assert_true(combinations.dimension(0) == 1, LOG);
   assert_true(combinations.dimension(1) == 1, LOG);
   assert_true(combinations(0,0) == 7.0, LOG);

    // Test

   perceptron_layer.set(2, 2);
   perceptron_layer.set_parameters_constant(1);

   inputs.resize(1,2);

   inputs.setConstant(1.0);

   combinations = perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.rank() == 2, LOG);
   assert_true(combinations.dimension(0) == 1, LOG);
   assert_true(combinations.dimension(1) == 2, LOG);
   assert_true(combinations(0,0) == 3.0, LOG);

   //Test

   perceptron_layer.set(3,4);

   synaptic_weights.resize(3,4);

   synaptic_weights.setConstant(1.0);

   biases.resize(1,4);

   biases.setConstant(2.0);

   inputs.resize(2,3);

   inputs.setConstant(0.5);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   combinations=perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.rank() == 2, LOG);
   assert_true(combinations.dimension(0) == 2, LOG);
   assert_true(combinations.dimension(1) == 4, LOG);
   assert_true(combinations(0,0) == 3.5, LOG);

   // Test

   perceptron_layer.set(2, 4);
   perceptron_layer.initialize_biases(1);
   synaptic_weights.resize(2,4);
   synaptic_weights.setConstant(1.0);

   biases.resize(1,4);
   biases.setConstant(1.0);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   inputs.resize(1,2);
   inputs(0,0) = 0.5;
   inputs(0,1) = 0.5;

   combinations = perceptron_layer.calculate_combinations(inputs);

   assert_true(combinations.rank() == 2, LOG);
   assert_true(combinations.dimension(0) == 1, LOG);
   assert_true(combinations.dimension(1) == 4, LOG);
   assert_true(combinations(0,0) == 2.0, LOG);

   //Test
   PerceptronLayer perceptron_layer_3(1,4);
   Tensor<type,2> combinations_3(10, 4);

   perceptron_layer_3.initialize_biases(1);

   perceptron_layer_3.initialize_synaptic_weights(1);

   Tensor<type, 2> inputs_3(10,1);
   inputs_3.setConstant(1);

   combinations_3 = perceptron_layer.calculate_combinations(inputs_3, perceptron_layer_3.get_synaptic_weights(), perceptron_layer_3.get_biases());

//   synaptic_weights.setConstant(1.0);

//   biases.resize(1,4);
//   biases.setConstant(1.0);

//   perceptron_layer.set_synaptic_weights(synaptic_weights);
//   perceptron_layer.set_biases(biases);

//   inputs.resize(1,2);
//   inputs(0,0) = 0.5;
//   inputs(0,1) = 0.5;



//   assert_true(combinations.rank() == 2, LOG);
//   assert_true(combinations.dimension(0) == 1, LOG);
//   assert_true(combinations.dimension(1) == 4, LOG);
//   assert_true(combinations(0,0) == 2.0, LOG);

//   perceptron_layer.set(3, 4);

//   biases.resize(1,4);

//   synaptic_weights.resize(3,4);

//   perceptron_layer.set_synaptic_weights(synaptic_weights);
//   perceptron_layer.set_biases(biases);

//   inputs.resize(2,3);

//   combinations = perceptron_layer.calculate_combinations(inputs, synaptic_weights, biases);

//   assert_true(combinations.rank() == 2, LOG);
//   assert_true(combinations.dimension(0) == 2, LOG);
//   assert_true(combinations.dimension(1) == 4, LOG);

}


void PerceptronLayerTest::test_calculate_activations()
{

   cout << "test_calculate_activations\n";
/*
   PerceptronLayer perceptron_layer;

   Tensor<type, 1> biases;
   Tensor<type, 2> synaptic_weights;
   Tensor<type, 1> parameters;

   Tensor<type, 2> inputs;
   Tensor<type, 2> activations;
   Tensor<type, 2> combinations;

   // Test

   perceptron_layer.set(1,1);

   biases.resize(1);
   biases.setConstant(1.0);

   synaptic_weights.resize(1,1);
   synaptic_weights.setConstant(1.0);

   synaptic_weights.setConstant(1.0);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);

   inputs.resize(1,1);

   inputs.setConstant(1.0);

   combinations = perceptron_layer.calculate_combinations(inputs);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 1, LOG);
   assert_true(activations.dimension(1) == 1, LOG);
//   assert_true(abs(activations.sum() - 2.0) < numeric_limits<type>::min(), LOG);

   // Test

   perceptron_layer.set(1, 1);
   perceptron_layer.set_parameters_constant(2);

   inputs.resize(2,1);
   inputs.setConstant(2);

   combinations = perceptron_layer.calculate_combinations(inputs);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 2, LOG);
   assert_true(activations.dimension(1) == 1, LOG);
   assert_true(activations(0,0) == 6.0, LOG);

   // Test

   perceptron_layer.set(2, 2);
   parameters.resize(6);

   parameters.setConstant(0.0);

   combinations.resize(1,2);

   combinations.setConstant(0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activations= perceptron_layer.calculate_activations(combinations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 1, LOG);
   assert_true(activations.dimension(1) == 2, LOG);
   assert_true(activations(0,0) == 0.0, LOG);

   // Test

   perceptron_layer.set(1, 2);
   parameters.resize(4);
   perceptron_layer.set_parameters_constant(0.0);

   combinations.resize(2,2);

   combinations.setConstant(0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 2, LOG);
   assert_true(activations.dimension(1) == 2, LOG);
//   assert_true(activations == 1.0 , LOG);

   // Test

   perceptron_layer.set(1, 2);
   perceptron_layer.set_parameters_constant(0.0);

   combinations.resize(2,2);

   combinations.setConstant(-2.0);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 2, LOG);
   assert_true(activations.dimension(1) == 2, LOG);
//   assert_true(activations == -1.0, LOG);

   // Test

   perceptron_layer.set(1, 2);
   perceptron_layer.set_parameters_constant(0.0);

   combinations.resize(2,2);

   combinations.setConstant(4.0);


   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations = perceptron_layer.calculate_activations(combinations);

   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 2, LOG);
   assert_true(activations.dimension(1) == 2, LOG);
//   assert_true(activations == 4.0, LOG);

   // Test

   perceptron_layer.set(3, 2);

   parameters.resize(8);

   parameters.setConstant(1.0);

   perceptron_layer.set_parameters(parameters);

   inputs.resize(1,3);

   inputs.setConstant(0.5);

   combinations = perceptron_layer.calculate_combinations(inputs);
   assert_true(combinations.rank() == 2, LOG);
   assert_true(combinations.dimension(0) == 1, LOG);
   assert_true(combinations.dimension(1) == 2, LOG);
   assert_true(combinations(0,0) == 2.5, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations.rank() == 2, LOG);
   assert_true(activations.dimension(0) == 1, LOG);
   assert_true(activations.dimension(1) == 2, LOG);
   assert_true(activations(0,0) == 1.0, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations(0,0) == 1.0, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   activations = perceptron_layer.calculate_activations(combinations);
//   assert_true(abs(activations(0,0) - 1.0/(1.0+exp(-2.5))) < numeric_limits<type>::min(), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations(0,0) == tanh(2.5), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations = perceptron_layer.calculate_activations(combinations);
   assert_true(activations(0,0) == 2.5, LOG);
   */
}


void PerceptronLayerTest::test_calculate_activations_derivatives()
{

   cout << "test_calculate_activation_derivative\n";
/*
   NumericalDifferentiation numerical_differentiation;

   PerceptronLayer perceptron_layer;
   Tensor<type, 1> parameters;
   Tensor<type, 2> inputs;
   Tensor<type, 2> combinations;
   Tensor<type, 2> activations_derivatives;
   Tensor<type, 2> numerical_activation_derivative;

   numerical_differentiation_tests = true;

   // Test

   perceptron_layer.set(1, 1);

   combinations.resize(1,1);

   combinations.setConstant(0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
//   assert_true(activations_derivatives == 0.25, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
//   assert_true(activations_derivatives == 1.0, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
//   assert_true(activations_derivatives == 1.0, LOG);

   // Test

   if(numerical_differentiation_tests)
   {
      perceptron_layer.set(2, 4);

      combinations.resize(1,4);
      combinations(0,0) = 1.56;
      combinations(0,2) = -0.68;
      combinations(0,2)= 0.91;
      combinations(0,3) = -1.99;

      perceptron_layer.set_activation_function(PerceptronLayer::Logistic);

      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);

//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);


      perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);

      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);

//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);


      perceptron_layer.set_activation_function(PerceptronLayer::Linear);

      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);

//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      perceptron_layer.set(4, 2);

      parameters.resize(10);
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

      inputs.resize(1,4);

      inputs(0, 0) = 0.85;
      inputs(0, 1) = -0.25;
      inputs(0, 2) = 0.29;
      inputs(0, 3) = -0.77;

      combinations = perceptron_layer.calculate_combinations(inputs);

      perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::Linear);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);
   }
   */
}


void PerceptronLayerTest::test_calculate_outputs()
{

    PerceptronLayer perceptron_layer;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> biases;

    Tensor<type, 2> inputs;
    Tensor<type, 1> parameters;

    perceptron_layer.set(3, 4);

    synaptic_weights.resize(3, 4);
    biases.resize(1, 4);
    inputs.resize(1, 3);

    inputs.setConstant(1);
    biases.setConstant(1);
    synaptic_weights.setValues({{1,1,1,1},
                               {2,2,2,2},
                               {3,3,3,3}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    perceptron_layer.set_activation_function(PerceptronLayer::Linear);

    Tensor<type,2>outputs = perceptron_layer.calculate_outputs(inputs);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(outputs(0,0) == 7, LOG);

/*

    biases(0,0) = static_cast<type>(0.85);
    biases(0,1) = -static_cast<type>(0.25);
    biases(0,2) = static_cast<type>(0.29);
    biases(0,3) = -static_cast<type>(0.77)*/;

//    synaptic_weights(0,0) = -static_cast<type>(0.04);
//    synaptic_weights(1,0) = static_cast<type>(0.87);
//    synaptic_weights(0,1) = static_cast<type>(0.25);
//    synaptic_weights(1,1) = -static_cast<type>(0.27);
//    synaptic_weights(0,2) = -static_cast<type>(0.57);
//    synaptic_weights(1,2) = static_cast<type>(0.15);
//    synaptic_weights(0,3) = static_cast<type>(0.96);
//    synaptic_weights(1,3) = -static_cast<type>(0.48);

//    perceptron_layer.set_synaptic_weights(synaptic_weights);
//    perceptron_layer.set_biases(biases);




    /*
    cout << "test_calculate_activation_derivative\n";

   PerceptronLayer perceptron_layer;

   Tensor<type, 1> parameters;
   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;
   Tensor<type, 2> potential_outputs;

   // Test 

   perceptron_layer.set(3, 2);
   perceptron_layer.set_parameters_constant(0.0);

   inputs.resize(1,3);
   inputs.setConstant(0.0);

   outputs = perceptron_layer.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);
//   assert_true(outputs == 0.0, LOG);

   // Test

   perceptron_layer.set(4, 2);

   parameters.resize(10);
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

   inputs.resize(1,4);
   inputs(0, 0) = 0.85;
   inputs(0, 1) = -0.25;
   inputs(0, 2) = 0.29;
   inputs(0, 3) = -0.77;

   outputs = perceptron_layer.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);

   // Test

   inputs.resize(1,1);
   inputs.setConstant((3.0));

   perceptron_layer.set(1, 1);

   perceptron_layer.set_parameters_constant(2.0);

   outputs = perceptron_layer.calculate_outputs(inputs);

   parameters.resize(2);

   parameters.setConstant(1.0);

   potential_outputs = perceptron_layer.calculate_outputs(inputs, parameters);

//   assert_true(outputs != potential_outputs, LOG);

   // Test

   perceptron_layer.set(1, 1);

   inputs.resize(1,1);
   inputs.setRandom();

   parameters = perceptron_layer.get_parameters();

//   assert_true(perceptron_layer.calculate_outputs(inputs) == perceptron_layer.calculate_outputs(inputs, parameters), LOG);

*/
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

   // Parameters initialization methods

   test_set_parameters_constant();

   test_initialize_biases();

   test_initialize_synaptic_weights();

   test_set_parameters_random();

   // Parameters initialization methods

   test_set_parameters_constant();

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
