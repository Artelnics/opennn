//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer_test.h"
#include "perceptron_layer.h"

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
    assert_true(perceptron_layer_2.get_biases_number() == 3, LOG);
    assert_true(perceptron_layer_2.get_synaptic_weights_number() == 30, LOG);
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

   // Test 0
   perceptron_layer.set();
   assert_true(perceptron_layer.get_inputs_number() == 0, LOG);

   // Test 1
   perceptron_layer.set(1, 1);
   assert_true(perceptron_layer.get_inputs_number() == 1, LOG);
}

void PerceptronLayerTest::test_get_neurons_number()
{
   cout << "test_get_size\n";

   PerceptronLayer perceptron_layer;

   // Test 0
   perceptron_layer.set();
   assert_true(perceptron_layer.get_neurons_number() == 0, LOG);

   // Test 1
   perceptron_layer.set(1, 1);
   assert_true(perceptron_layer.get_neurons_number() == 1, LOG);
}

void PerceptronLayerTest::test_get_activation_function()
{
   cout << "test_get_activation_function\n";

   PerceptronLayer perceptron_layer;

   // Test 0
   perceptron_layer.set();

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::Threshold, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::SymmetricThreshold, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::Logistic, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::HyperbolicTangent, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::Linear, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::RectifiedLinear);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::RectifiedLinear, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::ExponentialLinear);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ExponentialLinear, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::ScaledExponentialLinear);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ScaledExponentialLinear, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SoftPlus);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::SoftPlus, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SoftSign);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::SoftSign, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HardSigmoid);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::HardSigmoid, LOG);

   // Test 1
   perceptron_layer.set(1, 1);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::Threshold, LOG);
}

void PerceptronLayerTest::test_write_activation_function()
{
   cout << "test_write_activation_function\n";

   PerceptronLayer perceptron_layer;

   // Test 0
   perceptron_layer.set();

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   assert_true(perceptron_layer.write_activation_function() == "Threshold", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   assert_true(perceptron_layer.write_activation_function() == "SymmetricThreshold", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   assert_true(perceptron_layer.write_activation_function() == "Logistic", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   assert_true(perceptron_layer.write_activation_function() == "HyperbolicTangent", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   assert_true(perceptron_layer.write_activation_function() == "Linear", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::RectifiedLinear);
   assert_true(perceptron_layer.write_activation_function() == "RectifiedLinear", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::ExponentialLinear);
   assert_true(perceptron_layer.write_activation_function() == "ExponentialLinear", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SoftPlus);
   assert_true(perceptron_layer.write_activation_function() == "SoftPlus", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SoftSign);
   assert_true(perceptron_layer.write_activation_function() == "SoftSign", LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HardSigmoid);
   assert_true(perceptron_layer.write_activation_function() == "HardSigmoid", LOG);

   // Test 1
   perceptron_layer.set(1, 1);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   assert_true(perceptron_layer.write_activation_function() == "Threshold", LOG);

}

void PerceptronLayerTest::test_get_parameters_number()
{
   cout << "test_get_parameters_number\n";

   PerceptronLayer perceptron_layer;

   // Test 0
   perceptron_layer.set();
   assert_true(perceptron_layer.get_parameters_number() == 0, LOG);

   // Test 1
   perceptron_layer.set(3, 1);
   assert_true(perceptron_layer.get_parameters_number() == 4, LOG);

   // Test 2
   perceptron_layer.set(2, 4);
   assert_true(perceptron_layer.get_parameters_number() == 12, LOG);
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

   // Test  0
   perceptron_layer.set();
   perceptron_layer.set_parameters_constant(0);
   biases = perceptron_layer.get_biases();

   assert_true(biases.size() == 0, LOG);
//   assert_true(biases(0) < numeric_limits<type>::min(), LOG);

   // Test 1
   perceptron_layer.set(1, 1);
   perceptron_layer.set_parameters_constant(0);
   biases = perceptron_layer.get_biases();

   assert_true(biases.size() == 1, LOG);
//   assert_true(biases(0) < numeric_limits<type>::min(), LOG);

   cout << "test_get_biases_with_parameters\n";

   PerceptronLayer perceptron_layer_2;

   // Test  1
   Tensor<type, 2> biases_2(1, 4);
   biases_2.setValues({{9},{8},{7},{6}});

   Tensor<type, 2> synaptic_weights(2, 4);
   synaptic_weights.setValues({{11, 12, 13, 14},{21, 22, 23, 24}});

   Tensor<type, 1> parameters(12);
   perceptron_layer_2.set(2, 4);

   perceptron_layer_2.set_synaptic_weights(synaptic_weights);
   perceptron_layer_2.set_biases(biases_2);

   parameters = perceptron_layer_2.get_parameters();
   biases = perceptron_layer_2.get_biases(parameters);

   assert_true(biases.size() == 4, LOG);
   assert_true(abs(biases(0,0) - 9) < static_cast<type>(1e-5), LOG);
   assert_true(abs(biases(0,3) - 6) < static_cast<type>(1e-5), LOG);
}

void PerceptronLayerTest::test_get_synaptic_weights()
{
   cout << "test_get_synaptic_weights\n";

   PerceptronLayer perceptron_layer;
   Tensor<type, 2> synaptic_weights;

   // Test 0
   perceptron_layer.set(1, 1);

   perceptron_layer.set_parameters_constant(0.0);

   synaptic_weights = perceptron_layer.get_synaptic_weights();

   assert_true(synaptic_weights.dimension(0) == 1, LOG);
   assert_true(synaptic_weights.dimension(1) == 1, LOG);
   assert_true(synaptic_weights(0,0) < numeric_limits<type>::min(), LOG);

   cout << "test_get_synaptic_weight_with_parameters\n";

   PerceptronLayer perceptron_layer_2;

   // Test 1
   Tensor<type, 2> biases_2(1, 4);
   biases_2.setValues({{9},{-8},{7},{-6}});

   Tensor<type, 2> synaptic_weights_2(2, 4);
   synaptic_weights_2.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

   Tensor<type, 1> parameters(12);
   perceptron_layer_2.set(2, 4);

   perceptron_layer_2.set_synaptic_weights(synaptic_weights_2);
   perceptron_layer_2.set_biases(biases_2);

   parameters = perceptron_layer_2.get_parameters();
   synaptic_weights_2 = perceptron_layer_2.get_synaptic_weights(parameters);

   assert_true(synaptic_weights_2.size() == 8, LOG);

   assert_true(abs(synaptic_weights_2(0,0) + 11) < static_cast<type>(1e-5), LOG);
   assert_true(abs(synaptic_weights_2(1,3) + 24) < static_cast<type>(1e-5), LOG);
}

void PerceptronLayerTest::test_get_parameters()
{

   cout << "test_get_parameters\n";

   PerceptronLayer perceptron_layer;
   Tensor<type, 2> synaptic_weights;
   Tensor<type, 1> parameters;

   // Test 0
   perceptron_layer.set(1, 1);
   perceptron_layer.set_parameters_constant(1.0);

   parameters = perceptron_layer.get_parameters();

   assert_true(parameters.size() == 2, LOG);
   assert_true(abs(parameters(0) - 1) < numeric_limits<type>::min(), LOG);

   // Test 1
   PerceptronLayer perceptron_layer_2;

   Tensor<type, 2> biases_2(1, 4);
   biases_2.setValues({{9},{-8},{7},{-6}});

   Tensor<type, 2> synaptic_weights_2(2, 4);
   synaptic_weights_2.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

   perceptron_layer_2.set_synaptic_weights(synaptic_weights_2);
   perceptron_layer_2.set_biases(biases_2);

   Tensor<type,1>new_parameters = perceptron_layer_2.get_parameters();

   assert_true(new_parameters.size() == 12, LOG);
   assert_true(abs(new_parameters(0) - 9) < static_cast<type>(1e-5), LOG);
   assert_true(abs(new_parameters(4) + 11) < static_cast<type>(1e-5), LOG);
   assert_true(abs(new_parameters(7) + 22) < static_cast<type>(1e-5), LOG);
   }

void PerceptronLayerTest::test_set_biases()
{

   cout << "test_set_biases\n";

    PerceptronLayer perceptron_layer;

    Tensor<type, 2> biases(1, 4);

    // Test 0
    perceptron_layer.set(1, 4);

    biases.setZero();

    perceptron_layer.set_biases(biases);

    assert_true(perceptron_layer.get_biases_number() == 4, LOG);

    assert_true(abs(perceptron_layer.get_biases()(0)) < static_cast<type>(1e-5), LOG);
    assert_true(abs(perceptron_layer.get_biases()(3)) < static_cast<type>(1e-5), LOG);

    // Test 1
    perceptron_layer.set(1, 4);

    biases.setValues({{9},{-8},{7},{-6}});

    perceptron_layer.set_biases(biases);

    assert_true(perceptron_layer.get_biases_number() == 4, LOG);

    assert_true(abs(perceptron_layer.get_biases()(0) - 9) < static_cast<type>(1e-5), LOG);
    assert_true(abs(perceptron_layer.get_biases()(3) + 6) < static_cast<type>(1e-5), LOG);
}

void PerceptronLayerTest::test_set_synaptic_weights()
{
   cout << "test_set_synaptic_weights\n";

    PerceptronLayer perceptron_layer(1, 2);

    Tensor<type, 2> synaptic_weights(2, 1);

    // Test 0
    synaptic_weights.setZero();

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    assert_true(perceptron_layer.get_synaptic_weights().size() == 2, LOG);

    assert_true(abs(perceptron_layer.get_synaptic_weights()(0)) < static_cast<type>(1e-5), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(1)) < static_cast<type>(1e-5), LOG);

    // Test 1
    synaptic_weights.setValues({{-11},{21}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    assert_true(perceptron_layer.get_synaptic_weights().size() == 2, LOG);

    assert_true(abs(perceptron_layer.get_synaptic_weights()(0) + 11) < static_cast<type>(1e-5), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(1) - 21) < static_cast<type>(1e-5), LOG);
}

void PerceptronLayerTest::test_set_inputs_number()
{

   cout << "test_set_inputs_number\n";

    PerceptronLayer perceptron_layer;
    Tensor<type, 2> new_biases;
    Tensor<type, 2> new_synaptic_weights;

    //Test 0
    Tensor<type, 2> biases;
    Tensor<type, 2> synaptic_weights;

    perceptron_layer.set_biases(biases);
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_inputs_number = 0;

    perceptron_layer.set_inputs_number(new_inputs_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() == new_synaptic_weights.size(), LOG);

    //Test 1
    perceptron_layer.set(2, 2);

    Tensor<type, 2> biases_2(1, 4);
    biases_2.setValues({{9},{-8},{7},{-6}});

    Tensor<type, 2> synaptic_weights_2(2, 4);
    synaptic_weights_2.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

    perceptron_layer.set_biases(biases_2);
    perceptron_layer.set_synaptic_weights(synaptic_weights_2);

    Index new_inputs_number_2 = 6;

    perceptron_layer.set_inputs_number(new_inputs_number_2);

    Tensor<type, 2>new_biases_2 = perceptron_layer.get_biases();
    Tensor<type, 2>new_synaptic_weights_2 = perceptron_layer.get_synaptic_weights();

    assert_true(biases_2.size() == new_biases_2.size(), LOG);
    assert_true(synaptic_weights_2.size() != new_synaptic_weights_2.size(), LOG);
}

void PerceptronLayerTest::test_set_perceptrons_number()
{
   cout << "test_set_perceptrons_number\n";

    PerceptronLayer perceptron_layer;
    Tensor<type, 2> new_biases;
    Tensor<type, 2> new_synaptic_weights;

    perceptron_layer.set(3, 2);

    Tensor<type, 2> biases(1, 4);
    biases.setValues({{9},{-8},{7},{-6}});

    Tensor<type, 2> synaptic_weights(2, 4);
    synaptic_weights.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_perceptrons_number = 1;

    perceptron_layer.set_neurons_number(new_perceptrons_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() != new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
}

void PerceptronLayerTest::test_set_parameters()
{
  cout << "test_set_parameters\n";

    PerceptronLayer perceptron_layer;

    //Test
    perceptron_layer.set(1, 2);

    Tensor<type, 1> parameters_2(4);

    parameters_2.setValues({11,12,21,22});

    perceptron_layer.set_parameters(parameters_2);

    assert_true(perceptron_layer.get_biases()(0) - parameters_2(0) < static_cast<type>(1e-5), LOG);
    assert_true(perceptron_layer.get_synaptic_weights()(0) - parameters_2(2)  < static_cast<type>(1e-5), LOG);
}

void PerceptronLayerTest::test_get_display()
{
   cout << "test_get_display\n";
}

void PerceptronLayerTest::test_set_activation_function()
{
   cout << "test_set_activation_function\n";

   PerceptronLayer perceptron_layer;

   // Test
   string new_activation_function_name = "Threshold";
   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   assert_true(perceptron_layer.get_activation_function() == 0, LOG);

   new_activation_function_name = "SymmetricThreshold";
   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   assert_true(perceptron_layer.get_activation_function() == 1, LOG);

   new_activation_function_name = "Logistic";
   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   assert_true(perceptron_layer.get_activation_function() == 2, LOG);

   new_activation_function_name = "HyperbolicTangent";
   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   assert_true(perceptron_layer.get_activation_function() == 3, LOG);

   new_activation_function_name = "Linear";
   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   assert_true(perceptron_layer.get_activation_function() == 4, LOG);

   new_activation_function_name = "RectifiedLinear";
   perceptron_layer.set_activation_function(PerceptronLayer::RectifiedLinear);
   assert_true(perceptron_layer.get_activation_function() == 5, LOG);

   new_activation_function_name = "ExponentialLinear";
   perceptron_layer.set_activation_function(PerceptronLayer::ExponentialLinear);
   assert_true(perceptron_layer.get_activation_function() == 6, LOG);

   new_activation_function_name = "ScaledExponentialLinear";
   perceptron_layer.set_activation_function(PerceptronLayer::ScaledExponentialLinear);
   assert_true(perceptron_layer.get_activation_function() == 7, LOG);

   new_activation_function_name = "SoftPlus";
   perceptron_layer.set_activation_function(PerceptronLayer::SoftPlus);
   assert_true(perceptron_layer.get_activation_function() == 8, LOG);

   new_activation_function_name = "SoftSign";
   perceptron_layer.set_activation_function(PerceptronLayer::SoftSign);
   assert_true(perceptron_layer.get_activation_function() == 9, LOG);

   new_activation_function_name = "HardSigmoid";
   perceptron_layer.set_activation_function(PerceptronLayer::HardSigmoid);
   assert_true(perceptron_layer.get_activation_function() == 10, LOG);
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
   perceptron_layer.set_parameters_constant(0);

   parameters = perceptron_layer.get_parameters();

   assert_true(static_cast<Index>(parameters(0)) == 0, LOG);
   assert_true(static_cast<Index>(parameters(1)) == 0, LOG);
}

void PerceptronLayerTest::test_set_synaptic_weights_constant()
{
   cout << "test_set_synaptic_weights_constant\n";

   PerceptronLayer perceptron_layer;

   Tensor<type, 2> synaptic_weights;
   Tensor<type, 2> biases;

   // Test
   perceptron_layer.set(1, 1);
   perceptron_layer.set_synaptic_weights_constant(0);

   synaptic_weights = perceptron_layer.get_synaptic_weights();

   assert_true(static_cast<Index>(synaptic_weights(0,0)) == 0, LOG);
}

void PerceptronLayerTest::test_set_parameters_random()
{
   cout << "test_set_parameters_random\n";

   PerceptronLayer perceptron_layer;
   Tensor<type, 1> parameters;;

   // Test
   perceptron_layer.set(1,1);
   perceptron_layer.set_parameters_random();

   parameters = perceptron_layer.get_parameters();

   assert_true((parameters(0) >= static_cast<type>(1e-5)) && (parameters(0) >= static_cast<type>(1e-5)), LOG);
   assert_true((parameters(1) >= static_cast<type>(1e-5)) && (parameters(1) >= static_cast<type>(1e-5)), LOG);
}

void PerceptronLayerTest::test_calculate_combinations()
{

   cout << "test_calculate_combinations\n";

   PerceptronLayer perceptron_layer;

   Tensor<type, 2> biases(1,1);
   Tensor<type, 2> synaptic_weights(1,1);
   Tensor<type, 1> parameters(1);

   Tensor<type, 2> inputs(1,1);
   Tensor<type, 2> combinations_2d(1,1);

   Device device(Device::EigenSimpleThreadPool);
   perceptron_layer.set_device_pointer(&device);

   // Test 0

   biases.setConstant(1.0);
   synaptic_weights.setConstant(2.0);

   perceptron_layer.set(1,1);
   inputs.setConstant(3.0);

   perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations_2d);

   assert_true(combinations_2d.rank() == 2, LOG);
   assert_true(combinations_2d.dimension(0) == 1, LOG);
   assert_true(combinations_2d.dimension(1) == 1, LOG);
   assert_true(abs(combinations_2d(0,0) - 7) < static_cast<type>(1e-5) , LOG);

    // Test 1

   combinations_2d.resize(1, 2);
   combinations_2d.setZero();

   perceptron_layer.set(2, 2);
   perceptron_layer.set_parameters_constant(1);

   inputs.resize(1,2);
   inputs.setConstant(1.0);

   perceptron_layer.calculate_combinations(inputs, perceptron_layer.get_biases(), perceptron_layer.get_synaptic_weights(), combinations_2d);

   assert_true(combinations_2d.rank() == 2, LOG);
   assert_true(combinations_2d.dimension(0) == 1, LOG);
   assert_true(combinations_2d.dimension(1) == 2, LOG);
   assert_true(abs(combinations_2d(0,0) - 3) < static_cast<type>(1e-5), LOG);

   //Test 2

   combinations_2d.resize(2, 4);
   combinations_2d.setZero();

   perceptron_layer.set(3,4);

   synaptic_weights.resize(3,4);
   synaptic_weights.setConstant(1.0);
   biases.resize(1,4);
   biases.setConstant(2.0);

   inputs.resize(2,3);
   inputs.setConstant(0.5);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   perceptron_layer.calculate_combinations(inputs, perceptron_layer.get_biases(), perceptron_layer.get_synaptic_weights(), combinations_2d);

   assert_true(combinations_2d.rank() == 2, LOG);
   assert_true(combinations_2d.dimension(0) == 2, LOG);
   assert_true(combinations_2d.dimension(1) == 4, LOG);
   assert_true(abs(combinations_2d(0,0) - static_cast<type>(3.5)) < static_cast<type>(1e-5), LOG);

   // Test 3

   combinations_2d.resize(1, 4);
   combinations_2d.setZero();

   perceptron_layer.set(2, 4);

   synaptic_weights.resize(2,4);
   synaptic_weights.setConstant(1.0);
   biases.resize(1,4);
   biases.setConstant(1.0);

   perceptron_layer.set_synaptic_weights(synaptic_weights);
   perceptron_layer.set_biases(biases);

   inputs.resize(1,2);
   inputs.setValues({{0.5, 0.5}});

   perceptron_layer.calculate_combinations(inputs, perceptron_layer.get_biases(), perceptron_layer.get_synaptic_weights(), combinations_2d);

   assert_true(combinations_2d.rank() == 2, LOG);
   assert_true(combinations_2d.dimension(0) == 1, LOG);
   assert_true(combinations_2d.dimension(1) == 4, LOG);
   assert_true(static_cast<Index>(combinations_2d(0,0)) == 2, LOG);
}

void PerceptronLayerTest::test_calculate_activations()
{

   cout << "test_calculate_activations\n";

   PerceptronLayer perceptron_layer;

   Tensor<type, 2> biases(1,1);
   Tensor<type, 2> synaptic_weights(1,1);
   Tensor<type, 1> parameters(1);

   Tensor<type, 2> inputs(1,1);
   Tensor<type, 2> combinations_2d(1,1);
   Tensor<type, 2> activations_2d(1,1);

   Device device(Device::EigenSimpleThreadPool);
   perceptron_layer.set_device_pointer(&device);

   // Test 1

   perceptron_layer.set(1,1);

   biases.setConstant(1.0);
   synaptic_weights.setConstant(1.0);

   inputs.setConstant(1);

   perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations_2d);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);

   perceptron_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 1, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 2 , LOG);

   // Test 2

   perceptron_layer.set(2, 2);
   perceptron_layer.set_parameters_constant(2);

   combinations_2d.resize(1,2);
   combinations_2d.setZero();

   activations_2d.resize(1,2);
   activations_2d.setZero();

   inputs.resize(1,2);
   inputs.setConstant(2);

   perceptron_layer.calculate_combinations(inputs, perceptron_layer.get_biases(), perceptron_layer.get_synaptic_weights(), combinations_2d);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);

   perceptron_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 10, LOG);

   // Test 3

   perceptron_layer.set(2, 2);
   parameters.resize(6);

   parameters.setConstant(0.0);

   combinations_2d.resize(1,2);
   combinations_2d.setConstant(0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);

   perceptron_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 0, LOG);

   // Test 4

   perceptron_layer.set(1, 2);
   parameters.resize(4);

   parameters.setConstant(0.0);

   combinations_2d.resize(1,2);
   combinations_2d.setConstant(0.0);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);

   perceptron_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 1 , LOG);

   // Test 5

   perceptron_layer.set(2, 2);
   parameters.resize(6);

   parameters.setConstant(0.0);

   combinations_2d.resize(1,2);
   combinations_2d.setConstant(-2.0);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);

   perceptron_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0))  == -1, LOG);

   // Test 6

   perceptron_layer.set(1, 2);
   perceptron_layer.set_parameters_constant(0.0);

   combinations_2d.resize(2,2);
   combinations_2d.setConstant(4.0);

   activations_2d.resize(2,2);
   activations_2d.setZero();

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);

   perceptron_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 2, LOG);
   assert_true(activations_2d.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 4.0, LOG);

   // Test 7

   perceptron_layer.set(3, 2);
   perceptron_layer.set_parameters_constant(1);

   inputs.resize(1,3);
   inputs.setConstant(0.5);

   combinations_2d.resize(1,2);
   activations_2d.resize(1,2);

   perceptron_layer.calculate_combinations(inputs, perceptron_layer.get_biases(), perceptron_layer.get_synaptic_weights(), combinations_2d);

   assert_true(combinations_2d.rank() == 2, LOG);
   assert_true(combinations_2d.dimension(0) == 1, LOG);
   assert_true(combinations_2d.dimension(1) == 2, LOG);
   assert_true(abs(combinations_2d(0,0) - static_cast<type>(2.5)) < static_cast<type>(1e-5), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);

   perceptron_layer.calculate_activations(combinations_2d, activations_2d);
   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 1, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   perceptron_layer.calculate_activations(combinations_2d, activations_2d);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 1, LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   perceptron_layer.calculate_activations(combinations_2d, activations_2d);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(1.0/(1.0+exp(-2.5)))) < static_cast<type>(1e-5), LOG);


   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   perceptron_layer.calculate_activations(combinations_2d, activations_2d);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(tanh(2.5))) < static_cast<type>(1e-5), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   perceptron_layer.calculate_activations(combinations_2d, activations_2d);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(2.5)) < static_cast<type>(1e-5), LOG);
}

void PerceptronLayerTest::test_calculate_activations_derivatives()
{

   cout << "test_calculate_activations_derivatives\n";

   NumericalDifferentiation numerical_differentiation;
   PerceptronLayer perceptron_layer;

   Tensor<type, 1> parameters(1);
   Tensor<type, 2> inputs(1,1);
   Tensor<type, 2> combinations_2d(1,1);
   Tensor<type, 2> activations_2d(1,1);
   Tensor<type, 2> activations_derivatives(1,1);

   Device device(Device::EigenSimpleThreadPool);
   perceptron_layer.set_device_pointer(&device);

   // Test 1

   perceptron_layer.set(1,1);
   perceptron_layer.set_parameters_constant(1);

   inputs.setConstant(1);

   combinations_2d.setConstant(1);

   activations_derivatives.setZero();

   //@todo fail logistic_derivatives perceptron layer - dimension <type, 2> <-> <type, 3>

   perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

   assert_true(activations_derivatives.rank() == 2, LOG);
   assert_true(activations_derivatives.dimension(0) == 1, LOG);
   assert_true(activations_derivatives.dimension(1) == 1, LOG);
   assert_true(abs(activations_2d(0,0) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(activations_derivatives(0,0) - 0) < static_cast<type>(1e-5), LOG);


   perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - 0) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.196)) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(0.761)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.41997)) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::Linear);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(activations_derivatives(0,0) - 1) < static_cast<type>(1e-5), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::RectifiedLinear);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::ExponentialLinear);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::ScaledExponentialLinear);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(1.05)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(1.05)) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SoftPlus);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(1.313)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::SoftSign);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);

   perceptron_layer.set_activation_function(PerceptronLayer::HardSigmoid);
   perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(0.7)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.2)) < static_cast<type>(1e-3), LOG);

   numerical_differentiation_tests = true;

   // Test 2

   if(numerical_differentiation_tests)
   {
      perceptron_layer.set(2, 4);
      perceptron_layer.set_parameters_constant(1);

      combinations_2d.resize(1,4);
      combinations_2d.setValues({{1.56f, -0.68f, 0.91f, -1.99f}});

      activations_2d.resize(1,4);

      activations_derivatives.resize(1,4);
      activations_derivatives.setZero();

      // Test 2_1
      perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_differentiation.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);
      Tensor<type, 2> numerical_activation_derivative(1,4);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_2
      perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_3
      perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_4
      perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_5
      perceptron_layer.set_activation_function(PerceptronLayer::Linear);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_6
      perceptron_layer.set_activation_function(PerceptronLayer::RectifiedLinear);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_7
      perceptron_layer.set_activation_function(PerceptronLayer::ExponentialLinear);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_8
      perceptron_layer.set_activation_function(PerceptronLayer::ScaledExponentialLinear);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_9
      perceptron_layer.set_activation_function(PerceptronLayer::SoftPlus);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_10
      perceptron_layer.set_activation_function(PerceptronLayer::SoftSign);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

      // Test 2_11
      perceptron_layer.set_activation_function(PerceptronLayer::HardSigmoid);
      perceptron_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations_2d);

      assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);
   }

   // Test
/*
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

      combinations_2d = perceptron_layer.calculate_combinations(inputs);

      perceptron_layer.set_activation_function(PerceptronLayer::Threshold);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations_2d);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations_2d);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::SymmetricThreshold);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations_2d);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations_2d);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::Logistic);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations_2d);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations_2d);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::HyperbolicTangent);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations_2d);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations_2d);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);

      perceptron_layer.set_activation_function(PerceptronLayer::Linear);
      activations_derivatives = perceptron_layer.calculate_activations_derivatives(combinations_2d);
      numerical_activation_derivative = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, combinations_2d);
//      assert_true((activations_derivatives - numerical_activation_derivative).abs() < 1.0e-3, LOG);
   }
*/

}

void PerceptronLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    PerceptronLayer perceptron_layer;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> biases;

    Tensor<type, 2> inputs;
    Tensor<type, 1> parameters;

    Device device(Device::EigenSimpleThreadPool);
    perceptron_layer.set_device_pointer(&device);

    // Test 1

    perceptron_layer.set(3, 4);

    synaptic_weights.resize(3, 4);
    biases.resize(1, 4);
    inputs.resize(1, 3);

    inputs.setConstant(1);
    biases.setConstant(1);
    synaptic_weights.setValues({{1,-1,0,1},{2,-2,0,2},{3,-3,0,3}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    perceptron_layer.set_activation_function(PerceptronLayer::Linear);

    Tensor<type,2>outputs = perceptron_layer.calculate_outputs(inputs);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index >(outputs(0,0)) == 7, LOG);
    assert_true(static_cast<Index >(outputs(1,0)) == -5, LOG);
    assert_true(static_cast<Index >(outputs(2,0)) == 1, LOG);

    // Test 2

    Tensor<type, 2> biases_2(1, 4);
    biases.setValues({{9},{-8},{7},{-6}});

    Tensor<type, 2> synaptic_weights_2(2, 4);

    synaptic_weights.resize(2, 4);
    synaptic_weights.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    inputs.resize(1, 2);
    inputs.setConstant(1);

    perceptron_layer.set_activation_function(PerceptronLayer::Threshold);

    outputs = perceptron_layer.calculate_outputs(inputs);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index >(outputs(0,0)) == 1, LOG);
    assert_true(static_cast<Index >(outputs(1,0)) == 0, LOG);
    assert_true(static_cast<Index >(outputs(2,0)) == 1, LOG);

   // Test  3

   perceptron_layer.set(3, 2);
   perceptron_layer.set_parameters_constant(0.0);

   inputs.resize(1,3);
   inputs.setConstant(0.0);

   outputs = perceptron_layer.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);
   assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-5), LOG);

   // Test 4

   perceptron_layer.set(4, 2);
   parameters.resize(10);
   parameters.setValues({-1,2,-3,4,-5,6,-7,8,-9,10});
   perceptron_layer.set_parameters(parameters);

   inputs.resize(1,4);
   inputs.setValues({{4,-3,2,-1}});

   outputs = perceptron_layer.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);
   assert_true(abs(outputs(0,0) + 1) < static_cast<type>(1e-5), LOG);

   // Test 5

   inputs.resize(1,1);
   inputs.setConstant((3.0));

   perceptron_layer.set(1, 1);

   perceptron_layer.set_parameters_constant(-2.0);

   outputs = perceptron_layer.calculate_outputs(inputs);

   parameters.resize(2);

   parameters.setConstant(1.0);

   Tensor<type,2>potential_outputs = perceptron_layer.calculate_outputs(inputs, parameters);

   assert_true(abs(outputs(0,0) - potential_outputs(0,0)) > static_cast<type>(1e-3), LOG);

   // Test

   perceptron_layer.set(1, 1);

   inputs.resize(1,1);
   inputs.setRandom();

   parameters = perceptron_layer.get_parameters();

   assert_true(abs(perceptron_layer.calculate_outputs(inputs)(0,0) - perceptron_layer.calculate_outputs(inputs, parameters)(0,0)) < static_cast<type>(1e-3), LOG);
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


   // Inputs and perceptrons

   test_get_inputs_number();

   test_get_neurons_number();


   // Parameters

   test_get_parameters_number();

   test_get_biases();

   test_get_synaptic_weights();

   test_get_parameters();


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

   test_set_synaptic_weights_constant();

   test_set_parameters_random();


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
