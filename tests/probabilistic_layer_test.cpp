//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "probabilistic_layer_test.h"
#include <omp.h>

ProbabilisticLayerTest::ProbabilisticLayerTest() : UnitTesting()
{

}


ProbabilisticLayerTest::~ProbabilisticLayerTest()
{

}


void ProbabilisticLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default constructor

   ProbabilisticLayer probabilistic_layer_l1;

   assert_true(probabilistic_layer_l1.get_inputs_number() == 0, LOG);
   assert_true(probabilistic_layer_l1.get_neurons_number() == 0, LOG);
   assert_true(probabilistic_layer_l1.get_biases_number() == 0, LOG);
   assert_true(probabilistic_layer_l1.get_synaptic_weights_number() == 0, LOG);
   assert_true(probabilistic_layer_l1.get_parameters_number() == 0, LOG);


   // Probabilistic neurons number constructor

   ProbabilisticLayer probabilistic_layer_l2;

   probabilistic_layer_l2.set_neurons_number(3);

   assert_true(probabilistic_layer_l2.get_inputs_number() == 0, LOG);
   assert_true(probabilistic_layer_l2.get_neurons_number() == 3, LOG);
   assert_true(probabilistic_layer_l2.get_biases_number() == 3, LOG);
   assert_true(probabilistic_layer_l2.get_synaptic_weights_number() == 0, LOG);
   assert_true(probabilistic_layer_l2.get_parameters_number() == 3, LOG);

   // Copy constructor

   ProbabilisticLayer probabilistic_layer_l3(probabilistic_layer_l2);

   probabilistic_layer_l3.set_neurons_number(probabilistic_layer_l2.get_neurons_number());

   assert_true(probabilistic_layer_l3.get_inputs_number() == 0, LOG);
   assert_true(probabilistic_layer_l3.get_neurons_number() == 3, LOG);
   assert_true(probabilistic_layer_l3.get_biases_number() == 3, LOG);
   assert_true(probabilistic_layer_l3.get_synaptic_weights_number() == 0, LOG);
   assert_true(probabilistic_layer_l3.get_parameters_number() == 3, LOG);
}

void ProbabilisticLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}

void ProbabilisticLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   ProbabilisticLayer probabilistic_layer_l1;
   ProbabilisticLayer probabilistic_layer_l2;

   probabilistic_layer_l1.set_neurons_number(2);
   probabilistic_layer_l2.set_neurons_number(probabilistic_layer_l1.get_neurons_number());

   assert_true(probabilistic_layer_l2.get_inputs_number() == 0, LOG);
   assert_true(probabilistic_layer_l2.get_neurons_number() == 2, LOG);
   assert_true(probabilistic_layer_l2.get_biases_number() == 2, LOG);
   assert_true(probabilistic_layer_l2.get_synaptic_weights_number() == 0, LOG);
   assert_true(probabilistic_layer_l2.get_parameters_number() == 2, LOG);
}

void ProbabilisticLayerTest::test_get_inputs_number()
{
   cout << "test_get_inputs_number\n";

   ProbabilisticLayer probabilistic_layer;

   // Test 0
   probabilistic_layer.set();
   assert_true(probabilistic_layer.get_inputs_number() == 0, LOG);

   // Test 1
   probabilistic_layer.set_inputs_number(1);
   assert_true(probabilistic_layer.get_inputs_number() == 1, LOG);
}

void ProbabilisticLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   ProbabilisticLayer probabilistic_layer;

   // Test 0
   probabilistic_layer.set();
   assert_true(probabilistic_layer.get_neurons_number() == 0, LOG);

   // Test 1
   probabilistic_layer.set_neurons_number(1);
   assert_true(probabilistic_layer.get_neurons_number() == 1, LOG);
}

void ProbabilisticLayerTest::test_get_biases()
{
   cout << "test_get_biases\n";

   ProbabilisticLayer probabilistic_layer;
   Index biases_numb;
   Tensor<type, 2> biases;

   // Test 0
   probabilistic_layer.set();
   probabilistic_layer.set_parameters_constant(0);

   biases_numb = probabilistic_layer.get_biases_number();
   biases = probabilistic_layer.get_biases();

   assert_true(biases_numb == 0, LOG);
   assert_true(biases.size() == 0, LOG);

   // Test 1
   probabilistic_layer.set(1, 1);
   probabilistic_layer.set_parameters_constant(0);

   biases_numb = probabilistic_layer.get_biases_number();
   biases = probabilistic_layer.get_biases();

   assert_true(biases_numb == 1, LOG);
   assert_true(biases.size() == 1, LOG);
   assert_true(biases(0) < numeric_limits<type>::min(), LOG);


   cout << "test_get_biases_with_parameters\n";

   ProbabilisticLayer probabilistic_layer_2;

   // Test  1
   Tensor<type, 2> biases_2(1, 4);
   biases_2.setValues({{9},{8},{7},{6}});

   Tensor<type, 2> synaptic_weights(2, 4);
   synaptic_weights.setValues({{11, 12, 13, 14},{21, 22, 23, 24}});

   Tensor<type, 1> parameters(12);
   probabilistic_layer_2.set(2, 4);

   probabilistic_layer.set_synaptic_weights(synaptic_weights);
   probabilistic_layer_2.set_biases(biases_2);

   parameters = probabilistic_layer_2.get_parameters();
   biases = probabilistic_layer_2.get_biases(parameters);

   assert_true(biases.size() == 4, LOG);
   assert_true(abs(biases(0,0) - 9) < static_cast<type>(1e-5), LOG);
   assert_true(abs(biases(0,3) - 6) < static_cast<type>(1e-5), LOG);
}

void ProbabilisticLayerTest::test_get_synaptic_weights()
{
    cout << "test_get_synaptic_weights\n";

    ProbabilisticLayer probabilistic_layer;
    Tensor<type, 2> synaptic_weights;

    // Test 0
    probabilistic_layer.set(1, 1);

    probabilistic_layer.set_parameters_constant(0.0);

    synaptic_weights = probabilistic_layer.get_synaptic_weights();

    assert_true(synaptic_weights.dimension(0) == 1, LOG);
    assert_true(synaptic_weights.dimension(1) == 1, LOG);
    assert_true(synaptic_weights(0,0) < numeric_limits<type>::min(), LOG);

    cout << "test_get_synaptic_weight_with_parameters\n";

    ProbabilisticLayer probabilistic_layer_2;

    // Test 1
    Tensor<type, 2> biases_2(1, 4);
    biases_2.setValues({{9},{-8},{7},{-6}});

    Tensor<type, 2> synaptic_weights_2(2, 4);
    synaptic_weights_2.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

    Tensor<type, 1> parameters(12);
    probabilistic_layer_2.set(2, 4);

    probabilistic_layer_2.set_synaptic_weights(synaptic_weights_2);
    probabilistic_layer_2.set_biases(biases_2);

    parameters = probabilistic_layer_2.get_parameters();
    synaptic_weights_2 = probabilistic_layer_2.get_synaptic_weights(parameters);

    assert_true(synaptic_weights_2.size() == 8, LOG);

    assert_true(abs(synaptic_weights_2(0,0) + 11) < static_cast<type>(1e-5), LOG);
    assert_true(abs(synaptic_weights_2(1,3) + 24) < static_cast<type>(1e-5), LOG);
}

void ProbabilisticLayerTest::test_get_parameters()
{
   cout << "test_get_parameters\n";

   ProbabilisticLayer probabilistic_layer;
   Tensor<type, 2> synaptic_weights;
   Tensor<type, 1> parameters;

   // Test 0
   probabilistic_layer.set(1, 1);
   probabilistic_layer.set_parameters_constant(1.0);

   parameters = probabilistic_layer.get_parameters();

   assert_true(parameters.size() == 2, LOG);
   assert_true(abs(parameters(0) - 1) < numeric_limits<type>::min(), LOG);

   // Test 1
   ProbabilisticLayer probabilistic_layer_2;

   Tensor<type, 2> biases_2(1, 4);
   biases_2.setValues({{9},{-8},{7},{-6}});

   Tensor<type, 2> synaptic_weights_2(2, 4);
   synaptic_weights_2.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

   probabilistic_layer_2.set_synaptic_weights(synaptic_weights_2);
   probabilistic_layer_2.set_biases(biases_2);

   Tensor<type,1>new_parameters = probabilistic_layer_2.get_parameters();

   assert_true(new_parameters.size() == 12, LOG);
   assert_true(abs(new_parameters(0) - 9) < static_cast<type>(1e-5), LOG);
   assert_true(abs(new_parameters(4) + 11) < static_cast<type>(1e-5), LOG);
   assert_true(abs(new_parameters(7) + 22) < static_cast<type>(1e-5), LOG);
   }

void ProbabilisticLayerTest::test_get_decision_threshold()
{
   cout << "test_get_decision_threshold\n";

   ProbabilisticLayer probabilistic_layer;
   probabilistic_layer.set_decision_threshold(0.5);

   assert_true(abs(probabilistic_layer.get_decision_threshold() - static_cast<type>(0.5)) < static_cast<type>(1e-5), LOG);
   }

void ProbabilisticLayerTest::test_set()
{
   cout << "test_set\n";

   ProbabilisticLayer probabilistic_layer;
   probabilistic_layer.set();

   assert_true(probabilistic_layer.get_biases_number() == 0, LOG);
   assert_true(probabilistic_layer.get_synaptic_weights_number() == 0, LOG);
}

void ProbabilisticLayerTest::test_set_default()
{
   cout << "test_set_default\n";

   ProbabilisticLayer probabilistic_layer;
   probabilistic_layer.set_neurons_number(2);

   probabilistic_layer.set_default();

   assert_true(probabilistic_layer.get_activation_function() == OpenNN::ProbabilisticLayer::Softmax, LOG);
   assert_true(abs(probabilistic_layer.get_decision_threshold() - 0.5) < static_cast<type>(1e-5), LOG);
   assert_true(probabilistic_layer.get_display() == true, LOG);

   probabilistic_layer.set_neurons_number(1);

   probabilistic_layer.set_default();

   assert_true(probabilistic_layer.get_activation_function() == OpenNN::ProbabilisticLayer::Logistic, LOG);
}

void ProbabilisticLayerTest::test_set_biases()
{
   cout << "test_set_biases\n";

    ProbabilisticLayer probabilistic_layer;

    Tensor<type, 2> biases(1, 4);

    probabilistic_layer.set(1, 4);

    biases.setZero();

    probabilistic_layer.set_biases(biases);

    assert_true(probabilistic_layer.get_biases_number() == 4, LOG);

    assert_true(abs(probabilistic_layer.get_biases()(0)) < static_cast<type>(1e-5), LOG);
    assert_true(abs(probabilistic_layer.get_biases()(3)) < static_cast<type>(1e-5), LOG);
}

void ProbabilisticLayerTest::test_set_synaptic_weights()
{
   cout << "test_set_synaptic_weights\n";

    ProbabilisticLayer probabilistic_layer(1, 2);

    Tensor<type, 2> synaptic_weights(2, 1);

    synaptic_weights.setZero();

    probabilistic_layer.set_synaptic_weights(synaptic_weights);

    assert_true(probabilistic_layer.get_synaptic_weights().size() == 2, LOG);

    assert_true(abs(probabilistic_layer.get_synaptic_weights()(0)) < static_cast<type>(1e-5), LOG);
    assert_true(abs(probabilistic_layer.get_synaptic_weights()(1)) < static_cast<type>(1e-5), LOG);
}

void ProbabilisticLayerTest::test_set_parameters()
{
  cout << "test_set_parameters\n";

    ProbabilisticLayer probabilistic_layer;

    probabilistic_layer.set(1, 2);

    Tensor<type, 1> parameters_2(4);

    parameters_2.setValues({11,12,21,22});

    probabilistic_layer.set_parameters(parameters_2);

    assert_true(probabilistic_layer.get_biases()(0) - parameters_2(0) < static_cast<type>(1e-5), LOG);
    assert_true(probabilistic_layer.get_synaptic_weights()(0) - parameters_2(2)  < static_cast<type>(1e-5), LOG);
}

void ProbabilisticLayerTest::test_set_decision_threshold()
{
   cout << "test_set_decision_threshold\n";

   ProbabilisticLayer probabilistic_layer;
   probabilistic_layer.set_decision_threshold(static_cast<type>(0.7));

   assert_true(abs(probabilistic_layer.get_decision_threshold() - static_cast<type>(0.7)) < static_cast<type>(1e-5), LOG);
}

void ProbabilisticLayerTest::test_write_activation_function()
{
   cout << "test_write_activation_function\n";

   ProbabilisticLayer probabilistic_layer;

   probabilistic_layer.set();

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Binary);
   assert_true(probabilistic_layer.write_activation_function() == "Binary", LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Logistic);
   assert_true(probabilistic_layer.write_activation_function() == "Logistic", LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Competitive);
   assert_true(probabilistic_layer.write_activation_function() == "Competitive", LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);
   assert_true(probabilistic_layer.write_activation_function() == "Softmax", LOG);
}

void ProbabilisticLayerTest::test_write_activation_function_text()
{
    cout << "test_write_activation_function_text\n";

    ProbabilisticLayer probabilistic_layer;

    probabilistic_layer.set();

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Binary);
    assert_true(probabilistic_layer.write_activation_function_text() == "binary", LOG);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Competitive);
    assert_true(probabilistic_layer.write_activation_function_text() == "competitive", LOG);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Logistic);
    assert_true(probabilistic_layer.write_activation_function_text() == "logistic", LOG);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);
    assert_true(probabilistic_layer.write_activation_function_text() == "softmax", LOG);
}

void ProbabilisticLayerTest::test_set_activation_function()
{
   cout << "test_set_activation_function\n";

   ProbabilisticLayer probabilistic_layer;

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);
   assert_true(probabilistic_layer.get_activation_function() == ProbabilisticLayer::Softmax, LOG);

   probabilistic_layer.set_activation_function("Softmax");
   assert_true(probabilistic_layer.get_activation_function() == ProbabilisticLayer::Softmax, LOG);
}

void ProbabilisticLayerTest::test_get_display()
{
   cout << "test_get_display\n";

   ProbabilisticLayer probabilistic_layer;

   probabilistic_layer.set_display(true);

   assert_true(probabilistic_layer.get_display() == true, LOG);
}

void ProbabilisticLayerTest::test_set_display()
{
   cout << "test_set_display\n";
}

void ProbabilisticLayerTest::test_calculate_combinations()
{
   cout << "test_calculate_combinations\n";

   ProbabilisticLayer probabilistic_layer;

   Tensor<type, 2> biases(1,1);
   Tensor<type, 2> synaptic_weights(1,1);
   Tensor<type, 1> parameters(1);

   Tensor<type, 2> inputs(1,1);
   Tensor<type, 2> combinations_2d(1,1);

   biases.setConstant(1.0);
   synaptic_weights.setConstant(2.0);

   probabilistic_layer.set(1,1);
   inputs.setConstant(3.0);

   probabilistic_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations_2d);

   assert_true(combinations_2d.rank() == 2, LOG);
   assert_true(combinations_2d.dimension(0) == 1, LOG);
   assert_true(combinations_2d.dimension(1) == 1, LOG);
   assert_true(abs(combinations_2d(0,0) - 7) < static_cast<type>(1e-5) , LOG);

}

void ProbabilisticLayerTest::test_calculate_activations()
{
   cout << "test_calculate_activations\n";

   ProbabilisticLayer probabilistic_layer;

   Tensor<type, 2> biases(1,1);
   Tensor<type, 2> synaptic_weights(1,1);
   Tensor<type, 1> parameters(1);

   Tensor<type, 2> inputs(1,1);
   Tensor<type, 2> combinations_2d(1,1);
   Tensor<type, 2> activations_2d(1,1);

   // Test 1

   probabilistic_layer.set(1,1);

   biases.setConstant(1.0);
   synaptic_weights.setConstant(1.0);

   inputs.setConstant(-1);

   probabilistic_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations_2d);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Binary);

   probabilistic_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 1, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 1 , LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Logistic);

   probabilistic_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 1, LOG);
   assert_true(activations_2d(0,0) - static_cast<type>(0.5) < static_cast<type>(1e-5), LOG);

   // Test 2

   probabilistic_layer.set(2, 2);
   probabilistic_layer.set_parameters_constant(2);

   combinations_2d.resize(1,2);
   combinations_2d.setZero();

   activations_2d.resize(1,2);
   activations_2d.setZero();

   inputs.resize(1,2);
   inputs.setConstant(2);

   probabilistic_layer.calculate_combinations(inputs, probabilistic_layer.get_biases(), probabilistic_layer.get_synaptic_weights(), combinations_2d);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Competitive);

   probabilistic_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 2, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 1, LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);
   probabilistic_layer.calculate_activations(combinations_2d, activations_2d);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);

   // Test 3

   probabilistic_layer.set(3, 3);

   combinations_2d.resize(1,3);
   combinations_2d.setValues({{1,0,-1}});

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Competitive);

   activations_2d.resize(1,3);
   probabilistic_layer.calculate_activations(combinations_2d, activations_2d);

   assert_true(activations_2d.rank() == 2, LOG);
   assert_true(activations_2d.dimension(0) == 1, LOG);
   assert_true(activations_2d.dimension(1) == 3, LOG);
   assert_true(static_cast<Index>(activations_2d(0,0)) == 1, LOG);
   assert_true(static_cast<Index>(activations_2d(0,1)) == 0, LOG);
   assert_true(static_cast<Index>(activations_2d(0,2)) == 0, LOG);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);
   probabilistic_layer.calculate_activations(combinations_2d, activations_2d);
   assert_true(abs(activations_2d(0,0) - static_cast<type>(0.6652)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_2d(0,1) - static_cast<type>(0.2447)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(activations_2d(0,2) - static_cast<type>(0.09)) < static_cast<type>(1e-3), LOG);

}

void ProbabilisticLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_derivatives_activations\n";

    NumericalDifferentiation numerical_differentiation;
    ProbabilisticLayer probabilistic_layer;

    Tensor<type, 2> combinations_2d;
    Tensor<type, 2> activations_2d;
    Tensor<type, 3> activations_derivatives;

    // Test 1

    probabilistic_layer.set(1,3);

    combinations_2d.resize(1,3);
    combinations_2d.setValues({{1, 2, 3}});
    activations_2d.resize(1,3);
    activations_derivatives.resize(3,3,1);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);
    probabilistic_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

    assert_true(activations_derivatives.rank() == 3, LOG);
    assert_true(activations_derivatives.dimension(0) == 3, LOG);
    assert_true(activations_derivatives.dimension(1) == 3, LOG);
    assert_true(activations_derivatives.dimension(2) == 1, LOG);

    assert_true(abs(activations_2d(0,0) - static_cast<type>(0.09)) < static_cast<type>(1e-3), LOG);

    assert_true(abs(activations_derivatives(0,0,0) - static_cast<type>(0.0819)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(1,1,0) - static_cast<type>(0.1848)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(2,2,0) - static_cast<type>(0.2227)) < static_cast<type>(1e-3), LOG);

    // Test 2

    probabilistic_layer.set(1,4);

    combinations_2d.resize(1,4);
    combinations_2d.setValues({{-1, 2, -3, -4}});
    activations_2d.resize(1,4);
    activations_derivatives.resize(4,4,1);

    probabilistic_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

    assert_true(abs(activations_derivatives(3,0,0) + static_cast<type>(0.00011)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(3,1,0) + static_cast<type>(0.00221)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(3,2,0) + static_cast<type>(0.00001)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(3,3,0) - static_cast<type>(0.00233)) < static_cast<type>(1e-3), LOG);

    // Test 3

    probabilistic_layer.set(1,1);

    combinations_2d.resize(1,1);
    combinations_2d.setValues({{-1.55f}});
    activations_2d.resize(1,1);
    activations_derivatives.resize(1,1,1);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Logistic);
    probabilistic_layer.calculate_activations_derivatives(combinations_2d, activations_2d, activations_derivatives);

    assert_true(abs(activations_2d(0,0) - static_cast<type>(0.175)) < static_cast<type>(1e-2), LOG);

    assert_true(activations_derivatives.rank() == 3, LOG);
    assert_true(activations_derivatives.dimension(0) == 1, LOG);
    assert_true(activations_derivatives.dimension(1) == 1, LOG);
    assert_true(activations_derivatives.dimension(2) == 1, LOG);
    assert_true(abs(activations_derivatives(0,0,0) - static_cast<type>(0.1444)) < static_cast<type>(1e-3), LOG);

}

void ProbabilisticLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    ProbabilisticLayer probabilistic_layer;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> biases;

    Tensor<type, 2> inputs;
    Tensor<type, 1> parameters;

    // Test 1

    probabilistic_layer.set(3, 4);

    synaptic_weights.resize(3, 4);
    biases.resize(1, 4);
    inputs.resize(1, 3);

    inputs.setConstant(1);
    biases.setConstant(1);
    synaptic_weights.setValues({{1,-1,0,1},{2,-2,0,2},{3,-3,0,3}});

    probabilistic_layer.set_synaptic_weights(synaptic_weights);
    probabilistic_layer.set_biases(biases);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);

    Tensor<type,2>outputs = probabilistic_layer.calculate_outputs(inputs);

    Tensor<type,1>perceptron_sol(4);
    perceptron_sol.setValues({7,-5,1,7});

    Tensor<type,0>div = perceptron_sol.exp().sum();
    Tensor<type,1>sol_ = perceptron_sol.exp() / div(0);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs(0,0)) == static_cast<Index >(sol_(0)), LOG);
    assert_true(static_cast<Index>(outputs(1,0)) == static_cast<Index >(sol_(1)), LOG);
    assert_true(static_cast<Index>(outputs(2,0)) == static_cast<Index >(sol_(2)), LOG);

    // Test 1_2

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Competitive);

    Tensor<type,2>outputs_2 = probabilistic_layer.calculate_outputs(inputs);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs_2(0,0)) == 1, LOG);
    assert_true(static_cast<Index>(outputs_2(1,0)) == 0, LOG);
    assert_true(static_cast<Index>(outputs_2(2,0)) == 0, LOG);

    // Test 2

    Tensor<type, 2> biases_2(1, 4);
    biases.setValues({{9},{-8},{7},{-6}});

    Tensor<type, 2> synaptic_weights_2(2, 4);

    synaptic_weights.resize(2, 4);
    synaptic_weights.setValues({{-11, 12, -13, 14},{21, -22, 23, -24}});

    probabilistic_layer.set_synaptic_weights(synaptic_weights);
    probabilistic_layer.set_biases(biases);

    inputs.resize(1, 2);
    inputs.setConstant(1);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);

    outputs = probabilistic_layer.calculate_outputs(inputs);

    Tensor<type,1>perceptron_sol_3(4);
    perceptron_sol.setValues({7,-5,1,7});

    div = perceptron_sol.exp().sum();
    sol_ = perceptron_sol.exp() / div(0);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index >(outputs(0,0)) == static_cast<Index >(sol_(0)), LOG);
    assert_true(static_cast<Index >(outputs(1,0)) == static_cast<Index >(sol_(1)), LOG);
    assert_true(static_cast<Index >(outputs(2,0)) == static_cast<Index >(sol_(2)), LOG);

   // Test  3

   probabilistic_layer.set(3, 2);
   probabilistic_layer.set_parameters_constant(0.0);

   inputs.resize(1,3);
   inputs.setConstant(0.0);

   outputs = probabilistic_layer.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);
   assert_true(abs(outputs(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-5), LOG);

}


void ProbabilisticLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    ProbabilisticLayer probabilistic_layer(2,2);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);

    Tensor<type, 1> parameters(6);
    Tensor<type, 2> inputs(1,2);

    probabilistic_layer.set_parameters_constant(1);
    inputs.setConstant(1);

    Layer::ForwardPropagation forward_propagation(1, &probabilistic_layer);
    probabilistic_layer.forward_propagate(inputs, forward_propagation);

    assert_true(forward_propagation.combinations_2d.rank() == 2, LOG);
    assert_true(forward_propagation.combinations_2d.dimension(0) == 1, LOG);
    assert_true(forward_propagation.combinations_2d.dimension(1) == 2, LOG);
    assert_true(abs(forward_propagation.combinations_2d(0,0) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.combinations_2d(0,1) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.activations_2d(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.activations_2d(0,1) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.activations_derivatives_3d(0,0,0) - static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.activations_derivatives_3d(0,1,0) + static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);
}

void ProbabilisticLayerTest::test_calculate_output_delta()
{
    cout << "test_calculate_output_delta\n";

    ProbabilisticLayer probabilistic_layer(2,2);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);

    Tensor<type,2> output_delta(1,2);

    Tensor<type, 1> parameters(6);
    Tensor<type, 2> inputs(1,2);

    probabilistic_layer.set_parameters_constant(1);
    inputs.setConstant(1);
    Tensor<type,2> activations_2d(1,2);

    Layer::ForwardPropagation forward_propagation(1, &probabilistic_layer);
    probabilistic_layer.forward_propagate(inputs, forward_propagation);

    Tensor<type,2> output_gradient(1,2);
    output_gradient.setValues({{1,0}});

    probabilistic_layer.calculate_output_delta(forward_propagation, output_gradient, output_delta);

    assert_true(output_delta.rank() == 2, LOG);
    assert_true(output_delta.dimension(0) == 1, LOG);
    assert_true(output_delta.dimension(1) == 2, LOG);
    assert_true(abs(output_delta(0,0) - static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(output_delta(0,1) + static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);
}

void ProbabilisticLayerTest::test_calculate_error_gradient()
{
    cout << "test_calculate_error_gradient\n";

    ProbabilisticLayer probabilistic_layer(2,2);

    probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);

    Tensor<type, 1> parameters(6);
    Tensor<type, 2> inputs(1,2);

    Tensor<type, 2> output_gradient(1,2);

    Tensor<type, 2> output_delta(1,2);

    // Test 1

    parameters.setValues({1,1, 1,1,1,1});
    probabilistic_layer.set_parameters(parameters);

    inputs.setValues({{0,1}});

    Layer::ForwardPropagation forward_propagation(1, &probabilistic_layer);

    probabilistic_layer.forward_propagate(inputs, forward_propagation);

    Layer::BackPropagation back_propagation(1, &probabilistic_layer);

    output_gradient.setValues({{1,-7}});

    probabilistic_layer.calculate_output_delta(forward_propagation,output_gradient, output_delta);

    back_propagation.delta = output_delta;

    probabilistic_layer.calculate_error_gradient(inputs, forward_propagation, back_propagation);

    assert_true(back_propagation.biases_derivatives.rank() == 1, LOG);
    assert_true(back_propagation.biases_derivatives.dimension(0) == 2, LOG);
    assert_true(abs(back_propagation.biases_derivatives(0) - static_cast<type>(2)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(back_propagation.biases_derivatives(1) + static_cast<type>(2)) < static_cast<type>(1e-3), LOG);

    assert_true(back_propagation.synaptic_weights_derivatives.rank() == 2, LOG);
    assert_true(back_propagation.synaptic_weights_derivatives.dimension(0) == 2, LOG);
    assert_true(back_propagation.synaptic_weights_derivatives.dimension(1) == 2, LOG);
    assert_true(abs(back_propagation.synaptic_weights_derivatives(0,0) - static_cast<type>(0)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(back_propagation.synaptic_weights_derivatives(0,1) - static_cast<type>(0)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(back_propagation.synaptic_weights_derivatives(1,0) - static_cast<type>(2)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(back_propagation.synaptic_weights_derivatives(1,1) + static_cast<type>(2)) < static_cast<type>(1e-3), LOG);

}


void ProbabilisticLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   ProbabilisticLayer probabilistic_layer(2,2);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);

   Tensor<string, 1> inputs_names(2);
   inputs_names.setValues({"Uno_in","Dos_in"});

   Tensor<string, 1> outputs_names(2);
   outputs_names.setValues({"Uno_out","Dos_out"});

   string expression;

   expression = probabilistic_layer.write_expression(inputs_names,outputs_names);

   assert_true(expression.empty() == false, LOG);
}



void ProbabilisticLayerTest::run_test_case()
{
   cout << "Running probabilistic layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();


   // Assignment operators methods

   test_assignment_operator();


   // Get methods

   test_get_biases();
   test_get_synaptic_weights();
   test_get_parameters();
   test_get_decision_threshold();


   // Layer architecture

   test_get_inputs_number();
   test_get_neurons_number();


   // Display messages

   test_get_display();


   // Set methods

   test_set();
   test_set_default();
   test_set_biases();
   test_set_synaptic_weights();
   test_set_parameters();
   test_set_decision_threshold();


   //Activation function

   test_write_activation_function();
   test_write_activation_function_text();
   test_set_activation_function();


   // Display messages

   test_set_display();


   // Probabilistic post-processing

   test_calculate_combinations();
   test_calculate_activations();
   test_calculate_activations_derivatives();
   test_calculate_outputs();


   // Forward propagate

   test_forward_propagate();


   // Hidden delta

   test_calculate_output_delta();

   // Gradient

   test_calculate_error_gradient();


   //Write expression

   test_write_expression();

   cout << "End of probabilistic layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques, SL.
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
