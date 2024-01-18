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
    assert_true(perceptron_layer_1.get_type() == Layer::Type::Perceptron, LOG);

    // Architecture constructor

    PerceptronLayer perceptron_layer_3(10, 3, PerceptronLayer::ActivationFunction::Linear);
    assert_true(perceptron_layer_3.write_activation_function() == "Linear", LOG);
    assert_true(perceptron_layer_3.get_inputs_number() == 10, LOG);
    assert_true(perceptron_layer_3.get_neurons_number() == 3, LOG);
    assert_true(perceptron_layer_3.get_biases_number() == 3, LOG);
    assert_true(perceptron_layer_3.get_synaptic_weights_number() == 30, LOG);
    assert_true(perceptron_layer_3.get_parameters_number() == 33, LOG);

    PerceptronLayer perceptron_layer_4(0, 0, PerceptronLayer::ActivationFunction::Logistic);
    assert_true(perceptron_layer_4.write_activation_function() == "Logistic", LOG);
    assert_true(perceptron_layer_4.get_inputs_number() == 0, LOG);
    assert_true(perceptron_layer_4.get_neurons_number() == 0, LOG);
    assert_true(perceptron_layer_4.get_biases_number() == 0, LOG);
    assert_true(perceptron_layer_4.get_synaptic_weights_number() == 0, LOG);
    assert_true(perceptron_layer_4.get_parameters_number() == 0, LOG);

    PerceptronLayer perceptron_layer_5(1, 1, PerceptronLayer::ActivationFunction::Linear);
    assert_true(perceptron_layer_5.write_activation_function() == "Linear", LOG);
    assert_true(perceptron_layer_5.get_inputs_number() == 1, LOG);
    assert_true(perceptron_layer_5.get_neurons_number() == 1, LOG);
    assert_true(perceptron_layer_5.get_biases_number() == 1, LOG);
    assert_true(perceptron_layer_5.get_synaptic_weights_number() == 1, LOG);
    assert_true(perceptron_layer_5.get_parameters_number() == 2, LOG);
}


void PerceptronLayerTest::test_destructor()
{
    cout << "test_destructor\n";

    PerceptronLayer* perceptron_layer = new PerceptronLayer;

    delete perceptron_layer;
}


void PerceptronLayerTest::test_calculate_combinations()
{
    cout << "test_calculate_combinations\n";

    Index parameters_number;

    Tensor<type, 1> biases;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 1> parameters;

    Tensor<type, 2> inputs;
    Tensor<type, 2> combinations;

    // Test

    inputs_number = 1;
    neurons_number = 1;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(0));

    inputs.resize(samples_number, inputs_number);
    inputs.setZero();

    combinations.resize(samples_number, neurons_number);
    combinations.setConstant(type(3.1416));

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

    // Test

    biases.setConstant(type(1));
    synaptic_weights.setConstant(type(2));

    inputs.setConstant(type(3));

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(abs(combinations(0,0) - type(7)) < type(1e-5) , LOG);

    // Test

    inputs_number = 2;
    neurons_number = 2;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    combinations.resize(samples_number, neurons_number);
    combinations.setZero();

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == samples_number, LOG);
    assert_true(combinations.dimension(1) == neurons_number, LOG);
    assert_true(abs(combinations(0,0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    combinations.resize(2, 4);
    combinations.setZero();

    perceptron_layer.set(3, 4);

    synaptic_weights.resize(3, 4);
    synaptic_weights.setConstant(type(1));
    biases.resize(4);
    biases.setConstant(type(2));

    inputs.resize(2,3);
    inputs.setConstant(type(0.5));

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 2, LOG);
    assert_true(combinations.dimension(1) == 4, LOG);
    assert_true(abs(combinations(0,0) - type(3.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 2;
    neurons_number = 4;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number);

    synaptic_weights.resize(inputs_number, neurons_number);
    synaptic_weights.setConstant(type(1));
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    biases.resize( neurons_number);
    biases.setConstant(type(1));
    perceptron_layer.set_biases(biases);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({{type(0.5), type(0.5)}});

    combinations.resize(samples_number, neurons_number);
    combinations.setZero();

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == samples_number, LOG);
    assert_true(combinations.dimension(1) == neurons_number, LOG);
    assert_true(Index(combinations(0,0)) == 2, LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);

    synaptic_weights.resize(inputs_number, neurons_number);
    synaptic_weights.setValues({{type(1)}});
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    biases.resize( neurons_number);
    biases.setValues({type(-0.5)});
    perceptron_layer.set_biases(biases);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({{type(-0.8)}});

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases(0) - type(-0.5) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(synaptic_weights(0, 0) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    parameters_number = perceptron_layer.get_parameters_number();

    assert_true(parameters_number == 2, LOG);

    parameters = perceptron_layer.get_parameters();

    assert_true(parameters(0) - type(-0.5) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(parameters(1) - type(1.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number);

    inputs.resize(samples_number, inputs_number);
    inputs.setZero();

    biases.resize( neurons_number);
    biases.setZero();

    synaptic_weights.resize(inputs_number, neurons_number);
    synaptic_weights.setZero();

    combinations.resize(samples_number, neurons_number);

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(combinations.dimension(1) == 1, LOG);

    assert_true(abs(combinations(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);

}


void PerceptronLayerTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";

    Tensor<type, 1> biases;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 1> parameters;

    Tensor<type, 2> inputs;
    Tensor<type, 2> combinations;
    Tensor<type, 2> activations;

    // Test

    inputs_number = 1;
    neurons_number = 1;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(1));

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    combinations.resize(samples_number, neurons_number);

    activations.resize(samples_number, neurons_number);

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == 1, LOG);
    assert_true(activations.dimension(1) == 1, LOG);
    assert_true(Index(activations(0,0)) == 2 , LOG);

    // Test

    inputs_number = 2;
    neurons_number = 3;
    samples_number = 4;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(2));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(2));

    combinations.resize(samples_number, neurons_number);

    activations.resize(samples_number, neurons_number);

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    perceptron_layer.calculate_combinations(inputs, biases, synaptic_weights, combinations);
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(Index(activations(0,0)) == 10, LOG);

    // Test

    inputs_number = 6;
    neurons_number = 3;
    samples_number = 2;

    perceptron_layer.set(inputs_number, neurons_number);
    parameters.resize(2*neurons_number);

    parameters.setConstant(type(0));

    combinations.resize(samples_number, neurons_number);
    combinations.setConstant(type(0));

    activations.resize(samples_number, neurons_number);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(Index(activations(0,0)) == 0, LOG);

    // Test

    inputs_number = 1;
    neurons_number = 2;
    samples_number = 1;

    perceptron_layer.set(1, 2);

    parameters.resize(4);
    parameters.setConstant(type(0));

    combinations.resize(samples_number, neurons_number);
    combinations.setConstant(type(0));

    activations.resize(samples_number, neurons_number);
    activations.setConstant(type(0));

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(Index(activations(0,0)) == 1 , LOG);

    // Test

    inputs_number = 2;
    neurons_number = 2;
    samples_number = 1;

    perceptron_layer.set(2, 2);
    parameters.resize(6);
    parameters.setConstant(type(0));

    combinations.resize(samples_number, neurons_number);
    combinations.setConstant(-2.0);

    activations.resize(samples_number, neurons_number);
    activations.setConstant(type(0));

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(Index(activations(0,0))  == -1, LOG);

    // Test

    perceptron_layer.set(1, 2);
    perceptron_layer.set_parameters_constant(type(0));

    combinations.resize(2,2);
    combinations.setConstant(4.0);

    activations.resize(2,2);
    activations.setZero();

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == 2, LOG);
    assert_true(activations.dimension(1) == 2, LOG);
    assert_true(Index(activations(0,0)) == 4.0, LOG);

    // Test

    inputs_number = 3;
    neurons_number = 2;
    samples_number = 1;

    perceptron_layer.set(3, 2);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(1,3);
    inputs.setConstant(0.5);

    combinations.resize(1,2);
    activations.resize(1,2);

    perceptron_layer.calculate_combinations(inputs,
                                            perceptron_layer.get_biases(),
                                            perceptron_layer.get_synaptic_weights(),
                                            combinations);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(combinations.dimension(1) == 2, LOG);
    assert_true(abs(combinations(0,0) - type(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == 1, LOG);
    assert_true(activations.dimension(1) == 2, LOG);
    assert_true(Index(activations(0,0)) == 1, LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(Index(activations(0,0)) == 1, LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::PerceptronLayer::ActivationFunction::Logistic);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(abs(activations(0,0) - type(1.0/(1.0+exp(-2.5)))) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(abs(activations(0,0) - type(tanh(2.5))) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations, activations);

    assert_true(abs(activations(0,0) - type(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void PerceptronLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activations_derivatives\n";

    Tensor<type, 1> parameters(1);
    Tensor<type, 2> inputs(1,1);
    Tensor<type, 2> combinations(1,1);
    Tensor<type, 2> activations(1,1);
    Tensor<type, 2> activations_derivatives(1,1);

    Tensor<Index, 1> dimensions(2);
    dimensions = get_dimensions(activations_derivatives);

    // Test

    perceptron_layer.set(1,1);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.setConstant(type(1));

    combinations.setConstant(type(1));

    activations_derivatives.setZero();

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    assert_true(activations_derivatives.rank() == 2, LOG);
    assert_true(activations_derivatives.dimension(0) == 1, LOG);
    assert_true(activations_derivatives.dimension(1) == 1, LOG);
    assert_true(abs(activations(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(activations_derivatives(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(1)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(0.731)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(0.196)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(0.761)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(0.41997)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(1)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(1)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ExponentialLinear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(1)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(1)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ScaledExponentialLinear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(1.05)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(1.05)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftPlus);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(1.313)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(0.731)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftSign);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(0.5)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(0.25)) < type(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HardSigmoid);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);
    assert_true(abs(activations(0,0) - type(0.7)) < type(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(0.2)) < type(1e-3), LOG);

    // Test

    perceptron_layer.set(2, 4);
    perceptron_layer.set_parameters_constant(type(1));

    combinations.resize(1,4);
    combinations.setValues({{type(1.56f), type(-0.68f), type(0.91f), type(-1.99f)}});

    activations.resize(1,4);

    activations_derivatives.resize(1,4);
    activations_derivatives.setZero();

    dimensions = get_dimensions(activations_derivatives);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    Tensor<type, 2> numerical_activation_derivative(1,4);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ExponentialLinear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ScaledExponentialLinear);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftPlus);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftSign);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);

    // Test

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HardSigmoid);
    perceptron_layer.calculate_activations_derivatives(combinations, activations, activations_derivatives);

    numerical_activation_derivative
            = numerical_differentiation.calculate_derivatives(*this, &PerceptronLayerTest::get_activations, combinations);

    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < type(1e-3), LOG);
}


void PerceptronLayerTest::test_forward_propagate()
{

    cout << "test_forward_propagate\n";

    Tensor<type, 1> parameters;
    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<type, 1> potential_parameters;

    pair<type*, dimensions> inputs_pair;

    // Test

    samples_number = 2;
    inputs_number = 2;
    neurons_number = 2;
    bool is_training = true;

    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));


    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    perceptron_layer.forward_propagate(inputs_pair, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(abs(outputs(0,0) - type(3)) < type(1e-3), LOG);
    assert_true(abs(outputs(0,1) - type(3)) < type(1e-3), LOG);

    assert_true(abs(perceptron_layer_forward_propagation.activations_derivatives(0,0) - type(1)) < type(1e-3), LOG);
    assert_true(abs(perceptron_layer_forward_propagation.activations_derivatives(0,1) - type(1)) < type(1e-3), LOG);

    // Test

    samples_number = 2;
    inputs_number = 2;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));    

    potential_parameters = perceptron_layer.get_parameters();

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    perceptron_layer.forward_propagate(inputs_pair, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(abs(outputs(0,0) - type(0.99505)) < type(1e-3), LOG);
    assert_true(abs(outputs(0,1) - type(0.99505)) < type(1e-3), LOG);
    assert_true(abs(perceptron_layer_forward_propagation.activations_derivatives(0,0) - type(0.00986)) < type(1e-3), LOG);
    assert_true(abs(perceptron_layer_forward_propagation.activations_derivatives(0,1) - type(0.00986)) < type(1e-3), LOG);

    Tensor<type, 1> biases;
    Tensor<type, 2> synaptic_weights;
    Tensor<Index, 1> outputs_dimensions;

    // Test

    samples_number = 1;
    inputs_number = 3;
    neurons_number = 4;

    perceptron_layer.set(inputs_number, neurons_number);

    synaptic_weights.resize(inputs_number, neurons_number);
    biases.resize( neurons_number);
    inputs.resize(samples_number, inputs_number);
    outputs.resize(1, neurons_number);

    inputs.setConstant(type(1));
    biases.setConstant(type(1));
    synaptic_weights.setValues({{type(1),type(-1),type(0),type(1)},
                                {type(2),type(-2),type(0),type(2)},
                                {type(3),type(-3),type(0),type(3)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    perceptron_layer.forward_propagate(inputs_pair, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(Index(outputs(0,0)) == 7, LOG);
    assert_true(Index(outputs(1,0)) == -5, LOG);
    assert_true(Index(outputs(2,0)) == 1, LOG);

    // Test

    inputs_number = 2;
    neurons_number = 4;

    biases.resize( neurons_number);
    biases.setValues({type(9),
                      type(-8),
                      type(7),
                      type(-6)});

    synaptic_weights.resize(2, 4);

    synaptic_weights.setValues({{type(-11), type(12), type(-13), type(14)},
                                {type(21), type(-22), type(23), type(-24)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    outputs.resize(1, neurons_number);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    perceptron_layer.forward_propagate(inputs_pair, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(Index(outputs(0,0)) == 1, LOG);
    assert_true(Index(outputs(1,0)) == 0, LOG);
    assert_true(Index(outputs(2,0)) == 1, LOG);

    // Test

    inputs_number = 3;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(0));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(0));

    outputs.resize(1, neurons_number);

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    perceptron_layer.forward_propagate(inputs_pair, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 2, LOG);
    assert_true(abs(outputs(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 4;
    neurons_number = 2;

    perceptron_layer.set(4, 2);
    parameters.resize(10);

    parameters.setValues({type(-1),type(2),type(-3),type(4),type(-5),type(6),type(-7),type(8),type(-9),type(10) });

    perceptron_layer.set_parameters(parameters);

    inputs.resize(samples_number,inputs_number);
    inputs.setValues({{type(4),type(-3),type(2),type(-1)}});

    outputs.resize(1, neurons_number);

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    perceptron_layer.forward_propagate(inputs_pair, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;

    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 2, LOG);
    assert_true(abs(outputs(0,0) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test 5

    inputs_number = 1;
    neurons_number = 2;

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(3.0));

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(-2.0));

    outputs.resize(1, neurons_number); 

    perceptron_layer_forward_propagation.set(samples_number, &perceptron_layer);

    inputs_pair.first = inputs.data();
    inputs_pair.second = {{samples_number, inputs_number}};

    perceptron_layer.forward_propagate(inputs_pair, &perceptron_layer_forward_propagation, is_training);

    outputs = perceptron_layer_forward_propagation.outputs;
    parameters.resize(2);
    parameters.setConstant(type(1));

    // Test

    perceptron_layer.set(1, 1);

    inputs.resize(1,1);
    inputs.setRandom();

    parameters = perceptron_layer.get_parameters();
}


void PerceptronLayerTest::run_test_case()
{
    cout << "Running perceptron layer test case...\n";

    // Constructor and destructor

    test_constructor();
    test_destructor();

    // Combinations

    test_calculate_combinations();

    // Activation

    test_calculate_activations();
    test_calculate_activations_derivatives();

    // Forward propagate

    test_forward_propagate();

    cout << "End of perceptron layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
