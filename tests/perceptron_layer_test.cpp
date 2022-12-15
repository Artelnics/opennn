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


void PerceptronLayerTest::test_set()
{
    cout << "test_set\n";

    // Test

    perceptron_layer.set(1,1);

    perceptron_layer.set();

    assert_true(perceptron_layer.get_inputs_number() == 0, LOG);
    assert_true(perceptron_layer.get_neurons_number() == 0, LOG);

    // Test

    perceptron_layer.set();

    perceptron_layer.set(1,1);

    assert_true(perceptron_layer.get_inputs_number() == 1, LOG);
    assert_true(perceptron_layer.get_neurons_number() == 1, LOG);
}


void PerceptronLayerTest::test_set_default()
{
    cout << "test_set_default\n";

    perceptron_layer.set_default();

    assert_true(perceptron_layer.get_display(), LOG);
    assert_true(perceptron_layer.get_type() == opennn::Layer::Type::Perceptron, LOG);
}


void PerceptronLayerTest::test_set_biases()
{

    cout << "test_set_biases\n";

    Tensor<type, 2> biases;

    // Test

    perceptron_layer.set(1, 4);

    biases.resize(1, 4);
    biases.setZero();

    perceptron_layer.set_biases(biases);

    assert_true(perceptron_layer.get_biases_number() == 4, LOG);

    assert_true(abs(perceptron_layer.get_biases()(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_biases()(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test
    perceptron_layer.set(1, 4);

    biases.setValues({
                         {type(9)},
                         {type(-8)},
                         {type(7)},
                         {type(-6)}});

    perceptron_layer.set_biases(biases);

    assert_true(perceptron_layer.get_biases_number() == 4, LOG);

    assert_true(abs(perceptron_layer.get_biases()(0) - type(9)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_biases()(3) + type(6)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    perceptron_layer.set(2, 3);

    biases.resize(2, 3);
    biases.setValues({{type(9),type(1),type(-8)},{type(7),type(3),type(-6)}});

    perceptron_layer.set_biases(biases);

    assert_true(perceptron_layer.get_biases_number() == 6, LOG);

    assert_true(abs(perceptron_layer.get_biases()(0,0) - type(9)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_biases()(0,1) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_biases()(0,2) + type(8)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_biases()(1,0) - type(7)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_biases()(1,1) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_biases()(1,2) + type(6)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void PerceptronLayerTest::test_set_synaptic_weights()
{
    cout << "test_set_synaptic_weights\n";

    Tensor<type, 2> synaptic_weights;

    // Test

    perceptron_layer.set(1, 2);

    synaptic_weights.resize(2, 1);

    // Test
    synaptic_weights.setZero();

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    assert_true(perceptron_layer.get_synaptic_weights().size() == 2, LOG);

    assert_true(abs(perceptron_layer.get_synaptic_weights()(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test
    synaptic_weights.setValues({{type(-11)},{type(21)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);

    assert_true(perceptron_layer.get_synaptic_weights().size() == 2, LOG);

    assert_true(abs(perceptron_layer.get_synaptic_weights()(0) + type(11)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(1) - type(21)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test
    Tensor<type, 2> synaptic_weights_1(3,2);

    synaptic_weights_1.setValues({
                                     {type(1),type(-2)},
                                     {type(3),type(-4)},
                                     {type(5),type(-6)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights_1);

    assert_true(perceptron_layer.get_synaptic_weights().size() == 6, LOG);

    assert_true(abs(perceptron_layer.get_synaptic_weights()(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(0,1) + type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(1,0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(1,1) + type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(2,0) - type(5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(2,1) + type(6)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void PerceptronLayerTest::test_set_inputs_number()
{
    cout << "test_set_inputs_number\n";

    Tensor<type, 2> biases;
    Tensor<type, 2> synaptic_weights;

    Tensor<type, 2> new_biases;
    Tensor<type, 2> new_synaptic_weights;

    // Test

    perceptron_layer.set_biases(biases);
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_inputs_number = 0;

    perceptron_layer.set_inputs_number(new_inputs_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() == new_synaptic_weights.size(), LOG);

    // Test

    perceptron_layer.set(1,1);

    biases.resize(2, 2);
    biases.setValues({{type(7),type(3)},{type(-1),type(1)}});

    synaptic_weights.resize(1, 2);
    synaptic_weights.setValues({{type(-11)},{type(21)}});

    perceptron_layer.set_biases(biases);
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_inputs_number_1 = 1;

    perceptron_layer.set_inputs_number(new_inputs_number_1);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);

    // Test
    perceptron_layer.set(2, 2);

    biases.resize(1, 4);
    biases.setValues({
                         {type(9)},
                         {type(-8)},
                         {type(7)},
                         {type(-6)}});

    synaptic_weights.resize(2, 4);

    synaptic_weights.setValues({
                                   {type(-11), type(12), type(-13), type(14)},
                                   {type(21), type(-22), type(23), type(-24)}});

    perceptron_layer.set_biases(biases);
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_inputs_number_2 = 6;

    perceptron_layer.set_inputs_number(new_inputs_number_2);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
}


void PerceptronLayerTest::test_set_perceptrons_number()
{
    cout << "test_set_perceptrons_number\n";


    Tensor<type, 2> new_biases;
    Tensor<type, 2> new_synaptic_weights;

    // Test

    Tensor<type, 2> biases;
    Tensor<type, 2> synaptic_weights;

    perceptron_layer.set_biases(biases);
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    Index new_perceptrons_number = 0;

    perceptron_layer.set_inputs_number(new_perceptrons_number);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() == new_synaptic_weights.size(), LOG);

    // Test

    perceptron_layer.set(3, 2);

    Tensor<type, 2> biases_1(1, 4);

    biases_1.setValues({
                           {type(9)},
                           {type(-8)},
                           {type(7)},
                           {type(-6)}});

    Tensor<type, 2> synaptic_weights_1(2, 4);

    synaptic_weights_1.setValues({
                                     {type(-11), type(12), type(-13), type(14)},
                                     {type(21), type(-22), type(23), type(-24)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights_1);

    Index new_perceptrons_number_1= 1;

    perceptron_layer.set_neurons_number(new_perceptrons_number_1);

    new_biases = perceptron_layer.get_biases();
    new_synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases_1.size() != new_biases.size(), LOG);
    assert_true(synaptic_weights_1.size() != new_synaptic_weights.size(), LOG);
}


void PerceptronLayerTest::test_set_parameters()
{
    cout << "test_set_parameters\n";

    // Test

    Index neurons_number = 4;

    perceptron_layer.set(1, neurons_number);

    Tensor<type, 1> parameters_0(2 * neurons_number);

    parameters_0.setZero();

    perceptron_layer.set_parameters(parameters_0, 0);

    assert_true(abs(perceptron_layer.get_biases()(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(0) - parameters_0(0)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neurons_number = 2;

    perceptron_layer.set(1, neurons_number);

    Tensor<type, 1> parameters_1(2*neurons_number);

    parameters_1.setValues({ type(11),type(12),type(21),type(22)});

    perceptron_layer.set_parameters(parameters_1, 0);

    assert_true(abs(perceptron_layer.get_biases()(0) - parameters_1(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(perceptron_layer.get_synaptic_weights()(0) - parameters_1(2))  < type(NUMERIC_LIMITS_MIN), LOG);
}


void PerceptronLayerTest::test_set_activation_function()
{
    cout << "test_set_activation_function\n";

    string new_activation_function_name = "Threshold";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::Threshold, LOG);

    new_activation_function_name = "SymmetricThreshold";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::SymmetricThreshold, LOG);

    new_activation_function_name = "Logistic";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::Logistic, LOG);

    new_activation_function_name = "HyperbolicTangent";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::HyperbolicTangent, LOG);

    new_activation_function_name = "Linear";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::Linear, LOG);

    new_activation_function_name = "RectifiedLinear";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::RectifiedLinear, LOG);

    new_activation_function_name = "ExponentialLinear";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ExponentialLinear);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::ExponentialLinear, LOG);

    new_activation_function_name = "ScaledExponentialLinear";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ScaledExponentialLinear);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::ScaledExponentialLinear, LOG);

    new_activation_function_name = "SoftPlus";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftPlus);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::SoftPlus, LOG);

    new_activation_function_name = "SoftSign";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftSign);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::SoftSign, LOG);

    new_activation_function_name = "HardSigmoid";
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HardSigmoid);
    assert_true(perceptron_layer.get_activation_function() == PerceptronLayer::ActivationFunction::HardSigmoid, LOG);
}


void PerceptronLayerTest::test_set_parameters_constant()
{
    cout << "test_set_parameters_constant\n";


    Tensor<type, 1> parameters;

    // Test

    perceptron_layer.set(2, 1);
    perceptron_layer.set_parameters_constant(type(0));

    parameters = perceptron_layer.get_parameters();

    assert_true(static_cast<Index>(parameters(0)) == 0, LOG);
    assert_true(static_cast<Index>(parameters(2)) == 0, LOG);

    // Test

    perceptron_layer.set(1, 1);
    perceptron_layer.set_parameters_constant(type(5));

    parameters = perceptron_layer.get_parameters();

    assert_true(static_cast<Index>(parameters(0)) == 5, LOG);
    assert_true(static_cast<Index>(parameters(1)) == 5, LOG);
}


void PerceptronLayerTest::test_set_parameters_random()
{
    cout << "test_set_parameters_random\n";

    Tensor<type, 1> parameters;

    // Test
    perceptron_layer.set(1,1);
    perceptron_layer.set_parameters_random();

    parameters = perceptron_layer.get_parameters();

    assert_true((parameters(0) >= static_cast<type>(1e-5)) || (parameters(0) <= static_cast<type>(1e-5)), LOG);
    assert_true((parameters(1) >= static_cast<type>(1e-5)) || (parameters(1) <= static_cast<type>(1e-5)), LOG);
}


void PerceptronLayerTest::test_calculate_combinations()
{
    cout << "test_calculate_combinations\n";

    Index parameters_number;

    Tensor<type, 2> biases;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 1> parameters;

    Tensor<type, 2> inputs;
    Tensor<type, 2> combinations;

    Tensor<Index, 1> inputs_dimensions;
    Tensor<Index, 1> combinations_dims;

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

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(), inputs_dimensions);

    // Test

    biases.setConstant(type(1));
    synaptic_weights.setConstant(type(2));

    inputs.setConstant(type(3));

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(), inputs_dimensions);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(abs(combinations(0,0) - type(7)) < static_cast<type>(1e-5) , LOG);

    // Test

    inputs_number = 2;
    neurons_number = 2;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));
    inputs_dimensions = get_dimensions(inputs);

    combinations.resize(samples_number, neurons_number);
    combinations.setZero();

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(), inputs_dimensions);

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
    biases.resize(1,4);
    biases.setConstant(type(2));

    inputs.resize(2,3);
    inputs.setConstant(type(0.5));
    inputs_dimensions = get_dimensions(inputs);

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(), inputs_dimensions);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 2, LOG);
    assert_true(combinations.dimension(1) == 4, LOG);
    assert_true(abs(combinations(0,0) - static_cast<type>(3.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 2;
    neurons_number = 4;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number);

    synaptic_weights.resize(inputs_number, neurons_number);
    synaptic_weights.setConstant(type(1));
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    biases.resize(1, neurons_number);
    biases.setConstant(type(1));
    perceptron_layer.set_biases(biases);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({{type(0.5), type(0.5)}});
    inputs_dimensions = get_dimensions(inputs);

    combinations.resize(samples_number, neurons_number);
    combinations.setZero();

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(), inputs_dimensions);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == samples_number, LOG);
    assert_true(combinations.dimension(1) == neurons_number, LOG);
    assert_true(static_cast<Index>(combinations(0,0)) == 2, LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    samples_number = 1;

    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);

    synaptic_weights.resize(inputs_number, neurons_number);
    synaptic_weights.setValues({{type(1)}});
    perceptron_layer.set_synaptic_weights(synaptic_weights);

    biases.resize(1, neurons_number);
    biases.setValues({{type(-0.5)}});
    perceptron_layer.set_biases(biases);

    inputs.resize(samples_number, inputs_number);
    inputs.setValues({{type(-0.8)}});

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    assert_true(biases(0, 0) - type(-0.5) < type(NUMERIC_LIMITS_MIN), LOG);
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
    inputs_dimensions = get_dimensions(inputs);

    biases.resize(1, neurons_number);
    biases.setZero();

    synaptic_weights.resize(inputs_number, neurons_number);
    synaptic_weights.setZero();

    combinations.resize(samples_number, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(),inputs_dimensions);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(combinations.dimension(1) == 1, LOG);

    assert_true(abs(combinations(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void PerceptronLayerTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";

    Tensor<type, 2> biases;
    Tensor<type, 2> synaptic_weights;
    Tensor<type, 1> parameters;

    Tensor<type, 2> inputs;
    Tensor<type, 2> combinations;
    Tensor<type, 2> activations;

    Tensor<Index, 1> inputs_dimensions;
    Tensor<Index, 1> combinations_dims;
    Tensor<Index, 1> activations_dims;

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
    inputs_dimensions = get_dimensions(inputs);

    combinations.resize(samples_number, neurons_number);

    activations.resize(samples_number, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);
    activations_dims = get_dimensions(activations);

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(), inputs_dimensions);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == 1, LOG);
    assert_true(activations.dimension(1) == 1, LOG);
    assert_true(static_cast<Index>(activations(0,0)) == 2 , LOG);

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

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);
    activations_dims = get_dimensions(activations);

    biases = perceptron_layer.get_biases();
    synaptic_weights = perceptron_layer.get_synaptic_weights();

    perceptron_layer.calculate_combinations(inputs.data(), biases, synaptic_weights, combinations.data(), inputs_dimensions);
    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(static_cast<Index>(activations(0,0)) == 10, LOG);

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

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);
    activations_dims = get_dimensions(activations);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(static_cast<Index>(activations(0,0)) == 0, LOG);

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

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);
    activations_dims = get_dimensions(activations);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(static_cast<Index>(activations(0,0)) == 1 , LOG);

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

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);
    activations_dims = get_dimensions(activations);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == samples_number, LOG);
    assert_true(activations.dimension(1) == neurons_number, LOG);
    assert_true(static_cast<Index>(activations(0,0))  == -1, LOG);

    // Test

    perceptron_layer.set(1, 2);
    perceptron_layer.set_parameters_constant(type(0));

    combinations.resize(2,2);
    combinations.setConstant(4.0);

    activations.resize(2,2);
    activations.setZero();

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);
    activations_dims = get_dimensions(activations);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == 2, LOG);
    assert_true(activations.dimension(1) == 2, LOG);
    assert_true(static_cast<Index>(activations(0,0)) == 4.0, LOG);

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

    inputs_dimensions = get_dimensions(inputs);
    combinations_dims = get_dimensions(combinations);
    activations_dims = get_dimensions(activations);

    perceptron_layer.calculate_combinations(inputs.data(), perceptron_layer.get_biases(),
                                            perceptron_layer.get_synaptic_weights(), combinations.data(), inputs_dimensions);

    assert_true(combinations.rank() == 2, LOG);
    assert_true(combinations.dimension(0) == 1, LOG);
    assert_true(combinations.dimension(1) == 2, LOG);
    assert_true(abs(combinations(0,0) - static_cast<type>(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(activations.rank() == 2, LOG);
    assert_true(activations.dimension(0) == 1, LOG);
    assert_true(activations.dimension(1) == 2, LOG);
    assert_true(static_cast<Index>(activations(0,0)) == 1, LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(static_cast<Index>(activations(0,0)) == 1, LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::PerceptronLayer::ActivationFunction::Logistic);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(abs(activations(0,0) - static_cast<type>(1.0/(1.0+exp(-2.5)))) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(abs(activations(0,0) - static_cast<type>(tanh(2.5))) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    perceptron_layer.calculate_activations(combinations.data(), combinations_dims, activations.data(), activations_dims);

    assert_true(abs(activations(0,0) - static_cast<type>(2.5)) < type(NUMERIC_LIMITS_MIN), LOG);

}


void PerceptronLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activations_derivatives\n";

    Tensor<type, 1> parameters(1);
    Tensor<type, 2> inputs(1,1);
    Tensor<type, 2> combinations(1,1);
    Tensor<type, 2> activations(1,1);
    Tensor<type, 2> activations_derivatives(1,1);

    Tensor<Index, 1> dims(2);
    dims = get_dimensions(activations_derivatives);

    // Test

    perceptron_layer.set(1,1);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.setConstant(type(1));

    combinations.setConstant(type(1));

    activations_derivatives.setZero();

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

    assert_true(activations_derivatives.rank() == 2, LOG);
    assert_true(activations_derivatives.dimension(0) == 1, LOG);
    assert_true(activations_derivatives.dimension(1) == 1, LOG);
    assert_true(abs(activations(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(activations_derivatives(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - type(1)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.196)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(0.761)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.41997)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(activations_derivatives(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ExponentialLinear);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ScaledExponentialLinear);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(1.05)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(1.05)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftPlus);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(1.313)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftSign);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HardSigmoid);
    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);
    assert_true(abs(activations(0,0) - static_cast<type>(0.7)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(activations_derivatives(0,0) - static_cast<type>(0.2)) < static_cast<type>(1e-3), LOG);

    // Test

    perceptron_layer.set(2, 4);
    perceptron_layer.set_parameters_constant(type(1));

    combinations.resize(1,4);
    combinations.setValues({{type(1.56f), type(-0.68f), type(0.91f), type(-1.99f)}});

    activations.resize(1,4);

    activations_derivatives.resize(1,4);
    activations_derivatives.setZero();

    dims = get_dimensions(activations_derivatives);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    Tensor<type, 2> numerical_activation_derivative(1,4);
//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SymmetricThreshold);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ExponentialLinear);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::ScaledExponentialLinear);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test
//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftPlus);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test

//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::SoftSign);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);

//    // Test

//    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::HardSigmoid);
//    perceptron_layer.calculate_activations_derivatives(combinations.data(), dims, activations.data(), dims, activations_derivatives.data(), dims);

//    numerical_activation_derivative
//            = numerical_differentiation.calculate_derivatives(perceptron_layer, &PerceptronLayer::calculate_activations, 0, combinations);

//    assert_true(activations_derivatives(0,0) - numerical_activation_derivative(0,0) < static_cast<type>(1e-3), LOG);
}

/*
void PerceptronLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    Tensor<type, 2> biases;
    Tensor<type, 2> synaptic_weights;

    Tensor<type, 2> inputs;
    Tensor<type, 1> parameters;

    Tensor<Index, 1> outputs_dimensions;

    // Test

    inputs_number = 3;
    neurons_number = 4;

    perceptron_layer.set(inputs_number, neurons_number);

    synaptic_weights.resize(inputs_number, neurons_number);
    biases.resize(1, neurons_number);
    inputs.resize(1, inputs_number);

    inputs.setConstant(type(1));
    biases.setConstant(type(1));
    synaptic_weights.setValues({{type(1),type(-1),type(0),type(1)},
                                {type(2),type(-2),type(0),type(2)},
                                {type(3),type(-3),type(0),type(3)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    perceptron_layer.forward_propagate(inputs.data(), inputs_dimensions, &forward_propagation);

    const TensorMap<Tensor<type, 2>> outputs(forward_propagation.outputs_data, 1, neurons_number);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs(0,0)) == 7, LOG);
    assert_true(static_cast<Index>(outputs(1,0)) == -5, LOG);
    assert_true(static_cast<Index>(outputs(2,0)) == 1, LOG);

    // Test

    inputs_number = 2;
    neurons_number = 4;

    biases.resize(1, neurons_number);
    biases.setValues({
                         {type(9)},
                         {type(-8)},
                         {type(7)},
                         {type(-6)}});

    synaptic_weights.resize(2, 4);

    synaptic_weights.resize(2, 4);
    synaptic_weights.setValues({
                                   {type(-11), type(12), type(-13), type(14)},
                                   {type(21), type(-22), type(23), type(-24)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    inputs.resize(1, 2);
    inputs.setConstant(type(1));

//    outputs.resize(1, neurons_number);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs(0,0)) == 1, LOG);
    assert_true(static_cast<Index>(outputs(1,0)) == 0, LOG);
    assert_true(static_cast<Index>(outputs(2,0)) == 1, LOG);

    // Test

    inputs_number = 3;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(0));

    inputs.resize(1, inputs_number);
    inputs.setConstant(type(0));

    outputs.resize(1, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    assert_true(outputs.rank() == 2, LOG);
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

    inputs.resize(1,4);
    inputs.setValues({{type(4),type(-3),type(2),type(-1)}});

    outputs.resize(1, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 2, LOG);
    assert_true(abs(outputs(0,0) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test 5

    inputs_number = 1;
    neurons_number = 2;

    inputs.resize(1, inputs_number);
    inputs.setConstant(type(3.0));

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(-2.0));

    outputs.resize(1, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    parameters.resize(2);
    parameters.setConstant(type(1));

    // Test

    perceptron_layer.set(1, 1);

    inputs.resize(1,1);
    inputs.setRandom();

    parameters = perceptron_layer.get_parameters();
}
*/

void PerceptronLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    Tensor<type, 1> parameters;
    Tensor<type, 2> inputs;
    Tensor<Index, 1> inputs_dimensions;

    Tensor<type, 1> potential_parameters;

    // Test

    samples_number = 2;
    inputs_number = 2;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::Linear);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));
    inputs_dimensions = get_dimensions(inputs);

    forward_propagation.set(samples_number, &perceptron_layer);

    perceptron_layer.forward_propagate(inputs.data(), inputs_dimensions, &forward_propagation);

    assert_true(forward_propagation.combinations.rank() == 2, LOG);
    assert_true(forward_propagation.combinations.dimension(0) == samples_number, LOG);
    assert_true(forward_propagation.combinations.dimension(1) == neurons_number, LOG);

    assert_true(abs(forward_propagation.combinations(0,0) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.combinations(0,1) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);

    TensorMap<Tensor<type, 2>> outputs(forward_propagation.outputs_data, forward_propagation.outputs_dimensions(0), forward_propagation.outputs_dimensions(1));

    assert_true(abs(outputs(0,0) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);

    assert_true(abs(outputs(0,1) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);

    assert_true(abs(forward_propagation.activations_derivatives(0,0) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.activations_derivatives(0,1) - static_cast<type>(1)) < static_cast<type>(1e-3), LOG);

    // Test

    samples_number = 2;
    inputs_number = 2;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);
    perceptron_layer.set_parameters_constant(type(1));

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));
    inputs_dimensions = get_dimensions(inputs);

//    potential_parameters = perceptron_layer.get_parameters();

    forward_propagation.set(samples_number, &perceptron_layer);

    perceptron_layer.forward_propagate(inputs.data(), inputs_dimensions, &forward_propagation);

    TensorMap< Tensor<type, 2>> outputs_2(forward_propagation.outputs_data, forward_propagation.outputs_dimensions(0), forward_propagation.outputs_dimensions(1));

    assert_true(forward_propagation.combinations.rank() == 2, LOG);
    assert_true(forward_propagation.combinations.dimension(0) == samples_number, LOG);
    assert_true(forward_propagation.combinations.dimension(1) == neurons_number, LOG);
    assert_true(abs(forward_propagation.combinations(0,0) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.combinations(0,1) - static_cast<type>(3)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(outputs_2(0,0) - static_cast<type>(0.99505)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(outputs_2(0,1) - static_cast<type>(0.99505)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.activations_derivatives(0,0) - static_cast<type>(0.00986)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(forward_propagation.activations_derivatives(0,1) - static_cast<type>(0.00986)) < static_cast<type>(1e-3), LOG);

    /*
    cout << "test_calculate_outputs\n";

    Tensor<type, 2> biases;
    Tensor<type, 2> synaptic_weights;

    Tensor<type, 2> inputs;
    Tensor<type, 1> parameters;

    Tensor<Index, 1> outputs_dimensions;

    // Test

    inputs_number = 3;
    neurons_number = 4;

    perceptron_layer.set(inputs_number, neurons_number);

    synaptic_weights.resize(inputs_number, neurons_number);
    biases.resize(1, neurons_number);
    inputs.resize(1, inputs_number);

    inputs.setConstant(type(1));
    biases.setConstant(type(1));
    synaptic_weights.setValues({{type(1),type(-1),type(0),type(1)},
                                {type(2),type(-2),type(0),type(2)},
                                {type(3),type(-3),type(0),type(3)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    perceptron_layer.forward_propagate(inputs.data(), inputs_dimensions, &forward_propagation);

    const TensorMap<Tensor<type, 2>> outputs(forward_propagation.outputs_data, 1, neurons_number);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs(0,0)) == 7, LOG);
    assert_true(static_cast<Index>(outputs(1,0)) == -5, LOG);
    assert_true(static_cast<Index>(outputs(2,0)) == 1, LOG);

    // Test

    inputs_number = 2;
    neurons_number = 4;

    biases.resize(1, neurons_number);
    biases.setValues({
                         {type(9)},
                         {type(-8)},
                         {type(7)},
                         {type(-6)}});

    synaptic_weights.resize(2, 4);

    synaptic_weights.resize(2, 4);
    synaptic_weights.setValues({
                                   {type(-11), type(12), type(-13), type(14)},
                                   {type(21), type(-22), type(23), type(-24)}});

    perceptron_layer.set_synaptic_weights(synaptic_weights);
    perceptron_layer.set_biases(biases);

    inputs.resize(1, 2);
    inputs.setConstant(type(1));

//    outputs.resize(1, neurons_number);

    perceptron_layer.set_activation_function(PerceptronLayer::ActivationFunction::Threshold);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 4, LOG);
    assert_true(static_cast<Index>(outputs(0,0)) == 1, LOG);
    assert_true(static_cast<Index>(outputs(1,0)) == 0, LOG);
    assert_true(static_cast<Index>(outputs(2,0)) == 1, LOG);

    // Test

    inputs_number = 3;
    neurons_number = 2;

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(0));

    inputs.resize(1, inputs_number);
    inputs.setConstant(type(0));

    outputs.resize(1, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    assert_true(outputs.rank() == 2, LOG);
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

    inputs.resize(1,4);
    inputs.setValues({{type(4),type(-3),type(2),type(-1)}});

    outputs.resize(1, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    assert_true(outputs.rank() == 2, LOG);
    assert_true(outputs.dimension(0) == 1, LOG);
    assert_true(outputs.dimension(1) == 2, LOG);
    assert_true(abs(outputs(0,0) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test 5

    inputs_number = 1;
    neurons_number = 2;

    inputs.resize(1, inputs_number);
    inputs.setConstant(type(3.0));

    perceptron_layer.set(inputs_number, neurons_number);
    perceptron_layer.set_parameters_constant(type(-2.0));

    outputs.resize(1, neurons_number);

    inputs_dimensions = get_dimensions(inputs);
    outputs_dimensions = get_dimensions(outputs);

//    perceptron_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    parameters.resize(2);
    parameters.setConstant(type(1));

    // Test

    perceptron_layer.set(1, 1);

    inputs.resize(1,1);
    inputs.setRandom();

    parameters = perceptron_layer.get_parameters();*/
}


void PerceptronLayerTest::run_test_case()
{
    cout << "Running perceptron layer test case...\n";

    // Constructor and destructor

    test_constructor();
    test_destructor();

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

    // Parameters initialization methods

    test_set_parameters_constant();
    test_set_parameters_random();

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
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
