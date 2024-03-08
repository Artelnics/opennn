//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "transformer_test.h"

TransformerTest::TransformerTest() : UnitTesting()
{
}


TransformerTest::~TransformerTest()
{
}


void TransformerTest::test_constructor()
{
    cout << "test_constructor\n";

    Tensor<Layer*, 1> layers;

    // Default constructor

    Transformer transformer_0;

    assert_true(transformer_0.is_empty(), LOG);
    assert_true(transformer_0.get_layers_number() == 0, LOG);

    // Tensor constructor test

    inputs_length = 1;
    context_length = 1;
    inputs_dimension = 1;
    context_dimension = 1;
    embedding_depth = 1;
    perceptron_depth = 1;
    heads_number = 1;
    number_of_layers = 1;

    Tensor<Index, 1> architecture(8);
    architecture.setValues({ inputs_length, context_length, inputs_dimension, context_dimension,
                             embedding_depth, perceptron_depth, heads_number, number_of_layers });

    Transformer transformer_1(architecture);

    assert_true(transformer_1.get_layers_number() == 2 + 3 * number_of_layers + 4 * number_of_layers + 1, LOG);

    // List constructor test

    Transformer transformer_2({ inputs_length, context_length, inputs_dimension, context_dimension,
                                embedding_depth, perceptron_depth, heads_number, number_of_layers });

    assert_true(transformer_2.get_layers_number() == 2 + 3 * number_of_layers + 4 * number_of_layers + 1, LOG);

    // Test 3

    inputs_length = 2;
    context_length = 3;
    inputs_dimension = 5;
    context_dimension = 6;
    embedding_depth = 10;
    perceptron_depth = 12;
    heads_number = 4;
    number_of_layers = 1;

    Transformer transformer_3({ inputs_length, context_length, inputs_dimension, context_dimension,
                                embedding_depth, perceptron_depth, heads_number, number_of_layers });

    assert_true(transformer_3.get_layers_number() == 2 + 3 * number_of_layers + 4 * number_of_layers + 1, LOG);

    // Test 4

    number_of_layers = 3;

    Transformer transformer_4({ inputs_length, context_length, inputs_dimension, context_dimension,
                                embedding_depth, perceptron_depth, heads_number, number_of_layers });

    assert_true(transformer_4.get_layers_number() == 2 + 3 * number_of_layers + 4 * number_of_layers + 1, LOG);
}


void TransformerTest::test_destructor()
{
    cout << "test_destructor\n";

    Transformer* transformer = new Transformer;

    delete transformer;
}


void TransformerTest::test_calculate_parameters_norm()
{
    cout << "test_calculate_parameters_norm\n";

    type parameters_norm = type(0);

    // Test
    {
        inputs_length = 1;
        context_length = 1;
        perceptron_depth = 1;
        heads_number = 1;

        inputs_dimension = 0;
        context_dimension = 0;
        embedding_depth = 0;
        number_of_layers = 0;

        transformer.set({ inputs_length, context_length, inputs_dimension, context_dimension,
                          embedding_depth, perceptron_depth, heads_number, number_of_layers });

        parameters_norm = transformer.calculate_parameters_norm();

        assert_true(abs(parameters_norm) < type(NUMERIC_LIMITS_MIN), LOG);
    }
    // Test
    
    {
        inputs_dimension = 1;
        context_dimension = 1;
        embedding_depth = 1;
        number_of_layers = 1;

        transformer.set({ inputs_length, context_length, inputs_dimension, context_dimension,
                          embedding_depth, perceptron_depth, heads_number, number_of_layers });

        transformer.set_parameters_constant(type(1));

        parameters_norm = transformer.calculate_parameters_norm();

        Index parameters_number = transformer.get_parameters_number();

        assert_true(abs(parameters_norm - sqrt(type(parameters_number))) < type(NUMERIC_LIMITS_MIN), LOG);
    }
}


void TransformerTest::test_calculate_outputs()
{/*
    cout << "test_calculate_outputs\n";

    Tensor<type, 2> input;
    Tensor<type, 2> context;
    Tensor<type, 2> outputs;

    Index parameters_number;

    Tensor<type, 1> parameters;

    // Test two layers perceptron with all zeros

    inputs_length = 1;
    context_length = 1;
    inputs_dimension = 1;
    context_dimension = 1;
    embedding_depth = 1;
    perceptron_depth = 1;
    heads_number = 1;
    number_of_layers = 1;

    transformer.set({ inputs_length, context_length, inputs_dimension, context_dimension,
                      embedding_depth, perceptron_depth, heads_number, number_of_layers });
    transformer.set_parameters_constant(type(0));

    input.resize(batch_samples_number, inputs_length);
    input.setConstant(type(0));

    context.resize(batch_samples_number, inputs_length);
    context.setConstant(type(0));

    outputs = transformer.calculate_outputs(input);

    assert_true(outputs.dimension(0) == batch_samples_number, LOG);
    assert_true(outputs.dimension(1) == inputs_length, LOG);
    assert_true(outputs.dimension(2) == inputs_dimension, LOG);

    //assert_true(outputs.abs() < type(NUMERIC_LIMITS_MIN), LOG);
    
    // Test

    batch_samples_number = 3;
    inputs_number = 2;
    neurons_number = 4;
    outputs_number = 5;

    transformer.set(Transformer::ModelType::Approximation, { inputs_number, neurons_number, outputs_number });

    transformer.set_parameters_constant(type(0));

    inputs.resize(batch_samples_number, inputs_number);
    inputs.setConstant(type(0));

    outputs = transformer.calculate_outputs(inputs);

    assert_true(outputs.size() == batch_samples_number * outputs_number, LOG);
    assert_true(abs(outputs(0, 0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0, 1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0, 2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0, 3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0, 4)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    transformer.set(Transformer::ModelType::Approximation, { 1, 2 });

    transformer.set_parameters_constant(type(1));

    inputs.resize(1, 1);
    inputs.setConstant(2);

    outputs = transformer.calculate_outputs(inputs);

    assert_true(outputs.size() == 2, LOG);
    assert_true(abs(outputs(0, 0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0, 1) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    transformer.set(Transformer::ModelType::Approximation, { 4, 3, 3 });

    inputs.resize(1, 4);
    inputs.setZero();

    transformer.set_parameters_constant(type(1));

    outputs = transformer.calculate_outputs(inputs);

    assert_true(outputs.size() == 3, LOG);

    assert_true(abs(outputs(0, 0) - 3.2847) < type(1e-3), LOG);
    assert_true(abs(outputs(0, 1) - 3.2847) < type(1e-3), LOG);
    assert_true(abs(outputs(0, 2) - 3.2847) < type(1e-3), LOG);

    // Test

    transformer.set(Transformer::ModelType::Approximation, { 1, 2 });

    inputs_number = transformer.get_inputs_number();
    parameters_number = transformer.get_parameters_number();
    outputs_number = transformer.get_outputs_number();

    inputs.resize(1, inputs_number);
    inputs.setZero();

    parameters.resize(parameters_number);
    parameters.setConstant(type(0));

    transformer.set_parameters(parameters);

    outputs = transformer.calculate_outputs(inputs);

    assert_true(outputs.size() == outputs_number, LOG);
    assert_true(abs(outputs(0, 0) - 0) < type(1e-3), LOG);
    assert_true(abs(outputs(0, 1) - 0) < type(1e-3), LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    transformer.set(Transformer::ModelType::Approximation, { inputs_number, neurons_number, outputs_number });

    transformer.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();

    outputs = transformer.calculate_outputs(inputs);

    assert_true(outputs.size() == 1, LOG);
    assert_true(abs(outputs(0, 0) - 0) < type(1e-3), LOG);

    // Test

    transformer.set(Transformer::ModelType::Classification, { 1,1 });

    transformer.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();

    outputs = transformer.calculate_outputs(inputs);

    assert_true(outputs.size() == 1, LOG);
    assert_true(abs(outputs(0, 0) - type(0.5)) < type(1e-3), LOG);

    // Test 7

    transformer.set(Transformer::ModelType::Approximation, { 1,3,3,3,1 });

    batch_samples_number = 2;
    inputs_number = transformer.get_inputs_number();
    outputs_number = transformer.get_outputs_number();

    inputs.resize(batch_samples_number, inputs_number);
    inputs.setConstant(type(0));

    parameters_number = transformer.get_parameters_number();
    parameters.resize(parameters_number);
    parameters.setConstant(type(0));

    transformer.set_parameters(parameters);

    outputs = transformer.calculate_outputs(inputs);

    assert_true(outputs.dimension(1) == outputs_number, LOG);
    assert_true(abs(outputs(0, 0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1, 0)) < type(NUMERIC_LIMITS_MIN), LOG);

*/}


void TransformerTest::test_calculate_directional_inputs()
{/*
    cout << "test_calculate_directional_inputs\n";

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;
    Tensor<type, 2> trainable_outputs;

    Tensor<type, 1> parameters;

    Tensor<type, 1> point;

    Tensor<type, 2> directional_inputs;

    // Test

    transformer.set(Transformer::ModelType::Approximation, { 3, 4, 2 });
    transformer.set_parameters_constant(type(0));

    inputs.resize(2, 3);
    inputs.setValues({ {type(-5),type(-1),-type(3)},
                      {type(7),type(3),type(1)} });

    point.resize(3);
    point.setValues({ type(0),type(0),type(0) });

    directional_inputs = transformer.calculate_directional_inputs(0, point, type(0), type(0), 0);

    assert_true(directional_inputs.rank() == 2, LOG);
    assert_true(directional_inputs.dimension(0) == 0, LOG);

    // Test

    point.setValues({ type(1), type(2), type(3) });

    directional_inputs = transformer.calculate_directional_inputs(2, point, type(-1), type(1), 3);

    assert_true(directional_inputs.rank() == 2, LOG);
    assert_true(directional_inputs.dimension(0) == 3, LOG);
    assert_true(directional_inputs.dimension(1) == 3, LOG);
    assert_true(abs(directional_inputs(0, 2) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(1, 2) - type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(2, 2) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    point.setValues({ type(1), type(2), type(3) });

    directional_inputs = transformer.calculate_directional_inputs(0, point, type(-4), type(0), 5);

    assert_true(directional_inputs.rank() == 2, LOG);
    assert_true(directional_inputs.dimension(0) == 5, LOG);
    assert_true(directional_inputs.dimension(1) == 3, LOG);
    assert_true(abs(directional_inputs(0, 0) + type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(1, 0) + type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(2, 0) + type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(3, 0) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(4, 0) + type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
*/}


void TransformerTest::test_forward_propagate()
{

    cout << "test_forward_propagate\n";

    {
        // Test

        batch_samples_number = 1;

        inputs_length = 2;
        context_length = 3;
        inputs_dimension = 5;
        context_dimension = 6;

        embedding_depth = 10;
        perceptron_depth = 12;
        heads_number = 4;
        number_of_layers = 1;

        bool is_training = false;

        data.resize(batch_samples_number, context_length + 2 * inputs_length);

        for (Index i = 0; i < batch_samples_number; i++)
        {
            for (Index j = 0; j < context_length; j++)
                data(i, j) = type(rand() % context_dimension);

            for(Index j = 0; j < 2 * inputs_length; j++)
                data(i, j + context_length) = type(rand() % inputs_dimension);
        }
        
        data_set.set(data);

        data_set.set_training();

        for (Index i = 0; i < context_length; i++)
            data_set.set_raw_variable_use(i, DataSet::VariableUse::Context);

        for (Index i = 0; i < inputs_length; i++)
            data_set.set_raw_variable_use(i + context_length, DataSet::VariableUse::Input);

        for (Index i = 0; i < inputs_length; i++)
            data_set.set_raw_variable_use(i + context_length + inputs_length, DataSet::VariableUse::Target);

        training_samples_indices = data_set.get_training_samples_indices();
        context_variables_indices = data_set.get_context_variables_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices, context_variables_indices);

        Tensor<pair<type*, dimensions>, 1>& input_pair = batch.get_inputs_pair();
        TensorMap<Tensor<type, 2>> input(input_pair(0).first, input_pair(0).second[0], input_pair(0).second[1]);
        cout << "Input:" << endl << input << endl;

        Tensor<pair<type*, dimensions>, 1>& context_pair = batch.get_context_pair();
        TensorMap<Tensor<type, 2>> context(context_pair(0).first, context_pair(0).second[0], context_pair(0).second[1]);
        cout << "Context:" << endl << context << endl;

        transformer.set({ inputs_length, context_length, inputs_dimension, context_dimension,
                          embedding_depth, perceptron_depth, heads_number, number_of_layers });

        ForwardPropagation forward_propagation(data_set.get_training_samples_number(), &transformer);

        cout << "Before forward propagate" << endl;
        transformer.forward_propagate(batch, forward_propagation, is_training);
        cout << "After forward propagate" << endl;
        /*
        ProbabilisticLayer3DForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation.layers[transformer.get_layers_number() - 1]);

        Tensor<type, 2> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;

        assert_true(probabilistic_activations.rank() == 3, LOG);
        assert_true(probabilistic_activations.dimension(0) == batch_samples_number, LOG);
        assert_true(probabilistic_activations.dimension(1) == inputs_length, LOG);
        assert_true(probabilistic_activations.dimension(2) == inputs_dimension, LOG);
        /*
        assert_true(abs(perceptron_activations(0, 0) - type(0.952)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(1, 0) - type(0.993)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(2, 0) - type(0.999)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(3, 0) - type(0.731)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(4, 0) - type(0.731)) < type(1e-3), LOG);
        */
    }
    /*
    {
        // Test

        inputs_number = 4;
        outputs_number = 2;
        batch_samples_number = 3;
        bool is_training = false;

        data.resize(batch_samples_number, inputs_number + outputs_number);
        data.setValues({ {-1,1,-1,1,1,0},{-2,2,3,1,1,0},{-3,3,5,1,1,0} });
        data_set.set(data);
        data_set.set_target();
        data_set.set_training();

        Tensor<Index, 1> input_raw_variables_indices(inputs_number);
        input_raw_variables_indices.setValues({ 0,1,2,3 });

        Tensor<bool, 1> input_raw_variables_use(4);
        input_raw_variables_use.setConstant(true);

        data_set.set_input_raw_variables(input_raw_variables_indices, input_raw_variables_use);

        training_samples_indices = data_set.get_training_samples_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        transformer.set();

        PerceptronLayer* perceptron_layer = new PerceptronLayer(inputs_number, outputs_number);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
        const Index neurons_number_perceptron = perceptron_layer->get_neurons_number();

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(outputs_number, outputs_number);
        probabilistic_layer->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
        const Index neurons_number_probabilistic = probabilistic_layer->get_neurons_number();

        Tensor<type, 1> biases_perceptron(outputs_number);
        biases_perceptron.setConstant(5);
        perceptron_layer->set_biases(biases_perceptron);

        Tensor<type, 2> synaptic_weights_perceptron(inputs_number, neurons_number_perceptron);
        synaptic_weights_perceptron.setConstant(type(-1));
        perceptron_layer->set_synaptic_weights(synaptic_weights_perceptron);

        Tensor<type, 1> biases_probabilistic(outputs_number);
        biases_probabilistic.setConstant(3);
        probabilistic_layer->set_biases(biases_probabilistic);

        Tensor<type, 2> synaptic_weights_probabilistic(neurons_number_perceptron, neurons_number_probabilistic);
        synaptic_weights_probabilistic.setConstant(type(1));
        probabilistic_layer->set_synaptic_weights(synaptic_weights_probabilistic);

        Tensor<Layer*, 1> layers(2);
        layers.setValues({ perceptron_layer, probabilistic_layer });
        transformer.set_layers(layers);

        ForwardPropagation forward_propagation(data_set.get_training_samples_number(), &transformer);

        transformer.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
            = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers[0]);

        Tensor<type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

        ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.layers[1]);

        Tensor<type, 2> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;

        assert_true(perceptron_activations.dimension(0) == 3, LOG);
        assert_true(abs(perceptron_activations(0, 0) - type(0.993)) < type(1e-3)
            && abs(perceptron_activations(1, 0) - type(0.731)) < type(1e-3)
            && abs(perceptron_activations(2, 0) - type(0.268)) < type(1e-3), LOG);

        assert_true(probabilistic_activations.dimension(0) == 3, LOG);
        assert_true(abs(probabilistic_activations(0, 0) - 0.5) < type(1e-3)
            && abs(probabilistic_activations(1, 0) - 0.5) < type(1e-3)
            && abs(probabilistic_activations(2, 0) - 0.5) < type(1e-3), LOG);
    }
*/}


void TransformerTest::run_test_case()
{
    cout << "Running transformer test case...\n";

    // Constructor and destructor methods

    test_constructor();

    test_destructor();

    // Parameters norm / descriptives / histogram

    test_calculate_parameters_norm();

    // Output
    /*
    test_calculate_outputs();

    test_calculate_directional_inputs();
    */
    //Forward propagate

    test_forward_propagate();

    cout << "End of transformer test case.\n\n";
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

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
