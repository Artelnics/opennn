//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "transformer_test.h"

#include "../opennn/probabilistic_layer_3d.h"

namespace opennn
{

TransformerTest::TransformerTest() : UnitTesting()
{
}


void TransformerTest::test_constructor()
{
    cout << "test_constructor\n";
    
    Tensor<unique_ptr<Layer>, 1> layers;

    Transformer transformer_0;

    assert_true(transformer_0.is_empty(), LOG);
    assert_true(transformer_0.get_layers_number() == 0, LOG);
    
    // Tensor constructor test

    input_length = 1;
    context_length = 1;
    input_dimensions = 1;
    context_dimension = 1;
    embedding_depth = 1;
    perceptron_depth = 1;
    heads_number = 1;
    layers_number = 1;

    Tensor<Index, 1> architecture(8);
    architecture.setValues({ input_length, context_length, input_dimensions, context_dimension,
                             embedding_depth, perceptron_depth, heads_number, layers_number });
    
    Transformer transformer_1(architecture);
    
    assert_true(transformer_1.get_layers_number() == 2 + 7 * layers_number + 10 * layers_number + 1, LOG);
    
    // List constructor test

    Transformer transformer_2({ input_length, context_length, input_dimensions, context_dimension,
                                embedding_depth, perceptron_depth, heads_number, layers_number });

    assert_true(transformer_2.get_layers_number() == 2 + 7 * layers_number + 10 * layers_number + 1, LOG);

    // Test 3

    input_length = 2;
    context_length = 3;
    input_dimensions = 5;
    context_dimension = 6;
    embedding_depth = 10;
    perceptron_depth = 12;
    heads_number = 4;
    layers_number = 1;

    Transformer transformer_3({ input_length, context_length, input_dimensions, context_dimension,
                                embedding_depth, perceptron_depth, heads_number, layers_number });

    assert_true(transformer_3.get_layers_number() == 2 + 7 * layers_number + 10 * layers_number + 1, LOG);

    // Test 4

    layers_number = 3;

    Transformer transformer_4({ input_length, context_length, input_dimensions, context_dimension,
                                embedding_depth, perceptron_depth, heads_number, layers_number });

    assert_true(transformer_4.get_layers_number() == 2 + 7 * layers_number + 10 * layers_number + 1, LOG);
}


void TransformerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    Tensor<type, 2> inputs;
    Tensor<type, 2> context;
    Tensor<type, 2> outputs;

    Index parameters_number;

    Tensor<type, 1> parameters;

    // Test two layers perceptron with all zeros

    input_length = 1;
    context_length = 1;
    input_dimensions = 1;
    context_dimension = 1;
    embedding_depth = 1;
    perceptron_depth = 1;
    heads_number = 1;
    layers_number = 1;
/*
    transformer.set({ input_length, context_length, input_dimensions, context_dimension,
                      embedding_depth, perceptron_depth, heads_number, layers_number });
    transformer.set_parameters_constant(type(0));

    inputs.resize(batch_samples_number, input_length);
    inputs.setConstant(type(0));

    context.resize(batch_samples_number, input_length);
    context.setConstant(type(0));

    outputs = transformer.calculate_outputs(inputs, context);

    assert_true(outputs.dimension(0) == batch_samples_number, LOG);
    assert_true(outputs.dimension(1) == input_length, LOG);
    assert_true(outputs.dimension(2) == input_dimensions, LOG);

    //assert_true(outputs.abs() < type(NUMERIC_LIMITS_MIN), LOG);
    
    // Test

    batch_samples_number = 3;
    Index inputs_number = 2;
    Index neurons_number = 4;
    Index outputs_number = 5;

    transformer.set(Transformer::ModelType::Approximation, { inputs_number}, {neurons_number}, {outputs_number });

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

    transformer.set(Transformer::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});

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



void TransformerTest::test_forward_propagate()
{

    cout << "test_forward_propagate\n";

    {
        // Test

        batch_samples_number = 1;

        input_length = 4;
        context_length = 3;
        input_dimensions = 5;
        context_dimension = 6;

        embedding_depth = 4;
        perceptron_depth = 6;
        heads_number = 4;
        layers_number = 1;

        bool is_training = true;

        data.resize(batch_samples_number, context_length + 2 * input_length);

        for(Index i = 0; i < batch_samples_number; i++)
        {
            for(Index j = 0; j < context_length; j++)
                data(i, j) = type(rand() % context_dimension);
        
            for(Index j = 0; j < 2 * input_length; j++)
                data(i, j + context_length) = type(rand() % input_dimensions);
        }
        
        data_set.set(data);

        data_set.set_training();

        for(Index i = 0; i < context_length; i++)
            data_set.set_raw_variable_use(i, DataSet::VariableUse::Context);

        for(Index i = 0; i < input_length; i++)
            data_set.set_raw_variable_use(i + context_length, DataSet::VariableUse::Input);

        for(Index i = 0; i < input_length; i++)
            data_set.set_raw_variable_use(i + context_length + input_length, DataSet::VariableUse::Target);

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        context_variables_indices = data_set.get_context_variables_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices, context_variables_indices);
        
        transformer.set({ input_length, context_length, input_dimensions, context_dimension,
                          embedding_depth, perceptron_depth, heads_number, layers_number });

        ForwardPropagation forward_propagation(data_set.get_samples_number(DataSet::SampleUse::Training), &transformer);

        transformer.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);
/*
        ProbabilisticLayer3DForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation.layers[transformer.get_layers_number() - 1]);
        
        Tensor<type, 3> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;
        
        assert_true(probabilistic_activations.rank() == 3, LOG);
        assert_true(probabilistic_activations.dimension(0) == batch_samples_number, LOG);
        assert_true(probabilistic_activations.dimension(1) == input_length, LOG);
        assert_true(probabilistic_activations.dimension(2) == input_dimensions + 1, LOG);

        assert_true(check_activations_sums(probabilistic_activations), LOG);
*/
    }
    
    {
        // Test

        batch_samples_number = 4;

        input_length = 2;
        context_length = 3;
        input_dimensions = 5;
        context_dimension = 6;

        embedding_depth = 4;
        perceptron_depth = 6;
        heads_number = 4;
        layers_number = 3;

        bool is_training = true;

        data.resize(batch_samples_number, context_length + 2 * input_length);

        for(Index i = 0; i < batch_samples_number; i++)
        {
            for(Index j = 0; j < context_length; j++)
                data(i, j) = type(rand() % context_dimension);

            for(Index j = 0; j < 2 * input_length; j++)
                data(i, j + context_length) = type(rand() % input_dimensions);
        }

        data_set.set(data);

        data_set.set_training();

        for(Index i = 0; i < context_length; i++)
            data_set.set_raw_variable_use(i, DataSet::VariableUse::Context);

        for(Index i = 0; i < input_length; i++)
            data_set.set_raw_variable_use(i + context_length, DataSet::VariableUse::Input);

        for(Index i = 0; i < input_length; i++)
            data_set.set_raw_variable_use(i + context_length + input_length, DataSet::VariableUse::Target);

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        context_variables_indices = data_set.get_context_variables_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices, context_variables_indices);

        transformer.set({ input_length, context_length, input_dimensions, context_dimension,
                          embedding_depth, perceptron_depth, heads_number, layers_number });

        ForwardPropagation forward_propagation(data_set.get_samples_number(DataSet::SampleUse::Training), &transformer);

        transformer.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);
/*
        ProbabilisticLayer3DForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation.layers[transformer.get_layers_number() - 1]);

        Tensor<type, 3> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;

        assert_true(probabilistic_activations.rank() == 3, LOG);
        assert_true(probabilistic_activations.dimension(0) == batch_samples_number, LOG);
        assert_true(probabilistic_activations.dimension(1) == input_length, LOG);
        assert_true(probabilistic_activations.dimension(2) == input_dimensions + 1, LOG);

        assert_true(check_activations_sums(probabilistic_activations), LOG);
*/
    }
}


void TransformerTest::run_test_case()
{
    cout << "Running transformer test case...\n";

    test_constructor();

    test_forward_propagate();

    test_calculate_outputs();

    cout << "End of transformer test case.\n\n";
}

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
