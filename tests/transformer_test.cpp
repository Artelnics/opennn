#include "pch.h"

#include "../opennn/standard_networks.h"

using namespace opennn;

TEST(Transformer, DefaultConstructor)
{
    Transformer transformer;

    EXPECT_EQ(transformer.is_empty(), true);
    EXPECT_EQ(transformer.get_layers_number(), 0);
}


TEST(Transformer, GeneralConstructor)
{
    const Index input_sequence_length = 5;
    const Index decoder_sequence_length = 4;
    const Index input_vocabulary_size = 100;
    const Index output_vocabulary_size = 120;
    const Index embedding_dimension = 8;
    const Index heads_number = 2;
    const Index feed_forward_dimension = 16;
    const Index layers_number = 1;

    Transformer transformer(input_sequence_length,
                            decoder_sequence_length,
                            input_vocabulary_size,
                            output_vocabulary_size,
                            embedding_dimension,
                            heads_number,
                            feed_forward_dimension,
                            layers_number);

    EXPECT_EQ(transformer.get_layers_number(), 20);

    EXPECT_EQ(transformer.get_layer_index("decoder"), -1);
    EXPECT_EQ(transformer.get_layer_index("input"), -2);

    EXPECT_EQ(transformer.get_layer_index("decoder_embedding"), 0);
    EXPECT_EQ(transformer.get_layer_index("encoder_embedding"), 1);

    EXPECT_EQ(transformer.get_layer_index("encoder_self_attention_1"), 2);
    EXPECT_EQ(transformer.get_layer_index("encoder_self_attention_addition_1"), 3);
    EXPECT_EQ(transformer.get_layer_index("encoder_self_attention_normalization_1"), 4);
    EXPECT_EQ(transformer.get_layer_index("encoder_internal_dense_1"), 5);
    EXPECT_EQ(transformer.get_layer_index("encoder_external_dense_1"), 6);
    EXPECT_EQ(transformer.get_layer_index("encoder_dense_addition_1"), 7);
    EXPECT_EQ(transformer.get_layer_index("encoder_dense_normalization_1"), 8);

    EXPECT_EQ(transformer.get_layer_index("decoder_self_attention_1"), 9);
    EXPECT_EQ(transformer.get_layer_index("decoder_self_attention_addition_1"), 10);
    EXPECT_EQ(transformer.get_layer_index("decoder_self_attention_normalization_1"), 11);
    EXPECT_EQ(transformer.get_layer_index("cross_attention_1"), 12);
    EXPECT_EQ(transformer.get_layer_index("cross_attention_addition_1"), 13);
    EXPECT_EQ(transformer.get_layer_index("cross_attention_normalization_1"), 14);
    EXPECT_EQ(transformer.get_layer_index("decoder_internal_dense_1"), 15);
    EXPECT_EQ(transformer.get_layer_index("decoder_external_dense_1"), 16);
    EXPECT_EQ(transformer.get_layer_index("decoder_dense_addition_1"), 17);
    EXPECT_EQ(transformer.get_layer_index("decoder_dense_normalization_1"), 18);

    EXPECT_EQ(transformer.get_layer_index("output_projection"), 19);

    const vector<vector<Index>>& in = transformer.get_layer_input_indices();

    ASSERT_EQ(in.size(), 20);

    EXPECT_EQ(in[0], (vector<Index>{-1}));
    EXPECT_EQ(in[1], (vector<Index>{-2}));

    EXPECT_EQ(in[2], (vector<Index>{1}));
    EXPECT_EQ(in[3], (vector<Index>{1, 2}));
    EXPECT_EQ(in[4], (vector<Index>{3}));
    EXPECT_EQ(in[5], (vector<Index>{4}));
    EXPECT_EQ(in[6], (vector<Index>{5}));
    EXPECT_EQ(in[7], (vector<Index>{4, 6}));
    EXPECT_EQ(in[8], (vector<Index>{7}));

    EXPECT_EQ(in[9],  (vector<Index>{0}));
    EXPECT_EQ(in[10], (vector<Index>{0, 9}));
    EXPECT_EQ(in[11], (vector<Index>{10}));
    EXPECT_EQ(in[12], (vector<Index>{11, 8}));
    EXPECT_EQ(in[13], (vector<Index>{11, 12}));
    EXPECT_EQ(in[14], (vector<Index>{13}));
    EXPECT_EQ(in[15], (vector<Index>{14}));
    EXPECT_EQ(in[16], (vector<Index>{15}));
    EXPECT_EQ(in[17], (vector<Index>{14, 16}));
    EXPECT_EQ(in[18], (vector<Index>{17}));

    EXPECT_EQ(in[19], (vector<Index>{18}));

    const vector<vector<Index>> out = transformer.get_layer_output_indices();

    ASSERT_EQ(out.size(), 20);

    EXPECT_EQ(out[0], (vector<Index>{9, 10}));
    EXPECT_EQ(out[1],  (vector<Index>{2, 3}));
    EXPECT_EQ(out[2],  (vector<Index>{3}));
    EXPECT_EQ(out[3],  (vector<Index>{4}));
    EXPECT_EQ(out[4],  (vector<Index>{5, 7}));
    EXPECT_EQ(out[5],  (vector<Index>{6}));
    EXPECT_EQ(out[6],  (vector<Index>{7}));
    EXPECT_EQ(out[7],  (vector<Index>{8}));
    EXPECT_EQ(out[8],  (vector<Index>{12}));
    EXPECT_EQ(out[9],  (vector<Index>{10}));
    EXPECT_EQ(out[10], (vector<Index>{11}));
    EXPECT_EQ(out[11], (vector<Index>{12, 13}));
    EXPECT_EQ(out[12], (vector<Index>{13}));
    EXPECT_EQ(out[13], (vector<Index>{14}));
    EXPECT_EQ(out[14], (vector<Index>{15, 17}));
    EXPECT_EQ(out[15], (vector<Index>{16}));
    EXPECT_EQ(out[16], (vector<Index>{17}));
    EXPECT_EQ(out[17], (vector<Index>{18}));
    EXPECT_EQ(out[18], (vector<Index>{19}));
    EXPECT_EQ(out[19], (vector<Index>{-1}));
}

/*
TEST(Transformer, Outputs)
{
    Tensor2 inputs;
    Tensor2 context;
    Tensor2 outputs;

    Index parameters_number;

    Tensor1 parameters;

    // Test two dense layers with all zeros

    Index input_length = 1;
    Index context_length = 1;
    Index input_shape = 1;
    Index context_dimension = 1;
    Index embedding_depth = 1;
    Index dense_depth = 1;
    Index heads_number = 1;
    Index layers_number = 1;
    Index batch_size = 1;

    Transformer transformer(input_length, context_length, input_shape, context_dimension,
                      embedding_depth, dense_depth, heads_number, layers_number);

    transformer.set_parameters_constant(type(0));

    inputs.resize(batch_size, input_length);
    inputs.setConstant(type(0));

    context.resize(batch_size, input_length);
    context.setConstant(type(0));

    outputs = transformer.calculate_outputs(inputs, context);

    EXPECT_EQ(outputs.dimension(0), batch_size);
    EXPECT_EQ(outputs.cols(), input_length);
    EXPECT_EQ(outputs.dimension(2), input_shape);

    //EXPECT_EQ(outputs.abs() < type(EPSILON));
    
    // Test

    batch_size = 3;
    Index inputs_number = 2;
    Index neurons_number = 4;
    Index outputs_number = 5;
/*
    transformer.set(Transformer::ModelType::Approximation, { inputs_number}, {neurons_number}, {outputs_number });

    transformer.set_parameters_constant(type(0));

    inputs.resize(batch_size, inputs_number);
    inputs.setConstant(type(0));

    outputs = transformer.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == batch_size * outputs_number);
    EXPECT_EQ(abs(outputs(0, 0)) < type(EPSILON));
    EXPECT_EQ(abs(outputs(0, 1)) < type(EPSILON));
    EXPECT_EQ(abs(outputs(0, 2)) < type(EPSILON));
    EXPECT_EQ(abs(outputs(0, 3)) < type(EPSILON));
    EXPECT_EQ(abs(outputs(0, 4)) < type(EPSILON));

    // Test

    transformer.set(Transformer::ModelType::Approximation, { 1, 2 });

    transformer.set_parameters_constant(type(1));

    inputs.resize(1, 1);
    inputs.setConstant(2);

    outputs = transformer.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 2);
    EXPECT_EQ(abs(outputs(0, 0) - type(3)) < type(EPSILON));
    EXPECT_EQ(abs(outputs(0, 1) - type(3)) < type(EPSILON));

    // Test

    transformer.set(Transformer::ModelType::Approximation, { 4, 3, 3 });

    inputs.resize(1, 4);
    inputs.setZero();

    transformer.set_parameters_constant(type(1));

    outputs = transformer.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 3);

    EXPECT_EQ(abs(outputs(0, 0) - 3.2847) < type(1e-3));
    EXPECT_EQ(abs(outputs(0, 1) - 3.2847) < type(1e-3));
    EXPECT_EQ(abs(outputs(0, 2) - 3.2847) < type(1e-3));

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

    EXPECT_EQ(outputs.size() == outputs_number);
    EXPECT_EQ(abs(outputs(0, 0) - 0) < type(1e-3));
    EXPECT_EQ(abs(outputs(0, 1) - 0) < type(1e-3));

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    transformer.set(Transformer::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});

    transformer.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();

    outputs = transformer.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 1);
    EXPECT_EQ(abs(outputs(0, 0) - 0) < type(1e-3));

    // Test

    transformer.set(Transformer::ModelType::Classification, { 1,1 });

    transformer.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();

    outputs = transformer.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 1);
    EXPECT_EQ(abs(outputs(0, 0) - type(0.5)) < type(1e-3));

    // Test 7

    transformer.set(Transformer::ModelType::Approximation, { 1,3,3,3,1 });

    batch_size = 2;
    inputs_number = transformer.get_inputs_number();
    outputs_number = transformer.get_outputs_number();

    inputs.resize(batch_size, inputs_number);
    inputs.setConstant(type(0));

    parameters_number = transformer.get_parameters_number();
    parameters.resize(parameters_number);
    parameters.setConstant(type(0));

    transformer.set_parameters(parameters);

    outputs = transformer.calculate_outputs(inputs);

    EXPECT_EQ(outputs.cols() == outputs_number);
    EXPECT_EQ(abs(outputs(0, 0)) < type(EPSILON));
    EXPECT_EQ(abs(outputs(1, 0)) < type(EPSILON));
}


TEST(Transformer, ForwardPropagate)
{
    Index batch_size = 1;

    Index input_length = 4;
    Index context_length = 3;
    Index input_shape = 5;
    Index context_dimension = 6;

    Index embedding_depth = 4;
    Index dense_depth = 6;
    Index heads_number = 4;
    Index layers_number = 1;

    bool is_training = true;
/*
    data.resize(batch_size, context_length + 2 * input_length);

    for(Index i = 0; i < batch_size; i++)
    {
        for(Index j = 0; j < context_length; j++)
            data(i, j) = type(rand() % context_dimension);
        
        for(Index j = 0; j < 2 * input_length; j++)
            data(i, j + context_length) = type(rand() % input_shape);
    }
        
    dataset.set(data);

    dataset.set_sample_roles("Training");

    for(Index i = 0; i < context_length; i++)
        dataset.set_variable_role(i, string::Context);

    for(Index i = 0; i < input_length; i++)
        dataset.set_variable_role(i + context_length, "Input");

    for(Index i = 0; i < input_length; i++)
        dataset.set_variable_role(i + context_length + input_length, "Target");

    training_samples_indices = dataset.get_sample_indices("Training");
    decoder_variables_indices = dataset.get_variable_indices(string::Context);
    input_variables_indices = dataset.get_variable_indices("Input");
    target_variables_indices = dataset.get_variable_indices("Target");

    batch.set(batch_size, &dataset);

    batch.fill(training_samples_indices, input_variables_indices, decoder_variables_indices, target_variables_indices);
        
    transformer.set({ input_length, context_length, input_shape, context_dimension,
                        embedding_depth, dense_depth, heads_number, layers_number });

    ForwardPropagation forward_propagation(dataset.get_samples_number("Training"), &transformer);

    transformer.forward_propagate(batch.get_inputs(), forward_propagation, is_training);

    Dense3DForwardPropagation* dense_layer_forward_propagation
        = static_cast<Dense3DForwardPropagation*>(forward_propagation.layers[transformer.get_layers_number() - 1]);
        
    Tensor3 dense_activations = dense_layer_forward_propagation->outputs;
        
    EXPECT_EQ(dense_activations.rank() == 3);
    EXPECT_EQ(dense_activations.dimension(0) == batch_size);
    EXPECT_EQ(dense_activations.cols() == input_length);
    EXPECT_EQ(dense_activations.dimension(2) == input_shape + 1);

    EXPECT_EQ(check_activations_sums(dense_activations));

    {
        // Test

        batch_size = 4;

        input_length = 2;
        context_length = 3;
        input_shape = 5;
        context_dimension = 6;

        embedding_depth = 4;
        dense_depth = 6;
        heads_number = 4;
        layers_number = 3;

        bool is_training = true;

        data.resize(batch_size, context_length + 2 * input_length);

        for(Index i = 0; i < batch_size; i++)
        {
            for(Index j = 0; j < context_length; j++)
                data(i, j) = type(rand() % context_dimension);

            for(Index j = 0; j < 2 * input_length; j++)
                data(i, j + context_length) = type(rand() % input_shape);
        }

        dataset.set(data);

        dataset.set_sample_roles("Training");

        for(Index i = 0; i < context_length; i++)
            dataset.set_variable_role(i, string::Context);

        for(Index i = 0; i < input_length; i++)
            dataset.set_variable_role(i + context_length, "Input");

        for(Index i = 0; i < input_length; i++)
            dataset.set_variable_role(i + context_length + input_length, "Target");

        training_samples_indices = dataset.get_sample_indices("Training");
        decoder_variables_indices = dataset.get_variable_indices(string::Context);
        input_variables_indices = dataset.get_variable_indices("Input");
        target_variables_indices = dataset.get_variable_indices("Target");

        batch.set(batch_size, &dataset);

        batch.fill(training_samples_indices, input_variables_indices, decoder_variables_indices, target_variables_indices);

        transformer.set({ input_length, context_length, input_shape, context_dimension,
                          embedding_depth, dense_depth, heads_number, layers_number });

        ForwardPropagation forward_propagation(dataset.get_samples_number("Training"), &transformer);

        transformer.forward_propagate(batch.get_inputs(), forward_propagation, is_training);

        Dense3DForwardPropagation* dense_layer_forward_propagation
            = static_cast<Dense3DForwardPropagation*>(forward_propagation.layers[transformer.get_layers_number() - 1]);

        Tensor3 dense_activations = dense_layer_forward_propagation->outputs;

        EXPECT_EQ(dense_activations.rank() == 3);
        EXPECT_EQ(dense_activations.dimension(0) == batch_size);
        EXPECT_EQ(dense_activations.cols() == input_length);
        EXPECT_EQ(dense_activations.dimension(2) == input_shape + 1);

        EXPECT_EQ(check_activations_sums(dense_activations));
    }
}
*/

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.
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
