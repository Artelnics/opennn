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

    const Index input_length = random_integer(1, 10);
    const Index context_length = random_integer(1, 10);
    const Index input_shape = random_integer(1, 10);
    const Index context_dimension = random_integer(1, 10);
    const Index embedding_depth = random_integer(1, 10);
    const Index dense_depth = random_integer(1, 10);
    const Index heads_number = random_integer(1, 10);
    const Index layers_number = random_integer(1, 10);
/*
    Transformer transformer(input_length, 
                            context_length, 
                            input_shape, 
                            context_dimension, 
                            embedding_depth, 
                            dense_depth,
                            heads_number, 
                            layers_number);

//    EXPECT_EQ(transformer_1.get_layers_number() == 2 + 7 * layers_number + 10 * layers_number + 1);

    Tensor<unique_ptr<Layer>, 1> layers;

    Transformer transformer_0;

    EXPECT_EQ(transformer_0.is_empty());
    EXPECT_EQ(transformer_0.get_layers_number() == 0);

    // Test 3

    input_length = 2;
    context_length = 3;
    input_shape = 5;
    context_dimension = 6;
    embedding_depth = 10;
    dense_depth = 12;
    heads_number = 4;
    layers_number = 1;

    Transformer transformer_3({ input_length, context_length, input_shape, context_dimension,
                                embedding_depth, dense_depth, heads_number, layers_number });

    EXPECT_EQ(transformer_3.get_layers_number() == 2 + 7 * layers_number + 10 * layers_number + 1);

    // Test 4

    layers_number = 3;

    Transformer transformer_4({ input_length, context_length, input_shape, context_dimension,
                                embedding_depth, dense_depth, heads_number, layers_number });

    EXPECT_EQ(transformer_4.get_layers_number() == 2 + 7 * layers_number + 10 * layers_number + 1);
*/
}


TEST(Transformer, Outputs)
{
    /*
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
    */
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
*/
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
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
