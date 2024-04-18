//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_3d_test.h"

CrossEntropyError3DTest::CrossEntropyError3DTest() : UnitTesting()
{
    cross_entropy_error_3d.set(&neural_network, &data_set);
    cross_entropy_error_3d.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


CrossEntropyError3DTest::~CrossEntropyError3DTest()
{
}


void CrossEntropyError3DTest::test_constructor()
{
    cout << "test_constructor\n";

    CrossEntropyError3D cross_entropy_error_3d;
}


void CrossEntropyError3DTest::test_destructor()
{
    cout << "test_destructor\n";

    CrossEntropyError3D* cross_entropy_error_3d = new CrossEntropyError3D;

    delete cross_entropy_error_3d;
}


void CrossEntropyError3DTest::test_back_propagate()
{
    cout << "test_back_propagate\n";


    // Test all zero
    {
        batch_samples_number = 1;
        inputs_number = 1;
        inputs_dimension = 0;
        depth = 1;
        
        // Data set

        data.resize(batch_samples_number, 2 * inputs_number);
        data.setValues({ {0, 0} });

        data_set.set_data(data);

        data_set.set_raw_variable_use(0, DataSet::VariableUse::Input);
        data_set.set_raw_variable_use(1, DataSet::VariableUse::Target);

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();

        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);
        
        // Neural network

        neural_network.set();
        
        EmbeddingLayer* embedding_layer = new EmbeddingLayer(inputs_dimension, inputs_number, depth);
        neural_network.add_layer(embedding_layer);

        ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D(inputs_number, depth, inputs_dimension + 1);
        neural_network.add_layer(probabilistic_layer_3d);
        
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(batch_samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);
        
        // Loss index

        back_propagation.set(batch_samples_number, &cross_entropy_error_3d);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);
        
        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
        assert_true(back_propagation.gradient.size() == neural_network.get_parameters_number(), LOG);

        assert_true(is_zero(back_propagation.gradient), LOG);
    }
    
    // Test all random
    {
        batch_samples_number = type(1) + rand() % 5;
        inputs_number = type(1) + rand() % 5;
        inputs_dimension = type(1) + rand() % 5;
        depth = type(1) + rand() % 5;

        // Data set

        data.resize(batch_samples_number, 2 * inputs_number);

        for (Index i = 0; i < batch_samples_number; i++)
        {
            for (Index j = 0; j < 2 * inputs_number; j++)
                data(i, j) = type(rand() % (inputs_dimension+1));
        }

        data_set.set_data(data);

        for(Index i = 0; i < inputs_number; i++)
            data_set.set_raw_variable_use(i, DataSet::VariableUse::Input);

        for (Index i = 0; i < inputs_number; i++)
            data_set.set_raw_variable_use(i + inputs_number, DataSet::VariableUse::Target);

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();

        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set();

        EmbeddingLayer* embedding_layer = new EmbeddingLayer(inputs_dimension, inputs_number, depth);
        neural_network.add_layer(embedding_layer);

        ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D(inputs_number, depth, inputs_dimension + 1);
        neural_network.add_layer(probabilistic_layer_3d);

        forward_propagation.set(batch_samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(batch_samples_number, &cross_entropy_error_3d);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);

        assert_true(back_propagation.gradient.size() == neural_network.get_parameters_number(), LOG);

        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();
        
        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }

    // Test multi-layer
    {
        batch_samples_number = type(1) + rand() % 5;
        inputs_number = type(1) + rand() % 5;
        inputs_dimension = type(1) + rand() % 5;
        depth = type(1) + rand() % 5;

        Index hidden_depth = type(1) + rand() % 5;

        // Data set

        data.resize(batch_samples_number, 2 * inputs_number);

        for (Index i = 0; i < batch_samples_number; i++)
        {
            for (Index j = 0; j < 2 * inputs_number; j++)
                data(i, j) = type(rand() % (inputs_dimension + 1));
        }

        data_set.set_data(data);

        for (Index i = 0; i < inputs_number; i++)
            data_set.set_raw_variable_use(i, DataSet::VariableUse::Input);

        for (Index i = 0; i < inputs_number; i++)
            data_set.set_raw_variable_use(i + inputs_number, DataSet::VariableUse::Target);

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();

        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set();

        EmbeddingLayer* embedding_layer = new EmbeddingLayer(inputs_dimension, inputs_number, depth);
        neural_network.add_layer(embedding_layer);

        PerceptronLayer3D* perceptron_layer_3d_internal = new PerceptronLayer3D(inputs_number, depth, hidden_depth);
        neural_network.add_layer(perceptron_layer_3d_internal);

        PerceptronLayer3D* perceptron_layer_3d_external = new PerceptronLayer3D(inputs_number, hidden_depth, depth);
        neural_network.add_layer(perceptron_layer_3d_external);

        ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D(inputs_number, depth, inputs_dimension + 1);
        neural_network.add_layer(probabilistic_layer_3d);

        forward_propagation.set(batch_samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(batch_samples_number, &cross_entropy_error_3d);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);

        assert_true(back_propagation.gradient.size() == neural_network.get_parameters_number(), LOG);

        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }
}


void CrossEntropyError3DTest::test_calculate_gradient_transformer()
{
    //cout << "test_calculate_gradient_transformer\n";

    Index context_length;
    Index context_dimension;

    LanguageDataSet data_set;

    Index perceptron_depth;
    Index heads_number;
    Index number_of_layers;

    Tensor<Index, 1> context_variables_indices;

    Transformer transformer;

    cross_entropy_error_3d.set(&transformer, &data_set);

    // Test
    {
        batch_samples_number = 1;
        
        inputs_number = 2;
        context_length = 3;
        inputs_dimension = 5;
        context_dimension = 6;

        depth = 4; 
        perceptron_depth = 6; 
        heads_number = 4;
        number_of_layers = 1;
        
        bool is_training = true;
        
        data_set.set_data_random_language_model(batch_samples_number, inputs_number, context_length, inputs_dimension, context_dimension);

        data_set.set_training();

        training_samples_indices = data_set.get_training_samples_indices();
        context_variables_indices = data_set.get_context_variables_indices();
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();
        
        batch.set(batch_samples_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices, context_variables_indices);
        
        transformer.set({ inputs_number, context_length, inputs_dimension, context_dimension,
                          depth, perceptron_depth, heads_number, number_of_layers });
        
        ForwardPropagation forward_propagation(data_set.get_training_samples_number(), &transformer);
        
        transformer.forward_propagate(batch.get_inputs_pair(), forward_propagation, is_training);
        
        // Loss index

        back_propagation.set(batch_samples_number, &cross_entropy_error_3d);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);
        
        assert_true(back_propagation.gradient.size() == transformer.get_parameters_number(), LOG);
        
        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

        const bool equal_gradients = are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-1));
        
        assert_true(equal_gradients, LOG);
        
        // debugging

        /*
        *    input_embedding from parameter 0 to 23
        *    context_embedding from parameter 24 to 51
        *    context_self_attention_1 from parameter 52 to 359
        *    context_self_attention_normalization_1 from parameter 360 to 367
        *    encoder_internal_perceptron_1 from parameter 368 to 397
        *    encoder_external_perceptron_1 from parameter 398 to 425
        *    encoder_perceptron_normalization_1 from parameter 426 to 433
        *    input_self_attention_1 from parameter 434 to 741
        *    input_self_attention_normalization_1 from parameter 742 to 749
        *    cross_attention_1 from parameter 750 to 1057
        *    cross_attention_normalization_1 from parameter 1058 to 1065
        *    decoder_internal_perceptron_1 from parameter 1066 to 1095
        *    decoder_external_perceptron_1 from parameter 1096 to 1123
        *    decoder_perceptron_normalization_1 from parameter 1124 to 1131
        *    probabilistic from parameter 1132 to 1161
        */

        /*
        cout << "Gradient min = " << back_propagation.gradient.minimum() << " at index: " << back_propagation.gradient.argmin()
            << " and max = " << back_propagation.gradient.maximum() << " at index: " << back_propagation.gradient.argmax() << endl;


        Index parameter_index = 0;
        for (Index i = 0; i < transformer.get_layers().size(); i++)
        {
            cout << transformer.get_layer(i)->get_name() << " from parameter " << parameter_index << " to " << parameter_index + transformer.get_layer(i)->get_parameters_number() - 1 << endl;
            parameter_index += transformer.get_layer(i)->get_parameters_number();
        }
        */
        
        if (!equal_gradients)
        {
            cout << endl;

            Tensor<Index, 0> max_difference_index = (back_propagation.gradient - numerical_gradient).abs().argmax();
            cout << "Test failed with max difference: " << (back_propagation.gradient - numerical_gradient).abs().maximum() << " at index: " << max_difference_index(0) << endl;
            cout << "Gradient = " << back_propagation.gradient(max_difference_index(0)) << endl;
            cout << "Numerical gradient = " << numerical_gradient(max_difference_index(0)) << endl;    
        }
        
        for (Index i = numerical_gradient.size() - 1; i >= 0 ; i--)
        {
            if (abs(numerical_gradient(i) - back_propagation.gradient(i)) > type(1e-1))
            {
                cout << "First difference greater than 0.1 at index: " << i << endl;
                break;
            }
        }
        /*
        
        Tensor<type, 0> average_difference = (back_propagation.gradient - numerical_gradient).abs().mean();
        
        assert_true(average_difference(0) < type(1.0e-2), LOG);

        Tensor<type, 0> max_difference = (back_propagation.gradient - numerical_gradient).abs().maximum();
        cout << "Max difference: " << max_difference(0) << endl;        
        
        cout << "Numerical gradient:" << endl << numerical_gradient << endl;
        cout << "Gradient:" << endl << back_propagation.gradient << endl;
        
        Index count = 0;

        Tensor<Index, 1> zeroes_indices(0);
        zeroes_indices.setZero();
        Tensor<Index, 1> aux(0);

        Index embedding_parameters_number = transformer.get_layer(0)->get_parameters_number() + transformer.get_layer(1)->get_parameters_number();

        for (Index i = embedding_parameters_number; i < numerical_gradient.size(); i++)
        {
            if (numerical_gradient(i) == 0)
            {
                aux = zeroes_indices;

                zeroes_indices.resize(count + 1);

                for (Index j = 0; j < count; j++)    zeroes_indices(j) = aux(j);

                zeroes_indices(count) = i;

                count++;
            }
        }

        if (count > 0)
        {
            cout << "Elements of numerical gradient (non-embedding) that are 0: [ ";
            for (Index i = 0; i < count - 1; i++)   cout << zeroes_indices(i) << "\t";
            cout << zeroes_indices(count - 1) << " ]" << endl;
        }

        /*
        Index embedding_parameters_number = transformer.get_layer(0)->get_parameters_number() + transformer.get_layer(1)->get_parameters_number();
        
        Index count = 0;
        for (Index i = embedding_parameters_number; i < numerical_gradient.size(); i++)
        {
            if (numerical_gradient(i) == 0) count++;
        }

        cout << "Number of 0s in numerical gradient (non-embedding): " << count << " of " << numerical_gradient.size() - embedding_parameters_number << endl;
        

        /*
        Tensor<type, 1> abs_difference = (back_propagation.gradient - numerical_gradient).abs();

        Index count = 0;

        Tensor<Index, 1> diff_indices(0);
        diff_indices.setZero();
        Tensor<Index, 1> aux(0);

        for (Index i = 0; i < abs_difference.size(); i++)
        {
            if (abs_difference(i) > type(1.0e-2))
            {
                aux = diff_indices;

                diff_indices.resize(count + 1);

                for (Index j = 0; j < count; j++)    diff_indices(j) = aux(j);

                diff_indices(count) = i;

                count++;
            }
        }
        
        if (count > 0)
        {
            cout << "Elements with differences greater than 1.0e-2: [ ";
            for (Index i = 0; i < count - 1; i++)   cout << diff_indices(i) << "\t";
            cout << diff_indices(count - 1) << " ]" << endl;

            cout << "Gradient: [ ";
            for (Index i = 0; i < count - 1; i++)   cout << back_propagation.gradient(diff_indices(i)) << "\t";
            cout << back_propagation.gradient(diff_indices(count - 1)) << " ]" << endl;

            cout << "Numerical gradient: [ ";
            for (Index i = 0; i < count - 1; i++)   cout << numerical_gradient(i) << "\t";
            cout << numerical_gradient(diff_indices(count - 1)) << " ]" << endl;

            cout << endl;
        }
        //else
            //cout << "No elements with differences greater than 1.0e-2" << endl;
        */
    }
}


void CrossEntropyError3DTest::run_test_case()
{
    cout << "Running cross-entropy error test case...\n";
    
    // Test constructor
    
    test_constructor();
    test_destructor();
    
    // Back-propagation methods
    
    //test_back_propagate();
    
    // Transformer test (Must be last since we change &neural_network to &transformer)

    cout << "test_calculate_gradient_transformer\n";
    for(Index i = 0; i < 10; i++)   test_calculate_gradient_transformer();
    
    cout << "End of cross-entropy error test case.\n\n";
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
