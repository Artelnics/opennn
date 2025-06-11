#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/cross_entropy_error_3d.h"
#include "../opennn/probabilistic_layer_3d.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/perceptron_layer_3d.h"
#include "../opennn/language_dataset.h"
#include "../opennn/transformer.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/back_propagation.h"

TEST(CrossEntropyError3DTest, DefaultConstructor)
{
    CrossEntropyError3d cross_entropy_error_3d;

    EXPECT_EQ(cross_entropy_error_3d.has_neural_network(), false);
    EXPECT_EQ(cross_entropy_error_3d.has_data_set(), false);
}


TEST(CrossEntropyError3DTest, BackPropagateZero)
{

    const Index samples_number = get_random_index(1, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    // Data set

    DataSet data_set;
    data_set.set(DataSet::SampleUse::Training);

    Batch batch(1, &data_set);
    //batch.fill({0}, {0}, {}, {1});

    // Neural network

    NeuralNetwork neural_network;

//    Embedding* embedding_layer = new Embedding(input_dimensions, inputs_number, depth);
//    neural_network.add_layer(embedding_layer);

//    Probabilistic3d* probabilistic_layer_3d = new Probabilistic3d(inputs_number, depth, input_dimensions + 1);
//    neural_network.add_layer(probabilistic_layer_3d);

    neural_network.set_parameters_constant(type(0));
/*
    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    CrossEntropyError3d cross_entropy_error_3d(&neural_network, &data_set);

    BackPropagation back_propagation(samples_number, &cross_entropy_error_3d);
    cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);

//    EXPECT_EQ(abs(back_propagation.error) < NUMERIC_LIMITS_MIN);
//    EXPECT_EQ(back_propagation.gradient.size() == neural_network.get_parameters_number());

//    EXPECT_EQ(is_zero(back_propagation.gradient));
*/
}


TEST(CrossEntropyError3DTest, BackPropagateRandom)
{
/*
    batch_size = type(1) + rand() % 5;
    inputs_number = type(1) + rand() % 5;
    input_dimensions = type(1) + rand() % 5;
    depth = type(1) + rand() % 5;

    // Data set

    data.resize(batch_size, 2 * inputs_number);

    for (Index i = 0; i < batch_size; i++)
        for (Index j = 0; j < 2 * inputs_number; j++)
            data(i, j) = type(rand() % (input_dimensions + 1));

    data_set.set_data(data);

    for (Index i = 0; i < inputs_number; i++)
        data_set.set_raw_variable_use(i, DataSet::VariableUse::Input);

    for (Index i = 0; i < inputs_number; i++)
        data_set.set_raw_variable_use(i + inputs_number, DataSet::VariableUse::Target);

    data_set.set(DataSet::SampleUse::Training);

    training_samples_indices = data_set.get_sample_indices(SampleUse::Training);

    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    batch.set(batch_size, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

    // Neural network

    neural_network.set();

    Embedding* embedding_layer = new Embedding(input_dimensions, inputs_number, depth);
    neural_network.add_layer(embedding_layer);

    Probabilistic3d* probabilistic_layer_3d = new Probabilistic3d(inputs_number, depth, input_dimensions + 1);
    neural_network.add_layer(probabilistic_layer_3d);

    forward_propagation.set(batch_size, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

    // Loss index

    back_propagation.set(batch_size, &cross_entropy_error_3d);
    cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);

    EXPECT_EQ(back_propagation.gradient.size() == neural_network.get_parameters_number());

    numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)));
*/
}


/*
void CrossEntropyError3DTest::test_calculate_gradient_transformer()
{
    Index context_length;
    Index context_dimension;

    LanguageDataSet data_set;

    Index perceptron_depth;
    Index heads_number;
    Index layers_number;

    vector<Index> decoder_variables_indices;

    Transformer transformer;

    cross_entropy_error_3d.set(&transformer, &data_set);

    // Test
    {
        batch_size = 2;
        
        inputs_number = 4;
        context_length = 6;
        input_dimensions = 11;
        context_dimension = 10;

        depth = 4; 
        perceptron_depth = 6; 
        heads_number = 4;
        layers_number = 1;
        
        bool is_training = true;
        
        data_set.set_data_random_language_model(batch_size, inputs_number, context_length, input_dimensions, context_dimension);

        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        decoder_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Decoder);
        input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
        target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);
        
        batch.set(batch_size, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, decoder_variables_indices, target_variables_indices);
        
        transformer.set({ inputs_number, context_length, input_dimensions, context_dimension,
                          depth, perceptron_depth, heads_number, layers_number });
        
        ForwardPropagation forward_propagation(data_set.get_samples_number(DataSet::SampleUse::Training), &transformer);
        
        transformer.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);
        
        // Loss index

        back_propagation.set(batch_size, &cross_entropy_error_3d);
        cross_entropy_error_3d.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        cross_entropy_error_3d.back_propagate(batch, forward_propagation, back_propagation);
        
        EXPECT_EQ(back_propagation.gradient.size() == transformer.get_parameters_number());
        
        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

        const bool equal_gradients = are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-1));
        
        EXPECT_EQ(equal_gradients);
        
        // debugging

        
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
        
        Index parameter_index = 0;
        for(Index i = 0; i < transformer.get_layers().size(); i++)
        {
            cout << transformer.get_layer(i)->get_name() << " from parameter " << parameter_index << " to " << parameter_index + transformer.get_layer(i)->get_parameters_number() - 1 << endl;
            parameter_index += transformer.get_layer(i)->get_parameters_number();
        }
                
        if(!equal_gradients)
        {
            cout << endl;

            Tensor<type, 1> difference = back_propagation.gradient - numerical_gradient;
            Tensor<Index, 0> max_difference_index = difference.abs().argmax();
            cout << "Test failed with max difference: " << difference(max_difference_index(0)) << " at index: " << max_difference_index(0) << endl;
            cout << "Gradient = " << back_propagation.gradient(max_difference_index(0)) << endl;
            cout << "Numerical gradient = " << numerical_gradient(max_difference_index(0)) << endl;
        }
        
        for(Index i = numerical_gradient.size() - 1; i >= 0 ; i--)
        {
            if(abs(numerical_gradient(i) - back_propagation.gradient(i)) > type(1e-1))
            {
                cout << "First difference greater than 0.1 at index: " << i << endl;
                break;
            }
        }
                
        Tensor<type, 0> average_difference = (back_propagation.gradient - numerical_gradient).abs().mean();
        
        EXPECT_EQ(average_difference(0) < type(1.0e-2));

        Tensor<type, 0> max_difference = (back_propagation.gradient - numerical_gradient).abs().maximum();
        cout << "Max difference: " << max_difference(0) << endl;        
        
        cout << "Numerical gradient:" << endl << numerical_gradient << endl;
        cout << "Gradient:" << endl << back_propagation.gradient << endl;
        
        Index count = 0;

        Tensor<Index, 1> zeroes_indices(0);
        zeroes_indices.setZero();
        Tensor<Index, 1> aux(0);

        Index embedding_parameters_number = transformer.get_layer(0)->get_parameters_number() + transformer.get_layer(1)->get_parameters_number();

        for(Index i = embedding_parameters_number; i < numerical_gradient.size(); i++)
        {
            if(numerical_gradient(i) == 0)
            {
                aux = zeroes_indices;

                zeroes_indices.resize(count + 1);

                for(Index j = 0; j < count; j++)    zeroes_indices(j) = aux(j);

                zeroes_indices(count) = i;

                count++;
            }
        }

        if(count > 0)
        {
            cout << "Elements of numerical gradient (non-embedding) that are 0: [ ";
            for(Index i = 0; i < count - 1; i++)   cout << zeroes_indices(i) << "\t";
            cout << zeroes_indices(count - 1) << " ]" << endl;
        }


        Index embedding_parameters_number = transformer.get_layer(0)->get_parameters_number() + transformer.get_layer(1)->get_parameters_number();
        
        Index count = 0;

        for(Index i = embedding_parameters_number; i < numerical_gradient.size(); i++)
            if(numerical_gradient(i) == 0)
                count++;

        cout << "Number of 0s in numerical gradient (non-embedding): " << count << " of " << numerical_gradient.size() - embedding_parameters_number << endl;
        
        Tensor<type, 1> abs_difference = (back_propagation.gradient - numerical_gradient).abs();

        Index count = 0;

        Tensor<Index, 1> diff_indices(0);
        diff_indices.setZero();
        Tensor<Index, 1> aux(0);

        for(Index i = 0; i < abs_difference.size(); i++)
        {
            if(abs_difference(i) > type(1.0e-2))
            {
                aux = diff_indices;

                diff_indices.resize(count + 1);

                for(Index j = 0; j < count; j++)    diff_indices(j) = aux(j);

                diff_indices(count) = i;

                count++;
            }
        }
        
        if(count > 0)
        {
            cout << "Elements with differences greater than 1.0e-2: [ ";
            for(Index i = 0; i < count - 1; i++)   cout << diff_indices(i) << "\t";
            cout << diff_indices(count - 1) << " ]" << endl;

            cout << "Gradient: [ ";
            for(Index i = 0; i < count - 1; i++)   cout << back_propagation.gradient(diff_indices(i)) << "\t";
            cout << back_propagation.gradient(diff_indices(count - 1)) << " ]" << endl;

            cout << "Numerical gradient: [ ";
            for(Index i = 0; i < count - 1; i++)   cout << numerical_gradient(i) << "\t";
            cout << numerical_gradient(diff_indices(count - 1)) << " ]" << endl;

            cout << endl;
        }
        //else
            //cout << "No elements with differences greater than 1.0e-2" << endl;
        
    }
}
*/
