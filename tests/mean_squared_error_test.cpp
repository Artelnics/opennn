#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/dataset.h"
#include "../opennn/language_dataset.h"
#include "../opennn/perceptron_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/mean_squared_error.h"

using namespace opennn;

TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    MeanSquaredError mean_squared_error;

    EXPECT_EQ(mean_squared_error.has_neural_network(), false);
    EXPECT_EQ(mean_squared_error.has_data_set(), false);
}


TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;
    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    EXPECT_EQ(mean_squared_error.has_neural_network(), true);
    EXPECT_EQ(mean_squared_error.has_data_set(), true);
}


TEST(MeanSquaredErrorTest, BackPropagate)
{
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_data_random();
    dataset.set(Dataset::SampleUse::Training);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense2d>(dimensions{ inputs_number }, dimensions{ targets_number }));
    neural_network.set_parameters_random();

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    const type error = mean_squared_error.calculate_numerical_error();

    EXPECT_GE(error, 0);

    const Tensor<type, 1> gradient = mean_squared_error.calculate_gradient();
    const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
}


TEST(MeanSquaredErrorTest, BackPropagateLm)
{
/*
    const Index samples_number = get_random_index(2, 10);
    const Index inputs_number = get_random_index(1, 1);
    const Index outputs_number = get_random_index(1, 1);
    const Index neurons_number = get_random_index(1, 1);

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set(Dataset::SampleUse::Training);

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices(Dataset::SampleUse::Training),
               dataset.get_variable_indices(Dataset::VariableUse::Input),
               dataset.get_variable_indices(Dataset::VariableUse::Decoder),
               dataset.get_variable_indices(Dataset::VariableUse::Target));

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, 
                                { inputs_number }, { neurons_number }, { outputs_number });
    neural_network.set_parameters_random();

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    BackPropagation back_propagation(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    BackPropagationLM back_propagation_lm(samples_number, &mean_squared_error);
    mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

    const Tensor<type, 2> numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();
    const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();
    const Tensor<type, 2> numerical_hessian = mean_squared_error.calculate_numerical_hessian();

    EXPECT_NEAR(back_propagation_lm.error(), back_propagation.error(), type(1.0e-3));
    EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian), true);
    EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1e-2)), true);
    //EXPECT_EQ(are_equal(back_propagation_lm.hessian, numerical_hessian, type(1.0e-1)), true);
*/
}


TEST(MeanSquaredErrorTest, BackPropagateMultiheadAttention)
{

    NeuralNetwork neural_network;

    //const Index sequence_length = 3;More actions
    const Index embedding_dimension = 4;
    const Index heads_number = 2;
    //const Index batch_size = 3;

    LanguageDataset language_dataset;//("../data/amazon_cells_labelled.txt");
/*
    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Embedding>(language_dataset.get_input_dimensions(), embedding_dimension));
    neural_network.add_layer(make_unique<MultiHeadAttention>(neural_network.get_output_dimensions(), heads_number), {0,0});
    neural_network.add_layer(make_unique<Flatten3d>(neural_network.get_output_dimensions()));
    neural_network.add_layer(make_unique<Dense2d>(neural_network.get_output_dimensions(), language_dataset.get_target_dimensions(), Dense2d::Activation::Logistic));
*/
}
