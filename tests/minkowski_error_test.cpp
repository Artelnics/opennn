#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/data_set.h"
#include "../opennn/minkowski_error.h"
#include "../opennn/forward_propagation.h"


TEST(MinkowskiErrorTest, DefaultConstructor)
{
    MinkowskiError minkowski_error;

    EXPECT_EQ(minkowski_error.has_neural_network(), false);
    EXPECT_EQ(minkowski_error.has_data_set(), false);
}


TEST(MinkowskiErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    DataSet data_set;

    MinkowskiError minkowski_error(&neural_network, &data_set);

    EXPECT_EQ(minkowski_error.has_neural_network(), true);
    EXPECT_EQ(minkowski_error.has_data_set(), true);
}


TEST(MinkowskiErrorTest, BackPropagateApproximationZero)
{

    DataSet data_set(1, { 1 }, { 1 });
    data_set.set_data_constant(type(0));
    data_set.set(DataSet::SampleUse::Training);

    Batch batch(1, &data_set);
    batch.fill({0}, {0}, {1});
 
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {1}, {1}, {1});
    neural_network.set_parameters_constant(type(0));

    ForwardPropagation forward_propagation(1, &neural_network);
/*
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    back_propagation.set(samples_number, &minkowski_error);
    minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

    EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
    EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

    EXPECT_EQ(abs(back_propagation.error()) < NUMERIC_LIMITS_MIN);
    EXPECT_EQ(back_propagation.gradient.size() == inputs_number + inputs_number * neurons_number + outputs_number + outputs_number * neurons_number);

    EXPECT_EQ(is_zero(back_propagation.gradient));
*/
}

/*
void MinkowskiErrorTest::test_back_propagate()
{
    // Test approximation trivial
    {
        samples_number = 1;
        inputs_number = 1;
        outputs_number = 1;
        neurons_number = 1;
        bool is_training = true;

        // Data set

    }

    // Test approximation all random
    {
        samples_number = 1 + rand()%5;
        inputs_number = 1 + rand()%5;
        outputs_number = 1 + rand()%5;
        neurons_number = 1 + rand()%5;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();

        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
        target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        neural_network.print();

        // Loss index

        back_propagation.set(samples_number, &minkowski_error);

        minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = minkowski_error.calculate_numerical_gradient();

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)));

    }

    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
        target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {}, {outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &minkowski_error);
        minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = minkowski_error.calculate_numerical_gradient();

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.errors.dimension(0) == 1);
        EXPECT_EQ(back_propagation.errors.dimension(1) == 1);
        EXPECT_EQ(back_propagation.error() - type(0.5) < type(NUMERIC_LIMITS_MIN));

        EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)));
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_binary_random();
        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
        target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);

        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &minkowski_error);

        minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = minkowski_error.calculate_numerical_gradient();

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.error() >= 0);

        // @todo
        EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)));

    }

    // Test forecasting trivial
    {

        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);
        input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
        target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Forecasting, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &minkowski_error);
        minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(is_zero(back_propagation.gradient, type(1e-3)));

    }

    // Test forecasting random samples, inputs, outputs, neurons
    {

        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();
        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Forecasting, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &minkowski_error);
        minkowski_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = minkowski_error.calculate_numerical_gradient();


        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.error() >= type(0));

        EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)));

    }
}

}
*/
