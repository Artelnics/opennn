#include "pch.h"

#include "../opennn/normalized_squared_error.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/back_propagation.h"
#include "../opennn/tensors.h"

TEST(NormalizedSquaredErrorTest, DefaultConstructor)
{
    NormalizedSquaredError normalized_squared_error;

    EXPECT_EQ(normalized_squared_error.has_neural_network(), false);
    EXPECT_EQ(normalized_squared_error.has_data_set(), false);
}


TEST(NormalizedSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    DataSet data_set;

    NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);

    EXPECT_EQ(normalized_squared_error.has_neural_network(), true);
    EXPECT_EQ(normalized_squared_error.has_data_set(), true);
}


TEST(NormalizedSquaredErrorTest, BackPropagateApproximation)
{
    const Index samples_number = get_random_index(1, 10);
    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = get_random_index(1, 10);
    const Index neurons_number = get_random_index(1, 10);
/*
    DataSet data_set(samples_number, { inputs_number }, { targets_number });
    data_set.set_data_random();
    data_set.set(DataSet::SampleUse::Training);

    Batch batch(samples_number, &data_set);
    batch.fill(data_set.get_sample_indices(DataSet::SampleUse::Training),
        data_set.get_variable_indices(DataSet::VariableUse::Input),
        data_set.get_variable_indices(DataSet::VariableUse::Target));

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
        { inputs_number }, { neurons_number }, { targets_number });

    neural_network.set_parameters_random();

    ForwardPropagation forward_propagation(samples_number, &neural_network);

    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);
    normalized_squared_error.set_normalization_coefficient();
    BackPropagation back_propagation(samples_number, &normalized_squared_error);
    normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);
    
    const Tensor<type, 1> numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), true);
*/
}


/*
void NormalizedSquaredErrorTest::test_back_propagate()
{

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

        normalized_squared_error.set_normalization_coefficient();

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.errors.dimension(0) == 1);
        EXPECT_EQ(back_propagation.errors.dimension(1) == 1);
        EXPECT_NEAR(back_propagation.error() - type(0.25) < NUMERIC_LIMITS_MIN);

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

        normalized_squared_error.set_normalization_coefficient();

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.error() >= 0);

        //EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)));

    }

    // Test forecasting trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(0));

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Forecasting, {inputs_number, outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);

        bool is_training = true;
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.error() < type(1e-1));
        EXPECT_EQ(is_zero(back_propagation.gradient,type(1e-1)));
    }

    // Test forecasting random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

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

        bool is_training = true;
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        normalized_squared_error.set_normalization_coefficient(type(1));

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

        EXPECT_EQ(back_propagation.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation.error() >= type(0));
        EXPECT_EQ(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-1)));
    }
}


void NormalizedSquaredErrorTest::test_back_propagate_lm()
{
    normalized_squared_error.set_normalization_coefficient();

    // Test approximation random samples, inputs, outputs, neurons
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

        neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running
        back_propagation_lm.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = normalized_squared_error.calculate_numerical_gradient();
        numerical_jacobian = normalized_squared_error.calculate_numerical_jacobian();

        EXPECT_EQ(back_propagation_lm.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation_lm.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation_lm.error >= type(0));
        EXPECT_EQ(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-1));

        //EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-1)));
        //EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-1)));
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

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running
        back_propagation_lm.set(samples_number, &normalized_squared_error);
        //normalized_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = normalized_squared_error.calculate_numerical_gradient();
        numerical_jacobian = normalized_squared_error.calculate_numerical_jacobian();

        EXPECT_EQ(back_propagation_lm.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation_lm.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation_lm.error() >= type(0));
        //EXPECT_EQ(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2));

        //EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)));
        //EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)));

    }

    // Test multiple classification random samples, inputs, outputs, neurons
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

        back_propagation.set(samples_number, &normalized_squared_error);
        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // visual studio not running
        back_propagation_lm.set(samples_number, &normalized_squared_error);
        //normalized_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = normalized_squared_error.calculate_numerical_gradient();
        numerical_jacobian = normalized_squared_error.calculate_numerical_jacobian();

        EXPECT_EQ(back_propagation_lm.errors.dimension(0) == samples_number);
        EXPECT_EQ(back_propagation_lm.errors.dimension(1) == outputs_number);

        EXPECT_EQ(back_propagation_lm.error() >= type(0));
        //EXPECT_EQ(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2));

        //EXPECT_EQ(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)));
        //EXPECT_EQ(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)));
    }

    // Forecasting incompatible with LM
}


void NormalizedSquaredErrorTest::test_calculate_normalization_coefficient()
{
    Index samples_number;
    Index inputs_number;
    Index outputs_number;

    Tensor<string, 1> uses;

    Tensor<type, 1> targets_mean;
    Tensor<type, 2> target_data;

    // Test

    samples_number = 4;
    inputs_number = 4;
    outputs_number = 4;

    data_set.generate_random_data(samples_number, inputs_number + outputs_number);

    uses.resize(8);
    uses.setValues({"Input", "Input", "Input", "Input", "Target", "Target", "Target", "Target"});

    data_set.set_raw_variable_uses(uses);

    target_data = data_set.get_data(DataSet::VariableUse::Target);

    Eigen::array<int, 1> dimensions({0});
    targets_mean = target_data.mean(dimensions);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {2}, {outputs_number});
    neural_network.set_parameters_random();

    type normalization_coefficient = normalized_squared_error.calculate_normalization_coefficient(target_data, targets_mean);
    EXPECT_EQ(normalization_coefficient > 0);
}

}
*/
