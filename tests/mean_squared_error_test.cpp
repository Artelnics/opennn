#include "pch.h"

#include "../opennn/data_set.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/back_propagation.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network_forward_propagation.h"


TEST(MeanSquaredErrorTest, DefaultConstructor)
{
    MeanSquaredError mean_squared_error;

    EXPECT_EQ(mean_squared_error.has_neural_network(), false);
    EXPECT_EQ(mean_squared_error.has_data_set(), false);
}


TEST(MeanSquaredErrorTest, GeneralConstructor)
{
    NeuralNetwork neural_network;
    DataSet data_set;
    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    EXPECT_EQ(mean_squared_error.has_neural_network(), true);
    EXPECT_EQ(mean_squared_error.has_data_set(), true);
}


TEST(MeanSquaredErrorTest, BackPropagateApproximationZero)
{
    /*
    DataSet data_set(1, 1, 1);
    data_set.set_data_constant(type(0));

    data_set.set(DataSet::SampleUse::Training);
    
    Batch batch(1, &data_set);

    //batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {1}, {1}, {1});
    neural_network.set_parameters_constant(type(0));
    
    ForwardPropagation forward_propagation(1, &neural_network);

    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    BackPropagation back_propagation(1, &mean_squared_error);
    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
    assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

    assert_true(abs(back_propagation.error()) < NUMERIC_LIMITS_MIN, LOG);
    assert_true(back_propagation.gradient.size() == inputs_number + inputs_number * neurons_number + outputs_number + outputs_number * neurons_number, LOG);

    assert_true(is_zero(back_propagation.gradient), LOG);
*/

    EXPECT_EQ(1, 1);
}


/*

void MeanSquaredErrorTest::test_back_propagate_perceptron()
{
    // Test approximation all zero
    {


    }

    // Test approximation all random
    {
        samples_number = type(1) + rand() % 5;
        inputs_number = type(1) + rand() % 5;
        outputs_number = type(1) + rand() % 5;
        neurons_number = type(1) + rand() % 5;

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

        neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number }, { neurons_number }, {outputs_number });
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }

    {
        samples_number = 10;
        inputs_number = 8;
        outputs_number = 9;
        neurons_number = 12;

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

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);
    }
}


void MeanSquaredErrorTest::test_back_propagate_probabilistic()
{
    cout << "test_back_propagate_probabilistic\n";
    /*
    // Test binary classification trivial
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

        neural_network.set(NeuralNetwork::ModelType::Classification, { inputs_number }, {}, {outputs_number });
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error() - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = type(1) + rand() % 10;
        inputs_number = type(1) + rand() % 10;
        neurons_number = type(1) + rand() % 10;
        outputs_number = type(1) + rand() % 10;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_binary_random();
        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, { inputs_number }, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);

        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        //back_propagation.print();

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error() >= 0, LOG);
        //cout << "back_propagation.gradient:\n" << back_propagation.gradient << endl;
        //cout << "numerical_gradient:\n" << numerical_gradient << endl;
        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }

}


void MeanSquaredErrorTest::test_back_propagate_convolutional()
{
    cout << "test_back_propagate_convolutional\n";
}


void MeanSquaredErrorTest::test_back_propagate_recurrent()
{
    cout << "test_back_propagate_recurrent\n";

    Index samples_number = 100;
    Index time_steps = 5;
    Index inputs_number = 12;
    Index neurons_number = 10;
    Index targets_number = 4;

    // Data set

    Tensor<type, 2> data(samples_number, inputs_number + targets_number);

    data.setRandom();

    DataSet data_set(data);

    for(Index i = 0; i < inputs_number; i++)
        data_set.set_raw_variable_use(i, DataSet::VariableUse::Input);

    for(Index i = 0; i < targets_number; i++)
        data_set.set_raw_variable_use(i + inputs_number, DataSet::VariableUse::Target);

    data_set.set(DataSet::SampleUse::Training);

    Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);

    Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
    Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

    Batch batch(samples_number, &data_set);

    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    // Neural network

    NeuralNetwork neural_network;

    RecurrentLayer* recurrent_layer = new RecurrentLayer(inputs_number, neurons_number, time_steps);
    neural_network.add_layer(recurrent_layer);

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    MeanSquaredError error(&neural_network, &data_set);

    BackPropagation back_propagation(samples_number, &error);

    error.back_propagate(batch, forward_propagation, back_propagation);

    Tensor<type, 1> numerical_gradient = error.calculate_numerical_gradient();

    assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);

}


void MeanSquaredErrorTest::test_back_propagate_long_short_term_memory()
{
    cout << "test_back_propagate_long_short_term_memory\n";

    Index samples_number = 100;
    Index time_steps = 5;
    Index inputs_number = 12;
    Index neurons_number = 10;
    Index targets_number = 4;

    // Data set

    Tensor<type, 2> data(samples_number, inputs_number + targets_number);

    data.setRandom();

    DataSet data_set(data);

    for(Index i = 0; i < inputs_number; i++)
        data_set.set_raw_variable_use(i, DataSet::VariableUse::Input);

    for(Index i = 0; i < targets_number; i++)
        data_set.set_raw_variable_use(i + inputs_number, DataSet::VariableUse::Target);

    data_set.set(DataSet::SampleUse::Training);

    Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);

    Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
    Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

    Batch batch(samples_number, &data_set);

    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    // Neural network

    NeuralNetwork neural_network;

    LongShortTermMemoryLayer* lstm_layer = new LongShortTermMemoryLayer(inputs_number, neurons_number, time_steps);
    neural_network.add_layer(lstm_layer);

    ForwardPropagation forward_propagation(samples_number, &neural_network);
    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

    // Loss index

    MeanSquaredError error(&neural_network, &data_set);

    BackPropagation back_propagation(samples_number, &error);

    error.back_propagate(batch, forward_propagation, back_propagation);

    Tensor<type, 1> numerical_gradient = error.calculate_numerical_gradient();

    assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);

}


void MeanSquaredErrorTest::test_back_propagate()
{
    cout << "test_back_propagate\n";

}


void MeanSquaredErrorTest::test_back_propagate_lm()
{
    cout << "test_back_propagate_lm\n";

    // Test approximation random samples, inputs, outputs, neurons
    {
        samples_number = type(1) + rand()%10;
        inputs_number = type(1) + rand()%10;
        outputs_number = type(1) + rand()%10;
        neurons_number = type(1) + rand()%10;

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

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // not running in  visual studio

        back_propagation_lm.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();
        numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-1)), LOG);

        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-1)), LOG);
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {

        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_binary_random();
        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // not running in visual studio

        back_propagation_lm.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();
        numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        cout << back_propagation_lm.gradient << endl;
        cout << endl;
        cout << numerical_gradient << endl;

        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)), LOG);

    }

    // Test multiple classification random samples, inputs, outputs, neurons
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

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        // not running in visual studio

        back_propagation_lm.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();
        numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();

        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation_lm.error >= type(0), LOG);
        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)), LOG);
        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)), LOG);

    }

    // Test convolutional
    {

//        samples_number = 1;
//        inputs_number = 1;
//        outputs_number = 1;
//        neurons_number = 1;
//        bool is_training = true;

//        const Index channels = 3;

//        const Index input_heigth = 4;
//        const Index input_width = 4;
//        const Index kernel_height = 3;
//        const Index kernel_width = 3;

//        //set dimensions

//        Tensor<type,4> input_batch(input_heigth, input_width, channels, input_images);
//        Tensor<type,4> kernel(kernel_height, kernel_width, channels, kernels_number);
//        Tensor<type, 1> bias(kernels_number);

//        const Index inputs_number_convolution = (input_heigth)*(input_width)*channels*input_images;
//        const Index output_number_convolution = (input_heigth - kernel_height + 1)*(input_width - kernel_width + 1)*kernels_number*input_images;

//        //set values

//        input_batch.setConstant(1.);

//        input_batch.chip(0,3).chip(0,2).setConstant(2.);
//        input_batch.chip(0,3).chip(1,2).setConstant(3.);
//        input_batch.chip(0,3).chip(2,2).setConstant(4.);

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation, is_training);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);

//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

//        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

//        assert_true(abs(back_propagation.error) < NUMERIC_LIMITS_MIN, LOG);
//        assert_true(back_propagation.gradient.size() == inputs_number+inputs_number*neurons_number+outputs_number+outputs_number*neurons_number, LOG);

//        assert_true(is_zero(back_propagation.gradient), LOG);

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

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }

    // Test binary classification trivial
    {
        inputs_number = 1;
        outputs_number = 1;
        samples_number = 1;
        bool is_training = true;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_constant(type(1));

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {}, {outputs_number});
        neural_network.set_parameters_constant(type(0));

        forward_propagation.set(samples_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.errors.dimension(0) == 1, LOG);
        assert_true(back_propagation.errors.dimension(1) == 1, LOG);
        assert_true(back_propagation.error() - type(0.25) < type(NUMERIC_LIMITS_MIN), LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-3)), LOG);
    }

    // Test binary classification random samples, inputs, outputs, neurons
    {
        samples_number = 1 + rand()%10;
        inputs_number = 1 + rand()%10;
        outputs_number = 1 + rand()%10;
        neurons_number = 1 + rand()%10;

        // Data set

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_binary_random();
        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        // Neural network

        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});
        neural_network.set_parameters_random();

        forward_propagation.set(samples_number, &neural_network);
        bool is_training = true;
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        // Loss index

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error() >= 0, LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-2)), LOG);
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

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error() < type(1e-1), LOG);
        assert_true(is_zero(back_propagation.gradient, type(1e-1)), LOG);
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

        back_propagation.set(samples_number, &mean_squared_error);
        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();


        assert_true(back_propagation.errors.dimension(0) == samples_number, LOG);
        assert_true(back_propagation.errors.dimension(1) == outputs_number, LOG);

        assert_true(back_propagation.error() >= type(0), LOG);

        assert_true(are_equal(back_propagation.gradient, numerical_gradient, type(1.0e-1)), LOG);
    }

    // Test convolutional
   if(false)
    {

//       const Index input_images = 3;
//       const Index kernels_number = 3;

//       const Index channels = 3;

//       const Index input_heigth = 3;
//       const Index input_width = 3;
//       const Index kernel_height = 2;
//       const Index kernel_width = 2;

//       data_set.set_data_source_path("../../blank/test-6px-python-bmp/");

//       image_data_set.read_bmp();

//       const Index samples_number = data_set.get_training_samples_number();

//       const Tensor<Index, 1> samples_indices = data_set.get_sample_indices(SampleUse::Training);
//       const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//       const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

//       Batch batch(samples_number, &data_set);

//       batch.fill(samples_indices, input_variables_indices, target_variables_indices);

//       Eigen::array<Eigen::Index, 4> extents = {0, 0, 0, 0};
//       Eigen::array<Eigen::Index, 4> offsets = {batch.inputs_4d.dimension(0),
//                                                batch.inputs_4d.dimension(1)-1, //padding
//                                                batch.inputs_4d.dimension(2),
//                                                batch.inputs_4d.dimension(3)};

//       //remove padding
//       Tensor<float, 4> new_batch = batch.inputs_4d.slice(extents, offsets);
//       batch.inputs_4d = new_batch;

//       //set dimensions

//       Tensor<type,4> kernel(kernel_height, kernel_width, channels, kernels_number);
//       Tensor<type, 1> bias(kernels_number);

//       const Index inputs_number_convolution = (input_heigth)*(input_width)*channels;
//       const Index output_number_convolution = (input_heigth - kernel_height + 1)*(input_width - kernel_width + 1)*kernels_number;

//       //set values

//       kernel.chip(0,3).setConstant(type(1./3.));
//       kernel.chip(1,3).setConstant(type(1./9.));
//       kernel.chip(2,3).setConstant(type(1./27.));

//       bias.setValues({0, 0, 0});

//       neural_network.set(NeuralNetwork::ModelType::ImageClassification,
//                          {inputs_number_convolution, output_number_convolution, 1});

//       ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(neural_network.get_layer(0));
//       FlattenLayer* flatten_layer = static_cast<FlattenLayer*>(neural_network.get_layer(1));
//       PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer(2));

//       //set_dims //this should be inside nn contructor.

//       cout<<batch.inputs_4d<<endl;
//       getchar();
//       convolutional_layer->set(batch.inputs_4d, kernel, bias);

//       //set dimensions //this should be inside nn contructor.
//       flatten_layer->set(convolutional_layer->get_output_dimensions());


//       //set values
//       convolutional_layer->set_synaptic_weights(kernel);
//       convolutional_layer->set_biases(bias);

//       perceptron_layer->set_synaptic_weights_constant(1.);
//       perceptron_layer->set_biases_constant(0);

//       //start

//       forward_propagation.set(input_images, &neural_network);
//       neural_network.forward_propagate(batch, forward_propagation);
//       forward_propagation.print();

//       cout<<"parameters"<<endl;
//       cout<<neural_network.get_parameters()<<endl;

//       // create Dataset object to load data.
//       numerical_gradient = mean_squared_error.calculate_numerical_gradient();
   }

}


void MeanSquaredErrorTest::test_calculate_gradient_convolutional_network()
{
    cout << "test_calculate_gradient_convolutional_network\n";

//    const Index images_number = 2;

//    Tensor<Index, 1> input_dimensions(3);
//    input_dimensions(0) = 2;
//    input_dimensions(1) = 2;
//    input_dimensions(2) = 2;

//    Tensor<type, 2> data(images_number,9);

//    // Image 1

//    data(0,0) = 1;
//    data(0,1) = 5;
//    data(0,2) = 2;
//    data(0,3) = 6;

//    data(0,4) = 3;
//    data(0,5) = 7;
//    data(0,6) = 4;
//    data(0,7) = 8;

//    data(0,8) = 0; // Target

//    // Image 2

//    data(1,0) = 9;
//    data(1,1) = 13;
//    data(1,2) = 10;
//    data(1,3) = 14;
//    forward_propagation.set(samples_number, &neural_network);
//    bool is_training = true;
//    neural_network.forward_propagate(batch, forward_propagation, is_training);

//    data(1,4) = 11;
//    data(1,5) = 15;
//    data(1,6) = 12;
//    data(1,7) = 16;

//    data(1,8) = 1; // Target

//    DataSet data_set(images_number,1,1);
//    data_set.set_data_constant(3.1416);
//    data_set.set_input_dimensions(input_dimensions);
//    data_set.set_data(data); // 2d data

//    const Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
//    const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//    const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

//    Batch batch(images_number, &data_set);

//    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//    //    batch.print();

//    Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
//    convolutional_layer_inputs_dimensions(0) = 2;
//    convolutional_layer_inputs_dimensions(1) = 2;
//    convolutional_layer_inputs_dimensions(2) = 2;
//    convolutional_layer_inputs_dimensions(3) = images_number;
        // not running in  visual studio
//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        numerical_gradient = mean_squared_error.calculate_numerical_gradient();
//        numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-1)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-1)), LOG);
//    }

//    NeuralNetwork neural_network;

//    Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
//    convolutional_layer_kernels_dimensions(0) = 1;
//    convolutional_layer_kernels_dimensions(1) = 1;
//    convolutional_layer_kernels_dimensions(2) = images_number;
//    convolutional_layer_kernels_dimensions(3) = 1;

//    ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

//    convolutional_layer->set_synaptic_weights_constant(0.5);
//    convolutional_layer->set_biases_constant(0);

//    Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
//    flatten_layer_inputs_dimensions(0) = 2;
//    flatten_layer_inputs_dimensions(1) = 2;
//    flatten_layer_inputs_dimensions(2) = 1;
//    flatten_layer_inputs_dimensions(3) = 2;

//    FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

//    PerceptronLayer* perceptron_layer = new PerceptronLayer(4, 1);

//    perceptron_layer->set_synaptic_weights_constant(1);
//    perceptron_layer->set_biases_constant(0);
//    perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
//    forward_propagation.set(samples_number, &neural_network);
//    bool is_training = true;
//    neural_network.forward_propagate(batch, forward_propagation, is_training);

//    neural_network.add_layer(convolutional_layer);
//    neural_network.add_layer(flatten_layer);
//    neural_network.add_layer(perceptron_layer);

//    NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

//    neural_network.forward_propagate(batch, forward_propagation);
//    //    forward_propagation.print();

//    MeanSquaredError mean_squared_error(&neural_network, &data_set);

//    BackPropagation back_propagation(2, &mean_squared_error);

//    mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//    //    cout << "Gradient: " << back_propagation.gradient << endl;

        // not running in visual studio

//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        numerical_gradient = mean_squared_error.calculate_numerical_gradient();
//        numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)), LOG);
//    }

//    numerical_gradient = mean_squared_error.calculate_numerical_gradient();

//    cout << "Numerical gradient: " << numerical_gradient << endl;

        // Data set

//        data_set.set(samples_number, inputs_number, outputs_number);
//        data_set.set_data_random();
//        data_set.set(DataSet::SampleUse::Training);

//        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
//        input_variables_indices = data_set.get_input_variables_indices();
//        target_variables_indices = data_set.get_target_variables_indices();

//        batch.set(samples_number, &data_set);
//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        // Neural network

//        neural_network.set(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});
//        neural_network.set_parameters_random();

//        forward_propagation.set(samples_number, &neural_network);
//        bool is_training = true;
//        neural_network.forward_propagate(batch, forward_propagation, is_training);

//        forward_propagation.set(samples_number, &neural_network);
//        neural_network.forward_propagate(batch, forward_propagation);

//        // Loss index

//        back_propagation.set(samples_number, &mean_squared_error);
//        MeanSquaredError.back_propagate(batch, forward_propagation, back_propagation);
        // not running in visual studio

//        back_propagation_lm.set(samples_number, &mean_squared_error);
//        mean_squared_error.back_propagate_lm(batch, forward_propagation, back_propagation_lm);

//        numerical_gradient = mean_squared_error.calculate_numerical_gradient();
//        numerical_jacobian = mean_squared_error.calculate_numerical_jacobian();

//        assert_true(back_propagation_lm.errors.dimension(0) == samples_number, LOG);
//        assert_true(back_propagation_lm.errors.dimension(1) == outputs_number, LOG);

//        assert_true(back_propagation_lm.error >= type(0), LOG);
//        assert_true(abs(back_propagation.error-back_propagation_lm.error) < type(1.0e-2), LOG);

//        assert_true(are_equal(back_propagation_lm.gradient, numerical_gradient, type(1.0e-2)), LOG);
//        assert_true(are_equal(back_propagation_lm.squared_errors_jacobian, numerical_jacobian, type(1.0e-2)), LOG);
//    }

//}
//    {
//        const Index images_number = 2;

//        Tensor<Index, 1> input_dimensions(3);
//        input_dimensions(0) = 2;
//        input_dimensions(1) = 2;
//        input_dimensions(2) = 2;

//        Tensor<type, 2> data(images_number,9);

//        // Image 1

//        data(0,0) = 1;
//        data(0,1) = 5;
//        data(0,2) = 2;
//        data(0,3) = 6;

//        data(0,4) = 3;
//        data(0,5) = 7;
//        data(0,6) = 4;
//        data(0,7) = 8;

//        data(0,8) = 0; // Target

//        // Image 2

//        data(1,0) = 9;
//        data(1,1) = 13;
//        data(1,2) = 10;
//        data(1,3) = 14;

//        data(1,4) = 11;
//        data(1,5) = 15;
//        data(1,6) = 12;
//        data(1,7) = 16;

//        data(1,8) = 1; // Target

//        DataSet data_set(images_number,1,1);
//        data_set.set_input_dimensions(input_dimensions);
//        data_set.set_data(data); // 2d data

//        const Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
//        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

//        Batch batch(images_number, &data_set);

//        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        //    batch.print();

//        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
//        convolutional_layer_inputs_dimensions(0) = 2;
//        convolutional_layer_inputs_dimensions(1) = 2;
//        convolutional_layer_inputs_dimensions(2) = 2;
//        convolutional_layer_inputs_dimensions(3) = images_number;

//        NeuralNetwork neural_network;

//        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
//        convolutional_layer_kernels_dimensions(0) = 1;
//        convolutional_layer_kernels_dimensions(1) = 1;
//        convolutional_layer_kernels_dimensions(2) = images_number;
//        convolutional_layer_kernels_dimensions(3) = 1;

//        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

//        Tensor<type, 4> kernels(1,1,2,1);

//        kernels(0) = 0.5;
//        kernels(1) = 0.7;

//        convolutional_layer->set_synaptic_weights(kernels);

//        //    convolutional_layer->set_synaptic_weights_constant(0.5);
//        convolutional_layer->set_biases_constant(0);

//        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
//        flatten_layer_inputs_dimensions(0) = 2;
//        flatten_layer_inputs_dimensions(1) = 2;
//        flatten_layer_inputs_dimensions(2) = 1;
//        flatten_layer_inputs_dimensions(3) = 2;

//        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

//        PerceptronLayer* perceptron_layer = new PerceptronLayer(4, 2);

//        Tensor<type, 2> synaptic_weights(4,2);
//        synaptic_weights(0,0) = 1;
//        synaptic_weights(1,0) = 1;
//        synaptic_weights(2,0) = 1;
//        synaptic_weights(3,0) = 1;

//        synaptic_weights(0,1) = 2;
//        synaptic_weights(1,1) = 2;
//        synaptic_weights(2,1) = 2;
//        synaptic_weights(3,1) = 2;

//        //    perceptron_layer->set_synaptic_weights_constant(1);
//        perceptron_layer->set_synaptic_weights(synaptic_weights);
//        perceptron_layer->set_biases_constant(0);
//        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

//        cout << "Perceptron layer weights: " << perceptron_layer->get_synaptic_weights() << endl;

//        neural_network.add_layer(convolutional_layer);
//        neural_network.add_layer(flatten_layer);
//        neural_network.add_layer(perceptron_layer);

//        NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

//        neural_network.forward_propagate(batch, forward_propagation);
//        forward_propagation.print();

//        MeanSquaredError mean_squared_error(&neural_network, &data_set);

//        BackPropagation back_propagation(2, &mean_squared_error);

//        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

//        //    cout << "Gradient: " << back_propagation.gradient << endl;

//        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

//        cout << "Numerical gradient: " << numerical_gradient << endl;
//    }

    // Inputs 3x3x2x2; Filters: 1x1x2; Perceptrons: 2 --> Test ok

    {
        const Index images_number = 2;

        Tensor<Index, 1> input_dimensions(3);
        input_dimensions(0) = 2; // Channels number
        input_dimensions(1) = 3; // rows number
        input_dimensions(2) = 3; // columns number

        Tensor<type, 2> data(images_number,19);

        // Image 1

        data(0,0) = 1;
        data(0,1) = 10;
        data(0,2) = 2;
        data(0,3) = 11;
        data(0,4) = 3;
        data(0,5) = 12;
        data(0,6) = 4;
        data(0,7) = 13;
        data(0,8) = 5;
        data(0,9) = 14;
        data(0,10) = 6;
        data(0,11) = 15;
        data(0,12) = 7;
        data(0,13) = 16;
        data(0,14) = 8;
        data(0,15) = 17;
        data(0,16) = 9;
        data(0,17) = 18;

        data(0,18) = 0; // Target

        // Image 2

        data(1,0) = 19;
        data(1,1) = 28;
        data(1,2) = 20;
        data(1,3) = 29;
        data(1,4) = 21;
        data(1,5) = 30;
        data(1,6) = 22;
        data(1,7) = 31;
        data(1,8) = 23;
        data(1,9) = 32;
        data(1,10) = 24;
        data(1,11) = 33;
        data(1,12) = 25;
        data(1,13) = 34;
        data(1,14) = 26;
        data(1,15) = 35;
        data(1,16) = 27;
        data(1,17) = 36;

        data(1,18) = 1; // Target

        DataSet data_set(images_number,1,1);
        data_set.set_input_dimensions(input_dimensions);
        data_set.set_data(data); // 2d data

        const Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        Batch batch(images_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        batch.print();

        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
        convolutional_layer_inputs_dimensions(0) = 3;
        convolutional_layer_inputs_dimensions(1) = 3;
        convolutional_layer_inputs_dimensions(2) = 2;
        convolutional_layer_inputs_dimensions(3) = images_number;

        NeuralNetwork neural_network;

        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
        convolutional_layer_kernels_dimensions(0) = 1;
        convolutional_layer_kernels_dimensions(1) = 1;
        convolutional_layer_kernels_dimensions(2) = images_number;
        convolutional_layer_kernels_dimensions(3) = 1;

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

        Tensor<type, 4> kernels(1,1,2,1);

        kernels(0) = 0.5;
        kernels(1) = 0.7;

        convolutional_layer->set_synaptic_weights(kernels);

    //    convolutional_layer->set_synaptic_weights_constant(0.5);
        convolutional_layer->set_biases_constant(0);

        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
        flatten_layer_inputs_dimensions(0) = 3;
        flatten_layer_inputs_dimensions(1) = 3;
        flatten_layer_inputs_dimensions(2) = 1;
        flatten_layer_inputs_dimensions(3) = 2;

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

        PerceptronLayer* perceptron_layer = new PerceptronLayer(9, 2);

        Tensor<type, 2> synaptic_weights(9,2);
        synaptic_weights(0,0) = 1;
        synaptic_weights(1,0) = 1;
        synaptic_weights(2,0) = 1;
        synaptic_weights(3,0) = 1;
        synaptic_weights(4,0) = 1;
        synaptic_weights(5,0) = 1;
        synaptic_weights(6,0) = 1;
        synaptic_weights(7,0) = 1;
        synaptic_weights(8,0) = 1;

        synaptic_weights(0,1) = 2;
        synaptic_weights(1,1) = 2;
        synaptic_weights(2,1) = 2;
        synaptic_weights(3,1) = 2;
        synaptic_weights(4,1) = 2;
        synaptic_weights(5,1) = 2;
        synaptic_weights(6,1) = 2;
        synaptic_weights(7,1) = 2;
        synaptic_weights(8,1) = 2;
    //    perceptron_layer->set_synaptic_weights_constant(1);
        perceptron_layer->set_synaptic_weights(synaptic_weights);
        perceptron_layer->set_biases_constant(0);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        cout << "Perceptron layer weights: " << perceptron_layer->get_synaptic_weights() << endl;

        neural_network.add_layer(convolutional_layer);
        neural_network.add_layer(flatten_layer);
        neural_network.add_layer(perceptron_layer);

        NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

        neural_network.forward_propagate(batch, forward_propagation);
//        forward_propagation.print();

        MeanSquaredError mean_squared_error(&neural_network, &data_set);

        BackPropagation back_propagation(2, &mean_squared_error);

        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

    //    cout << "Gradient: " << back_propagation.gradient << endl;

        numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        cout << "Numerical gradient: " << endl << numerical_gradient << endl;

    }

    // Inputs 3x3x2x2; Filters: 2x2x2; Perceptrons: 1 --> Working (4-Jan-23)

    {
        bool is_training = true;

        const Index input_channels = 2;
        const Index input_height = 3;
        const Index input_raw_variables_number = 3;

        const Index images_number = 2;

        Tensor<Index, 1> input_dimensions(3);
        input_dimensions(0) = input_channels; // Channels number
        input_dimensions(1) = input_height; // rows number
        input_dimensions(2) = input_raw_variables_number; // columns number

        Tensor<type, 2> data(images_number,19);

        // Image 1

        data(0,0) = 1;
        data(0,1) = 10;
        data(0,2) = 2;
        data(0,3) = 11;
        data(0,4) = 3;
        data(0,5) = 12;
        data(0,6) = 4;
        data(0,7) = 13;
        data(0,8) = 5;
        data(0,9) = 14;
        data(0,10) = 6;
        data(0,11) = 15;
        data(0,12) = 7;
        data(0,13) = 16;
        data(0,14) = 8;
        data(0,15) = 17;
        data(0,16) = 9;
        data(0,17) = 18;

        data(0,18) = 0; // Target

        // Image 2

        data(1,0) = 19;
        data(1,1) = 28;
        data(1,2) = 20;
        data(1,3) = 29;
        data(1,4) = 21;
        data(1,5) = 30;
        data(1,6) = 22;
        data(1,7) = 31;
        data(1,8) = 23;
        data(1,9) = 32;
        data(1,10) = 24;
        data(1,11) = 33;
        data(1,12) = 25;
        data(1,13) = 34;
        data(1,14) = 26;
        data(1,15) = 35;
        data(1,16) = 27;
        data(1,17) = 36;

        data(1,18) = 1; // Target

        DataSet data_set(images_number,1,1);
        data_set.set_input_dimensions(input_dimensions);

        data_set.set_data(data); // 2d data

        const Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
//        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        Tensor<Index, 1> input_variables_indices(18);
        input_variables_indices(0) = 0;
        input_variables_indices(1) = 1;
        input_variables_indices(2) = 2;
        input_variables_indices(3) = 3;
        input_variables_indices(4) = 4;
        input_variables_indices(5) = 5;
        input_variables_indices(6) = 6;
        input_variables_indices(7) = 7;
        input_variables_indices(8) = 8;
        input_variables_indices(9) = 9;
        input_variables_indices(10) = 10;
        input_variables_indices(11) = 11;
        input_variables_indices(12) = 12;
        input_variables_indices(13) = 13;
        input_variables_indices(14) = 14;
        input_variables_indices(15) = 15;
        input_variables_indices(16) = 16;
        input_variables_indices(17) = 17;

        Tensor<Index, 1> target_variables_indices(1);
        target_variables_indices(0) = 18;

        Batch batch(images_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        Tensor<Index, 1> convolutional_layer_inputs_dimensions(4);
        convolutional_layer_inputs_dimensions(0) = input_height;
        convolutional_layer_inputs_dimensions(1) = input_raw_variables_number;
        convolutional_layer_inputs_dimensions(2) = input_channels;
        convolutional_layer_inputs_dimensions(3) = images_number;

        NeuralNetwork neural_network;

        const Index kernel_height = 2;
        const Index kernels_raw_variables_number = 2;
        const Index kernel_channels = input_channels;
        const Index kernels_number = 2;

        Tensor<Index, 1> convolutional_layer_kernels_dimensions(4);
        convolutional_layer_kernels_dimensions(0) = kernel_height;
        convolutional_layer_kernels_dimensions(1) = kernels_raw_variables_number;
        convolutional_layer_kernels_dimensions(2) = kernels_number;
        convolutional_layer_kernels_dimensions(3) = kernel_channels;

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

        Tensor<type, 4> kernels(kernel_height,kernels_raw_variables_number,images_number,kernel_channels);

        kernels(0,0,0,0) = type(0.5);
        kernels(0,1,0,0) = type(0.5);
        kernels(1,0,0,0) = type(0.5);
        kernels(1,1,0,0) = type(0.5);

        kernels(0,0,0,1) = type(0.5);
        kernels(0,1,0,1) = type(0.5);
        kernels(1,0,0,1) = type(0.5);
        kernels(1,1,0,1) = type(0.5);

        kernels(0,0,1,0) = type(0.7);
        kernels(0,1,1,0) = type(0.7);
        kernels(1,0,1,0) = type(0.7);
        kernels(1,1,1,0) = type(0.7);

        kernels(0,0,1,1) = type(0.7);
        kernels(0,1,1,1) = type(0.7);
        kernels(1,0,1,1) = type(0.7);
        kernels(1,1,1,1) = type(0.7);

        convolutional_layer->set_synaptic_weights(kernels);

        convolutional_layer->set_biases_constant(0);

        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
        flatten_layer_inputs_dimensions(0) = input_height-kernel_height+1;
        flatten_layer_inputs_dimensions(1) = input_raw_variables_number-kernels_raw_variables_number+1;
        flatten_layer_inputs_dimensions(2) = kernels_number;
        flatten_layer_inputs_dimensions(3) = images_number;

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

        const Index perceptron_layer_inputs_number = flatten_layer_inputs_dimensions(0)*flatten_layer_inputs_dimensions(1)*flatten_layer_inputs_dimensions(2);
        const Index perceptrons_number = 1;

        PerceptronLayer* perceptron_layer = new PerceptronLayer(perceptron_layer_inputs_number, perceptrons_number);

        Tensor<type, 2> synaptic_weights(perceptron_layer_inputs_number,perceptrons_number);

        for(Index i = 0; i < perceptron_layer_inputs_number; i++)
        {
            for(Index j = 0; j < perceptrons_number; j++)
            {
                synaptic_weights(i,j) = type(j+1);
            }
        }

//        synaptic_weights(0,0) = 1;
//        synaptic_weights(1,0) = 1;
//        synaptic_weights(2,0) = 1;
//        synaptic_weights(3,0) = 1;

//        synaptic_weights(0,1) = 2;
//        synaptic_weights(1,1) = 2;
//        synaptic_weights(2,1) = 2;
//        synaptic_weights(3,1) = 2;

        perceptron_layer->set_synaptic_weights(synaptic_weights);
        perceptron_layer->set_biases_constant(0);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        neural_network.add_layer(convolutional_layer);
        neural_network.add_layer(flatten_layer);
        neural_network.add_layer(perceptron_layer);

        NeuralNetworkForwardPropagation forward_propagation(images_number, &neural_network);

        neural_network.forward_propagate(batch, forward_propagation, is_training);
//        forward_propagation.print();

        MeanSquaredError mean_squared_error(&neural_network, &data_set);

        BackPropagation back_propagation(images_number, &mean_squared_error);

        mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

        cout << "Gradient: " << endl << back_propagation.gradient << endl;

        const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();

        cout << "Numerical gradient: " << endl << numerical_gradient << endl;
    }

    // Inputs 3x3x2x2; Filters: 2x2x2; Perceptrons: 1
    if(true)
    {
        bool is_training = true;

        const Index input_channels = 2;
        const Index input_height = 3;
        const Index input_raw_variables_number = 3;

        const Index images_number = 2;

        dimensions input_dimensions(3);
        input_dimensions[0] = input_channels; // Channels number
        input_dimensions[1] = input_height; // rows number
        input_dimensions[2] = input_raw_variables_number; // columns number

        Tensor<type, 2> data(images_number,19);

        // Image 1

        data(0,0) = 1;//17;
        data(0,1) = 1;//16;
        data(0,2) = 1;//11;
        data(0,3) = 1;//10;
        data(0,4) = 1;//5;
        data(0,5) = 1;//4;
        data(0,6) = 1;//15;
        data(0,7) = 1;//14;
        data(0,8) = 1;//9;
        data(0,9) = 1;//8;
        data(0,10) = 1;//3;
        data(0,11) = 1;//2;
        data(0,12) = 1;//13;
        data(0,13) = 1;//12;
        data(0,14) = 1;//7;
        data(0,15) = 1;//6;
        data(0,16) = 1;//1;
        data(0,17) = 1;//0;

        data(0,18) = 1; // Target

        // Image 2

        data(1,0) = 2;//36;
        data(1,1) = 2;//35;
        data(1,2) = 2;//30;
        data(1,3) = 2;//29;
        data(1,4) = 2;//24;
        data(1,5) = 2;//23;
        data(1,6) = 2;//34;
        data(1,7) = 2;//33;
        data(1,8) = 2;//28;
        data(1,9) = 2;//27;
        data(1,10) = 2;//22;
        data(1,11) = 2;//21;
        data(1,12) = 2;//32;
        data(1,13) = 2;//31;
        data(1,14) = 2;//26;
        data(1,15) = 2;//25;
        data(1,16) = 2;//20;
        data(1,17) = 2;//19;

        data(1,18) = 0; // Target

        DataSet data_set(images_number,1,1);

        data_set.set_data(data); // 2d data
//        data_set.set_data_random();

        cout << "Data: " << endl << data_set.get_data() << endl;

        data_set.set_input_dimensions(input_dimensions);

        data_set.set(DataSet::SampleUse::Training);

        const Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
//        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        Tensor<Index, 1> input_variables_indices(18);
        input_variables_indices(0) = 0;
        input_variables_indices(1) = 1;
        input_variables_indices(2) = 2;
        input_variables_indices(3) = 3;
        input_variables_indices(4) = 4;
        input_variables_indices(5) = 5;
        input_variables_indices(6) = 6;
        input_variables_indices(7) = 7;
        input_variables_indices(8) = 8;
        input_variables_indices(9) = 9;
        input_variables_indices(10) = 10;
        input_variables_indices(11) = 11;
        input_variables_indices(12) = 12;
        input_variables_indices(13) = 13;
        input_variables_indices(14) = 14;
        input_variables_indices(15) = 15;
        input_variables_indices(16) = 16;
        input_variables_indices(17) = 17;

        Tensor<Index, 1> target_variables_indices(1);
        target_variables_indices(0) = 18;

        Batch batch(images_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        dimensions convolutional_layer_1_inputs_dimensions({input_height,
                                                         input_raw_variables_number,
                                                         input_channels,
                                                         images_number});

        NeuralNetwork neural_network;

        const Index kernels1_rows_number = 2;
        const Index kernels1_columns_number = 2;
        const Index kernels1_channels_number = input_channels;
        const Index kernels1_number = 2;

        dimensions convolutional_layer_1_kernels_dimensions({kernels1_rows_number,
                                                          kernels1_columns_number,
                                                          kernels1_channels_number,
                                                          kernels1_number});

        ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer(convolutional_layer_1_inputs_dimensions,
                                                                           convolutional_layer_1_kernels_dimensions);

        dimensions convolutional_layer1_outputs_dimensions = convolutional_layer_1->get_output_dimensions();

        dimensions convolutional_layer2_inputs_dimensions({convolutional_layer1_outputs_dimensions[0],
                                                           convolutional_layer1_outputs_dimensions[1],
                                                           convolutional_layer1_outputs_dimensions[2],
                                                           images_number});

        const Index kernels2_rows_number = 1;
        const Index kernels2_columns_number = 1;
        const Index kernels2_channels_number = kernels1_number;
        const Index kernels2_number = 1;

        dimensions convolutional_layer2_kernels_dimensions({kernels2_rows_number,
                                                          kernels2_columns_number,
                                                          kernels2_channels_number,
                                                          kernels2_number});

        ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(convolutional_layer2_inputs_dimensions,
                                                                           convolutional_layer2_kernels_dimensions);

        convolutional_layer_2->set_activation_function(ConvolutionalLayer::ActivationFunction::Linear);

        convolutional_layer_2->set_parameters_random();

        dimensions convolutional_layer2_outputs_dimensions = convolutional_layer_2->get_output_dimensions();

        dimensions flatten_layer_inputs_dimensions({convolutional_layer2_outputs_dimensions[0],
                                                    convolutional_layer2_outputs_dimensions[1],
                                                    convolutional_layer2_outputs_dimensions[2],
                                                    images_number});

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

        const Index perceptron_layer_inputs_number = convolutional_layer2_outputs_dimensions[0]
                                                    *convolutional_layer2_outputs_dimensions[1]
                                                    *convolutional_layer2_outputs_dimensions[2];
        const Index perceptrons_number = 1;

        PerceptronLayer* perceptron_layer = new PerceptronLayer(perceptron_layer_inputs_number, perceptrons_number);

        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
        perceptron_layer->set_parameters_random();

        neural_network.add_layer(convolutional_layer_1);
        neural_network.add_layer(convolutional_layer_2);
        neural_network.add_layer(flatten_layer);
        neural_network.add_layer(perceptron_layer);

        ForwardPropagation forward_propagation(images_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        cout << "End forward propagate" << endl;

        forward_propagation.print();

      MeanSquaredError mean_squared_error(&neural_network, &data_set);
      BackPropagation back_propagation(images_number, &mean_squared_error);
      mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);

      const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();

      cout << "Gradient   ;    Numerical gradient  ; Error" << endl;

      for(Index i = 0; i < back_propagation.gradient.size(); i++)
      {
          cout << back_propagation.gradient(i) << " ; " << numerical_gradient(i) <<  " ; " <<
                  abs((back_propagation.gradient(i) - numerical_gradient(i))/numerical_gradient(i)*100)
               << "%" << endl;

      }
    }

    // Pooling layer

    if(false)
    {
        bool is_training = true;

        const Index input_channels = 2;
        const Index input_height = 3;
        const Index input_raw_variables_number = 3;

        const Index images_number = 2;

        dimensions input_dimensions(3);
        input_dimensions[0] = input_channels;
        input_dimensions[1] = input_height;
        input_dimensions[2] = input_raw_variables_number;

        Tensor<type, 2> data(images_number, 19);

        // Image 1

        data(0,0) = 1;//17;
        data(0,1) = 1;//16;
        data(0,2) = 1;//11;
        data(0,3) = 1;//10;
        data(0,4) = 1;//5;
        data(0,5) = 1;//4;
        data(0,6) = 1;//15;
        data(0,7) = 1;//14;
        data(0,8) = 1;//9;
        data(0,9) = 1;//8;
        data(0,10) = 1;//3;
        data(0,11) = 1;//2;
        data(0,12) = 1;//13;
        data(0,13) = 1;//12;
        data(0,14) = 1;//7;
        data(0,15) = 1;//6;
        data(0,16) = 1;//1;
        data(0,17) = 1;//0;

        data(0,18) = 1; // Target

        // Image 2

        data(1,0) = 2;//36;
        data(1,1) = 2;//35;
        data(1,2) = 2;//30;
        data(1,3) = 2;//29;
        data(1,4) = 2;//24;
        data(1,5) = 2;//23;
        data(1,6) = 2;//34;
        data(1,7) = 2;//33;
        data(1,8) = 2;//28;
        data(1,9) = 2;//27;
        data(1,10) = 2;//22;
        data(1,11) = 2;//21;
        data(1,12) = 2;//32;
        data(1,13) = 2;//31;
        data(1,14) = 2;//26;
        data(1,15) = 2;//25;
        data(1,16) = 2;//20;
        data(1,17) = 2;//19;

        data(1,18) = 0; // Target

        DataSet data_set(images_number,1,1);

        data_set.set_data(data); // 2d data
        data_set.set_data_random();

        data_set.set_input_dimensions(input_dimensions);

        data_set.set(DataSet::SampleUse::Training);

        const Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
//        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        Tensor<Index, 1> input_variables_indices(18);
        input_variables_indices(0) = 0;
        input_variables_indices(1) = 1;
        input_variables_indices(2) = 2;
        input_variables_indices(3) = 3;
        input_variables_indices(4) = 4;
        input_variables_indices(5) = 5;
        input_variables_indices(6) = 6;
        input_variables_indices(7) = 7;
        input_variables_indices(8) = 8;
        input_variables_indices(9) = 9;
        input_variables_indices(10) = 10;
        input_variables_indices(11) = 11;
        input_variables_indices(12) = 12;
        input_variables_indices(13) = 13;
        input_variables_indices(14) = 14;
        input_variables_indices(15) = 15;
        input_variables_indices(16) = 16;
        input_variables_indices(17) = 17;

        Tensor<Index, 1> target_variables_indices(1);
        target_variables_indices(0) = 18;

        Batch batch(images_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

//        batch.print();

        dimensions convolutional_layer_inputs_dimensions({input_height, input_raw_variables_number, input_channels, images_number});

        NeuralNetwork neural_network;

        const Index kernel_height = 2;
        const Index kernels_raw_variables_number = 2;
        const Index kernel_channels = input_channels;
        const Index kernels_number = 2;

        dimensions convolutional_layer_kernels_dimensions({kernel_height,kernels_raw_variables_number, kernel_channels, kernels_number});

        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(convolutional_layer_inputs_dimensions, convolutional_layer_kernels_dimensions);

        convolutional_layer->set_parameters_random();

        dimensions pooling_layer_inputs_dimensions = convolutional_layer->get_output_dimensions();
        dimensions pooling_layer_pools_dimensions(2);
        pooling_layer_pools_dimensions[0] = 2;
        pooling_layer_pools_dimensions[1] = 2;

        PoolingLayer* pooling_layer = new PoolingLayer(pooling_layer_inputs_dimensions, pooling_layer_pools_dimensions);
        pooling_layer->set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

//        Tensor<Index, 1> flatten_layer_inputs_dimensions(4);
//        flatten_layer_inputs_dimensions(0) = input_height-kernel_height+1;
//        flatten_layer_inputs_dimensions(1) = input_raw_variables_number-kernels_raw_variables_number+1;
//        flatten_layer_inputs_dimensions(2) = kernels_number;
//        flatten_layer_inputs_dimensions(3) = images_number;

        dimensions flatten_layer_inputs_dimensions({input_height-kernel_height+1,
                                                    input_raw_variables_number-kernels_raw_variables_number+1,
                                                    kernels_number,
                                                    images_number});

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);

        const Index perceptron_layer_inputs_number = (input_height-kernel_height+1)
                                                    *(input_raw_variables_number-kernels_raw_variables_number+1)
                                                    *kernels_number;
        const Index perceptrons_number = 1;

        PerceptronLayer* perceptron_layer = new PerceptronLayer(perceptron_layer_inputs_number, perceptrons_number);

        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
        perceptron_layer->set_parameters_random();

        neural_network.add_layer(convolutional_layer);
        neural_network.add_layer(pooling_layer);
        neural_network.add_layer(flatten_layer);
        neural_network.add_layer(perceptron_layer);

        ForwardPropagation forward_propagation(images_number, &neural_network);
        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);


      MeanSquaredError mean_squared_error(&neural_network, &data_set);

      BackPropagation back_propagation(images_number, &mean_squared_error);
      mean_squared_error.back_propagate(batch, forward_propagation, back_propagation);
      cout << "Gradient: " << endl << back_propagation.gradient << endl;

      const Tensor<type, 1> numerical_gradient = mean_squared_error.calculate_numerical_gradient();

      for(Index i = 0; i < back_propagation.gradient.size(); i++)
      {
          cout << back_propagation.gradient(i) << " ; " << numerical_gradient(i) <<  " ; " <<
                  abs((back_propagation.gradient(i) - numerical_gradient(i))/numerical_gradient(i)*100)
               << "%" << endl;
      }
    }

}

}
*/
