#include "pch.h"

#include "../opennn/forward_propagation.h"
#include "../opennn/back_propagation.h"
#include "../opennn/learning_rate_algorithm.h"
#include "../opennn/mean_squared_error.h"


TEST(LearningRateAlgorithmTest, DefaultConstructor)
{
    LearningRateAlgorithm learning_rate_algorithm;

    EXPECT_EQ(learning_rate_algorithm.has_loss_index(), false);
}


TEST(LearningRateAlgorithmTest, GeneralConstructor)
{
    MeanSquaredError mean_squared_error;
    LearningRateAlgorithm learning_rate_algorithm(&mean_squared_error);

    EXPECT_EQ(learning_rate_algorithm.has_loss_index(), true);
}


TEST(LearningRateAlgorithmTest, BracketingTriplet)
{
    Index samples_number;
    Index inputs_number;
    Index targets_number;
    Index neurons_number;

    Batch batch;

    ForwardPropagation forward_propagation;

    BackPropagation back_propagation;

    LearningRateAlgorithm::Triplet triplet;

    OptimizationAlgorithmData optimization_data;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;
    neurons_number = 1;

    DataSet data_set(samples_number, inputs_number, targets_number);
    data_set.set_data_random();

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { inputs_number }, { neurons_number }, { targets_number });
/*
    batch.set(samples_number, &data_set);
    forward_propagation.set(samples_number, &neural_network);
    back_propagation.set(samples_number, &sum_squared_error);

    //triplet = learning_rate_algorithm.calculate_bracketing_triplet(batch, forward_propagation, back_propagation, optimization_data);

    Tensor<Index, 3> samples_indices(0, 1, samples_number);

    LearningRateAlgorithm learning_rate_algorithm(&sum_squared_error);

    type loss = 0.0;
    Tensor<type, 1> training_direction;
    type initial_learning_rate = 0.0;

//    assert_true(triplet.A.first <= triplet.U.first, LOG);
//    assert_true(triplet.U.first <= triplet.B.first, LOG);
//    assert_true(triplet.A.second >= triplet.U.second, LOG);
//    assert_true(triplet.U.second <= triplet.B.second, LOG);
*/
}

/*

void LearningRateAlgorithmTest::test_calculate_bracketing_triplet()
{
    cout << "test_calculate_bracketing_triplet\n";


    // Test

    sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::L2);

    neural_network.set_parameters_random();

    //loss = sum_squared_error.calculate_training_loss();
    //training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);

    initial_learning_rate = 0.01;

    //triplet = learning_rate_algorithm.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

    assert_true(triplet.A.first <= triplet.U.first, LOG);
    assert_true(triplet.U.first <= triplet.B.first, LOG);
    assert_true(triplet.A.second >= triplet.U.second, LOG);
    assert_true(triplet.U.second <= triplet.B.second, LOG);

    // Test

    neural_network.set_parameters_constant(type(0));

    initial_learning_rate = 0.01;

    //triplet = learning_rate_algorithm.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

    // Test

    neural_network.set_parameters_constant(type(1));

    initial_learning_rate = 0.0;

    //triplet = learning_rate_algorithm.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

    // Test

    data_set.set(1, 1, 1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_random();

    initial_learning_rate = 0.001;

    //triplet = learning_rate_algorithm.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

    assert_true(triplet.A.first <= triplet.U.first, LOG);
    assert_true(triplet.U.first <= triplet.B.first, LOG);
    assert_true(triplet.A.second >= triplet.U.second, LOG);
    assert_true(triplet.U.second <= triplet.B.second, LOG);

    // Test

    data_set.set(3, 1, 1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {targets_number});
    neural_network.set_parameters_random();

    initial_learning_rate = 0.001;

    //triplet = learning_rate_algorithm.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

    assert_true(triplet.A.first <= triplet.U.first, LOG);
    assert_true(triplet.U.first <= triplet.B.first, LOG);
    assert_true(triplet.A.second >= triplet.U.second, LOG);
    assert_true(triplet.U.second <= triplet.B.second, LOG);
}


void LearningRateAlgorithmTest::test_calculate_golden_section_directional_point()
{
    cout << "test_calculate_golden_section_directional_point\n";

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Index neurons_number;

    data_set.set(1, 1, 1);

    Tensor<Index, 1> indices(1, 1, data_set.get_samples_number()-1);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {1, 1});

    LearningRateAlgorithm learning_rate_algorithm(&sum_squared_error);

    neural_network.set_parameters_constant(type(1));

    Tensor<type, 1> training_direction = gradient*(-1.0);
    type initial_learning_rate = 0.001;

    type loss_tolerance = 1.0e-6;
    learning_rate_algorithm.set_loss_tolerance(loss_tolerance);

    pair<type, type> directional_point
            = learning_rate_algorithm.calculate_golden_section_directional_point(loss, training_direction, initial_learning_rate);

    assert_true(directional_point.first >= type(0), LOG);
    assert_true(directional_point.second < loss, LOG);

}


void LearningRateAlgorithmTest::test_calculate_Brent_method_directional_point()
{
    cout << "test_calculate_Brent_method_directional_point\n";

    Index samples_number = 1;
    Index inputs_number = 1;
    Index targets_number = 1;

    Index neurons_number = 5;

    data_set.set(samples_number, inputs_number, targets_number);

    Tensor<Index, 1> indices(3);

    indices.setValues({inputs_number,targets_number,samples_number-1});

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {targets_number}, {neurons_number});
    neural_network.set_parameters_constant(type(1));

    // @todo loss_index.calculate_training_loss not available

    Tensor<type, 1> gradient = sum_squared_error.calculate_numerical_gradient();

    Tensor<type, 1> training_direction = gradient*(type(-1.0));

    type initial_learning_rate = 0.001;

    pair<type, type> directional_point
           = learning_rate_algorithm.calculate_directional_point(1e-2, training_direction, initial_learning_rate);

    assert_true(directional_point.first >= type(0), LOG);
    assert_true(directional_point.second < 1e-2, LOG);

}

}
*/
