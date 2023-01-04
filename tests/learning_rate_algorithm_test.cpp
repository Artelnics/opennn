//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "learning_rate_algorithm_test.h"


LearningRateAlgorithmTest::LearningRateAlgorithmTest() : UnitTesting()
{
    sum_squared_error.set(&neural_network, &data_set);

    learning_rate_algorithm.set(&sum_squared_error);
}


LearningRateAlgorithmTest::~LearningRateAlgorithmTest()
{
}


void LearningRateAlgorithmTest::test_constructor()
{
    cout << "test_constructor\n";

    SumSquaredError sum_squared_error;

    // Test

    LearningRateAlgorithm tra1(&sum_squared_error);

    assert_true(tra1.has_loss_index(), LOG);

    // Test

    LearningRateAlgorithm tra2;

    assert_true(!tra2.has_loss_index(), LOG);
}


void LearningRateAlgorithmTest::test_destructor()
{
    cout << "test_destructor\n";

    SumSquaredError sum_squared_error;

    LearningRateAlgorithm* tra1 = new LearningRateAlgorithm(&sum_squared_error);

    delete tra1;
}


void LearningRateAlgorithmTest::test_calculate_bracketing_triplet()
{
    cout << "test_calculate_bracketing_triplet\n";

    Index samples_number;
    Index inputs_number;
    Index targets_number;
    Index neurons_number;

    DataSetBatch batch;

    NeuralNetworkForwardPropagation forward_propagation;

    LossIndexBackPropagation back_propagation;

    LearningRateAlgorithm::Triplet triplet;

    OptimizationAlgorithmData optimization_data;

    /// @todo
//    Test

//    samples_number = 1;
//    inputs_number = 1;
//    targets_number = 1;
//    neurons_number = 1;

//    data_set.set(samples_number, inputs_number, targets_number);
//    data_set.set_data_random();

//    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, targets_number});

//    batch.set(samples_number, &data_set);
//    forward_propagation.set(samples_number, &neural_network);
//    back_propagation.set(samples_number, &sum_squared_error);

//    triplet = learning_rate_algorithm.calculate_bracketing_triplet(batch, forward_propagation, back_propagation, optimization_data);

//    Tensor<Index, 3> samples_indices(0, 1, samples_number);


//    LearningRateAlgorithm learning_rate_algorithm(&sum_squared_error);

//    type loss = 0.0;
//    Tensor<type, 1> training_direction;
//    type initial_learning_rate = 0.0;

//    assert_true(triplet.A.first <= triplet.U.first, LOG);
//    assert_true(triplet.U.first <= triplet.B.first, LOG);
//    assert_true(triplet.A.second >= triplet.U.second, LOG);
//    assert_true(triplet.U.second <= triplet.B.second, LOG);

    // Test

//            sum_squared_error.set_regularization_method(LossIndex::RegularizationMethod::L2);

//    neural_network.set_parameters_random();

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.01;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

//    assert_true(triplet.A.first <= triplet.U.first, LOG);
//    assert_true(triplet.U.first <= triplet.B.first, LOG);
//    assert_true(triplet.A.second >= triplet.U.second, LOG);
//    assert_true(triplet.U.second <= triplet.B.second, LOG);

//    Test

//            neural_network.set_parameters_constant(type(0));

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.01;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

//    Test

//            neural_network.set_parameters_constant(type(1));

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.0;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

//    Test

//            data_set.set(1, 1, 1);
//    data_set.set_data_random();

//    samples_indices.set(0, 1, data_set.get_samples_number()-1);

//    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
//    neural_network.set_parameters_random();

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.001;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

//    assert_true(triplet.A.first <= triplet.U.first, LOG);
//    assert_true(triplet.U.first <= triplet.B.first, LOG);
//    assert_true(triplet.A.second >= triplet.U.second, LOG);
//    assert_true(triplet.U.second <= triplet.B.second, LOG);

//    Test

//            data_set.set(3, 1, 1);
//    data_set.set_data_random();

//    samples_indices.set(0, 1, data_set.get_samples_number()-1);

//    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number});
//    neural_network.set_parameters_random();

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.001;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

//    assert_true(triplet.A.first <= triplet.U.first, LOG);
//    assert_true(triplet.U.first <= triplet.B.first, LOG);
//    assert_true(triplet.A.second >= triplet.U.second, LOG);
//    assert_true(triplet.U.second <= triplet.B.second, LOG);
}


void LearningRateAlgorithmTest::test_calculate_golden_section_directional_point()
{
    cout << "test_calculate_golden_section_directional_point\n";

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Index neurons_number;

    data_set.set(1, 1, 1);
//    Tensor<Index, 1> indices(1,1,data_set.get_samples_number()-1);

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1, 1});

//    LearningRateAlgorithm learning_rate_algorithm(&sum_squared_error);

//    neural_network.set_parameters_constant(type(1));

//    type loss = sum_squared_error.calculate_training_loss();
//    Tensor<type, 1> gradient = sum_squared_error.calculate_training_loss_gradient();

//    Tensor<type, 1> training_direction = gradient*(-1.0);
//    type initial_learning_rate = 0.001;

//    type loss_tolerance = 1.0e-6;
//    tra.set_loss_tolerance(loss_tolerance);

//    pair<type,type> directional_point
//            = tra.calculate_golden_section_directional_point(loss, training_direction, initial_learning_rate);

//    assert_true(directional_point.first >= type(0), LOG);
//    assert_true(directional_point.second < loss, LOG);
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

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, targets_number,neurons_number});
    neural_network.set_parameters_constant(type(1));

    /// @todo loss_index.calculate_training_loss not available

//    type loss = sum_squared_error.calculate_training_loss();
//    Tensor<type, 1> gradient = sum_squared_error.calculate_numerical_differentiation_gradient();

//    Tensor<type, 1> training_direction = gradient*(-1.0);
//    type initial_learning_rate = 0.001;

//    pair<type,type> directional_point
//            = learning_rate_algorithm.calculate_directional_point(1e-2, training_direction, initial_learning_rate);

//    assert_true(directional_point.first >= type(0), LOG);
//    assert_true(directional_point.second < 1e-2, LOG);

}


void LearningRateAlgorithmTest::run_test_case()
{
    cout << "Running learning rate algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Training methods

    test_calculate_bracketing_triplet();

    test_calculate_golden_section_directional_point();

    test_calculate_Brent_method_directional_point();

    cout << "End of learning rate algorithm test case.\n\n";
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
