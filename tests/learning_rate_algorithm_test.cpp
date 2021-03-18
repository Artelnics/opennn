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
}


LearningRateAlgorithmTest::~LearningRateAlgorithmTest()
{
}


void LearningRateAlgorithmTest::test_constructor()
{
   cout << "test_constructor\n"; 

   SumSquaredError sum_squared_error;

   LearningRateAlgorithm tra1(&sum_squared_error);

   assert_true(tra1.has_loss_index(), LOG);

   LearningRateAlgorithm tra2;

   assert_true(!tra2.has_loss_index(), LOG);
}


void LearningRateAlgorithmTest::test_destructor()
{
   cout << "test_destructor\n"; 
}


void LearningRateAlgorithmTest::test_get_loss_index_pointer()
{
   cout << "test_get_loss_index_pointer\n"; 
   
   SumSquaredError sum_squared_error;

   LearningRateAlgorithm tra(&sum_squared_error);

   LossIndex* pfp = tra.get_loss_index_pointer();

   assert_true(pfp != nullptr, LOG);
}


void LearningRateAlgorithmTest::test_get_learning_rate_method()
{
   cout << "test_get_learning_rate_method\n";

   LearningRateAlgorithm tra;

   tra.set_learning_rate_method(LearningRateAlgorithm::GoldenSection);
   assert_true(tra.get_learning_rate_method() == LearningRateAlgorithm::GoldenSection, LOG);

   tra.set_learning_rate_method(LearningRateAlgorithm::BrentMethod);
   assert_true(tra.get_learning_rate_method() == LearningRateAlgorithm::BrentMethod, LOG);
}


void LearningRateAlgorithmTest::test_get_learning_rate_method_name()
{
   cout << "test_get_learning_rate_method_name\n";
}


void LearningRateAlgorithmTest::test_get_display()
{
   cout << "test_get_display\n";

   LearningRateAlgorithm tra;

   tra.set_display(false);

   assert_true(!tra.get_display(), LOG);
}


void LearningRateAlgorithmTest::test_get_loss_tolerance()
{
   cout << "test_get_loss_tolerance\n"; 
}


void LearningRateAlgorithmTest::test_set()
{
   cout << "test_set\n"; 
}


void LearningRateAlgorithmTest::test_set_default()
{
   cout << "test_set_default\n"; 
}


void LearningRateAlgorithmTest::test_set_loss_index_pointer()
{
   cout << "test_set_loss_index_pointer\n"; 
}


void LearningRateAlgorithmTest::test_set_display()
{
   cout << "test_set_display\n"; 
}


void LearningRateAlgorithmTest::test_set_learning_rate_method()
{
   cout << "test_set_learning_rate_method\n";
}


void LearningRateAlgorithmTest::test_set_loss_tolerance()
{
   cout << "test_set_loss_tolerance\n"; 
}


void LearningRateAlgorithmTest::test_calculate_directional_point()
{
   cout << "test_calculate_directional_point\n";
}


void LearningRateAlgorithmTest::test_calculate_fixed_directional_point()
{
   cout << "test_calculate_fixed_directional_point\n";

//   Tensor<Index, 1> indices;

//   NeuralNetwork neural_network;

//   Tensor<Index, 1> architecture;
//   Tensor<type, 1> parameters;

//   SumSquaredError sum_squared_error(&neural_network);

//   type loss;
//   Tensor<type, 1> gradient;

//   LearningRateAlgorithm tra(&sum_squared_error);

//   Tensor<type, 1> training_direction;
//   type learning_rate;

//   pair<type,type> directional_point;

   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);

//   neural_network.set_parameters_constant(1.0);

//   loss = sum_squared_error.calculate_training_loss();

//   gradient = sum_squared_error.calculate_training_loss_gradient();

//   training_direction = -gradient;
//   learning_rate = 0.001;

//   directional_point = tra.calculate_fixed_directional_point(loss, training_direction, learning_rate);

//   assert_true(directional_point.second < loss, LOG);

//   assert_true(abs(directional_point.second - sum_squared_error.calculate_training_loss(training_direction, learning_rate)) <= numeric_limits<type>::min(), LOG);

//   parameters = neural_network.get_parameters();

//   neural_network.set_parameters(parameters + training_direction*learning_rate);

//   assert_true(abs(directional_point.second - sum_squared_error.calculate_training_loss()) <= numeric_limits<type>::min(), LOG);

   // Test

//   architecture.setValues({1,1});

//   neural_network.set(NeuralNetwork::Approximation, architecture);

//   neural_network.set_parameters_constant(1.0);

//   training_direction.set(2, -1.0);
//   learning_rate = 1.0;

//   directional_point = tra.calculate_fixed_directional_point(3.14, training_direction, learning_rate);

//   assert_true(directional_point.first == 1.0, LOG);
//   assert_true(directional_point.second == 0.0, LOG);
}


void LearningRateAlgorithmTest::test_calculate_bracketing_triplet() // @todo
{
    cout << "test_calculate_bracketing_triplet\n";

//    DataSet data_set(2, 1, 1);

//    data_set.set_data_random();

//    Tensor<Index, 1> samples_indices(0, 1, data_set.get_samples_number()-1);

//    Tensor<Index, 1> architecture;

//    architecture.setValues({1,1});

//    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);

//    SumSquaredError sum_squared_error(&neural_network, &data_set);

//    LearningRateAlgorithm tra(&sum_squared_error);

//    type loss = 0.0;
//    Tensor<type, 1> training_direction;
//    type initial_learning_rate = 0.0;

//    LearningRateAlgorithm::Triplet triplet;

    // Test

//    sum_squared_error.set_regularization_method(LossIndex::L2);

//    neural_network.set_parameters_random();

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.01;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

//    assert_true(triplet.A.first <= triplet.U.first, LOG);
//    assert_true(triplet.U.first <= triplet.B.first, LOG);
//    assert_true(triplet.A.second >= triplet.U.second, LOG);
//    assert_true(triplet.U.second <= triplet.B.second, LOG);

    // Test

//    neural_network.set_parameters_constant(0.0);

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.01;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

    // Test

//    neural_network.set_parameters_constant(1.0);

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.0;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

    // Test

//    data_set.set(1, 1, 1);
//    data_set.set_data_random();

//    samples_indices.set(0, 1, data_set.get_samples_number()-1);

//    Tensor<Index, 1> architecture;

//    architecture.setValues({1,1});

//    neural_network.set(NeuralNetwork::Approximation, architecture);
//    neural_network.set_parameters_random();

//    loss = sum_squared_error.calculate_training_loss();
//    training_direction = sum_squared_error.calculate_training_loss_gradient()*(-1.0);
//    initial_learning_rate = 0.001;

//    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

//    assert_true(triplet.A.first <= triplet.U.first, LOG);
//    assert_true(triplet.U.first <= triplet.B.first, LOG);
//    assert_true(triplet.A.second >= triplet.U.second, LOG);
//    assert_true(triplet.U.second <= triplet.B.second, LOG);

    // Test

//    data_set.set(3, 1, 1);
//    data_set.set_data_random();

//    samples_indices.set(0, 1, data_set.get_samples_number()-1);

//    architecture.setValues({1,1});

//    neural_network.set(NeuralNetwork::Approximation, architecture);
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

//   DataSet data_set(1, 1, 1);
//   Tensor<Index, 1> indices(1,1,data_set.get_samples_number()-1);

//   Tensor<Index, 1> architecture;

//   architecture.setValues({1,1});

//   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);

//   SumSquaredError sum_squared_error(&neural_network);

//   LearningRateAlgorithm tra(&sum_squared_error);

//   neural_network.set_parameters_constant(1.0);

//   type loss = sum_squared_error.calculate_training_loss();
//   Tensor<type, 1> gradient = sum_squared_error.calculate_training_loss_gradient();

//   Tensor<type, 1> training_direction = gradient*(-1.0);
//   type initial_learning_rate = 0.001;

//   type loss_tolerance = 1.0e-6;
//   tra.set_loss_tolerance(loss_tolerance);
  
//   pair<type,type> directional_point
//   = tra.calculate_golden_section_directional_point(loss, training_direction, initial_learning_rate);

//   assert_true(directional_point.first >= 0.0, LOG);
//   assert_true(directional_point.second < loss, LOG);
}


void LearningRateAlgorithmTest::test_calculate_Brent_method_directional_point()
{
   cout << "test_calculate_Brent_method_directional_point\n";

   DataSet data_set(1, 1, 1);
   Tensor<Index, 1> indices(3);
   indices.setValues({1,1,data_set.get_samples_number()-1});

   Tensor<Index, 1> architecture(2);

   architecture.setValues({1,1});

   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_constant(1.0);
//   type loss = sum_squared_error.calculate_training_loss();
//   Tensor<type, 1> gradient = sum_squared_error.calculate_training_loss_gradient();

//   Tensor<type, 1> training_direction = gradient*(-1.0);
//   type initial_learning_rate = 0.001;

//   type loss_tolerance = 1.0e-6;
//   tra.set_loss_tolerance(loss_tolerance);

//   pair<type,type> directional_point
//   = tra.calculate_Brent_method_directional_point(loss, training_direction, initial_learning_rate);

//   assert_true(directional_point.first >= 0.0, LOG);
//   assert_true(directional_point.second < loss, LOG);
}


void LearningRateAlgorithmTest::test_to_XML()
{
   cout << "test_to_XML\n";

   LearningRateAlgorithm  tra;

//   tinyxml2::XMLDocument* document = tra.to_XML();

//   assert_true(document != nullptr, LOG);

//   delete document;
}


void LearningRateAlgorithmTest::run_test_case()
{
   cout << "Running training rate algorithm test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_loss_index_pointer();

   // Training operators

   test_get_learning_rate_method();
   test_get_learning_rate_method_name();

   // Training parameters

   test_get_loss_tolerance();

   // Utilities
   
   test_get_display();

   // Set methods

   test_set();
   test_set_default();   
   test_set_loss_index_pointer();

   // Training operators

   test_set_learning_rate_method();

   // Training parameters

   test_set_loss_tolerance();

   // Utilities

   test_set_display();

   // Training methods

   test_calculate_bracketing_triplet();
   test_calculate_fixed_directional_point();
   test_calculate_golden_section_directional_point();
   test_calculate_Brent_method_directional_point();
   test_calculate_directional_point();

   // Serialization methods

   test_to_XML();

   cout << "End of training rate algorithm test case.\n\n";
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
