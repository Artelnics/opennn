//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   T E S T   C L A S S             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "conjugate_gradient_test.h"

using namespace OpenNN;


ConjugateGradientTest::ConjugateGradientTest() : UnitTesting() 
{
    sum_squared_error.set(&neural_network, &data_set);

    conjugate_gradient.set_loss_index_pointer(&sum_squared_error);
}


ConjugateGradientTest::~ConjugateGradientTest()
{
}


void ConjugateGradientTest::test_constructor()
{
   cout << "test_constructor\n"; 

   // Default constructor

   ConjugateGradient cg1; 
   assert_true(!cg1.has_loss_index(), LOG);

   // Loss index constructor

   SumSquaredError sum_squared_error;

   ConjugateGradient cg2(&sum_squared_error);
   assert_true(cg2.has_loss_index(), LOG);
}


void ConjugateGradientTest::test_destructor()
{
   cout << "test_destructor\n";
}


void ConjugateGradientTest::test_get_training_direction_method()
{
   cout << "test_get_training_direction_method\n";

   conjugate_gradient.set_training_direction_method(ConjugateGradient::PR);

   ConjugateGradient::TrainingDirectionMethod training_direction_method = conjugate_gradient.get_training_direction_method();

   assert_true(training_direction_method == ConjugateGradient::PR, LOG);
}


void ConjugateGradientTest::test_get_training_direction_method_name()
{
   cout << "test_get_training_direction_method_name\n";
}


void ConjugateGradientTest::test_set_training_direction_method()
{
   cout << "test_set_training_direction_method\n";

   conjugate_gradient.set_training_direction_method(ConjugateGradient::FR);
   assert_true(conjugate_gradient.get_training_direction_method() == ConjugateGradient::FR, LOG);

   conjugate_gradient.set_training_direction_method(ConjugateGradient::PR);
   assert_true(conjugate_gradient.get_training_direction_method() == ConjugateGradient::PR, LOG);
}


/// @todo

void ConjugateGradientTest::test_calculate_PR_parameter()
{
   cout << "test_calculate_PR_parameter\n";

   data_set.set(1, 1, 2);
   data_set.set_data_random();

   neural_network.set(NeuralNetwork::Approximation, {1,1});

   neural_network.set_parameters_constant(2.0);
//   Tensor<type, 1> old_gradient = sum_squared_error.calculate_gradient();

   neural_network.set_parameters_constant(1.0);
//   Tensor<type, 1> gradient = sum_squared_error.calculate_gradient();

//   type PR_parameter = conjugate_gradient.calculate_PR_parameter(old_gradient, gradient);

//   assert_true(PR_parameter >= 0.0, LOG);
//   assert_true(PR_parameter <= 1.0, LOG);
}


/// @todo

void ConjugateGradientTest::test_calculate_FR_parameter()
{
   cout << "test_calculate_FR_parameter\n";

   data_set.set(1, 1, 2);
   data_set.set_data_random();

   neural_network.set(NeuralNetwork::Approximation, {1,1});

   neural_network.set_parameters_constant(2.0);
//   Tensor<type, 1> old_gradient = sum_squared_error.calculate_training_loss_gradient();

   neural_network.set_parameters_constant(1.0);
//   Tensor<type, 1> gradient = sum_squared_error.calculate_training_loss_gradient();

//   type FR_parameter = conjugate_gradient.calculate_FR_parameter(old_gradient, gradient);

//   assert_true(FR_parameter >= 0.0, LOG);
//   assert_true(FR_parameter <= 1.0, LOG);

}


void ConjugateGradientTest::test_calculate_PR_training_direction()
{
   cout << "test_calculate_PR_training_direction\n";


   neural_network.set(NeuralNetwork::Approximation, {1,1});

   Tensor<type, 1> old_gradient;
   Tensor<type, 1> gradient;
   Tensor<type, 1> old_training_direction;
   Tensor<type, 1> training_direction;

   Index parameters_number;

   // Test

   data_set.set(1, 1, 2);
   data_set.set_data_random();

   neural_network.set(NeuralNetwork::Approximation, {1,1});

   parameters_number = neural_network.get_parameters_number();

   old_gradient.resize(parameters_number);
   gradient.resize(parameters_number);
   old_training_direction.resize(parameters_number);
   training_direction.resize(parameters_number);

   old_gradient.setRandom();
   gradient.setRandom();
   old_training_direction.setRandom();
   training_direction.setRandom();

   conjugate_gradient.calculate_PR_training_direction(old_gradient, gradient, old_training_direction, training_direction);

   assert_true(training_direction.size() == parameters_number, LOG);
}


void ConjugateGradientTest::test_calculate_FR_training_direction()
{
   cout << "test_calculate_FR_training_direction\n";

   neural_network.set(NeuralNetwork::Approximation, {1,1});

   Tensor<type, 1> old_gradient;
   Tensor<type, 1> gradient;
   Tensor<type, 1> old_training_direction;
   Tensor<type, 1> training_direction;

   Index parameters_number;

   // Test

   data_set.set(1, 1, 2);
   data_set.set_data_random();

   neural_network.set(NeuralNetwork::Approximation, {1,1});

   parameters_number = neural_network.get_parameters_number();

   old_gradient.resize(parameters_number);
   gradient.resize(parameters_number);
   old_training_direction.resize(parameters_number);
   training_direction.resize(parameters_number);

   old_gradient.setRandom();
   gradient.setRandom();
   old_training_direction.setRandom();
   training_direction.setRandom();

   conjugate_gradient.calculate_FR_training_direction(old_gradient, gradient, old_training_direction, training_direction);

   assert_true(training_direction.size() == parameters_number, LOG);
}


void ConjugateGradientTest::test_calculate_training_direction()
{
   cout << "test_calculate_training_direction\n";

}


void ConjugateGradientTest::test_perform_training()
{
   cout << "test_perform_training\n";

   neural_network.set(NeuralNetwork::Approximation, {1, 2});
   neural_network.set_parameters_random();

   // Test

   data_set.set(1,1,1);
   data_set.set_data_constant(0.0);

   neural_network.set(NeuralNetwork::Approximation, {1, 1});
   neural_network.set_parameters_constant(0.0);

   conjugate_gradient.perform_training();
/*
   DataSet data_set(1, 1, 1);
   data_set.set_data_random();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1,1,1});

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   type old_loss;
   type loss;

   type training_loss_goal;

   ConjugateGradient conjugate_gradient(&sum_squared_error);

   type minimum_parameters_increment_norm;

   // Test

   neural_network.set_parameters_random();

//   old_loss = sum_squared_error.calculate_training_loss();

   conjugate_gradient.set_display(false);
   conjugate_gradient.set_maximum_epochs_number(1);

//   conjugate_gradient.perform_training();

//   loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   neural_network.set_parameters_constant(-1.0);

   minimum_parameters_increment_norm = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   conjugate_gradient.set_loss_goal(0.0);
   conjugate_gradient.set_minimum_loss_decrease(0.0);
   conjugate_gradient.set_gradient_norm_goal(0.0);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

//   conjugate_gradient.perform_training();

   // Performance goal

   neural_network.set_parameters_constant(-1.0);

   training_loss_goal = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(0.0);
   conjugate_gradient.set_loss_goal(training_loss_goal);
   conjugate_gradient.set_minimum_loss_decrease(0.0);
   conjugate_gradient.set_gradient_norm_goal(0.0);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

//   conjugate_gradient.perform_training();

//   loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < training_loss_goal, LOG);

   // Minimum evaluation improvement

   neural_network.set_parameters_constant(-1.0);

   type minimum_loss_decrease = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(0.0);
   conjugate_gradient.set_loss_goal(0.0);
   conjugate_gradient.set_minimum_loss_decrease(minimum_loss_decrease);
   conjugate_gradient.set_gradient_norm_goal(0.0);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

//   conjugate_gradient.perform_training();

   // Gradient norm goal 

   neural_network.set_parameters_constant(-1.0);

   type gradient_norm_goal = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(0.0);
   conjugate_gradient.set_loss_goal(0.0);
   conjugate_gradient.set_minimum_loss_decrease(0.0);
   conjugate_gradient.set_gradient_norm_goal(gradient_norm_goal);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

//   conjugate_gradient.perform_training();

//   type gradient_norm = sum_squared_error.calculate_gradient().calculate_norm();

//   assert_true(gradient_norm < gradient_norm_goal, LOG);
*/
}


void ConjugateGradientTest::test_to_XML()   
{
   cout << "test_to_XML\n";

//   tinyxml2::XMLDocument* cgd = conjugate_gradient.to_XML();
//   assert_true(cgd != nullptr, LOG);
}


void ConjugateGradientTest::test_from_XML()
{
   cout << "test_from_XML\n";

   ConjugateGradient cg1;
   ConjugateGradient cg2;

   tinyxml2::XMLDocument* document;

   // Test

   cg1.set_display(true);

//   document = cg1.to_XML();

//   cg2.from_XML(*document);

//   delete document;
}


void ConjugateGradientTest::run_test_case()
{
   cout << "Running conjugate gradient test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();


   // Get methods

   test_get_training_direction_method();
   test_get_training_direction_method_name();


   // Set methods

   test_set_training_direction_method();

   // Training methods

   test_calculate_PR_parameter();
   test_calculate_FR_parameter();
   test_calculate_FR_training_direction();
   test_calculate_PR_training_direction();
   test_calculate_training_direction();
   test_perform_training();

   // Serialization methods

   test_to_XML();   
   test_from_XML();

   cout << "End of conjugate gradient test case.\n\n";
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
