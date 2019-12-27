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
}


ConjugateGradientTest::~ConjugateGradientTest()
{
}


void ConjugateGradientTest::test_constructor()
{
   cout << "test_constructor\n"; 

   SumSquaredError sum_squared_error;

   // Default constructor

   ConjugateGradient cg1; 
   assert_true(cg1.has_loss_index() == false, LOG);

   // Loss index constructor

   ConjugateGradient cg2(&sum_squared_error);
   assert_true(cg2.has_loss_index() == true, LOG);
}


void ConjugateGradientTest::test_destructor()
{
   cout << "test_destructor\n";
}


void ConjugateGradientTest::test_get_training_direction_method()
{
   cout << "test_get_training_direction_method\n";

   ConjugateGradient conjugate_gradient;

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

   ConjugateGradient conjugate_gradient;

   conjugate_gradient.set_training_direction_method(ConjugateGradient::FR);
   assert_true(conjugate_gradient.get_training_direction_method() == ConjugateGradient::FR, LOG);

   conjugate_gradient.set_training_direction_method(ConjugateGradient::PR);
   assert_true(conjugate_gradient.get_training_direction_method() == ConjugateGradient::PR, LOG);
}


void ConjugateGradientTest::test_set_reserve_all_training_history()
{
   cout << "test_set_reserve_all_training_history\n";

   ConjugateGradient conjugate_gradient;
   conjugate_gradient.set_reserve_all_training_history(true);

   assert_true(conjugate_gradient.get_reserve_training_error_history() == true, LOG);
   assert_true(conjugate_gradient.get_reserve_selection_error_history() == true, LOG);
}


/// @todo

void ConjugateGradientTest::test_calculate_PR_parameter()
{
   cout << "test_calculate_PR_parameter\n";

   DataSet data_set(1, 1, 2);
   data_set.randomize_data_normal();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1 ,1});
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   ConjugateGradient conjugate_gradient(&sum_squared_error);

   neural_network.initialize_parameters(2.0);
//   Vector<double> old_gradient = sum_squared_error.calculate_gradient();

   neural_network.initialize_parameters(1.0);
//   Vector<double> gradient = sum_squared_error.calculate_gradient();

//   double PR_parameter = conjugate_gradient.calculate_PR_parameter(old_gradient, gradient);

//   assert_true(PR_parameter >= 0.0, LOG);
//   assert_true(PR_parameter <= 1.0, LOG);

}


/// @todo

void ConjugateGradientTest::test_calculate_FR_parameter()
{
   cout << "test_calculate_FR_parameter\n";
/*
   DataSet data_set(1, 1, 2);
   data_set.randomize_data_normal();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1 ,1});
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   ConjugateGradient conjugate_gradient(&sum_squared_error);

   neural_network.initialize_parameters(2.0);
   Vector<double> old_gradient = sum_squared_error.calculate_training_loss_gradient();

   neural_network.initialize_parameters(1.0);
   Vector<double> gradient = sum_squared_error.calculate_training_loss_gradient();

   double FR_parameter = conjugate_gradient.calculate_FR_parameter(old_gradient, gradient);

   assert_true(FR_parameter >= 0.0, LOG);
   assert_true(FR_parameter <= 1.0, LOG);
*/
}


/// @todo

void ConjugateGradientTest::test_calculate_PR_training_direction()
{
   cout << "test_calculate_PR_training_direction\n";
/*
   DataSet data_set(1, 1, 2);
   data_set.randomize_data_normal();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1 ,1});
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   ConjugateGradient conjugate_gradient(&sum_squared_error);

   neural_network.initialize_parameters(2.0);
   Vector<double> old_gradient = sum_squared_error.calculate_training_error_gradient();
   Vector<double> old_training_direction = old_gradient;   

   neural_network.initialize_parameters(1.0);
   Vector<double> gradient = sum_squared_error.calculate_training_error_gradient();

   Vector<double> PR_training_direction 
   = conjugate_gradient.calculate_PR_training_direction(old_gradient, gradient, old_training_direction);

   size_t parameters_number = neural_network.get_parameters_number();

   assert_true(PR_training_direction.size() == parameters_number, LOG);
*/
}


/// @todo

void ConjugateGradientTest::test_calculate_FR_training_direction()
{
   cout << "test_calculate_FR_training_direction\n";
/*
   DataSet data_set(1, 1, 2);
   data_set.randomize_data_normal();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1 ,1});
   SumSquaredError sum_squared_error(&neural_network, &data_set);
   ConjugateGradient conjugate_gradient(&sum_squared_error);

   neural_network.initialize_parameters(2.0);
   Vector<double> old_gradient = sum_squared_error.calculate_training_error_gradient();
   Vector<double> old_training_direction = old_gradient;   

   neural_network.initialize_parameters(1.0);
   Vector<double> gradient = sum_squared_error.calculate_training_error_gradient();
	
   Vector<double> FR_training_direction 
   = conjugate_gradient.calculate_FR_training_direction(old_gradient, gradient, old_training_direction);

   size_t parameters_number = neural_network.get_parameters_number();

   assert_true(FR_training_direction.size() == parameters_number, LOG);
*/
}


void ConjugateGradientTest::test_calculate_training_direction()
{
   cout << "test_calculate_training_direction\n";

}


void ConjugateGradientTest::test_perform_training()
{
   cout << "test_perform_training\n";

   DataSet data_set(1, 1, 1);
   data_set.randomize_data_normal();

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1, 1, 1});

   SumSquaredError sum_squared_error(&neural_network, &data_set);

   double old_loss;
   double loss;

   double loss_goal;

   ConjugateGradient conjugate_gradient(&sum_squared_error);

   double minimum_parameters_increment_norm;

   // Test

   neural_network.randomize_parameters_normal();

   old_loss = sum_squared_error.calculate_training_loss();

   conjugate_gradient.set_display(false);
   conjugate_gradient.set_maximum_epochs_number(1);

   conjugate_gradient.perform_training();

   loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < old_loss, LOG);

   // Minimum parameters increment norm

   neural_network.initialize_parameters(-1.0);

   minimum_parameters_increment_norm = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
   conjugate_gradient.set_loss_goal(0.0);
   conjugate_gradient.set_minimum_loss_decrease(0.0);
   conjugate_gradient.set_gradient_norm_goal(0.0);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

   conjugate_gradient.perform_training();

   // Performance goal

   neural_network.initialize_parameters(-1.0);

   loss_goal = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(0.0);
   conjugate_gradient.set_loss_goal(loss_goal);
   conjugate_gradient.set_minimum_loss_decrease(0.0);
   conjugate_gradient.set_gradient_norm_goal(0.0);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

   conjugate_gradient.perform_training();

   loss = sum_squared_error.calculate_training_loss();

   assert_true(loss < loss_goal, LOG);

   // Minimum evaluation improvement

   neural_network.initialize_parameters(-1.0);

   double minimum_loss_increase = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(0.0);
   conjugate_gradient.set_loss_goal(0.0);
   conjugate_gradient.set_minimum_loss_decrease(minimum_loss_increase);
   conjugate_gradient.set_gradient_norm_goal(0.0);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

   conjugate_gradient.perform_training();

   // Gradient norm goal 

   neural_network.initialize_parameters(-1.0);

   double gradient_norm_goal = 0.1;

   conjugate_gradient.set_minimum_parameters_increment_norm(0.0);
   conjugate_gradient.set_loss_goal(0.0);
   conjugate_gradient.set_minimum_loss_decrease(0.0);
   conjugate_gradient.set_gradient_norm_goal(gradient_norm_goal);
   conjugate_gradient.set_maximum_epochs_number(1000);
   conjugate_gradient.set_maximum_time(1000.0);

   conjugate_gradient.perform_training();

//   double gradient_norm = sum_squared_error.calculate_gradient().calculate_norm();

//   assert_true(gradient_norm < gradient_norm_goal, LOG);

}


void ConjugateGradientTest::test_to_XML()   
{
   cout << "test_to_XML\n";

   ConjugateGradient conjugate_gradient;

   tinyxml2::XMLDocument* cgd = conjugate_gradient.to_XML();
   assert_true(cgd != nullptr, LOG);
}


void ConjugateGradientTest::test_from_XML()
{
   cout << "test_from_XML\n";

   ConjugateGradient cg1;
   ConjugateGradient cg2;

   tinyxml2::XMLDocument* document;

   // Test

   cg1.set_display(true);

   document = cg1.to_XML();

   cg2.from_XML(*document);

   delete document;

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

   // Training history methods

   test_set_reserve_all_training_history();

   // Serialization methods

   test_to_XML();   
   test_from_XML();

   cout << "End of conjugate gradient test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
