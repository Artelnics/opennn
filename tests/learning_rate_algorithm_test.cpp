/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   R A T E   A L G O R I T H M   T E S T   C L A S S                                        */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "learning_rate_algorithm_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR 

LearningRateAlgorithmTest::LearningRateAlgorithmTest() : UnitTesting()
{
}


// DESTRUCTOR

LearningRateAlgorithmTest::~LearningRateAlgorithmTest()
{
}


// METHODS

void LearningRateAlgorithmTest::test_constructor()
{
   message += "test_constructor\n"; 

   SumSquaredError sse;

   LearningRateAlgorithm tra1(&sse);

   assert_true(tra1.has_loss_index() == true, LOG);

   LearningRateAlgorithm tra2;

   assert_true(tra2.has_loss_index() == false, LOG);
}


void LearningRateAlgorithmTest::test_destructor()
{
   message += "test_destructor\n"; 
}


void LearningRateAlgorithmTest::test_get_loss_index_pointer()
{
   message += "test_get_loss_index_pointer\n"; 
   
   SumSquaredError sse;

   LearningRateAlgorithm tra(&sse);

   LossIndex* pfp = tra.get_loss_index_pointer();

   assert_true(pfp != nullptr, LOG);
}


void LearningRateAlgorithmTest::test_get_training_rate_method()
{
   message += "test_get_training_rate_method\n"; 

   LearningRateAlgorithm tra;

   tra.set_training_rate_method(LearningRateAlgorithm::Fixed);
   assert_true(tra.get_training_rate_method() == LearningRateAlgorithm::Fixed, LOG);

   tra.set_training_rate_method(LearningRateAlgorithm::GoldenSection);
   assert_true(tra.get_training_rate_method() == LearningRateAlgorithm::GoldenSection, LOG);

   tra.set_training_rate_method(LearningRateAlgorithm::BrentMethod);
   assert_true(tra.get_training_rate_method() == LearningRateAlgorithm::BrentMethod, LOG);
}


void LearningRateAlgorithmTest::test_get_training_rate_method_name()
{
   message += "test_get_training_rate_method_name\n"; 
}


void LearningRateAlgorithmTest::test_get_warning_training_rate()
{
   message += "test_get_warning_training_rate\n";

   LearningRateAlgorithm tra;

   tra.set_warning_training_rate(0.0);

   assert_true(tra.get_warning_training_rate() == 0.0, LOG);
}


void LearningRateAlgorithmTest::test_get_error_training_rate()
{
   message += "test_get_error_training_rate\n";

   LearningRateAlgorithm tra;

   tra.set_error_training_rate(0.0);

   assert_true(tra.get_error_training_rate() == 0.0, LOG);
}


void LearningRateAlgorithmTest::test_get_display()
{
   message += "test_get_warning_gradient_norm\n"; 

   LearningRateAlgorithm tra;

   tra.set_display(false);

   assert_true(tra.get_display() == false, LOG);
}


void LearningRateAlgorithmTest::test_get_loss_tolerance()
{
   message += "test_get_loss_tolerance\n"; 
}


void LearningRateAlgorithmTest::test_set()
{
   message += "test_set\n"; 
}


void LearningRateAlgorithmTest::test_set_default()
{
   message += "test_set_default\n"; 
}


void LearningRateAlgorithmTest::test_set_loss_index_pointer()
{
   message += "test_set_loss_index_pointer\n"; 
}


void LearningRateAlgorithmTest::test_set_display()
{
   message += "test_set_display\n"; 
}


void LearningRateAlgorithmTest::test_set_training_rate_method()
{
   message += "test_set_training_rate_method\n"; 
}


void LearningRateAlgorithmTest::test_set_loss_tolerance()
{
   message += "test_set_loss_tolerance\n"; 
}


void LearningRateAlgorithmTest::test_set_warning_training_rate()
{
   message += "test_set_warning_training_rate\n"; 
}


void LearningRateAlgorithmTest::test_set_error_training_rate()
{
   message += "test_set_error_training_rate\n"; 
}


void LearningRateAlgorithmTest::test_calculate_directional_point()
{
   message += "test_calculate_directional_point\n";
}


void LearningRateAlgorithmTest::test_calculate_fixed_directional_point()
{
   message += "test_calculate_fixed_directional_point\n";
/*
   Vector<size_t> indices;

   NeuralNetwork nn;

   Vector<double> parameters;

   SumSquaredError sse(&nn);

   double loss;
   Vector<double> gradient;

   LearningRateAlgorithm tra(&sse);

   Vector<double> training_direction;
   double training_rate;

   Vector<double> directional_point;

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   loss = sse.calculate_loss();

   gradient = sse.calculate_loss_gradient();

   training_direction = gradient*(-1.0);
   training_rate = 0.001;

   directional_point = tra.calculate_fixed_directional_point(loss, training_direction, training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[1] < loss, LOG);

   assert_true(fabs(directional_point[1] - sse.calculate_training_loss(training_direction, training_rate)) <= numeric_limits<double>::min(), LOG);

   parameters = nn.get_parameters();

   nn.set_parameters(parameters + training_direction*training_rate);

   assert_true(fabs(directional_point[1] - sse.calculate_loss()) <= numeric_limits<double>::min(), LOG);

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   training_direction.set(2, -1.0);
   training_rate = 1.0;

   directional_point = tra.calculate_fixed_directional_point(3.14, training_direction, training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[0] == 1.0, LOG);
   assert_true(directional_point[1] == 0.0, LOG);
*/
}


void LearningRateAlgorithmTest::test_calculate_bracketing_triplet()
{
    message += "test_calculate_bracketing_triplet\n";
/*
    DataSet ds(2, 1, 1);

    ds.randomize_data_normal();

    Vector<size_t> instances_indices(0, 1, ds.get_instances().get_instances_number()-1);

    NeuralNetwork nn(1, 1);

    SumSquaredError sse(&nn, &ds);

    LearningRateAlgorithm tra(&sse);

    double loss = 0.0;
    Vector<double> training_direction;
    double initial_training_rate = 0.0;

    LearningRateAlgorithm::Triplet triplet;

    // Test

    sse.set_regularization_method(LossIndex::L2);

    nn.randomize_parameters_normal();

    loss = sse.calculate_loss();
    training_direction = sse.calculate_loss_gradient()*(-1.0);
    initial_training_rate = 0.01;

    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_training_rate);

    assert_true(triplet.A[0] <= triplet.U[0], LOG);
    assert_true(triplet.U[0] <= triplet.B[0], LOG);
    assert_true(triplet.A[1] >= triplet.U[1], LOG);
    assert_true(triplet.U[1] <= triplet.B[1], LOG);

    // Test

    nn.initialize_parameters(0.0);

    loss = sse.calculate_loss();
    training_direction = sse.calculate_loss_gradient()*(-1.0);
    initial_training_rate = 0.01;

    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_training_rate);

    assert_true(triplet.has_length_zero(), LOG);

    // Test

    nn.initialize_parameters(1.0);

    loss = sse.calculate_loss();
    training_direction = sse.calculate_loss_gradient()*(-1.0);
    initial_training_rate = 0.0;

    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_training_rate);

    assert_true(triplet.has_length_zero(), LOG);

    // Test

    ds.set(1, 1, 1);
    ds.randomize_data_normal();

    instances_indices.set(0, 1, ds.get_instances().get_instances_number()-1);

    nn.set(1, 1);
    nn.randomize_parameters_normal();

    loss = sse.calculate_loss();
    training_direction = sse.calculate_loss_gradient()*(-1.0);
    initial_training_rate = 0.001;

    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_training_rate);

    assert_true(triplet.A[0] <= triplet.U[0], LOG);
    assert_true(triplet.U[0] <= triplet.B[0], LOG);
    assert_true(triplet.A[1] >= triplet.U[1], LOG);
    assert_true(triplet.U[1] <= triplet.B[1], LOG);

    // Test

    ds.set(3, 1, 1);
    ds.randomize_data_normal();

    instances_indices.set(0, 1, ds.get_instances().get_instances_number()-1);

    nn.set(1, 1);
    nn.randomize_parameters_normal();

    loss = sse.calculate_loss();
    training_direction = sse.calculate_loss_gradient()*(-1.0);
    initial_training_rate = 0.001;

    triplet = tra.calculate_bracketing_triplet(loss, training_direction, initial_training_rate);

    assert_true(triplet.A[0] <= triplet.U[0], LOG);
    assert_true(triplet.U[0] <= triplet.B[0], LOG);
    assert_true(triplet.A[1] >= triplet.U[1], LOG);
    assert_true(triplet.U[1] <= triplet.B[1], LOG);
*/
}


void LearningRateAlgorithmTest::test_calculate_golden_section_directional_point()
{
   message += "test_calculate_golden_section_directional_point\n";
/*
   DataSet ds(1, 1, 1);
   Vector<size_t> indices(1,1,ds.get_instances().get_instances_number()-1);

   NeuralNetwork nn(1, 1);

   SumSquaredError sse(&nn);

   LearningRateAlgorithm tra(&sse);

   nn.initialize_parameters(1.0);

   double loss = sse.calculate_loss();
   Vector<double> gradient = sse.calculate_loss_gradient();

   Vector<double> training_direction = gradient*(-1.0);
   double initial_training_rate = 0.001;

   double loss_tolerance = 1.0e-6;
   tra.set_loss_tolerance(loss_tolerance);
  
   Vector<double> directional_point
   = tra.calculate_golden_section_directional_point(loss, training_direction, initial_training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[0] >= 0.0, LOG);
   assert_true(directional_point[1] < loss, LOG);
*/
}


void LearningRateAlgorithmTest::test_calculate_Brent_method_directional_point()
{
   message += "test_calculate_Brent_method_directional_point\n";
/*
   DataSet ds(1, 1, 1);
   Vector<size_t> indices(1,1,ds.get_instances().get_instances_number()-1);

   NeuralNetwork nn(1, 1);
   SumSquaredError sse(&nn);

   LearningRateAlgorithm tra(&sse);

   nn.initialize_parameters(1.0);

   double loss = sse.calculate_loss();
   Vector<double> gradient = sse.calculate_loss_gradient();

   Vector<double> training_direction = gradient*(-1.0);
   double initial_training_rate = 0.001;

   double loss_tolerance = 1.0e-6;
   tra.set_loss_tolerance(loss_tolerance);

   Vector<double> directional_point
   = tra.calculate_Brent_method_directional_point(loss, training_direction, initial_training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[0] >= 0.0, LOG);
   assert_true(directional_point[1] < loss, LOG);
*/
}


void LearningRateAlgorithmTest::test_to_XML()
{
   message += "test_to_XML\n";

   LearningRateAlgorithm  tra;

   tinyxml2::XMLDocument* document = tra.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;
}


void LearningRateAlgorithmTest::run_test_case()
{
   message += "Running training rate algorithm test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_loss_index_pointer();

   // Training operators

   test_get_training_rate_method();
   test_get_training_rate_method_name();

   // Training parameters

   test_get_loss_tolerance();

   test_get_warning_training_rate();
   test_get_error_training_rate();

   // Utilities
   
   test_get_display();

   // Set methods

   test_set();
   test_set_default();   

   test_set_loss_index_pointer();

   // Training operators

   test_set_training_rate_method();

   // Training parameters

   test_set_loss_tolerance();

   test_set_warning_training_rate();

   test_set_error_training_rate();

   // Utilities

   test_set_display();

   // Training methods

   test_calculate_bracketing_triplet();
//   test_calculate_fixed_directional_point();
//   test_calculate_golden_section_directional_point();
//   test_calculate_Brent_method_directional_point();
//   test_calculate_directional_point();

   // Serialization methods

   test_to_XML();

   message += "End of training rate algorithm test case.\n";
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
