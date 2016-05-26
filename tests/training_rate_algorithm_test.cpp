/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   R A T E   A L G O R I T H M   T E S T   C L A S S                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "training_rate_algorithm_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR 

TrainingRateAlgorithmTest::TrainingRateAlgorithmTest(void) : UnitTesting() 
{
}


// DESTRUCTOR

TrainingRateAlgorithmTest::~TrainingRateAlgorithmTest(void)
{
}


// METHODS

void TrainingRateAlgorithmTest::test_constructor(void)
{
   message += "test_constructor\n"; 

   PerformanceFunctional pf;

   TrainingRateAlgorithm tra1(&pf);

   assert_true(tra1.has_performance_functional() == true, LOG);

   TrainingRateAlgorithm tra2;

   assert_true(tra2.has_performance_functional() == false, LOG);
}


void TrainingRateAlgorithmTest::test_destructor(void)
{
   message += "test_destructor\n"; 
}


void TrainingRateAlgorithmTest::test_get_performance_functional_pointer(void)
{
   message += "test_get_performance_functional_pointer\n"; 
   
   PerformanceFunctional pf;

   TrainingRateAlgorithm tra(&pf);

   PerformanceFunctional* pfp = tra.get_performance_functional_pointer();

   assert_true(pfp != NULL, LOG);
}


void TrainingRateAlgorithmTest::test_get_training_rate_method(void)
{
   message += "test_get_training_rate_method\n"; 

   TrainingRateAlgorithm tra;

   tra.set_training_rate_method(TrainingRateAlgorithm::Fixed);
   assert_true(tra.get_training_rate_method() == TrainingRateAlgorithm::Fixed, LOG);

   tra.set_training_rate_method(TrainingRateAlgorithm::GoldenSection);
   assert_true(tra.get_training_rate_method() == TrainingRateAlgorithm::GoldenSection, LOG);

   tra.set_training_rate_method(TrainingRateAlgorithm::BrentMethod);
   assert_true(tra.get_training_rate_method() == TrainingRateAlgorithm::BrentMethod, LOG);
}


void TrainingRateAlgorithmTest::test_get_training_rate_method_name(void)
{
   message += "test_get_training_rate_method_name\n"; 
}


void TrainingRateAlgorithmTest::test_get_warning_training_rate(void)
{
   message += "test_get_warning_training_rate\n";

   TrainingRateAlgorithm tra;

   tra.set_warning_training_rate(0.0);

   assert_true(tra.get_warning_training_rate() == 0.0, LOG);
}


void TrainingRateAlgorithmTest::test_get_error_training_rate(void)
{
   message += "test_get_error_training_rate\n";

   TrainingRateAlgorithm tra;

   tra.set_error_training_rate(0.0);

   assert_true(tra.get_error_training_rate() == 0.0, LOG);
}


void TrainingRateAlgorithmTest::test_get_display(void)
{
   message += "test_get_warning_gradient_norm\n"; 

   TrainingRateAlgorithm tra;

   tra.set_display(false);

   assert_true(tra.get_display() == false, LOG);
}


void TrainingRateAlgorithmTest::test_get_bracketing_factor(void)   
{
   message += "test_get_bracketing_factor\n"; 

   TrainingRateAlgorithm tra;

   tra.set_bracketing_factor(0.0);

   assert_true(tra.get_bracketing_factor() == 0.0, LOG);
}


void TrainingRateAlgorithmTest::test_get_training_rate_tolerance(void)
{
   message += "test_get_training_rate_tolerance\n"; 
}


void TrainingRateAlgorithmTest::test_set(void)
{
   message += "test_set\n"; 
}


void TrainingRateAlgorithmTest::test_set_default(void)
{
   message += "test_set_default\n"; 
}


void TrainingRateAlgorithmTest::test_set_performance_functional_pointer(void)
{
   message += "test_set_performance_functional_pointer\n"; 
}


void TrainingRateAlgorithmTest::test_set_display(void)
{
   message += "test_set_display\n"; 
}


void TrainingRateAlgorithmTest::test_set_training_rate_method(void)
{
   message += "test_set_training_rate_method\n"; 
}


void TrainingRateAlgorithmTest::test_set_training_rate_tolerance(void)
{
   message += "test_set_training_rate_tolerance\n"; 
}


void TrainingRateAlgorithmTest::test_set_warning_training_rate(void)
{
   message += "test_set_warning_training_rate\n"; 
}


void TrainingRateAlgorithmTest::test_set_error_training_rate(void)
{
   message += "test_set_error_training_rate\n"; 
}


void TrainingRateAlgorithmTest::test_calculate_directional_point(void)
{
   message += "test_calculate_directional_point\n";

//   NeuralNetwork nn;

//   PerformanceFunctional pf(&nn);

//   TrainingRateAlgorithm tra(&pf);

//   tra.calculate_directional_point();
}


void TrainingRateAlgorithmTest::test_calculate_fixed_directional_point(void)
{
   message += "test_calculate_fixed_directional_point\n";

   NeuralNetwork nn;

   Vector<double> parameters;

   PerformanceFunctional pf(&nn);

   pf.destruct_all_terms();
   pf.set_regularization_type(PerformanceFunctional::NEURAL_PARAMETERS_NORM);

   double performance;
   Vector<double> gradient;

   TrainingRateAlgorithm tra(&pf);

   Vector<double> training_direction;
   double training_rate;

   Vector<double> directional_point;

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   performance = pf.calculate_performance();

   gradient = pf.calculate_gradient();

   training_direction = gradient*(-1.0);
   training_rate = 0.001;

   directional_point = tra.calculate_fixed_directional_point(performance, training_direction, training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[1] < performance, LOG);

   assert_true(directional_point[1] == pf.calculate_performance(training_direction, training_rate), LOG);

   parameters = nn.arrange_parameters();

   nn.set_parameters(parameters + training_direction*training_rate);

   assert_true(directional_point[1] == pf.calculate_performance(), LOG);

   // Test

   nn.set(1, 1);

   nn.initialize_parameters(1.0);

   training_direction.set(2, -1.0);
   training_rate = 1.0;

   directional_point = tra.calculate_fixed_directional_point(3.14, training_direction, training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[0] == 1.0, LOG);
   assert_true(directional_point[1] == 0.0, LOG);
}


void TrainingRateAlgorithmTest::test_calculate_bracketing_triplet(void)
{
    message += "test_calculate_bracketing_triplet\n";

    DataSet ds(2, 1, 1);

    NeuralNetwork nn(1, 1);

    PerformanceFunctional pf(&nn, &ds);

    TrainingRateAlgorithm tra(&pf);

    double performance;
    Vector<double> training_direction;
    double initial_training_rate;

    TrainingRateAlgorithm::Triplet triplet;

    // Test

    pf.destruct_all_terms();
    pf.set_regularization_type(PerformanceFunctional::NEURAL_PARAMETERS_NORM);

    nn.randomize_parameters_normal();

    performance = pf.calculate_performance();
    training_direction = pf.calculate_gradient()*(-1.0);
    initial_training_rate = 0.01;

    triplet = tra.calculate_bracketing_triplet(performance, training_direction, initial_training_rate);

    assert_true(triplet.A[0] <= triplet.U[0], LOG);
    assert_true(triplet.U[0] <= triplet.B[0], LOG);
    assert_true(triplet.A[1] >= triplet.U[1], LOG);
    assert_true(triplet.U[1] <= triplet.B[1], LOG);

    // Test

    nn.initialize_parameters(0.0);

    performance = pf.calculate_performance();
    training_direction = pf.calculate_gradient()*(-1.0);
    initial_training_rate = 0.01;

    triplet = tra.calculate_bracketing_triplet(performance, training_direction, initial_training_rate);

    assert_true(triplet.has_length_zero(), LOG);

    // Test

    nn.initialize_parameters(1.0);

    performance = pf.calculate_performance();
    training_direction = pf.calculate_gradient()*(-1.0);
    initial_training_rate = 0.0;

    triplet = tra.calculate_bracketing_triplet(performance, training_direction, initial_training_rate);

    assert_true(triplet.has_length_zero(), LOG);

    // Test

    ds.set(1, 1, 1);
    ds.randomize_data_normal();

    nn.set(1, 1);
    nn.randomize_parameters_normal();

    pf.destruct_all_terms();
    pf.set_error_type(PerformanceFunctional::SUM_SQUARED_ERROR);

    performance = pf.calculate_performance();
    training_direction = pf.calculate_gradient()*(-1.0);
    initial_training_rate = 0.001;

    triplet = tra.calculate_bracketing_triplet(performance, training_direction, initial_training_rate);

    assert_true(triplet.A[0] <= triplet.U[0], LOG);
    assert_true(triplet.U[0] <= triplet.B[0], LOG);
    assert_true(triplet.A[1] >= triplet.U[1], LOG);
    assert_true(triplet.U[1] <= triplet.B[1], LOG);

    // Test

    ds.set(3, 1, 1);
    ds.randomize_data_normal();

    nn.set(1, 1);
    nn.randomize_parameters_normal();

    pf.destruct_all_terms();
    pf.set_error_type(PerformanceFunctional::NORMALIZED_SQUARED_ERROR);

    performance = pf.calculate_performance();
    training_direction = pf.calculate_gradient()*(-1.0);
    initial_training_rate = 0.001;

    triplet = tra.calculate_bracketing_triplet(performance, training_direction, initial_training_rate);

    assert_true(triplet.A[0] <= triplet.U[0], LOG);
    assert_true(triplet.U[0] <= triplet.B[0], LOG);
    assert_true(triplet.A[1] >= triplet.U[1], LOG);
    assert_true(triplet.U[1] <= triplet.B[1], LOG);
}


void TrainingRateAlgorithmTest::test_calculate_golden_section_directional_point(void)
{
   message += "test_calculate_golden_section_directional_point\n";

   NeuralNetwork nn(1, 1);

   PerformanceFunctional pf(&nn);

   pf.destruct_all_terms();
   pf.set_regularization_type(PerformanceFunctional::NEURAL_PARAMETERS_NORM);

   TrainingRateAlgorithm tra(&pf);

   nn.initialize_parameters(1.0);

   double performance = pf.calculate_performance();
   Vector<double> gradient = pf.calculate_gradient();

   Vector<double> training_direction = gradient*(-1.0);
   double initial_training_rate = 0.001;

   double training_rate_tolerance = 1.0e-6;
   tra.set_training_rate_tolerance(training_rate_tolerance);
  
   Vector<double> directional_point
   = tra.calculate_golden_section_directional_point(performance, training_direction, initial_training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[0] >= 0.0, LOG);
   assert_true(directional_point[1] < performance, LOG);
}


void TrainingRateAlgorithmTest::test_calculate_Brent_method_directional_point(void)
{
   message += "test_calculate_Brent_method_directional_point\n";

   NeuralNetwork nn(1, 1);
   PerformanceFunctional pf(&nn);

   pf.destruct_all_terms();
   pf.set_regularization_type(PerformanceFunctional::NEURAL_PARAMETERS_NORM);

   TrainingRateAlgorithm tra(&pf);

   nn.initialize_parameters(1.0);

   double performance = pf.calculate_performance();
   Vector<double> gradient = pf.calculate_gradient();

   Vector<double> training_direction = gradient*(-1.0);
   double initial_training_rate = 0.001;

   double training_rate_tolerance = 1.0e-6;
   tra.set_training_rate_tolerance(training_rate_tolerance);

   Vector<double> directional_point
   = tra.calculate_Brent_method_directional_point(performance, training_direction, initial_training_rate);

   assert_true(directional_point.size() == 2, LOG);
   assert_true(directional_point[0] >= 0.0, LOG);
   assert_true(directional_point[1] < performance, LOG);
}


void TrainingRateAlgorithmTest::test_to_XML(void)
{
   message += "test_to_XML\n";

   TrainingRateAlgorithm  tra;

   tinyxml2::XMLDocument* document = tra.to_XML();

   assert_true(document != NULL, LOG);

   delete document;
}


void TrainingRateAlgorithmTest::run_test_case(void)
{
   message += "Running training rate algorithm test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_performance_functional_pointer();

   // Training operators

   test_get_training_rate_method();
   test_get_training_rate_method_name();

   // Training parameters

   test_get_bracketing_factor();   
   test_get_training_rate_tolerance();

   test_get_warning_training_rate();
   test_get_error_training_rate();

   // Utilities
   
   test_get_display();

   // Set methods

   test_set();
   test_set_default();   

   test_set_performance_functional_pointer();

   // Training operators

   test_set_training_rate_method();

   // Training parameters

   test_set_training_rate_tolerance();

   test_set_warning_training_rate();

   test_set_error_training_rate();

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

   message += "End of training rate algorithm test case.\n";
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
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
