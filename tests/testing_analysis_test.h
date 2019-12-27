//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   T E S T   C L A S S   H E A D E R   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TESTINGANALYSISTEST_H
#define TESTINGANALYSISTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class TestingAnalysisTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:  

   explicit TestingAnalysisTest();   

   virtual ~TestingAnalysisTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_neural_network_pointer();
   void test_get_data_set_pointer();
   
   void test_get_display();

   // Set methods

   void test_set_neural_network_pointer();
   void test_set_data_set_pointer();

   void test_set_display();

   // Error data methods

   void test_calculate_error_data();
   void test_calculate_percentage_error_data();
   void test_calculate_forecasting_error_data();

   void test_calculate_absolute_errors_statistics();

   void test_calculate_percentage_errors_statistics();

   void test_calculate_error_data_statistics();
   void test_print_error_data_statistics();
   void test_calculate_error_data_statistics_matrices();

   void test_calculate_error_data_histograms();

   void test_calculate_maximal_errors();

   // Linear regression parameters methods

   void test_linear_regression();
   void test_print_linear_regression_correlation();
   void test_get_linear_regression_correlations_std();
   void test_save_linear_regression();

   void test_perform_linear_regression();

   void test_print_linear_regression_analysis();
   void test_save_linear_regression_analysis();

   // Binary classification test methods

   void test_calculate_binary_classification_test();
   void test_print_binary_classification_test();

   // Confusion matrix methods

   void test_calculate_confusion();
   void test_print_confusion();

   // ROC curve methods

   void test_calculate_Wilcoxon_parameter();

   void test_calculate_roc_curve();
   void test_calculate_area_under_curve();
   void test_calculate_optimal_threshold ();

   // Lift chart methods

   void test_calculate_cumulative_gain();
   void test_calculate_lift_chart();

   // Calibration plot

   void test_calculate_calibration_plot();

   // Binary classificaton rates

   void test_calculate_true_positive_instances();
   void test_calculate_false_positive_instances();
   void test_calculate_false_negative_instances();
   void test_calculate_true_negative_instances();

   // Multiple classification rates

   void test_calculate_multiple_classification_rates();

   // Unit testing methods

   void run_test_case();

};


#endif

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
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
