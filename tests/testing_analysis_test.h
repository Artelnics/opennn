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

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/testing_analysis.h"

namespace opennn
{


class TestingAnalysisTest : public UnitTesting 
{

public: 

    explicit TestingAnalysisTest();

    virtual ~TestingAnalysisTest();

    // Constructor and destructor

    void test_constructor();

    void test_destructor();

    // Error data

    void test_calculate_error_data();
    void test_calculate_percentage_error_data();
    void test_calculate_forecasting_error_data();
    void test_calculate_absolute_errors_descriptives();
    void test_calculate_percentage_errors_descriptives();
    void test_calculate_error_data_descriptives();
    void test_calculate_error_data_histograms();
    void test_calculate_maximal_errors();

    // Linear regression parameters

    void test_linear_regression();
    void test_save();
    void test_perform_linear_regression();

    // Binary classification test

    void test_calculate_binary_classification_test();

    // Confusion matrix

    void test_calculate_confusion();

    // ROC curve

    void test_calculate_roc_curve();
    void test_calculate_area_under_curve();
    void test_calculate_optimal_threshold ();

    // Lift chart

    void test_calculate_cumulative_gain();
    void test_calculate_lift_chart();

    // Calibration plot

    void test_calculate_calibration_plot();

    // Binary classificaton rates

    void test_calculate_true_positive_samples();
    void test_calculate_false_positive_samples();
    void test_calculate_false_negative_samples();
    void test_calculate_true_negative_samples();

    // Multiple classification rates

    void test_calculate_multiple_classification_rates();

    // Unit testing

    void run_test_case();

private:

    Index samples_number = 0;
    Index inputs_number = 0;
    Index targets_number = 0;
    Index neurons_number = 0;

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    DataSet data_set;

    NeuralNetwork neural_network;

    TestingAnalysis testing_analysis;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
