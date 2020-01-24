//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S   H E A D E R             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TESTINGANALYSIS_H
#define TESTINGANALYSIS_H

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

// OpenNN includes

#include "config.h"
#include "metrics.h"
#include "correlations.h"
#include "data_set.h"
#include "neural_network.h"

namespace OpenNN
{

/// This class contains tools for testing neural networks in different learning tasks.

///
/// In particular, it can be used for testing function regression, classification
/// or time series prediction problems.

class TestingAnalysis
{

public:  

   // Constructors

   explicit TestingAnalysis();

   explicit TestingAnalysis(NeuralNetwork*);

   explicit TestingAnalysis(DataSet*);

   explicit TestingAnalysis(NeuralNetwork*, DataSet*);

   explicit TestingAnalysis(const tinyxml2::XMLDocument&);

   explicit TestingAnalysis(const string&);

    // Destructor

   virtual ~TestingAnalysis();

    /// Structure with the results from a linear regression analysis.

    struct LinearRegressionAnalysis
    {
       /// Target data from data set and output data from neural network.

       type correlation = 0.0;

       type intercept = 0.0;

       type slope = 0.0;

       Tensor<type, 1> targets;
       Tensor<type, 1> outputs;

       void save(const string&) const;
    };


    /// Structure with the results from a roc curve analysis.

    struct RocAnalysisResults
    {
        /// Matrix containing the data of a ROC curve.

        Tensor<type, 2> roc_curve;

        /// Area under a ROC curve.

        type area_under_curve;

        /// Confidence limit

        type confidence_limit;

        /// Optimal threshold of a ROC curve.

        type optimal_threshold;
    };


    /// Structure with the results from Kolmogorov-Smirnov analysis.

    struct KolmogorovSmirnovResults
    {
        /// Matrix containing the data of a positive cumulative gain

        Tensor<type, 2> positive_cumulative_gain;

        /// Matrix containing the data of a negative cumulative gain.

        Tensor<type, 2> negative_cumulative_gain;

        /// Maximum gain of the cumulative gain analysis

        Tensor<type, 1> maximum_gain;
    };


    /// Structure with the binary classification rates

    struct BinaryClassifcationRates
    {
        /// Vector with the indices of the instances which are true positive.

        Tensor<Index, 1> true_positives_indices;

        /// Vector with the indices of the instances which are false positive.

        Tensor<Index, 1> false_positives_indices;

        /// Vector with the indices of the instances which are false negative.

        Tensor<Index, 1> false_negatives_indices;

        /// Vector with the indices of the instances which are true negative.

        Tensor<Index, 1> true_negatives_indices;
    };

   // Get methods

   NeuralNetwork* get_neural_network_pointer() const;
   DataSet* get_data_set_pointer() const;

   const bool& get_display() const;

   // Set methods

   void set_neural_network_pointer(NeuralNetwork*);
   void set_data_set_pointer(DataSet*);

   void set_display(const bool&);

   void set_default();

   // Checking methods

   void check() const;

   // Error data methods

   Tensor<Tensor<type, 2>, 1> calculate_error_data() const;
   Tensor<Tensor<type, 1>, 1> calculate_percentage_error_data() const;

   Tensor<Descriptives, 1> calculate_absolute_errors_statistics() const;
   Tensor<Descriptives, 1> calculate_absolute_errors_statistics(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<Descriptives, 1> calculate_percentage_errors_statistics() const;
   Tensor<Descriptives, 1> calculate_percentage_errors_statistics(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<Tensor<Descriptives, 1>, 1> calculate_error_data_statistics() const;
   void print_error_data_statistics() const;

   Tensor<Tensor<type, 2>, 1> calculate_error_data_statistics_matrices() const;

   Tensor<Histogram, 1> calculate_error_data_histograms(const Index& = 10) const;

   Tensor<Tensor<Index, 1>, 1> calculate_maximal_errors(const Index& = 10) const;

   Tensor<type, 2> calculate_errors() const;
   Tensor<type, 2> calculate_binary_classification_errors() const;
   Tensor<type, 2> calculate_multiple_classification_errors() const;

   Tensor<type, 1> calculate_training_errors() const;
   Tensor<type, 1> calculate_binary_classification_training_errors() const;
   Tensor<type, 1> calculate_multiple_classification_training_errors() const;

   Tensor<type, 1> calculate_selection_errors() const;
   Tensor<type, 1> calculate_binary_classification_selection_errors() const;
   Tensor<type, 1> calculate_multiple_classification_selection_errors() const;

   Tensor<type, 1> calculate_testing_errors() const;
   Tensor<type, 1> calculate_binary_classification_testing_errors() const;
   Tensor<type, 1> calculate_multiple_classification_testing_errors() const;

   type calculate_testing_normalized_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   type calculate_testing_cross_entropy_error(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   type calculate_testing_weighted_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 1>& = Tensor<type, 1>()) const;

   // Linear regression analysis methods

   Tensor<RegressionResults, 1> linear_regression() const;
   Tensor<RegressionResults, 1> linear_regression(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   void print_linear_regression_correlations() const;
   Tensor<type, 1> get_linear_regression_correlations_std() const;

   Tensor<LinearRegressionAnalysis, 1> perform_linear_regression_analysis() const;
   void perform_linear_regression_analysis_void() const;

   // Binary classifcation methods

   Tensor<type, 1> calculate_binary_classification_tests() const;

   type calculate_logloss() const;

   // Confusion methods

   Tensor<Index, 2> calculate_confusion_binary_classification(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&) const;
   Tensor<Index, 2> calculate_confusion_multiple_classification(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<Index, 1> calculate_positives_negatives_rate(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<Index, 2> calculate_confusion(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   Tensor<Index, 2> calculate_confusion() const;

   // ROC curve

   RocAnalysisResults perform_roc_analysis() const;

   type calculate_Wilcoxon_parameter(const type&, const type&) const;

   Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>& ,const Tensor<type, 2>&) const;
   type calculate_area_under_curve(const Tensor<type, 2>& ,const Tensor<type, 2>&) const;
   type calculate_area_under_curve_confidence_limit(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   type calculate_area_under_curve_confidence_limit(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&) const;
   type calculate_optimal_threshold(const Tensor<type, 2>& ,const Tensor<type, 2>&) const;
   type calculate_optimal_threshold(const Tensor<type, 2>& ,const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Lift Chart

   Tensor<type, 2> perform_cumulative_gain_analysis() const;
   Tensor<type, 2> calculate_cumulative_gain(const Tensor<type, 2>& ,const Tensor<type, 2>&) const;
   Tensor<type, 2> calculate_negative_cumulative_gain(const Tensor<type, 2>&, const Tensor<type, 2>&)const;

   Tensor<type, 2> perform_lift_chart_analysis() const;
   Tensor<type, 2> calculate_lift_chart(const Tensor<type, 2>&) const;

   KolmogorovSmirnovResults perform_Kolmogorov_Smirnov_analysis() const;
   Tensor<type, 1> calculate_maximum_gain(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Calibration plot

   Tensor<type, 2> perform_calibration_plot_analysis() const;

   Tensor<type, 2> calculate_calibration_plot(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Output histogram

   Tensor<Histogram, 1> calculate_output_histogram(const Tensor<type, 2>&, const Index& = 10) const;

   // Binary classification rates

   BinaryClassifcationRates calculate_binary_classification_rates() const;

   Tensor<Index, 1> calculate_true_positive_instances(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;
   Tensor<Index, 1> calculate_false_positive_instances(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;
   Tensor<Index, 1> calculate_false_negative_instances(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;
   Tensor<Index, 1> calculate_true_negative_instances(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;

   // Multiple classification rates

   Tensor<Tensor<Index, 1>, 2> calculate_multiple_classification_rates() const;

   Tensor<Tensor<Index, 1>, 2> calculate_multiple_classification_rates(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&) const;

   // Forecasting methods

   Tensor<Tensor<type, 1>, 1> calculate_error_autocorrelation(const Index& = 10) const;

   Tensor<Tensor<type, 1>, 1> calculate_inputs_errors_cross_correlation(const Index& = 10) const;

   // Serialization methods

   string object_to_string() const;

   void print() const;

   virtual tinyxml2::XMLDocument* to_XML() const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void save(const string&) const;
   void load(const string&);


private: 

   /// Pointer to the neural network object to be tested. 

   NeuralNetwork* neural_network_pointer = nullptr;

   /// Pointer to a data set object.

   DataSet* data_set_pointer = nullptr;

   /// Display messages to screen.
   
   bool display;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
