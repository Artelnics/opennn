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

#include "vector.h"
#include "matrix.h"
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

       double correlation = 0.0;

       double intercept = 0.0;

       double slope = 0.0;

       Vector<double> targets;
       Vector<double> outputs;

       void save(const string&) const;
    };


    /// Structure with the results from a roc curve analysis.

    struct RocAnalysisResults
    {
        /// Matrix containing the data of a ROC curve.

        Matrix<double> roc_curve;

        /// Area under a ROC curve.

        double area_under_curve;

        /// Confidence limit

        double confidence_limit;

        /// Optimal threshold of a ROC curve.

        double optimal_threshold;
    };


    /// Structure with the results from Kolmogorov-Smirnov analysis.

    struct KolmogorovSmirnovResults
    {
        /// Matrix containing the data of a positive cumulative gain

        Matrix<double> positive_cumulative_gain;

        /// Matrix containing the data of a negative cumulative gain.

        Matrix<double> negative_cumulative_gain;

        /// Maximum gain of the cumulative gain analysis

        Vector<double> maximum_gain;
    };


    /// Structure with the binary classification rates

    struct BinaryClassifcationRates
    {
        /// Vector with the indices of the instances which are true positive.

        Vector<size_t> true_positives_indices;

        /// Vector with the indices of the instances which are false positive.

        Vector<size_t> false_positives_indices;

        /// Vector with the indices of the instances which are false negative.

        Vector<size_t> false_negatives_indices;

        /// Vector with the indices of the instances which are true negative.

        Vector<size_t> true_negatives_indices;
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

   Vector<Matrix<double>> calculate_error_data() const;
   Vector<Vector<double>> calculate_percentage_error_data() const;

   Vector<Descriptives> calculate_absolute_errors_statistics() const;
   Vector<Descriptives> calculate_absolute_errors_statistics(const Tensor<double>&, const Tensor<double>&) const;

   Vector<Descriptives> calculate_percentage_errors_statistics() const;
   Vector<Descriptives> calculate_percentage_errors_statistics(const Tensor<double>&, const Tensor<double>&) const;

   Vector<Vector<Descriptives>> calculate_error_data_statistics() const;
   void print_error_data_statistics() const;

   Vector<Matrix<double>> calculate_error_data_statistics_matrices() const;

   Vector<Histogram> calculate_error_data_histograms(const size_t& = 10) const;

   Vector<Vector<size_t>> calculate_maximal_errors(const size_t& = 10) const;

   Matrix<double> calculate_errors() const;
   Matrix<double> calculate_binary_classification_errors() const;
   Matrix<double> calculate_multiple_classification_errors() const;

   Vector<double> calculate_training_errors() const;
   Vector<double> calculate_binary_classification_training_errors() const;
   Vector<double> calculate_multiple_classification_training_errors() const;

   Vector<double> calculate_selection_errors() const;
   Vector<double> calculate_binary_classification_selection_errors() const;
   Vector<double> calculate_multiple_classification_selection_errors() const;

   Vector<double> calculate_testing_errors() const;
   Vector<double> calculate_binary_classification_testing_errors() const;
   Vector<double> calculate_multiple_classification_testing_errors() const;

   double calculate_testing_normalized_squared_error(const Tensor<double>&, const Tensor<double>&) const;
   double calculate_testing_cross_entropy_error(const Tensor<double>&, const Tensor<double>&) const;
   double calculate_testing_weighted_squared_error(const Tensor<double>&, const Tensor<double>&, const Vector<double>& = Vector<double>()) const;

   // Linear regression analysis methods

   Vector<RegressionResults> linear_regression() const;
   Vector<RegressionResults> linear_regression(const Tensor<double>&, const Tensor<double>&) const;

   void print_linear_regression_correlations() const;
   vector<double> get_linear_regression_correlations_std() const;

   Vector<LinearRegressionAnalysis> perform_linear_regression_analysis() const;
   void perform_linear_regression_analysis_void() const;

   // Binary classifcation methods

   Vector<double> calculate_binary_classification_tests() const;

   double calculate_logloss() const;

   // Confusion methods

   Matrix<size_t> calculate_confusion_binary_classification(const Tensor<double>&, const Tensor<double>&, const double&) const;
   Matrix<size_t> calculate_confusion_multiple_classification(const Tensor<double>&, const Tensor<double>&) const;

   Vector<size_t> calculate_positives_negatives_rate(const Tensor<double>&, const Tensor<double>&) const;

   Matrix<size_t> calculate_confusion(const Matrix<double>&, const Matrix<double>&) const;
   Matrix<size_t> calculate_confusion() const;

   // ROC curve

   RocAnalysisResults perform_roc_analysis() const;

   double calculate_Wilcoxon_parameter(const double&, const double&) const;

   Matrix<double> calculate_roc_curve(const Tensor<double>& ,const Tensor<double>&) const;
   double calculate_area_under_curve(const Tensor<double>& ,const Tensor<double>&) const;
   double calculate_area_under_curve_confidence_limit(const Tensor<double>&, const Tensor<double>&) const;
   double calculate_area_under_curve_confidence_limit(const Tensor<double>&, const Tensor<double>&, const double&) const;
   double calculate_optimal_threshold(const Tensor<double>& ,const Tensor<double>&) const;
   double calculate_optimal_threshold(const Tensor<double>& ,const Tensor<double>&, const Matrix<double>&) const;

   // Lift Chart

   Matrix<double> perform_cumulative_gain_analysis() const;
   Matrix<double> calculate_cumulative_gain(const Tensor<double>& ,const Tensor<double>&) const;
   Matrix<double> calculate_negative_cumulative_gain(const Tensor<double>&, const Tensor<double>&)const;

   Matrix<double> perform_lift_chart_analysis() const;
   Matrix<double> calculate_lift_chart(const Matrix<double>&) const;

   KolmogorovSmirnovResults perform_Kolmogorov_Smirnov_analysis() const;
   Vector<double> calculate_maximum_gain(const Matrix<double>&, const Matrix<double>&) const;

   // Calibration plot

   Matrix<double> perform_calibration_plot_analysis() const;

   Matrix<double> calculate_calibration_plot(const Tensor<double>&, const Tensor<double>&) const;

   // Output histogram

   Vector<Histogram> calculate_output_histogram(const Tensor<double>&, const size_t& = 10) const;

   // Binary classification rates

   BinaryClassifcationRates calculate_binary_classification_rates() const;

   Vector<size_t> calculate_true_positive_instances(const Tensor<double>&, const Tensor<double>&, const Vector<size_t>&, const double&) const;
   Vector<size_t> calculate_false_positive_instances(const Tensor<double>&, const Tensor<double>&, const Vector<size_t>&, const double& ) const;
   Vector<size_t> calculate_false_negative_instances(const Tensor<double>&, const Tensor<double>&, const Vector<size_t>&, const double& ) const;
   Vector<size_t> calculate_true_negative_instances(const Tensor<double>&, const Tensor<double>&, const Vector<size_t>&, const double& ) const;

   // Multiple classification rates

   Matrix<Vector<size_t>> calculate_multiple_classification_rates() const;

   Matrix<Vector<size_t>> calculate_multiple_classification_rates(const Tensor<double>&, const Tensor<double>&, const Vector<size_t>&) const;

   // Forecasting methods

   Vector<Vector<double>> calculate_error_autocorrelation(const size_t& = 10) const;

   Vector<Vector<double>> calculate_inputs_errors_cross_correlation(const size_t& = 10) const;

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
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
