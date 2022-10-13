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

#include <numeric>

// OpenNN includes

#include "config.h"

#include "correlations.h"
#include "data_set.h"
#include "neural_network.h"

//Eigen includes

namespace opennn
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

   explicit TestingAnalysis(NeuralNetwork*, DataSet*);

    // Destructor

   virtual ~TestingAnalysis();

    /// Structure with the results from a goodness-of-fit analysis.

    struct GoodnessOfFitAnalysis
    {
       /// Target data from data set and output data from neural network.

       type determination = type(0);

       Tensor<type, 1> targets;
       Tensor<type, 1> outputs;

       /// @todo

       void save(const string&) const
       {
       }

       void print() const
       {
           cout << "Goodness-of-fit analysis" << endl;
           cout << "Determination: " << determination << endl;
       }

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

    struct BinaryClassificationRates
    {
        /// Vector with the indices of the samples which are true positive.

        Tensor<Index, 1> true_positives_indices;

        /// Vector with the indices of the samples which are false positive.

        Tensor<Index, 1> false_positives_indices;

        /// Vector with the indices of the samples which are false negative.

        Tensor<Index, 1> false_negatives_indices;

        /// Vector with the indices of the samples which are true negative.

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

   void set_threads_number(const int&);

   // Checking methods

   void check() const;

   // Error data methods

   Tensor<type, 3> calculate_error_data() const;
   Tensor<type, 2> calculate_percentage_error_data() const;

   Tensor<Descriptives, 1> calculate_absolute_errors_descriptives() const;
   Tensor<Descriptives, 1> calculate_absolute_errors_descriptives(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<Descriptives, 1> calculate_percentage_errors_descriptives() const;
   Tensor<Descriptives, 1> calculate_percentage_errors_descriptives(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<Tensor<Descriptives, 1>, 1> calculate_error_data_descriptives() const;
   void print_error_data_descriptives() const;

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

   type calculate_normalized_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   type calculate_cross_entropy_error(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   type calculate_weighted_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 1>& = Tensor<type, 1>()) const;
   type calculate_Minkowski_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const type = type(1.5)) const;

   type calculate_determination_coefficient(const Tensor<type,1>&, const Tensor<type,1>&) const;

   // Goodness-of-fit analysis methods

   Tensor<Correlation, 1> linear_correlation() const;
   Tensor<Correlation, 1> linear_correlation(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   void print_linear_regression_correlations() const;

   Tensor<GoodnessOfFitAnalysis, 1> perform_goodness_of_fit_analysis() const;
   void print_goodness_of_fit_analysis() const;

   // Binary classifcation methods

   Tensor<type, 1> calculate_binary_classification_tests() const;

   void print_binary_classification_tests() const;

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

   Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

//   type calculate_area_under_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   type calculate_area_under_curve(const Tensor<type, 2>&) const;
   type calculate_area_under_curve_confidence_limit(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
//   type calculate_area_under_curve_confidence_limit(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&) const;
//   type calculate_optimal_threshold(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   type calculate_optimal_threshold(const Tensor<type, 2>&) const;

   // Lift Chart

   Tensor<type, 2> perform_cumulative_gain_analysis() const;
   Tensor<type, 2> calculate_cumulative_gain(const Tensor<type, 2>&, const Tensor<type, 2>&) const;
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

   BinaryClassificationRates calculate_binary_classification_rates() const;

   Tensor<Index, 1> calculate_true_positive_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;
   Tensor<Index, 1> calculate_false_positive_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;
   Tensor<Index, 1> calculate_false_negative_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;
   Tensor<Index, 1> calculate_true_negative_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&, const type&) const;

   // Multiple classification tests

   Tensor<type, 1> calculate_multiple_classification_precision() const;
   Tensor<type, 2> calculate_multiple_classification_tests() const;
   void save_confusion(const string&) const;
   void save_multiple_classification_tests(const string&) const;

   // Multiple classification rates

   Tensor<Tensor<Index,1>, 2> calculate_multiple_classification_rates() const;

   Tensor<Tensor<Index,1>, 2> calculate_multiple_classification_rates(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<Index, 1>&) const;

   Tensor<string, 2> calculate_well_classified_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&) const;

   Tensor<string, 2> calculate_misclassified_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&) const;

   void save_well_classified_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&, const string&);

   void save_misclassified_samples(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&, const string&);

   void save_well_classified_samples_statistics(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&, const string&);

   void save_misclassified_samples_statistics(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&, const string&);

   void save_well_classified_samples_probability_histogram(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&, const string&) const;

   void save_well_classified_samples_probability_histogram(const Tensor<string, 2>&, const string&) const;

   void save_misclassified_samples_probability_histogram(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<string, 1>&, const string&);

   void save_misclassified_samples_probability_histogram(const Tensor<string, 2>&, const string&);

   // Forecasting methods

   Tensor<Tensor<type, 1>, 1> calculate_error_autocorrelation(const Index& = 10) const;

   Tensor<Tensor<type, 1>, 1> calculate_inputs_errors_cross_correlation(const Index& = 10) const;

   // Serialization methods

   void print() const;

   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void save(const string&) const;
   void load(const string&);


private: 

   ThreadPool* thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device = nullptr;

   /// Pointer to the neural network object to be tested. 

   NeuralNetwork* neural_network_pointer = nullptr;

   /// Pointer to a data set object.

   DataSet* data_set_pointer = nullptr;

   /// Display messages to screen.
   
   bool display = true;

   const Eigen::array<IndexPair<Index>, 2> SSE = {IndexPair<Index>(0, 0), IndexPair<Index>(1, 1)};
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
