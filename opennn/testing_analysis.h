//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

class Dataset;
class NeuralNetwork;

struct Descriptives;
struct Histogram;
struct Correlation;

class TestingAnalysis
{

public:

    TestingAnalysis(NeuralNetwork* = nullptr, Dataset* = nullptr);

    struct GoodnessOfFitAnalysis
    {
        type determination = type(0);

        VectorR targets;
        VectorR outputs;

        void set(const VectorR&, const VectorR&, type);

        void save(const filesystem::path&) const;

        void print() const;
    };

    struct RocAnalysis
    {
        MatrixR roc_curve;

        type area_under_curve = 0;

        type confidence_limit = 0;

        type optimal_threshold = 0;

        void print() const;
    };

    struct KolmogorovSmirnovResults
    {
        MatrixR positive_cumulative_gain;

        MatrixR negative_cumulative_gain;

        VectorR maximum_gain;
    };

    struct BinaryClassificationRates
    {
        vector<Index> true_positives_indices;

        vector<Index> false_positives_indices;

        vector<Index> false_negatives_indices;

        vector<Index> true_negatives_indices;
    };

    // Get

    const NeuralNetwork* get_neural_network() const { return neural_network; }
    const Dataset* get_dataset() const { return dataset; }

    // Set

    void set_neural_network(NeuralNetwork* nn) { neural_network = nn; }
    void set_dataset(Dataset* ds) { dataset = ds; }

    // Checking

    void check() const;

    // Error data

    MatrixR calculate_error() const;

    Tensor3 calculate_error_data() const;
    MatrixR calculate_percentage_error_data() const;

    vector<Descriptives> calculate_absolute_errors_descriptives() const;
    vector<Descriptives> calculate_absolute_errors_descriptives(const MatrixR&, const MatrixR&) const;

    vector<Descriptives> calculate_percentage_errors_descriptives() const;
    vector<Descriptives> calculate_percentage_errors_descriptives(const MatrixR&, const MatrixR&) const;

    vector<vector<Descriptives>> calculate_error_data_descriptives() const;
    void print_error_data_descriptives() const;

    vector<Histogram> calculate_error_data_histograms(const Index = 10) const;

    Tensor<VectorI, 1> calculate_maximal_errors(const Index = 10) const;

    MatrixR calculate_errors() const;
    VectorR calculate_errors(const MatrixR&, const MatrixR&) const;
    VectorR calculate_errors(const string&) const;

    MatrixR calculate_binary_classification_errors() const;
    VectorR calculate_binary_classification_errors(const string&) const;

    MatrixR calculate_multiple_classification_errors() const;
    VectorR calculate_multiple_classification_errors(const string&) const;

    type calculate_masked_accuracy(const Tensor3&, const MatrixR&) const;

    type calculate_determination(const VectorR&, const VectorR&) const;

    // Goodness-of-fit analysis

    Tensor<Correlation, 1> linear_correlation(const MatrixR&, const MatrixR&) const;

    void print_linear_correlations() const;

    Tensor<GoodnessOfFitAnalysis, 1> perform_goodness_of_fit_analysis() const;
    void print_goodness_of_fit_analysis() const;

    // Binary classifcation

    VectorR calculate_binary_classification_tests(const type = 0.50) const;

    void print_binary_classification_tests() const;

    // Confusion

    vector<MatrixI> calculate_multilabel_confusion(const type) const;
    MatrixI calculate_confusion(const MatrixR&, const MatrixR&, type = 0.50) const;
    MatrixI calculate_confusion(const type = 0.50) const;

    VectorI calculate_positives_negatives_rate(const MatrixR&, const MatrixR&) const;

    // ROC curve

    RocAnalysis perform_roc_analysis() const;

    MatrixR calculate_roc_curve(const MatrixR&, const MatrixR&) const;

    type calculate_area_under_curve(const MatrixR&) const;
    type calculate_area_under_curve_confidence_limit(const MatrixR&, const MatrixR&) const;
    type calculate_optimal_threshold(const MatrixR&) const;

    // Lift Chart

    MatrixR perform_cumulative_gain_analysis() const;
    MatrixR calculate_cumulative_gain(const MatrixR&, const MatrixR&) const;
    MatrixR calculate_negative_cumulative_gain(const MatrixR&, const MatrixR&)const;

    MatrixR perform_lift_chart_analysis() const;
    MatrixR calculate_lift_chart(const MatrixR&) const;

    KolmogorovSmirnovResults perform_Kolmogorov_Smirnov_analysis() const;
    VectorR calculate_maximum_gain(const MatrixR&, const MatrixR&) const;

    // Output histogram

    vector<Histogram> calculate_output_histogram(const MatrixR&, Index = 10) const;

    // Binary classification rates

    BinaryClassificationRates calculate_binary_classification_rates(const type = 0.50) const;

    vector<Index> calculate_true_positive_samples(const MatrixR&, const MatrixR&, const vector<Index>&, type) const;
    vector<Index> calculate_false_positive_samples(const MatrixR&, const MatrixR&, const vector<Index>&, type) const;
    vector<Index> calculate_false_negative_samples(const MatrixR&, const MatrixR&, const vector<Index>&, type) const;
    vector<Index> calculate_true_negative_samples(const MatrixR&, const MatrixR&, const vector<Index>&, type) const;

    // Multiple classification tests

    VectorR calculate_multiple_classification_precision() const;
    MatrixR calculate_multiple_classification_tests() const;

    // Multiple classification rates

    Tensor<VectorI, 2> calculate_multiple_classification_rates() const;

    Tensor<VectorI, 2> calculate_multiple_classification_rates(const MatrixR&, const MatrixR&, const vector<Index>&) const;

    Tensor<string, 2> calculate_well_classified_samples(const MatrixR&, const MatrixR&, const vector<string>&) const;

    Tensor<string, 2> calculate_misclassified_samples(const MatrixR&, const MatrixR&, const vector<string>&) const;

    // Save

    void save_confusion(const filesystem::path&) const;

    void save_multiple_classification_tests(const filesystem::path&) const;

    void save_well_classified_samples(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    void save_misclassified_samples(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    void save_well_classified_samples_statistics(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    void save_misclassified_samples_statistics(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    // Forecasting

    vector<VectorR> calculate_error_autocorrelation(const Index = 10) const;

    vector<VectorR> calculate_inputs_errors_cross_correlation(const Index = 10) const;

    // Transformer

    pair<type, type> test_transformer() const;

    string test_transformer(const vector<string>& context_string, bool imported_vocabulary) const;

    // Serialization

    void from_XML(const XmlDocument&);

    void to_XML(XmlPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

#ifdef OPENNN_WITH_CUDA

    MatrixI calculate_confusion_cuda(const type = 0.50) const;

#endif

private:

    pair<MatrixR, MatrixR> get_targets_and_outputs(const string&) const;

    vector<Index> filter_classification_samples(const MatrixR&, const MatrixR&, const vector<Index>&, type,
                                                bool target_positive, bool output_positive) const;

    MatrixR calculate_cumulative_gain_impl(const MatrixR&, const MatrixR&, bool) const;

    Tensor<string, 2> classify_samples(const MatrixR&, const MatrixR&, const vector<string>&, bool match) const;

    static VectorR extract_probabilities(const Tensor<string, 2>&);

    void save_classified_samples_csv(const Tensor<string, 2>&, const filesystem::path&) const;
    void save_classified_samples_statistics_csv(const Tensor<string, 2>&, const filesystem::path&) const;
    void save_classified_samples_probability_histogram(const Tensor<string, 2>&, const filesystem::path&) const;

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    Index batch_size = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
