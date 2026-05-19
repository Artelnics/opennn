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

/// @brief Performs post-training analysis of a neural network: errors, confusion matrices, ROC, gain charts, etc.
class TestingAnalysis
{

public:

    /// @brief Constructs the analyser bound to an optional neural network and dataset.
    TestingAnalysis(NeuralNetwork* = nullptr, Dataset* = nullptr);

    /// @brief Coefficient of determination and the matching target/output series for a single output variable.
    struct GoodnessOfFitAnalysis
    {
        float determination = 0.0f;

        VectorR targets;
        VectorR outputs;

        /// @brief Stores the target and output series together with the determination coefficient.
        void set(const VectorR&, const VectorR&, float);

        /// @brief Saves the analysis to disk.
        void save(const filesystem::path&) const;

        /// @brief Prints the analysis to stdout.
        void print() const;
    };

    /// @brief Results of a ROC analysis: ROC curve, area under it and optimal threshold.
    struct RocAnalysis
    {
        MatrixR roc_curve;

        float area_under_curve = 0;

        float confidence_limit = 0;

        float optimal_threshold = 0;

        /// @brief Prints the ROC analysis to stdout.
        void print() const;
    };

    /// @brief Results of a Kolmogorov-Smirnov analysis: cumulative gains and maximum gain.
    struct KolmogorovSmirnovResults
    {
        MatrixR positive_cumulative_gain;

        MatrixR negative_cumulative_gain;

        VectorR maximum_gain;
    };

    /// @brief Sample indices split into the four cells of a binary classification confusion matrix.
    struct BinaryClassificationRates
    {
        vector<Index> true_positives_indices;

        vector<Index> false_positives_indices;

        vector<Index> false_negatives_indices;

        vector<Index> true_negatives_indices;
    };
    const NeuralNetwork* get_neural_network() const { return neural_network; }
    const Dataset* get_dataset() const { return dataset; }
    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }
    void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }
    void set_batch_size(Index new_batch_size) { batch_size = new_batch_size; }
    Index get_batch_size() const { return batch_size; }

    /// @brief Verifies that the neural network and dataset are consistent for testing analysis.
    void check() const;

    /// @brief Computes the overall error matrix between targets and outputs on the testing samples.
    MatrixR calculate_error() const;

    /// @brief Computes the per-sample, per-variable error tensor on the testing samples.
    Tensor3 calculate_error_data() const;

    /// @brief Computes the per-sample percentage error matrix on the testing samples.
    MatrixR calculate_percentage_error_data() const;

    /// @brief Computes descriptive statistics of the absolute errors over the testing samples.
    vector<Descriptives> calculate_absolute_errors_descriptives() const;

    /// @brief Computes descriptive statistics of the absolute errors between the supplied targets and outputs.
    vector<Descriptives> calculate_absolute_errors_descriptives(const MatrixR&, const MatrixR&) const;

    /// @brief Computes descriptive statistics of the percentage errors over the testing samples.
    vector<Descriptives> calculate_percentage_errors_descriptives() const;

    /// @brief Computes descriptive statistics of the percentage errors between the supplied targets and outputs.
    vector<Descriptives> calculate_percentage_errors_descriptives(const MatrixR&, const MatrixR&) const;

    /// @brief Computes descriptive statistics of the per-sample error data on the testing samples.
    vector<vector<Descriptives>> calculate_error_data_descriptives() const;

    /// @brief Prints the error-data descriptive statistics to stdout.
    void print_error_data_descriptives() const;

    /// @brief Builds histograms of the per-variable error data on the testing samples.
    vector<Histogram> calculate_error_data_histograms(const Index = 10) const;

    /// @brief Returns the indices of the samples with the largest errors per output variable.
    Tensor<VectorI, 1> calculate_maximal_errors(const Index = 10) const;

    /// @brief Computes the per-variable error metrics on the testing samples.
    MatrixR calculate_errors() const;

    /// @brief Computes the error metrics for the supplied targets and outputs.
    VectorR calculate_errors(const MatrixR&, const MatrixR&) const;

    /// @brief Computes the error metrics for the sample subset with the given role name.
    VectorR calculate_errors(const string&) const;

    /// @brief Computes binary classification error metrics on the testing samples.
    MatrixR calculate_binary_classification_errors() const;

    /// @brief Computes binary classification error metrics on the samples with the given role name.
    VectorR calculate_binary_classification_errors(const string&) const;

    /// @brief Computes multi-class classification error metrics on the testing samples.
    MatrixR calculate_multiple_classification_errors() const;

    /// @brief Computes multi-class classification error metrics on the samples with the given role name.
    VectorR calculate_multiple_classification_errors(const string&) const;

    /// @brief Computes accuracy when a masking matrix indicates which tokens to consider (e.g. for language models).
    float calculate_masked_accuracy(const Tensor3&, const MatrixR&) const;

    /// @brief Computes the coefficient of determination R^2 between the supplied target and output series.
    float calculate_determination(const VectorR&, const VectorR&) const;

    /// @brief Computes the linear correlation between each target/output column pair.
    Tensor<Correlation, 1> linear_correlation(const MatrixR&, const MatrixR&) const;

    /// @brief Prints the linear correlations between targets and outputs to stdout.
    void print_linear_correlations() const;

    /// @brief Performs goodness-of-fit analysis for each output variable.
    Tensor<GoodnessOfFitAnalysis, 1> perform_goodness_of_fit_analysis() const;

    /// @brief Prints the goodness-of-fit analysis to stdout.
    void print_goodness_of_fit_analysis() const;

    /// @brief Computes the standard binary classification metrics for the given decision threshold.
    VectorR calculate_binary_classification_tests(const float = 0.50) const;

    /// @brief Prints binary classification metrics to stdout.
    void print_binary_classification_tests() const;

    /// @brief Computes per-label binary confusion matrices for multi-label classification.
    vector<MatrixI> calculate_multilabel_confusion(const float) const;

    /// @brief Computes the confusion matrix from the supplied targets and outputs.
    MatrixI calculate_confusion(const MatrixR&, const MatrixR&, float = 0.50) const;

    /// @brief Computes the confusion matrix on the testing samples for the given decision threshold.
    MatrixI calculate_confusion(const float = 0.50) const;

    /// @brief Counts positives and negatives in targets and outputs (returns TP, FP, FN, TN).
    VectorI calculate_positives_negatives_rate(const MatrixR&, const MatrixR&) const;

    /// @brief Performs ROC analysis on the testing samples.
    RocAnalysis perform_roc_analysis() const;

    /// @brief Computes the ROC curve from the supplied targets and outputs.
    MatrixR calculate_roc_curve(const MatrixR&, const MatrixR&) const;

    /// @brief Computes the area under the supplied ROC curve.
    float calculate_area_under_curve(const MatrixR&) const;

    /// @brief Computes the confidence limit of the area under the ROC curve.
    float calculate_area_under_curve_confidence_limit(const MatrixR&, const MatrixR&) const;

    /// @brief Computes the decision threshold that maximizes the ROC criterion.
    float calculate_optimal_threshold(const MatrixR&) const;

    /// @brief Performs a cumulative gain analysis on the testing samples.
    MatrixR perform_cumulative_gain_analysis() const;

    /// @brief Computes the positive cumulative gain curve from the supplied targets and outputs.
    MatrixR calculate_cumulative_gain(const MatrixR&, const MatrixR&) const;

    /// @brief Computes the negative cumulative gain curve from the supplied targets and outputs.
    MatrixR calculate_negative_cumulative_gain(const MatrixR&, const MatrixR&)const;

    /// @brief Performs a lift chart analysis on the testing samples.
    MatrixR perform_lift_chart_analysis() const;

    /// @brief Computes the lift chart from the supplied cumulative gain matrix.
    MatrixR calculate_lift_chart(const MatrixR&) const;

    /// @brief Performs a Kolmogorov-Smirnov analysis on the testing samples.
    KolmogorovSmirnovResults perform_Kolmogorov_Smirnov_analysis() const;

    /// @brief Computes the maximum gain between positive and negative cumulative gain curves.
    VectorR calculate_maximum_gain(const MatrixR&, const MatrixR&) const;

    /// @brief Builds histograms of the supplied outputs for the given number of bins.
    vector<Histogram> calculate_output_histogram(const MatrixR&, Index = 10) const;

    /// @brief Returns sample indices in the four cells of the binary confusion matrix for the given threshold.
    BinaryClassificationRates calculate_binary_classification_rates(const float = 0.50) const;

    /// @brief Returns the indices of true positive samples given targets, outputs and a candidate index list.
    vector<Index> calculate_true_positive_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;

    /// @brief Returns the indices of false positive samples given targets, outputs and a candidate index list.
    vector<Index> calculate_false_positive_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;

    /// @brief Returns the indices of false negative samples given targets, outputs and a candidate index list.
    vector<Index> calculate_false_negative_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;

    /// @brief Returns the indices of true negative samples given targets, outputs and a candidate index list.
    vector<Index> calculate_true_negative_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;

    /// @brief Computes the per-class precision for multi-class classification.
    VectorR calculate_multiple_classification_precision() const;

    /// @brief Computes the standard multi-class classification metrics.
    MatrixR calculate_multiple_classification_tests() const;

    /// @brief Returns the per-cell sample indices of the multi-class confusion matrix for the testing samples.
    Tensor<VectorI, 2> calculate_multiple_classification_rates() const;

    /// @brief Returns the per-cell sample indices of the multi-class confusion matrix for the supplied data.
    Tensor<VectorI, 2> calculate_multiple_classification_rates(const MatrixR&, const MatrixR&, const vector<Index>&) const;

    /// @brief Returns the well-classified samples annotated with their target and output labels.
    Tensor<string, 2> calculate_well_classified_samples(const MatrixR&, const MatrixR&, const vector<string>&) const;

    /// @brief Returns the misclassified samples annotated with their target and output labels.
    Tensor<string, 2> calculate_misclassified_samples(const MatrixR&, const MatrixR&, const vector<string>&) const;

    /// @brief Saves the confusion matrix of the testing samples to disk.
    void save_confusion(const filesystem::path&) const;

    /// @brief Saves the multi-class classification metrics of the testing samples to disk.
    void save_multiple_classification_tests(const filesystem::path&) const;

    /// @brief Saves the well-classified samples annotated table to disk.
    void save_well_classified_samples(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    /// @brief Saves the misclassified samples annotated table to disk.
    void save_misclassified_samples(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    /// @brief Saves per-class statistics of the well-classified samples to disk.
    void save_well_classified_samples_statistics(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    /// @brief Saves per-class statistics of the misclassified samples to disk.
    void save_misclassified_samples_statistics(const MatrixR&, const MatrixR&, const vector<string>&, const filesystem::path&) const;

    /// @brief Computes the autocorrelation of the residual errors up to the given lag.
    vector<VectorR> calculate_error_autocorrelation(const Index = 10) const;

    /// @brief Computes the cross-correlation between input variables and residual errors up to the given lag.
    vector<VectorR> calculate_inputs_errors_cross_correlation(const Index = 10) const;

    /// @brief Computes loss and accuracy of a transformer model on the testing samples.
    pair<float, float> test_transformer() const;

    /// @brief Generates a transformer prediction string from a context.
    /// @param context_string Tokenized context for the model.
    /// @param imported_vocabulary True if the vocabulary was imported from outside the dataset.
    string test_transformer(const vector<string>& context_string, bool imported_vocabulary) const;

    /// @brief Loads the testing analysis configuration from a JSON document.
    void from_JSON(const JsonDocument&);

    /// @brief Writes the testing analysis configuration to a JSON writer.
    void to_JSON(JsonWriter&) const;

    /// @brief Saves the testing analysis configuration to disk.
    void save(const filesystem::path&) const;

    /// @brief Loads the testing analysis configuration from disk.
    void load(const filesystem::path&);

private:

    pair<MatrixR, MatrixR> get_targets_and_outputs(const string&) const;

    vector<Index> filter_classification_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float,
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
