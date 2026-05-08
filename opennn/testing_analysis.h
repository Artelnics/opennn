//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file testing_analysis.h
 * @brief Declares the TestingAnalysis class.
 *
 * TestingAnalysis evaluates a trained NeuralNetwork against the testing
 * samples of a Dataset, producing error statistics, confusion matrices,
 * ROC curves, lift charts and other diagnostic artefacts.
 */

#pragma once

#include "pch.h"

namespace opennn
{

class Dataset;
class NeuralNetwork;

struct Descriptives;
struct Histogram;
struct Correlation;

/**
 * @class TestingAnalysis
 * @brief Computes diagnostic metrics for a trained network on testing data.
 *
 * Holds non-owning pointers to a NeuralNetwork and a Dataset and runs the
 * network on the dataset's testing partition. Exposes:
 *   - regression diagnostics: errors, error histograms, descriptives,
 *     goodness-of-fit, linear correlation;
 *   - binary-classification diagnostics: confusion matrix, ROC, AUC,
 *     cumulative gain, lift, Kolmogorov-Smirnov;
 *   - multi-class diagnostics: precision, multilabel confusion, per-class rates;
 *   - utilities to enumerate well-classified and misclassified samples.
 */
class TestingAnalysis
{

public:

    /**
     * @brief Constructs a testing analysis bound to a network and dataset.
     * @param new_neural_network Non-owning pointer to the trained network.
     * @param new_dataset Non-owning pointer to the dataset providing testing samples.
     */
    TestingAnalysis(NeuralNetwork* new_neural_network = nullptr, Dataset* new_dataset = nullptr);

    /**
     * @struct GoodnessOfFitAnalysis
     * @brief Per-output regression goodness-of-fit summary.
     *
     * Captures the coefficient of determination together with the raw
     * target/output vectors used to compute it.
     */
    struct GoodnessOfFitAnalysis
    {
        /// Coefficient of determination (R^2) for this output.
        float determination = 0.0f;

        /// Target values for the testing samples.
        VectorR targets;
        /// Network outputs for the testing samples.
        VectorR outputs;

        /**
         * @brief Populates the struct from raw vectors and a precomputed R^2.
         * @param targets Target values.
         * @param outputs Network outputs.
         * @param determination R^2 between targets and outputs.
         */
        void set(const VectorR& targets, const VectorR& outputs, float determination);

        /**
         * @brief Saves the struct to a CSV file.
         * @param file_name Destination path.
         */
        void save(const filesystem::path& file_name) const;

        /// Prints the struct to standard output.
        void print() const;
    };

    /**
     * @struct RocAnalysis
     * @brief Output of perform_roc_analysis().
     */
    struct RocAnalysis
    {
        /// ROC curve as a matrix of [false-positive-rate, true-positive-rate] rows.
        MatrixR roc_curve;

        /// Area under the ROC curve.
        float area_under_curve = 0;

        /// 95% confidence limit on the AUC.
        float confidence_limit = 0;

        /// Threshold that maximizes Youden's J on the curve.
        float optimal_threshold = 0;

        /// Prints the analysis to standard output.
        void print() const;
    };

    /**
     * @struct KolmogorovSmirnovResults
     * @brief Output of perform_Kolmogorov_Smirnov_analysis().
     */
    struct KolmogorovSmirnovResults
    {
        /// Cumulative-gain curve for the positive class.
        MatrixR positive_cumulative_gain;

        /// Cumulative-gain curve for the negative class.
        MatrixR negative_cumulative_gain;

        /// Maximum vertical distance between the two cumulative-gain curves.
        VectorR maximum_gain;
    };

    /**
     * @struct BinaryClassificationRates
     * @brief Sample indices grouped by binary-classification outcome.
     */
    struct BinaryClassificationRates
    {
        /// Indices of samples classified as positive whose target is positive.
        vector<Index> true_positives_indices;

        /// Indices of samples classified as positive whose target is negative.
        vector<Index> false_positives_indices;

        /// Indices of samples classified as negative whose target is positive.
        vector<Index> false_negatives_indices;

        /// Indices of samples classified as negative whose target is negative.
        vector<Index> true_negatives_indices;
    };

    /**
     * @brief Returns the network being evaluated.
     * @return Const pointer to the network (may be nullptr).
     */
    const NeuralNetwork* get_neural_network() const { return neural_network; }

    /**
     * @brief Returns the dataset providing testing samples.
     * @return Const pointer to the dataset (may be nullptr).
     */
    const Dataset* get_dataset() const { return dataset; }

    /**
     * @brief Replaces the network being evaluated.
     * @param new_neural_network Non-owning pointer to the new network.
     */
    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }

    /**
     * @brief Replaces the dataset providing testing samples.
     * @param new_dataset Non-owning pointer to the new dataset.
     */
    void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }

    /**
     * @brief Sets the batch size used when running the network.
     * @param new_batch_size Number of samples per batch (0 = full pass).
     */
    void set_batch_size(Index new_batch_size) { batch_size = new_batch_size; }

    /**
     * @brief Returns the batch size used when running the network.
     * @return Configured batch size.
     */
    Index get_batch_size() const { return batch_size; }

    /**
     * @brief Validates that network and dataset are configured.
     * @throws runtime_error if either is missing.
     */
    void check() const;

    /**
     * @brief Computes the per-output mean error on the testing samples.
     * @return Matrix of mean errors (one row per error type, one column per output).
     */
    MatrixR calculate_error() const;

    /**
     * @brief Computes raw error data for every testing sample.
     * @return Rank-3 tensor [error_type, sample, output] of error values.
     */
    Tensor3 calculate_error_data() const;

    /**
     * @brief Computes percentage errors for every testing sample.
     * @return Matrix of percentage errors (sample x output).
     */
    MatrixR calculate_percentage_error_data() const;

    /**
     * @brief Computes descriptive statistics of absolute errors on the testing partition.
     * @return One Descriptives entry per network output.
     */
    vector<Descriptives> calculate_absolute_errors_descriptives() const;

    /**
     * @brief Computes descriptive statistics of absolute errors between the supplied tensors.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return One Descriptives entry per output column.
     */
    vector<Descriptives> calculate_absolute_errors_descriptives(const MatrixR& targets, const MatrixR& outputs) const;

    /**
     * @brief Computes descriptive statistics of percentage errors on the testing partition.
     * @return One Descriptives entry per network output.
     */
    vector<Descriptives> calculate_percentage_errors_descriptives() const;

    /**
     * @brief Computes descriptive statistics of percentage errors between the supplied tensors.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return One Descriptives entry per output column.
     */
    vector<Descriptives> calculate_percentage_errors_descriptives(const MatrixR& targets, const MatrixR& outputs) const;

    /**
     * @brief Computes descriptive statistics for every error type on the testing partition.
     * @return Outer vector indexed by error type, inner vector by output.
     */
    vector<vector<Descriptives>> calculate_error_data_descriptives() const;

    /// Prints the result of calculate_error_data_descriptives() to standard output.
    void print_error_data_descriptives() const;

    /**
     * @brief Builds histograms of the per-sample errors.
     * @param bins_number Number of bins per histogram.
     * @return One Histogram per network output.
     */
    vector<Histogram> calculate_error_data_histograms(const Index bins_number = 10) const;

    /**
     * @brief Returns the indices of the samples with the largest errors.
     * @param maximal_number Number of indices to return.
     * @return Tensor of vectors of indices (one entry per error type).
     */
    Tensor<VectorI, 1> calculate_maximal_errors(const Index maximal_number = 10) const;

    /**
     * @brief Computes per-output regression error metrics on the testing partition.
     * @return Matrix [metric x output] containing absolute, relative and percentage errors.
     */
    MatrixR calculate_errors() const;

    /**
     * @brief Computes regression error metrics between the supplied tensors.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return Vector of metrics aggregated across outputs.
     */
    VectorR calculate_errors(const MatrixR& targets, const MatrixR& outputs) const;

    /**
     * @brief Computes regression error metrics on a specific samples subset.
     * @param samples_role Sample-role filter ("Training", "Validation", "Testing").
     * @return Vector of metrics for the requested partition.
     */
    VectorR calculate_errors(const string& samples_role) const;

    /**
     * @brief Computes binary-classification error metrics on the testing partition.
     * @return Matrix of metrics suitable for downstream reporting.
     */
    MatrixR calculate_binary_classification_errors() const;

    /**
     * @brief Computes binary-classification error metrics on a specific samples subset.
     * @param samples_role Sample-role filter.
     * @return Vector of metrics for the requested partition.
     */
    VectorR calculate_binary_classification_errors(const string& samples_role) const;

    /**
     * @brief Computes multi-class classification error metrics on the testing partition.
     * @return Matrix of metrics suitable for downstream reporting.
     */
    MatrixR calculate_multiple_classification_errors() const;

    /**
     * @brief Computes multi-class classification error metrics on a specific samples subset.
     * @param samples_role Sample-role filter.
     * @return Vector of metrics for the requested partition.
     */
    VectorR calculate_multiple_classification_errors(const string& samples_role) const;

    /**
     * @brief Computes accuracy ignoring positions flagged by a mask (e.g. padding tokens).
     * @param outputs Network outputs as a rank-3 tensor (e.g. [batch, time, vocab]).
     * @param mask Binary mask matching the [batch, time] dims; non-zero positions count.
     * @return Fraction of unmasked positions correctly classified.
     */
    float calculate_masked_accuracy(const Tensor3& outputs, const MatrixR& mask) const;

    /**
     * @brief Computes the coefficient of determination R^2.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return R^2 between the two vectors.
     */
    float calculate_determination(const VectorR& targets, const VectorR& outputs) const;

    /**
     * @brief Computes per-output linear correlation between targets and outputs.
     * @param targets Target values (samples x outputs).
     * @param outputs Network outputs (samples x outputs).
     * @return One Correlation per output column.
     */
    Tensor<Correlation, 1> linear_correlation(const MatrixR& targets, const MatrixR& outputs) const;

    /// Prints the linear correlations to standard output.
    void print_linear_correlations() const;

    /**
     * @brief Runs goodness-of-fit analysis for every output.
     * @return One GoodnessOfFitAnalysis per output.
     */
    Tensor<GoodnessOfFitAnalysis, 1> perform_goodness_of_fit_analysis() const;

    /// Prints the result of perform_goodness_of_fit_analysis() to standard output.
    void print_goodness_of_fit_analysis() const;

    /**
     * @brief Computes a battery of binary-classification metrics at a given threshold.
     * @param decision_threshold Probability threshold separating the positive and negative class.
     * @return Vector with accuracy, precision, recall, specificity, F1, MCC, etc.
     */
    VectorR calculate_binary_classification_tests(const float decision_threshold = 0.50) const;

    /// Prints the result of calculate_binary_classification_tests() to standard output.
    void print_binary_classification_tests() const;

    /**
     * @brief Computes one confusion matrix per label for a multilabel problem.
     * @param decision_threshold Probability threshold per label.
     * @return One MatrixI per output label.
     */
    vector<MatrixI> calculate_multilabel_confusion(const float decision_threshold) const;

    /**
     * @brief Computes a confusion matrix from the supplied targets and outputs.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param decision_threshold Probability threshold (binary case only).
     * @return Integer confusion matrix [true class x predicted class].
     */
    MatrixI calculate_confusion(const MatrixR& targets, const MatrixR& outputs, float decision_threshold = 0.50) const;

    /**
     * @brief Computes the confusion matrix on the testing partition.
     * @param decision_threshold Probability threshold (binary case only).
     * @return Integer confusion matrix [true class x predicted class].
     */
    MatrixI calculate_confusion(const float decision_threshold = 0.50) const;

    /**
     * @brief Counts samples that are positive vs. negative per class.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return Vector with the positive and negative counts.
     */
    VectorI calculate_positives_negatives_rate(const MatrixR& targets, const MatrixR& outputs) const;

    /**
     * @brief Runs full ROC analysis on the testing partition.
     * @return RocAnalysis with curve, AUC, confidence limit and optimal threshold.
     */
    RocAnalysis perform_roc_analysis() const;

    /**
     * @brief Computes the ROC curve from the supplied targets and outputs.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return Matrix with [false-positive-rate, true-positive-rate] rows.
     */
    MatrixR calculate_roc_curve(const MatrixR& targets, const MatrixR& outputs) const;

    /**
     * @brief Computes the area under a ROC curve.
     * @param roc_curve Curve produced by calculate_roc_curve().
     * @return AUC value in [0, 1].
     */
    float calculate_area_under_curve(const MatrixR& roc_curve) const;

    /**
     * @brief Computes the 95% confidence limit on the AUC.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return Confidence limit on the AUC.
     */
    float calculate_area_under_curve_confidence_limit(const MatrixR& targets, const MatrixR& outputs) const;

    /**
     * @brief Returns the threshold that maximizes Youden's J on a ROC curve.
     * @param roc_curve Curve produced by calculate_roc_curve().
     * @return Optimal probability threshold.
     */
    float calculate_optimal_threshold(const MatrixR& roc_curve) const;

    /**
     * @brief Runs cumulative-gain analysis on the testing partition.
     * @return Matrix with the positive class cumulative-gain curve.
     */
    MatrixR perform_cumulative_gain_analysis() const;

    /**
     * @brief Computes the positive-class cumulative-gain curve.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return Cumulative-gain curve.
     */
    MatrixR calculate_cumulative_gain(const MatrixR& targets, const MatrixR& outputs) const;

    /**
     * @brief Computes the negative-class cumulative-gain curve.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @return Cumulative-gain curve.
     */
    MatrixR calculate_negative_cumulative_gain(const MatrixR& targets, const MatrixR& outputs)const;

    /**
     * @brief Runs lift-chart analysis on the testing partition.
     * @return Matrix with the lift curve.
     */
    MatrixR perform_lift_chart_analysis() const;

    /**
     * @brief Computes the lift curve from a cumulative-gain curve.
     * @param cumulative_gain Curve produced by calculate_cumulative_gain().
     * @return Lift curve.
     */
    MatrixR calculate_lift_chart(const MatrixR& cumulative_gain) const;

    /**
     * @brief Runs the Kolmogorov-Smirnov analysis on the testing partition.
     * @return KolmogorovSmirnovResults with cumulative-gain curves and maximum gain.
     */
    KolmogorovSmirnovResults perform_Kolmogorov_Smirnov_analysis() const;

    /**
     * @brief Computes the maximum vertical distance between two cumulative-gain curves.
     * @param positive_cumulative_gain Positive-class curve.
     * @param negative_cumulative_gain Negative-class curve.
     * @return Vector containing the maximum gain value and its index.
     */
    VectorR calculate_maximum_gain(const MatrixR& positive_cumulative_gain, const MatrixR& negative_cumulative_gain) const;

    /**
     * @brief Builds histograms of the network output values.
     * @param outputs Network outputs.
     * @param bins_number Number of bins per histogram.
     * @return One Histogram per output.
     */
    vector<Histogram> calculate_output_histogram(const MatrixR& outputs, Index bins_number = 10) const;

    /**
     * @brief Splits the testing samples into TP/FP/FN/TN groups.
     * @param decision_threshold Probability threshold.
     * @return BinaryClassificationRates with sample indices for each group.
     */
    BinaryClassificationRates calculate_binary_classification_rates(const float decision_threshold = 0.50) const;

    /**
     * @brief Returns the indices of the true-positive samples in the supplied subset.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param indices Index subset to consider.
     * @param decision_threshold Probability threshold.
     * @return Indices of true-positive samples.
     */
    vector<Index> calculate_true_positive_samples(const MatrixR& targets, const MatrixR& outputs, const vector<Index>& indices, float decision_threshold) const;

    /**
     * @brief Returns the indices of the false-positive samples in the supplied subset.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param indices Index subset to consider.
     * @param decision_threshold Probability threshold.
     * @return Indices of false-positive samples.
     */
    vector<Index> calculate_false_positive_samples(const MatrixR& targets, const MatrixR& outputs, const vector<Index>& indices, float decision_threshold) const;

    /**
     * @brief Returns the indices of the false-negative samples in the supplied subset.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param indices Index subset to consider.
     * @param decision_threshold Probability threshold.
     * @return Indices of false-negative samples.
     */
    vector<Index> calculate_false_negative_samples(const MatrixR& targets, const MatrixR& outputs, const vector<Index>& indices, float decision_threshold) const;

    /**
     * @brief Returns the indices of the true-negative samples in the supplied subset.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param indices Index subset to consider.
     * @param decision_threshold Probability threshold.
     * @return Indices of true-negative samples.
     */
    vector<Index> calculate_true_negative_samples(const MatrixR& targets, const MatrixR& outputs, const vector<Index>& indices, float decision_threshold) const;

    /**
     * @brief Computes the per-class precision in a multi-class problem.
     * @return Vector of precision values, one per class.
     */
    VectorR calculate_multiple_classification_precision() const;

    /**
     * @brief Computes a battery of multi-class classification metrics.
     * @return Matrix of metrics suitable for downstream reporting.
     */
    MatrixR calculate_multiple_classification_tests() const;

    /**
     * @brief Splits multi-class testing samples by (true class, predicted class).
     * @return Matrix of vectors of sample indices.
     */
    Tensor<VectorI, 2> calculate_multiple_classification_rates() const;

    /**
     * @brief Splits multi-class samples by (true class, predicted class) on supplied data.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param indices Index subset to consider.
     * @return Matrix of vectors of sample indices.
     */
    Tensor<VectorI, 2> calculate_multiple_classification_rates(const MatrixR& targets, const MatrixR& outputs, const vector<Index>& indices) const;

    /**
     * @brief Returns the well-classified samples annotated with predicted/target labels.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param labels_names Class names matching the target columns.
     * @return Per-sample table of predicted vs. target labels.
     */
    Tensor<string, 2> calculate_well_classified_samples(const MatrixR& targets, const MatrixR& outputs, const vector<string>& labels_names) const;

    /**
     * @brief Returns the misclassified samples annotated with predicted/target labels.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param labels_names Class names matching the target columns.
     * @return Per-sample table of predicted vs. target labels.
     */
    Tensor<string, 2> calculate_misclassified_samples(const MatrixR& targets, const MatrixR& outputs, const vector<string>& labels_names) const;

    /**
     * @brief Saves the testing-partition confusion matrix to a CSV file.
     * @param file_name Destination path.
     */
    void save_confusion(const filesystem::path& file_name) const;

    /**
     * @brief Saves the multi-class classification tests to a CSV file.
     * @param file_name Destination path.
     */
    void save_multiple_classification_tests(const filesystem::path& file_name) const;

    /**
     * @brief Saves the well-classified samples to a CSV file.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param labels_names Class names matching the target columns.
     * @param file_name Destination path.
     */
    void save_well_classified_samples(const MatrixR& targets, const MatrixR& outputs, const vector<string>& labels_names, const filesystem::path& file_name) const;

    /**
     * @brief Saves the misclassified samples to a CSV file.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param labels_names Class names matching the target columns.
     * @param file_name Destination path.
     */
    void save_misclassified_samples(const MatrixR& targets, const MatrixR& outputs, const vector<string>& labels_names, const filesystem::path& file_name) const;

    /**
     * @brief Saves descriptive statistics of the well-classified samples to CSV.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param labels_names Class names matching the target columns.
     * @param file_name Destination path.
     */
    void save_well_classified_samples_statistics(const MatrixR& targets, const MatrixR& outputs, const vector<string>& labels_names, const filesystem::path& file_name) const;

    /**
     * @brief Saves descriptive statistics of the misclassified samples to CSV.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param labels_names Class names matching the target columns.
     * @param file_name Destination path.
     */
    void save_misclassified_samples_statistics(const MatrixR& targets, const MatrixR& outputs, const vector<string>& labels_names, const filesystem::path& file_name) const;

    /**
     * @brief Computes the autocorrelation of the residuals at increasing lags.
     * @param maximum_lags_number Maximum lag to compute.
     * @return One autocorrelation vector per output.
     */
    vector<VectorR> calculate_error_autocorrelation(const Index maximum_lags_number = 10) const;

    /**
     * @brief Computes the cross-correlation between inputs and residuals at increasing lags.
     * @param maximum_lags_number Maximum lag to compute.
     * @return One cross-correlation vector per input.
     */
    vector<VectorR> calculate_inputs_errors_cross_correlation(const Index maximum_lags_number = 10) const;

    /**
     * @brief Runs perplexity-style evaluation on a transformer-like network.
     * @return Pair of (loss, accuracy) on the testing partition.
     */
    pair<float, float> test_transformer() const;

    /**
     * @brief Runs free-form generation on a transformer-like network.
     * @param context_string Tokens prepended to the model as context.
     * @param imported_vocabulary Whether to load the vocabulary from disk.
     * @return Generated text continuation.
     */
    string test_transformer(const vector<string>& context_string, bool imported_vocabulary) const;

    /**
     * @brief Restores the analysis state from a JSON document.
     * @param document Parsed JSON produced by to_JSON().
     */
    void from_JSON(const JsonDocument& document);

    /**
     * @brief Serializes the analysis state to JSON.
     * @param writer JSON writer that receives the state tree.
     */
    void to_JSON(JsonWriter& writer) const;

    /**
     * @brief Saves the analysis state to a JSON file on disk.
     * @param file_name Destination path.
     */
    void save(const filesystem::path& file_name) const;

    /**
     * @brief Loads the analysis state from a JSON file on disk.
     * @param file_name Source path.
     */
    void load(const filesystem::path& file_name);

private:

    /**
     * @brief Returns the (targets, outputs) pair for a sample-role partition.
     * @param samples_role Role filter ("Training", "Validation", "Testing").
     * @return Pair (targets, outputs) for the requested partition.
     */
    pair<MatrixR, MatrixR> get_targets_and_outputs(const string& samples_role) const;

    /**
     * @brief Filters classification samples by predicted/target sign.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param indices Index subset to consider.
     * @param decision_threshold Probability threshold.
     * @param target_positive Whether to require targets in the positive class.
     * @param output_positive Whether to require predictions in the positive class.
     * @return Indices that match the requested combination.
     */
    vector<Index> filter_classification_samples(const MatrixR& targets, const MatrixR& outputs, const vector<Index>& indices, float decision_threshold, bool target_positive, bool output_positive) const;

    /**
     * @brief Shared implementation for positive/negative cumulative-gain curves.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param positive_class Whether to compute the positive-class curve.
     * @return Cumulative-gain curve.
     */
    MatrixR calculate_cumulative_gain_impl(const MatrixR& targets, const MatrixR& outputs, bool positive_class) const;

    /**
     * @brief Annotates samples with predicted/target labels and a match flag.
     * @param targets Target values.
     * @param outputs Network outputs.
     * @param labels_names Class names matching the target columns.
     * @param match Whether to keep matched (true) or mismatched (false) samples.
     * @return Per-sample table of predicted vs. target labels.
     */
    Tensor<string, 2> classify_samples(const MatrixR& targets, const MatrixR& outputs, const vector<string>& labels_names, bool match) const;

    /**
     * @brief Extracts the probability column from a classify_samples() output.
     * @param classification Output table from classify_samples().
     * @return Vector of probabilities aligned to the rows of the input table.
     */
    static VectorR extract_probabilities(const Tensor<string, 2>&);

    /**
     * @brief Saves a classify_samples() output to CSV.
     * @param classification Output table from classify_samples().
     * @param file_name Destination path.
     */
    void save_classified_samples_csv(const Tensor<string, 2>&, const filesystem::path&) const;

    /**
     * @brief Saves descriptive statistics of a classify_samples() output to CSV.
     * @param classification Output table from classify_samples().
     * @param file_name Destination path.
     */
    void save_classified_samples_statistics_csv(const Tensor<string, 2>&, const filesystem::path&) const;

    /**
     * @brief Saves a probability histogram of a classify_samples() output to CSV.
     * @param classification Output table from classify_samples().
     * @param file_name Destination path.
     */
    void save_classified_samples_probability_histogram(const Tensor<string, 2>&, const filesystem::path&) const;

    /// Non-owning pointer to the network being evaluated.
    NeuralNetwork* neural_network = nullptr;

    /// Non-owning pointer to the dataset providing testing samples.
    Dataset* dataset = nullptr;

    /// Number of samples per batch when running the network. 0 means full pass.
    Index batch_size = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
