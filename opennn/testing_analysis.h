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

    explicit TestingAnalysis(NeuralNetwork* = nullptr, Dataset* = nullptr);

    struct GoodnessOfFitAnalysis
    {
        float determination = 0.0f;

        VectorR targets;
        VectorR outputs;

        void set(const VectorR&, const VectorR&, float);

        void save(const filesystem::path&) const;

        void print() const;
    };

    struct RocAnalysis
    {
        MatrixR roc_curve;

        float area_under_curve = 0;

        float confidence_limit = 0;

        float optimal_threshold = 0;

        void print() const;
    };

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
    void check() const;
    MatrixR calculate_error() const;

    Tensor3 calculate_error_data() const;
    MatrixR calculate_percentage_error_data() const;

    vector<Descriptives> calculate_absolute_errors_descriptives() const;
    vector<Descriptives> calculate_absolute_errors_descriptives(const MatrixR&, const MatrixR&) const;

    vector<Descriptives> calculate_percentage_errors_descriptives() const;
    vector<Descriptives> calculate_percentage_errors_descriptives(const MatrixR&, const MatrixR&) const;

    vector<vector<Descriptives>> calculate_error_data_descriptives() const;

    vector<Histogram> calculate_error_data_histograms(const Index = 10) const;

    Tensor<VectorI, 1> calculate_maximal_errors(const Index = 10) const;

    MatrixR calculate_errors() const;
    VectorR calculate_errors(const MatrixR&, const MatrixR&) const;
    VectorR calculate_errors(const string&) const;

    MatrixR calculate_binary_classification_errors() const;
    VectorR calculate_binary_classification_errors(const string&) const;

    MatrixR calculate_multiple_classification_errors() const;
    VectorR calculate_multiple_classification_errors(const string&) const;

    float calculate_determination(const VectorR&, const VectorR&) const;
    Tensor<Correlation, 1> linear_correlation(const MatrixR&, const MatrixR&) const;

    Tensor<GoodnessOfFitAnalysis, 1> perform_goodness_of_fit_analysis() const;
    void print_goodness_of_fit_analysis() const;
    VectorR calculate_binary_classification_tests(const float = 0.50) const;

    void print_binary_classification_tests() const;
    MatrixI calculate_confusion(const MatrixR&, const MatrixR&, float = 0.50) const;
    MatrixI calculate_confusion(const float = 0.50) const;

    VectorI calculate_positives_negatives_rate(const MatrixR&, const MatrixR&) const;
    RocAnalysis perform_roc_analysis() const;

    MatrixR calculate_roc_curve(const MatrixR&, const MatrixR&) const;

    float calculate_area_under_curve(const MatrixR&) const;
    float calculate_area_under_curve_confidence_limit(const MatrixR&, const MatrixR&) const;
    float calculate_optimal_threshold(const MatrixR&) const;

    // Lift chart: cumulative-gains analysis over the testing samples sorted by
    // descending output score. Column 0 is the studied-samples ratio, column 1
    // the lift (positives found over positives expected at random).
    MatrixR perform_lift_chart_analysis() const;
    MatrixR calculate_cumulative_gain(const MatrixR&, const MatrixR&) const;
    MatrixR calculate_lift_chart(const MatrixR&) const;

    BinaryClassificationRates calculate_binary_classification_rates(const float = 0.50) const;

    vector<Index> calculate_true_positive_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;
    vector<Index> calculate_false_positive_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;
    vector<Index> calculate_false_negative_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;
    vector<Index> calculate_true_negative_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float) const;
    Tensor<VectorI, 2> calculate_multiple_classification_rates() const;

    Tensor<VectorI, 2> calculate_multiple_classification_rates(const MatrixR&, const MatrixR&, const vector<Index>&) const;

    pair<MatrixR, MatrixR> get_targets_and_outputs(const string&) const;

private:

    vector<Index> filter_classification_samples(const MatrixR&, const MatrixR&, const vector<Index>&, float,
                                                bool, bool) const;

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    Index batch_size = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
