//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T E R I A L I Z E D   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "dataset.h"
#include "correlations.h"
#include "statistics.h"

namespace opennn
{

class MaterializedDataset : public Dataset
{

public:

    MaterializedDataset(const Index = 0,
                        const Shape& = {0},
                        const Shape& = {0});

    MaterializedDataset(const filesystem::path&,
                        const string&,
                        bool = true,
                        bool = false,
                        const Codification& = Codification::UTF8);

    Index get_samples_number() const override { return data.rows(); }
    using Dataset::get_samples_number;

    const MatrixR& get_data() const { return data; }
    MatrixR get_feature_data(const string&) const;
    MatrixR get_data(const string&, const string&) const override;
    MatrixR get_data_from_indices(const vector<Index>&, const vector<Index>&) const override;

    VectorR get_sample_data(const Index) const;

    MatrixR get_variable_data(const Index) const;
    MatrixR get_variable_data(const Index, const vector<Index>&) const;
    MatrixR get_variable_data(const string&) const;

    void set(const Index = 0, const Shape& = {}, const Shape& = {});
    void set(const filesystem::path&,
             const string&,
             bool = true,
             bool = false,
             const Codification& = Codification::UTF8);
    void set(const filesystem::path&);

    void set_data(const MatrixR&);

    vector<string> unuse_uncorrelated_variables(const float = 0.25f);
    vector<string> unuse_collinear_variables(const float = 0.95f);

    void set_data_constant(const float);

    vector<Descriptives> calculate_feature_descriptives() const;
    vector<Descriptives> calculate_feature_descriptives(const string&) const override;

    vector<Descriptives> calculate_variable_descriptives_positive_samples() const;
    vector<Descriptives> calculate_variable_descriptives_negative_samples() const;
    vector<Descriptives> calculate_variable_descriptives_categories(const Index) const;

    vector<Histogram> calculate_variable_distributions(const Index = 10) const;
    vector<BoxPlot> calculate_variables_box_plots() const;

    Tensor<Correlation, 2> calculate_input_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), Correlation::Method, const string&) const;
    Tensor<Correlation, 2> calculate_input_variable_pearson_correlations() const;
    Tensor<Correlation, 2> calculate_input_variable_spearman_correlations() const;

    Tensor<Correlation, 2> calculate_input_target_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), const string&) const;
    Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const override;
    Tensor<Correlation, 2> calculate_input_target_variable_spearman_correlations() const;

    VectorI calculate_correlations_rank() const override;

    vector<Descriptives> scale_data();
    vector<Descriptives> scale_features(const string&) override;
    void unscale_features(const string&, const vector<Descriptives>&) override;

    VectorI calculate_target_distribution() const override;
    vector<vector<Index>> calculate_Tukey_outliers(const float = 1.5f, bool = false);
    vector<vector<Index>> replace_Tukey_outliers_with_NaN(const float = 1.5f);
    void unuse_Tukey_outliers(const float = 1.5f);

    void set_data_random() override;
    void set_data_integer(const Index vocabulary_size) override;
    void set_data_rosenbrock();
    void set_data_binary_classification();

    void from_JSON(const JsonDocument&) override;

    void save_data() const;
    void save_data_binary(const filesystem::path&) const;
    void load_data_binary();

    bool has_nan() const override;
    bool has_nan_row(const Index) const;

    void impute_missing_values_unuse() override;
    void impute_missing_values_statistic(const MissingValuesMethod&);
    void impute_missing_values_interpolate() override;

    void scrub_missing_values() override;
    void calculate_missing_values_statistics();

    VectorI count_nans_per_variable() const;
    Index count_variables_with_nan() const;
    Index count_rows_with_nan() const;
    Index count_nan() const;

    void set_binary_variables();

    void read_csv() override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool = true,
                     int contiguous = -1) const override;

    void fill_decoder(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool = true,
                      int contiguous = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool = true,
                      int contiguous = -1) const override;

protected:

    MatrixR data;

    void infer_variable_types_from_data();
    vector<Index> filter_used_samples_by_column(Index, bool) const;
    void apply_scaler(Index, const string&, const Descriptives&, bool);

    void samples_from_JSON(const Json*);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
