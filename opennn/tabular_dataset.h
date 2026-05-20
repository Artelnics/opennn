//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T A B U L A R   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "dataset.h"
#include "correlations.h"
#include "statistics.h"

namespace opennn
{

class TabularDataset : public Dataset
{

public:

    enum class MissingValuesMethod{Unuse, Mean, Median, Interpolation};

    TabularDataset(const Index = 0,
                        const Shape& = {0},
                        const Shape& = {0});

    TabularDataset(const filesystem::path&,
                        const string&,
                        bool = true,
                        bool = false,
                        const Codification& = Codification::UTF8);

    [[nodiscard]] Index get_samples_number() const override { return data.rows(); }

    [[nodiscard]] const MatrixR& get_data() const { return data; }
    void set_data(const MatrixR&);
    void set_data_constant(float);

    [[nodiscard]] MatrixR get_data(const string&, const string&) const;
    [[nodiscard]] MatrixR get_data_from_indices(const vector<Index>&, const vector<Index>&) const;

    [[nodiscard]] VectorR get_sample_data(Index) const;

    [[nodiscard]] MatrixR get_variable_data(Index) const;
    [[nodiscard]] MatrixR get_variable_data(Index, const vector<Index>&) const;
    [[nodiscard]] MatrixR get_variable_data(const string&) const;

    [[nodiscard]] MatrixR get_feature_data(const string&) const;

    void set(Index = 0, const Shape& = {}, const Shape& = {});
    void set(const filesystem::path&,
             const string&,
             bool = true,
             bool = false,
             const Codification& = Codification::UTF8);
    void set(const filesystem::path&);

    vector<string> get_feature_scalers(const string&) const;

    void set_variable_scalers(const string&);
    void set_variable_scalers(const vector<string>&);
    void set_default_variable_scalers();

    void set_gmt(const Index new_gmt) { gmt = new_gmt; }

    DateFormat infer_dataset_date_format(const vector<Variable>&, const vector<vector<string_view>>&, bool, const string&);

    MissingValuesMethod get_missing_values_method() const { return missing_values_method; }
    string get_missing_values_method_string() const;
    const string& get_missing_values_label() const { return missing_values_label; }
    Index get_missing_values_number() const { return missing_values_number; }

    void set_missing_values_label(string label) { missing_values_label = move(label); }
    void set_missing_values_method(const MissingValuesMethod& method) { missing_values_method = method; }
    void set_missing_values_method(const string&);

    bool has_missing_values(const vector<string_view>&) const;

    void scrub_missing_values() override;
    void calculate_missing_values_statistics();
    void impute_missing_values_statistic(const MissingValuesMethod&);
    virtual void impute_missing_values_unuse();
    virtual void impute_missing_values_interpolate();

    vector<string> unuse_uncorrelated_variables(const float = 0.25f);
    vector<string> unuse_collinear_variables(const float = 0.95f);

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

    [[nodiscard]] bool has_nan() const override;
    [[nodiscard]] bool has_nan_row(Index) const;

    [[nodiscard]] VectorI count_nans_per_variable() const;
    [[nodiscard]] Index count_variables_with_nan() const;
    [[nodiscard]] Index count_rows_with_nan() const;
    [[nodiscard]] Index count_nan() const;

    void save_data() const;
    void save_data_binary(const filesystem::path&) const;
    void load_data_binary();

    void set_binary_variables();

    void set_data_random();
    void set_data_integer(const Index vocabulary_size);
    void set_data_rosenbrock();
    void set_data_binary_classification();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void read_csv();

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     bool is_training,
                     bool parallelize = true,
                     int contiguous = -1) const override;

    void fill_decoder(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      bool parallelize = true,
                      int contiguous = -1) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      bool is_training,
                      bool parallelize = true,
                      int contiguous = -1) const override;

protected:

    MatrixR data;

    string missing_values_label = "NA";
    MissingValuesMethod missing_values_method = MissingValuesMethod::Mean;
    Index missing_values_number = 0;
    VectorI variables_missing_values_number;
    Index rows_missing_values_number = 0;

    Index gmt = 0;

    void missing_values_to_JSON(JsonWriter&) const;
    void missing_values_from_JSON(const Json*);

    void infer_variable_types_from_data();
    void resize_data_from_JSON(Index) override;

    void infer_column_types(const vector<vector<string_view>>&);

    vector<Index> filter_used_samples_by_column(Index, bool) const;
    void apply_scaler(Index, const string&, const Descriptives&, bool);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
