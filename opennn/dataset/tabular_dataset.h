//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T A B U L A R   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/dataset/dataset.h"
#include "opennn/dataset/correlations.h"
#include "opennn/core/statistics.h"
#include "opennn/core/io_utilities.h"
#include "opennn/dataset/field_parsing.h"

#include <utility>

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
                   bool has_header = true,
                   bool has_sample_ids = false,
                   const Codification& = Codification::UTF8);

    Index get_samples_number() const noexcept override
    {
        return storage_mode == StorageMode::BinaryFile
             ? ssize(sample_roles)
             : data.rows();
    }
    using Dataset::get_samples_number;

    using Dataset::get_data;
    MatrixR get_data(const string& sample_role, const string& variable_role) const { return get_data_from_indices(get_sample_indices(sample_role), get_feature_indices(variable_role)); }
    MatrixR get_data_from_indices(const vector<Index>&, const vector<Index>&) const;

    MatrixR get_variable_data(Index) const;
    MatrixR get_variable_data(Index, const vector<Index>&) const;

    MatrixR get_feature_data(const string&) const;

    void set(Index = 0, const Shape& = {}, const Shape& = {});
    void set(const filesystem::path&,
             const string&,
             bool new_has_header = true,
             bool new_has_ids = false,
             const Codification& = Codification::UTF8);
    void set(const filesystem::path&);

    using Dataset::set_storage_mode;
    void set_storage_mode(StorageMode) override;

    void set_binary_cache_path(const filesystem::path&);

    vector<string> get_feature_scalers(const string&) const;

    void set_variable_scalers(const string&);
    void set_variable_scalers(const vector<string>&);
    void set_default_variable_scalers();

    MissingValuesMethod get_missing_values_method() const { return missing_values_method; }
    string get_missing_values_method_string() const;
    const string& get_missing_values_label() const { return missing_values_label; }
    Index get_missing_values_number() const { return missing_values_number; }

    const NumberFormat& get_number_format() const { return number_format; }

    void set_number_format(const NumberFormat& new_number_format)
    {
        number_format = new_number_format;
        number_format_automatic = false;
    }

    void set_number_format_auto() { number_format = {}; number_format_automatic = true; }

    void set_missing_values_label(string label) { missing_values_label = std::move(label); }
    void set_missing_values_method(const MissingValuesMethod& method) { missing_values_method = method; }
    void set_missing_values_method(const string&);

    void scrub_missing_values() override;
    void calculate_missing_values_statistics();
    void impute_missing_values_statistic(const MissingValuesMethod&);
    void reuse_input_incomplete_rows_binary();
    virtual void impute_missing_values_unuse();
    virtual void impute_missing_values_interpolate();

    vector<string> unuse_uncorrelated_variables(const float = 0.25f);
    vector<string> unuse_collinear_variables(const float = 0.95f);
    vector<string> unuse_least_correlated_variables(const Index inputs_to_keep);

    vector<Descriptives> calculate_feature_descriptives() const;
    vector<Descriptives> calculate_feature_descriptives(const string&) const;
    vector<Descriptives> calculate_feature_descriptives(const string&, const vector<Index>&) const;

    vector<Descriptives> calculate_variable_descriptives_positive_samples() const { return calculate_variable_descriptives_samples(true); }
    vector<Descriptives> calculate_variable_descriptives_negative_samples() const { return calculate_variable_descriptives_samples(false); }
    vector<Descriptives> calculate_variable_descriptives_categories(Index) const;

    vector<Histogram> calculate_variable_distributions(const Index = 10) const;
    vector<BoxPlot> calculate_variables_box_plots() const;

    Tensor<Correlation, 2> calculate_input_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), Correlation::Method, const string&) const;
    Tensor<Correlation, 2> calculate_input_variable_pearson_correlations() const { return calculate_input_variable_correlations(correlation, Correlation::Method::Pearson, "pearson"); }
    Tensor<Correlation, 2> calculate_input_variable_spearman_correlations() const { return calculate_input_variable_correlations(correlation_spearman, Correlation::Method::Spearman, "spearman"); }

    Tensor<Correlation, 2> calculate_input_target_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), const string&) const;
    Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const { return calculate_input_target_variable_correlations(correlation, "pearson"); }
    Tensor<Correlation, 2> calculate_input_target_variable_spearman_correlations() const { return calculate_input_target_variable_correlations(correlation_spearman, "spearman"); }

    MatrixR calculate_input_target_correlation_values() const override { return get_correlation_values(calculate_input_target_variable_pearson_correlations()); }

    FeatureScaling calculate_used_feature_scaling(VariableRole) const override;

    vector<Descriptives> scale_data();
    vector<Descriptives> scale_features(const string&);
    void unscale_features(const string&, const vector<Descriptives>&);

    FeatureScaling prepare_training_scaling(
        VariableRole,
        const FeatureScaling&,
        Index) override;
    void clear_training_scaling() noexcept override;
    void enable_device_residency() override;

    VectorI calculate_target_distribution() const override;
    vector<vector<Index>> calculate_Tukey_outliers(float cleaning_parameter = 1.5f, bool replace_with_nan = false);
    vector<vector<Index>> replace_Tukey_outliers_with_NaN(const float = 1.5f);

    bool has_nan() const override;
    bool has_nan_row(Index row_index) const { return data.row(row_index).array().isNaN().any(); }

    VectorI count_nans_per_variable() const;
    Index count_rows_with_nan() const { return data.array().isNaN().rowwise().any().count(); }
    Index count_nan() const { return data.array().isNaN().count(); }

    void set_data_random();
    void set_data_integer(const Index);
    void set_data_binary_classification();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void read_csv();

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     float*,
                     FillMode,
                     ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    void fill_decoder(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      FillMode,
                      ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    void fill_targets(const vector<Index>&,
                      const vector<Index>&,
                      float*,
                      FillMode,
                      ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

protected:

    string missing_values_label = "NA";
    NumberFormat number_format;
    bool number_format_automatic = true;
    MissingValuesMethod missing_values_method = MissingValuesMethod::Mean;
    Index missing_values_number = 0;
    VectorI variables_missing_values_number;
    Index rows_missing_values_number = 0;

    void missing_values_to_JSON(JsonWriter&) const;
    void missing_values_from_JSON(const Json*) override;

    void resize_data_from_JSON(Index) override;

    filesystem::path cache_file_path() const;

    void fill_features(const vector<Index>&,
                       const vector<Index>&,
                       float*,
                       ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const;

    void fill_from_binary_cache(const vector<Index>&,
                                const vector<Index>&,
                                float*,
                                ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const;

    float apply_training_scaling(Index, float) const;
    void apply_training_scaling(const vector<Index>&,
                                float*,
                                Index) const;

    vector<Index> filter_used_samples_by_column(Index column, bool positive) const;

    vector<Descriptives> calculate_variable_descriptives_samples(bool positive) const;

    vector<ScalerMethod> get_feature_scaler_methods() const;
    vector<ScalerMethod> get_feature_scaler_methods(VariableRole) const;

    void unuse_samples_with_missing_targets(const vector<Index>&, const vector<Index>&);

    void on_used_samples_changed() override;
    void clear_cache_derived_state();
    void refresh_cache_statistics();
    vector<Descriptives> compute_descriptives_streaming(const vector<Index>&) const;

    DateFormat configure_csv_columns(vector<string_view>&, char, bool);
    void load_csv_data(const vector<string_view>&, char, bool, DateFormat);

    filesystem::path cache_path;
    filesystem::path cache_path_override;
    mutable FileReader cache_reader;
    Index cache_columns_number = 0;

    vector<Descriptives> cache_feature_descriptives;
    vector<Descriptives> cache_transform_descriptives;
    vector<float> cache_feature_replacement;
    vector<ScalerMethod> cache_feature_transforms;

    struct TrainingTransform
    {
        Descriptives descriptives;
        ScalerMethod scaler = ScalerMethod::None;
        float min_range = -1.0f;
        float max_range = 1.0f;
        bool configured = false;

        float apply(float value) const
        {
            return configured
                ? scale_value(scaler, descriptives, value, min_range, max_range)
                : value;
        }
    };

    vector<TrainingTransform> training_transforms;

    DateFormat infer_column_types(const vector<string_view>&, char, bool has_quotes = false);

    void apply_scaler(Index feature_index, ScalerMethod scaler, const Descriptives& descriptives, bool unscale);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
