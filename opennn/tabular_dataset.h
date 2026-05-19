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

/// @brief Tabular dataset with CSV loading, scaling, descriptive statistics and correlation analysis.
class TabularDataset : public Dataset
{

public:

    /// @brief Strategy for handling missing values in tabular data.
    enum class MissingValuesMethod{Unuse, Mean, Median, Interpolation};

    /// @brief Creates a tabular dataset with the given sample count and input/target shapes.
    /// @param sample_count Number of samples to allocate.
    /// @param input_shape Shape of input features.
    /// @param target_shape Shape of target features.
    TabularDataset(const Index = 0,
                        const Shape& = {0},
                        const Shape& = {0});

    /// @brief Creates a tabular dataset by reading the given file with the given separator.
    /// @param data_path Path to the source CSV/text file.
    /// @param separator Field separator name ("Space", "Tab", "Comma", "Semicolon").
    /// @param has_header Whether the first row contains column names.
    /// @param has_ids Whether the first column contains sample identifiers.
    /// @param codification Text encoding of the file.
    TabularDataset(const filesystem::path&,
                        const string&,
                        bool = true,
                        bool = false,
                        const Codification& = Codification::UTF8);

    using Dataset::set;
    /// @brief Resets the dataset from the given file using the given parsing options.
    /// @param data_path Path to the source CSV/text file.
    /// @param separator Field separator name.
    /// @param has_header Whether the first row contains column names.
    /// @param has_ids Whether the first column contains sample identifiers.
    /// @param codification Text encoding of the file.
    void set(const filesystem::path&,
             const string&,
             bool = true,
             bool = false,
             const Codification& = Codification::UTF8);
    /// @brief Resets the dataset by reading from the given path with default parsing options.
    void set(const filesystem::path&);

    /// @brief Returns the scaler names for variables matching the given role.
    vector<string> get_feature_scalers(const string&) const;

    /// @brief Sets every variable's scaler from the given name.
    void set_variable_scalers(const string&);
    /// @brief Sets each variable's scaler from the corresponding name in the list.
    void set_variable_scalers(const vector<string>&);
    /// @brief Assigns default scalers based on each variable's type.
    void set_default_variable_scalers();

    void set_gmt(const Index new_gmt) { gmt = new_gmt; }

    /// @brief Infers the date/time format used in the given file preview rows.
    /// @param variables Variable descriptors used to locate date columns.
    /// @param sample_rows Preview rows of the source file.
    /// @param has_header Whether the first row is a header.
    /// @param separator Field separator string.
    /// @return Detected date format.
    DateFormat infer_dataset_date_format(const vector<Variable>&, const vector<vector<string_view>>&, bool, const string&);

    MissingValuesMethod get_missing_values_method() const { return missing_values_method; }
    /// @brief Returns the missing-values method as its enumerator name.
    string get_missing_values_method_string() const;
    const string& get_missing_values_label() const { return missing_values_label; }
    Index get_missing_values_number() const { return missing_values_number; }

    void set_missing_values_label(string label) { missing_values_label = move(label); }
    void set_missing_values_method(const MissingValuesMethod& method) { missing_values_method = method; }
    /// @brief Sets the missing-values method from its enumerator name.
    void set_missing_values_method(const string&);

    /// @brief Returns true if any field in the row matches the missing-values label.
    bool has_missing_values(const vector<string_view>&) const;

    /// @brief Removes or imputes missing values using the configured MissingValuesMethod.
    void scrub_missing_values() override;
    /// @brief Refreshes per-variable and per-row missing-value counts.
    void calculate_missing_values_statistics();
    /// @brief Imputes missing values using the given statistic-based method (Mean/Median).
    void impute_missing_values_statistic(const MissingValuesMethod&);
    /// @brief Marks samples containing missing values as unused.
    virtual void impute_missing_values_unuse();
    /// @brief Fills missing values by interpolating between neighboring samples.
    virtual void impute_missing_values_interpolate();

    /// @brief Marks input variables whose absolute correlation with targets is below the threshold as unused.
    /// @return Names of the variables that were unused.
    vector<string> unuse_uncorrelated_variables(const float = 0.25f);
    /// @brief Marks input variables that are highly collinear (above the threshold) as unused.
    /// @return Names of the variables that were unused.
    vector<string> unuse_collinear_variables(const float = 0.95f);

    /// @brief Returns descriptive statistics for every feature across all samples.
    vector<Descriptives> calculate_feature_descriptives() const;
    /// @brief Returns descriptive statistics for features with the given role name.
    vector<Descriptives> calculate_feature_descriptives(const string&) const override;

    /// @brief Returns variable descriptives computed only over samples with positive target.
    vector<Descriptives> calculate_variable_descriptives_positive_samples() const;
    /// @brief Returns variable descriptives computed only over samples with negative target.
    vector<Descriptives> calculate_variable_descriptives_negative_samples() const;
    /// @brief Returns variable descriptives restricted to the given target class index.
    vector<Descriptives> calculate_variable_descriptives_categories(const Index) const;

    /// @brief Returns per-variable histograms with the given bin count.
    vector<Histogram> calculate_variable_distributions(const Index = 10) const;
    /// @brief Returns per-variable Tukey box-plot summaries.
    vector<BoxPlot> calculate_variables_box_plots() const;

    /// @brief Returns the input-input correlation matrix using the given correlation function and method.
    Tensor<Correlation, 2> calculate_input_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), Correlation::Method, const string&) const;
    /// @brief Returns the Pearson correlation matrix among input variables.
    Tensor<Correlation, 2> calculate_input_variable_pearson_correlations() const;
    /// @brief Returns the Spearman correlation matrix among input variables.
    Tensor<Correlation, 2> calculate_input_variable_spearman_correlations() const;

    /// @brief Returns the input-target correlation matrix using the given correlation function.
    Tensor<Correlation, 2> calculate_input_target_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), const string&) const;
    /// @brief Returns the Pearson correlation matrix between input and target variables.
    Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const override;
    /// @brief Returns the Spearman correlation matrix between input and target variables.
    Tensor<Correlation, 2> calculate_input_target_variable_spearman_correlations() const;

    /// @brief Returns input variable indices ranked by absolute correlation with the target.
    VectorI calculate_correlations_rank() const override;

    /// @brief Scales every feature column using its configured scaler; returns the applied descriptives.
    vector<Descriptives> scale_data();
    /// @brief Scales the features with the given role using their configured scalers.
    vector<Descriptives> scale_features(const string&) override;
    /// @brief Reverts the scaling of features with the given role using the supplied descriptives.
    void unscale_features(const string&, const vector<Descriptives>&) override;

    /// @brief Returns the distribution of target classes for classification datasets.
    VectorI calculate_target_distribution() const override;
    /// @brief Detects Tukey outliers per variable using the given fence multiplier.
    /// @param cleaning_parameter Multiplier applied to the interquartile range.
    /// @param use_only_used_samples If true, restrict the analysis to in-use samples.
    /// @return Outlier row indices per variable.
    vector<vector<Index>> calculate_Tukey_outliers(const float = 1.5f, bool = false);
    /// @brief Replaces detected Tukey outliers with NaN and returns the affected indices.
    vector<vector<Index>> replace_Tukey_outliers_with_NaN(const float = 1.5f);
    /// @brief Marks samples containing Tukey outliers as unused.
    void unuse_Tukey_outliers(const float = 1.5f);

    /// @brief Fills the data matrix with random values.
    void set_data_random() override;
    /// @brief Fills the data matrix with random integers up to the given vocabulary size.
    void set_data_integer(const Index vocabulary_size) override;
    /// @brief Fills the data matrix with samples drawn from the Rosenbrock function (regression test data).
    void set_data_rosenbrock();
    /// @brief Fills the data matrix with a synthetic binary classification dataset.
    void set_data_binary_classification();

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    /// @brief Reads the configured CSV file into the dataset, inferring types and headers.
    void read_csv();

protected:

    string missing_values_label = "NA";
    MissingValuesMethod missing_values_method = MissingValuesMethod::Mean;
    Index missing_values_number = 0;
    VectorI variables_missing_values_number;
    Index rows_missing_values_number = 0;

    Index gmt = 0;

    void missing_values_to_JSON(JsonWriter&) const;
    void missing_values_from_JSON(const Json*);

    void infer_column_types(const vector<vector<string_view>>&);

    vector<Index> filter_used_samples_by_column(Index, bool) const;
    void apply_scaler(Index, const string&, const Descriptives&, bool);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
