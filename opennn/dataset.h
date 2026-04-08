//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "correlations.h"
#include "statistics.h"
#include "tensor_utilities.h"
#include "string_utilities.h"
#include "variable.h"

namespace opennn
{

class Dataset
{

public:

    enum class Codification { UTF8, SHIFT_JIS };

    Dataset(const Index = 0,
            const Shape& = {0},
            const Shape& = {0});

    Dataset(const filesystem::path&,
            const string&,
            bool = true,
            bool = false,
            const Codification& = Codification::UTF8);

    // Enumerations

    enum class Separator{Space, Tab, Comma, Semicolon};

    enum class MissingValuesMethod{Unuse, Mean, Median, Interpolation};

    // Samples get

    inline Index get_samples_number() const {return data.rows();}

    Index get_samples_number(const string&) const;

    Index get_used_samples_number() const;

    vector<Index> get_sample_indices(const string&) const;

    vector<Index> get_used_sample_indices() const;

    const vector<string>& get_sample_roles() const { return sample_roles; }

    vector<Index> get_sample_roles_vector() const;

    VectorI get_sample_role_numbers() const;

    inline Index get_variables_number() const { return variables.size(); }
    Index get_variables_number(const string&) const;
    Index get_used_variables_number() const;

    const vector<Variable>& get_variables() const { return variables; }
    vector<Variable> get_variables(const string&) const;

    Index get_variable_index(const string&) const;
    Index get_variable_index(const Index) const;

    vector<Index> get_variable_indices(const string&) const;
    vector<Index> get_used_variables_indices() const;

    vector<string> get_variable_names() const;
    vector<string> get_variable_names(const string&) const;

    VariableType get_variable_type(const Index index) const { return variables[index].type; }

    vector<VariableType> get_variable_types(const vector<Index> indices) const;

    // Variables get

    Index get_features_number() const;
    Index get_features_number(const string&) const;
    Index get_used_features_number() const;

    vector<string> get_feature_names() const;
    vector<string> get_feature_names(const string&) const;

    vector<vector<Index>> get_feature_indices() const;
    vector<Index> get_feature_indices(const Index) const;
    vector<Index> get_feature_indices(const string&) const;
    vector<Index> get_used_feature_indices() const;

    vector<Index> get_feature_dimensions() const;

    Shape get_shape(const string&) const;

    vector<string> get_feature_scalers(const string&) const;

    virtual vector<vector<Index>> get_batches(const vector<Index>&, Index, bool) const;

    const MatrixR& get_data() const { return data; }
    MatrixR get_feature_data(const string&) const;
    MatrixR get_data(const string&, const string&) const;
    MatrixR get_data_from_indices(const vector<Index>&, const vector<Index>&) const;

    VectorR get_sample_data(const Index) const;

    MatrixR get_variable_data(const Index) const;
    MatrixR get_variable_data(const Index, const vector<Index>&) const;
    MatrixR get_variable_data(const string&) const;

    const vector<vector<string>>& get_data_file_preview() const { return data_file_preview; }

    // Members get

    MissingValuesMethod get_missing_values_method() const { return missing_values_method; }
    string get_missing_values_method_string() const;

    const filesystem::path& get_data_path() const { return data_path; }

    const Separator& get_separator() const { return separator; }
    string get_separator_string() const;
    string get_separator_name() const;

    const Codification& get_codification() const { return codification; }
    const string get_codification_string() const;

    const string& get_missing_values_label() const { return missing_values_label; }

    bool get_display() const { return display; }

    bool is_empty() const { return data.size() == 0; }

    Shape get_input_shape() const { return input_shape; }
    Shape get_target_shape() const { return target_shape; }

    // Set

    void set(const Index = 0, const Shape& = {}, const Shape& = {});

    void set(const filesystem::path&,
             const string&,
             bool = true,
             bool = false,
             const Dataset::Codification& = Codification::UTF8);

    void set(const filesystem::path&);

    void set_default();

    // Samples set

    void set_sample_roles(const string&);

    void set_sample_role(const Index, const string&);

    void set_sample_roles(const vector<string>&);
    void set_sample_roles(const vector<Index>&, const string&);

    // Variables set

    void set_variables(const vector<Variable>& v) { variables = v; }

    void set_default_variable_names();

    void set_default_variable_roles();
    void set_default_variable_roles_forecasting();
    virtual void set_variable_roles(const vector<string>&);

    void set_variables(const string&);
    void set_variable_indices(const vector<Index>&, const vector<Index>&);
    void set_input_variables_unused();

    void set_variable_role(const Index, const string&);
    void set_variable_role(const string&, const string&);

    void set_variable_type(const Index, const VariableType&);
    void set_variable_type(const string&, const VariableType&);

    void set_variable_types(const VariableType&);

    void set_variable_names(const vector<string>&);

    void set_variables_number(const Index n) { variables.resize(n); }

    void set_variable_scalers(const string&);

    void set_variable_scalers(const vector<string>&);

    void infer_variable_types_from_data();
    void set_binary_variables();
    void unuse_constant_variables();

    // Variables set

    void set_feature_names(const vector<string>&);

    void set_variable_roles(const string&);

    void set_shape(const string&, const Shape&);

    // Dataset

    void set_data(const MatrixR&);

    // Members set

    void set_data_path(const filesystem::path& p) { data_path = p; }

    void set_has_header(bool h) { has_header = h; }
    void set_has_ids(bool h) { has_sample_ids = h; }

    void set_separator(const Separator& s) { separator = s; }
    void set_separator_string(const string&);
    void set_separator_name(const string&);

    void set_codification(const Codification& c) { codification = c; }
    void set_codification(const string&);

    void set_missing_values_label(const string& l) { missing_values_label = l; }
    void set_missing_values_method(const MissingValuesMethod& m) { missing_values_method = m; }
    void set_missing_values_method(const string&);

    void set_gmt(const Index g) { gmt = g; }

    void set_display(bool d) { display = d; }

    bool is_sample_used(const Index i) const { return sample_roles[i] != "None"; }

    bool has_binary_variables() const;
    bool has_categorical_variables() const;
    bool has_binary_or_categorical_variables() const;
    bool has_time_variable() const;

    bool has_validation() const;

    bool has_missing_values(const vector<string>&) const;

    // Splitting

    void split_samples(const type training_ratio = type(0.6),
                       type selection_ratio = type(0.2),
                       type testing_ratio = type(0.2),
                       bool shuffle = true);

    void split_samples_sequential(const type training_ratio = type(0.6),
                                  type selection_ratio = type(0.2),
                                  type testing_ratio = type(0.2));

    void split_samples_random(const type training_ratio = type(0.6),
                              type selection_ratio = type(0.2),
                              type testing_ratio = type(0.2));

    // Unusing

    vector<string> unuse_uncorrelated_variables(const type = type(0.25));
    vector<string> unuse_collinear_variables(const type = type(0.95));

    // Initialization

    void set_data_constant(const type);

    // Descriptives

    vector<Descriptives> calculate_feature_descriptives() const;

    vector<Descriptives> calculate_variable_descriptives_positive_samples() const;
    vector<Descriptives> calculate_variable_descriptives_negative_samples() const;
    vector<Descriptives> calculate_variable_descriptives_categories(const Index) const;

    vector<Descriptives> calculate_feature_descriptives(const string&) const;

    // Distribution

    vector<Histogram> calculate_variable_distributions(const Index = 10) const;

    // Box plots

    vector<BoxPlot> calculate_variables_box_plots() const;

    // Inputs correlations

    Tensor<Correlation, 2> calculate_input_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), Correlation::Method, const string&) const;

    Tensor<Correlation, 2> calculate_input_variable_pearson_correlations() const;

    Tensor<Correlation, 2> calculate_input_variable_spearman_correlations() const;

    // Input-target correlations

    Tensor<Correlation, 2> calculate_input_target_variable_correlations(
        Correlation (*)(const MatrixR&, const MatrixR&), const string&) const;

    Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const;
    Tensor<Correlation, 2> calculate_input_target_variable_spearman_correlations() const;

    VectorI calculate_correlations_rank() const;

    // Scaling2d

    void set_default_variable_scalers();

    // Data scaling

    vector<Descriptives> scale_data();

    virtual vector<Descriptives> scale_features(const string&);

    // Data unscaling

    void unscale_features(const string&, const vector<Descriptives>&);

    // Classification

    VectorI calculate_target_distribution() const;

    // Tuckey outlier detection

    vector<vector<Index>> calculate_Tukey_outliers(const type = type(1.5), bool = false);

    vector<vector<Index>> replace_Tukey_outliers_with_NaN(const type = type(1.5));

    void unuse_Tukey_outliers(const type = type(1.5));

    // Data generation

    virtual void set_data_random();
    virtual void set_data_integer(const Index vocabulary_size);
    void set_data_rosenbrock();
    void set_data_binary_classification();

    // Serialization

    virtual void from_XML(const XMLDocument&);
    virtual void to_XML(XMLPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    void save_data() const;

    void save_data_binary(const filesystem::path&) const;

    void load_data_binary();

    // Missing values

    inline Index get_missing_values_number() const { return missing_values_number; }

    bool has_nan() const;

    bool has_nan_row(const Index) const;

    virtual void impute_missing_values_unuse();
    void impute_missing_values_statistic(const MissingValuesMethod&);
    virtual void impute_missing_values_interpolate();

    void scrub_missing_values();
    void calculate_missing_values_statistics();

    VectorI count_nans_per_variable() const;
    Index count_variables_with_nan() const;
    Index count_rows_with_nan() const;
    Index count_nan() const;

    // Eigen

    vector<vector<Index>> split_samples(const vector<Index>&, Index) const;

    //bool get_has_text_data() const;

    // Reader

    //void decode(string&) const;

    virtual void read_csv();

    DateFormat infer_dataset_date_format(const vector<Variable>&, const vector<vector<string>>&, bool, const string&);

    virtual void fill_inputs(const vector<Index>&,
                             const vector<Index>&,
                             type*,
                             bool = true) const;

    virtual void augment_inputs(type*, Index) const {}

    virtual void fill_decoder(const vector<Index>&,
                              const vector<Index>&,
                              type*,
                              bool = true) const;

    virtual void fill_targets(const vector<Index>&,
                              const vector<Index>&,
                              type*,
                              bool = true) const;

private:

    string get_sample_role(const Index i) const { return sample_roles[i]; }
    vector<Index> filter_used_samples_by_column(Index, bool) const;
    void apply_scaler(Index, const string&, const Descriptives&, bool);
    void infer_column_types(const vector<vector<string>>&);
    void read_data_file_preview(const vector<vector<string>>&);
    void check_separators(const string&) const;

protected:

    // DATA

    MatrixR data;

    // Dimensions

    Shape input_shape;
    Shape target_shape;
    Shape decoder_shape;

    // Samples

    vector<string> sample_roles;

    vector<string> sample_ids;

    // Variables

    vector<Variable> variables;

    // Data File

    filesystem::path data_path;

    Separator separator = Separator::Comma;

    string missing_values_label = "NA";

    //VectorB nans_variables;

    bool has_header = false;

    bool has_sample_ids = false;

    Codification codification = Codification::UTF8;

    vector<vector<string>> data_file_preview;

    Index gmt = 0;

    // Missing Values

    MissingValuesMethod missing_values_method = MissingValuesMethod::Mean;

    Index missing_values_number = 0;

    VectorI variables_missing_values_number;

    Index rows_missing_values_number = 0;

    // Display

    bool display = true;

    const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
    const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};

    void variables_to_XML(XMLPrinter&) const;
    void samples_to_XML(XMLPrinter&) const;
    void missing_values_to_XML(XMLPrinter&) const;
    void preview_data_to_XML(XMLPrinter&) const;

    void variables_from_XML(const XMLElement*);
    void samples_from_XML(const XMLElement*);
    void missing_values_from_XML(const XMLElement*);
    void preview_data_from_XML(const XMLElement*);
};

}

#define STRINGIFY_ENUM(x) #x

#define ENUM_TO_STRING(x) STRINGIFY_ENUM(x)

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
