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
#include "enum_map.h"
#include "string_utilities.h"
#include "variable.h"

namespace opennn
{

/// @brief Role of a sample in a dataset split.
enum class SampleRole
{
    Training,
    Validation,
    Testing,
    None
};

/// @brief Returns the bidirectional string/enum map for SampleRole.
[[nodiscard]] inline const EnumMap<SampleRole>& sample_role_map()
{
    static const vector<pair<SampleRole, string>> entries = {
        {SampleRole::Training,   "Training"},
        {SampleRole::Validation, "Validation"},
        {SampleRole::Testing,    "Testing"},
        {SampleRole::None,       "None"}
    };
    static const EnumMap<SampleRole> map{entries};
    return map;
}

/// @brief Returns the canonical string name for a SampleRole.
[[nodiscard]] inline const string& sample_role_to_string(SampleRole role)
{
    return sample_role_map().to_string(role);
}

/// @brief Parses a string (name or "0"/"1"/"2"/"3") into the matching SampleRole.
[[nodiscard]] inline SampleRole string_to_sample_role(const string& name)
{
    if (name == "0") return SampleRole::Training;
    if (name == "1") return SampleRole::Validation;
    if (name == "2") return SampleRole::Testing;
    if (name == "3") return SampleRole::None;
    return sample_role_map().from_string(name);
}

/// @brief Abstract base class for OpenNN datasets, owning samples, variables, and metadata.
class Dataset
{

public:

    /// @brief Text encoding of the source data file.
    enum class Codification { UTF8, SHIFT_JIS };

    virtual ~Dataset() = default;

    /// @brief Field separator used when reading delimited text files.
    enum class Separator{Space, Tab, Comma, Semicolon};

    /// @brief Returns the total number of samples (rows) in the data matrix.
    [[nodiscard]] virtual Index get_samples_number() const { return data.rows(); }

    /// @brief Returns the number of samples with the given role ("Training", "Validation", ...).
    [[nodiscard]] Index get_samples_number(const string&) const;

    /// @brief Returns the number of samples whose role is not None.
    [[nodiscard]] Index get_used_samples_number() const;

    /// @brief Returns indices of samples with the given role name.
    [[nodiscard]] vector<Index> get_sample_indices(const string&) const;

    /// @brief Returns indices of all samples whose role is not None.
    [[nodiscard]] vector<Index> get_used_sample_indices() const;

    /// @brief Returns the per-sample role vector.
    [[nodiscard]] const vector<SampleRole>& get_sample_roles() const { return sample_roles; }

    /// @brief Returns the per-sample roles as integer indices.
    [[nodiscard]] vector<Index> get_sample_roles_vector() const;

    /// @brief Returns the per-sample roles as a tensor of integer indices.
    [[nodiscard]] VectorI get_sample_role_numbers() const;

    /// @brief Returns the total number of variables (columns descriptors).
    [[nodiscard]] Index get_variables_number() const { return variables.size(); }
    /// @brief Returns the number of variables with the given role name.
    [[nodiscard]] Index get_variables_number(const string&) const;
    /// @brief Returns the number of variables whose role is in use.
    [[nodiscard]] Index get_used_variables_number() const;

    /// @brief Returns the variable descriptors.
    [[nodiscard]] const vector<Variable>& get_variables() const { return variables; }
    /// @brief Returns the variables with the given role name.
    [[nodiscard]] vector<Variable> get_variables(const string&) const;

    /// @brief Returns the index of the variable with the given name.
    [[nodiscard]] Index get_variable_index(const string&) const;
    /// @brief Returns the variable index corresponding to a flat feature index.
    [[nodiscard]] Index get_variable_index(const Index) const;

    /// @brief Returns the indices of variables matching the given role name.
    [[nodiscard]] vector<Index> get_variable_indices(const string&) const;
    /// @brief Returns the indices of all in-use variables.
    [[nodiscard]] vector<Index> get_used_variables_indices() const;

    /// @brief Returns the names of all variables.
    [[nodiscard]] vector<string> get_variable_names() const;
    /// @brief Returns the names of variables with the given role name.
    [[nodiscard]] vector<string> get_variable_names(const string&) const;

    /// @brief Returns the VariableType of the variable at the given index.
    [[nodiscard]] VariableType get_variable_type(const Index index) const { return variables[index].type; }

    /// @brief Returns the VariableType for each of the given variable indices.
    [[nodiscard]] vector<VariableType> get_variable_types(const vector<Index>& indices) const;
    /// @brief Returns the total number of features (data matrix columns).
    [[nodiscard]] Index get_features_number() const;
    /// @brief Returns the number of features for variables with the given role name.
    [[nodiscard]] Index get_features_number(const string&) const;
    /// @brief Returns the number of features for in-use variables.
    [[nodiscard]] Index get_used_features_number() const;

    /// @brief Returns the expanded feature names (one entry per data matrix column).
    [[nodiscard]] vector<string> get_feature_names() const;
    /// @brief Returns the expanded feature names restricted to the given role.
    [[nodiscard]] vector<string> get_feature_names(const string&) const;

    /// @brief Returns the data column indices grouped per variable.
    [[nodiscard]] vector<vector<Index>> get_feature_indices() const;
    /// @brief Returns the data column indices that belong to the given variable index.
    [[nodiscard]] vector<Index> get_feature_indices(const Index) const;
    /// @brief Returns the data column indices that belong to variables with the given role name.
    [[nodiscard]] vector<Index> get_feature_indices(const string&) const;
    /// @brief Returns the data column indices that belong to in-use variables.
    [[nodiscard]] vector<Index> get_used_feature_indices() const;

    /// @brief Returns the per-variable feature dimension counts.
    [[nodiscard]] vector<Index> get_feature_dimensions() const;

    /// @brief Returns the configured Shape for the given role ("Input", "Target", "Decoder").
    [[nodiscard]] Shape get_shape(const string&) const;

    /// @brief Splits sample indices into batches and writes them into @p batches.
    /// @param indices Sample indices to draw from.
    /// @param batch_size Number of samples per batch.
    /// @param shuffle Whether to shuffle indices before batching.
    /// @param batches Output vector receiving the batched index lists.
    virtual void get_batches(const vector<Index>&, Index, bool, vector<vector<Index>>&) const;

    /// @brief Returns the parsed preview rows captured during the last file read.
    [[nodiscard]] const vector<vector<string>>& get_data_file_preview() const { return data_file_preview; }

    /// @brief Returns the configured data file path.
    [[nodiscard]] const filesystem::path& get_data_path() const { return data_path; }

    /// @brief Returns the current field Separator.
    [[nodiscard]] const Separator& get_separator() const { return separator; }
    /// @brief Returns the separator as the literal character used in files.
    [[nodiscard]] string get_separator_string() const;
    /// @brief Returns the separator as its enumerator name ("Space", "Tab", ...).
    [[nodiscard]] string get_separator_name() const;

    /// @brief Returns the configured text Codification.
    [[nodiscard]] const Codification& get_codification() const { return codification; }
    /// @brief Returns the codification as its enumerator name.
    [[nodiscard]] string get_codification_string() const;

    /// @brief Returns whether progress messages are printed.
    [[nodiscard]] bool get_display() const { return display; }

    /// @brief Returns true when the dataset contains no samples.
    [[nodiscard]] virtual bool is_empty() const { return get_samples_number() == 0; }

    /// @brief Returns the configured input tensor shape.
    [[nodiscard]] Shape get_input_shape() const { return input_shape; }
    /// @brief Returns the configured target tensor shape.
    [[nodiscard]] Shape get_target_shape() const { return target_shape; }

    /// @brief Returns the raw data matrix (rows = samples, columns = features).
    [[nodiscard]] const MatrixR& get_data() const { return data; }
    /// @brief Replaces the underlying data matrix.
    void set_data(const MatrixR&);
    /// @brief Fills the data matrix with the given constant value.
    void set_data_constant(const float);

    /// @brief Restores default dataset settings.
    void set_default();
    /// @brief Assigns the same role to every sample.
    void set_sample_roles(const string&);

    /// @brief Sets the role of a single sample by index.
    void set_sample_role(const Index, const string&);

    /// @brief Assigns a role per sample from a string vector.
    void set_sample_roles(const vector<string>&);
    /// @brief Assigns the same role to the given sample indices.
    void set_sample_roles(const vector<Index>&, const string&);
    void set_variables(const vector<Variable>& new_variables) { variables = new_variables; }

    /// @brief Assigns default placeholder names to all variables.
    void set_default_variable_names();

    /// @brief Sets the role of every variable from the given string list.
    void set_variable_roles(const vector<string>&);

    /// @brief Sets which variable indices are inputs and which are targets.
    /// @param input_indices Indices for variables to mark as Input.
    /// @param target_indices Indices for variables to mark as Target.
    void set_variable_indices(const vector<Index>&, const vector<Index>&);
    /// @brief Marks all input variables as unused (role None).
    void set_input_variables_unused();

    /// @brief Sets the role of the variable at the given index.
    void set_variable_role(const Index, const string&);
    /// @brief Sets the role of the variable with the given name.
    void set_variable_role(const string&, const string&);

    /// @brief Sets the type of the variable at the given index.
    void set_variable_type(const Index, const VariableType&);
    /// @brief Sets the type of the variable with the given name.
    void set_variable_type(const string&, const VariableType&);

    /// @brief Sets every variable to the given type.
    void set_variable_types(const VariableType&);
    /// @brief Detects and marks variables with binary values as VariableType::Binary.
    void set_binary_variables();

    /// @brief Assigns names to all variables from the given list.
    void set_variable_names(const vector<string>&);

    void set_variables_number(const Index new_size) { variables.resize(new_size); }

    /// @brief Assigns expanded feature names, propagating categories back to variables.
    void set_feature_names(const vector<string>&);

    /// @brief Assigns the same role to every variable.
    void set_variable_roles(const string&);

    /// @brief Sets the tensor Shape associated with the given role ("Input"/"Target"/"Decoder").
    void set_shape(const string&, const Shape&);
    /// @brief Resizes the input shape to the given flat feature count.
    virtual void resize_input_shape(Index input_features_count) { set_shape("Input", {input_features_count}); }
    void set_data_path(const filesystem::path& new_data_path) { data_path = new_data_path; }

    void set_has_header(bool new_has_header) { has_header = new_has_header; }
    void set_has_ids(bool new_has_ids) { has_sample_ids = new_has_ids; }

    void set_separator(const Separator& new_separator) { separator = new_separator; }
    /// @brief Sets the separator from its literal character.
    void set_separator_string(const string&);
    /// @brief Sets the separator from its enumerator name.
    void set_separator_name(const string&);

    void set_codification(const Codification& new_codification) { codification = new_codification; }
    /// @brief Sets the codification from its enumerator name.
    void set_codification(const string&);

    void set_display(bool new_display) { display = new_display; }

    /// @brief Returns true if the sample at @p i has a role other than None.
    [[nodiscard]] bool is_sample_used(const Index i) const { return sample_roles[i] != SampleRole::None; }

    /// @brief Returns true if any variable has type Binary.
    [[nodiscard]] bool has_binary_variables() const;
    /// @brief Returns true if any variable has type Categorical.
    [[nodiscard]] bool has_categorical_variables() const;
    /// @brief Returns true if any variable is Binary or Categorical.
    [[nodiscard]] bool has_binary_or_categorical_variables() const;
    /// @brief Returns true if any variable has role Time.
    [[nodiscard]] bool has_time_variable() const;

    /// @brief Returns true if any sample is assigned the Validation role.
    [[nodiscard]] bool has_validation() const;

    /// @brief Splits samples into Training/Validation/Testing roles, optionally shuffled.
    /// @param training_ratio Fraction of samples assigned to Training.
    /// @param selection_ratio Fraction of samples assigned to Validation.
    /// @param testing_ratio Fraction of samples assigned to Testing.
    /// @param shuffle If true, shuffles samples before splitting.
    void split_samples(const float training_ratio = 0.6f,
                       float selection_ratio = 0.2f,
                       float testing_ratio = 0.2f,
                       bool shuffle = true);

    /// @brief Splits samples sequentially without shuffling.
    void split_samples_sequential(const float training_ratio = 0.6f,
                                  float selection_ratio = 0.2f,
                                  float testing_ratio = 0.2f);

    /// @brief Splits samples randomly across roles.
    void split_samples_random(const float training_ratio = 0.6f,
                              float selection_ratio = 0.2f,
                              float testing_ratio = 0.2f);

    /// @brief Splits the given indices into chunks of the requested size.
    [[nodiscard]] vector<vector<Index>> split_samples(const vector<Index>&, Index) const;

    /// @brief Scales the features with the configured method; returns the applied descriptives.
    virtual vector<Descriptives> scale_features(const string&) { return {}; }

    /// @brief Fills the data matrix with random values (no-op in base class).
    virtual void set_data_random() {}
    /// @brief Fills the data matrix with random integers up to the given vocabulary size.
    virtual void set_data_integer(const Index) {}

    /// @brief Loads dataset state from a JSON document.
    virtual void from_JSON(const JsonDocument&) = 0;
    /// @brief Writes dataset state to a JSON writer.
    virtual void to_JSON(JsonWriter&) const {}

    /// @brief Returns the data submatrix for the given sample role and feature role.
    [[nodiscard]] MatrixR get_data(const string&, const string&) const;
    /// @brief Returns a submatrix from the data using explicit row and column indices.
    [[nodiscard]] MatrixR get_data_from_indices(const vector<Index>&, const vector<Index>&) const;

    /// @brief Returns the row of the data matrix at the given sample index.
    [[nodiscard]] VectorR get_sample_data(const Index) const;

    /// @brief Returns the data columns belonging to the variable at @p index.
    [[nodiscard]] MatrixR get_variable_data(const Index) const;
    /// @brief Returns the variable data restricted to the given sample indices.
    [[nodiscard]] MatrixR get_variable_data(const Index, const vector<Index>&) const;
    /// @brief Returns the data columns of the variable with the given name.
    [[nodiscard]] MatrixR get_variable_data(const string&) const;

    /// @brief Returns the data columns belonging to features with the given role name.
    [[nodiscard]] MatrixR get_feature_data(const string&) const;

    /// @brief Resets the dataset with the given sample count and input/target shapes.
    /// @param sample_count Number of samples to allocate.
    /// @param input_shape Shape of input features.
    /// @param target_shape Shape of target features.
    void set(const Index = 0, const Shape& = {}, const Shape& = {});

    /// @brief Returns true if any entry in the data matrix is NaN.
    [[nodiscard]] bool has_nan() const;
    /// @brief Returns true if the sample at the given row contains a NaN.
    [[nodiscard]] bool has_nan_row(const Index) const;

    /// @brief Returns the NaN count for each variable.
    [[nodiscard]] VectorI count_nans_per_variable() const;
    /// @brief Returns the number of variables that contain at least one NaN.
    [[nodiscard]] Index count_variables_with_nan() const;
    /// @brief Returns the number of rows that contain at least one NaN.
    [[nodiscard]] Index count_rows_with_nan() const;
    /// @brief Returns the total NaN count in the data matrix.
    [[nodiscard]] Index count_nan() const;

    /// @brief Removes or imputes missing values using the configured strategy.
    virtual void scrub_missing_values() {}

    /// @brief Returns descriptive statistics for features with the given role (subclass-specific).
    [[nodiscard]] virtual vector<Descriptives> calculate_feature_descriptives(const string&) const { return {}; }
    /// @brief Returns the Pearson correlations between input and target variables.
    [[nodiscard]] virtual Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const { return {}; }
    /// @brief Returns the distribution of target values.
    [[nodiscard]] virtual VectorI calculate_target_distribution() const { return {}; }
    /// @brief Returns input variables ranked by absolute correlation with the target.
    [[nodiscard]] virtual VectorI calculate_correlations_rank() const { return {}; }

    /// @brief Reverts a previously applied scaling using the supplied descriptives.
    virtual void unscale_features(const string&, const vector<Descriptives>&) {}

    /// @brief Saves the dataset metadata to a JSON file at the given path.
    void save(const filesystem::path&) const;
    /// @brief Loads the dataset metadata from the JSON file at the given path.
    void load(const filesystem::path&);

    /// @brief Saves the data matrix to the configured data path.
    void save_data() const;
    /// @brief Saves the data matrix to the given path in binary form.
    void save_data_binary(const filesystem::path&) const;
    /// @brief Loads the data matrix from the binary file at the configured data path.
    void load_data_binary();

    /// @brief Copies input features of the given samples into a destination buffer.
    /// @param sample_indices Row indices to read.
    /// @param variable_indices Variable indices selecting which features to copy.
    /// @param destination Output pointer that receives the flattened tensor.
    /// @param is_training True when the call is part of a training batch.
    /// @param parallelize If true, parallelize the copy.
    /// @param contiguous Stride hint for contiguous copies (-1 = auto).
    virtual void fill_inputs(const vector<Index>&,
                             const vector<Index>&,
                             float*,
                             bool is_training,
                             bool parallelize = true,
                             int contiguous = -1) const;

    /// @brief Applies data augmentation in place to the input buffer (no-op in base class).
    virtual void augment_inputs(float*, Index) const {}

    /// @brief Copies decoder input features into a destination buffer (sequence models).
    virtual void fill_decoder(const vector<Index>&,
                              const vector<Index>&,
                              float*,
                              bool is_training,
                              bool parallelize = true,
                              int contiguous = -1) const;

    /// @brief Copies target features of the given samples into a destination buffer.
    virtual void fill_targets(const vector<Index>&,
                              const vector<Index>&,
                              float*,
                              bool is_training,
                              bool parallelize = true,
                              int contiguous = -1) const;

protected:

    Dataset() = default;

    void set_default_variable_roles();
    void set_default_variable_roles_forecasting();

    void infer_variable_types_from_data();
    void read_data_file_preview(const vector<vector<string_view>>&);
    void check_separators(string_view) const;
    void samples_from_JSON(const Json*);

    Shape input_shape;
    Shape target_shape;
    Shape decoder_shape;

    vector<SampleRole> sample_roles;
    vector<string> sample_ids;

    vector<Variable> variables;

    MatrixR data;

    filesystem::path data_path;
    Separator separator = Separator::Comma;
    bool has_header = false;
    bool has_sample_ids = false;
    Codification codification = Codification::UTF8;
    vector<vector<string>> data_file_preview;

    bool display = true;

    const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
    const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};

    void variables_to_JSON(JsonWriter&) const;
    void samples_to_JSON(JsonWriter&) const;
    void preview_data_to_JSON(JsonWriter&) const;

    void variables_from_JSON(const Json*);
    void preview_data_from_JSON(const Json*);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
