//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file dataset.h
 * @brief Declares the Dataset class and the SampleRole enum.
 *
 * Dataset is the base data container for OpenNN. Specialized subclasses
 * (TabularDataset, ImageDataset, LanguageDataset, TimeSeriesDataset) extend
 * it for specific data formats; this header provides the shared abstraction.
 */

#pragma once

#include "correlations.h"
#include "statistics.h"
#include "tensor_utilities.h"
#include "enum_map.h"
#include "string_utilities.h"
#include "variable.h"

namespace opennn
{

/**
 * @enum SampleRole
 * @brief Role of a sample within a dataset partition.
 *
 * - Training: used to fit the network parameters.
 * - Validation: used to monitor generalization during training.
 * - Testing: held out for the final evaluation.
 * - None: the sample is ignored (e.g. dropped due to missing values).
 */
enum class SampleRole
{
    Training,
    Validation,
    Testing,
    None
};

/**
 * @brief Returns the string<->enum mapping for SampleRole values.
 * @return Singleton EnumMap initialized once.
 */
inline const EnumMap<SampleRole>& sample_role_map()
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

/**
 * @brief Converts a SampleRole to its canonical string name.
 * @param role The sample role.
 * @return Canonical string ("Training", "Validation", "Testing", "None").
 */
inline const string& sample_role_to_string(SampleRole role)
{
    return sample_role_map().to_string(role);
}

/**
 * @brief Parses a SampleRole from string.
 *
 * Accepts both the canonical names and the numeric encodings ("0", "1", "2", "3").
 *
 * @param name String to parse.
 * @return Matching SampleRole; throws if the string is unrecognized.
 */
inline SampleRole string_to_sample_role(const string& name)
{
    if (name == "0") return SampleRole::Training;
    if (name == "1") return SampleRole::Validation;
    if (name == "2") return SampleRole::Testing;
    if (name == "3") return SampleRole::None;
    return sample_role_map().from_string(name);
}

/**
 * @class Dataset
 * @brief Base data container with samples, variables and per-variable metadata.
 *
 * Owns a dense matrix of values (rows are samples, columns are variables) plus
 * parallel metadata: per-sample role, per-variable Variable description
 * (role, type, scaler), input/target shapes, missing-value handling, source
 * file metadata.
 *
 * Provides utilities for loading from CSV, splitting into Training /
 * Validation / Testing partitions, scaling and unscaling features,
 * descriptive statistics, correlation analysis and Tukey-based outlier
 * handling.
 *
 * Specialized data formats are implemented by deriving from this class:
 * TabularDataset, ImageDataset, LanguageDataset, TimeSeriesDataset.
 */
class Dataset
{

public:

    /**
     * @enum Codification
     * @brief Source-file character encoding.
     */
    enum class Codification { UTF8, SHIFT_JIS };

    /**
     * @brief Constructs an empty dataset of given dimensions.
     * @param samples_number Number of samples (rows).
     * @param input_shape Shape of the input portion (columns assigned the "Input" role).
     * @param target_shape Shape of the target portion (columns assigned the "Target" role).
     */
    Dataset(const Index samples_number = 0,
            const Shape& input_shape = {0},
            const Shape& target_shape = {0});

    /**
     * @brief Constructs a dataset by loading a delimited text file.
     * @param data_path Path to the source file.
     * @param separator Field separator string ("," ";" "\t" or " ").
     * @param has_header Whether the first row contains column names.
     * @param has_ids Whether the first column contains sample identifiers.
     * @param codification Source-file character encoding.
     */
    Dataset(const filesystem::path& data_path,
            const string& separator,
            bool has_header = true,
            bool has_ids = false,
            const Codification& codification = Codification::UTF8);

    // Enumerations

    /**
     * @enum Separator
     * @brief Field-separator type for tabular files.
     */
    enum class Separator{Space, Tab, Comma, Semicolon};

    /**
     * @enum MissingValuesMethod
     * @brief Strategy for replacing missing values.
     */
    enum class MissingValuesMethod{Unuse, Mean, Median, Interpolation};

    /**
     * @brief Returns the total number of samples (rows of @p data).
     * @return Sample count.
     */
    Index get_samples_number() const {return data.rows();}

    /**
     * @brief Returns the number of samples assigned to a given role.
     * @param role_name "Training", "Validation", "Testing" or "None".
     * @return Sample count for the role.
     */
    Index get_samples_number(const string& role_name) const;

    /**
     * @brief Returns the number of samples that are not "None".
     * @return Used sample count.
     */
    Index get_used_samples_number() const;

    /**
     * @brief Returns the indices of the samples assigned to a given role.
     * @param role_name "Training", "Validation", "Testing" or "None".
     * @return Vector of sample indices for the role.
     */
    vector<Index> get_sample_indices(const string& role_name) const;

    /**
     * @brief Returns the indices of all samples that are not "None".
     * @return Vector of used sample indices.
     */
    vector<Index> get_used_sample_indices() const;

    /**
     * @brief Returns the per-sample role assignments.
     * @return Const reference to the role vector.
     */
    const vector<SampleRole>& get_sample_roles() const { return sample_roles; }

    /**
     * @brief Returns the per-sample role indices as plain integers.
     * @return Vector with the integer encoding of each role.
     */
    vector<Index> get_sample_roles_vector() const;

    /**
     * @brief Counts the samples assigned to each role.
     * @return Length-4 vector: [Training, Validation, Testing, None] counts.
     */
    VectorI get_sample_role_numbers() const;

    /**
     * @brief Returns the total number of variables (columns of @p data).
     * @return Variable count.
     */
    Index get_variables_number() const { return variables.size(); }

    /**
     * @brief Returns the number of variables assigned to a given role.
     * @param role_name Variable role ("Input", "Target", "Time", "Unused", "Decoder").
     * @return Variable count for the role.
     */
    Index get_variables_number(const string& role_name) const;

    /**
     * @brief Returns the number of variables that are not "Unused".
     * @return Used variable count.
     */
    Index get_used_variables_number() const;

    /**
     * @brief Returns the per-variable metadata.
     * @return Const reference to the variables vector.
     */
    const vector<Variable>& get_variables() const { return variables; }

    /**
     * @brief Returns the variables assigned to a given role.
     * @param role_name Variable role.
     * @return Vector of matching Variable entries.
     */
    vector<Variable> get_variables(const string& role_name) const;

    /**
     * @brief Returns the column index of the variable with a given name.
     * @param name Variable name.
     * @return Column index in @p data.
     */
    Index get_variable_index(const string& name) const;

    /**
     * @brief Returns the column index of the variable with a given numeric id.
     * @param id Variable id.
     * @return Column index in @p data.
     */
    Index get_variable_index(const Index id) const;

    /**
     * @brief Returns the column indices of the variables assigned to a given role.
     * @param role_name Variable role.
     * @return Vector of column indices.
     */
    vector<Index> get_variable_indices(const string& role_name) const;

    /**
     * @brief Returns the column indices of all variables that are not "Unused".
     * @return Vector of used column indices.
     */
    vector<Index> get_used_variables_indices() const;

    /**
     * @brief Returns the names of every variable.
     * @return Vector of variable names.
     */
    vector<string> get_variable_names() const;

    /**
     * @brief Returns the names of the variables assigned to a given role.
     * @param role_name Variable role.
     * @return Vector of variable names matching the role.
     */
    vector<string> get_variable_names(const string& role_name) const;

    /**
     * @brief Returns the type of a variable (Numeric, Binary, Categorical, ...).
     * @param index Column index.
     * @return Variable type.
     */
    VariableType get_variable_type(const Index index) const { return variables[index].type; }

    /**
     * @brief Returns the types of a list of variables.
     * @param indices Column indices.
     * @return Variable types in the same order as @p indices.
     */
    vector<VariableType> get_variable_types(const vector<Index> indices) const;

    /**
     * @brief Returns the total number of features.
     *
     * One categorical variable expands into N features (one per class), so
     * features_number >= variables_number in general.
     *
     * @return Feature count summed across variables.
     */
    Index get_features_number() const;

    /**
     * @brief Returns the number of features assigned to a given role.
     * @param role_name Variable role.
     * @return Feature count for the role.
     */
    Index get_features_number(const string& role_name) const;

    /**
     * @brief Returns the number of features that are not "Unused".
     * @return Used feature count.
     */
    Index get_used_features_number() const;

    /**
     * @brief Returns the names of every feature.
     * @return Vector of feature names.
     */
    vector<string> get_feature_names() const;

    /**
     * @brief Returns the names of the features assigned to a given role.
     * @param role_name Variable role.
     * @return Vector of feature names matching the role.
     */
    vector<string> get_feature_names(const string& role_name) const;

    /**
     * @brief Returns the per-variable feature indices.
     * @return Outer vector indexed by variable, inner by feature within that variable.
     */
    vector<vector<Index>> get_feature_indices() const;

    /**
     * @brief Returns the feature indices for a single variable.
     * @param variable_index Column index of the variable.
     * @return Feature indices generated by that variable.
     */
    vector<Index> get_feature_indices(const Index variable_index) const;

    /**
     * @brief Returns the feature indices for variables of a given role.
     * @param role_name Variable role.
     * @return Vector of feature indices.
     */
    vector<Index> get_feature_indices(const string& role_name) const;

    /**
     * @brief Returns the feature indices for all variables that are not "Unused".
     * @return Vector of used feature indices.
     */
    vector<Index> get_used_feature_indices() const;

    /**
     * @brief Returns the per-variable feature dimension (1 for Numeric, N for Categorical).
     * @return Vector of feature dimensions.
     */
    vector<Index> get_feature_dimensions() const;

    /**
     * @brief Returns the input or target shape used by the network.
     * @param role_name "Input" or "Target".
     * @return Shape with the number of features for that role.
     */
    Shape get_shape(const string& role_name) const;

    /**
     * @brief Returns the scaler chosen for each variable of a given role.
     * @param role_name Variable role.
     * @return Vector of scaler names ("MinimumMaximum", "MeanStandardDeviation", ...).
     */
    vector<string> get_feature_scalers(const string& role_name) const;

    /**
     * @brief Splits a list of sample indices into batches.
     *
     * Optionally shuffles the indices before batching.
     *
     * @param sample_indices Sample indices to batch.
     * @param batch_size Target batch size.
     * @param shuffle Whether to shuffle indices.
     * @param batches Output vector of batches; populated in place.
     */
    virtual void get_batches(const vector<Index>& sample_indices, Index batch_size, bool shuffle, vector<vector<Index>>& batches) const;

    /**
     * @brief Returns the raw data matrix.
     * @return Const reference to the [samples x variables] matrix.
     */
    const MatrixR& get_data() const { return data; }

    /**
     * @brief Returns the data matrix restricted to the features of a given role.
     * @param role_name Variable role.
     * @return Matrix [used_samples x features_number(role)].
     */
    MatrixR get_feature_data(const string& role_name) const;

    /**
     * @brief Returns the data restricted to a sample-role and variable-role intersection.
     * @param sample_role Sample role.
     * @param variable_role Variable role.
     * @return Matrix [samples_number(sample_role) x features_number(variable_role)].
     */
    MatrixR get_data(const string& sample_role, const string& variable_role) const;

    /**
     * @brief Returns the data restricted to specific samples and variables.
     * @param sample_indices Row indices.
     * @param variable_indices Column indices.
     * @return Submatrix in row-major order.
     */
    MatrixR get_data_from_indices(const vector<Index>& sample_indices, const vector<Index>& variable_indices) const;

    /**
     * @brief Returns a single sample as a row vector.
     * @param sample_index Row index.
     * @return Vector with all variable values for that sample.
     */
    VectorR get_sample_data(const Index sample_index) const;

    /**
     * @brief Returns the data for a single variable across all samples.
     * @param variable_index Column index.
     * @return Column matrix.
     */
    MatrixR get_variable_data(const Index variable_index) const;

    /**
     * @brief Returns the data for a single variable on a subset of samples.
     * @param variable_index Column index.
     * @param sample_indices Row indices.
     * @return Column matrix restricted to @p sample_indices.
     */
    MatrixR get_variable_data(const Index variable_index, const vector<Index>& sample_indices) const;

    /**
     * @brief Returns the data for a single variable identified by name.
     * @param variable_name Variable name.
     * @return Column matrix.
     */
    MatrixR get_variable_data(const string& variable_name) const;

    /**
     * @brief Returns the cached preview of the source file (first rows).
     * @return Const reference to the preview rows.
     */
    const vector<vector<string>>& get_data_file_preview() const { return data_file_preview; }

    /**
     * @brief Returns the configured missing-value strategy.
     * @return MissingValuesMethod enum value.
     */
    MissingValuesMethod get_missing_values_method() const { return missing_values_method; }

    /**
     * @brief Returns the missing-value strategy as a string.
     * @return One of "Unuse", "Mean", "Median", "Interpolation".
     */
    string get_missing_values_method_string() const;

    /**
     * @brief Returns the path to the source data file.
     * @return Const reference to the data path.
     */
    const filesystem::path& get_data_path() const { return data_path; }

    /**
     * @brief Returns the configured field separator.
     * @return Separator enum value.
     */
    const Separator& get_separator() const { return separator; }

    /**
     * @brief Returns the field separator as the actual delimiter character(s).
     * @return Separator string ("," ";" "\t" or " ").
     */
    string get_separator_string() const;

    /**
     * @brief Returns the field separator as a human-readable name.
     * @return Separator name ("Comma", "Semicolon", "Tab", "Space").
     */
    string get_separator_name() const;

    /**
     * @brief Returns the configured source-file codification.
     * @return Codification enum value.
     */
    const Codification& get_codification() const { return codification; }

    /**
     * @brief Returns the codification as a string.
     * @return One of "UTF8", "SHIFT_JIS".
     */
    const string get_codification_string() const;

    /**
     * @brief Returns the label that marks missing values in the source file.
     * @return Configured missing-value label (defaults to "NA").
     */
    const string& get_missing_values_label() const { return missing_values_label; }

    /**
     * @brief Reports whether progress messages are printed.
     * @return true when display is on.
     */
    bool get_display() const { return display; }

    /**
     * @brief Reports whether the data matrix is empty.
     * @return true when data has zero elements.
     */
    bool is_empty() const { return data.size() == 0; }

    /**
     * @brief Returns the input shape.
     * @return Shape used to construct the network's first layer.
     */
    Shape get_input_shape() const { return input_shape; }

    /**
     * @brief Returns the target shape.
     * @return Shape used to construct the network's output layer.
     */
    Shape get_target_shape() const { return target_shape; }

    /**
     * @brief Resets the dataset to a synthetic shape.
     * @param samples_number Number of samples.
     * @param input_shape Shape of the input portion.
     * @param target_shape Shape of the target portion.
     */
    void set(const Index samples_number = 0, const Shape& input_shape = {}, const Shape& target_shape = {});

    /**
     * @brief Resets the dataset by loading a delimited text file.
     * @param data_path Path to the source file.
     * @param separator Field separator string.
     * @param has_header Whether the first row contains column names.
     * @param has_ids Whether the first column contains sample identifiers.
     * @param codification Source-file character encoding.
     */
    void set(const filesystem::path& data_path,
             const string& separator,
             bool has_header = true,
             bool has_ids = false,
             const Dataset::Codification& codification = Codification::UTF8);

    /**
     * @brief Resets the dataset by loading a previously serialized JSON state.
     * @param file_name Path to the JSON file.
     */
    void set(const filesystem::path& file_name);

    /**
     * @brief Resets configuration members to defaults.
     */
    void set_default();

    /**
     * @brief Assigns the same role to every sample.
     * @param role_name Sample role.
     */
    void set_sample_roles(const string& role_name);

    /**
     * @brief Assigns a role to a single sample.
     * @param sample_index Row index.
     * @param role_name Sample role.
     */
    void set_sample_role(const Index sample_index, const string& role_name);

    /**
     * @brief Assigns roles to all samples from a parallel string vector.
     * @param role_names One name per sample.
     */
    void set_sample_roles(const vector<string>& role_names);

    /**
     * @brief Assigns the same role to a list of samples.
     * @param sample_indices Indices to update.
     * @param role_name Sample role.
     */
    void set_sample_roles(const vector<Index>& sample_indices, const string& role_name);

    /**
     * @brief Replaces the per-variable metadata.
     * @param new_variables New variables vector.
     */
    void set_variables(const vector<Variable>& new_variables) { variables = new_variables; }

    /**
     * @brief Sets default names ("variable_1", "variable_2", ...) for every variable.
     */
    void set_default_variable_names();

    /**
     * @brief Assigns roles to all variables from a parallel string vector.
     * @param role_names One role per variable.
     */
    virtual void set_variable_roles(const vector<string>& role_names);

    /**
     * @brief Re-creates the variables vector from an input/target shape descriptor.
     * @param description Compact descriptor parsed by the implementation.
     */
    void set_variables(const string& description);

    /**
     * @brief Marks selected variables as Input and others as Target.
     * @param input_indices Indices of input variables.
     * @param target_indices Indices of target variables.
     */
    void set_variable_indices(const vector<Index>& input_indices, const vector<Index>& target_indices);

    /**
     * @brief Marks all input variables as Unused.
     */
    void set_input_variables_unused();

    /**
     * @brief Sets the role of a single variable by index.
     * @param variable_index Column index.
     * @param role_name New role.
     */
    void set_variable_role(const Index variable_index, const string& role_name);

    /**
     * @brief Sets the role of a single variable by name.
     * @param variable_name Variable name.
     * @param role_name New role.
     */
    void set_variable_role(const string& variable_name, const string& role_name);

    /**
     * @brief Sets the type of a single variable by index.
     * @param variable_index Column index.
     * @param type New variable type.
     */
    void set_variable_type(const Index variable_index, const VariableType& type);

    /**
     * @brief Sets the type of a single variable by name.
     * @param variable_name Variable name.
     * @param type New variable type.
     */
    void set_variable_type(const string& variable_name, const VariableType& type);

    /**
     * @brief Sets every variable to a given type.
     * @param type Variable type to apply.
     */
    void set_variable_types(const VariableType& type);

    /**
     * @brief Replaces the names of every variable.
     * @param new_variable_names One name per variable.
     */
    void set_variable_names(const vector<string>& new_variable_names);

    /**
     * @brief Resizes the variables vector.
     * @param new_size New variable count.
     */
    void set_variables_number(const Index new_size) { variables.resize(new_size); }

    /**
     * @brief Sets the same scaler on every variable.
     * @param scaler_name Scaler name.
     */
    void set_variable_scalers(const string& scaler_name);

    /**
     * @brief Sets one scaler per variable.
     * @param scaler_names One name per variable.
     */
    void set_variable_scalers(const vector<string>& scaler_names);

    /**
     * @brief Detects binary variables (two distinct values) and tags them accordingly.
     */
    void set_binary_variables();

    /**
     * @brief Names every feature.
     *
     * Each categorical variable contributes one name per class.
     *
     * @param new_feature_names One name per feature.
     */
    void set_feature_names(const vector<string>& new_feature_names);

    /**
     * @brief Assigns the same role to every variable.
     * @param role_name Variable role.
     */
    void set_variable_roles(const string& role_name);

    /**
     * @brief Sets the input or target shape.
     * @param role_name "Input" or "Target".
     * @param new_shape New shape.
     */
    void set_shape(const string& role_name, const Shape& new_shape);

    /**
     * @brief Replaces the data matrix.
     *
     * The number of rows and columns must match the existing samples and
     * variables count.
     *
     * @param new_data Replacement matrix.
     */
    void set_data(const MatrixR& new_data);

    /**
     * @brief Sets the path to the source data file.
     * @param new_data_path New path.
     */
    void set_data_path(const filesystem::path& new_data_path) { data_path = new_data_path; }

    /**
     * @brief Sets whether the source file has a header row.
     * @param new_has_header true if the first row contains names.
     */
    void set_has_header(bool new_has_header) { has_header = new_has_header; }

    /**
     * @brief Sets whether the source file has a sample-id column.
     * @param new_has_ids true if the first column contains identifiers.
     */
    void set_has_ids(bool new_has_ids) { has_sample_ids = new_has_ids; }

    /**
     * @brief Sets the field separator.
     * @param new_separator Separator enum value.
     */
    void set_separator(const Separator& new_separator) { separator = new_separator; }

    /**
     * @brief Sets the field separator from its delimiter character(s).
     * @param new_separator_string Delimiter ("," ";" "\t" or " ").
     */
    void set_separator_string(const string& new_separator_string);

    /**
     * @brief Sets the field separator from its human-readable name.
     * @param new_separator_name "Comma", "Semicolon", "Tab" or "Space".
     */
    void set_separator_name(const string& new_separator_name);

    /**
     * @brief Sets the source-file codification.
     * @param new_codification Codification enum value.
     */
    void set_codification(const Codification& new_codification) { codification = new_codification; }

    /**
     * @brief Sets the source-file codification from its name.
     * @param new_codification "UTF8" or "SHIFT_JIS".
     */
    void set_codification(const string& new_codification);

    /**
     * @brief Sets the label used for missing values in the source file.
     * @param label New label.
     */
    void set_missing_values_label(string label) { missing_values_label = move(label); }

    /**
     * @brief Sets the missing-value handling strategy.
     * @param method MissingValuesMethod enum value.
     */
    void set_missing_values_method(const MissingValuesMethod& method) { missing_values_method = method; }

    /**
     * @brief Sets the missing-value handling strategy from its name.
     * @param method_name "Unuse", "Mean", "Median" or "Interpolation".
     */
    void set_missing_values_method(const string& method_name);

    /**
     * @brief Sets the GMT offset for time variables.
     * @param new_gmt Offset in hours.
     */
    void set_gmt(const Index new_gmt) { gmt = new_gmt; }

    /**
     * @brief Toggles progress messages.
     * @param new_display true to enable.
     */
    void set_display(bool new_display) { display = new_display; }

    /**
     * @brief Reports whether a sample is used (any role other than None).
     * @param i Sample index.
     * @return true when the sample is used.
     */
    bool is_sample_used(const Index i) const { return sample_roles[i] != SampleRole::None; }

    /**
     * @brief Reports whether at least one variable is binary.
     * @return true when any variable has type Binary.
     */
    bool has_binary_variables() const;

    /**
     * @brief Reports whether at least one variable is categorical.
     * @return true when any variable has type Categorical.
     */
    bool has_categorical_variables() const;

    /**
     * @brief Reports whether the dataset has any binary or categorical variable.
     * @return true when at least one variable is Binary or Categorical.
     */
    bool has_binary_or_categorical_variables() const;

    /**
     * @brief Reports whether at least one variable plays the Time role.
     * @return true when a Time variable exists.
     */
    bool has_time_variable() const;

    /**
     * @brief Reports whether at least one sample is assigned to Validation.
     * @return true when validation samples exist.
     */
    bool has_validation() const;

    /**
     * @brief Reports whether the dataset has missing values matching any of the supplied labels.
     * @param labels Candidate missing-value labels.
     * @return true when any sample contains one of the labels.
     */
    bool has_missing_values(const vector<string>& labels) const;

    /**
     * @brief Splits samples into training/validation/testing partitions.
     *
     * Convenience entry point that delegates to split_samples_random() (when
     * @p shuffle is true) or split_samples_sequential() (otherwise).
     *
     * @param training_ratio Fraction of samples for training.
     * @param selection_ratio Fraction for validation.
     * @param testing_ratio Fraction for testing.
     * @param shuffle Whether to shuffle the indices before splitting.
     */
    void split_samples(const float training_ratio = 0.6f, float selection_ratio = 0.2f, float testing_ratio = 0.2f, bool shuffle = true);

    /**
     * @brief Splits samples into partitions in their original order.
     * @param training_ratio Fraction of samples for training.
     * @param selection_ratio Fraction for validation.
     * @param testing_ratio Fraction for testing.
     */
    void split_samples_sequential(const float training_ratio = 0.6f, float selection_ratio = 0.2f, float testing_ratio = 0.2f);

    /**
     * @brief Splits samples into partitions after random shuffling.
     * @param training_ratio Fraction of samples for training.
     * @param selection_ratio Fraction for validation.
     * @param testing_ratio Fraction for testing.
     */
    void split_samples_random(const float training_ratio = 0.6f, float selection_ratio = 0.2f, float testing_ratio = 0.2f);

    /**
     * @brief Marks variables with low correlation against the target as Unused.
     * @param minimum_correlation Threshold; variables below it are unused.
     * @return Names of the variables marked as unused.
     */
    vector<string> unuse_uncorrelated_variables(const float minimum_correlation = 0.25f);

    /**
     * @brief Marks variables strongly correlated against another input as Unused.
     * @param maximum_correlation Threshold; variables above it (with another input) are unused.
     * @return Names of the variables marked as unused.
     */
    vector<string> unuse_collinear_variables(const float maximum_correlation = 0.95f);

    /**
     * @brief Fills the data matrix with a constant value.
     * @param value Value used to fill every cell.
     */
    void set_data_constant(const float value);

    /**
     * @brief Computes descriptive statistics for every feature.
     * @return One Descriptives entry per feature.
     */
    vector<Descriptives> calculate_feature_descriptives() const;

    /**
     * @brief Computes descriptives for inputs restricted to positive-target samples.
     * @return One Descriptives entry per input variable.
     */
    vector<Descriptives> calculate_variable_descriptives_positive_samples() const;

    /**
     * @brief Computes descriptives for inputs restricted to negative-target samples.
     * @return One Descriptives entry per input variable.
     */
    vector<Descriptives> calculate_variable_descriptives_negative_samples() const;

    /**
     * @brief Computes descriptives per category of a categorical variable.
     * @param variable_index Column index of the categorical variable.
     * @return One Descriptives entry per (input variable, category) pair.
     */
    vector<Descriptives> calculate_variable_descriptives_categories(const Index variable_index) const;

    /**
     * @brief Computes feature descriptives for a single role.
     * @param role_name Variable role.
     * @return One Descriptives entry per feature in the role.
     */
    vector<Descriptives> calculate_feature_descriptives(const string& role_name) const;

    /**
     * @brief Builds histograms of every variable.
     * @param bins_number Number of bins per histogram.
     * @return One Histogram per variable.
     */
    vector<Histogram> calculate_variable_distributions(const Index bins_number = 10) const;

    /**
     * @brief Computes box-plot statistics for every variable.
     * @return One BoxPlot per variable.
     */
    vector<BoxPlot> calculate_variables_box_plots() const;

    /**
     * @brief Computes a custom correlation between every pair of input variables.
     * @param correlation_function Function used to compute the pair correlation.
     * @param method Correlation method (Pearson, Spearman, ...).
     * @param samples_role Sample-role filter.
     * @return Symmetric Correlation matrix [inputs x inputs].
     */
    Tensor<Correlation, 2> calculate_input_variable_correlations(
        Correlation (*correlation_function)(const MatrixR&, const MatrixR&), Correlation::Method method, const string& samples_role) const;

    /**
     * @brief Computes Pearson correlations between every pair of input variables.
     * @return Symmetric Correlation matrix [inputs x inputs].
     */
    Tensor<Correlation, 2> calculate_input_variable_pearson_correlations() const;

    /**
     * @brief Computes Spearman rank correlations between every pair of input variables.
     * @return Symmetric Correlation matrix [inputs x inputs].
     */
    Tensor<Correlation, 2> calculate_input_variable_spearman_correlations() const;

    /**
     * @brief Computes a custom correlation between inputs and targets.
     * @param correlation_function Function used to compute the pair correlation.
     * @param samples_role Sample-role filter.
     * @return Correlation matrix [inputs x targets].
     */
    Tensor<Correlation, 2> calculate_input_target_variable_correlations(
        Correlation (*correlation_function)(const MatrixR&, const MatrixR&), const string& samples_role) const;

    /**
     * @brief Computes Pearson correlations between inputs and targets.
     * @return Correlation matrix [inputs x targets].
     */
    Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const;

    /**
     * @brief Computes Spearman rank correlations between inputs and targets.
     * @return Correlation matrix [inputs x targets].
     */
    Tensor<Correlation, 2> calculate_input_target_variable_spearman_correlations() const;

    /**
     * @brief Returns the rank of every input variable by absolute Pearson correlation against targets.
     * @return Vector of ranks aligned to input variables.
     */
    VectorI calculate_correlations_rank() const;

    /**
     * @brief Picks default scalers for every variable based on its type.
     */
    void set_default_variable_scalers();

    /**
     * @brief Scales the entire data matrix.
     * @return One Descriptives entry per variable, computed before scaling.
     */
    vector<Descriptives> scale_data();

    /**
     * @brief Scales the features of a given role.
     *
     * Reads the scalers from the per-variable metadata, computes the
     * descriptives of the unscaled data, scales the data in place and returns
     * the descriptives so they can be reused (for inverse-scaling outputs or
     * configuring the network's Scaling layer).
     *
     * @param role_name Variable role to scale (typically "Input").
     * @return One Descriptives entry per feature in the role.
     */
    virtual vector<Descriptives> scale_features(const string& role_name);

    /**
     * @brief Inverse-scales the features of a given role.
     * @param role_name Variable role to unscale.
     * @param feature_descriptives Descriptives produced by scale_features().
     */
    void unscale_features(const string& role_name, const vector<Descriptives>& feature_descriptives);

    /**
     * @brief Counts the samples of every target class.
     * @return Vector with one count per class.
     */
    VectorI calculate_target_distribution() const;

    /**
     * @brief Detects Tukey outliers per variable.
     * @param tukey_factor Tukey-fence multiplier (1.5 = mild, 3.0 = extreme).
     * @param replace Whether to also replace the outliers with NaN.
     * @return Outer vector indexed by variable, inner by outlier sample index.
     */
    vector<vector<Index>> calculate_Tukey_outliers(const float tukey_factor = 1.5f, bool replace = false);

    /**
     * @brief Detects Tukey outliers and replaces them with NaN.
     * @param tukey_factor Tukey-fence multiplier.
     * @return Outer vector indexed by variable, inner by sample index that was replaced.
     */
    vector<vector<Index>> replace_Tukey_outliers_with_NaN(const float tukey_factor = 1.5f);

    /**
     * @brief Marks Tukey-outlier samples as Unused.
     * @param tukey_factor Tukey-fence multiplier.
     */
    void unuse_Tukey_outliers(const float tukey_factor = 1.5f);

    /**
     * @brief Fills the data matrix with uniform random values.
     */
    virtual void set_data_random();

    /**
     * @brief Fills the data matrix with random integers in [0, vocabulary_size).
     * @param vocabulary_size Exclusive upper bound for the integers.
     */
    virtual void set_data_integer(const Index vocabulary_size);

    /**
     * @brief Fills the data matrix with samples from the Rosenbrock function.
     */
    void set_data_rosenbrock();

    /**
     * @brief Fills the data matrix with synthetic binary-classification data.
     */
    void set_data_binary_classification();

    /**
     * @brief Restores the dataset state from a JSON document.
     * @param document Parsed JSON produced by to_JSON().
     */
    virtual void from_JSON(const JsonDocument& document);

    /**
     * @brief Serializes the dataset state to JSON.
     * @param writer JSON writer that receives the dataset tree.
     */
    virtual void to_JSON(JsonWriter& writer) const;

    /**
     * @brief Saves the dataset state to a JSON file on disk.
     * @param file_name Destination path.
     */
    void save(const filesystem::path& file_name) const;

    /**
     * @brief Loads the dataset state from a JSON file on disk.
     * @param file_name Source path.
     */
    void load(const filesystem::path& file_name);

    /**
     * @brief Saves the data matrix back to the configured source file.
     */
    void save_data() const;

    /**
     * @brief Saves the data matrix as a binary file.
     * @param file_name Destination path.
     */
    void save_data_binary(const filesystem::path& file_name) const;

    /**
     * @brief Loads the data matrix from a binary file produced by save_data_binary().
     */
    void load_data_binary();

    /**
     * @brief Returns the total number of cells flagged as missing.
     * @return Missing-value count.
     */
    Index get_missing_values_number() const { return missing_values_number; }

    /**
     * @brief Reports whether the data matrix contains any NaN.
     * @return true when at least one cell is NaN.
     */
    bool has_nan() const;

    /**
     * @brief Reports whether a row contains any NaN.
     * @param row_index Row to inspect.
     * @return true when at least one cell of @p row_index is NaN.
     */
    bool has_nan_row(const Index row_index) const;

    /**
     * @brief Marks samples with missing values as Unused.
     */
    virtual void impute_missing_values_unuse();

    /**
     * @brief Replaces missing values with a per-variable statistic.
     * @param method MissingValuesMethod (Mean, Median).
     */
    void impute_missing_values_statistic(const MissingValuesMethod& method);

    /**
     * @brief Replaces missing values via linear interpolation along each variable.
     */
    virtual void impute_missing_values_interpolate();

    /**
     * @brief Removes samples that contain missing values.
     */
    void scrub_missing_values();

    /**
     * @brief Updates the cached missing-value statistics (counts, indices).
     */
    void calculate_missing_values_statistics();

    /**
     * @brief Counts NaN cells per variable.
     * @return Vector with one count per variable.
     */
    VectorI count_nans_per_variable() const;

    /**
     * @brief Counts variables that contain at least one NaN.
     * @return Number of affected variables.
     */
    Index count_variables_with_nan() const;

    /**
     * @brief Counts samples that contain at least one NaN.
     * @return Number of affected rows.
     */
    Index count_rows_with_nan() const;

    /**
     * @brief Counts the total number of NaN cells.
     * @return Total NaN count.
     */
    Index count_nan() const;

    /**
     * @brief Splits a list of sample indices into chunks of (roughly) equal size.
     * @param indices Sample indices to split.
     * @param parts_number Number of chunks.
     * @return Outer vector indexed by chunk, inner by sample index.
     */
    vector<vector<Index>> split_samples(const vector<Index>& indices, Index parts_number) const;

    /**
     * @brief Reads the configured CSV file into the data matrix.
     */
    virtual void read_csv();

    /**
     * @brief Infers the date format used by date-typed variables in the source file.
     * @param variables Variables to inspect.
     * @param data_file_preview Preview rows from the source file.
     * @param has_header Whether the preview includes a header row.
     * @param missing_values_label Missing-value label to ignore.
     * @return Inferred date format.
     */
    DateFormat infer_dataset_date_format(const vector<Variable>& variables, const vector<vector<string>>& data_file_preview, bool has_header, const string& missing_values_label);

    /**
     * @brief Fills a contiguous float buffer with input data for a batch of samples.
     *
     * Used by Loss/Optimizer to build forward-pass inputs without reallocating.
     *
     * @param sample_indices Row indices for the batch.
     * @param feature_indices Column indices for the input features.
     * @param buffer Destination buffer; must hold sample_indices.size() *
     *               feature_indices.size() floats.
     * @param transpose Whether to write feature-major (true) or sample-major (false).
     * @param contiguous Stride hint when sample_indices is a contiguous range; -1 disables.
     */
    virtual void fill_inputs(const vector<Index>& sample_indices, const vector<Index>& feature_indices, float* buffer, bool transpose = true, int contiguous = -1) const;

    /**
     * @brief Optionally augments inputs in-place after fill_inputs() (e.g. random crops).
     *
     * Default implementation is a no-op; subclasses override.
     *
     * @param buffer Buffer produced by fill_inputs().
     * @param batch_size Number of samples in the buffer.
     */
    virtual void augment_inputs(float* buffer, Index batch_size) const {}

    /**
     * @brief Fills a contiguous buffer with decoder-side inputs (transformer-style models).
     * @param sample_indices Row indices for the batch.
     * @param feature_indices Column indices for the decoder features.
     * @param buffer Destination buffer.
     * @param transpose Whether to write feature-major or sample-major.
     * @param contiguous Stride hint.
     */
    virtual void fill_decoder(const vector<Index>& sample_indices,
                              const vector<Index>& feature_indices,
                              float* buffer,
                              bool transpose = true,
                              int contiguous = -1) const;

    /**
     * @brief Fills a contiguous buffer with target data for a batch of samples.
     * @param sample_indices Row indices for the batch.
     * @param feature_indices Column indices for the target features.
     * @param buffer Destination buffer.
     * @param transpose Whether to write feature-major or sample-major.
     * @param contiguous Stride hint.
     */
    virtual void fill_targets(const vector<Index>& sample_indices, const vector<Index>& feature_indices, float* buffer, bool transpose = true, int contiguous = -1) const;

private:

    /// Inspects @p data_file_preview and assigns a VariableType to each variable.
    void infer_variable_types_from_data();

    /// Marks variables whose values do not change as Unused.
    void unuse_constant_variables();

    /**
     * @brief Returns the role name of a sample by index.
     * @param i Row index.
     * @return Role name as string.
     */
    string get_sample_role(const Index i) const { return sample_role_to_string(sample_roles[i]); }

    /**
     * @brief Returns the role of a sample as enum.
     * @param i Row index.
     * @return SampleRole enum value.
     */
    SampleRole get_sample_role_type(const Index i) const { return sample_roles[i]; }

    /**
     * @brief Selects used samples whose value at a column is positive or negative.
     * @param variable_index Column index.
     * @param positive Whether to keep positive (true) or negative (false) values.
     * @return Row indices that match.
     */
    vector<Index> filter_used_samples_by_column(Index, bool) const;

    /**
     * @brief Applies (or reverses) a single scaler to a single feature in @p data.
     * @param feature_index Column index in @p data.
     * @param scaler Scaler name.
     * @param descriptives Descriptives used by the scaler.
     * @param inverse Whether to reverse the scaling.
     */
    void apply_scaler(Index feature_index, const string& scaler, const Descriptives& descriptives, bool inverse);

    /**
     * @brief Infers the type of every column from the preview rows.
     * @param data_file_preview Preview rows from the source file.
     */
    void infer_column_types(const vector<vector<string>>& data_file_preview);

    /**
     * @brief Caches the preview rows for later inspection.
     * @param data_file_preview Preview rows.
     */
    void read_data_file_preview(const vector<vector<string>>& data_file_preview);

    /**
     * @brief Validates that the file uses the configured separator only.
     * @param file_path Path to the file.
     * @throws runtime_error if multiple separator candidates are detected.
     */
    void check_separators(const string& file_path) const;

protected:

    /// Sets default Input/Target roles based on column position (last column = target).
    void set_default_variable_roles();

    /// Sets default roles for forecasting (typical pattern: lagged inputs + future target).
    void set_default_variable_roles_forecasting();

    // DATA

    /// Dense data matrix [samples x variables].
    MatrixR data;

    // Dimensions

    /// Shape of the input portion (rank may exceed 1 for image/sequence data).
    Shape input_shape;
    /// Shape of the target portion.
    Shape target_shape;
    /// Shape of the decoder portion (transformer-style models).
    Shape decoder_shape;

    // Samples

    /// Per-sample role (Training/Validation/Testing/None).
    vector<SampleRole> sample_roles;

    /// Optional per-sample identifiers (when the source file has an id column).
    vector<string> sample_ids;

    // Variables

    /// Per-variable metadata (name, role, type, scaler).
    vector<Variable> variables;

    // Data File

    /// Path to the source data file.
    filesystem::path data_path;

    /// Field separator used in the source file.
    Separator separator = Separator::Comma;

    /// Label that marks missing values in the source file.
    string missing_values_label = "NA";

    //VectorB nans_variables;

    /// Whether the source file's first row contains column names.
    bool has_header = false;

    /// Whether the source file's first column contains sample identifiers.
    bool has_sample_ids = false;

    /// Source-file character encoding.
    Codification codification = Codification::UTF8;

    /// Cached preview of the first rows of the source file.
    vector<vector<string>> data_file_preview;

    /// GMT offset for time variables, in hours.
    Index gmt = 0;

    // Missing Values

    /// Strategy used to handle missing values.
    MissingValuesMethod missing_values_method = MissingValuesMethod::Mean;

    /// Total number of missing cells.
    Index missing_values_number = 0;

    /// Per-variable missing-value count.
    VectorI variables_missing_values_number;

    /// Number of rows that contain at least one missing value.
    Index rows_missing_values_number = 0;

    // Display

    /// Whether to print progress messages.
    bool display = true;

    /// Strings interpreted as positive when parsing binary variables.
    const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
    /// Strings interpreted as negative when parsing binary variables.
    const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};

    /// Serializes the variables vector to JSON.
    void variables_to_JSON(JsonWriter&) const;
    /// Serializes the per-sample roles to JSON.
    void samples_to_JSON(JsonWriter&) const;
    /// Serializes the missing-value statistics to JSON.
    void missing_values_to_JSON(JsonWriter&) const;
    /// Serializes the source-file preview to JSON.
    void preview_data_to_JSON(JsonWriter&) const;

    /// Restores the variables vector from JSON.
    void variables_from_JSON(const Json*);
    /// Restores the per-sample roles from JSON.
    void samples_from_JSON(const Json*);
    /// Restores the missing-value statistics from JSON.
    void missing_values_from_JSON(const Json*);
    /// Restores the source-file preview from JSON.
    void preview_data_from_JSON(const Json*);
};

}

#define STRINGIFY_ENUM(x) #x

#define ENUM_TO_STRING(x) STRINGIFY_ENUM(x)

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
