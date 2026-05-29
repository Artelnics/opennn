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

enum class SampleRole
{
    Training,
    Validation,
    Testing,
    None
};

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

inline const string& sample_role_to_string(SampleRole role)
{
    return sample_role_map().to_string(role);
}

inline SampleRole string_to_sample_role(const string& name)
{
    if (name == "0") return SampleRole::Training;
    if (name == "1") return SampleRole::Validation;
    if (name == "2") return SampleRole::Testing;
    if (name == "3") return SampleRole::None;
    return sample_role_map().from_string(name);
}


struct DeviceResidentData
{
    Buffer input{Device::CUDA};    // rows x input_features  (FP32, row-major)
    Buffer target{Device::CUDA};   // rows x target_features (FP32)
    vector<Index> row_of;          // dataset sample index -> resident row (-1 if absent)
    Index rows = 0;
    Index input_features = 0;
    Index target_features = 0;
    bool  valid = false;
};

class Dataset
{

public:

    enum class Codification { UTF8, SHIFT_JIS };

    virtual ~Dataset() = default;

    enum class Separator{Space, Tab, Comma, Semicolon};

    virtual Index get_samples_number() const { return ssize(sample_roles); }

    Index get_samples_number(const string&) const;

    Index get_used_samples_number() const;

    vector<Index> get_sample_indices(const string&) const;

    vector<Index> get_used_sample_indices() const;

    const vector<SampleRole>& get_sample_roles() const { return sample_roles; }

    vector<Index> get_sample_roles_vector() const;

    VectorI get_sample_role_numbers() const;

    Index get_variables_number() const { return variables.size(); }
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

    vector<VariableType> get_variable_types(const vector<Index>& indices) const;
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

    virtual void get_batches(const vector<Index>&, Index, bool, vector<vector<Index>>&) const;

    virtual bool supports_device_residency() const { return false; }
    virtual bool ensure_device_resident(const string& /*sample_role*/) { return false; }
    virtual const DeviceResidentData* get_device_resident(const string& /*sample_role*/) const { return nullptr; }

    const vector<vector<string>>& get_data_file_preview() const { return data_file_preview; }

    const filesystem::path& get_data_path() const { return data_path; }

    const Separator& get_separator() const { return separator; }
    string get_separator_string() const;
    string get_separator_name() const;

    const Codification& get_codification() const { return codification; }
    string get_codification_string() const;

    bool get_display() const { return display; }

    virtual bool is_empty() const { return get_samples_number() == 0; }

    Shape get_input_shape() const { return input_shape; }
    Shape get_target_shape() const { return target_shape; }

    void set_default();
    void set_sample_roles(const string&);

    void set_sample_role(const Index, const string&);

    void set_sample_roles(const vector<string>&);
    void set_sample_roles(const vector<Index>&, const string&);
    void set_variables(const vector<Variable>& new_variables) { variables = new_variables; }

    void set_default_variable_names();

    void set_variable_roles(const vector<string>&);

    void set_variable_indices(const vector<Index>&, const vector<Index>&);
    void set_input_variables_unused();

    void set_variable_role(const Index, const string&);
    void set_variable_role(const string&, const string&);

    void set_variable_type(const Index, const VariableType&);
    void set_variable_type(const string&, const VariableType&);

    void set_variable_types(const VariableType&);
    void set_variable_names(const vector<string>&);

    void set_variables_number(const Index new_size) { variables.resize(new_size); }

    void set_feature_names(const vector<string>&);

    void set_variable_roles(const string&);

    void set_shape(const string&, const Shape&);
    virtual void resize_input_shape(Index input_features_count) { set_shape("Input", {input_features_count}); }
    void set_data_path(const filesystem::path& new_data_path) { data_path = new_data_path; }

    void set_has_header(bool new_has_header) { has_header = new_has_header; }
    void set_has_ids(bool new_has_ids) { has_sample_ids = new_has_ids; }

    void set_separator(const Separator& new_separator) { separator = new_separator; }
    void set_separator_string(const string&);
    void set_separator_name(const string&);

    void set_codification(const Codification& new_codification) { codification = new_codification; }
    void set_codification(const string&);

    void set_display(bool new_display) { display = new_display; }

    bool is_sample_used(const Index i) const { return sample_roles[i] != SampleRole::None; }

    bool has_binary_variables() const;
    bool has_categorical_variables() const;
    bool has_binary_or_categorical_variables() const;
    bool has_time_variable() const;

    bool has_validation() const;

    void split_samples(const float training_ratio = 0.6f,
                       float validation_ratio = 0.2f,
                       float testing_ratio = 0.2f,
                       bool shuffle = true);

    void split_samples_sequential(const float training_ratio = 0.6f,
                                  float validation_ratio = 0.2f,
                                  float testing_ratio = 0.2f);

    void split_samples_random(const float training_ratio = 0.6f,
                              float validation_ratio = 0.2f,
                              float testing_ratio = 0.2f);

    vector<vector<Index>> split_samples(const vector<Index>&, Index) const;

    virtual vector<Descriptives> scale_features(const string&) { return {}; }

    virtual void from_JSON(const JsonDocument&) = 0;
    virtual void to_JSON(JsonWriter&) const {}

    virtual bool has_nan() const { return false; }

    virtual void scrub_missing_values() {}

    virtual vector<Descriptives> calculate_feature_descriptives(const string&) const { return {}; }
    virtual Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const { return {}; }
    virtual VectorI calculate_target_distribution() const { return {}; }
    virtual VectorI calculate_correlations_rank() const { return {}; }

    virtual void unscale_features(const string&, const vector<Descriptives>&) {}

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    virtual void fill_inputs(const vector<Index>&,
                             const vector<Index>&,
                             float*,
                             bool is_training,
                             bool parallelize = true,
                             int contiguous = -1) const;

    virtual void augment_inputs(float*, Index, bool = true) const {}

    virtual void fill_decoder(const vector<Index>&,
                              const vector<Index>&,
                              float*,
                              bool is_training,
                              bool parallelize = true,
                              int contiguous = -1) const;

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

    void read_data_file_preview(const vector<vector<string_view>>&);
    void check_separators(string_view) const;
    void samples_from_JSON(const Json*);
    virtual void resize_data_from_JSON(Index) {}

    Shape input_shape;
    Shape target_shape;
    Shape decoder_shape;

    vector<SampleRole> sample_roles;
    vector<string> sample_ids;

    vector<Variable> variables;

    filesystem::path data_path;
    Separator separator = Separator::Comma;
    bool has_header = false;
    bool has_sample_ids = false;
    Codification codification = Codification::UTF8;
    vector<vector<string>> data_file_preview;

    bool display = true;

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
