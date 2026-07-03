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
#include "tensor_types.h"
#include "enum_map.h"
#include "string_utilities.h"
#include "variable.h"

namespace opennn
{

struct Batch;

enum class SampleRole{Training, Validation, Testing, None};

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

class Dataset
{

public:

    enum class Codification { UTF8, SHIFT_JIS };

    virtual ~Dataset() = default;

    enum class Separator{Space, Tab, Comma, Semicolon};
    enum class StorageMode{Matrix, BinaryFile, GPUPersistantData};

    virtual Index get_samples_number() const noexcept { return ssize(sample_roles); }

    Index get_samples_number(const string&) const;

    Index get_used_samples_number() const;

    vector<Index> get_sample_indices(const string&) const;

    vector<Index> get_used_sample_indices() const;

    const vector<SampleRole>& get_sample_roles() const noexcept { return sample_roles; }

    vector<Index> get_sample_roles_vector() const;

    VectorI get_sample_role_numbers() const;

    Index get_variables_number() const noexcept { return variables.size(); }
    Index get_variables_number(const string&) const;
    Index get_used_variables_number() const;

    const vector<Variable>& get_variables() const noexcept { return variables; }
    vector<Variable> get_variables(const string&) const;

    Index get_variable_index(const string&) const;
    Index get_variable_index(const Index) const;

    vector<Index> get_variable_indices(const string&) const;
    vector<Index> get_used_variables_indices() const;

    vector<string> get_variable_names() const;
    vector<string> get_variable_names(const string&) const;

    VariableType get_variable_type(const Index index) const { return variables[index].type; }

    vector<VariableType> get_variable_types(const vector<Index>&) const;
    Index get_features_number() const;
    Index get_features_number(const string&) const;

    vector<string> get_feature_names() const;
    vector<string> get_feature_names(const string&) const;

    vector<vector<Index>> get_feature_indices() const;
    vector<Index> get_feature_indices(const Index) const;
    vector<Index> get_feature_indices(const string&) const;
    vector<Index> get_used_feature_indices() const;

    vector<Index> get_feature_dimensions() const;

    Shape get_shape(const string&) const;

    virtual void get_batches(const vector<Index>&, Index, bool, vector<vector<Index>>&) const;

    const vector<vector<string>>& get_data_file_preview() const noexcept { return data_file_preview; }

    const filesystem::path& get_data_path() const noexcept { return data_path; }
    StorageMode get_storage_mode() const noexcept { return storage_mode; }
    string get_storage_mode_string() const;

    const Separator& get_separator() const noexcept { return separator; }
    string get_separator_string() const;
    string get_separator_name() const;

    const Codification& get_codification() const noexcept { return codification; }
    string get_codification_string() const;

    bool get_display() const noexcept { return display; }

    virtual bool is_empty() const noexcept { return get_samples_number() == 0; }

    Shape get_input_shape() const noexcept { return input_shape; }
    Shape get_target_shape() const noexcept { return target_shape; }

    const MatrixR& get_data() const noexcept { return data; }
    void set_data(const MatrixR&);
    void set_data_constant(float);

    virtual void enable_device_residency();
    void disable_device_residency() { data_device.resize_bytes(0, Device::CUDA); }
    bool is_device_resident() const noexcept { return data_device.data != nullptr; }
    const float* get_device_data() const { return data_device.as<float>(); }
    Index get_data_columns() const noexcept { return data.cols(); }
    Index get_device_data_columns() const noexcept { return device_data_columns; }

    void set_default();
    void set_sample_roles(const string&);

    void set_sample_role(const Index, const string&);

    void set_sample_roles(const vector<string>&);
    void set_sample_roles(const vector<Index>&, const string&);
    void set_variables(const vector<Variable>&);

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

    void set_variables_number(const Index);

    void set_feature_names(const vector<string>&);

    void set_variable_roles(const string&);

    void set_shape(const string&, const Shape&);
    virtual void resize_input_shape(Index input_features_count) { set_shape("Input", {input_features_count}); }
    virtual void set_data_path(const filesystem::path&);
    virtual void set_storage_mode(StorageMode);
    virtual void set_storage_mode(const string&);

    void set_has_header(bool new_has_header) { has_header = new_has_header; }
    void set_has_ids(bool new_has_ids) { has_sample_ids = new_has_ids; }

    bool get_header_line() const { return has_header; }
    bool get_has_sample_ids() const { return has_sample_ids; }
    const vector<string>& get_sample_ids() const { return sample_ids; }

    // Methods with default no-op implementations; subclasses override as needed
    virtual vector<string> get_feature_scalers(const string&) const { return {}; }
    virtual MatrixR get_variable_data(Index) const { return {}; }
    virtual MatrixR get_variable_data(Index, const vector<Index>&) const { return {}; }
    virtual vector<Histogram> calculate_variable_distributions(Index = 10) const { return {}; }
    virtual vector<BoxPlot> calculate_variables_box_plots() const { return {}; }
    virtual Tensor<Correlation, 2> calculate_input_variable_pearson_correlations() const { return {}; }
    virtual Tensor<Correlation, 2> calculate_input_variable_spearman_correlations() const { return {}; }
    virtual Tensor<Correlation, 2> calculate_input_target_variable_spearman_correlations() const { return calculate_input_target_variable_pearson_correlations(); }
    virtual vector<string> unuse_uncorrelated_variables(float = 0.25f) { return {}; }
    virtual vector<string> unuse_collinear_variables(float = 0.9f) { return {}; }
    virtual vector<Descriptives> calculate_variable_descriptives_positive_samples() const { return {}; }
    virtual vector<Descriptives> calculate_variable_descriptives_negative_samples() const { return {}; }
    virtual vector<Descriptives> calculate_variable_descriptives_categories(Index) const { return {}; }
    virtual VectorI filter_data(const VectorR&, const VectorR&) const { return {}; }

    bool has_categorical_variables() const
    {
        for(const auto& v : get_variables())
            if(v.type == VariableType::Categorical) return true;
        return false;
    }

    void set_separator(const Separator& new_separator) { separator = new_separator; }
    void set_separator_string(const string&);
    void set_separator_name(const string&);

    void set_codification(const Codification& new_codification) { codification = new_codification; }
    void set_codification(const string&);

    void set_display(bool new_display) { display = new_display; }

    bool is_sample_used(const Index i) const { return sample_roles[i] != SampleRole::None; }

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
                             bool,
                             int contiguous = -1) const;

    virtual void augment_inputs(float*, Index) const {}

    virtual void fill_decoder(const vector<Index>&,
                              const vector<Index>&,
                              float*,
                              bool,
                              int contiguous = -1) const;

    virtual void fill_targets(const vector<Index>&,
                              const vector<Index>&,
                              float*,
                              bool,
                              int contiguous = -1) const;

    virtual bool supports_bf16_inputs() const { return true; }

    virtual void fill_batch(Batch&,
                            const vector<Index>&,
                            const vector<Index>&,
                            const vector<Index>&,
                            const vector<Index>&,
                            bool) const;

protected:

    Dataset() = default;

    void fill_batch_host(Batch&,
                         const vector<Index>&,
                         const vector<Index>&,
                         const vector<Index>&,
                         const vector<Index>&,
                         bool) const;

    void set_default_variable_roles();
    void set_default_variable_roles_forecasting();
    void set_default_variable_roles_implementation(bool);

    void read_data_file_preview(const vector<string_view>&, char);
    void check_separators(string_view) const;
    void samples_from_JSON(const Json*);
    virtual void resize_data_from_JSON(Index) {}

    virtual void mark_data_changed() const {}

    StorageMode storage_mode = StorageMode::Matrix;

    void upload_device_matrix(const MatrixR&);

    MatrixR data;

    Buffer data_device{Device::CUDA};
    Index device_data_columns = 0;

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
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
