//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/dataset/batch.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/enum_map.h"
#include "opennn/core/scaling.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/variable.h"

namespace opennn
{

enum class SampleRole{Training, Validation, Testing, None};

inline const EnumMap<SampleRole>& sample_role_map()
{
    static const EnumMap<SampleRole> map{
        {SampleRole::Training,   "Training"},
        {SampleRole::Validation, "Validation"},
        {SampleRole::Testing,    "Testing"},
        {SampleRole::None,       "None"}
    };
    return map;
}

inline const string& sample_role_to_string(SampleRole role)
{
    return sample_role_map().to_string(role);
}

inline SampleRole string_to_sample_role(string_view name)
{
    if (name == "0") return SampleRole::Training;
    if (name == "1") return SampleRole::Validation;
    if (name == "2") return SampleRole::Testing;
    if (name == "3") return SampleRole::None;
    return sample_role_map().from_string(name);
}

enum class Shuffle { No, Yes };

class Dataset
{

public:

    enum class Codification { UTF8, SHIFT_JIS };

    virtual ~Dataset() = default;

    enum class Separator{Space, Tab, Comma, Semicolon};
    enum class StorageMode{Matrix, BinaryFile, GPUPersistantData};

    virtual Index get_samples_number() const noexcept { return ssize(sample_roles); }

    Index get_samples_number(SampleRole role_type) const { return ranges::count(active_sample_roles(), role_type); }
    Index get_samples_number(string_view role) const { return get_samples_number(string_to_sample_role(role)); }

    Index get_used_samples_number() const { return get_samples_number() - get_samples_number(SampleRole::None); }

    vector<Index> get_sample_indices(SampleRole) const;
    vector<Index> get_sample_indices(string_view role) const { return get_sample_indices(string_to_sample_role(role)); }

    vector<Index> get_used_sample_indices() const;

    const vector<SampleRole>& get_sample_roles() const noexcept { return sample_roles; }

    void set_fold_split(const vector<Index>& training, const vector<Index>& validation);
    void clear_fold_split() noexcept { fold_split_roles.reset(); }

    Index get_variables_number() const noexcept { return variables.size(); }
    Index get_variables_number(VariableRole) const;
    Index get_variables_number(string_view role) const { return get_variables_number(string_to_variable_role(role)); }
    Index get_used_variables_number() const;

    const vector<Variable>& get_variables() const noexcept { return variables; }
    vector<Variable> get_variables(VariableRole) const;
    vector<Variable> get_variables(string_view role) const { return get_variables(string_to_variable_role(role)); }

    Index get_variable_index(const string&) const;
    Index get_variable_index(const Index) const;

    vector<Index> get_variable_indices(VariableRole) const;
    vector<Index> get_variable_indices(string_view role) const { return get_variable_indices(string_to_variable_role(role)); }
    vector<Index> get_used_variables_indices() const;

    vector<string> get_variable_names() const;
    vector<string> get_variable_names(VariableRole) const;
    vector<string> get_variable_names(string_view role) const { return get_variable_names(string_to_variable_role(role)); }

    VariableType get_variable_type(const Index index) const { return variables[index].type; }

    Index get_features_number() const;
    Index get_features_number(VariableRole) const;
    Index get_features_number(string_view role) const { return get_features_number(string_to_variable_role(role)); }

    vector<string> get_feature_names() const;
    vector<string> get_feature_names(VariableRole) const;
    vector<string> get_feature_names(string_view role) const { return get_feature_names(string_to_variable_role(role)); }

    virtual vector<Variable> get_model_input_variables() const
    {
        return get_variables(VariableRole::Input);
    }

    virtual bool sample_order_matters() const noexcept { return false; }

    virtual MatrixR calculate_input_target_correlation_values() const { throw runtime_error("Dataset does not support input-target correlations."); }
    virtual FeatureScaling calculate_used_feature_scaling(VariableRole) const;

    vector<vector<Index>> get_feature_indices() const;
    vector<Index> get_feature_indices(const Index) const;
    vector<Index> get_feature_indices(VariableRole) const;
    vector<Index> get_feature_indices(string_view role) const { return get_feature_indices(string_to_variable_role(role)); }
    vector<Index> get_used_feature_indices() const;

    Shape get_shape(VariableRole) const;
    Shape get_shape(string_view role) const { return get_shape(string_to_variable_role(role)); }

    void get_batches(const vector<Index>&, Index, Shuffle, vector<vector<Index>>&,
                     optional<unsigned> shuffle_seed = nullopt) const;

    const vector<vector<string>>& get_data_file_preview() const noexcept { return data_file_preview; }

    const filesystem::path& get_data_path() const noexcept { return data_path; }

    void set_cache_directory(const filesystem::path& new_cache_directory) { cache_directory = new_cache_directory; }

    StorageMode get_storage_mode() const noexcept { return storage_mode; }
    string get_storage_mode_string() const;

    string get_separator_string() const;
    string get_separator_name() const;

    string get_codification_string() const;

    bool get_display() const noexcept { return display; }

    bool is_empty() const noexcept { return get_samples_number() == 0; }

    Shape get_input_shape() const noexcept { return input_shape; }
    Shape get_target_shape() const noexcept { return target_shape; }

    void record_memory(const string& stage) const;

    const MatrixR& get_data() const noexcept { return data; }
    void set_data(const MatrixR&);
    void set_data(MatrixR&&);
    void set_data_constant(float new_value) { data.setConstant(new_value); }

    virtual void enable_device_residency();
    void disable_device_residency() { data_device.resize_bytes(0, Device::CUDA); }
    bool is_device_resident() const noexcept { return data_device.data() != nullptr; }
    bool requests_device_residency() const noexcept
    {
        return storage_mode == StorageMode::GPUPersistantData;
    }
    bool uses_device_residency() const noexcept
    {
        return is_device_resident();
    }
    const float* get_device_data() const { return data_device.as<float>(); }
    Index get_device_data_columns() const noexcept { return device_data_columns; }

    virtual FeatureScaling prepare_training_scaling(
        VariableRole,
        const FeatureScaling&,
        Index);
    virtual void clear_training_scaling() noexcept {}

    void set_sample_roles(SampleRole);
    void set_sample_roles(string_view role) { set_sample_roles(string_to_sample_role(role)); }

    void set_sample_role(Index, SampleRole);
    void set_sample_role(Index index, string_view role) { set_sample_role(index, string_to_sample_role(role)); }

    void set_sample_roles(const vector<string>&);
    void set_sample_roles(const vector<Index>&, SampleRole);
    void set_sample_roles(const vector<Index>& indices, string_view role)
    {
        set_sample_roles(indices, string_to_sample_role(role));
    }

    void set_default_variable_names();
    void set_default_variable_roles() { set_default_variable_roles_implementation(false); }

    void set_variable_roles(const vector<string>&);

    void set_variable_indices(const vector<Index>&, const vector<Index>&);
    void set_input_variables_unused();

    void set_variable_role(Index, VariableRole);
    void set_variable_role(Index index, string_view role) { set_variable_role(index, string_to_variable_role(role)); }
    void set_variable_role(const string&, VariableRole);
    void set_variable_role(const string& name, string_view role)
    {
        set_variable_role(name, string_to_variable_role(role));
    }

    void set_variable_names(const vector<string>&);

    void set_variables_number(const Index new_size) { variables.resize(new_size); }

    void set_variable_roles(VariableRole);
    void set_variable_roles(string_view role) { set_variable_roles(string_to_variable_role(role)); }

    void set_shape(VariableRole, const Shape&);
    void set_shape(string_view role, const Shape& shape) { set_shape(string_to_variable_role(role), shape); }
    virtual void resize_input_shape(Index input_features_count) { set_shape(VariableRole::Input, {input_features_count}); }
    virtual void set_data_path(const filesystem::path& new_data_path) { data_path = new_data_path; }
    virtual void set_storage_mode(StorageMode);
    virtual void set_storage_mode(const string&);

    void set_has_header(bool new_has_header) { has_header = new_has_header; }
    void set_has_ids(bool new_has_ids) { has_sample_ids = new_has_ids; }

    bool get_header_line() const { return has_header; }
    bool get_has_sample_ids() const { return has_sample_ids; }
    const vector<string>& get_sample_ids() const { return sample_ids; }

    VectorI filter_data(const VectorR&, const VectorR&);

    bool has_categorical_variables() const
    {
        return ranges::any_of(get_variables(),
            [](const Variable& variable) { return variable.type == VariableType::Categorical; });
    }

    void set_separator(const Separator& new_separator) { separator = new_separator; }
    void set_separator_string(const string&);
    void set_separator_name(const string&);

    void set_codification(const Codification& new_codification) { codification = new_codification; }
    void set_codification(const string&);

    void set_display(bool new_display) { display = new_display; }

    bool has_validation() const { return get_samples_number(SampleRole::Validation) != 0; }

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

    virtual void from_JSON(const JsonDocument&) = 0;
    virtual void to_JSON(JsonWriter&) const {}

    virtual bool has_nan() const { return false; }

    virtual void scrub_missing_values() {}

    virtual VectorI calculate_target_distribution() const { return {}; }

    pair<Index, Index> count_binary_targets(const string& sample_role) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    virtual void fill_inputs(const vector<Index>&,
                             const vector<Index>&,
                             float*,
                             FillMode,
                             ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const;

    virtual void fill_decoder(const vector<Index>&,
                              const vector<Index>&,
                              float*,
                              FillMode,
                              ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const;

    virtual void fill_targets(const vector<Index>&,
                              const vector<Index>&,
                              float*,
                              FillMode,
                              ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const;

    virtual bool supports_bf16_inputs() const { return true; }

    FeatureSelection get_feature_selection() const;

    virtual void fill_batch(Batch&,
                            const vector<Index>& sample_indices,
                            const FeatureSelection&,
                            FillMode) const;

protected:

    Dataset() = default;

    void fill_batch_host(Batch&,
                         const vector<Index>& sample_indices,
                         const FeatureSelection&,
                         FillMode) const;

    bool can_device_gather(const Batch&, const FeatureSelection&) const;

    DeviceGather& start_device_gather(Batch&,
                                      const vector<Index>& sample_indices,
                                      const FeatureSelection&) const;

    void set_default_variable_roles_forecasting() { set_default_variable_roles_implementation(true); }
    void set_default_variable_roles_implementation(bool forecasting);

    void read_data_file_preview(const vector<string_view>&, char, bool has_quotes = false);
    void check_separators(string_view) const;
    void samples_from_JSON(const Json*);
    virtual void resize_data_from_JSON(Index) {}
    virtual void on_used_samples_changed() {}

    void require_in_memory_data(string_view what) const
    {
        throw_if(storage_mode == StorageMode::BinaryFile,
                 "{} is not available with BinaryFile storage; it needs the data matrix in memory.",
                 what);
    }

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

    optional<vector<SampleRole>> fold_split_roles;

    const vector<SampleRole>& active_sample_roles() const noexcept
    { return fold_split_roles ? *fold_split_roles : sample_roles; }

    vector<Variable> variables;

    filesystem::path data_path;

    filesystem::path cache_directory;

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

    virtual void missing_values_from_JSON(const Json*) {}

    void write_json_header(JsonWriter&, initializer_list<pair<const char*, Json>>) const;

    void write_json_footer(JsonWriter&) const;

    void read_json_blocks(const Json*);
};

struct FoldScope
{
    Dataset& dataset;

    FoldScope(Dataset& dataset, const vector<Index>& training, const vector<Index>& validation)
        : dataset(dataset) { dataset.set_fold_split(training, validation); }

    ~FoldScope() { dataset.clear_fold_split(); }

    FoldScope(const FoldScope&) = delete;
    FoldScope& operator=(const FoldScope&) = delete;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
