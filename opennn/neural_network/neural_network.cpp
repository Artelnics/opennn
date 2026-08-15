//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/neural_network.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <utility>

#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/kernel_tensor.cuh"
#include "opennn/core/enum_map.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/variable.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/model_expression.h"
#include "opennn/neural_network/operators/combination_operator.h"
#include "opennn/registry.h"

namespace opennn
{

namespace
{

// Binary snapshots written by OpenNN carry this fixed-width little-endian
// header. The loaders also accept their historical headerless FP32 payloads
// when the byte count matches exactly.
using SnapshotMagic = array<unsigned char, 8>;
constexpr SnapshotMagic PARAMETER_FILE_MAGIC = {
    'O', 'P', 'E', 'N', 'N', 'N', 'P', 0
};
constexpr SnapshotMagic STATE_FILE_MAGIC = {
    'O', 'P', 'E', 'N', 'N', 'N', 'S', 0
};
constexpr uint32_t SNAPSHOT_FILE_VERSION = 1;
constexpr uint32_t SNAPSHOT_FILE_HEADER_SIZE = 56;
constexpr uint32_t SNAPSHOT_FILE_ENDIAN_MARKER = 0x01020304;
constexpr uint32_t SNAPSHOT_FILE_SCALAR_FP32 = 1;
constexpr uint64_t FNV1A_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV1A_PRIME = 1099511628211ULL;
constexpr string_view SAVE_TRANSACTION_MAGIC = "OPENNN_SAVE_TRANSACTION_V1";

filesystem::path parameter_file_path(const filesystem::path& model_path)
{
    filesystem::path parameter_path = model_path;
    parameter_path.replace_extension(".bin");
    return parameter_path;
}

filesystem::path append_path_suffix(filesystem::path path, string_view suffix)
{
    path += suffix;
    return path;
}

void validate_model_file_paths(const filesystem::path& model_path)
{
    throw_if(model_path.empty(), "NeuralNetwork: the model file path is empty.");

    const filesystem::path parameter_path = parameter_file_path(model_path);
    throw_if(model_path == parameter_path
             || ascii_lowercase(model_path.extension().string()) == ".bin",
             "NeuralNetwork: the model file must not use the .bin extension because "
             "that path is reserved for its parameter snapshot.");

    throw_if(filesystem::exists(model_path) && filesystem::is_directory(model_path),
             "NeuralNetwork: the model path is a directory: {}.", model_path.string());
    throw_if(filesystem::exists(parameter_path) && filesystem::is_directory(parameter_path),
             "NeuralNetwork: the parameter path is a directory: {}.",
             parameter_path.string());
}

void remove_transaction_artifact(const filesystem::path& path)
{
    if (!filesystem::exists(path)) return;

    throw_if(filesystem::is_directory(path),
             "Cannot remove model save transaction artifact because it is a directory: {}.",
             path.string());
    filesystem::remove(path);
}

void remove_transaction_artifacts(initializer_list<filesystem::path> paths)
{
    for (const filesystem::path& path : paths) remove_transaction_artifact(path);
}

void write_save_transaction_marker(const filesystem::path& marker_path,
                                   bool had_model, bool had_parameters)
{
    const filesystem::path temporary_marker = append_path_suffix(marker_path, ".tmp");
    remove_transaction_artifact(temporary_marker);

    ofstream marker(temporary_marker, ios::trunc);
    throw_if(!marker.is_open(), "Cannot create save transaction marker: {}.",
             temporary_marker.string());
    marker << SAVE_TRANSACTION_MAGIC << '\n'
           << int(had_model) << ' ' << int(had_parameters) << '\n';
    marker.close();
    throw_if(!marker, "Cannot write save transaction marker: {}.",
             temporary_marker.string());

    filesystem::rename(temporary_marker, marker_path);
}

pair<bool, bool> read_save_transaction_marker(const filesystem::path& marker_path)
{
    ifstream marker(marker_path);
    throw_if(!marker.is_open(), "Cannot open save transaction marker: {}.",
             marker_path.string());

    string magic;
    int model_flag = -1;
    int parameter_flag = -1;
    marker >> magic >> model_flag >> parameter_flag;
    throw_if(!marker || magic != SAVE_TRANSACTION_MAGIC
             || (model_flag != 0 && model_flag != 1)
             || (parameter_flag != 0 && parameter_flag != 1),
             "Invalid save transaction marker: {}.", marker_path.string());

    return {model_flag != 0, parameter_flag != 0};
}

void restore_transaction_file(const filesystem::path& final_path,
                              const filesystem::path& backup_path,
                              bool existed_before_transaction)
{
    if (!existed_before_transaction)
    {
        remove_transaction_artifact(final_path);
        remove_transaction_artifact(backup_path);
        return;
    }

    if (filesystem::exists(backup_path))
    {
        remove_transaction_artifact(final_path);
        filesystem::rename(backup_path, final_path);
        return;
    }

    throw_if(!filesystem::exists(final_path),
             "Cannot recover interrupted model save: both {} and its backup are missing.",
             final_path.string());
}

void recover_model_save_transaction(const filesystem::path& model_path)
{
    const filesystem::path parameter_path = parameter_file_path(model_path);
    const filesystem::path model_temporary = append_path_suffix(model_path, ".tmp");
    const filesystem::path parameter_temporary = append_path_suffix(parameter_path, ".tmp");
    const filesystem::path model_backup = append_path_suffix(model_path, ".bak");
    const filesystem::path parameter_backup = append_path_suffix(parameter_path, ".bak");
    const filesystem::path marker_path = append_path_suffix(model_path, ".save-transaction");
    const filesystem::path marker_temporary = append_path_suffix(marker_path, ".tmp");

    if (filesystem::exists(marker_path))
    {
        const auto [had_model, had_parameters] =
            read_save_transaction_marker(marker_path);

        restore_transaction_file(model_path, model_backup, had_model);
        restore_transaction_file(parameter_path, parameter_backup, had_parameters);
        remove_transaction_artifacts({model_temporary, parameter_temporary,
                                      marker_temporary, marker_path});
        return;
    }

    // A missing marker means that no replacement started, or that both new
    // files committed. Any remaining siblings are therefore stale.
    remove_transaction_artifacts({model_temporary, parameter_temporary,
                                  model_backup, parameter_backup,
                                  marker_temporary});
}

uint64_t hash_bytes(uint64_t hash, const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= FNV1A_PRIME;
    }
    return hash;
}

uint64_t hash_uint64(uint64_t hash, uint64_t value)
{
    for (int shift = 0; shift < 64; shift += 8)
    {
        const unsigned char byte = static_cast<unsigned char>(value >> shift);
        hash = hash_bytes(hash, &byte, 1);
    }
    return hash;
}

uint64_t hash_string(uint64_t hash, string_view value)
{
    hash = hash_uint64(hash, value.size());
    return hash_bytes(hash, value.data(), value.size());
}

uint64_t hash_shape(uint64_t hash, const Shape& shape)
{
    hash = hash_uint64(hash, shape.rank);
    for (Index dimension : shape)
        hash = hash_uint64(hash, static_cast<uint64_t>(dimension));
    return hash;
}

uint64_t hash_layer_layout(uint64_t hash, const Layer& layer,
                           span<const Index> sources,
                           const vector<TensorSpec>& specs)
{
    hash = hash_string(hash, layer.get_name());
    hash = hash_shape(hash, layer.get_input_shape());
    hash = hash_shape(hash, layer.get_output_shape());

    hash = hash_uint64(hash, specs.size());
    for (const TensorSpec& spec : specs)
        hash = hash_shape(hash, spec.shape);

    hash = hash_uint64(hash, sources.size());
    for (Index source : sources)
        hash = hash_uint64(hash, static_cast<uint64_t>(source));
    return hash;
}

Index find_layer_index(const vector<unique_ptr<Layer>>& layers, const Layer* target)
{
    const auto found = ranges::find(layers, target, &unique_ptr<Layer>::get);
    return found == layers.end() ? Index(-1) : Index(found - layers.begin());
}

void store_uint32_le(unsigned char* destination, uint32_t value)
{
    for (int i = 0; i < 4; ++i)
        destination[i] = static_cast<unsigned char>(value >> (8 * i));
}

void store_uint64_le(unsigned char* destination, uint64_t value)
{
    for (int i = 0; i < 8; ++i)
        destination[i] = static_cast<unsigned char>(value >> (8 * i));
}

uint32_t load_uint32_le(const unsigned char* source)
{
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i)
        value |= uint32_t(source[i]) << (8 * i);
    return value;
}

uint64_t load_uint64_le(const unsigned char* source)
{
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= uint64_t(source[i]) << (8 * i);
    return value;
}

array<unsigned char, SNAPSHOT_FILE_HEADER_SIZE> make_snapshot_header(
    const SnapshotMagic& magic, uint64_t elements, uint64_t payload_bytes,
    uint64_t layout, uint64_t checksum)
{
    array<unsigned char, SNAPSHOT_FILE_HEADER_SIZE> header{};
    ranges::copy(magic, header.begin());
    store_uint32_le(header.data() + 8, SNAPSHOT_FILE_VERSION);
    store_uint32_le(header.data() + 12, SNAPSHOT_FILE_HEADER_SIZE);
    store_uint32_le(header.data() + 16, SNAPSHOT_FILE_ENDIAN_MARKER);
    store_uint32_le(header.data() + 20, SNAPSHOT_FILE_SCALAR_FP32);
    store_uint64_le(header.data() + 24, elements);
    store_uint64_le(header.data() + 32, payload_bytes);
    store_uint64_le(header.data() + 40, layout);
    store_uint64_le(header.data() + 48, checksum);
    return header;
}

bool is_versioned_snapshot(ifstream& file, uintmax_t file_bytes,
                           uint64_t legacy_payload_bytes,
                           const SnapshotMagic& magic,
                           const filesystem::path& file_name,
                           const char* caller)
{
    SnapshotMagic leading_bytes{};
    bool has_versioned_magic = false;
    if (file_bytes >= leading_bytes.size())
    {
        file.read(reinterpret_cast<char*>(leading_bytes.data()), leading_bytes.size());
        throw_if(!file, "NeuralNetwork::{}: cannot inspect {}.",
                 caller, file_name.string());
        has_versioned_magic = ranges::equal(magic, leading_bytes);
        file.seekg(0);
    }

    return has_versioned_magic || file_bytes != legacy_payload_bytes;
}

uint64_t read_snapshot_header(ifstream& file, uintmax_t file_bytes,
                              const SnapshotMagic& magic,
                              uint64_t expected_elements,
                              uint64_t expected_payload_bytes,
                              uint64_t expected_layout,
                              const filesystem::path& file_name,
                              const char* caller, const char* snapshot_name)
{
    throw_if(file_bytes < SNAPSHOT_FILE_HEADER_SIZE,
             "NeuralNetwork::{}: size mismatch for {} "
             "(got {} bytes, expected {} legacy bytes or at least {} versioned bytes).",
             caller, file_name.string(), file_bytes, expected_payload_bytes,
             SNAPSHOT_FILE_HEADER_SIZE);

    array<unsigned char, SNAPSHOT_FILE_HEADER_SIZE> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    throw_if(!file, "NeuralNetwork::{}: cannot read header from {}.",
             caller, file_name.string());

    throw_if(!ranges::equal(magic, span(header).first(magic.size())),
             "NeuralNetwork::{}: {} has an unrecognized header and is not a "
             "legacy raw snapshot of the expected size.",
             caller, file_name.string());

    const uint32_t version = load_uint32_le(header.data() + 8);
    const uint32_t header_size = load_uint32_le(header.data() + 12);
    const uint32_t endian_marker = load_uint32_le(header.data() + 16);
    const uint32_t scalar_type = load_uint32_le(header.data() + 20);
    const uint64_t stored_elements = load_uint64_le(header.data() + 24);
    const uint64_t stored_payload_bytes = load_uint64_le(header.data() + 32);
    const uint64_t stored_layout = load_uint64_le(header.data() + 40);
    const uint64_t stored_checksum = load_uint64_le(header.data() + 48);

    throw_if(version != SNAPSHOT_FILE_VERSION,
             "NeuralNetwork::{}: unsupported {} file version {} in {} "
             "(supported version {}).",
             caller, snapshot_name, version, file_name.string(),
             SNAPSHOT_FILE_VERSION);
    throw_if(header_size != SNAPSHOT_FILE_HEADER_SIZE,
             "NeuralNetwork::{}: invalid version-{} header size {} in {}.",
             caller, version, header_size, file_name.string());
    throw_if(endian_marker != SNAPSHOT_FILE_ENDIAN_MARKER,
             "NeuralNetwork::{}: unsupported byte order in {}.",
             caller, file_name.string());
    throw_if(scalar_type != SNAPSHOT_FILE_SCALAR_FP32,
             "NeuralNetwork::{}: unsupported scalar type {} in {}.",
             caller, scalar_type, file_name.string());
    throw_if(stored_elements != expected_elements
             || stored_payload_bytes != expected_payload_bytes,
             "NeuralNetwork::{}: payload size mismatch for {} "
             "(file has {} FP32 elements/{} bytes, network expects {}/{}).",
             caller, file_name.string(), stored_elements, stored_payload_bytes,
             expected_elements, expected_payload_bytes);
    throw_if(file_bytes != uintmax_t(header_size) + stored_payload_bytes,
             "NeuralNetwork::{}: file size mismatch for {} "
             "(got {} bytes, header describes {}).",
             caller, file_name.string(), file_bytes,
             uintmax_t(header_size) + stored_payload_bytes);
    throw_if(stored_layout != expected_layout,
             "NeuralNetwork::{}: {} layout mismatch for {} "
             "(file {:016x}, network {:016x}).",
             caller, snapshot_name, file_name.string(), stored_layout,
             expected_layout);

    return stored_checksum;
}

const EnumMap<NetworkTask>& network_task_map()
{
    static const vector<EnumMap<NetworkTask>::Entry> entries = {
        {NetworkTask::Generic,             "Generic"},
        {NetworkTask::Approximation,       "Approximation"},
        {NetworkTask::Classification,      "Classification"},
        {NetworkTask::Forecasting,         "Forecasting"},
        {NetworkTask::AutoAssociation,     "AutoAssociation"},
        {NetworkTask::ImageClassification, "ImageClassification"},
        {NetworkTask::ObjectDetection,     "ObjectDetection"},
        {NetworkTask::TextClassification,  "TextClassification"},
        {NetworkTask::LanguageModeling,    "LanguageModeling"}
    };

    static const EnumMap<NetworkTask> map{entries};
    return map;
}

void wire_drelu_fusions(vector<unique_ptr<Layer>>& layers,
                        const vector<vector<Index>>& source_layers,
                        Device device,
                        Type training_type)
{
    for (auto& layer : layers)
        if (auto* dense = dynamic_cast<Dense*>(layer.get()))
            dense->reset_drelu_fusion();

    if (device != Device::CUDA || training_type != Type::FP32)
        return;

    if (!env_flag_enabled("OPENNN_DRELU_FUSION"))
        return;

    vector<Index> consumer_count(layers.size(), 0);
    for (const auto& layer_sources : source_layers)
        for (Index source : layer_sources)
            if (source >= 0) ++consumer_count[size_t(source)];

    for (size_t i = 0; i < source_layers.size(); ++i)
    {
        const auto& sources = source_layers[i];
        if (sources.size() != 1 || sources[0] < 0) continue;
        if (consumer_count[size_t(sources[0])] != 1) continue;

        auto* consumer = dynamic_cast<Dense*>(layers[i].get());
        auto* producer = dynamic_cast<Dense*>(layers[size_t(sources[0])].get());

        if (consumer && producer)
            consumer->try_wire_drelu_fusion(*producer);
    }
}

}

static void validate_source_indices(const vector<Index>&, Index, Index);
static void validate_source_arity(const Layer&, const vector<Index>&, Index);

NeuralNetwork::NeuralNetwork()
    : NeuralNetwork(NetworkTask::Generic)
{
}

NeuralNetwork::NeuralNetwork(NetworkTask new_task)
    : task(new_task)
{
    clear();
}

NeuralNetwork::NeuralNetwork(const filesystem::path& file_name)
    : NeuralNetwork(file_name, NetworkTask::Generic)
{
}

NeuralNetwork::NeuralNetwork(const filesystem::path& file_name, NetworkTask new_task)
    : NeuralNetwork(new_task)
{
    load(file_name);
}

void NeuralNetwork::add_layer(unique_ptr<Layer> layer, const vector<Index>& sources)
{
    throw_if(!layer, "NeuralNetwork: cannot add a null layer.");

    const Index old_layers_number = get_layers_number() - 1;

    if (!layers.empty())
        throw_if(!layers.back()->allows_successors(),
                 "No layers can be added after a {} layer.\n",
                 layers.back()->get_name());

    const vector<Index> resolved_sources = sources.empty()
        ? vector<Index>{old_layers_number}
        : sources;

    validate_source_indices(resolved_sources, ssize(layers), ssize(layers));
    validate_source_arity(*layer, resolved_sources, ssize(layers));

    layers.push_back(std::move(layer));

    source_layers.push_back(resolved_sources);

    first_trainable_cache_ = -1;
    last_trainable_cache_  = -1;
}

void NeuralNetwork::compile()
{
    if (get_layers_number() == 0) return;
    compile(Configuration::instance().resolve());
}

void NeuralNetwork::compile(const Device device)
{
    if (get_layers_number() == 0) return;
    compile(Configuration::instance().resolve_for(device));
}

void NeuralNetwork::compile(Configuration::Resolved new_config)
{
    config = new_config;

    stale_configuration_warned = false;

    for (auto& layer : layers)
    {
        layer->set_compute_device(get_device());
        layer->set_compute_dtype(get_training_type());
    }

    parameters.resize_bytes(get_aligned_bytes(get_parameter_specs(), Type::FP32), Device::CPU);
    parameters.setZero();

    clear_low_precision_parameter_storage();

    link_parameters();

    states.resize_bytes(get_states_size() * Index(sizeof(float)), Device::CPU);
    states.setZero();

    link_states();

    wire_drelu_fusions(layers, source_layers, get_device(), get_training_type());
}

void NeuralNetwork::clear_low_precision_parameter_storage()
{
    parameters_bf16_mirror.resize_bytes(0, Device::CUDA);
    parameters_bf16_mirror_compact = false;
    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
    parameters_int8_storage.resize_bytes(0, Device::CUDA);
}

void NeuralNetwork::warn_if_stale_configuration() const
{
    if (stale_configuration_warned
        || config.generation == Configuration::instance().get_generation())
        return;

    stale_configuration_warned = true;

    cerr << "Warning: Configuration::set() was called after this network was compiled, "
            "so it has no effect on it. The network keeps the settings resolved at "
            "compile() time; call Configuration::set() before constructing the network.\n";
}

bool NeuralNetwork::has(const string& name) const
{
    return has(string_to_layer_type(name));
}

bool NeuralNetwork::has(LayerType type) const
{
    return ranges::any_of(layers,
                          [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

bool NeuralNetwork::has_recurrent_layers() const
{
    return ranges::any_of(layers, [](const unique_ptr<Layer>& layer)
    {
        return is_one_of(layer->get_type(), LayerType::Recurrent,
                         LayerType::LongShortTermMemory);
    });
}

bool NeuralNetwork::supports_compact_cnn_memory_layout() const noexcept
{
    return ranges::all_of(
        layers,
        [](const unique_ptr<Layer>& layer)
        {
            const LayerType type = layer->get_type();
            return is_one_of(type, LayerType::Scaling, LayerType::Convolutional,
                             LayerType::Pooling, LayerType::Flatten, LayerType::Dense);
        });
}

vector<string> NeuralNetwork::get_input_feature_names() const
{
    return get_variable_feature_names(input_variables);
}

vector<string> NeuralNetwork::get_output_feature_names() const
{
    return get_variable_feature_names(output_variables);
}

const unique_ptr<Layer>& NeuralNetwork::get_layer(const string& label) const
{
    auto it = ranges::find_if(layers,
                              [&label](const unique_ptr<Layer>& layer) { return layer->get_label() == label; });

    if (it != layers.end())
        return *it;

    throw runtime_error("Layer not found in neural network");
}

Index NeuralNetwork::get_layer_index(const string& new_label) const
{
    if (contains({"Dataset", "decoder"}, new_label))
        return -1;

    if (new_label == "input")
        return -2;

    auto it = ranges::find_if(layers,
                              [&new_label](const unique_ptr<Layer>& layer) { return layer->get_label() == new_label; });

    if (it != layers.end())
        return distance(layers.begin(), it);

    throw runtime_error(format("Layer not found: {}", new_label));
}

const Layer* NeuralNetwork::get_first(const string& name) const
{
    return get_first(string_to_layer_type(name));
}

const Layer* NeuralNetwork::get_first(LayerType type) const
{
    auto it = ranges::find_if(layers,
                              [type](const unique_ptr<Layer>& layer) { return layer->get_type() == type; });

    return it != layers.end() ? it->get() : nullptr;
}

Layer* NeuralNetwork::get_first(const string& name)
{
    return get_first(string_to_layer_type(name));
}

Layer* NeuralNetwork::get_first(LayerType type)
{
    return const_cast<Layer*>(static_cast<const NeuralNetwork*>(this)->get_first(type));
}

static void define_variables_from_names(vector<Variable>& variables,
                                        const vector<string>& names,
                                        VariableRole role)
{
    variables.assign(names.size(), Variable());

    for (size_t i = 0; i < names.size(); ++i)
    {
        variables[i].name = names[i];
        variables[i].role = role;
        variables[i].type = VariableType::Numeric;
    }
}

static void set_variable_names(vector<Variable>& variables, const vector<string>& new_names)
{

    if (ranges::any_of(variables,
                       [](const Variable& v) { return !v.is_categorical() && v.features > 1; }))
        return define_variables_from_names(variables, new_names,
                                           variables.empty() ? VariableRole::None : variables[0].role);

    const size_t total = new_names.size();
    size_t name_index = 0;
    for (size_t i = 0; i < variables.size(); ++i)
    {
        if (variables[i].is_categorical())
        {
            const size_t num_cats = variables[i].get_categories_number();
            throw_if(name_index + num_cats > total,
                     "set_variable_names: not enough names for categorical variable {} (need {}, have {}).",
                            i, num_cats, total - name_index);
            variables[i].categories.assign(new_names.begin() + name_index,
                                           new_names.begin() + name_index + num_cats);
            name_index += num_cats;
        }
        else
        {
            throw_if(name_index >= total,
                     "set_variable_names: not enough names for scalar variable {}.", i);
            variables[i].name = new_names[name_index];
            ++name_index;
        }
    }

    throw_if(name_index != total,
             "set_variable_names: received {} names but variables expected {}.",
                    total, name_index);
}

void NeuralNetwork::set_input_names(const vector<string>& new_input_names)
{
    if (input_variables.empty() && !new_input_names.empty())
        return define_variables_from_names(input_variables, new_input_names, VariableRole::Input);

    set_variable_names(input_variables, new_input_names);
}

void NeuralNetwork::set_output_names(const vector<string>& new_output_names)
{
    if (output_variables.empty() && !new_output_names.empty())
        return define_variables_from_names(output_variables, new_output_names, VariableRole::Target);

    set_variable_names(output_variables, new_output_names);
}

void NeuralNetwork::set_input_shape(const Shape& new_input_shape)
{
    if (get_features_number(input_variables) != new_input_shape.size())
    {
        input_variables.assign(1, Variable());
        input_variables[0].name = "input";
        input_variables[0].role = VariableRole::Input;
        input_variables[0].type = VariableType::Numeric;
        input_variables[0].features = new_input_shape.size();
    }

    if (Layer* scaling = get_first(LayerType::Scaling))
        scaling->set_input_shape(new_input_shape);

    layers[get_first_trainable_layer_index()]->set_input_shape(new_input_shape);

    const Index layers_number = get_layers_number();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        if (sources.size() == 1 && sources[0] >= 0)
            layers[i]->set_input_shape(layers[sources[0]]->get_output_shape());
    }
}

void NeuralNetwork::clear()
{
    layers.clear();

    source_layers.clear();

    input_variables.clear();

    output_variables.clear();

    first_trainable_cache_ = -1;
    last_trainable_cache_  = -1;
}

void NeuralNetwork::steal_from(NeuralNetwork& src)
{
    clear();
    task             = src.task;
    layers           = std::move(src.layers);
    source_layers    = std::move(src.source_layers);
    input_variables  = std::move(src.input_variables);
    output_variables = std::move(src.output_variables);
    first_trainable_cache_ = src.first_trainable_cache_;
    last_trainable_cache_  = src.last_trainable_cache_;
    src.first_trainable_cache_ = -1;
    src.last_trainable_cache_  = -1;
    link_parameters();
}

static void validate_source_indices(const vector<Index>& sources, Index layer_index, Index layers_count)
{
    for (const Index src : sources | views::filter([](Index source) { return source >= 0; }))
        throw_if(src >= layers_count || src >= layer_index,
                 "NeuralNetwork: source index {} is not a previous layer for layer {}.", src, layer_index);
}

static void validate_source_arity(const Layer& layer,
                                  const vector<Index>& sources,
                                  Index layer_index)
{
    const Index expected_sources = layer.get_sources_number();

    throw_if(ssize(sources) != expected_sources,
             "NeuralNetwork: {} layer {} expects {} sources, got {}.",
             layer.get_name(), layer_index, expected_sources, sources.size());
}

Index NeuralNetwork::get_inputs_number() const
{
    if (layers.empty())
        return 0;

    if (get_first(LayerType::Embedding))
        return get_layer(0)->get_inputs_number();

    for (const LayerType type : {LayerType::Recurrent, LayerType::LongShortTermMemory})
        if (const Layer* layer = get_first(type))
            return layer->get_input_shape()[1];

    return layers[0]->get_input_shape().size();
}

Index NeuralNetwork::get_outputs_number() const
{
    if (layers.empty()) return 0;

    return layers.back()->get_output_shape().size();
}

Shape NeuralNetwork::get_input_shape() const
{
    if (layers.empty())
        return {};

    return layers[0]->get_input_shape();
}

Shape NeuralNetwork::get_output_shape() const
{
    if (layers.empty())
        return {};

    return layers.back()->get_output_shape();
}

ActivationFunction NeuralNetwork::get_output_activation() const
{
    const Index last_index = get_last_trainable_layer_index();
    if (last_index < 0 || static_cast<size_t>(last_index) >= layers.size())
        return ActivationFunction::Identity;

    return layers[last_index]->get_output_activation();
}

Index NeuralNetwork::get_parameters_number() const
{
    return transform_reduce(layers.begin(), layers.end(), Index(0), plus<>{},
        [](const unique_ptr<Layer>& layer) { return layer->get_parameters_number(); });
}

uint64_t NeuralNetwork::parameter_layout_fingerprint() const
{
    uint64_t hash = hash_string(FNV1A_OFFSET, "OpenNN parameter layout v1");
    hash = hash_uint64(hash, layers.size());
    hash = hash_uint64(hash, static_cast<uint64_t>(get_parameters_buffer_size()));

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index)
    {
        const Layer& layer = *layers[layer_index];
        hash = hash_layer_layout(hash, layer, source_layers[layer_index],
                                 layer.get_parameter_specs());

        const Layer::TiedWeight tied_weight = layer.get_tied_weight();
        const Index tied_source_index = find_layer_index(layers, tied_weight.source);

        throw_if(tied_weight.source && tied_source_index == -1,
                 "NeuralNetwork::parameter_layout_fingerprint: tied weight source is not in the network.");

        hash = hash_uint64(hash, static_cast<uint64_t>(tied_source_index));
        hash = hash_uint64(hash, tied_weight.spec_index);
        hash = hash_uint64(hash, tied_weight.source_spec_index);
    }

    return hash;
}

uint64_t NeuralNetwork::state_layout_fingerprint() const
{
    uint64_t hash = hash_string(FNV1A_OFFSET, "OpenNN state layout v1");
    hash = hash_uint64(hash, layers.size());
    hash = hash_uint64(hash, static_cast<uint64_t>(get_states_buffer_size()));

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index)
        hash = hash_layer_layout(hash, *layers[layer_index],
                                 source_layers[layer_index],
                                 layers[layer_index]->get_state_specs());

    return hash;
}

Index NeuralNetwork::get_first_trainable_layer_index() const
{
    if (first_trainable_cache_ >= 0) return first_trainable_cache_;

    auto it = ranges::find_if(layers,
                              [](const unique_ptr<Layer>& layer) { return layer->get_is_trainable(); });

    // No trainable layers yet: return the -1 sentinel instead of throwing.
    // Callers guard with < 0 (e.g. get_output_activation here, and the Neural
    // Designer engine's resync); throwing turned that guard into dead code and
    // broke TrainingStrategy::set_default() on a not-yet-built network.
    if (it == layers.end()) return -1;

    first_trainable_cache_ = distance(layers.begin(), it);
    return first_trainable_cache_;
}

Index NeuralNetwork::get_last_trainable_layer_index() const
{
    if (last_trainable_cache_ >= 0) return last_trainable_cache_;

    const Index layers_number = get_layers_number();
    for (Index i = layers_number - 1; i >= 0; --i)
        if (layers[i]->get_is_trainable())
            return last_trainable_cache_ = i;

    // No trainable layers yet: return the -1 sentinel instead of throwing.
    // Callers guard with < 0 (get_output_activation, the ND engine's resync).
    return -1;
}

Index NeuralNetwork::get_layers_number(const string& name) const
{
    return get_layers_number(string_to_layer_type(name));
}

Index NeuralNetwork::get_layers_number(LayerType type) const
{
    return ranges::count_if(layers,
                            [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

static bool upload_host_vector(Buffer& buffer, const VectorR& values)
{
    const Index byte_count = values.size() * Index(sizeof(float));

    if (buffer.device_type == Device::CUDA)
    {
        buffer.resize_bytes(byte_count, Device::CUDA);
        if (byte_count > 0)
        {
            cudaStream_t stream = Backend::get_compute_stream();
            device::copy_async(buffer.data, values.data(), byte_count,
                               device::CopyKind::HostToDevice,
                               stream);
            device::synchronize(stream);
        }
        return true;
    }

    buffer.resize_bytes(byte_count, Device::CPU);
    if (byte_count > 0)
        memcpy(buffer.data, values.data(), static_cast<size_t>(byte_count));
    return false;
}

void NeuralNetwork::set_parameters(const VectorR& new_parameters)
{
    throw_if(new_parameters.size() == 0,
             "NeuralNetwork::set_parameters: refusing to apply an empty parameter vector.");

    const Index expected_size = get_parameters_buffer_size();
    throw_if(expected_size > 0 && new_parameters.size() != expected_size,
             "NeuralNetwork::set_parameters: size mismatch (got {}, expected {}). Make sure the network is compiled with the same architecture as the one that produced this snapshot.", new_parameters.size(), expected_size);

    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);

    if (upload_host_vector(parameters, new_parameters))
        cast_parameters_to_bf16();

    link_parameters();
}

void NeuralNetwork::set_states(const VectorR& new_states)
{
    const Index expected_size = get_states_buffer_size();

    if (expected_size == 0)
    {
        throw_if(new_states.size() != 0, "NeuralNetwork::set_states: network has no state buffer.");
        return;
    }

    throw_if(new_states.size() != expected_size,
             "NeuralNetwork::set_states: size mismatch (got {}, expected {}).", new_states.size(), expected_size);

    upload_host_vector(states, new_states);

    link_states();
}

void NeuralNetwork::initialize_parameters(void (Operator::*initializer)())
{
    const HostParametersGuard guard(*this);

    for (const auto& layer : layers)
        for (Operator* op : layer->get_operators())
            (op->*initializer)();
}

void NeuralNetwork::set_parameters_random()
{
    initialize_parameters(&Operator::set_parameters_random);
}

void NeuralNetwork::set_parameters_glorot()
{
    initialize_parameters(&Operator::set_parameters_glorot);
}

void NeuralNetwork::set_parameters_pytorch()
{
    initialize_parameters(&Operator::set_parameters_pytorch);
}

Tensor3 NeuralNetwork::calculate_outputs(const Tensor3& inputs_1, const Tensor3& inputs_2)
{
    const Index layers_number = get_layers_number();

    if (layers_number == 0)
        return {};

    warn_if_stale_configuration();

    const Index batch_size = inputs_1.dimension(0);

    ForwardPropagation forward_propagation(batch_size, this,
                                           ForwardPropagationMode::Inference);

    const vector<TensorView> input_views = {TensorView(const_cast<float*>(inputs_1.data()), {{inputs_1.dimension(0), inputs_1.dimension(1), inputs_1.dimension(2)}}),
                                            TensorView(const_cast<float*>(inputs_2.data()), {{inputs_2.dimension(0), inputs_2.dimension(1), inputs_2.dimension(2)}})};

    if (is_gpu())
    {
        const MatrixR result_matrix = calculate_outputs_device(input_views, forward_propagation);
        const TensorView out = forward_propagation.get_outputs();
        throw_if(out.shape.rank < 3,
                 "calculate_outputs(Tensor3, Tensor3): expected rank-3 output, got rank {}",
                        out.shape.rank);
        Tensor3 result(out.shape[0], out.shape[1], out.shape[2]);
        memcpy(result.data(), result_matrix.data(),
                    size_t(result.size()) * sizeof(float));
        return result;
    }

    forward_propagate(input_views, forward_propagation, false);

    return forward_propagation.get_outputs().as_tensor<3>();
}

MatrixR NeuralNetwork::calculate_outputs(const vector<TensorView>& input_views)
{
    if (layers.empty() || input_views.empty()) return {};

    warn_if_stale_configuration();

    const Index batch_size = input_views[0].shape[0];

    if (is_gpu())
    {
        ForwardPropagation forward_propagation(batch_size, this,
                                               ForwardPropagationMode::Inference);
        return calculate_outputs_device(input_views, forward_propagation);
    }

    constexpr Index tile_budget_bytes = Index(1024) * 1024 * 1024;

    const Index row_bytes = max(Index(1), get_aligned_bytes(get_forward_specs(1)));
    const Index tile_rows_max = clamp((tile_budget_bytes / row_bytes) & ~Index(15),
                                      Index(16), Index(65536));

    const bool tileable = batch_size > tile_rows_max
        && ranges::all_of(input_views,
            [batch_size](const TensorView& view)
            {
                return view.shape.rank >= 2
                    && view.shape[0] == batch_size
                    && view.is_fp32()
                    && !view.is_cuda();
            });

    if (!tileable)
    {
        ForwardPropagation forward_propagation(batch_size, this,
                                               ForwardPropagationMode::Inference);
        forward_propagate(input_views, forward_propagation, false);
        return forward_propagation.get_outputs().as_matrix();
    }

    ForwardPropagation tile_propagation(tile_rows_max, this,
                                        ForwardPropagationMode::Inference);
    unique_ptr<ForwardPropagation> tail_propagation;

    MatrixR outputs;

    for (Index start = 0; start < batch_size; start += tile_rows_max)
    {
        const Index rows = min(tile_rows_max, batch_size - start);

        ForwardPropagation* propagation = &tile_propagation;
        if (rows != tile_rows_max)
        {
            tail_propagation = make_unique<ForwardPropagation>(
                rows, this, ForwardPropagationMode::Inference);
            propagation = tail_propagation.get();
        }

        vector<TensorView> tile_views;
        tile_views.reserve(input_views.size());
        for (const TensorView& view : input_views)
        {
            Shape tile_shape = view.shape;
            tile_shape[0] = rows;
            const Index row_elements = view.size() / batch_size;
            tile_views.emplace_back(view.as<float>() + start * row_elements,
                                    tile_shape, Type::FP32);
        }

        forward_propagate(tile_views, *propagation, false);

        const TensorView tile_outputs = propagation->get_outputs();
        const Index output_columns = tile_outputs.size() / rows;
        if (outputs.size() == 0)
            outputs.resize(batch_size, output_columns);

        memcpy(outputs.data() + start * output_columns, tile_outputs.data,
               size_t(rows) * size_t(output_columns) * sizeof(float));
    }

    return outputs;
}

MatrixR NeuralNetwork::calculate_outputs(const MatrixR& inputs)
{
    return calculate_outputs(vector<TensorView>{TensorView(const_cast<float*>(inputs.data()), {inputs.rows(), inputs.cols()}, Type::FP32)});
}

MatrixR NeuralNetwork::calculate_outputs(const Tensor3& inputs)
{
    return calculate_outputs(vector<TensorView>{TensorView(const_cast<float*>(inputs.data()), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2)}, Type::FP32)});
}

MatrixR NeuralNetwork::calculate_outputs(const Tensor4& inputs)
{
    return calculate_outputs(vector<TensorView>{TensorView(const_cast<float*>(inputs.data()), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)}, Type::FP32)});
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      bool is_training) const
{
    throw_if(parameters.size_in_floats() != get_aligned_size(get_parameter_specs()),
             "Network shapes changed since compile(); call compile() again.");

    Index first_layer_index = 0;
    if (is_training || forward_propagation.inputs_pre_scaled)
        while (first_layer_index < get_layers_number()
               && layers[first_layer_index]->get_type() == LayerType::Scaling)
            ++first_layer_index;
    const Index last_layer_index = get_layers_number() - 1;

#ifdef OPENNN_HAS_CUDA
    if (is_gpu())
    {
        NeuralNetwork* self = const_cast<NeuralNetwork*>(this);

        const bool needs_parameter_device_copy =
            parameters.device_type != Device::CUDA
            || (!parameters.empty()
                && ((config.training_type == Type::BF16 && parameters_bf16_mirror.empty())
                    || (config.training_type == Type::INT8 && parameters_int8_storage.empty())));

        if (needs_parameter_device_copy)
            self->copy_parameters_device();

        self->copy_states_device();

        vector<TensorView>& device_inputs =
            forward_propagation.staged_inputs;
        device_inputs.assign(input_view.begin(), input_view.end());
        forward_propagation.staged_input_storage.resize(input_view.size());

        const bool uses_bf16_activations =
            activation_dtype(config.training_type) == Type::BF16;
        if (uses_bf16_activations)
            forward_propagation.host_bf16_input_scratch.resize(input_view.size());

        const auto input_feeds_token_ids = [&](size_t input_index)
        {
            const Index external_source = -static_cast<Index>(input_index) - 1;

            for (size_t layer_index = 0; layer_index < source_layers.size(); ++layer_index)
                for (const Index source : source_layers[layer_index])
                    if (source == external_source
                        && is_one_of(layers[layer_index]->get_type(),
                                     LayerType::Embedding, LayerType::Tokenizer))
                        return true;

            return false;
        };

        cudaStream_t stream = Backend::get_compute_stream();
        bool inputs_staged = false;

        if (has(LayerType::GroupedQueryAttention))
            forward_propagation.stage_position(stream);

        for (size_t i = 0; i < input_view.size(); ++i)
        {
            const TensorView& source = input_view[i];
            if (source.empty()) continue;
            if (source.is_cuda()) continue;

            throw_if(source.device == Device::Auto,
                     "NeuralNetwork::forward_propagate: input device must be CPU or CUDA.");

            const bool cast_input_to_bf16 = uses_bf16_activations
                                         && source.is_fp32()
                                         && !input_feeds_token_ids(i);

            Buffer& input_buffer = forward_propagation.staged_input_storage[i];
            const auto ensure_cuda_capacity = [&](Index required_bytes)
            {
                if (input_buffer.device_type != Device::CUDA)
                    input_buffer.resize_bytes(required_bytes, Device::CUDA);
                else
                    input_buffer.grow_to(required_bytes);
            };

            if (cast_input_to_bf16)
            {
                const Index n = source.size();
                vector<uint16_t>& bf16_cpu = forward_propagation.host_bf16_input_scratch[i];
                bf16_cpu.resize(size_t(n));
                float_2_bfloat16_host(n, source.as<float>(), bf16_cpu.data());
                ensure_cuda_capacity(n * Index(sizeof(uint16_t)));
                device::copy_async(input_buffer.data,
                                   bf16_cpu.data(),
                                   size_t(n) * sizeof(uint16_t),
                                   device::CopyKind::HostToDevice,
                                   stream);
                device_inputs[i].type = Type::BF16;
            }
            else
            {
                ensure_cuda_capacity(source.byte_size());
                device::copy_async(input_buffer.data,
                                   source.data,
                                   source.byte_size(),
                                   device::CopyKind::HostToDevice,
                                   stream);
            }

            device_inputs[i].data = input_buffer.data;
            device_inputs[i].device = Device::CUDA;
            inputs_staged = true;
        }

        forward_propagate(device_inputs, forward_propagation, is_training, first_layer_index, last_layer_index);

        if (inputs_staged)
            device::synchronize(stream);

        return;
    }
#endif

    forward_propagate(input_view, forward_propagation, is_training, first_layer_index, last_layer_index);
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      bool is_training,
                                      Index first_layer_index,
                                      Index last_layer_index) const
{
    throw_if(is_training
             && forward_propagation.mode != ForwardPropagationMode::Training,
             "NeuralNetwork::forward_propagate: an inference ForwardPropagation "
             "cannot be used for training.");

    const auto pick_input = [&](size_t input_index) -> const TensorView& {
        throw_if(input_index >= input_view.size(),
                 "NeuralNetwork::forward_propagate: input index {} out of range (have {} inputs). Network wiring expects more inputs than were provided.",
                        input_index, input_view.size());
        return input_view[input_index];
    };

    for (const auto& [layer_i, source_j, ext_idx] : forward_propagation.passthrough_overrides)
        if (Index(layer_i) >= first_layer_index)
            forward_propagation.inputs[layer_i][source_j] = pick_input(ext_idx);

    for (Index i = first_layer_index; i <= last_layer_index; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        auto& input_slot = forward_propagation.inputs[i];

        for (size_t source_index = 0; source_index < sources.size(); ++source_index)
        {
            const Index source_layer = sources[source_index];

            if (source_layer < 0)
                input_slot[source_index] = pick_input(size_t(-source_layer - 1));
            else if ((is_training || forward_propagation.inputs_pre_scaled)
                     && source_layer < first_layer_index)
                input_slot[source_index] = pick_input(source_index);
        }

        if (i == forward_propagation.get_final_output_layer())
            forward_propagation.gather_output_window();

        PROFILE_SCOPE("fwd:" + layers[i]->get_name());
        layers[i]->forward_propagate(forward_propagation, i, is_training);

        forward_propagation.inherit_valid_lengths(size_t(i));
    }
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      const VectorR& new_parameters,
                                      ForwardPropagation& forward_propagation)
{

    const Device original_parameters_device = parameters.device_type;
    const Index parameters_size = get_parameters_buffer_size();
    VectorR saved_parameters(parameters_size);
    if (parameters.device_type == Device::CUDA)
    {
        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(saved_parameters.data(), parameters.data,
                           parameters_size * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);
    }
    else
        memcpy(saved_parameters.data(), parameters.data,
               size_t(parameters_size) * sizeof(float));

    set_parameters(new_parameters);
    forward_propagate(input_view, forward_propagation, true);
    set_parameters(saved_parameters);

    if (parameters.device_type != original_parameters_device)
    {
        if (original_parameters_device == Device::CPU)
            copy_parameters_host();
        else if (original_parameters_device == Device::CUDA)
            copy_parameters_device();
    }
}

void NeuralNetwork::to_JSON(JsonWriter& printer) const
{

    const HostStatesGuard guard(*const_cast<NeuralNetwork*>(this),
                                parameters.device_type == Device::CUDA);

    const Index inputs_number = get_inputs_number();
    const Index layers_number = get_layers_number();
    const Index outputs_number = get_outputs_number();

    const auto write_variables_array = [&printer](const vector<Variable>& variables, const char* tag)
    {
        printer.begin_array(tag);

        for (size_t i = 0; i < variables.size(); ++i)
        {
            const Variable& variable = variables[i];

            printer.begin_array_object();
            add_json_field(printer, "Index", i + 1);
            add_json_field(printer, "Text", variable.name);
            add_json_field(printer, "Role", variable.get_role());
            add_json_field(printer, "Type", variable.get_type_string());
            add_json_field(printer, "Scaler", variable.get_scaler());

            if (variable.features > 1)
                add_json_field(printer, "Features", variable.features);

            if (is_one_of(variable.type, VariableType::Categorical, VariableType::Binary))
                add_json_field(printer, "Categories", vector_to_string(variable.categories, ";"));

            printer.end_array_object();
        }

        printer.end_array();
    };

    printer.open_element("NeuralNetwork");

    add_json_field(printer, "Task", network_task_map().to_string(task));
    add_json_field(printer, "TrainingActivationRecomputation",
                   training_activation_recomputation);

    printer.open_element("Inputs");
    add_json_field(printer, "InputsNumber", inputs_number);
    write_variables_array(input_variables, "Input");
    printer.close_element();

    printer.open_element("Layers");
    add_json_field(printer, "LayersNumber", layers_number);

    printer.begin_array("Items");
    for (Index i = 0; i < layers_number; ++i)
    {
        printer.begin_array_object();
        layers[i]->to_JSON(printer);
        printer.end_array_object();
    }
    printer.end_array();

    printer.open_element("SourceLayers");
    printer.begin_array("SourceLayer");
    for (size_t i = 0; i < source_layers.size(); ++i)
    {
        printer.begin_array_object();
        add_json_field(printer, "LayerIndex", i);
        add_json_field(printer, "Text", vector_to_string(source_layers[i]));
        printer.end_array_object();
    }
    printer.end_array();
    printer.close_element();

    printer.begin_array("TiedWeights");
    for (Index layer_index = 0; layer_index < layers_number; ++layer_index)
    {
        const Layer::TiedWeight tied_weight = layers[layer_index]->get_tied_weight();
        if (!tied_weight.source) continue;

        const Index source_layer_index = find_layer_index(layers, tied_weight.source);

        throw_if(source_layer_index < 0 || source_layer_index >= layer_index,
                 "NeuralNetwork::to_JSON: tied weight source for layer {} must be an earlier layer in the network.",
                 layer_index);

        printer.begin_array_object();
        add_json_field(printer, "LayerIndex", layer_index);
        add_json_field(printer, "SourceLayerIndex", source_layer_index);
        add_json_field(printer, "SpecIndex", tied_weight.spec_index);
        add_json_field(printer, "SourceSpecIndex", tied_weight.source_spec_index);
        printer.end_array_object();
    }
    printer.end_array();

    printer.close_element();

    printer.open_element("Outputs");
    const Index outputs_count = output_variables.empty()
                              ? outputs_number
                              : get_features_number(output_variables);
    add_json_field(printer, "OutputsNumber", outputs_count);
    write_variables_array(output_variables, "Output");
    printer.close_element();
    printer.close_element();
}

void NeuralNetwork::from_JSON(const JsonDocument& document)
{
    const Json* neural_network_element = get_json_root(document, "NeuralNetwork");

    if (neural_network_element->find("Task"))
        task = network_task_map().from_string(read_json_string(neural_network_element, "Task"));

    training_activation_recomputation =
        neural_network_element->has("TrainingActivationRecomputation")
        && read_json_bool(neural_network_element, "TrainingActivationRecomputation");

    const auto read_variables_array = [](const Json* parent, const char* tag,
                                         vector<Variable>& variables, const char* role)
    {
        const Json* items = parent->find(tag);
        const size_t entries_number = items && items->is_array()
                                    ? items->array_value.size()
                                    : 0;

        variables.assign(entries_number, Variable());

        for_json_items(parent, tag, entries_number, [&](size_t i, const Json* element) {
            Variable& variable = variables[i];

            variable.name = read_json_string(element, "Text");
            variable.set_role(element->has("Role")
                              ? read_json_string(element, "Role")
                              : role);
            variable.features = element->find("Features") ? read_json_index(element, "Features") : 1;

            if (element->has("Type"))
                variable.set_type(read_json_string(element, "Type"));
            else if (element->has("Categories"))
                variable.type = VariableType::Categorical;

            if (element->has("Scaler"))
                variable.set_scaler(read_json_string(element, "Scaler"));

            if (element->find("Categories"))
            {
                variable.categories = get_tokens(read_json_string(element, "Categories"), ";");
            }
        });
    };

    if (const Json* inputs_element = neural_network_element->find("Inputs"); inputs_element)
        read_variables_array(inputs_element, "Input", input_variables, "Input");

    const Json* layers_container = neural_network_element->find("Layers");
    throw_if(!layers_container, "layers container is nullptr.");

    const Index layers_number = read_json_index(layers_container, "LayersNumber");

    layers.clear();
    source_layers.clear();
    layers.reserve(layers_number);
    first_trainable_cache_ = -1;
    last_trainable_cache_  = -1;

    const Json* items_array = layers_container->find("Items");
    if (items_array && items_array->is_array())
    {
        for (const Json& item : items_array->array_value)
        {
            if (!item.is_object() || item.object_value.empty()) continue;

            const string& tag_name = item.object_value[0].first;

            unique_ptr<Layer> layer = create_layer(tag_name);

            JsonDocument layer_doc;
            layer_doc.root = item;
            layer->from_JSON(layer_doc);

            layers.push_back(std::move(layer));
        }
    }

    source_layers.resize(layers.size());

    if (const Json* source_layers_element = layers_container->find("SourceLayers"); source_layers_element)
    {
        const Json* indices_array = source_layers_element->find("SourceLayer");
        if (indices_array && indices_array->is_array())
        {
            for (const Json& entry : indices_array->array_value)
            {
                const long layer_index = read_json_index(&entry, "LayerIndex");
                const string text   = read_json_string(&entry, "Text");
                if (text.empty()) continue;

                throw_if(layer_index < 0 || layer_index >= ssize(layers),
                         "NeuralNetwork::from_JSON: SourceLayer index {} out of range (have {} layers).", layer_index, layers.size());

                const vector<Index> sources = parse_number_list<Index>(text, "SourceLayers");
                validate_source_indices(sources, layer_index, ssize(layers));
                validate_source_arity(*layers[layer_index], sources, layer_index);
                source_layers[layer_index] = sources;
            }
        }
    }

    if (const Json* tied_weights = layers_container->find("TiedWeights");
        tied_weights && tied_weights->is_array())
    {
        for (const Json& entry : tied_weights->array_value)
        {
            const Index layer_index = read_json_index(&entry, "LayerIndex");
            const Index source_layer_index = read_json_index(&entry, "SourceLayerIndex");
            const Index spec_index = entry.has("SpecIndex")
                                   ? read_json_index(&entry, "SpecIndex") : 0;
            const Index source_spec_index = entry.has("SourceSpecIndex")
                                          ? read_json_index(&entry, "SourceSpecIndex") : 0;

            throw_if(layer_index < 0 || layer_index >= ssize(layers)
                     || source_layer_index < 0 || source_layer_index >= layer_index
                     || spec_index < 0 || source_spec_index < 0,
                     "NeuralNetwork::from_JSON: invalid tied weight indices for layer {} and source {}.",
                     layer_index, source_layer_index);

            layers[size_t(layer_index)]->set_tied_weight({
                layers[size_t(source_layer_index)].get(),
                size_t(spec_index), size_t(source_spec_index)});
        }
    }

    if (const Json* outputs_element = neural_network_element->find("Outputs"); outputs_element)
        read_variables_array(outputs_element, "Output", output_variables, "Target");

    compile();

    if (items_array && items_array->is_array())
    {
        Index layer_index = 0;
        for (const Json& item : items_array->array_value)
        {
            if (!item.is_object() || item.object_value.empty()) continue;
            if (layer_index >= ssize(layers)) break;

            JsonDocument layer_doc;
            layer_doc.root = item;
            layers[layer_index]->load_state_from_JSON(layer_doc);
            ++layer_index;
        }
    }

    const Json* parameters_element = neural_network_element->find("Parameters");
    const string parameters_text   = parameters_element ? read_json_string(parameters_element, "Values") : string();
    if (parameters_text.empty()) return;

    VectorR json_parameters;
    string_to_vector(parameters_text, json_parameters);

    if (json_parameters.size() != parameters.size_in_floats())
    {
        cout << "Warning: JSON parameter size (" << json_parameters.size()
             << ") differs from Compiled size (" << parameters.size_in_floats() << ").\n";
    }

    const Index elements_to_copy = min(parameters.size_in_floats(), json_parameters.size());

    const HostParametersGuard guard(*this);
    // Qualified: opennn::copy is the tensor-view overload.
    std::copy(json_parameters.data(), json_parameters.data() + elements_to_copy, parameters.as<float>());
}

void NeuralNetwork::save(const filesystem::path& file_name) const
{
    validate_model_file_paths(file_name);
    recover_model_save_transaction(file_name);

    const filesystem::path binary_path = parameter_file_path(file_name);
    const filesystem::path temporary_model = append_path_suffix(file_name, ".tmp");
    const filesystem::path temporary_binary = append_path_suffix(binary_path, ".tmp");
    const filesystem::path model_backup = append_path_suffix(file_name, ".bak");
    const filesystem::path binary_backup = append_path_suffix(binary_path, ".bak");
    const filesystem::path marker_path =
        append_path_suffix(file_name, ".save-transaction");

    JsonWriter printer;
    to_JSON(printer);

    try
    {
        save_json_file(temporary_model, printer);
        save_parameters_binary(temporary_binary);

        const bool had_model = filesystem::exists(file_name);
        const bool had_parameters = filesystem::exists(binary_path);
        write_save_transaction_marker(marker_path, had_model, had_parameters);

        if (had_model) filesystem::rename(file_name, model_backup);
        if (had_parameters) filesystem::rename(binary_path, binary_backup);
        filesystem::rename(temporary_binary, binary_path);
        filesystem::rename(temporary_model, file_name);

        // Removing the marker is the commit point. Recovery before this
        // operation restores the backups; after it, the new pair is complete.
        filesystem::remove(marker_path);
        remove_transaction_artifacts({model_backup, binary_backup});
    }
    catch (const exception& save_error)
    {
        const string message = save_error.what();
        try
        {
            recover_model_save_transaction(file_name);
        }
        catch (const exception& recovery_error)
        {
            throw runtime_error(format(
                "Model save failed: {} Recovery also failed: {}",
                message, recovery_error.what()));
        }
        throw;
    }
}

static ofstream open_binary_output(const filesystem::path& file_name)
{
    ofstream file(file_name, ios::binary);

    throw_if(!file.is_open(),
             "Cannot open binary file for writing: {}\n", file_name.string());

    return file;
}

static void write_binary_payload(ofstream& file, const filesystem::path& file_name,
                                 const void* data, Index byte_count)
{
    if (byte_count > 0)
        file.write(static_cast<const char*>(data), byte_count);

    throw_if(!file, "Error writing binary file: {}\n", file_name.string());
}

static ifstream open_binary_input(const filesystem::path& file_name,
                                  uintmax_t expected_bytes, const char* caller)
{
    ifstream file(file_name, ios::binary);

    throw_if(!file.is_open(),
             "Cannot open binary file: {}\n", file_name.string());

    const uintmax_t file_bytes = filesystem::file_size(file_name);
    throw_if(file_bytes != expected_bytes,
             "NeuralNetwork::{}: size mismatch for {} (got {} bytes, expected {} bytes).",
                    caller,
                    file_name.string(),
                    file_bytes,
                    expected_bytes);

    return file;
}

static void save_binary_snapshot(const filesystem::path& file_name,
                                 const Buffer& storage,
                                 const SnapshotMagic& magic,
                                 uint64_t layout_fingerprint)
{
    const Index payload_bytes = storage.bytes;
    vector<char> staging;
    const void* payload = storage.data;

    if (storage.device_type == Device::CUDA && storage.data)
    {
        staging.resize(size_t(payload_bytes));
        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(staging.data(), storage.data, payload_bytes,
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);
        payload = staging.data();
    }

    const auto header = make_snapshot_header(
        magic, uint64_t(storage.size_in_floats()), uint64_t(payload_bytes),
        layout_fingerprint, hash_bytes(FNV1A_OFFSET, payload, size_t(payload_bytes)));

    ofstream file = open_binary_output(file_name);
    write_binary_payload(file, file_name, header.data(), Index(header.size()));
    write_binary_payload(file, file_name, payload, payload_bytes);
    file.close();
    throw_if(!file, "Error closing binary file: {}", file_name.string());
}

static void load_binary_snapshot(const filesystem::path& file_name,
                                 Buffer& storage,
                                 const SnapshotMagic& magic,
                                 uint64_t layout_fingerprint,
                                 const char* caller,
                                 const char* snapshot_name)
{
    const uint64_t payload_bytes = uint64_t(storage.bytes);
    ifstream file(file_name, ios::binary);
    throw_if(!file.is_open(), "Cannot open binary file: {}\n", file_name.string());

    const uintmax_t file_bytes = filesystem::file_size(file_name);
    const bool versioned = is_versioned_snapshot(
        file, file_bytes, payload_bytes, magic, file_name, caller);
    const uint64_t expected_checksum = versioned
        ? read_snapshot_header(file, file_bytes, magic,
                               uint64_t(storage.size_in_floats()), payload_bytes,
                               layout_fingerprint, file_name, caller, snapshot_name)
        : 0;

    vector<char> staging;
    void* destination = storage.data;
    if (versioned || (storage.device_type == Device::CUDA && storage.data))
    {
        staging.resize(size_t(payload_bytes));
        destination = staging.data();
    }

    if (payload_bytes > 0)
        file.read(static_cast<char*>(destination), streamsize(payload_bytes));
    throw_if(!file, "Error reading binary file: {}", file_name.string());

    if (versioned)
    {
        const uint64_t actual_checksum =
            hash_bytes(FNV1A_OFFSET, destination, size_t(payload_bytes));
        throw_if(actual_checksum != expected_checksum,
                 "NeuralNetwork::{}: payload checksum mismatch for {}.",
                 caller, file_name.string());
    }

    if (storage.device_type == Device::CUDA && storage.data && payload_bytes > 0)
    {
        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(storage.data, destination, storage.bytes,
                           device::CopyKind::HostToDevice, stream);
        device::synchronize(stream);
    }
    else if (versioned && payload_bytes > 0)
        memcpy(storage.data, destination, size_t(payload_bytes));
}

#ifdef OPENNN_HAS_CUDA
static inline Index quantization_channel(const Index element_index,
                                         const Index row_length,
                                         const Index channels,
                                         const int axis)
{
    return axis == 0 ? element_index / row_length : element_index % channels;
}

static inline void finalize_int8_scales(vector<float>& absolute_maxima)
{
    for (float& scale : absolute_maxima)
        scale = scale > 0.0f ? scale / 127.0f : 1.0f;
}

static inline void quantize_int8_host(const float* values, const Index count,
                                      const Index base_index, const Index row_length,
                                      const Index channels, const int axis,
                                      const float* scales, int8_t* out)
{
    #pragma omp parallel for if(count > 4096)
    for (Index i = 0; i < count; ++i)
    {
        const Index channel =
            quantization_channel(base_index + i, row_length, channels, axis);
        out[i] = int8_t(clamp<long>(lroundf(values[i] / scales[channel]), -127, 127));
    }
}
#endif

NeuralNetwork::ParameterSlotTotals NeuralNetwork::for_each_parameter_slot(
    const function<void(const ParameterSlot&)>& visit,
    const function<void(Layer&)>& begin_layer) const
{
    ParameterSlotTotals totals;
    Index master_elements = 0;

    for (const auto& layer : layers)
    {
        if (begin_layer) begin_layer(*layer);

        const auto specs = layer->get_parameter_specs();
        const auto quantization = layer->get_parameter_quantization();
        const Layer::TiedWeight tie = layer->get_tied_weight();

        for (size_t spec_index = 0; spec_index < specs.size(); ++spec_index)
        {
            const auto& [shape, dtype] = specs[spec_index];

            ParameterSlot slot;
            slot.layer = layer.get();
            slot.shape = shape;
            slot.dtype = dtype;
            slot.tied = tie.source && spec_index == tie.spec_index;
            slot.master_offset = master_elements;
            slot.bf16_offset = totals.bf16_elements;
            slot.int8_offset = totals.int8_elements;
            slot.fp32_offset = totals.fp32_elements;

            if (shape.empty())
            {
                if (visit) visit(slot);
                continue;
            }

            if (slot.dtype == Type::INT8 && !slot.tied)
            {
                const Operator::SlotQuantization slot_quantization =
                    spec_index < quantization.size() ? quantization[spec_index]
                                                     : Operator::SlotQuantization{};
                throw_if(slot_quantization.channels <= 0
                         || shape.size() % slot_quantization.channels != 0,
                         "NeuralNetwork: INT8 parameter slot without per-channel "
                         "quantization metadata in layer \"{}\".", layer->get_label());
                slot.scale_channels = slot_quantization.channels;
                slot.scale_axis = slot_quantization.axis;
            }

            if (visit) visit(slot);

            const Index aligned = get_aligned_size(shape.size());
            master_elements += aligned;
            if (slot.tied) continue;

            if (slot.dtype == Type::INT8)
            {
                totals.int8_elements += aligned;
                totals.fp32_elements += get_aligned_size(slot.scale_channels);
            }
            else if (slot.dtype == Type::BF16)
                totals.bf16_elements += aligned;
            else
                totals.fp32_elements += aligned;
        }
    }

    return totals;
}

void NeuralNetwork::allocate_compact_parameter_storage(const ParameterSlotTotals& totals)
{
    parameters_bf16_mirror.resize_bytes(
        totals.bf16_elements * Index(sizeof(bfloat16)), Device::CUDA);
    parameters_fp32_inference_storage.resize_bytes(
        totals.fp32_elements * Index(sizeof(float)), Device::CUDA);
    parameters_int8_storage.resize_bytes(totals.int8_elements, Device::CUDA);
    parameters_bf16_mirror_compact = true;
}

#ifdef OPENNN_HAS_CUDA

void NeuralNetwork::use_compact_parameter_storage()
{
    void* compact_storage = parameters_bf16_mirror.data;
    if (!compact_storage) compact_storage = parameters_int8_storage.data;
    if (!compact_storage) compact_storage = parameters_fp32_inference_storage.data;

    throw_if(!compact_storage,
             "NeuralNetwork: compact inference parameter storage is empty.");

    const Index master_bytes = parameters.bytes;
    parameters.resize_bytes(0, Device::CPU);
    parameters.set_view(compact_storage, master_bytes, Device::CUDA);
    link_parameters();
    activate_transposed_inference_weights();
}

#endif

void NeuralNetwork::save_parameters_binary(const filesystem::path& file_name) const
{
    throw_if(!parameters.owns,
             "NeuralNetwork::save_parameters_binary: the fp32 parameter master "
             "was released for quantized inference; reload the model before saving.");

    save_binary_snapshot(file_name, parameters, PARAMETER_FILE_MAGIC,
                         parameter_layout_fingerprint());
}

void NeuralNetwork::save_states_binary(const filesystem::path& file_name) const
{
    save_binary_snapshot(file_name, states, STATE_FILE_MAGIC,
                         state_layout_fingerprint());
}

void NeuralNetwork::load(const filesystem::path& file_name)
{
    validate_model_file_paths(file_name);
    recover_model_save_transaction(file_name);

    clear();

    from_JSON(load_json_file(file_name));

    const filesystem::path binary_path = parameter_file_path(file_name);

    if (filesystem::exists(binary_path))
        load_parameters_binary(binary_path);
}

void NeuralNetwork::load_parameters_binary(const filesystem::path& file_name)
{
    load_binary_snapshot(file_name, parameters, PARAMETER_FILE_MAGIC,
                         parameter_layout_fingerprint(),
                         "load_parameters_binary", "parameter");

    if (parameters.device_type == Device::CUDA && parameters.data)
        cast_parameters_to_bf16();

    link_parameters();
}

void NeuralNetwork::load_parameters_bf16_inference_binary(
    const filesystem::path& file_name)
{
    throw_if(parameters.empty() || !parameters.owns,
             "NeuralNetwork::load_parameters_bf16_inference_binary: "
             "the network must own its compiled parameter storage.");

    const Index parameters_number = parameters.size_in_floats();
    ifstream file = open_binary_input(
        file_name, uintmax_t(parameters_number) * sizeof(uint16_t),
        "load_parameters_bf16_inference_binary");

    constexpr Index chunk_elements = Index(8) * 1024 * 1024;
    vector<uint16_t> bf16_chunk(
        size_t(min(chunk_elements, max(Index(1), parameters_number))));
    vector<float> fp32_chunk(bf16_chunk.size());

#ifdef OPENNN_HAS_CUDA
    if (config.device == Device::CUDA)
    {
        throw_if(!is_one_of(config.training_type, Type::BF16, Type::INT8),
                 "NeuralNetwork::load_parameters_bf16_inference_binary: "
                 "CUDA direct loading requires BF16 or INT8 configuration.");

        const ParameterSlotTotals totals = for_each_parameter_slot({});
        allocate_compact_parameter_storage(totals);

        uint16_t* const mirror = parameters_bf16_mirror.as<uint16_t>();
        float* const fp32_compact = parameters_fp32_inference_storage.as<float>();
        int8_t* const int8_storage = parameters_int8_storage.as<int8_t>();
        cudaStream_t stream = Backend::get_compute_stream();

        const auto skip = [&](const Index count)
        {
            if (count <= 0) return;
            file.seekg(
                streamoff(count * Index(sizeof(uint16_t))), ios::cur);
            throw_if(!file,
                     "Error seeking through BF16 parameter file: {}",
                     file_name.string());
        };

        const auto read_bf16_to_device =
            [&](uint16_t* destination, const Index count)
        {
            Index copied = 0;
            while (copied < count)
            {
                const Index chunk =
                    min(chunk_elements, count - copied);
                file.read(
                    reinterpret_cast<char*>(bf16_chunk.data()),
                    streamsize(chunk * Index(sizeof(uint16_t))));
                throw_if(!file,
                         "Error reading BF16 parameter file: {}",
                         file_name.string());
                device::copy_async(
                    destination + copied, bf16_chunk.data(),
                    chunk * Index(sizeof(uint16_t)),
                    Device::CPU, Device::CUDA, stream);
                device::synchronize(stream);
                copied += chunk;
            }
        };

        const auto read_bf16_as_fp32_to_device =
            [&](float* destination, const Index count)
        {
            Index copied = 0;
            while (copied < count)
            {
                const Index chunk =
                    min(chunk_elements, count - copied);
                file.read(
                    reinterpret_cast<char*>(bf16_chunk.data()),
                    streamsize(chunk * Index(sizeof(uint16_t))));
                throw_if(!file,
                         "Error reading BF16 parameter file: {}",
                         file_name.string());
                ranges::transform(bf16_chunk | views::take(chunk), fp32_chunk.begin(),
                                  bfloat16_to_float_host);
                device::copy_async(
                    destination + copied, fp32_chunk.data(),
                    chunk * Index(sizeof(float)),
                    Device::CPU, Device::CUDA, stream);
                device::synchronize(stream);
                copied += chunk;
            }
        };

        vector<int8_t> int8_chunk(bf16_chunk.size());

        const auto read_bf16_quantize_int8_to_device =
            [&](int8_t* destination, float* scale_destination,
                const Index count, const Index channels, const int axis)
        {
            const Index row_length = count / channels;
            const streampos slot_start = file.tellg();

            vector<float> scales(size_t(channels), 0.0f);
            Index processed = 0;
            while (processed < count)
            {
                const Index chunk = min(chunk_elements, count - processed);
                file.read(reinterpret_cast<char*>(bf16_chunk.data()),
                          streamsize(chunk * Index(sizeof(uint16_t))));
                throw_if(!file, "Error reading BF16 parameter file: {}",
                         file_name.string());
                for (Index i = 0; i < chunk; ++i)
                {
                    const Index channel = quantization_channel(
                        processed + i, row_length, channels, axis);
                    scales[size_t(channel)] = max(scales[size_t(channel)],
                        abs(bfloat16_to_float_host(bf16_chunk[size_t(i)])));
                }
                processed += chunk;
            }
            finalize_int8_scales(scales);

            file.seekg(slot_start);
            throw_if(!file, "Error seeking through BF16 parameter file: {}",
                     file_name.string());

            processed = 0;
            while (processed < count)
            {
                const Index chunk = min(chunk_elements, count - processed);
                file.read(reinterpret_cast<char*>(bf16_chunk.data()),
                          streamsize(chunk * Index(sizeof(uint16_t))));
                throw_if(!file, "Error reading BF16 parameter file: {}",
                         file_name.string());
                ranges::transform(bf16_chunk | views::take(chunk), fp32_chunk.begin(),
                                  bfloat16_to_float_host);
                quantize_int8_host(fp32_chunk.data(), chunk, processed,
                                   row_length, channels, axis,
                                   scales.data(), int8_chunk.data());
                device::copy_async(
                    destination + processed, int8_chunk.data(),
                    chunk, Device::CPU, Device::CUDA, stream);
                device::synchronize(stream);
                processed += chunk;
            }

            device::copy_async(
                scale_destination, scales.data(),
                channels * Index(sizeof(float)),
                Device::CPU, Device::CUDA, stream);
            device::synchronize(stream);
        };

        for_each_parameter_slot([&](const ParameterSlot& slot)
        {
            if (slot.shape.empty()) return;

            const Index size = slot.shape.size();
            const Index aligned = get_aligned_size(size);
            if (slot.tied)
            {
                skip(aligned);
                return;
            }

            if (slot.dtype == Type::INT8)
                read_bf16_quantize_int8_to_device(
                    int8_storage + slot.int8_offset,
                    fp32_compact + slot.fp32_offset,
                    size, slot.scale_channels, slot.scale_axis);
            else if (slot.dtype == Type::BF16)
                read_bf16_to_device(mirror + slot.bf16_offset, size);
            else
                read_bf16_as_fp32_to_device(
                    fp32_compact + slot.fp32_offset, size);

            skip(aligned - size);
        });

        throw_if(file.peek() != ifstream::traits_type::eof(),
                 "NeuralNetwork::load_parameters_bf16_inference_binary: "
                 "unconsumed data remains in {}.",
                 file_name.string());

        use_compact_parameter_storage();
        return;
    }
#endif

    float* const host_parameters = parameters.as<float>();
    Index converted = 0;
    while (converted < parameters_number)
    {
        const Index chunk =
            min(chunk_elements, parameters_number - converted);
        file.read(reinterpret_cast<char*>(bf16_chunk.data()),
                  streamsize(chunk * Index(sizeof(uint16_t))));
        throw_if(!file,
                 "Error reading BF16 parameter file: {}",
                 file_name.string());
        ranges::transform(bf16_chunk | views::take(chunk), host_parameters + converted,
                          bfloat16_to_float_host);
        converted += chunk;
    }
    link_parameters();
}

void NeuralNetwork::load_states_binary(const filesystem::path& file_name)
{
    load_binary_snapshot(file_name, states, STATE_FILE_MAGIC,
                         state_layout_fingerprint(),
                         "load_states_binary", "state");

    link_states();
}

vector<string> NeuralNetwork::get_layer_labels() const
{
    vector<string> layer_labels(layers.size());
    ranges::transform(layers, layer_labels.begin(),
                      [](const unique_ptr<Layer>& layer) { return layer->get_label(); });
    return layer_labels;
}

void NeuralNetwork::link_parameters()
{
    float* fp32_base = parameters.as<float>();
    float* fp32_inference_base =
        parameters.device_type == Device::CUDA
        && !parameters.owns
        && !parameters_fp32_inference_storage.empty()
        ? parameters_fp32_inference_storage.as<float>()
        : nullptr;

    bfloat16* bf16_mirror_base = (parameters.device_type == Device::CUDA && !parameters_bf16_mirror.empty())
        ? parameters_bf16_mirror.as<bfloat16>()
        : nullptr;

    int8_t* int8_base = (parameters.device_type == Device::CUDA && !parameters_int8_storage.empty())
        ? parameters_int8_storage.as<int8_t>()
        : nullptr;

    Layer* current_layer = nullptr;

    for_each_parameter_slot([&](const ParameterSlot& slot)
    {
        auto& param_views = slot.layer->get_parameter_views();
        auto& param_scales = slot.layer->get_parameter_scales();

        if (slot.shape.empty())
        {
            param_views.emplace_back();
            param_scales.emplace_back();
            return;
        }

        const Type expected_type =
            slot.dtype == Type::INT8 && int8_base != nullptr ? Type::INT8
            : slot.dtype == Type::BF16 && bf16_mirror_base != nullptr ? Type::BF16
            : Type::FP32;

        if (slot.tied)
        {
            const Layer::TiedWeight tie = slot.layer->get_tied_weight();
            const auto& source_views = tie.source->get_parameter_views();
            throw_if(source_views.size() <= tie.source_spec_index
                     || source_views[tie.source_spec_index].empty(),
                     "NeuralNetwork::link_parameters: tied weight source is not linked.");
            const TensorView& source = source_views[tie.source_spec_index];
            throw_if(source.size() != slot.shape.size(),
                     "NeuralNetwork::link_parameters: tied weight sizes do not match.");
            throw_if(source.type != expected_type,
                     "NeuralNetwork::link_parameters: tied weight dtype mismatch "
                     "(the source table must be stored in the consumer's compute dtype).");

            param_views.emplace_back(source);

            const auto& source_scales = tie.source->get_parameter_scales();
            param_scales.emplace_back(source_scales.size() > tie.source_spec_index
                                      ? source_scales[tie.source_spec_index]
                                      : TensorView{});
            return;
        }

        float* const fp32_slot = fp32_base ? fp32_base + slot.master_offset : nullptr;

        void* slot_ptr = fp32_slot;
        Type view_type = Type::FP32;
        Device view_device = parameters.device_type;
        TensorView scale_view;

        if (slot.dtype == Type::INT8 && int8_base != nullptr)
        {
            throw_if(fp32_inference_base == nullptr,
                     "NeuralNetwork::link_parameters: INT8 parameters require compact FP32 scale storage.");

            slot_ptr = int8_base + slot.int8_offset;
            view_type = Type::INT8;
            view_device = Device::CUDA;
            scale_view = TensorView(fp32_inference_base + slot.fp32_offset,
                                    Shape{slot.scale_channels}, Type::FP32, Device::CUDA);
        }
        else if (slot.dtype == Type::BF16 && bf16_mirror_base != nullptr)
        {
            slot_ptr = bf16_mirror_base
                + (parameters_bf16_mirror_compact ? slot.bf16_offset : slot.master_offset);
            view_type = Type::BF16;
            view_device = Device::CUDA;
        }
        else if (fp32_inference_base != nullptr)
        {
            float* const compact_slot = fp32_inference_base + slot.fp32_offset;
            throw_if(!is_aligned(compact_slot),
                     "NeuralNetwork::link_parameters: unaligned compact fp32 parameter memory.");

            slot_ptr = compact_slot;
            view_type = Type::FP32;
            view_device = Device::CUDA;
        }
        else
        {
            throw_if(!is_aligned(fp32_slot),
                     "NeuralNetwork::link_parameters: unaligned parameter memory.");
        }

        param_views.emplace_back(slot_ptr, slot.shape, view_type, view_device);
        param_scales.emplace_back(scale_view);
    },
    [&](Layer& layer)
    {
        if (current_layer) current_layer->redistribute_parameters_to_operators();
        layer.get_parameter_views().clear();
        layer.get_parameter_scales().clear();
        current_layer = &layer;
    });

    if (current_layer) current_layer->redistribute_parameters_to_operators();
}

void NeuralNetwork::link_states()
{
    const Device state_device = states.empty()
        ? parameters.device_type
        : states.device_type;

    link_states(state_device);
}

void NeuralNetwork::link_states(Device device)
{
    float* state_pointer = states.as<float>();

    for (auto& layer : layers)
        state_pointer = layer->link_states(state_pointer, device);
}

#ifdef OPENNN_HAS_CUDA

void NeuralNetwork::copy_parameters_device()
{
    if (parameters.empty())
    {
        clear_low_precision_parameter_storage();
        return;
    }

    if (parameters.device_type == Device::CUDA && !parameters.owns)
    {
        const bool bf16_released = config.training_type == Type::BF16 && !parameters_bf16_mirror.empty();
        const bool int8_released = config.training_type == Type::INT8 && !parameters_int8_storage.empty();
        throw_if(!bf16_released && !int8_released,
                 "NeuralNetwork::copy_parameters_device: parameters are a non-owning view.");
        link_parameters();
        return;
    }

    if (config.training_type == Type::INT8)
    {
        throw_if(parameters.device_type != Device::CPU || !parameters.owns,
                 "NeuralNetwork::copy_parameters_device: INT8 inference requires "
                 "a host FP32 master to quantize.");
        upload_parameters_int8_inference();
        return;
    }

    cudaStream_t stream = Backend::get_compute_stream();
    parameters.migrate_to(Device::CUDA, stream);

    if (config.training_type == Type::BF16)
    {
        parameters_bf16_mirror.resize_bytes(parameters.size_in_floats() * Index(sizeof(bfloat16)), Device::CUDA);
        parameters_bf16_mirror_compact = false;
        parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
        parameters_int8_storage.resize_bytes(0, Device::CUDA);
        cast_parameters_to_bf16();
    }
    else
        clear_low_precision_parameter_storage();

    link_parameters();
}

void NeuralNetwork::cast_parameters_to_bf16()
{
    if (parameters_bf16_mirror.empty() || parameters.empty() || !parameters.owns) return;

    cast_fp32_to_bf16(parameters.size_in_floats(),
                           parameters.as<float>(),
                           parameters_bf16_mirror.as<bfloat16>());
}

void NeuralNetwork::release_bf16_fp32_parameter_master_for_inference()
{
    const bool can_release_parameter_master =
        config.training_type == Type::BF16
        && parameters.device_type == Device::CUDA
        && !parameters.empty()
        && !parameters_bf16_mirror.empty()
        && parameters.owns;

    if (!can_release_parameter_master) return;

    const auto specs = get_parameter_specs();

    Index fp32_keep_floats = 0;
    for (const auto& layer_specs : specs)
        for (const auto& [shape, dtype] : layer_specs)
            if (!shape.empty() && dtype != Type::BF16)
                fp32_keep_floats += get_aligned_size(shape.size());

    if (fp32_keep_floats > 0)
    {
        parameters_fp32_inference_storage.resize_bytes(fp32_keep_floats * Index(sizeof(float)), Device::CUDA);

        cudaStream_t stream = Backend::get_compute_stream();
        float* const source_base = parameters.as<float>();
        float* const destination_base = parameters_fp32_inference_storage.as<float>();

        Index source_offset = 0;
        Index destination_offset = 0;

        for (const auto& layer_specs : specs)
            for (const auto& [shape, dtype] : layer_specs)
            {
                if (shape.empty()) continue;

                const Index aligned = get_aligned_size(shape.size());
                if (dtype != Type::BF16)
                {
                    device::copy_async(destination_base + destination_offset,
                                       source_base + source_offset,
                                       aligned * Index(sizeof(float)),
                                       device::CopyKind::DeviceToDevice,
                                       stream);
                    destination_offset += aligned;
                }
                source_offset += aligned;
            }

        device::synchronize(stream);
        memory_debug::record("parameters",
                             "fp32_compact_inference",
                             parameters_fp32_inference_storage.bytes,
                             "bf16_release");
    }
    else
    {
        parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
    }

    const Index fp32_master_bytes = parameters.bytes;
    parameters.resize_bytes(0, Device::CUDA);
    parameters.set_view(parameters_bf16_mirror.data,
                        fp32_master_bytes,
                        Device::CUDA);
    link_parameters();
}

void NeuralNetwork::upload_parameters_bf16_inference()
{
    const bool can_upload_low_precision_parameters =
        config.device == Device::CUDA
        && is_one_of(config.training_type, Type::BF16, Type::INT8)
        && !parameters.empty()
        && parameters.device_type == Device::CPU
        && parameters.owns;

    if (!can_upload_low_precision_parameters)
    {
        copy_parameters_device();
        return;
    }

    cudaStream_t stream = Backend::get_compute_stream();
    const float* const host_fp32 = parameters.as<float>();

    const ParameterSlotTotals totals = for_each_parameter_slot({});
    allocate_compact_parameter_storage(totals);
    uint16_t* const mirror = parameters_bf16_mirror.as<uint16_t>();
    float* const fp32_compact = parameters_fp32_inference_storage.as<float>();
    int8_t* const int8_storage = parameters_int8_storage.as<int8_t>();

    vector<uint16_t> host_bf16;
    vector<int8_t> host_int8;
    vector<float> host_scales;

    for_each_parameter_slot([&](const ParameterSlot& slot)
    {
        if (slot.shape.empty() || slot.tied) return;

        const Index size = slot.shape.size();
        const float* const source = host_fp32 + slot.master_offset;

        if (slot.dtype == Type::INT8 && int8_storage)
        {
            const Index channels = slot.scale_channels;
            const Index row_length = size / channels;

            host_scales.assign(size_t(channels), 0.0f);
            for (Index i = 0; i < size; ++i)
            {
                const Index channel = quantization_channel(i, row_length, channels, slot.scale_axis);
                host_scales[size_t(channel)] = max(host_scales[size_t(channel)], abs(source[i]));
            }
            finalize_int8_scales(host_scales);

            host_int8.resize(size_t(size));
            quantize_int8_host(source, size, 0, row_length, channels, slot.scale_axis,
                               host_scales.data(), host_int8.data());

            device::copy_async(int8_storage + slot.int8_offset, host_int8.data(),
                               size, Device::CPU, Device::CUDA, stream);
            device::copy_async(fp32_compact + slot.fp32_offset, host_scales.data(),
                               channels * Index(sizeof(float)), Device::CPU, Device::CUDA, stream);
            device::synchronize(stream);
        }
        else if (slot.dtype == Type::BF16 && mirror)
        {
            host_bf16.resize(static_cast<size_t>(size));
            ranges::transform(span<const float>(source, static_cast<size_t>(size)),
                              host_bf16.begin(), float_to_bfloat16_host);
            device::copy_async(mirror + slot.bf16_offset, host_bf16.data(),
                               size * Index(sizeof(uint16_t)), Device::CPU, Device::CUDA, stream);
            device::synchronize(stream);
        }
        else if (fp32_compact)
            device::copy_async(fp32_compact + slot.fp32_offset, source,
                               size * Index(sizeof(float)), Device::CPU, Device::CUDA, stream);
    });
    device::synchronize(stream);

    use_compact_parameter_storage();
}

void NeuralNetwork::upload_parameters_int8_inference()
{
    throw_if(config.training_type != Type::INT8,
             "NeuralNetwork::upload_parameters_int8_inference: "
             "the network must be compiled with an INT8 configuration.");
    upload_parameters_bf16_inference();
}

void NeuralNetwork::activate_transposed_inference_weights()
{
    cudaStream_t stream = Backend::get_compute_stream();

    const auto transpose_in_place = [&](const TensorView& weight)
    {
        Buffer scratch{Device::CUDA};
        scratch.resize_bytes(weight.byte_size(), Device::CUDA);
        if (weight.is_int8())
            transpose_2d_cuda<int8_t>(weight.shape[0], weight.shape[1],
                                      weight.as<int8_t>(), scratch.as<int8_t>());
        else
            weight.dispatch([&]<typename T>()
            {
                transpose_2d_cuda<T>(weight.shape[0], weight.shape[1],
                                     weight.as<T>(), scratch.as<T>());
            });
        device::copy_async(weight.data, scratch.data, weight.byte_size(),
                           device::CopyKind::DeviceToDevice, stream);
        device::synchronize(stream);
    };

    const bool int8_training = get_training_type() == Type::INT8;

    for (const auto& layer : layers)
    {
        const bool has_tied_weight = bool(layer->get_tied_weight().source);
        vector<CombinationOperator*> combinations;

        for (Operator* op : layer->get_operators())
            if (auto* combination = dynamic_cast<CombinationOperator*>(op))
                combinations.push_back(combination);

        for (CombinationOperator* combination : combinations)
        {
            const TensorView& weight = combination->weights;
            const bool automatic_int8 = int8_training && weight.is_int8()
                && combination->fused_activation == ActivationFunction::Identity;
            const bool configured = combinations.size() == 1
                && combination->transposed_inference_preferred && !weight.is_int8();

            if (has_tied_weight || combination->transposed_inference_active
                || combination->tied_transposed || combination->use_bias
                || (!automatic_int8 && !configured)
                || !weight.is_cuda() || weight.get_rank() != 2)
                continue;

            transpose_in_place(weight);
            combination->transposed_inference_active = true;
        }
    }
}

void NeuralNetwork::copy_parameters_host()
{
    if (parameters.empty())
    {
        clear_low_precision_parameter_storage();
        return;
    }

    throw_if(parameters.device_type == Device::CUDA && !parameters.owns,
             "NeuralNetwork::copy_parameters_host: the fp32 CUDA parameter master "
             "was released for quantized inference and cannot be copied back.");

    parameters.migrate_to(Device::CPU, Backend::get_compute_stream());
    clear_low_precision_parameter_storage();

    for (const auto& layer : layers)
        for (Operator* op : layer->get_operators())
            if (auto* combination = dynamic_cast<CombinationOperator*>(op))
                combination->transposed_inference_active = false;

    link_parameters();
}

void NeuralNetwork::copy_states_device()
{
    if (!states.empty())
        states.migrate_to(Device::CUDA, Backend::get_compute_stream());

    link_states(Device::CUDA);
}

void NeuralNetwork::copy_states_host()
{
    if (!states.empty())
        states.migrate_to(Device::CPU, Backend::get_compute_stream());

    link_states(Device::CPU);
}

MatrixR NeuralNetwork::calculate_outputs_device(const vector<TensorView>& input_views_cpu,
                                                ForwardPropagation& forward_propagation)
{
    forward_propagate(input_views_cpu, forward_propagation, false);

    const TensorView out_view = forward_propagation.get_outputs();

    const Index batch_size = input_views_cpu[0].shape[0];
    const Index out_cols = out_view.size() / batch_size;
    MatrixR result(batch_size, out_cols);

    cudaStream_t stream = Backend::get_compute_stream();
    copy_device_to_host_float(out_view.data, out_view.type, out_view.size(),
                              result.data(), stream);

    return result;
}

namespace
{

bool same_input_pointers(const vector<TensorView>& inputs,
                         const vector<const void*>& captured)
{
    if (captured.size() != inputs.size()) return false;

    for (size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i].data != captured[i]) return false;

    return true;
}

constexpr Index inference_graph_warmup_calls = 2;

}

TensorView NeuralNetwork::calculate_outputs_resident(const vector<TensorView>& gpu_inputs,
                                                     ForwardPropagation& forward_propagation,
                                                     bool upload_parameters)
{

    if (upload_parameters)
    {
        copy_parameters_device();
        copy_states_device();

        forward_propagation.reset_cuda_graph();
    }

    if (!forward_propagation.use_cuda_graph || forward_propagation.cuda_graph_failed)
    {
        forward_propagate(gpu_inputs, forward_propagation, false);
        return forward_propagation.get_outputs();
    }

    const cudaStream_t compute = Backend::get_compute_stream();

    if (forward_propagation.inference_graph_exec)
    {
        if (same_input_pointers(gpu_inputs, forward_propagation.captured_input_pointers))
        {
            PROFILE_SCOPE_HOST("inference:graph_launch");

            if (forward_propagation.position_pinned)
                *static_cast<int*>(forward_propagation.position_pinned) = int(forward_propagation.past_length);
            device::launch_graph(forward_propagation.inference_graph_exec, compute);
            return forward_propagation.get_outputs();
        }

        forward_propagate(gpu_inputs, forward_propagation, false);
        return forward_propagation.get_outputs();
    }

    {
        device::CudaGraphWorkspaceScope workspace_measurement(
            forward_propagation.inference_graph_workspace_requirements);
        forward_propagate(gpu_inputs, forward_propagation, false);
    }

    if (++forward_propagation.cuda_graph_warmup_calls < inference_graph_warmup_calls)
        return forward_propagation.get_outputs();

    if (env_flag_enabled("OPENNN_GRAPH_TIMING"))
    {
        forward_propagation.cuda_graph_failed = true;
        cerr << "NeuralNetwork::calculate_outputs_resident: OPENNN_GRAPH_TIMING "
                "event timing cannot be captured; continuing eager.\n";
        return forward_propagation.get_outputs();
    }

    const bool profiler_was_enabled = ::opennn::enabled();
    ::opennn::enabled() = false;

    forward_propagation.prepare_cuda_graph_workspaces();
    const device::GraphWorkspaceViews graph_workspace_views =
        forward_propagation.get_cuda_graph_workspace_views();

    try
    {
        device::synchronize(compute);
        device::CudaAllocationGrowthGuard growth_guard(true);
        device::CudaGraphWorkspaceScope stable_workspaces(
            forward_propagation.inference_graph_workspace_requirements,
            &graph_workspace_views);
        device::StreamCapture capture(compute);

        forward_propagate(gpu_inputs, forward_propagation, false);

        capture.end(forward_propagation.inference_graph_exec);

        forward_propagation.captured_input_pointers.resize(gpu_inputs.size());
        ranges::transform(gpu_inputs, forward_propagation.captured_input_pointers.begin(),
                          [](const auto& gpu_input) { return gpu_input.data; });
    }
    catch (const exception& capture_error)
    {
        if (forward_propagation.cuda_graph_workspaces_need_growth())
        {

            forward_propagation.inference_graph_exec.reset();
            forward_propagation.captured_input_pointers.clear();
            forward_propagation.cuda_graph_warmup_calls =
                inference_graph_warmup_calls - 1;
        }
        else
        {
            forward_propagation.reset_cuda_graph();
            forward_propagation.cuda_graph_failed = true;
            cerr << "NeuralNetwork::calculate_outputs_resident: cuda graph capture "
                    "unavailable (" << capture_error.what() << "); continuing eager.\n";
        }
    }

    if (forward_propagation.inference_graph_exec)
        release_matmul_thread_workspaces();

    ::opennn::enabled() = profiler_was_enabled;

    return forward_propagation.get_outputs();
}

#else

void NeuralNetwork::copy_parameters_device() OPENNN_CUDA_STUB_BODY(NeuralNetwork::copy_parameters_device)

void NeuralNetwork::cast_parameters_to_bf16() OPENNN_CUDA_STUB_BODY(NeuralNetwork::cast_parameters_to_bf16)

void NeuralNetwork::release_bf16_fp32_parameter_master_for_inference()
{
}

void NeuralNetwork::upload_parameters_bf16_inference()
{
}

void NeuralNetwork::upload_parameters_int8_inference() OPENNN_CUDA_STUB_BODY(NeuralNetwork::upload_parameters_int8_inference)

void NeuralNetwork::copy_parameters_host()
{
    link_parameters();
}

void NeuralNetwork::copy_states_device() OPENNN_CUDA_STUB_BODY(NeuralNetwork::copy_states_device)

void NeuralNetwork::copy_states_host()
{
    link_states(Device::CPU);
}

MatrixR NeuralNetwork::calculate_outputs_device(const vector<TensorView>&,
                                                ForwardPropagation&) OPENNN_CUDA_STUB_BODY(NeuralNetwork::calculate_outputs_device)

TensorView NeuralNetwork::calculate_outputs_resident(const vector<TensorView>&,
                                                     ForwardPropagation&,
                                                     bool) OPENNN_CUDA_STUB_BODY(NeuralNetwork::calculate_outputs_resident)

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
