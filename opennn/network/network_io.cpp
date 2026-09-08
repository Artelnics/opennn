//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E T W O R K   P E R S I S T E N C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/network/network.h"
#include "opennn/network/network_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <utility>

#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/device_backend.h"
#include "opennn/core/enum_map.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/registry.h"

namespace opennn
{

using network_detail::validate_source_indices;
using network_detail::validate_source_arity;

#ifdef OPENNN_HAS_CUDA
using network_detail::quantization_channel;
using network_detail::finalize_int8_scales;
using network_detail::quantize_int8_host;
#endif

namespace
{

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
    throw_if(model_path.empty(), "Network: the model file path is empty.");

    const filesystem::path parameter_path = parameter_file_path(model_path);
    throw_if(model_path == parameter_path
             || ascii_lowercase(model_path.extension().string()) == ".bin",
             "Network: the model file must not use the .bin extension because "
             "that path is reserved for its parameter snapshot.");

    throw_if(filesystem::exists(model_path) && filesystem::is_directory(model_path),
             "Network: the model path is a directory: {}.", model_path.string());
    throw_if(filesystem::exists(parameter_path) && filesystem::is_directory(parameter_path),
             "Network: the parameter path is a directory: {}.",
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
        return remove_transaction_artifact(backup_path);
    }

    if (filesystem::exists(backup_path))
    {
        remove_transaction_artifact(final_path);
        return filesystem::rename(backup_path, final_path);
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
    hash = hash_uint64(hash, shape.get_rank());
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
    return uint32_t{source[0]}
         | uint32_t{source[1]} << 8
         | uint32_t{source[2]} << 16
         | uint32_t{source[3]} << 24;
}

uint64_t load_uint64_le(const unsigned char* source)
{
    return uint64_t{source[0]}
         | uint64_t{source[1]} << 8
         | uint64_t{source[2]} << 16
         | uint64_t{source[3]} << 24
         | uint64_t{source[4]} << 32
         | uint64_t{source[5]} << 40
         | uint64_t{source[6]} << 48
         | uint64_t{source[7]} << 56;
}

class HeaderWriter
{
public:

    explicit HeaderWriter(span<unsigned char> destination) : buffer(destination) {}

    void bytes(span<const unsigned char> value)
    {
        ranges::copy(value, buffer.begin() + used);
        used += Index(value.size());
    }

    void u32(uint32_t value) { store_uint32_le(buffer.data() + used, value); used += 4; }
    void u64(uint64_t value) { store_uint64_le(buffer.data() + used, value); used += 8; }

    Index size() const noexcept { return used; }

private:

    span<unsigned char> buffer;
    Index used = 0;
};

class HeaderReader
{
public:

    explicit HeaderReader(span<const unsigned char> source) : buffer(source) {}

    void skip(Index count) noexcept { used += count; }

    uint32_t u32() { const uint32_t v = load_uint32_le(buffer.data() + used); used += 4; return v; }
    uint64_t u64() { const uint64_t v = load_uint64_le(buffer.data() + used); used += 8; return v; }

private:

    span<const unsigned char> buffer;
    Index used = 0;
};

array<unsigned char, SNAPSHOT_FILE_HEADER_SIZE> make_snapshot_header(
    const SnapshotMagic& magic, uint64_t elements, uint64_t payload_bytes,
    uint64_t layout, uint64_t checksum)
{
    array<unsigned char, SNAPSHOT_FILE_HEADER_SIZE> header{};

    HeaderWriter write(header);
    write.bytes(magic);
    write.u32(SNAPSHOT_FILE_VERSION);
    write.u32(SNAPSHOT_FILE_HEADER_SIZE);
    write.u32(SNAPSHOT_FILE_ENDIAN_MARKER);
    write.u32(SNAPSHOT_FILE_SCALAR_FP32);
    write.u64(elements);
    write.u64(payload_bytes);
    write.u64(layout);
    write.u64(checksum);

    throw_if(write.size() != SNAPSHOT_FILE_HEADER_SIZE,
             "make_snapshot_header: wrote {} header bytes, expected {}.",
             write.size(), Index(SNAPSHOT_FILE_HEADER_SIZE));

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
        throw_if(!file, "Network::{}: cannot inspect {}.",
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
             "Network::{}: size mismatch for {} "
             "(got {} bytes, expected {} legacy bytes or at least {} versioned bytes).",
             caller, file_name.string(), file_bytes, expected_payload_bytes,
             SNAPSHOT_FILE_HEADER_SIZE);

    array<unsigned char, SNAPSHOT_FILE_HEADER_SIZE> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    throw_if(!file, "Network::{}: cannot read header from {}.",
             caller, file_name.string());

    throw_if(!ranges::equal(magic, span(header).first(magic.size())),
             "Network::{}: {} has an unrecognized header and is not a "
             "legacy raw snapshot of the expected size.",
             caller, file_name.string());

    HeaderReader read(header);
    read.skip(Index(magic.size()));

    const uint32_t version = read.u32();
    const uint32_t header_size = read.u32();
    const uint32_t endian_marker = read.u32();
    const uint32_t scalar_type = read.u32();
    const uint64_t stored_elements = read.u64();
    const uint64_t stored_payload_bytes = read.u64();
    const uint64_t stored_layout = read.u64();
    const uint64_t stored_checksum = read.u64();

    throw_if(version != SNAPSHOT_FILE_VERSION,
             "Network::{}: unsupported {} file version {} in {} "
             "(supported version {}).",
             caller, snapshot_name, version, file_name.string(),
             SNAPSHOT_FILE_VERSION);
    throw_if(header_size != SNAPSHOT_FILE_HEADER_SIZE,
             "Network::{}: invalid version-{} header size {} in {}.",
             caller, version, header_size, file_name.string());
    throw_if(endian_marker != SNAPSHOT_FILE_ENDIAN_MARKER,
             "Network::{}: unsupported byte order in {}.",
             caller, file_name.string());
    throw_if(scalar_type != SNAPSHOT_FILE_SCALAR_FP32,
             "Network::{}: unsupported scalar type {} in {}.",
             caller, scalar_type, file_name.string());
    throw_if(stored_elements != expected_elements
             || stored_payload_bytes != expected_payload_bytes,
             "Network::{}: payload size mismatch for {} "
             "(file has {} FP32 elements/{} bytes, network expects {}/{}).",
             caller, file_name.string(), stored_elements, stored_payload_bytes,
             expected_elements, expected_payload_bytes);
    throw_if(file_bytes != uintmax_t(header_size) + stored_payload_bytes,
             "Network::{}: file size mismatch for {} "
             "(got {} bytes, header describes {}).",
             caller, file_name.string(), file_bytes,
             uintmax_t(header_size) + stored_payload_bytes);
    throw_if(stored_layout != expected_layout,
             "Network::{}: {} layout mismatch for {} "
             "(file {:016x}, network {:016x}).",
             caller, snapshot_name, file_name.string(), stored_layout,
             expected_layout);

    return stored_checksum;
}

const EnumMap<NetworkTask>& network_task_map()
{
    static const EnumMap<NetworkTask> map{
        {NetworkTask::Generic,             "Generic"},
        {NetworkTask::Approximation,       "Approximation"},
        {NetworkTask::Classification,      "Classification"},
        {NetworkTask::Forecasting,         "Forecasting"},
        {NetworkTask::AnomalyDetection,     "AnomalyDetection"},
        {NetworkTask::AnomalyDetection,     "AutoAssociation"},
        {NetworkTask::ImageClassification, "ImageClassification"},
        {NetworkTask::ObjectDetection,     "ObjectDetection"},
        {NetworkTask::TextClassification,  "TextClassification"},
        {NetworkTask::LanguageModeling,    "LanguageModeling"}
    };
    return map;
}

}

uint64_t Network::parameter_layout_fingerprint() const
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
                 "Network::parameter_layout_fingerprint: tied weight source is not in the network.");

        hash = hash_uint64(hash, static_cast<uint64_t>(tied_source_index));
        hash = hash_uint64(hash, tied_weight.spec_index);
        hash = hash_uint64(hash, tied_weight.source_spec_index);
    }

    return hash;
}

uint64_t Network::state_layout_fingerprint() const
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

void Network::to_JSON(JsonWriter& printer) const
{

    const HostStatesGuard guard(*const_cast<Network*>(this));

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

    printer.open_element("Network");

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
                 "Network::to_JSON: tied weight source for layer {} must be an earlier layer in the network.",
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

void Network::from_JSON(const JsonDocument& document)
{
    const Json* network_element = get_json_root(document, "Network");

    if (network_element->find("Task"))
        task = network_task_map().from_string(read_json_string(network_element, "Task"));

    training_activation_recomputation =
        network_element->has("TrainingActivationRecomputation")
        && read_json_bool(network_element, "TrainingActivationRecomputation");

    const auto read_variables_array = [](const Json* parent, const char* tag,
                                         vector<Variable>& variables, const char* role)
    {
        const Json* items = parent->find(tag);
        const size_t entries_number = items && items->is_array()
                                    ? items->as_array().size()
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

    if (const Json* inputs_element = network_element->find("Inputs"); inputs_element)
        read_variables_array(inputs_element, "Input", input_variables, "Input");

    const Json* layers_container = network_element->find("Layers");
    throw_if(!layers_container, "layers container is nullptr.");

    const Index layers_number = read_json_index(layers_container, "LayersNumber");

    layers.clear();
    source_layers.clear();
    layers.reserve(layers_number);
    linked_gradient_base   = nullptr;

    const Json* items_array = layers_container->find("Items");
    if (items_array && items_array->is_array())
    {
        for (const Json& item : items_array->as_array())
        {
            if (!item.is_object() || item.as_object().empty()) continue;

            const string& tag_name = item.as_object().front().first;

            unique_ptr<Layer> layer = create_layer(tag_name);

            JsonDocument layer_doc;
            layer_doc.set_root(item);
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
            for (const Json& entry : indices_array->as_array())
            {
                const long layer_index = read_json_index(&entry, "LayerIndex");
                const string text   = read_json_string(&entry, "Text");
                if (text.empty()) continue;

                throw_if(layer_index < 0 || layer_index >= ssize(layers),
                         "Network::from_JSON: SourceLayer index {} out of range (have {} layers).", layer_index, layers.size());

                const vector<Index> sources = parse_number_list<Index>(text, "SourceLayers");
                validate_source_indices(sources, layer_index, ssize(layers));
                validate_source_arity(*layers[layer_index], sources, layer_index);
                source_layers[layer_index] = sources;
            }
        }
    }

    for (Index i = 0; i < ssize(layers); ++i)
    {
        if (!source_layers[size_t(i)].empty()) continue;

        throw_if(i == 0,
                 "Network::from_JSON: layer 0 has no source; the first layer must name "
                 "its input.");

        source_layers[size_t(i)] = vector<Index>{i - 1};
        validate_source_arity(*layers[size_t(i)], source_layers[size_t(i)], i);
    }

    if (const Json* tied_weights = layers_container->find("TiedWeights");
        tied_weights && tied_weights->is_array())
    {
        for (const Json& entry : tied_weights->as_array())
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
                     "Network::from_JSON: invalid tied weight indices for layer {} and source {}.",
                     layer_index, source_layer_index);

            layers[size_t(layer_index)]->set_tied_weight({
                layers[size_t(source_layer_index)].get(),
                size_t(spec_index), size_t(source_spec_index)});
        }
    }

    if (const Json* outputs_element = network_element->find("Outputs"); outputs_element)
        read_variables_array(outputs_element, "Output", output_variables, "Target");

    compile();

    if (items_array && items_array->is_array())
    {
        Index layer_index = 0;
        for (const Json& item : items_array->as_array())
        {
            if (!item.is_object() || item.as_object().empty()) continue;
            if (layer_index >= ssize(layers)) break;

            JsonDocument layer_doc;
            layer_doc.set_root(item);
            layers[layer_index]->load_state_from_JSON(layer_doc);
            ++layer_index;
        }
    }

    const Json* parameters_element = network_element->find("Parameters");
    const string parameters_text   = parameters_element ? read_json_string(parameters_element, "Values") : string();
    if (parameters_text.empty()) return;

    VectorR json_parameters;
    string_to_vector(parameters_text, json_parameters);

    throw_if(json_parameters.size() != parameters.size_in_floats(),
             "Network::from_JSON: embedded parameter size mismatch "
             "(got {}, expected {}). Supply the complete compiled parameter buffer, "
             "including alignment padding.",
             json_parameters.size(), parameters.size_in_floats());

    for (Index i = 0; i < json_parameters.size(); ++i)
        throw_if(!std::isfinite(json_parameters(i)),
                 "Network::from_JSON: non-finite embedded parameter at index {}. "
                 "All embedded parameter values must be finite.", i);

    const HostParametersGuard guard(*this);
    std::copy_n(json_parameters.data(), json_parameters.size(), parameters.as<float>());
}

void Network::save(const filesystem::path& file_name) const
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
             "Network::{}: size mismatch for {} (got {} bytes, expected {} bytes).",
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
    const Index payload_bytes = storage.byte_size();
    const bool cuda = storage.get_device() == Device::CUDA && storage.data();

    vector<char> staging(cuda ? size_t(payload_bytes) : 0);
    const void* payload = cuda ? staging.data() : storage.data();

    if (cuda)
    {
        cudaStream_t stream = device::get_compute_stream();
        device::copy_async(staging.data(), storage.data(), payload_bytes,
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);
    }

    const array<unsigned char, SNAPSHOT_FILE_HEADER_SIZE> header =
        make_snapshot_header(
            magic,
            uint64_t(storage.size_in_floats()),
            uint64_t(payload_bytes),
            layout_fingerprint,
            hash_bytes(FNV1A_OFFSET, payload, size_t(payload_bytes)));

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
    const uint64_t payload_bytes = uint64_t(storage.byte_size());

    ifstream file(file_name, ios::binary);
    throw_if(!file.is_open(), "Cannot open binary file: {}\n", file_name.string());

    const uintmax_t file_bytes = filesystem::file_size(file_name);
    const bool versioned =
        is_versioned_snapshot(file, file_bytes, payload_bytes, magic, file_name, caller);

    const uint64_t expected_checksum =
        versioned
            ? read_snapshot_header(file, file_bytes, magic,
                                   uint64_t(storage.size_in_floats()), payload_bytes,
                                   layout_fingerprint, file_name, caller, snapshot_name)
            : 0;

    const bool cuda = storage.get_device() == Device::CUDA && storage.data();
    const bool use_staging = versioned || cuda;

    vector<char> staging(use_staging ? size_t(payload_bytes) : 0);
    void* destination = use_staging ? staging.data() : storage.data();

    if (payload_bytes > 0)
        file.read(static_cast<char*>(destination), streamsize(payload_bytes));

    throw_if(!file, "Error reading binary file: {}", file_name.string());

    if (versioned)
    {
        const uint64_t checksum =
            hash_bytes(FNV1A_OFFSET, destination, size_t(payload_bytes));

        throw_if(checksum != expected_checksum,
                 "Network::{}: payload checksum mismatch for {}.",
                 caller, file_name.string());
    }

    if (payload_bytes == 0)
        return;

    if (cuda)
    {
        cudaStream_t stream = device::get_compute_stream();
        device::copy_async(storage.data(), destination, storage.byte_size(),
                           device::CopyKind::HostToDevice, stream);
        device::synchronize(stream);
    }
    else if (versioned)
    {
        memcpy(storage.data(), destination, size_t(payload_bytes));
    }
}

void Network::save_parameters_binary(const filesystem::path& file_name) const
{
    throw_if(!parameters.owns_memory(),
             "Network::save_parameters_binary: the fp32 parameter master "
             "was released for quantized inference; reload the model before saving.");

    save_binary_snapshot(file_name, parameters, PARAMETER_FILE_MAGIC,
                         parameter_layout_fingerprint());
}

void Network::save_states_binary(const filesystem::path& file_name) const
{
    save_binary_snapshot(file_name, states, STATE_FILE_MAGIC,
                         state_layout_fingerprint());
}

void Network::load(const filesystem::path& file_name)
{
    validate_model_file_paths(file_name);
    recover_model_save_transaction(file_name);

    const filesystem::path binary_path = parameter_file_path(file_name);
    const bool has_binary = filesystem::exists(binary_path);

    {
        const JsonDocument document = load_json_file(file_name);

        if (!has_binary)
        {
            const Json* root = get_json_root(document, "Network");
            const Json* embedded_parameters = root->find("Parameters");
            throw_if(!embedded_parameters
                     || read_json_string(embedded_parameters, "Values").empty(),
                     "Network::load: missing parameter file {} and no embedded "
                     "JSON weights. Restore the matching .bin file, or use "
                     "from_JSON(load_json_file(path)) explicitly for architecture-only loading.",
                     binary_path.string());
        }

        clear();
        from_JSON(document);
    } // Release the JSON document before allocating binary snapshot staging.

    if (has_binary)
        load_parameters_binary(binary_path);
}

void Network::load_parameters_binary(const filesystem::path& file_name)
{
    mark_parameters_changed();

    throw_if(fp32_master_released(),
             "Network::load_parameters_binary: the fp32 parameter master was released "
             "for quantized inference; reload the model before loading parameters.");

    load_binary_snapshot(file_name, parameters, PARAMETER_FILE_MAGIC,
                         parameter_layout_fingerprint(),
                         "load_parameters_binary", "parameter");

    if (parameters.get_device() == Device::CUDA && parameters.data())
        cast_parameters_to_bf16();

    link_parameters();
}

void Network::load_parameters_bf16_inference_binary(
    const filesystem::path& file_name)
{
    throw_if(parameters.empty() || !parameters.owns_memory(),
             "Network::load_parameters_bf16_inference_binary: "
             "the network must own its compiled parameter storage.");

    read_parameters_bf16_inference_binary(file_name, parameters.size_in_floats());
}

void Network::compile_and_load_parameters_bf16_inference_binary(
    const filesystem::path& file_name)
{
    throw_if(get_layers_number() == 0,
             "Network::compile_and_load_parameters_bf16_inference_binary: "
             "the network has no layers.");

    compile(Configuration::instance().resolve(), false);

    read_parameters_bf16_inference_binary(
        file_name, get_aligned_size(get_parameter_specs()));
}

void Network::read_parameters_bf16_inference_binary(
    const filesystem::path& file_name, const Index parameters_number)
{
    PROFILE_SCOPE_HOST("load:bf16_inference_binary");

    mark_parameters_changed();

    ifstream file = open_binary_input(
        file_name, uintmax_t(parameters_number) * sizeof(uint16_t),
        "load_parameters_bf16_inference_binary");

    constexpr Index chunk_elements = Index(8) * 1024 * 1024;
    vector<uint16_t> bf16_chunk(
        size_t(min(chunk_elements, max(Index(1), parameters_number))));
    vector<float> fp32_chunk(bf16_chunk.size());

    // Every reader below walks the file in the same fixed-size BF16 chunks and
    // reports the same error; only what each does with a chunk differs. The
    // loop was written out five times, once per reader.
    const auto for_each_bf16_chunk = [&](const Index count, auto&& consume)
    {
        Index done = 0;
        while (done < count)
        {
            const Index chunk = min(chunk_elements, count - done);
            file.read(reinterpret_cast<char*>(bf16_chunk.data()),
                      streamsize(chunk * Index(sizeof(uint16_t))));
            throw_if(!file,
                     "Error reading BF16 parameter file: {}",
                     file_name.string());
            consume(chunk, done);
            done += chunk;
        }
    };

#ifdef OPENNN_HAS_CUDA
    if (config.device == Device::CUDA)
    {
        throw_if(!is_one_of(config.training_type, Type::BF16, Type::INT8),
                 "Network::load_parameters_bf16_inference_binary: "
                 "CUDA direct loading requires BF16 or INT8 configuration.");

        const ParameterSlotTotals totals = for_each_parameter_slot({});
        allocate_compact_parameter_storage(totals);

        uint16_t* const mirror = parameters_bf16_mirror.as<uint16_t>();
        float* const fp32_compact = parameters_fp32_inference_storage.as<float>();
        int8_t* const int8_storage = parameters_int8_storage.as<int8_t>();
        cudaStream_t stream = device::get_compute_stream();

        const auto skip = [&](const Index count)
        {
            if (count <= 0) return;
            file.seekg(
                streamoff(count * Index(sizeof(uint16_t))), ios::cur);
            throw_if(!file,
                     "Error seeking through BF16 parameter file: {}",
                     file_name.string());
        };

        // Two pinned slots let the read of one chunk overlap the transfer of
        // the previous one: a slot is refilled only once the event recorded
        // behind its copy has fired, and the stream is drained once at the
        // end rather than after every chunk. Only the BF16 reader stages this
        // way -- it moves nearly every byte of a BF16 model -- so it has its
        // own loop instead of the shared one, whose buffer is fixed and whose
        // consumers wait per chunk. The slots outlive the try below because
        // the catch must drain the stream before they are released.
        struct StagingSlot
        {
            device::PinnedBuffer host;
            device::CudaEvent copied;
            bool pending = false;
        };

        array<StagingSlot, 2> staging;
        for (StagingSlot& slot : staging)
        {
            slot.host.resize_bytes(Index(bf16_chunk.size() * sizeof(uint16_t)));
            slot.copied.create();
        }
        size_t next_slot = 0;

        const auto read_bf16_to_device =
            [&](uint16_t* destination, const Index count)
        {
            Index done = 0;
            while (done < count)
            {
                StagingSlot& slot = staging[next_slot];
                next_slot = (next_slot + 1) % staging.size();
                if (slot.pending) device::synchronize_event(slot.copied.get());
                slot.pending = false;

                const Index chunk = min(chunk_elements, count - done);
                file.read(slot.host.as<char>(),
                          streamsize(chunk * Index(sizeof(uint16_t))));
                throw_if(!file,
                         "Error reading BF16 parameter file: {}",
                         file_name.string());
                device::copy_async(
                    destination + done, slot.host.data(),
                    chunk * Index(sizeof(uint16_t)),
                    Device::CPU, Device::CUDA, stream);
                device::record_event(slot.copied.get(), stream);
                slot.pending = true;
                done += chunk;
            }
        };

        const auto read_bf16_as_fp32_to_device =
            [&](float* destination, const Index count)
        {
            for_each_bf16_chunk(count, [&](const Index chunk, const Index copied)
            {
                ranges::transform(bf16_chunk | views::take(chunk), fp32_chunk.begin(),
                                  bfloat16_to_float_host);
                device::copy_async(
                    destination + copied, fp32_chunk.data(),
                    chunk * Index(sizeof(float)),
                    Device::CPU, Device::CUDA, stream);
                device::synchronize(stream);
            });
        };

        vector<int8_t> int8_chunk(bf16_chunk.size());

        const auto read_bf16_quantize_int8_to_device =
            [&](int8_t* destination, float* scale_destination,
                const Index count, const Index channels, const int axis)
        {
            const Index row_length = count / channels;
            const streampos slot_start = file.tellg();

            vector<float> scales(size_t(channels), 0.0f);
            for_each_bf16_chunk(count, [&](const Index chunk, const Index processed)
            {
                for (Index i = 0; i < chunk; ++i)
                {
                    const Index channel = quantization_channel(
                        processed + i, row_length, channels, axis);
                    scales[size_t(channel)] = max(scales[size_t(channel)],
                        abs(bfloat16_to_float_host(bf16_chunk[size_t(i)])));
                }
            });
            finalize_int8_scales(scales);

            file.seekg(slot_start);
            throw_if(!file, "Error seeking through BF16 parameter file: {}",
                     file_name.string());

            for_each_bf16_chunk(count, [&](const Index chunk, const Index processed)
            {
                ranges::transform(bf16_chunk | views::take(chunk), fp32_chunk.begin(),
                                  bfloat16_to_float_host);
                quantize_int8_host(fp32_chunk.data(), chunk, processed,
                                   row_length, channels, axis,
                                   scales.data(), int8_chunk.data());
                device::copy_async(
                    destination + processed, int8_chunk.data(),
                    chunk, Device::CPU, Device::CUDA, stream);
                device::synchronize(stream);
            });

            device::copy_async(
                scale_destination, scales.data(),
                channels * Index(sizeof(float)),
                Device::CPU, Device::CUDA, stream);
            device::synchronize(stream);
        };

        try
        {
            for_each_parameter_slot([&](const ParameterSlot& slot)
            {
                if (slot.shape.empty()) return;

                const Index size = slot.shape.size();
                const Index aligned = get_aligned_size(size);
                if (slot.tied)
                    return skip(aligned);

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
                     "Network::load_parameters_bf16_inference_binary: "
                     "unconsumed data remains in {}.",
                     file_name.string());
        }
        catch (...)
        {
            device::synchronize(stream);
            throw;
        }

        device::synchronize(stream);

        return use_compact_parameter_storage();
    }
#endif

    // A network compiled without its master gets one here: every aligned float
    // of it, padding included, is overwritten by the chunks below.
    if (parameters.empty())
        parameters.resize_bytes(parameters_number * Index(sizeof(float)), Device::CPU);

    float* const host_parameters = parameters.as<float>();

    for_each_bf16_chunk(parameters_number, [&](const Index chunk, const Index converted)
    {
        ranges::transform(bf16_chunk | views::take(chunk), host_parameters + converted,
                          bfloat16_to_float_host);
    });

    link_parameters();
}

void Network::load_states_binary(const filesystem::path& file_name)
{
    load_binary_snapshot(file_name, states, STATE_FILE_MAGIC,
                         state_layout_fingerprint(),
                         "load_states_binary", "state");

    link_states();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
