//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/layer.h"

#include "opennn/core/json.h"
#include "opennn/core/device_backend.h"
#include "opennn/registry.h"

namespace opennn
{

namespace
{

const Json* get_layer_json_root(const JsonDocument& document, const Layer& layer)
{
    if (const Json* root = document.first_child(layer.get_name()))
        return root;

    const Json& document_root = document.get_root();
    if (document_root.is_object() && document_root.as_object().size() == 1)
    {
        const auto& [serialized_name, value] = document_root.as_object().front();
        if (string_to_layer_type(serialized_name) == layer.get_type())
            return &value;
    }

    return get_json_root(document, layer.get_name());
}

}

vector<TensorSpec> Layer::get_parameter_specs() const
{
    vector<TensorSpec> result;
    for (Operator* op : get_operators())
    {
        auto specs = op->parameter_specs();
        result.insert(result.end(),
                      make_move_iterator(specs.begin()),
                      make_move_iterator(specs.end()));
    }

    return result;
}

vector<TensorSpec> Layer::get_state_specs() const
{
    vector<TensorSpec> result;
    for (Operator* op : get_operators())
    {
        auto specs = op->state_specs();
        result.insert(result.end(),
                      make_move_iterator(specs.begin()),
                      make_move_iterator(specs.end()));
    }

    return result;
}

vector<Operator::SlotQuantization> Layer::get_parameter_quantization() const
{
    vector<Operator::SlotQuantization> result;
    for (Operator* op : get_operators())
    {
        const size_t n = op->parameter_specs().size();
        auto quantization = op->parameter_quantization();
        quantization.resize(n);
        result.insert(result.end(), quantization.begin(), quantization.end());
    }

    return result;
}

void Layer::redistribute_parameters_to_operators()
{
    const bool has_scales = parameter_scales.size() == parameters.size();

    size_t offset = 0;
    for (Operator* op : get_operators())
    {
        const size_t n = op->parameter_specs().size();
        if (n == 0) continue;
        throw_if(offset + n > parameters.size(),
                 "Layer::redistribute_parameters_to_operators: missing parameter views in layer \"{}\"", get_name());
        op->link_parameters(span(parameters).subspan(offset, n));
        if (has_scales)
            op->link_parameter_scales(span(parameter_scales).subspan(offset, n));
        offset += n;
    }

    throw_if(offset != parameters.size(),
             "Layer::redistribute_parameters_to_operators: excess parameter views in layer \"{}\"", get_name());
}

Index Layer::get_parameters_number() const
{
    Index count = 0;

    for (Operator* op : get_operators())
        for (const auto& [shape, _] : op->parameter_specs())
            count += shape.size();

    return count;
}

float* Layer::link_views_to_operators(vector<TensorView>& views, float* pointer,
                                      vector<TensorSpec> (Operator::*specs_fn)() const,
                                      void (Operator::*link_fn)(span<const TensorView>),
                                      Device device)
{
    views.clear();

    for (Operator* op : get_operators())
    {
        const auto specs = (op->*specs_fn)();
        if (specs.empty()) continue;

        const size_t start = views.size();

        for (const auto& [shape, _] : specs)
        {
            if (shape.empty()) { views.emplace_back(); continue; }

            throw_if(!is_aligned(pointer),
                     "Layer::link_views_to_operators: unaligned memory in layer \"{}\"", get_name());

            views.emplace_back(pointer, shape, Type::FP32, device);
            pointer += get_aligned_size(shape.size());
        }

        (op->*link_fn)(span(views).subspan(start));
    }

    return pointer;
}

float* Layer::link_states(float* pointer, Device device)
{
    return link_views_to_operators(states, pointer,
                                   &Operator::state_specs,
                                   &Operator::link_states,
                                   device);
}

bool Layer::refresh_feature_storage(Buffer& storage, bool& dirty, Device device,
                                    Index features, Index columns,
                                    const function<void(float*)>& fill)
{
    const Index bytes = columns * features * Index(sizeof(float));
    if (!dirty && storage.byte_size() == bytes && storage.get_device() == device)
        return false;

    storage.resize_bytes(bytes, device);
    dirty = false;

    if (features == 0) return true;

    vector<float> staging(size_t(columns * features));
    fill(staging.data());

    if (device == Device::CUDA)
        opennn::device::copy_async(storage.data(), staging.data(), bytes,
                                   opennn::device::CopyKind::HostToDevice);
    else
        memcpy(storage.data(), staging.data(), size_t(bytes));

    return true;
}

float* Layer::link_gradients(float* pointer, Device device)
{
    vector<TensorView> gradients;
    return link_views_to_operators(gradients, pointer,
                                   &Operator::parameter_specs,
                                   &Operator::link_gradients,
                                   device);
}

void Layer::from_JSON(const JsonDocument& document)
{
    const Json* root = get_layer_json_root(document, *this);

    const string json_label = read_json_string(root, "Label");

    set_input_shape(string_to_shape(read_json_string(root, "InputDimensions")));
    set_output_shape(string_to_shape(read_json_string(root, "OutputDimensions")));
    set_label(json_label);
    if (root->has("Trainable"))
        set_is_trainable(read_json_bool(root, "Trainable"));

    read_JSON_body(root);
    for (Operator* op : get_operators())
        op->from_JSON(root);

    on_loaded();
}

void Layer::load_state_from_JSON(const JsonDocument& document)
{
    const Json* root = get_layer_json_root(document, *this);
    for (Operator* op : get_operators())
        op->load_state_from_JSON(root);
}

void Layer::to_JSON(JsonWriter& writer) const
{
    writer.open_element(get_name());

    add_json_field(writer, "Label", label);
    add_json_field(writer, "InputDimensions", shape_to_string(get_input_shape()));
    add_json_field(writer, "OutputDimensions", shape_to_string(get_output_shape()));
    add_json_field(writer, "Trainable", is_trainable);

    write_JSON_body(writer);

    for (Operator* op : get_operators())
        op->to_JSON(writer);

    writer.close_element();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
