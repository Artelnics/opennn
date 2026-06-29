//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "device_backend.h"
#include "tensor_types.h"
#include "bounding_layer.h"
#include "string_utilities.h"
#include "json.h"

namespace opennn
{

Bounding::Bounding(const Shape& new_output_shape, const string& new_name)
    : Layer(LayerType::Bounding, false)
{
    operators = {&bound};
    set(new_output_shape, new_name);
}

VectorR Bounding::get_lower_bounds() const
{
    return Map<const VectorR>(lower_bounds.data(), ssize(lower_bounds));
}

VectorR Bounding::get_upper_bounds() const
{
    return Map<const VectorR>(upper_bounds.data(), ssize(upper_bounds));
}

const EnumMap<Bounding::BoundingMethod>& Bounding::bounding_method_map()
{
    static const vector<pair<BoundingMethod, string>> entries = {
        {BoundingMethod::NoBounding, "NoBounding"},
        {BoundingMethod::NoBounding, "No bounding"},
        {BoundingMethod::Bounding,   "Bounding"},
        {BoundingMethod::Bounding,   "Positive outputs"},
        {BoundingMethod::Bounding,   "Data range"}
    };
    static const EnumMap<BoundingMethod> map{entries};
    return map;
}

void Bounding::set(const Shape& new_output_shape, const string& new_label)
{
    output_shape = new_output_shape;

    set_label(new_label);

    const Index features = output_shape.dim_or_zero(0);
    bound.method = BoundingMethod::Bounding;

    lower_bounds.assign(size_t(features), -numeric_limits<float>::max());
    upper_bounds.assign(size_t(features),  numeric_limits<float>::max());
    op_storage_dirty = true;
}

void Bounding::set_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape, label);
}

void Bounding::set_bounding_method(const BoundingMethod& new_method)
{
    bound.method = new_method;
}

void Bounding::set_bounding_method(const string& new_method_string)
{
    bound.method = bounding_method_map().from_string(new_method_string);
}

void Bounding::set_lower_bound(Index index, float new_lower_bound)
{
    throw_if(index < 0 || size_t(index) >= lower_bounds.size(),
             format("Bounding::set_lower_bound: index {} out of range [0, {}).",
                    index, lower_bounds.size()));
    lower_bounds[size_t(index)] = new_lower_bound;
    op_storage_dirty = true;
    refresh_op_storage(op_storage_device);
}

void Bounding::set_upper_bound(Index index, float new_upper_bound)
{
    throw_if(index < 0 || size_t(index) >= upper_bounds.size(),
             format("Bounding::set_upper_bound: index {} out of range [0, {}).",
                    index, upper_bounds.size()));
    upper_bounds[size_t(index)] = new_upper_bound;
    op_storage_dirty = true;
    refresh_op_storage(op_storage_device);
}

float* Bounding::link_states(float* pointer, Device device)
{
    refresh_op_storage(device);
    return pointer;
}

void Bounding::refresh_op_storage(Device device)
{
    op_storage_device = device;

    const Index features = ssize(lower_bounds);
    const Index bytes    = 2 * features * Index(sizeof(float));

    const bool needs = op_storage_dirty
                    || op_storage.bytes       != bytes
                    || op_storage.device_type != device;
    if (!needs) return;

    op_storage.resize_bytes(bytes, device);

    if (features == 0)
    {
        bound.lower = bound.upper = TensorView();
        op_storage_dirty = false;
        return;
    }

    const Index feature_bytes = features * Index(sizeof(float));

    if (device == Device::CUDA)
    {
        opennn::device::copy_async(op_storage.as<float>(), lower_bounds.data(),
                                   feature_bytes,
                                   opennn::device::CopyKind::HostToDevice);
        opennn::device::copy_async(op_storage.as<float>() + features, upper_bounds.data(),
                                   feature_bytes,
                                   opennn::device::CopyKind::HostToDevice);
    }
    else
    {
        memcpy(op_storage.as<float>(), lower_bounds.data(), static_cast<size_t>(feature_bytes));
        memcpy(op_storage.as<float>() + features, upper_bounds.data(), static_cast<size_t>(feature_bytes));
    }

    float* const base = op_storage.as<float>();
    const Shape shape{features};
    bound.lower = TensorView(base, shape, Type::FP32, device);
    bound.upper = TensorView(base + 1 * features, shape, Type::FP32, device);

    op_storage_dirty = false;
}

void Bounding::read_JSON_body(const Json* root_element)
{
    if (!root_element) return;

    set_bounding_method(read_json_string(root_element, "BoundingMethod"));

    auto parse_bounds = [&](const string& field, vector<float>& dest)
    {
        if (!root_element->has(field)) return;
        VectorR values;
        string_to_vector(read_json_string(root_element, field), values);
        throw_if(values.size() != ssize(dest),
                 format("Bounding::read_JSON_body: field \"{}\" has size {}, expected {}.",
                        field, values.size(), dest.size()));
        for (Index i = 0; i < values.size(); ++i)
            dest[size_t(i)] = values(i);
    };

    parse_bounds("LowerBounds", lower_bounds);
    parse_bounds("UpperBounds", upper_bounds);

    op_storage_dirty = true;
    refresh_op_storage(op_storage_device);
}

void Bounding::write_JSON_body(JsonWriter& printer) const
{
    if (bound.method == BoundingMethod::Bounding && !lower_bounds.empty())
    {
        add_json_field(printer, "LowerBounds", vector_to_string(get_lower_bounds()));
        add_json_field(printer, "UpperBounds", vector_to_string(get_upper_bounds()));
    }

    add_json_field(printer, "BoundingMethod", bounding_method_map().to_string(bound.method));
}

string Bounding::write_expression(const vector<string>& input_names,
                                  const vector<string>& output_names) const
{
    if (get_bounding_method() == BoundingMethod::NoBounding)
        return {};

    ostringstream buffer;
    buffer.precision(10);

    for (Index i = 0; i < output_shape[0]; ++i)
        buffer << output_names[i] << " = max(" << lower_bounds[size_t(i)] << ", " << input_names[i] << ")\n"
               << output_names[i] << " = min(" << upper_bounds[size_t(i)] << ", " << output_names[i] << ")\n";

    return buffer.str();
}

REGISTER(Layer, Bounding, "Bounding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
