//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "bounding_layer.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "string_utilities.h"
#include "cuda_dispatch.h"
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
    if (index < 0 || size_t(index) >= lower_bounds.size())
        throw runtime_error(format("Bounding::set_lower_bound: index {} out of range [0, {}).",
                                   index, lower_bounds.size()));
    lower_bounds[size_t(index)] = new_lower_bound;
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Bounding::set_lower_bounds(const VectorR& new_lower_bounds)
{
    if (new_lower_bounds.size() != ssize(lower_bounds))
        throw runtime_error(format("Bounding::set_lower_bounds: size mismatch (expected {}, got {}).",
                                   lower_bounds.size(), new_lower_bounds.size()));
    for (Index i = 0; i < new_lower_bounds.size(); ++i)
        lower_bounds[size_t(i)] = new_lower_bounds(i);
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Bounding::set_upper_bound(Index index, float new_upper_bound)
{
    if (index < 0 || size_t(index) >= upper_bounds.size())
        throw runtime_error(format("Bounding::set_upper_bound: index {} out of range [0, {}).",
                                   index, upper_bounds.size()));
    upper_bounds[size_t(index)] = new_upper_bound;
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Bounding::set_upper_bounds(const VectorR& new_upper_bounds)
{
    if (new_upper_bounds.size() != ssize(upper_bounds))
        throw runtime_error(format("Bounding::set_upper_bounds: size mismatch (expected {}, got {}).",
                                   upper_bounds.size(), new_upper_bounds.size()));
    for (Index i = 0; i < new_upper_bounds.size(); ++i)
        upper_bounds[size_t(i)] = new_upper_bounds(i);
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

float* Bounding::link_states(float* pointer)
{
    refresh_op_storage(current_device());
    return pointer;
}

void Bounding::refresh_op_storage(Device device)
{
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

    vector<float> staging(size_t(2 * features));
    for (Index i = 0; i < features; ++i)
    {
        staging[size_t(0 * features + i)] = lower_bounds[size_t(i)];
        staging[size_t(1 * features + i)] = upper_bounds[size_t(i)];
    }

#ifdef OPENNN_HAS_CUDA
    if (device == Device::CUDA)
    {
        CHECK_CUDA(cudaMemcpy(op_storage.data, staging.data(),
                              size_t(bytes), cudaMemcpyHostToDevice));
    }
    else
#endif
    {
        memcpy(op_storage.data, staging.data(), size_t(bytes));
    }

    float* const base = op_storage.as<float>();
    const Shape shape{features};
    bound.lower = TensorView(base + 0 * features, shape, Type::FP32);
    bound.upper = TensorView(base + 1 * features, shape, Type::FP32);

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
        if (values.size() != ssize(dest))
            throw runtime_error(format("Bounding::read_JSON_body: field \"{}\" has size {}, expected {}.",
                                       field, values.size(), dest.size()));
        for (Index i = 0; i < values.size(); ++i)
            dest[size_t(i)] = values(i);
    };

    parse_bounds("LowerBounds", lower_bounds);
    parse_bounds("UpperBounds", upper_bounds);

    op_storage_dirty = true;
    refresh_op_storage(current_device());
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
