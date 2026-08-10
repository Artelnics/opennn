//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/layers/bounding_layer.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/json.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/neural_network/layers/kernel_scaling.cuh"
#endif

namespace opennn
{

// Defined below: against the CUDA kernel, or as a throwing stub.
static void bound_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&);

static void bound_cpu(const TensorView& input,
               const TensorView& lower_bounds,
               const TensorView& upper_bounds,
               TensorView& output)
{
    const Index features = lower_bounds.size();

    const MatrixMap input_matrix = input.as_flat_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_flat_matrix();

    for (Index feature_index = 0; feature_index < features; ++feature_index)
        output_matrix.col(feature_index) = input_matrix.col(feature_index)
                                                        .cwiseMax(lower_bounds_vector(feature_index))
                                                        .cwiseMin(upper_bounds_vector(feature_index));
}

void bound(const TensorView& input,
           const TensorView& lower_bounds,
           const TensorView& upper_bounds,
           TensorView& output)
{
    if (input.is_cuda()) { bound_gpu(input, lower_bounds, upper_bounds, output); return; }
    bound_cpu(input, lower_bounds, upper_bounds, output);
}

#ifdef OPENNN_HAS_CUDA

static void bound_gpu(const TensorView& input,
               const TensorView& lower_bounds,
               const TensorView& upper_bounds,
               TensorView& output)
{
    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&]<typename TIn, typename TOut>() {
        bounding_cuda<TIn, TOut>(output.size(), to_int(lower_bounds.size()),
                                 input.as<TIn>(),
                                 lower_bounds.as_float(),
                                 upper_bounds.as_float(),
                                 output.as<TOut>());
    });
}

#else

static void bound_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&)
{
    throw runtime_error("bound_gpu: CUDA support not compiled in.");
}

#endif

void BoundOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    if (method == Method::NoBounding || !lower.data)
    {
        copy(input, output);
        return;
    }

    bound(input, lower, upper, output);
}

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

    lower_bounds.assign(size_t(features), -MAX);
    upper_bounds.assign(size_t(features),  MAX);
    op_storage_dirty = true;
}

void Bounding::apply_input_shape(const Shape& new_input_shape)
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
             "Bounding::set_lower_bound: index {} out of range [0, {}).",
                    index, lower_bounds.size());
    lower_bounds[size_t(index)] = new_lower_bound;
    op_storage_dirty = true;
    refresh_op_storage(op_storage.device_type);
}

void Bounding::set_upper_bound(Index index, float new_upper_bound)
{
    throw_if(index < 0 || size_t(index) >= upper_bounds.size(),
             "Bounding::set_upper_bound: index {} out of range [0, {}).",
                    index, upper_bounds.size());
    upper_bounds[size_t(index)] = new_upper_bound;
    op_storage_dirty = true;
    refresh_op_storage(op_storage.device_type);
}

float* Bounding::link_states(float* pointer, Device device)
{
    refresh_op_storage(device);
    return pointer;
}

void Bounding::refresh_op_storage(Device device)
{
    const Index features = ssize(lower_bounds);

    if (!refresh_feature_storage(op_storage, op_storage_dirty, device, features, 2,
            [&](float* staging)
            {
                memcpy(staging, lower_bounds.data(), size_t(features) * sizeof(float));
                memcpy(staging + features, upper_bounds.data(), size_t(features) * sizeof(float));
            }))
        return;

    if (features == 0)
    {
        bound.lower = bound.upper = TensorView();
        return;
    }

    float* const base = op_storage.as<float>();
    const Shape shape{features};
    bound.lower = TensorView(base, shape, Type::FP32, device);
    bound.upper = TensorView(base + 1 * features, shape, Type::FP32, device);
}

void Bounding::read_JSON_body(const Json* root_element)
{
    if (!root_element) return;

    set_bounding_method(read_json_string(root_element, "BoundingMethod"));

    const auto parse_bounds = [&](const string& field, vector<float>& dest)
    {
        if (!root_element->has(field)) return;
        VectorR values;
        string_to_vector(read_json_string(root_element, field), values);
        throw_if(values.size() != ssize(dest),
                 "Bounding::read_JSON_body: field \"{}\" has size {}, expected {}.",
                        field, values.size(), dest.size());
        for (Index i = 0; i < values.size(); ++i)
            dest[size_t(i)] = values(i);
    };

    parse_bounds("LowerBounds", lower_bounds);
    parse_bounds("UpperBounds", upper_bounds);

    op_storage_dirty = true;
    refresh_op_storage(op_storage.device_type);
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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
