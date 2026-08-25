//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C L A M P I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/clamping_layer.h"

#include "opennn/core/device_backend.h"
#include "opennn/core/json.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/registry.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/neural_network/layers/kernel_scaling.cuh"
#endif

namespace opennn
{

static void apply_clamping_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&);

static void apply_clamping_cpu(const TensorView& input,
               const TensorView& lower_bounds,
               const TensorView& upper_bounds,
               TensorView& output)
{
    const Index features = lower_bounds.size();

    const MatrixMap input_matrix = input.as_flat_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_flat_matrix();

    const Index columns = output_matrix.cols();

    for (Index column_index = 0; column_index < columns; ++column_index)
    {
        const Index feature_index = column_index % features;
        output_matrix.col(column_index) = input_matrix.col(column_index)
                                                        .cwiseMax(lower_bounds_vector(feature_index))
                                                        .cwiseMin(upper_bounds_vector(feature_index));
    }
}

void apply_clamping(const TensorView& input,
           const TensorView& lower_bounds,
           const TensorView& upper_bounds,
           TensorView& output)
{
    if (input.is_cuda()) { apply_clamping_gpu(input, lower_bounds, upper_bounds, output); return; }
    apply_clamping_cpu(input, lower_bounds, upper_bounds, output);
}

#ifdef OPENNN_HAS_CUDA

static void apply_clamping_gpu(const TensorView& input,
               const TensorView& lower_bounds,
               const TensorView& upper_bounds,
               TensorView& output)
{
    visit_type_pair<Type::FP32, Type::BF16>(input.get_type(), output.get_type(), [&]<typename TIn, typename TOut>() {
        clamping_cuda<TIn, TOut>(output.size(), to_int(lower_bounds.size()),
                                 input.as<TIn>(),
                                 lower_bounds.as_float(),
                                 upper_bounds.as_float(),
                                 output.as<TOut>());
    });
}

#else

OPENNN_CUDA_STUB(void, apply_clamping_gpu,
                 (const TensorView&, const TensorView&, const TensorView&, TensorView&))

#endif

void ClampingOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    if (method == Method::NoClamping || !lower.get_data())
        return copy(input, output);

    apply_clamping(input, lower, upper, output);
}

Clamping::Clamping(const Shape& new_output_shape, const string& new_name)
    : Layer(LayerType::Clamping, false)
{
    operators = {&clamping};
    set(new_output_shape, new_name);
}

const EnumMap<Clamping::ClampingMethod>& Clamping::clamping_method_map()
{
    static const EnumMap<ClampingMethod> map{
        {ClampingMethod::NoClamping, "NoClamping"},
        {ClampingMethod::NoClamping, "No clamping"},
        {ClampingMethod::Clamping,   "Clamping"},
        {ClampingMethod::Clamping,   "Positive outputs"},
        {ClampingMethod::Clamping,   "Data range"},
        {ClampingMethod::NoClamping, "NoBounding"},
        {ClampingMethod::NoClamping, "No bounding"},
        {ClampingMethod::Clamping,   "Bounding"}
    };
    return map;
}

void Clamping::set(const Shape& new_output_shape, const string& new_label)
{
    output_shape = new_output_shape;

    set_label(new_label);

    const Index features = output_shape.empty() ? 0 : output_shape.back();
    clamping.method = ClampingMethod::Clamping;

    lower_bounds.assign(size_t(features), -MAX);
    upper_bounds.assign(size_t(features),  MAX);
    op_storage_dirty = true;
}

void Clamping::set_clamping_method(const ClampingMethod& new_method)
{
    clamping.method = new_method;
}

void Clamping::set_clamping_method(const string& new_method_string)
{
    clamping.method = clamping_method_map().from_string(new_method_string);
}

void Clamping::set_lower_bound(Index index, float new_lower_bound)
{
    throw_if(index < 0 || size_t(index) >= lower_bounds.size(),
             "Clamping::set_lower_bound: index {} out of range [0, {}).",
                    index, lower_bounds.size());
    lower_bounds[size_t(index)] = new_lower_bound;
    op_storage_dirty = true;
    refresh_op_storage(op_storage.get_device());
}

void Clamping::set_upper_bound(Index index, float new_upper_bound)
{
    throw_if(index < 0 || size_t(index) >= upper_bounds.size(),
             "Clamping::set_upper_bound: index {} out of range [0, {}).",
                    index, upper_bounds.size());
    upper_bounds[size_t(index)] = new_upper_bound;
    op_storage_dirty = true;
    refresh_op_storage(op_storage.get_device());
}

float* Clamping::link_states(float* pointer, Device device)
{
    refresh_op_storage(device);
    return pointer;
}

void Clamping::refresh_op_storage(Device device)
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
        clamping.lower = clamping.upper = TensorView();
        return;
    }

    float* const base = op_storage.as<float>();
    const Shape shape{features};
    clamping.lower = TensorView(base, shape, Type::FP32, device);
    clamping.upper = TensorView(base + 1 * features, shape, Type::FP32, device);
}

void Clamping::read_JSON_body(const Json* root_element)
{
    if (!root_element) return;

    set_clamping_method(read_json_string_fallback(
        root_element, {"ClampingMethod", "BoundingMethod"}));

    const auto parse_bounds = [&](const string& field, vector<float>& dest)
    {
        if (!root_element->has(field)) return;
        VectorR values;
        string_to_vector(read_json_string(root_element, field), values);
        throw_if(values.size() != ssize(dest),
                 "Clamping::read_JSON_body: field \"{}\" has size {}, expected {}.",
                        field, values.size(), dest.size());
        for (Index i = 0; i < values.size(); ++i)
            dest[size_t(i)] = values(i);
    };

    parse_bounds("LowerBounds", lower_bounds);
    parse_bounds("UpperBounds", upper_bounds);

    op_storage_dirty = true;
    refresh_op_storage(op_storage.get_device());
}

void Clamping::write_JSON_body(JsonWriter& printer) const
{
    if (clamping.method == ClampingMethod::Clamping && !lower_bounds.empty())
    {
        add_json_field(printer, "LowerBounds", vector_to_string(get_lower_bounds()));
        add_json_field(printer, "UpperBounds", vector_to_string(get_upper_bounds()));
    }

    add_json_field(printer, "ClampingMethod", clamping_method_map().to_string(clamping.method));
}

string Clamping::write_expression(const vector<string>& input_names,
                                  const vector<string>& output_names) const
{
    if (get_clamping_method() == ClampingMethod::NoClamping)
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
