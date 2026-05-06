//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "pooling_layer_3d.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Pooling3d::Pooling3d(const Shape& new_input_shape,
                     const PoolingMethod& new_pooling_method,
                     const string& new_name) : Layer()
{
    name = "Pooling3d";
    layer_type = LayerType::Pooling3d;

    set(new_input_shape, new_pooling_method, new_name);
}

// Getters

Shape Pooling3d::get_output_shape() const
{
    return {input_features};
}

string Pooling3d::write_pooling_method() const
{
    return pooling_method_to_string(pooling_method);
}

vector<pair<Shape, Type>> Pooling3d::get_forward_specs(Index batch_size) const
{
    return {
        {(pooling_method == PoolingMethod::MaxPooling)
            ? Shape{batch_size, input_features}
            : Shape{},
         Type::FP32},                                  // MaximalIndices
        {{batch_size, input_features}, compute_dtype}, // Output (must be last)
    };
}

vector<pair<Shape, Type>> Pooling3d::get_backward_specs(Index batch_size) const
{
    return {{{batch_size, sequence_length, input_features}, compute_dtype}};
}

// Setters

void Pooling3d::set(const Shape& new_input_shape,
                    const PoolingMethod& new_pooling_method,
                    const string& new_label)
{
    sequence_length = new_input_shape.empty() ? Index(0) : new_input_shape[0];
    input_features  = new_input_shape.rank >= 2 ? new_input_shape[1] : Index(0);

    pooling_method = new_pooling_method;

    set_label(new_label);
}

void Pooling3d::set_pooling_method(const string& new_pooling_method)
{
    pooling_method = string_to_pooling_method(new_pooling_method);
}

// Forward / back propagation

void Pooling3d::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    if (pooling_method == PoolingMethod::MaxPooling)
        max_pooling_3d_forward(forward_views[Input][0],
                               forward_views[Output][0],
                               forward_views[MaximalIndices][0],
                               is_training);
    else
        average_pooling_3d_forward(forward_views[Input][0],
                                   forward_views[Output][0]);
}

void Pooling3d::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& delta_views = back_propagation.delta_views[layer];

    if (pooling_method == PoolingMethod::MaxPooling)
        max_pooling_3d_backward(forward_views[MaximalIndices][0],
                                delta_views[OutputDelta][0],
                                delta_views[InputDelta][0]);
    else
        average_pooling_3d_backward(forward_views[Input][0],
                                    delta_views[OutputDelta][0],
                                    delta_views[InputDelta][0]);
}

// Serialization

void Pooling3d::from_JSON(const JsonDocument& document)
{
    const Json* element = get_json_root(document, "Pooling3d");

    set_label(read_json_string(element, "Label"));
    set_input_shape(string_to_shape(read_json_string(element, "InputDimensions")));
    set_pooling_method(read_json_string(element, "PoolingMethod"));
}

void Pooling3d::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Pooling3d");

    write_json(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(get_input_shape())},
        {"PoolingMethod", write_pooling_method()}
    });

    printer.close_element();
}

REGISTER(Layer, Pooling3d, "Pooling3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
