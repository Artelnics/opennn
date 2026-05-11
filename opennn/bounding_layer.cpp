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

namespace opennn
{

Bounding::Bounding(const Shape& new_output_shape, const string& new_name)
    : Layer("Bounding", LayerType::Bounding, false)
{
    operators = {&bound};
    set(new_output_shape, new_name);
}

Shape Bounding::get_output_shape() const
{
    return output_shape;
}

VectorR Bounding::get_lower_bounds() const
{
    if (!bound.lower.data) return VectorR();
    return bound.lower.as_vector();
}

VectorR Bounding::get_upper_bounds() const
{
    if (!bound.upper.data) return VectorR();
    return bound.upper.as_vector();
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
    bound.set(BoundingMethod::Bounding, features);
    bound.input_slots  = {Input};
    bound.output_slots = {Output};
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
    if (!bound.lower.data)
        throw runtime_error("Bounding::set_lower_bound: layer not compiled yet (call NeuralNetwork::compile() first).");

    bound.lower.as<float>()[index] = new_lower_bound;
}

void Bounding::set_lower_bounds(const VectorR& new_lower_bounds)
{
    if (!bound.lower.data)
        throw runtime_error("Bounding::set_lower_bounds: layer not compiled yet (call NeuralNetwork::compile() first).");

    bound.lower.as_vector() = new_lower_bounds;
}

void Bounding::set_upper_bound(Index index, float new_upper_bound)
{
    if (!bound.upper.data)
        throw runtime_error("Bounding::set_upper_bound: layer not compiled yet (call NeuralNetwork::compile() first).");

    bound.upper.as<float>()[index] = new_upper_bound;
}

void Bounding::set_upper_bounds(const VectorR& new_upper_bounds)
{
    if (!bound.upper.data)
        throw runtime_error("Bounding::set_upper_bounds: layer not compiled yet (call NeuralNetwork::compile() first).");

    bound.upper.as_vector() = new_upper_bounds;
}

void Bounding::read_JSON_body(const Json* root_element)
{
    set_bounding_method(read_json_string(root_element, "BoundingMethod"));
}

void Bounding::write_JSON_body(JsonWriter& printer) const
{
    if (bound.method == BoundingMethod::Bounding && bound.lower.data)
    {
        add_json_field(printer, "LowerBounds", vector_to_string(bound.lower.as_vector()));
        add_json_field(printer, "UpperBounds", vector_to_string(bound.upper.as_vector()));
    }

    add_json_field(printer, "BoundingMethod", bounding_method_map().to_string(bound.method));
}

string Bounding::write_expression(const vector<string>& input_names,
                                  const vector<string>& output_names) const
{
    if (get_bounding_method() == BoundingMethod::NoBounding)
        return string();

    ostringstream buffer;
    buffer.precision(10);

    const Shape output_shape = get_output_shape();
    const VectorR& lower_bounds = get_lower_bounds();
    const VectorR& upper_bounds = get_upper_bounds();

    for (Index i = 0; i < output_shape[0]; ++i)
        buffer << output_names[i] << " = max(" << lower_bounds[i] << ", " << input_names[i] << ")\n"
               << output_names[i] << " = min(" << upper_bounds[i] << ", " << output_names[i] << ")\n";

    return buffer.str();
}

REGISTER(Layer, Bounding, "Bounding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
