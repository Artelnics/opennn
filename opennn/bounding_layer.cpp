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

namespace opennn
{

Bounding::Bounding(const Shape& output_shape, const string& new_name) : Layer()
{
    set(output_shape, new_name);
}

const Bounding::BoundingMethod& Bounding::get_bounding_method() const
{
    return bounding_method;
}

Shape Bounding::get_output_shape() const
{
    return output_shape;
}

VectorR Bounding::get_lower_bounds() const
{
    if(ssize(states) <= Lower || !states[Lower].data) return VectorR();
    return states[Lower].as_vector();
}

VectorR Bounding::get_upper_bounds() const
{
    if(ssize(states) <= Upper || !states[Upper].data) return VectorR();
    return states[Upper].as_vector();
}

void Bounding::set(const Shape& new_output_shape, const string& new_label)
{
    set_output_shape(new_output_shape);

    label = new_label;

    bounding_method = BoundingMethod::Bounding;

    name = "Bounding";
    layer_type = LayerType::Bounding;

    is_trainable = false;
}

void Bounding::set_bounding_method(const BoundingMethod& new_method)
{
    bounding_method = new_method;
}

void Bounding::set_bounding_method(const string& new_method_string)
{
    bounding_method = bounding_method_map().from_string(new_method_string);
}

void Bounding::set_input_shape(const Shape& new_input_shape)
{
    if(!output_shape.empty() && new_input_shape != output_shape)
        throw runtime_error("Bounding: input shape mismatch with output shape.");
}

void Bounding::set_output_shape(const Shape& new_output_shape)
{
    output_shape = new_output_shape;
}

// States[] is allocated by NN::compile() → Layer::link_states(). This override initializes
// the arena slots to defaults (±max) since they're zero-initialized by the base.
type* Bounding::link_states(type* pointer)
{
    type* next = Layer::link_states(pointer);

    if(bounding_method == BoundingMethod::NoBounding) return next;

    if(states[Lower].data)
        VectorMap(states[Lower].as<float>(), states[Lower].size()).setConstant(-numeric_limits<type>::max());

    if(states[Upper].data)
        VectorMap(states[Upper].as<float>(), states[Upper].size()).setConstant(numeric_limits<type>::max());

    return next;
}

void Bounding::set_lower_bound(const Index index, type new_lower_bound)
{
    if(ssize(states) <= Lower || !states[Lower].data)
        throw runtime_error("Bounding::set_lower_bound: layer not compiled yet (call NeuralNetwork::compile() first).");

    states[Lower].as<float>()[index] = new_lower_bound;
}

void Bounding::set_lower_bounds(const VectorR& new_lower_bounds)
{
    if(ssize(states) <= Lower || !states[Lower].data)
        throw runtime_error("Bounding::set_lower_bounds: layer not compiled yet (call NeuralNetwork::compile() first).");

    VectorMap(states[Lower].as<float>(), states[Lower].size()) = new_lower_bounds;
}

void Bounding::set_upper_bounds(const VectorR& new_upper_bounds)
{
    if(ssize(states) <= Upper || !states[Upper].data)
        throw runtime_error("Bounding::set_upper_bounds: layer not compiled yet (call NeuralNetwork::compile() first).");

    VectorMap(states[Upper].as<float>(), states[Upper].size()) = new_upper_bounds;
}

void Bounding::set_upper_bound(const Index index, type new_upper_bound)
{
    if(ssize(states) <= Upper || !states[Upper].data)
        throw runtime_error("Bounding::set_upper_bound: layer not compiled yet (call NeuralNetwork::compile() first).");

    states[Upper].as<float>()[index] = new_upper_bound;
}

void Bounding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer_index, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer_index];

    if(bounding_method == BoundingMethod::NoBounding)
    {
        copy(forward_views[Input][0], forward_views[Output][0]);
        return;
    }

    bounding(forward_views[Input][0],
             states[Lower],
             states[Upper],
             forward_views[Output][0]);
}

void Bounding::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Bounding");

    add_xml_element(printer, "NeuronsNumber", to_string(output_shape[0]));

    if(bounding_method == BoundingMethod::Bounding && ssize(states) > Upper && states[Lower].data)
    {
        add_xml_element(printer, "LowerBounds", vector_to_string(states[Lower].as_vector()));
        add_xml_element(printer, "UpperBounds", vector_to_string(states[Upper].as_vector()));
    }

    add_xml_element(printer, "BoundingMethod", bounding_method_map().to_string(bounding_method));

    printer.close_element();
}

// Phase 1: parse config only; states[] isn't allocated yet.
void Bounding::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "Bounding");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set({ neurons_number });

    set_bounding_method(read_xml_string(root_element, "BoundingMethod"));
}

// Phase 2: states[] is allocated; parse bounds directly into arena.
void Bounding::load_state_from_XML(const XmlDocument& document)
{
    if(bounding_method == BoundingMethod::NoBounding) return;
    if(ssize(states) <= Upper || !states[Lower].data) return;

    const XmlElement* root_element = get_xml_root(document, "Bounding");

    VectorR tmp;
    string_to_vector(read_xml_string(root_element, "LowerBounds"), tmp);
    if(tmp.size() == states[Lower].size())
        VectorMap(states[Lower].as<float>(), states[Lower].size()) = tmp;

    string_to_vector(read_xml_string(root_element, "UpperBounds"), tmp);
    if(tmp.size() == states[Upper].size())
        VectorMap(states[Upper].as<float>(), states[Upper].size()) = tmp;
}

REGISTER(Layer, Bounding, "Bounding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
