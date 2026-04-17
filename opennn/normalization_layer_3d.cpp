//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "normalization_layer_3d.h"
#include "neural_network.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Normalization3d::Normalization3d(const Shape& new_input_shape,
                                 const string& new_name) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_name);
}

Shape Normalization3d::get_input_shape() const
{
    return { sequence_length, embedding_dimension };
}

Shape Normalization3d::get_output_shape() const
{
    return { sequence_length, embedding_dimension };
}

vector<Shape> Normalization3d::get_parameter_shapes() const
{
    return {{embedding_dimension},
            {embedding_dimension}};
}

void Normalization3d::set(const Index new_sequence_length,
                          Index new_embedding_dimension,
                          const string& new_label)
{
    sequence_length = new_sequence_length;
    embedding_dimension = new_embedding_dimension;

    label = new_label;
    name = "Normalization3d";
    layer_type = LayerType::Normalization3d;
}

void Normalization3d::set_parameters_random()
{
    VectorMap(parameters[Gamma].data, parameters[Gamma].size()).setOnes();

    VectorMap(parameters[Beta].data, parameters[Beta].size()).setZero();
}

void Normalization3d::set_parameters_glorot()
{
    set_parameters_random();
}


void Normalization3d::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    layernorm_forward(forward_views[Input][0], 
                      parameters[Gamma], 
                      parameters[Beta],
                      forward_views[Means][0], 
                      forward_views[StandardDeviations][0], 
                      forward_views[NormalizedInput][0], 
                      forward_views[Output][0],
                      forward_propagation.batch_size, 
                      sequence_length, 
                      embedding_dimension);
}


void Normalization3d::back_propagate(ForwardPropagation& forward_propagation,
                                     BackPropagation& back_propagation,
                                     size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    layernorm_backward(forward_views[Input][0], 
                       backward_views[OutputGradient][0],
                       forward_views[Means][0], 
                       forward_views[StandardDeviations][0], 
                       forward_views[NormalizedInput][0], 
                       parameters[Gamma],
                       gradient_views[Gamma],
                       gradient_views[Beta],
                       backward_views[InputGradient][0],
                       forward_propagation.batch_size, 
                       sequence_length, 
                       embedding_dimension);
}


void Normalization3d::from_XML(const XmlDocument& document)
{
    const XmlElement* element = get_xml_root(document, "Normalization3d");

    const string new_name = read_xml_string(element, "Label");
    const Index new_sequence_length = read_xml_index(element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(element, "EmbeddingDimension");

    set(new_sequence_length, new_embedding_dimension, new_name);
}

void Normalization3d::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Normalization3d");
    write_xml_properties(printer, {
        {"Label", label},
        {"SequenceLength", to_string(get_sequence_length())},
        {"EmbeddingDimension", to_string(get_embedding_dimension())}
    });
    printer.close_element();
}

REGISTER(Layer, Normalization3d, "Normalization3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
