//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "addition_layer_3d.h"

namespace opennn
{

Addition3d::Addition3d(const Index& new_inputs_number, 
                       const Index& new_inputs_depth, 
                       const string& new_name) : Layer()
{
    layer_type = Type::Addition3D;

    set(new_inputs_number, new_inputs_depth, new_name);
}


Index Addition3d::get_sequence_length() const
{
    return sequence_length;
}


Index Addition3d::get_embedding_length() const
{
    return embedding_length;
}


dimensions Addition3d::get_input_dimensions() const
{
    return { sequence_length, embedding_length };
}


dimensions Addition3d::get_output_dimensions() const
{
    return { sequence_length, embedding_length };
}


void Addition3d::set(const Index& new_sequence_length, 
                     const Index& new_embedding_length, 
                     const string& new_name)
{
    sequence_length = new_sequence_length;

    embedding_length = new_embedding_length;

    name = new_name;
}


void Addition3d::set_sequence_length(const Index& new_sequence_length)
{
    sequence_length = new_sequence_length;
}


void Addition3d::set_embedding_length(const Index& new_embedding_length)
{
    embedding_length = new_embedding_length;
}


void Addition3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                   unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                   const bool&)
{
    const TensorMap<Tensor<type, 3>> positional_encodings = tensor_map_3(input_pairs[0]);

    // cout << "Is this part? (Addition layer)" << endl;

    // print_vector(input_pairs[1].second);

    const TensorMap<Tensor<type, 3>> input_embeddings = tensor_map_3(input_pairs[1]);

    // cout << "Is it this one? (Addition layer)" << endl;

    Addition3dForwardPropagation* addition_3d_forward_propagation =
        static_cast<Addition3dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = addition_3d_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = positional_encodings + input_embeddings;
}


void Addition3d::back_propagate(const vector<pair<type*, dimensions>>&,
                                const vector<pair<type*, dimensions>>& delta_pairs,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Back propagation

    Addition3dBackPropagation* addition_3d_back_propagation =
        static_cast<Addition3dBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& input_1_derivatives = addition_3d_back_propagation->input_1_derivatives;

    Tensor<type, 3>& input_2_derivatives = addition_3d_back_propagation->input_2_derivatives;

    input_1_derivatives.device(*thread_pool_device) = deltas;
    input_2_derivatives.device(*thread_pool_device) = deltas;
}


void Addition3d::from_XML(const XMLDocument& document)
{
    const auto* addition_layer_element = document.FirstChildElement("Addition3D");

    if (!addition_layer_element)
        throw runtime_error("Addition3D element is nullptr.\n");

    const string new_name = read_xml_string(addition_layer_element, "Name");
    const Index new_sequence_lenght = read_xml_index(addition_layer_element, "SequenceLength");
    const Index new_embedding_legth = read_xml_index(addition_layer_element, "EmbeddingLength");

    set(new_sequence_lenght, new_embedding_legth, new_name);
}


void Addition3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Addition3D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingLength", to_string(get_embedding_length()));

    printer.CloseElement();
}


Addition3dForwardPropagation::Addition3dForwardPropagation(const Index& new_batch_size, 
                                                                     Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Addition3dForwardPropagation::get_outputs_pair() const
{
    Addition3d* addition_layer_3d = static_cast<Addition3d*>(layer);

    const Index sequence_length = addition_layer_3d->get_sequence_length();
    const Index embedding_length = addition_layer_3d->get_embedding_length();

    return {(type*)outputs.data(), {batch_size, sequence_length, embedding_length}};
}


void Addition3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Addition3d* addition_layer_3d = static_cast<Addition3d*>(layer);

    batch_size = new_batch_size;

    const Index sequence_length = addition_layer_3d->get_sequence_length();
    const Index embedding_length = addition_layer_3d->get_embedding_length();

    outputs.resize(batch_size, sequence_length, embedding_length);
}


void Addition3dForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}


void Addition3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Addition3d* addition_layer_3d = static_cast<Addition3d*>(layer);

    batch_size = new_batch_size;

    const Index sequence_length = addition_layer_3d->get_sequence_length();
    const Index embedding_length = addition_layer_3d->get_embedding_length();

    input_1_derivatives.resize(batch_size, sequence_length, embedding_length);
    input_2_derivatives.resize(batch_size, sequence_length, embedding_length);
}


void Addition3dBackPropagation::print() const
{
}


Addition3dBackPropagation::Addition3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> Addition3dBackPropagation::get_input_derivative_pairs() const
{
    Addition3d* addition_layer_3d = static_cast<Addition3d*>(layer);

    const Index sequence_length = addition_layer_3d->get_sequence_length();
    const Index embedding_length = addition_layer_3d->get_embedding_length();

    return
    {{(type*)input_1_derivatives.data(), {batch_size, sequence_length, embedding_length}},
     {(type*)input_2_derivatives.data(), {batch_size, sequence_length, embedding_length}}};
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
