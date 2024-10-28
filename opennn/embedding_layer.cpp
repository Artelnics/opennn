//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "strings_utilities.h"
#include "embedding_layer.h"

namespace opennn
{

EmbeddingLayer::EmbeddingLayer() : Layer()
{
    set();

    layer_type = Type::Embedding;
}


EmbeddingLayer::EmbeddingLayer(const Index& new_inputs_dimension,
                               const Index& new_inputs_number,
                               const Index& new_depth,
                               const bool& new_positional_encoding) : Layer()
{
    set(new_inputs_dimension, new_inputs_number, new_depth, new_positional_encoding);

    layer_type = Type::Embedding;

    name = "embedding_layer";
}


Index EmbeddingLayer::get_input_dimension() const
{
    return input_dimensions;
}


Index EmbeddingLayer::get_inputs_number() const
{
    return inputs_number;
}


Index EmbeddingLayer::get_depth() const
{
    return depth;
}


bool EmbeddingLayer::get_positional_encoding() const
{
    return positional_encoding;
}


dimensions EmbeddingLayer::get_input_dimensions() const
{
    return {};
}


dimensions EmbeddingLayer::get_output_dimensions() const
{
    return { inputs_number, depth };
}


Index EmbeddingLayer::get_parameters_number() const
{
    return embedding_weights.size();
}


Tensor<type, 1> EmbeddingLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    memcpy(parameters.data(), embedding_weights.data(), embedding_weights.size()*sizeof(type));

    return parameters;
}


Index EmbeddingLayer::get_neurons_number() const
{
    return inputs_number * depth;
}


const bool& EmbeddingLayer::get_display() const
{
    return display;
}


void EmbeddingLayer::set()
{
    input_dimensions = 0;

    inputs_number = 0;

    depth = 0;

    positional_encoding = false;

    embedding_weights.resize(0, 0);

    set_default();
}


void EmbeddingLayer::set(const Index& new_inputs_dimension,
                         const Index& new_inputs_number,
                         const Index& new_depth,
                         const bool& new_positional_encoding)
{
    input_dimensions = new_inputs_dimension;

    inputs_number = new_inputs_number;

    depth = new_depth;

    set_embedding_weights();

    positional_encoding = new_positional_encoding;

    set_default();
}


void EmbeddingLayer::set_default()
{
    name = "embedding_layer";

    display = true;

    layer_type = Type::Embedding;
}


void EmbeddingLayer::set_input_dimensions(const Index& new_inputs_dimension)
{
    input_dimensions = new_inputs_dimension;

    set_embedding_weights();
}


void EmbeddingLayer::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


void EmbeddingLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_embedding_weights();
}


void EmbeddingLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void EmbeddingLayer::set_embedding_weights()
{
    embedding_weights.resize(input_dimensions, depth);

    set_parameters_random();
}


void EmbeddingLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    memcpy(embedding_weights.data(), new_parameters.data() + index, embedding_weights.size()*sizeof(type));
}


void EmbeddingLayer::set_parameters_random()
{
    const type minimum = type(-0.05);
    const type maximum = type(0.05);

    // First row must be 0s because input value 0 is padding
    
    embedding_weights.chip(0, 0).setConstant(0);
    
    #pragma omp parallel for

    for(Index i = 1; i < embedding_weights.dimension(0); i++)
        for(Index j = 0; j < embedding_weights.dimension(1); j++)
            embedding_weights(i, j) = minimum + (maximum - minimum)* type(rand() / (RAND_MAX + 1.0));
}


void EmbeddingLayer::set_parameters_constant(const type& value)
{
    embedding_weights.setConstant(value);
}


void EmbeddingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void EmbeddingLayer::dropout(Tensor<type, 3>& outputs) const
{
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    #pragma omp parallel for
    for(Index i = 0; i < outputs.size(); i++)
        outputs(i) = (calculate_random_uniform(type(0), type(1)) < dropout_rate)
            ? 0 
            : outputs(i) * scaling_factor;
}


void EmbeddingLayer::lookup_embedding(const Tensor<type, 2>& inputs, Tensor<type, 3>& outputs)
{
    const Index batch_size = inputs.dimension(0);

    #pragma omp parallel for
    for(Index row = 0; row < batch_size; row++)
        for(Index input_position = 0; input_position < inputs_number; input_position++)
            outputs.chip(row, 0).chip(input_position, 0)
                = embedding_weights.chip(inputs(row, input_position), 0);
}


void EmbeddingLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                       const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation =
        static_cast<EmbeddingLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = embedding_layer_forward_propagation->outputs;

    lookup_embedding(inputs, outputs);

    if(positional_encoding)
    {
        outputs.device(*thread_pool_device) = outputs * outputs.constant(sqrt(depth));

        const Tensor<type, 2>& positional_encoding = embedding_layer_forward_propagation->positional_encoding;
        
        for(Index batch_element = 0; batch_element < outputs.dimension(0); batch_element++)
            outputs.chip(batch_element, 0).device(*thread_pool_device) += positional_encoding;
    }

    if(dropout_rate > 0 && is_training)
        dropout(outputs);
}


void EmbeddingLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                    const vector<pair<type*, dimensions>>& delta_pairs,
                                    unique_ptr<LayerForwardPropagation>& forward_propagation,
                                    unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_samples_number = input_pairs[0].second[0];
    const Index inputs_number = input_pairs[0].second[1];

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    if(delta_pairs.size() > 1)     
        add_deltas(delta_pairs);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Back propagation

    EmbeddingLayerBackPropagation* embedding_layer_back_propagation =
        static_cast<EmbeddingLayerBackPropagation*>(back_propagation.get());

    Tensor<type, 2>& sample_deltas = embedding_layer_back_propagation->sample_deltas;
    Tensor<type, 2>& embedding_weights_derivatives = embedding_layer_back_propagation->embedding_weights_derivatives;
    
    embedding_weights_derivatives.setZero();
    
    for(Index i = 0; i < batch_samples_number; i++)
    {
        if(positional_encoding)
            sample_deltas.device(*thread_pool_device) 
                = deltas.chip(i, 0) * sample_deltas.constant(sqrt(depth));
        else
            sample_deltas.device(*thread_pool_device) = deltas.chip(i, 0);

        for(Index j = 0; j < inputs_number; j++)
            embedding_weights_derivatives.chip(Index(inputs(i, j)), 0).device(*thread_pool_device)
                += sample_deltas.chip(j, 0);
    }
}


void EmbeddingLayer::add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
{
    TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    for(Index i = 1; i < Index(delta_pairs.size()); i++)
        deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
}


void EmbeddingLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                     const Index& index,
                                     Tensor<type, 1>& gradient) const
{
    const Index embedding_weights_number = get_parameters_number();

    const EmbeddingLayerBackPropagation* embedding_layer_back_propagation =
        static_cast<EmbeddingLayerBackPropagation*>(back_propagation.get());

    const type* embedding_weights_derivatives_data = embedding_layer_back_propagation->embedding_weights_derivatives.data();

    type* gradient_data = gradient.data();

    memcpy(gradient_data + index, embedding_weights_derivatives_data, embedding_weights_number*sizeof(type));
}


void EmbeddingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    // Embedding layer

    const tinyxml2::XMLElement* embedding_layer_element = document.FirstChildElement("EmbeddingLayer");

    if(!embedding_layer_element)
        throw runtime_error("EmbeddingLayer element is nullptr.\n");

    set_name(read_xml_string(embedding_layer_element, "Name"));
    set_input_dimensions(read_xml_index(embedding_layer_element, "InputDimensions"));
    set_inputs_number(read_xml_index(embedding_layer_element, "InputsNumber"));
    set_depth(read_xml_index(embedding_layer_element, "Depth"));

    positional_encoding = read_xml_bool(embedding_layer_element, "PositionalEncoding");

    const tinyxml2::XMLElement* parameters_element = embedding_layer_element->FirstChildElement("Parameters");

    if (!parameters_element)
        throw std::runtime_error("Parameters element is nullptr.\n");

    if (parameters_element->GetText())
        set_parameters(to_type_vector(parameters_element->GetText(), " "));
}


void EmbeddingLayer::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("EmbeddingLayer");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(get_input_dimensions()));
    add_xml_element(printer, "InputsNumber", to_string(get_inputs_number()));
    add_xml_element(printer, "Depth", to_string(get_depth()));
    add_xml_element(printer, "PositionalEncoding", to_string(positional_encoding ? 1 : 0));
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();  
}


pair<type*, dimensions> EmbeddingLayerForwardPropagation::get_outputs_pair() const
{
    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(layer);

    const Index inputs_number = embedding_layer->get_inputs_number();

    const Index depth = embedding_layer->get_depth();
    
    return {outputs_data, {batch_samples_number, inputs_number, depth}};
}


void EmbeddingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(new_layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = embedding_layer->get_inputs_number();

    const Index depth = embedding_layer->get_depth();

    // Outputs

    outputs.resize(batch_samples_number, inputs_number, depth);

    outputs_data = outputs.data();

    if(embedding_layer->get_positional_encoding())
        build_positional_encoding_matrix();
}


void EmbeddingLayerForwardPropagation::build_positional_encoding_matrix()
{
    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(layer);

    const Index inputs_number = embedding_layer->get_inputs_number();
    const Index depth = embedding_layer->get_depth();

    positional_encoding.resize(inputs_number, depth);

    positional_encoding.setZero();

    const type half_depth = type(depth) / 2;

    #pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        for(Index j = 0; j < Index(depth); j++)
            positional_encoding(i, j) = (j < Index(half_depth))
                ? sin(i / pow(10000, j / half_depth))
                : cos(i / pow(10000, (j - Index(half_depth)) / half_depth));

    built_positional_encoding_matrix = true;
}


void EmbeddingLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(new_layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = embedding_layer->get_inputs_number();
    const Index depth = embedding_layer->get_depth();
    const Index input_dimension = embedding_layer->get_input_dimension();

    sample_deltas.resize(inputs_number, depth);
    embedding_weights_derivatives.resize(input_dimension, depth);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
