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

EmbeddingLayer::EmbeddingLayer(const Index& new_vocabulary_size,
                               const Index& new_sequence_length,
                               const Index& new_embedding_size,
                               const bool& new_positional_encoding,
                               const string& new_name) : Layer()
{
    set(new_vocabulary_size, new_sequence_length, new_embedding_size, new_positional_encoding, new_name);

    layer_type = Type::Embedding;

    name = "embedding_layer";
}


Index EmbeddingLayer::get_vocabulary_size() const
{
    return embedding_weights.dimension(0);
}


Index EmbeddingLayer::get_sequence_length() const
{
    return sequence_length;
}


Index EmbeddingLayer::get_embedding_dimension() const
{
    return embedding_weights.dimension(1);
}


bool EmbeddingLayer::get_positional_encoding() const
{
    return positional_encoding;
}


dimensions EmbeddingLayer::get_input_dimensions() const
{
    return { sequence_length };
}


dimensions EmbeddingLayer::get_output_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
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


void EmbeddingLayer::set(const Index& new_vocabulary_size,
                         const Index& new_sequence_length,
                         const Index& new_embedding_size,
                         const bool& new_positional_encoding,
                         const string& new_name)
{
    sequence_length = new_sequence_length;

    embedding_weights.resize(new_vocabulary_size, new_embedding_size);

    set_parameters_random();

    positional_encoding = new_positional_encoding;

    name = "embedding_layer";

    layer_type = Type::Embedding;
}


void EmbeddingLayer::set_vocabulary_size(const Index& new_vocabulary_size)
{
    const Index embedding_dimension = get_embedding_dimension();

    embedding_weights.resize(new_vocabulary_size, embedding_dimension);

    set_parameters_random();
}


void EmbeddingLayer::set_sequence_length(const Index& new_sequence_length)
{
    sequence_length = new_sequence_length;
}


void EmbeddingLayer::set_embedding_size(const Index& new_embedding_size)
{
    const Index vocabulary_size = get_vocabulary_size();

    embedding_weights.resize(vocabulary_size, new_embedding_size);

    set_parameters_random();
}


void EmbeddingLayer::set_positional_encoding(const bool& new_positional_encoding)
{
    positional_encoding = new_positional_encoding;
}


void EmbeddingLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void EmbeddingLayer::set_embedding_weights()
{
    const Index vocabulary_size = get_vocabulary_size();
    const Index embedding_dimension = get_embedding_dimension();

    embedding_weights.resize(vocabulary_size, embedding_dimension);

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

    if(embedding_weights.dimension(0) != 0)
    {
        embedding_weights.chip(0, 0).setConstant(0);

        #pragma omp parallel for

        for(Index i = 1; i < embedding_weights.dimension(0); i++)
            for(Index j = 0; j < embedding_weights.dimension(1); j++)
                embedding_weights(i, j) = minimum + (maximum - minimum)* type(rand() / (RAND_MAX + 1.0));
    }
}


void EmbeddingLayer::set_parameters_constant(const type& value)
{
    embedding_weights.setConstant(value);
}


void EmbeddingLayer::dropout(Tensor<type, 3>& outputs) const
{
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    #pragma omp parallel for
    for(Index i = 0; i < outputs.size(); i++)
        outputs(i) = (get_random_type(type(0), type(1)) < dropout_rate)
            ? 0 
            : outputs(i) * scaling_factor;
}


void EmbeddingLayer::lookup_embedding(const Tensor<type, 2>& inputs, Tensor<type, 3>& outputs)
{
    const Index batch_size = inputs.dimension(0);

    // #pragma omp parallel for collapse(2)
     for(Index row = 0; row < batch_size; row++)
        for(Index input_position = 0; input_position < sequence_length; input_position++)
             outputs.chip(row, 0).chip(input_position, 0)
                = embedding_weights.chip(inputs(row, input_position), 0);

}


void EmbeddingLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                       const bool& is_training)
{
    const Index embedding_dimension = get_embedding_dimension();

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation =
        static_cast<EmbeddingLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = embedding_layer_forward_propagation->outputs;

    lookup_embedding(inputs, outputs);

    if(positional_encoding)
    {
        outputs.device(*thread_pool_device) = outputs * sqrt(type(embedding_dimension));

        const Tensor<type, 2>& positional_encoding = embedding_layer_forward_propagation->positional_encoding;
        
        for(Index batch_element = 0; batch_element < outputs.dimension(0); batch_element++)
            outputs.chip(batch_element, 0).device(*thread_pool_device) += positional_encoding;
    }

    if(dropout_rate > 0 && is_training)
        dropout(outputs);
}


void EmbeddingLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                    const vector<pair<type*, dimensions>>& delta_pairs,
                                    unique_ptr<LayerForwardPropagation>&,
                                    unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index embedding_dimension = get_embedding_dimension();

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
        positional_encoding
            ? sample_deltas.device(*thread_pool_device) = deltas.chip(i, 0) * sqrt(type(embedding_dimension))
            : sample_deltas.device(*thread_pool_device) = deltas.chip(i, 0);

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


void EmbeddingLayer::from_XML(const XMLDocument& document)
{
    const XMLElement* embedding_layer_element = document.FirstChildElement("Embedding");

    if(!embedding_layer_element)
        throw runtime_error("Embedding element is nullptr.\n");

    set_name(read_xml_string(embedding_layer_element, "Name"));
    set_vocabulary_size(read_xml_index(embedding_layer_element, "VocabularySize"));
    set_sequence_length(read_xml_index(embedding_layer_element, "SequenceLength"));
    set_embedding_size(read_xml_index(embedding_layer_element, "EmbeddingSize"));
    set_positional_encoding(read_xml_bool(embedding_layer_element, "PositionalEncoding"));
    set_parameters(to_type_vector(read_xml_string(embedding_layer_element, "Parameters"), " "));
}


void EmbeddingLayer::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Embedding");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "VocabularySize", to_string(get_vocabulary_size()));
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingSize", to_string(get_embedding_dimension()));
    add_xml_element(printer, "PositionalEncoding", to_string(positional_encoding ? 1 : 0));
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();  
}


EmbeddingLayerForwardPropagation::EmbeddingLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> EmbeddingLayerForwardPropagation::get_outputs_pair() const
{
    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(layer);

    const Index sequence_length = embedding_layer->get_sequence_length();

    const Index embedding_size = embedding_layer->get_embedding_dimension();
    
    return {(type*)outputs.data(), {batch_samples_number, sequence_length, embedding_size}};
}


void EmbeddingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(new_layer);

    batch_samples_number = new_batch_samples_number;

    const Index sequence_length = embedding_layer->get_sequence_length();

    const Index embedding_size = embedding_layer->get_embedding_dimension();

    // Outputs

    outputs.resize(batch_samples_number, sequence_length, embedding_size);

    if(embedding_layer->get_positional_encoding())
        build_positional_encoding_matrix();
}


void EmbeddingLayerForwardPropagation::print() const
{
    cout << "Attention scores:" << endl;
    //       cout << attention_scores.dimensions() << endl;
    cout << "Outputs dimensions:" << endl;
    //       cout << output_dimensions << endl;
    cout << "Outputs:" << endl;
    //       cout << TensorMap<Tensor<type,3>>(outputs_data, output_dimensions(0), output_dimensions(1), output_dimensions(2)) << endl;
    cout << "Attention scores:" << endl;
    //       cout << attention_scores << endl;
}


void EmbeddingLayerForwardPropagation::build_positional_encoding_matrix()
{
    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(layer);

    const Index inputs_number = embedding_layer->get_sequence_length();
    const Index embedding_size = embedding_layer->get_embedding_dimension();

    positional_encoding.resize(inputs_number, embedding_size);

    positional_encoding.setZero();

    const type half_depth = type(embedding_size) / 2;

    #pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        for(Index j = 0; j < Index(embedding_size); j++)
            positional_encoding(i, j) = (j < Index(half_depth))
                ? sin(i / pow(10000, j / half_depth))
                : cos(i / pow(10000, (j - Index(half_depth)) / half_depth));

    built_positional_encoding_matrix = true;
}


EmbeddingLayerBackPropagation::EmbeddingLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> EmbeddingLayerBackPropagation::get_input_derivative_pairs() const
{
    return vector<pair<type*, dimensions>>();
}


void EmbeddingLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const EmbeddingLayer* embedding_layer = static_cast<EmbeddingLayer*>(new_layer);

    batch_samples_number = new_batch_samples_number;

    const Index sequence_length = embedding_layer->get_sequence_length();
    const Index embedding_size = embedding_layer->get_embedding_dimension();
    const Index vocabulary_size = embedding_layer->get_vocabulary_size();

    sample_deltas.resize(sequence_length, embedding_size);
    embedding_weights_derivatives.resize(vocabulary_size, embedding_size);
}


void EmbeddingLayerBackPropagation::print() const
{
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
