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

Embedding::Embedding(const Index& new_vocabulary_size,
                     const Index& new_sequence_length,
                     const Index& new_embedding_dimension,
                     const string& new_name) : Layer()
{
    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_name);

    layer_type = Type::Embedding;

    name = new_name;
}


Index Embedding::get_vocabulary_size() const
{
    return weights.dimension(0);
}


Index Embedding::get_sequence_length() const
{
    return sequence_length;
}


Index Embedding::get_embedding_dimension() const
{
    return weights.dimension(1);
}


dimensions Embedding::get_input_dimensions() const
{
    return { sequence_length };
}


dimensions Embedding::get_output_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


Index Embedding::get_parameters_number() const
{
    return weights.size();
}


Tensor<type, 1> Embedding::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    Index index = 0;

    copy_to_vector(parameters, weights, index);

    return parameters;
}


void Embedding::set(const Index& new_vocabulary_size,
                    const Index& new_sequence_length,
                    const Index& new_embedding_dimension,
                    const string& new_name)
{
    sequence_length = new_sequence_length;

    weights.resize(new_vocabulary_size, new_embedding_dimension);

    set_parameters_random();

    positional_encoding.resize(sequence_length, new_embedding_dimension);

    positional_encoding.setZero();

    const type half_depth = type(new_embedding_dimension)/2;

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < sequence_length; i++)
        for (Index j = 0; j < new_embedding_dimension; j++)
            positional_encoding(i, j) = (j < Index(half_depth))
            ? sin(i / pow(10000, j / half_depth))
            : cos(i / pow(10000, (j - Index(half_depth)) / half_depth));

    name = "embedding_layer";
}


void Embedding::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void Embedding::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(weights, new_parameters, index);
}


void Embedding::set_parameters_random()
{
    if(weights.size() == 0) return;

    const type minimum = type(-0.05);
    const type maximum = type(0.05);

    // First row must be 0s because input value 0 is padding

    weights.chip(0, 0).setConstant(0);

    #pragma omp parallel for

    for (Index i = 1; i < weights.dimension(0); i++)
        for (Index j = 0; j < weights.dimension(1); j++)
            weights(i, j) = get_random_type(minimum, maximum);
}


void Embedding::set_parameters_constant(const type& value)
{
    weights.setConstant(value);
}


void Embedding::dropout(Tensor<type, 3>& outputs) const
{
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    outputs.device(*thread_pool_device) = outputs.unaryExpr([this, scaling_factor](const type& value) {
        return get_random_type(type(0), type(1)) < dropout_rate 
            ? type(0) 
            : value * scaling_factor;
        });
/*
    #pragma omp parallel for

    for(Index i = 0; i < outputs.size(); i++)
        outputs(i) = get_random_type(type(0), type(1)) < dropout_rate
            ? 0 
            : outputs(i) * scaling_factor;
*/
}


void Embedding::embedding_lookup(const Tensor<type, 2>& inputs, Tensor<type, 3>& outputs)
{
    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index embedding_dimension = outputs.dimension(2);

    outputs.setConstant(sqrt(type(embedding_dimension)));

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++)
    {
        for (int t = 0; t < sequence_length; ++t)
        {
            const int token_id = inputs(b, t);

            for (int d = 0; d < embedding_dimension; d++)
                outputs(b, t, d) *= weights(token_id, d);
        }
    }

    //outputs.device(*thread_pool_device) = outputs * sqrt(type(embedding_dimension));
}


void Embedding::add_positional_encodings(Tensor<type, 3>& embeddings) const
{ 
    const int batch_size = embeddings.dimension(0);
    const int sequence_length = embeddings.dimension(1);
    const int embedding_dimension = embeddings.dimension(2);

    const Eigen::array<int, 3> broadcast_dimensions = { batch_size, 1, 1 };

    const Eigen::array<int, 3> reshape_dimensions = { 1, sequence_length, embedding_dimension};

    embeddings.device(*thread_pool_device) 
        += positional_encoding.reshape(reshape_dimensions).broadcast(broadcast_dimensions);

    //for (Index sample_index = 0; sample_index < batch_size; sample_index++)
    //    embeddings.chip(sample_index, 0).device(*thread_pool_device) += positional_encoding;
}


void Embedding::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                       const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    EmbeddingForwardPropagation* embedding_forward_propagation =
        static_cast<EmbeddingForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = embedding_forward_propagation->outputs;

    embedding_lookup(inputs, outputs);
        
    add_positional_encodings(outputs);

    if(dropout_rate > 0 && is_training)
        dropout(outputs);
}


void Embedding::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                               const vector<pair<type*, dimensions>>& delta_pairs,
                               unique_ptr<LayerForwardPropagation>&,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index embedding_dimension = get_embedding_dimension();

    const Index samples_number = input_pairs[0].second[0];
    const Index inputs_number = input_pairs[0].second[1];

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Back propagation

    EmbeddingBackPropagation* embedding_back_propagation =
        static_cast<EmbeddingBackPropagation*>(back_propagation.get());

    Tensor<type, 2>& sample_deltas = embedding_back_propagation->sample_deltas;
    Tensor<type, 2>& weight_derivatives = embedding_back_propagation->weight_derivatives;

    if (delta_pairs.size() > 1)
        add_deltas(delta_pairs);

    weight_derivatives.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        sample_deltas.device(*thread_pool_device) = deltas.chip(i, 0) * sqrt(type(embedding_dimension));

        for(Index j = 0; j < inputs_number; j++)
            weight_derivatives.chip(Index(inputs(i, j)), 0).device(*thread_pool_device)
                += sample_deltas.chip(j, 0);
    }
}


void Embedding::add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
{
    TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    for(Index i = 1; i < Index(delta_pairs.size()); i++)
        deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
}


void Embedding::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                     Index& index,
                                     Tensor<type, 1>& gradient) const
{
    const EmbeddingBackPropagation* embedding_back_propagation =
        static_cast<EmbeddingBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, embedding_back_propagation->weight_derivatives, index);
}


void Embedding::from_XML(const XMLDocument& document)
{
    const XMLElement* embedding_layer_element = document.FirstChildElement("Embedding");

    if(!embedding_layer_element)
        throw runtime_error("Embedding element is nullptr.\n");

    const string new_name = read_xml_string(embedding_layer_element, "Name");
    const Index new_vocabulary_size = read_xml_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_xml_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_name);

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(embedding_layer_element, "Parameters"), " "), index);
}


void Embedding::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Embedding");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "VocabularySize", to_string(get_vocabulary_size()));
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingSize", to_string(get_embedding_dimension()));
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();  
}


EmbeddingForwardPropagation::EmbeddingForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> EmbeddingForwardPropagation::get_outputs_pair() const
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);

    const Index sequence_length = embedding_layer->get_sequence_length();

    const Index embedding_dimension = embedding_layer->get_embedding_dimension();
    
    return {(type*)outputs.data(), {batch_size, sequence_length, embedding_dimension}};
}


void EmbeddingForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    const Embedding* embedding_layer = static_cast<Embedding*>(new_layer);

    batch_size = new_batch_size;

    const Index sequence_length = embedding_layer->get_sequence_length();

    const Index embedding_dimension = embedding_layer->get_embedding_dimension();

    // Outputs

    outputs.resize(batch_size, sequence_length, embedding_dimension);
}


void EmbeddingForwardPropagation::print() const
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


EmbeddingBackPropagation::EmbeddingBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> EmbeddingBackPropagation::get_input_derivative_pairs() const
{
    return vector<pair<type*, dimensions>>();
}


void EmbeddingBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    const Embedding* embedding_layer = static_cast<Embedding*>(new_layer);

    batch_size = new_batch_size;

    const Index sequence_length = embedding_layer->get_sequence_length();
    const Index embedding_dimension = embedding_layer->get_embedding_dimension();
    const Index vocabulary_size = embedding_layer->get_vocabulary_size();

    sample_deltas.resize(sequence_length, embedding_dimension);
    weight_derivatives.resize(vocabulary_size, embedding_dimension);
}


void EmbeddingBackPropagation::print() const
{
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
