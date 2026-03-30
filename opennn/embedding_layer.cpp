//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "embedding_layer.h"
#include "neural_network.h"

namespace opennn
{

Embedding::Embedding(const Shape& new_input_shape,
                     Index new_embedding_dimension,
                     const string& new_label) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_embedding_dimension, new_label);

    name = "Embedding";
}


Index Embedding::get_vocabulary_size() const
{
    return vocabulary_size;
}


Index Embedding::get_sequence_length() const
{
    return sequence_length;
}


Index Embedding::get_embedding_dimension() const
{
    return embedding_dimension;
}


Shape Embedding::get_input_shape() const
{
    return { sequence_length };
}


Shape Embedding::get_output_shape() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


vector<Shape> Embedding::get_parameter_shapes() const
{    
    return {{vocabulary_size, embedding_dimension}}; // weights
}


void Embedding::set(const Index new_vocabulary_size,
                    Index new_sequence_length,
                    Index new_embedding_dimension,
                    const string& new_label)
{
    parameters.resize(1);

    sequence_length = new_sequence_length;
    label = new_label;

    positional_encoding.resize(sequence_length, new_embedding_dimension);
    positional_encoding.setZero();

    const type half_depth = type(new_embedding_dimension)/2;

#pragma omp parallel for collapse(2)

    for(Index i = 0; i < sequence_length; i++)
        for(Index j = 0; j < new_embedding_dimension; j++)
            positional_encoding(i, j) = (j < Index(half_depth))
                ? sin(i / pow(10000, j / half_depth))
                : cos(i / pow(10000, (j - Index(half_depth)) / half_depth));

#ifdef CUDA

    weights_device.set_descriptor({new_vocabulary_size, new_embedding_dimension});

    positional_encoding_device.resize({sequence_length, new_embedding_dimension});

#endif
}

void Embedding::set_scale_embedding(bool new_scale_embedding)
{
    scale_embedding = new_scale_embedding;
}


void Embedding::set_add_positional_encoding(bool new_add_positional_encoding)
{
    add_positional_encoding = new_add_positional_encoding;
}


void Embedding::set_dropout_rate(const type new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void Embedding::set_parameters_random()
{
    if(parameters[Weights].empty()) return;

    const type scale = type(0.05);

    MatrixMap weights = matrix_map(parameters[Weights]);

    weights.setRandom();
    weights *= scale;

    weights.row(0).setZero();
}


void Embedding::set_parameters_glorot()
{
    if(parameters[Weights].empty()) return;

//    const Index vocabulary_size = weights.shape[0];
//    const Index embedding_dimension = weights.shape[1];

    const type limit = sqrt(type(6.0) / (vocabulary_size + embedding_dimension));

    MatrixMap weights = matrix_map(parameters[Weights]);

    weights.setRandom();
    weights *= limit;

    weights.row(0).setZero();
}


void Embedding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
#ifndef CUDA

    const Index batch_size = forward_propagation.batch_size;
    const Index total_tokens = batch_size * sequence_length;

    MatrixMap outputs = matrix_map(forward_propagation.views[layer][Outputs][0]);
    outputs.setZero();

    const MatrixMap weights = matrix_map(parameters[Weights]);

    const type* input_indices = forward_propagation.views[layer][Inputs][0].data;

    #pragma omp parallel for
    for(Index i = 0; i < total_tokens; i++)
    {
        const Index token_id = static_cast<Index>(input_indices[i]);

        if(token_id < 0 || token_id >= weights.rows())
        {
            outputs.row(i).setZero();
            continue;
        }

        outputs.row(i).noalias() = weights.row(token_id);
    }

    if(scale_embedding)
        outputs *= sqrt(static_cast<type>(embedding_dimension));

    if(add_positional_encoding)
    {
        #pragma omp parallel for
        for(Index b = 0; b < batch_size; b++)
            for(Index s = 0; s < sequence_length; s++)
                if (static_cast<Index>(input_indices[b * sequence_length + s]) > 0)
                    outputs.row(b * sequence_length + s) += positional_encoding.row(s);
    }

    //if(is_training && dropout_rate > 0)
    //    dropout(outputs, dropout_rate);

#else
    const Index batch_size = forward_propagation->batch_size;
    const Index sequence_length = this->sequence_length;
    const Index embedding_dimension = get_embedding_dimension();
    const Index vocabulary_size = get_vocabulary_size();

    const Index total_elements = batch_size * sequence_length * embedding_dimension;

    TensorView& outputs = forward_propagation->outputs;

    const float* inputs_data = forward_propagation->inputs[0].data;
    const float* weights_data = weights_device.data;

    if (add_positional_encoding && !pos_encoding_synced)
    {
        this->copy_positional_encoding_device();
        pos_encoding_synced = true;
    }

    const float* positional_encoding_data = add_positional_encoding ? positional_encoding_device.data : nullptr;

    float* outputs_ptr = outputs.data;

    embedding_forward_cuda(
        total_elements,
        inputs_data,
        weights_data,
        positional_encoding_data,
        outputs_ptr,
        sequence_length,
        embedding_dimension,
        vocabulary_size,
        scale_embedding,
        add_positional_encoding
        );
#endif
}


void Embedding::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t index) const
{
/*
    if(back_propagation->output_gradients.size() > 1)
        add_gradients(back_propagation->output_gradients);
*/
#ifndef CUDA
/*

    const Index embedding_dimension = get_embedding_dimension();
    const Index batch_size = forward_propagation->inputs[0].shape[0];
    const Index sequence_length = forward_propagation->inputs[0].shape[1];
    const Index total_elements = batch_size * sequence_length;

    const type* input_indices = forward_propagation->inputs[0].data;

    if(back_propagation->output_gradients.size() > 1)
        add_gradients(back_propagation->output_gradients);

    MatrixMap gradients_map(back_propagation->output_gradients[0].data, total_elements, embedding_dimension);

    if(scale_embedding)
        gradients_map *= sqrt(static_cast<type>(embedding_dimension));

    EmbeddingBackPropagation* embedding_back_propagation = static_cast<EmbeddingBackPropagation*>(back_propagation.get());
    MatrixMap weight_gradients = matrix_map(embedding_back_propagation->weight_gradients);
    weight_gradients.setZero();

    for(Index i = 0; i < total_elements; i++)
    {
        const Index vocabulary_index = static_cast<Index>(input_indices[i]);

        if(vocabulary_index < 0 || vocabulary_index >= weight_gradients.rows())
            continue;

        weight_gradients.row(vocabulary_index).noalias() += gradients_map.row(i);
    }

    weight_gradients.row(0).setZero();
*/
#else
    const Index batch_size = forward_propagation->batch_size;
    const Index sequence_length = this->sequence_length;
    const Index embedding_dimension = get_embedding_dimension();
    const Index vocabulary_size = get_vocabulary_size();
    const Index total_elements = batch_size * sequence_length * embedding_dimension;

    EmbeddingBackPropagationCuda* embedding_back_propagation = static_cast<EmbeddingBackPropagationCuda*>(back_propagation.get());

    float* weight_gradients_data = embedding_back_propagation->weight_gradients.data;

    CHECK_CUDA(cudaMemset(weight_gradients_data, 0, vocabulary_size * embedding_dimension * sizeof(float)));

    const float* inputs_data = forward_propagation->inputs[0].data;
    const float* output_gradients_data = back_propagation->output_gradients[0].data;

    embedding_backward_cuda(
        total_elements,
        inputs_data,
        output_gradients_data,
        weight_gradients_data,
        sequence_length,
        embedding_dimension,
        vocabulary_size,
        scale_embedding
        );
#endif
}


void Embedding::print() const
{    
    cout << "Embedding Layer" << endl
         << "Label: " << label << endl
         << "Type: Embedding" << endl
         << "Input shape: " << get_input_shape() << endl
         << "Output shape: " << get_output_shape() << endl
         << "Vocabulary size: " << get_vocabulary_size() << endl
         << "Sequence length: " << get_sequence_length() << endl
         << "Embedding dimension: " << get_embedding_dimension() << endl
         << "Dropout rate: " << dropout_rate << endl;
         //<< "Weights shape: " << weights.shape << endl;

    //cout << "Weights:\n " << weights.shape << endl;
    cout << "Positional encoding:\n" << positional_encoding << endl;
}


void Embedding::from_XML(const XMLDocument& document)
{
    const XMLElement* embedding_layer_element = document.FirstChildElement("Embedding");

    if(!embedding_layer_element)
        throw runtime_error("Embedding element is nullptr.\n");

    const string new_label = read_xml_string(embedding_layer_element, "Label");
    const Index new_vocabulary_size = read_xml_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_xml_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_label);
}


void Embedding::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Embedding");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "VocabularySize", to_string(get_vocabulary_size()));
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingSize", to_string(get_embedding_dimension()));

    printer.CloseElement();
}


#ifdef CUDA


void Embedding::forward_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation, bool)
{
    const Index batch_size = forward_propagation->batch_size;
    const Index sequence_length = this->sequence_length;
    const Index embedding_dimension = get_embedding_dimension();
    const Index vocabulary_size = get_vocabulary_size();

    const Index total_elements = batch_size * sequence_length * embedding_dimension;

    TensorViewCuda& outputs = forward_propagation->outputs;

    const float* inputs_data = forward_propagation->inputs[0].data;
    const float* weights_data = weights_device.data;

    if (add_positional_encoding && !pos_encoding_synced)
    {
        this->copy_positional_encoding_device();
        pos_encoding_synced = true;
    }

    const float* positional_encoding_data = add_positional_encoding ? positional_encoding_device.data : nullptr;

    float* outputs_ptr = outputs.data;

    embedding_forward_cuda(
        total_elements,
        inputs_data,
        weights_data,
        positional_encoding_data,
        outputs_ptr,
        sequence_length,
        embedding_dimension,
        vocabulary_size,
        scale_embedding,
        add_positional_encoding
        );
}


void Embedding::back_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                               unique_ptr<LayerBackPropagationCuda>& back_propagation) const
{
    const Index batch_size = forward_propagation->batch_size;
    const Index sequence_length = this->sequence_length;
    const Index embedding_dimension = get_embedding_dimension();
    const Index vocabulary_size = get_vocabulary_size();
    const Index total_elements = batch_size * sequence_length * embedding_dimension;

    EmbeddingBackPropagationCuda* embedding_back_propagation = static_cast<EmbeddingBackPropagationCuda*>(back_propagation.get());

    float* weight_gradients_data = embedding_back_propagation->weight_gradients.data;

    const float* inputs_data = forward_propagation->inputs[0].data;
    const float* output_gradients_data = back_propagation->output_gradients[0].data;

    embedding_backward_cuda(
        total_elements,
        inputs_data,
        output_gradients_data,
        weight_gradients_data,
        sequence_length,
        embedding_dimension,
        vocabulary_size,
        scale_embedding
        );
}


vector<TensorViewCuda*> Embedding::get_parameter_views_device()
{
    return {&weights_device};
}




void Embedding::copy_positional_encoding_device()
{
    if (positional_encoding.size() > 0)
    {
        CHECK_CUDA(cudaMemcpy(positional_encoding_device.data,
                              positional_encoding.data(),
                              positional_encoding.size() * sizeof(type),
                              cudaMemcpyHostToDevice));
    }
}


void EmbeddingForwardPropagationCuda::initialize()
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);

    const Index sequence_length = embedding_layer->get_sequence_length();
    const Index embedding_dimension = embedding_layer->get_embedding_dimension();

    outputs.set_descriptor({batch_size, sequence_length, embedding_dimension});
}


void EmbeddingForwardPropagationCuda::print() const
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);

    const Index sequence_length = embedding_layer->get_sequence_length();
    const Index embedding_dimension = embedding_layer->get_embedding_dimension();

    cout << "Embedding layer forward propagation CUDA" << endl;
    cout << "Batch size: " << batch_size << endl;
    cout << "Outputs dimensions: " << batch_size << "x" << sequence_length << "x" << embedding_dimension << endl;
}


void EmbeddingBackPropagationCuda::initialize()
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);

    const Index embedding_dimension = embedding_layer->get_embedding_dimension();
    const Index vocabulary_size = embedding_layer->get_vocabulary_size();

    weight_gradients.set_descriptor({vocabulary_size, embedding_dimension});
}


vector<TensorView*> EmbeddingBackPropagationCuda::get_gradient_views()
{
    return {&weight_gradients};
}


void EmbeddingBackPropagationCuda::print() const
{
    const Embedding* embedding_layer = static_cast<Embedding*>(layer);
    const Index embedding_dimension = embedding_layer->get_embedding_dimension();
    const Index vocabulary_size = embedding_layer->get_vocabulary_size();

    cout << "Embedding layer back propagation CUDA" << endl;
    cout << "Batch size: " << batch_size << endl;
    cout << "Weight gradients dimensions: " << vocabulary_size << "x" << embedding_dimension << endl;
}

#endif

REGISTER(Layer, Embedding, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
