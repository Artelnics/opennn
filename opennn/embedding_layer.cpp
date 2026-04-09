//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "embedding_layer.h"
#include "neural_network.h"
#include "loss.h"

namespace opennn
{

Embedding::Embedding(const Shape& new_input_shape,
                     Index new_embedding_dimension,
                     const string& new_label) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_embedding_dimension, new_label);

    name = "Embedding";
}

Shape Embedding::get_output_shape() const
{
    return {get_sequence_length(), embedding_dimension};
}

vector<Shape> Embedding::get_parameter_shapes() const
{
    return {{get_vocabulary_size(), embedding_dimension}}; // weights
}

void Embedding::set(const Index new_vocabulary_size,
                    Index new_sequence_length,
                    Index new_embedding_dimension,
                    const string& new_label)
{
    input_shape = {new_sequence_length};
    vocabulary_size = new_vocabulary_size;
    embedding_dimension = new_embedding_dimension;
    label = new_label;

    positional_encoding.resize(new_sequence_length, new_embedding_dimension);
    positional_encoding.setZero();

    const type half_depth = type(new_embedding_dimension) / 2;

    #pragma omp parallel for collapse(2)
    for(Index i = 0; i < new_sequence_length; i++)
        for(Index j = 0; j < new_embedding_dimension; j++)
            positional_encoding(i, j) = (j < Index(half_depth))
                ? sin(i / pow(type(10000), j / half_depth))
                : cos(i / pow(type(10000), (j - Index(half_depth)) / half_depth));
#ifdef CUDA

    weights_device.set_descriptor({new_vocabulary_size, new_embedding_dimension});

    positional_encoding_device.resize({new_sequence_length, new_embedding_dimension});

#endif
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

    const type limit = sqrt(type(6.0) / (get_vocabulary_size() + embedding_dimension));

    MatrixMap weights = matrix_map(parameters[Weights]);

    weights.setRandom();
    weights *= limit;

    weights.row(0).setZero();
}

void Embedding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
#ifndef CUDA

    const Index batch_size = forward_propagation.batch_size;
    const Index total_tokens = batch_size * get_sequence_length();

    const TensorView& output_view = forward_propagation.views[layer][Outputs][0];
    MatrixMap outputs(output_view.data, total_tokens, embedding_dimension);
    outputs.setZero();

    const MatrixMap weights(parameters[Weights].data, vocabulary_size, embedding_dimension);

    const type* input_indices = forward_propagation.views[layer][Inputs][0].data;

    const Index sequence_length = get_sequence_length();

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
        copy_positional_encoding_device();
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
                               size_t layer) const
{
#ifndef CUDA

    const TensorView& input_indices = forward_propagation.views[layer][Inputs][0];
    TensorView& output_gradient = back_propagation.backward_views[layer][0][0];
    TensorView& weight_gradient = back_propagation.gradient_views[layer][Weights];

    embedding_backward(input_indices, output_gradient, weight_gradient,
                       embedding_dimension, scale_embedding);

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

void Embedding::from_XML(const XMLDocument& document)
{
    const XMLElement* embedding_layer_element = get_xml_root(document, "Embedding");

    const string new_label = read_xml_string(embedding_layer_element, "Label");
    const Index new_vocabulary_size = read_xml_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_xml_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_label);

    set_scale_embedding(read_xml_bool(embedding_layer_element, "ScaleEmbedding"));
    set_add_positional_encoding(read_xml_bool(embedding_layer_element, "AddPositionalEncoding"));
}

void Embedding::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Embedding");

    write_xml_properties(printer, {
        {"Label", label},
        {"VocabularySize", to_string(get_vocabulary_size())},
        {"SequenceLength", to_string(get_sequence_length())},
        {"EmbeddingSize", to_string(get_embedding_dimension())},
        {"ScaleEmbedding", to_string(scale_embedding)},
        {"AddPositionalEncoding", to_string(add_positional_encoding)}
    });

    printer.CloseElement();
}

REGISTER(Layer, Embedding, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
