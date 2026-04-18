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
#include "forward_propagation.h"
#include "back_propagation.h"
#ifdef OPENNN_WITH_CUDA
#include "kernel.cuh"
#endif

namespace opennn
{

Embedding::Embedding(const Shape& new_input_shape,
                     Index new_embedding_dimension,
                     const string& new_label) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_embedding_dimension, new_label);

    name = "Embedding";
    layer_type = LayerType::Embedding;
}

Shape Embedding::get_output_shape() const
{
    return {sequence_length, embedding_dimension};
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
    sequence_length = new_sequence_length;
    vocabulary_size = new_vocabulary_size;
    embedding_dimension = new_embedding_dimension;
    embedding_scale = sqrt(static_cast<type>(new_embedding_dimension));
    label = new_label;

    positional_encoding.resize(new_sequence_length, new_embedding_dimension);
    positional_encoding.setZero();

    const type half_depth = type(new_embedding_dimension) / 2;

    VectorR divisors(new_embedding_dimension);
    for(Index j = 0; j < new_embedding_dimension; ++j)
        divisors(j) = pow(type(10000), (j < Index(half_depth) ? j : j - Index(half_depth)) / half_depth);

    #pragma omp parallel for collapse(2)
    for(Index i = 0; i < new_sequence_length; ++i)
        for(Index j = 0; j < new_embedding_dimension; ++j)
            positional_encoding(i, j) = (j < Index(half_depth))
                ? sin(i / divisors(j))
                : cos(i / divisors(j));
#ifdef OPENNN_WITH_CUDA

    const Index pe_size = new_sequence_length * new_embedding_dimension;
    positional_encoding_device.resize(pe_size);
    positional_encoding_device.resize_device(pe_size);
    CHECK_CUDA(cudaMemcpy(positional_encoding_device.device(),
                          positional_encoding.data(),
                          pe_size * sizeof(float),
                          cudaMemcpyHostToDevice));

#endif
}

void Embedding::set_parameters_random()
{
    if(parameters[Weight].empty()) return;

    const type scale = type(0.05);

    MatrixMap weights = matrix_map(parameters[Weight]);

    weights.setRandom();
    weights *= scale;

    weights.row(0).setZero();
}

void Embedding::set_parameters_glorot()
{
    if(parameters[Weight].empty()) return;

//    const Index vocabulary_size = weights.shape[0];
//    const Index embedding_dimension = weights.shape[1];

    const type limit = sqrt(type(6.0) / (vocabulary_size + embedding_dimension));

    MatrixMap weights = matrix_map(parameters[Weight]);

    weights.setRandom();
    weights *= limit;

    weights.row(0).setZero();
}

void Embedding::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    const Index batch_size = forward_propagation.batch_size;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const Index total_elements = batch_size * sequence_length * embedding_dimension;

        const float* positional_encoding_data = add_positional_encoding
            ? positional_encoding_device.device()
            : nullptr;

        embedding_forward_cuda(
            total_elements,
            forward_views[Input][0].data,
            parameters[Weight].data,
            positional_encoding_data,
            forward_views[Output][0].data,
            sequence_length,
            embedding_dimension,
            vocabulary_size,
            scale_embedding,
            add_positional_encoding);
        return;
    }
#endif

    const Index total_tokens = batch_size * sequence_length;

    const TensorView& output_view = forward_views[Output][0];
    MatrixMap outputs(output_view.data, total_tokens, embedding_dimension);

    const MatrixMap weights(parameters[Weight].data, vocabulary_size, embedding_dimension);

    const type* input_indices = forward_views[Input][0].data;

    #pragma omp parallel for
    for(Index i = 0; i < total_tokens; ++i)
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
        outputs *= embedding_scale;

    if(add_positional_encoding)
    {
        #pragma omp parallel for
        for(Index i = 0; i < total_tokens; ++i)
            if (static_cast<Index>(input_indices[i]) > 0)
                outputs.row(i) += positional_encoding.row(i % sequence_length);
    }
}

void Embedding::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        const Index total_elements = forward_propagation.batch_size * sequence_length * embedding_dimension;

        embedding_backward_cuda(
            total_elements,
            forward_views[Input][0].data,
            backward_views[OutputGradient][0].data,
            gradient_views[Weight].data,
            sequence_length,
            embedding_dimension,
            vocabulary_size,
            scale_embedding);

        return;
    }
#endif

    embedding_backward(forward_views[Input][0],
                       backward_views[OutputGradient][0],
                       gradient_views[Weight],
                       embedding_dimension,
                       scale_embedding);

}

void Embedding::from_XML(const XmlDocument& document)
{
    const XmlElement* embedding_layer_element = get_xml_root(document, "Embedding");

    const string new_label = read_xml_string(embedding_layer_element, "Label");
    const Index new_vocabulary_size = read_xml_index(embedding_layer_element, "VocabularySize");
    const Index new_sequence_length = read_xml_index(embedding_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(embedding_layer_element, "EmbeddingSize");

    set(new_vocabulary_size, new_sequence_length, new_embedding_dimension, new_label);

    set_scale_embedding(read_xml_bool(embedding_layer_element, "ScaleEmbedding"));
    set_add_positional_encoding(read_xml_bool(embedding_layer_element, "AddPositionalEncoding"));
}

void Embedding::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Embedding");

    write_xml_properties(printer, {
        {"Label", label},
        {"VocabularySize", to_string(get_vocabulary_size())},
        {"SequenceLength", to_string(get_sequence_length())},
        {"EmbeddingSize", to_string(get_embedding_dimension())},
        {"ScaleEmbedding", to_string(scale_embedding)},
        {"AddPositionalEncoding", to_string(add_positional_encoding)}
    });

    printer.close_element();
}

REGISTER(Layer, Embedding, "Embedding")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
